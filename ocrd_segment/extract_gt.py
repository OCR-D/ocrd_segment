from __future__ import absolute_import

import os.path
import json
import PIL.Image, PIL.ImageDraw
import numpy as np

from ocrd_utils import (
    getLogger, concat_padded,
    coordinates_of_segment,
    MIMETYPE_PAGE
)
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    AlternativeImageType,
    TextRegionType,
    to_xml
)
from ocrd import Processor

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-extract-gt'
LOG = getLogger('processor.ExtractGT')
CLASSES = { 'border': (255,255,255),
            'text': (200,0,0),
            'table': (0,100,20),
            'chart': (0,120,0),
            'chem': (0,140,0),
            'graphic': (0,0,200),
            'image': (0,20,180),
            'linedrawing': (20,0,180),
            'maths': (20,120,0),
            'music': (20,120,40),
            'noise': (50,50,50),
            'separator': (0,0,100),
            'unknown': (0,0,0)
        }

class ExtractGT(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(ExtractGT, self).__init__(*args, **kwargs)

    def process(self):
        """Extract region images and coordinates to files not managed in the workspace.
        
        Open and deserialize PAGE input files and their respective images,
        then iterate over the element hierarchy down to the region level.
        
        Get all regions with their types and coordinates relative to the page
        (possibly cropped and/or deskewed). Extract the page image, both in
        binarized and non-binarized form. In addition, create a page image
        which color-codes all regions. Create a JSON file with region types and
        coordinates.
        
        Write all files in the directory of the output file group, named like so:
        * ID + '.png': raw image
        * ID + '.bin.png': binarized image
        * ID + '.dbg.png': debug image
        * ID + '.json': region coordinates
        """
        # pylint: disable=attribute-defined-outside-init
        for n, input_file in enumerate(self.input_files):
            file_id = input_file.ID.replace(self.input_file_grp, self.output_file_grp)
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page = pcgts.get_Page()
            page_image, page_xywh, _ = self.workspace.image_from_page(
                page, page_id, feature_filter='binarized')
            file_path = self.workspace.save_image_file(page_image,
                                                       file_id,
                                                       page_id=page_id,
                                                       file_grp=self.output_file_grp)
            page_image_bin, page_xywh, _ = self.workspace.image_from_page(
                page, page_id, feature_selector='binarized')
            self.workspace.save_image_file(page_image_bin,
                                           file_id + '.bin',
                                           page_id=page_id,
                                           file_grp=self.output_file_grp)
            page_image_dbg = PIL.Image.new(mode='RGB', size=page_image.size, color=0)
            regions = { 'text': page.get_TextRegion(),
                        'table': page.get_TableRegion(),
                        'chart': page.get_ChartRegion(),
                        'chem': page.get_ChemRegion(),
                        'graphic': page.get_GraphicRegion(),
                        'image': page.get_ImageRegion(),
                        'linedrawing': page.get_LineDrawingRegion(),
                        'maths': page.get_MathsRegion(),
                        'music': page.get_MusicRegion(),
                        'noise': page.get_NoiseRegion(),
                        'separator': page.get_SeparatorRegion(),
                        'unknown': page.get_UnknownRegion()
                    }
            description = { 'angle': page.get_orientation() }
            for rtype, rlist in regions.items():
                for region in rlist:
                    coords = coordinates_of_segment(
                        region, page_image, page_xywh).tolist()
                    description.setdefault('regions', []).append(
                        { 'type': rtype,
                          'coords': coords
                    })
                    PIL.ImageDraw.Draw(page_image_dbg).polygon(list(map(tuple,coords)), fill=CLASSES[rtype])
                    PIL.ImageDraw.Draw(page_image_dbg).line(list(map(tuple,coords + [coords[0]])), 
                                                            fill=CLASSES['border'], width=3)
            self.workspace.save_image_file(page_image_dbg,
                                           file_id + '.dbg',
                                           page_id=page_id,
                                           file_grp=self.output_file_grp)
            file_path = file_path.replace('.png', '.json')
            json.dump(description, open(file_path, 'w'))


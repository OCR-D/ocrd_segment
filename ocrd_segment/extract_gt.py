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
        """
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
            page_image, page_xywh, _ = self.workspace.image_from_page(
                page, page_id)
            self.workspace.save_image_file(page_image,
                                           file_id + '.bin',
                                           page_id=page_id,
                                           file_grp=self.output_file_grp)
            debug_image = PIL.Image.new(mode='RGB', size=page_image.size, color=0)
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
                    PIL.ImageDraw.Draw(debug_image).polygon(list(map(tuple,coords)), fill=CLASSES[rtype])
                    PIL.ImageDraw.Draw(debug_image).line(list(map(tuple,coords + [coords[0]])), 
                                                         fill=CLASSES['border'], width=3)
            self.workspace.save_image_file(debug_image,
                                           file_id + '.debug',
                                           page_id=page_id,
                                           file_grp=self.output_file_grp)
            file_path = file_path.replace('.png', '.json')
            json.dump(description, open(file_path, 'w'))


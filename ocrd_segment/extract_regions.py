from __future__ import absolute_import

import json
from PIL import Image, ImageDraw

from ocrd_utils import (
    getLogger, concat_padded,
    coordinates_of_segment
)
from ocrd_models.ocrd_page import (
    LabelsType, LabelType,
    MetadataItemType
)
from ocrd_modelfactory import page_from_file
from ocrd import Processor

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-extract-regions'
LOG = getLogger('processor.ExtractRegions')
# region classes and their colours in debug images:
CLASSES = {'border': (255, 255, 255),
           'text': (200, 0, 0),
           'table': (0, 100, 20),
           'chart': (0, 120, 0),
           'chem': (0, 140, 0),
           'graphic': (0, 0, 200),
           'image': (0, 20, 180),
           'linedrawing': (20, 0, 180),
           'maths': (20, 120, 0),
           'music': (20, 120, 40),
           'noise': (50, 50, 50),
           'separator': (0, 0, 100),
           'unknown': (0, 0, 0)
}

class ExtractRegions(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(ExtractRegions, self).__init__(*args, **kwargs)

    def process(self):
        """Extract page images and region descriptions (type and coordinates) from the workspace.
        
        Open and deserialize PAGE input files and their respective images,
        then iterate over the element hierarchy down to the region level.
        
        Get all regions with their types (region element class), sub-types (@type)
        and coordinates relative to the page (which depending on the workflow could
        already be cropped, deskewed, dewarped, binarized etc). Extract the image of
        the page, both in binarized and non-binarized form. In addition, create a new
        image which color-codes all regions. Create a JSON file with region types and
        coordinates.
        
        Write all files in the directory of the output file group, named like so:
        * ID + '.png': raw image
        * ID + '.bin.png': binarized image
        * ID + '.dbg.png': debug image
        * ID + '.json': region coordinates
        
        (This is intended for training and evaluation of region segmentation models.)
        """
        # pylint: disable=attribute-defined-outside-init
        for n, input_file in enumerate(self.input_files):
            file_id = input_file.ID.replace(self.input_file_grp, self.output_file_grp)
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page = pcgts.get_Page()
            ptype = page.get_type()
            metadata = pcgts.get_Metadata() # ensured by from_file()
            metadata.add_MetadataItem(
                MetadataItemType(type_="processingStep",
                                 name=self.ocrd_tool['steps'][0],
                                 value=TOOL,
                                 Labels=[LabelsType(
                                     externalModel="ocrd-tool",
                                     externalId="parameters",
                                     Label=[LabelType(type_=name,
                                                      value=self.parameter[name])
                                            for name in self.parameter.keys()])]))
            page_image, page_coords, page_image_info = self.workspace.image_from_page(
                page, page_id,
                feature_filter='binarized',
                transparency=self.parameter['transparency'])
            if page_image_info.resolution != 1:
                dpi = page_image_info.resolution
                if page_image_info.resolutionUnit == 'cm':
                    dpi = round(dpi * 2.54)
            else:
                dpi = None
            file_path = self.workspace.save_image_file(page_image,
                                                       file_id,
                                                       self.output_file_grp,
                                                       page_id=page_id)
            page_image_bin, _, _ = self.workspace.image_from_page(
                page, page_id,
                feature_selector='binarized',
                transparency=self.parameter['transparency'])
            self.workspace.save_image_file(page_image_bin,
                                           file_id + '.bin',
                                           self.output_file_grp,
                                           page_id=page_id)
            page_image_dbg = Image.new(mode='RGB', size=page_image.size, color=0)
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
                    polygon = coordinates_of_segment(
                        region, page_image, page_coords).tolist()
                    description.setdefault('regions', []).append(
                        { 'type': rtype,
                          'subtype': region.get_type() if rtype in ['text', 'chart', 'graphic'] else None,
                          'coords': polygon,
                          'features': page_coords['features'],
                          'DPI': dpi,
                          'region.ID': region.id,
                          'page.ID': page_id,
                          'page.type': ptype,
                          'file_grp': self.input_file_grp,
                          'METS.UID': self.workspace.mets.unique_identifier
                        })
                    ImageDraw.Draw(page_image_dbg).polygon(list(map(tuple, polygon)),
                                                               fill=CLASSES[rtype])
                    ImageDraw.Draw(page_image_dbg).line(list(map(tuple, polygon + [polygon[0]])), 
                                                            fill=CLASSES['border'], width=3)
            self.workspace.save_image_file(page_image_dbg,
                                           file_id + '.dbg',
                                           self.output_file_grp,
                                           page_id=page_id)
            file_path = file_path.replace('.png', '.json')
            json.dump(description, open(file_path, 'w'))

from __future__ import absolute_import

import os.path
import json

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
                page, page_id)
            file_path = self.workspace.save_image_file(page_image,
                                                       file_id,
                                                       page_id=page_id,
                                                       file_grp=self.output_file_grp)
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
                    description.setdefault('regions', []).append(
                        { 'type': rtype,
                          'coords': coordinates_of_segment(
                              region, page_image, page_xywh).tolist()
                    })
            file_path = file_path.replace('.png', '.json')
            json.dump(description, open(file_path, 'w'))


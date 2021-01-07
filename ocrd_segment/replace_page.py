from __future__ import absolute_import

import os.path

from ocrd_utils import (
    getLogger, concat_padded,
    coordinates_for_segment,
    points_from_polygon,
    polygon_from_points,
    make_file_id,
    assert_file_grp_cardinality,
    MIMETYPE_PAGE
)
from ocrd_models.ocrd_page import (
    TextRegionType,
    to_xml
)
from ocrd_modelfactory import page_from_file
from ocrd import Processor

from .config import OCRD_TOOL
from .repair import ensure_consistent

TOOL = 'ocrd-segment-replace-page'

class ReplacePage(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(ReplacePage, self).__init__(*args, **kwargs)

    def process(self):
        """Replace everything below the page level with another annotation.
        
        Open and deserialize PAGE input files from both input file groups,
        then go to the page hierarchy level.
        
        Replace all regions (and their reading order) from the page of
        the first input file group with all regions from the page of
        the second input file group. Keep page-level annotations unchanged
        (i.e. Border, orientation, type, AlternativeImage etc).
        
        If ``transform_coordinates`` is true, then also retrieve the
        coordinate transform of the (cropped, deskewed, dewarped) page
        from the first input fileGrp, and use it to adjust all segment
        coordinates from the second input fileGrp, accordingly.
        (This assumes both are consistent, i.e. the second input was derived
        from the first input via ``ocrd-segment-replace-original`` or similar.)
        
        Produce a new output file by serialising the resulting hierarchy.
        """
        LOG = getLogger('processor.ReplacePage')
        assert_file_grp_cardinality(self.input_file_grp, 2, 'original, page')
        assert_file_grp_cardinality(self.output_file_grp, 1)
        adapt_coords = self.parameter['transform_coordinates']
        
        # collect input file tuples
        ifts = self.zip_input_files() # input file tuples
        # process input file tuples
        for n, ift in enumerate(ifts):
            input_file, page_file = ift
            if input_file is None or page_file is None:
                continue
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            page = pcgts.get_Page()
            pcgts2 = page_from_file(self.workspace.download_file(page_file))
            page2 = pcgts2.get_Page()
            # adjust all coordinates (recursively)
            if adapt_coords:
                try:
                    _, page_coords, _ = self.workspace.image_from_page(page, page_id)
                    for region in page2.get_AllRegions():
                        region_coords = region.get_Coords()
                        region_polygon = polygon_from_points(region_coords.points)
                        region_polygon = coordinates_for_segment(region_polygon, None, page_coords)
                        region_coords.set_points(points_from_polygon(region_polygon))
                        ensure_consistent(region)
                        if isinstance(region, TextRegionType):
                            for line in region.get_TextLine():
                                line_coords = line.get_Coords()
                                line_polygon = polygon_from_points(line_coords.points)
                                line_polygon = coordinates_for_segment(line_polygon, None, page_coords)
                                line_coords.set_points(points_from_polygon(line_polygon))
                                ensure_consistent(line)
                                for word in line.get_Word():
                                    word_coords = word.get_Coords()
                                    word_polygon = polygon_from_points(word_coords.points)
                                    word_polygon = coordinates_for_segment(word_polygon, None, page_coords)
                                    word_coords.set_points(points_from_polygon(word_polygon))
                                    ensure_consistent(word)
                                    for glyph in word.get_Glyph():
                                        glyph_coords = glyph.get_Coords()
                                        glyph_polygon = polygon_from_points(glyph_coords.points)
                                        glyph_polygon = coordinates_for_segment(glyph_polygon, None, page_coords)
                                        glyph_coords.set_points(points_from_polygon(glyph_polygon))
                                        ensure_consistent(glyph)
                except:
                    LOG.error('invalid coordinates on page %s', page_id)
                    continue
            # replace all regions
            page.set_ReadingOrder(page2.get_ReadingOrder())
            page.set_TextRegion(page2.get_TextRegion())
            page.set_ImageRegion(page2.get_ImageRegion())
            page.set_LineDrawingRegion(page2.get_LineDrawingRegion())
            page.set_GraphicRegion(page2.get_GraphicRegion())
            page.set_TableRegion(page2.get_TableRegion())
            page.set_ChartRegion(page2.get_ChartRegion())
            page.set_MapRegion(page2.get_MapRegion())
            page.set_SeparatorRegion(page2.get_SeparatorRegion())
            page.set_MathsRegion(page2.get_MathsRegion())
            page.set_ChemRegion(page2.get_ChemRegion())
            page.set_MusicRegion(page2.get_MusicRegion())
            page.set_AdvertRegion(page2.get_AdvertRegion())
            page.set_NoiseRegion(page2.get_NoiseRegion())
            page.set_UnknownRegion(page2.get_UnknownRegion())
            page.set_CustomRegion(page2.get_CustomRegion())

            # update METS (add the PAGE file):
            file_id = make_file_id(page_file, self.output_file_grp)
            out = self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                local_filename=os.path.join(self.output_file_grp, file_id + '.xml'),
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts))
            LOG.info('created file ID: %s, file_grp: %s, path: %s',
                     file_id, self.output_file_grp, out.local_filename)

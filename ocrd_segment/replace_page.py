from __future__ import absolute_import

import os.path
import numpy as np
from shapely.geometry import Polygon, asPolygon
from shapely.ops import unary_union

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
    PageType,
    to_xml
)
from ocrd_modelfactory import page_from_file
from ocrd import Processor

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-replace-page'
LOG = getLogger('processor.ReplacePage')

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
        assert_file_grp_cardinality(self.input_file_grp, 2, 'original, page')
        assert_file_grp_cardinality(self.output_file_grp, 1)
        adapt_coords = self.parameter['transform_coordinates']
        
        ifgs = self.input_file_grp.split(",") # input file groups
        # collect input file tuples
        ifts = self.zip_input_files(ifgs) # input file tuples
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
                        region_polygon = polygon_from_points(region.get_Coords().points)
                        region_polygon = coordinates_for_segment(region_polygon, None, page_coords)
                        region_polygon = polygon_for_parent(region_polygon, page)
                        region.get_Coords().points = points_from_polygon(region_polygon)
                        if isinstance(region, TextRegionType):
                            for line in region.get_TextLine():
                                line_polygon = polygon_from_points(line.get_Coords().points)
                                line_polygon = coordinates_for_segment(line_polygon, None, page_coords)
                                line_polygon = polygon_for_parent(line_polygon, region)
                                line.get_Coords().points = points_from_polygon(line_polygon)
                                for word in line.get_Word():
                                    word_polygon = polygon_from_points(word.get_Coords().points)
                                    word_polygon = coordinates_for_segment(word_polygon, None, page_coords)
                                    word_polygon = polygon_for_parent(word_polygon, line)
                                    word.get_Coords().points = points_from_polygon(word_polygon)
                                    for glyph in word.get_Glyph():
                                        glyph_polygon = polygon_from_points(glyph.get_Coords().points)
                                        glyph_polygon = coordinates_for_segment(glyph_polygon, None, page_coords)
                                        glyph_polygon = polygon_for_parent(glyph_polygon, word)
                                        glyph.get_Coords().points = points_from_polygon(glyph_polygon)
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
    
    def zip_input_files(self, ifgs):
        """Get a list (for each physical page) of tuples (for each input file group) of METS files."""
        ifts = list() # file tuples
        if self.page_id:
            pages = [self.page_id]
        else:
            pages = self.workspace.mets.physical_pages
        for page_id in pages:
            ifiles = list()
            for ifg in ifgs:
                #LOG.debug("adding input file group %s to page %s", ifg, page_id)
                files = self.workspace.mets.find_files(pageId=page_id, fileGrp=ifg)
                # find_files cannot filter by MIME type yet
                files = [file_ for file_ in files if (
                    file_.mimetype.startswith('image/') or
                    file_.mimetype == MIMETYPE_PAGE)]
                if not files:
                    # other fallback options?
                    LOG.error('found no page %s in file group %s',
                              page_id, ifg)
                    ifiles.append(None)
                else:
                    ifiles.append(files[0])
            if ifiles[0]:
                ifts.append(tuple(ifiles))
        return ifts

def polygon_for_parent(polygon, parent):
    """Clip polygon to parent polygon range.
    
    (Should be moved to ocrd_utils.coordinates_for_segment.)
    """
    childp = Polygon(polygon)
    if isinstance(parent, PageType):
        if parent.get_Border():
            parentp = Polygon(polygon_from_points(parent.get_Border().get_Coords().points))
        else:
            parentp = Polygon([[0,0], [0,parent.get_imageHeight()],
                               [parent.get_imageWidth(),parent.get_imageHeight()],
                               [parent.get_imageWidth(),0]])
    else:
        parentp = Polygon(polygon_from_points(parent.get_Coords().points))
    # check if clipping is necessary
    if childp.within(parentp):
        return polygon
    # ensure input coords have valid paths (without self-intersection)
    # (this can happen when shapes valid in floating point are rounded)
    childp = make_valid(childp)
    parentp = make_valid(parentp)
    # clip to parent
    interp = childp.intersection(parentp)
    # post-process
    if interp.is_empty or interp.area == 0.0:
        # FIXME: we need a better strategy against this
        raise Exception("intersection of would-be segment with parent is empty")
    if interp.type == 'GeometryCollection':
        # heterogeneous result: filter zero-area shapes (LineString, Point)
        interp = unary_union([geom for geom in interp.geoms if geom.area > 0])
    if interp.type == 'MultiPolygon':
        # homogeneous result: construct convex hull to connect
        # FIXME: construct concave hull / alpha shape
        interp = interp.convex_hull
    if interp.minimum_clearance < 1.0:
        # follow-up calculations will necessarily be integer;
        # so anticipate rounding here and then ensure validity
        interp = asPolygon(np.round(interp.exterior.coords))
        interp = make_valid(interp)
    return interp.exterior.coords[:-1] # keep open

def make_valid(polygon):
    for split in range(1, len(polygon.exterior.coords)-1):
        if polygon.is_valid or polygon.simplify(polygon.area).is_valid:
            break
        # simplification may not be possible (at all) due to ordering
        # in that case, try another starting point
        polygon = Polygon(polygon.exterior.coords[-split:]+polygon.exterior.coords[:-split])
    for tolerance in range(1, int(polygon.area)):
        if polygon.is_valid:
            break
        # simplification may require a larger tolerance
        polygon = polygon.simplify(tolerance)
    return polygon
    

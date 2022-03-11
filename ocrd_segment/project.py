from __future__ import absolute_import

import os.path
from itertools import chain
import numpy as np
from shapely.geometry import asPolygon, Polygon
from shapely.ops import unary_union
import alphashape

from ocrd import Processor
from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    coordinates_of_segment,
    polygon_from_points,
    points_from_polygon,
    MIMETYPE_PAGE
)
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    PageType,
    BorderType,
    CoordsType,
    to_xml
)
from .config import OCRD_TOOL

TOOL = 'ocrd-segment-project'

class ProjectHull(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super().__init__(*args, **kwargs)

    def process(self):
        """Make coordinates become the convex hull of their constituent segments with Shapely.
        
        Open and deserialize PAGE input files and their respective images,
        then iterate over the segment hierarchy down to the requested hierarchy
        ``level-of-operation``.
        
        For each segment (page border, region, line or word), update the coordinates
        to become the minimal convex hull of its constituent (lower-level) segments
        (regions, lines, words or glyphs), unless no such constituents exist.
        
        (A change in coordinates will automatically invalidate any AlternativeImage
        references on the segment. Therefore, you may need to rebinarize etc.)
        
        Finally, produce new output files by serialising the resulting hierarchy.
        """
        LOG = getLogger('processor.ProjectHull')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        level = self.parameter['level-of-operation']
        
        for n, input_file in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            page = pcgts.get_Page()
            if level == 'page':
                regions = (page.get_TextRegion() +
                           page.get_ImageRegion() +
                           page.get_LineDrawingRegion() +
                           page.get_GraphicRegion() +
                           page.get_TableRegion() +
                           page.get_ChartRegion() +
                           page.get_MapRegion() +
                           page.get_SeparatorRegion() +
                           page.get_MathsRegion() +
                           page.get_ChemRegion() +
                           page.get_MusicRegion() +
                           page.get_AdvertRegion() +
                           page.get_NoiseRegion() +
                           page.get_UnknownRegion() +
                           page.get_CustomRegion())
                if len(regions):
                    self._process_segment(page, regions, page_id)
            else:
                textregions = page.get_AllRegions(classes=['Text'])
                for region in textregions:
                    lines = region.get_TextLine()
                    if not len(lines):
                        continue
                    if level == 'region':
                        self._process_segment(region, lines, page_id)
                        continue
                    for line in lines:
                        words = line.get_Word()
                        if not len(words):
                            continue
                        if level == 'line':
                            self._process_segment(line, words, page_id)
                            continue
                        for word in words:
                            glyphs = word.get_Glyph()
                            if not len(glyphs):
                                continue
                            self._process_segment(word, glyphs, page_id)
            
            file_id = make_file_id(input_file, self.output_file_grp)
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(self.output_file_grp,
                                            file_id + '.xml'),
                content=to_xml(pcgts))
    
    def _process_segment(self, segment, constituents, page_id):
        """Shrink segment outline to become the minimal convex hull of its constituent segments."""
        LOG = getLogger('processor.ProjectHull')
        polygons = [Polygon(polygon_from_points(constituent.get_Coords().points))
                    for constituent in constituents]
        polygon = join_polygons(polygons).buffer(self.parameter['padding']).exterior.coords[:-1]
        if isinstance(segment, PageType):
            oldborder = segment.Border
            segment.Border = None # ensure interim parent is the page frame itself
        # make sure the segment still fits into its own parent
        polygon2 = polygon_for_parent(polygon, segment)
        if polygon2 is None:
            LOG.info('Ignoring extant segment: %s', segment.id)
            if isinstance(segment, PageType):
                segment.Border = oldborder
        else:
            polygon = polygon2
            points = points_from_polygon(polygon)
            coords = CoordsType(points=points)
            LOG.debug('Using new coordinates from %d constituents for segment "%s"',
                      len(constituents), segment.id)
            if isinstance(segment, PageType):
                segment.set_Border(BorderType(Coords=coords))
            else:
                segment.set_Coords(coords)

def join_polygons(polygons, loc='', scale=20):
    """construct concave hull (alpha shape) from input polygons"""
    LOG = getLogger('processor.ProjectHull')
    # ensure input polygons are simply typed
    polygons = list(chain.from_iterable([
        poly.geoms if poly.type in ['MultiPolygon', 'GeometryCollection']
        else [poly]
        for poly in polygons]))
    if len(polygons) == 1:
        return polygons[0]
    # get equidistant list of points along hull
    # (otherwise alphashape will jump across the interior)
    points = [poly.exterior.interpolate(dist).coords[0] # .xy
              for poly in polygons
              for dist in np.arange(0, poly.length, scale / 2)]
    #alpha = alphashape.optimizealpha(points) # too slow
    alpha = 0.03
    jointp = alphashape.alphashape(points, alpha)
    tries = 0
    # from descartes import PolygonPatch
    # import matplotlib.pyplot as plt
    while jointp.type in ['MultiPolygon', 'GeometryCollection'] or len(jointp.interiors):
        # plt.figure()
        # plt.gca().scatter(*zip(*points))
        # for geom in jointp.geoms:
        #     plt.gca().add_patch(PolygonPatch(geom, alpha=0.2))
        # plt.show()
        alpha *= 0.7
        tries += 1
        if tries > 10:
            LOG.warning("cannot find alpha for concave hull on '%s'", loc)
            alpha = 0
        jointp = alphashape.alphashape(points, alpha)
    if jointp.minimum_clearance < 1.0:
        # follow-up calculations will necessarily be integer;
        # so anticipate rounding here and then ensure validity
        jointp = asPolygon(np.round(jointp.exterior.coords))
        jointp = make_valid(jointp)
    return jointp
    
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
    # ensure input coords have valid paths (without self-intersection)
    # (this can happen when shapes valid in floating point are rounded)
    childp = make_valid(childp)
    parentp = make_valid(parentp)
    if not childp.is_valid:
        return None
    if not parentp.is_valid:
        return None
    # check if clipping is necessary
    if childp.within(parentp):
        return childp.exterior.coords[:-1]
    # clip to parent
    interp = make_intersection(childp, parentp)
    if not interp:
        return None
    return interp.exterior.coords[:-1] # keep open

def make_intersection(poly1, poly2):
    interp = poly1.intersection(poly2)
    # post-process
    if interp.is_empty or interp.area == 0.0:
        return None
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
    return interp

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

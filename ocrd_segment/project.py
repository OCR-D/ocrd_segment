from __future__ import absolute_import

import os.path
import itertools
import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union, nearest_points

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
            elif level == 'table':
                for region in page.get_AllRegions(classes=['Table']):
                    regions = region.get_TextRegion()
                    if not len(regions):
                        continue
                    self._process_segment(region, regions, page_id)
            else:
                for region in page.get_AllRegions(classes=['Text']):
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

def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def join_polygons(polygons, scale=20):
    """construct concave hull (alpha shape) from input polygons by connecting their pairwise nearest points"""
    # ensure input polygons are simply typed
    polygons = list(itertools.chain.from_iterable([
        poly.geoms if poly.type in ['MultiPolygon', 'GeometryCollection']
        else [poly]
        for poly in polygons]))
    npoly = len(polygons)
    if npoly == 1:
        return polygons[0]
    # find min-dist path through all polygons (travelling salesman)
    pairs = itertools.combinations(range(npoly), 2)
    paths = list(itertools.permutations(range(npoly)))
    dists = np.eye(npoly, dtype=float)
    for i, j in pairs:
        dists[i, j] = polygons[i].distance(polygons[j])
        dists[j, i] = dists[i, j]
    dists = [sum(dists[i, j] for i, j in pairwise(path))
             for path in paths]
    path = paths[min(enumerate(dists), key=lambda x: x[1])[0]]
    polygons = [polygons[i] for i in path]
    # iteratively join to next nearest neighbour
    jointp = polygons[0]
    for thisp, nextp in pairwise(polygons):
        nearest = nearest_points(jointp, nextp)
        bridgep = LineString(nearest).buffer(max(1, scale/5), resolution=1)
        jointp = unary_union([jointp, bridgep, nextp])
    if jointp.minimum_clearance < 1.0:
        # follow-up calculations will necessarily be integer;
        # so anticipate rounding here and then ensure validity
        jointp = Polygon(np.round(jointp.exterior.coords))
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
        interp = join_polygons(interp.geoms)
    if interp.minimum_clearance < 1.0:
        # follow-up calculations will necessarily be integer;
        # so anticipate rounding here and then ensure validity
        interp = Polygon(np.round(interp.exterior.coords))
        interp = make_valid(interp)
    return interp

def make_valid(polygon):
    points = list(polygon.exterior.coords)
    for split in range(1, len(points)):
        if polygon.is_valid or polygon.simplify(polygon.area).is_valid:
            break
        # simplification may not be possible (at all) due to ordering
        # in that case, try another starting point
        polygon = Polygon(points[-split:]+points[:-split])
    for tolerance in range(int(polygon.area)):
        if polygon.is_valid:
            break
        # simplification may require a larger tolerance
        polygon = polygon.simplify(tolerance + 1)
    return polygon

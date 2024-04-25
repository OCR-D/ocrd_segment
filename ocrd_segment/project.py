from __future__ import absolute_import

import os.path
import itertools
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from shapely.geometry import Polygon, LineString
from shapely.geometry.polygon import orient
from shapely import set_precision
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
            file_id = make_file_id(input_file, self.output_file_grp)
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            pcgts.set_pcGtsId(file_id)
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

            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(self.output_file_grp,
                                            file_id + '.xml'),
                content=to_xml(pcgts))

    def _process_segment(self, segment, constituents, page_id):
        """Overwrite segment outline to become the minimal convex hull of its constituent segments."""
        LOG = getLogger('processor.ProjectHull')
        polygons = [make_valid(Polygon(polygon_from_points(constituent.get_Coords().points)))
                    for constituent in constituents]
        polygon = join_polygons(polygons).buffer(self.parameter['padding']).exterior.coords[:-1]
        # make sure the segment still fits into its parent's parent
        if isinstance(segment, PageType):
            # ensure interim parent is the page frame itself
            parent = PageType(**segment.__dict__)
            parent.Border = None
        else:
            parent = segment.parent_object_
        polygon = polygon_for_parent(polygon, parent)
        if polygon is None:
            LOG.info('Ignoring extant segment: %s', segment.id)
        else:
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
    # ensure input polygons are simply typed and all oriented equally
    polygons = [orient(poly)
                for poly in itertools.chain.from_iterable(
                        [poly.geoms
                         if poly.geom_type in ['MultiPolygon', 'GeometryCollection']
                         else [poly]
                         for poly in polygons])]
    npoly = len(polygons)
    if npoly == 1:
        return polygons[0]
    # find min-dist path through all polygons (travelling salesman)
    pairs = itertools.combinations(range(npoly), 2)
    dists = np.zeros((npoly, npoly), dtype=float)
    for i, j in pairs:
        dist = polygons[i].distance(polygons[j])
        if dist < 1e-5:
            dist = 1e-5 # if pair merely touches, we still need to get an edge
        dists[i, j] = dist
        dists[j, i] = dist
    dists = minimum_spanning_tree(dists, overwrite=True)
    # add bridge polygons (where necessary)
    for prevp, nextp in zip(*dists.nonzero()):
        prevp = polygons[prevp]
        nextp = polygons[nextp]
        nearest = nearest_points(prevp, nextp)
        bridgep = orient(LineString(nearest).buffer(max(1, scale/5), resolution=1), -1)
        polygons.append(bridgep)
    jointp = unary_union(polygons)
    assert jointp.geom_type == 'Polygon', jointp.wkt
    # follow-up calculations will necessarily be integer;
    # so anticipate rounding here and then ensure validity
    jointp2 = set_precision(jointp, 1.0)
    if jointp2.geom_type != 'Polygon' or not jointp2.is_valid:
        jointp2 = Polygon(np.round(jointp.exterior.coords))
        jointp2 = make_valid(jointp2)
    assert jointp2.geom_type == 'Polygon', jointp2.wkt
    return jointp2

def polygon_for_parent(polygon, parent):
    """Clip polygon to parent polygon range.
    
    (Should be moved to ocrd_utils.coordinates_for_segment.)
    """
    childp = Polygon(polygon)
    childp = make_valid(childp)
    if not childp.is_valid:
        return None
    if isinstance(parent, PageType):
        border = parent.get_Border()
        if border and border.get_Coords():
            parentp = Polygon(polygon_from_points(border.get_Coords().points))
        else:
            parentp = Polygon([[0,0], [0,parent.get_imageHeight()],
                               [parent.get_imageWidth(),parent.get_imageHeight()],
                               [parent.get_imageWidth(),0]])
    else:
        if parent.get_Coords():
            parentp = Polygon(polygon_from_points(parent.get_Coords().points))
        else:
            return childp.exterior.coords[:-1]
    # ensure input coords have valid paths (without self-intersection)
    # (this can happen when shapes valid in floating point are rounded)
    parentp = make_valid(parentp)
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
    if interp.geom_type == 'GeometryCollection':
        # heterogeneous result: filter zero-area shapes (LineString, Point)
        interp = unary_union([geom for geom in interp.geoms if geom.area > 0])
    if interp.geom_type == 'MultiPolygon':
        # homogeneous result: construct convex hull to connect
        interp = join_polygons(interp.geoms)
    # follow-up calculations will necessarily be integer;
    # so anticipate rounding here and then ensure validity
    interp = set_precision(interp, 1.0)
    return interp

def make_valid(polygon):
    """Ensures shapely.geometry.Polygon object is valid by repeated rearrangement/simplification/enlargement."""
    points = list(polygon.exterior.coords)
    # try by re-arranging points
    for split in range(1, len(points)):
        if polygon.is_valid or polygon.simplify(polygon.area).is_valid:
            break
        # simplification may not be possible (at all) due to ordering
        # in that case, try another starting point
        polygon = Polygon(points[-split:]+points[:-split])
    # try by simplification
    for tolerance in range(int(polygon.area + 1.5)):
        if polygon.is_valid:
            break
        # simplification may require a larger tolerance
        polygon = polygon.simplify(tolerance + 1)
    # try by enlarging
    for tolerance in range(1, int(polygon.area + 2.5)):
        if polygon.is_valid:
            break
        # enlargement may require a larger tolerance
        polygon = polygon.buffer(tolerance)
    assert polygon.is_valid, polygon.wkt
    return polygon

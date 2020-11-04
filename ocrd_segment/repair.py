from __future__ import absolute_import

import os.path
from collections import namedtuple
from skimage import draw
from scipy.ndimage import filters, morphology
import cv2
import numpy as np
from shapely.geometry import asPolygon, Polygon, LineString
from shapely.ops import unary_union

from ocrd import Processor
from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    coordinates_for_segment,
    coordinates_of_segment,
    polygon_from_points,
    points_from_polygon,
    xywh_from_polygon,
    MIMETYPE_PAGE
)
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    PageType,
    CoordsType,
    to_xml
)
from ocrd_models.ocrd_page_generateds import (
    RegionRefType,
    RegionRefIndexedType,
    OrderedGroupType,
    OrderedGroupIndexedType,
    UnorderedGroupType,
    UnorderedGroupIndexedType,
    ReadingOrderType
)
from ocrd_validators.page_validator import (
    CoordinateConsistencyError,
    CoordinateValidityError,
    PageValidator
)
from .config import OCRD_TOOL

TOOL = 'ocrd-segment-repair'

class RepairSegmentation(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(RepairSegmentation, self).__init__(*args, **kwargs)


    def process(self):
        """Performs segmentation evaluation with Shapely on the workspace.
        
        Open and deserialize PAGE input files and their respective images,
        then iterate over the element hierarchy down to the region level.
        
        Return information on the plausibility of the segmentation into
        regions on the logging level.
        """
        LOG = getLogger('processor.RepairSegmentation')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        sanitize = self.parameter['sanitize']
        plausibilize = self.parameter['plausibilize']
        
        for (n, input_file) in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            page = pcgts.get_Page()
            
            #
            # validate segmentation (warn of children extending beyond their parents)
            #
            report = PageValidator.validate(ocrd_page=pcgts, 
                                            page_textequiv_consistency='off',
                                            check_baseline=False)
            if not report.is_valid:
                errors = report.errors
                report.errors = []
                for error in errors:
                    if isinstance(error, (CoordinateConsistencyError,CoordinateValidityError)):
                        if error.tag.endswith('Region'):
                            element = next((region
                                            for region in page.get_AllRegions()
                                            if region.id == error.ID), None)
                        elif error.tag == 'TextLine':
                            element = next((line
                                            for region in page.get_AllRegions(classes=['Text'])
                                            for line in region.get_TextLine()
                                            if line.id == error.ID), None)
                        elif error.tag == 'Word':
                            element = next((word
                                            for region in page.get_AllRegions(classes=['Text'])
                                            for line in region.get_TextLine()
                                            for word in line.get_Word()
                                            if word.id == error.ID), None)
                        elif error.tag == 'Glyph':
                            element = next((glyph
                                            for region in page.get_AllRegions(classes=['Text'])
                                            for line in region.get_TextLine()
                                            for word in line.get_Word()
                                            for glyph in word.get_Glyph()
                                            if glyph.id == error.ID), None)
                        else:
                            LOG.error("Unrepairable error for unknown segment type: %s",
                                      str(error))
                            report.add_error(error)
                            continue
                        if not element:
                            LOG.error("Unrepairable error for unknown segment element: %s",
                                      str(error))
                            report.add_error(error)
                            continue
                        if isinstance(error, CoordinateConsistencyError):
                            try:
                                ensure_consistent(element)
                            except Exception as e:
                                LOG.error(str(e))
                                report.add_error(error)
                                continue
                        else:
                            ensure_valid(element)
                        LOG.warning("Fixed %s for segment '%s'", error.__class__.__name__, element.id)
            if not report.is_valid:
                LOG.warning(report.to_xml())

            #
            # plausibilize region segmentation (remove redundant text regions)
            #
            ro = page.get_ReadingOrder()
            if ro:
                rogroup = ro.get_OrderedGroup() or ro.get_UnorderedGroup()
            else:
                rogroup = None
            mark_for_deletion = list() # what regions get removed?
            mark_for_merging = dict() # what regions get merged into which regions?
            # cover recursive region structure (but compare only at the same level)
            parents = list(set([region.parent_object_ for region in page.get_AllRegions(classes=['Text'])]))
            for parent in parents:
                regions = parent.get_TextRegion()
                # sort by area to ensure to arrive at a total ordering compatible
                # with the topological sort along containment/equivalence arcs
                # (so we can avoid substituting regions with superregions that have
                #  themselves been substituted/deleted):
                RegionPolygon = namedtuple('RegionPolygon', ['region', 'polygon'])
                regionspolys = sorted([RegionPolygon(region, Polygon(polygon_from_points(region.get_Coords().points)))
                                       for region in regions],
                                      key=lambda x: x.polygon.area)
                for i in range(0, len(regionspolys)):
                    for j in range(i+1, len(regionspolys)):
                        region1 = regionspolys[i].region
                        region2 = regionspolys[j].region
                        poly1 = regionspolys[i].polygon
                        poly2 = regionspolys[j].polygon
                        LOG.debug('Comparing regions "%s" and "%s"', region1.id, region2.id)

                        if poly1.almost_equals(poly2):
                            LOG.warning('Page "%s" region "%s" is almost equal to "%s" %s',
                                        page_id, region2.id, region1.id,
                                        '(removing)' if plausibilize else '')
                            mark_for_deletion.append(region2.id)
                        elif poly1.contains(poly2):
                            LOG.warning('Page "%s" region "%s" is within "%s" %s',
                                        page_id, region2.id, region1.id,
                                        '(removing)' if plausibilize else '')
                            mark_for_deletion.append(region2.id)
                        elif poly2.contains(poly1):
                            LOG.warning('Page "%s" region "%s" is within "%s" %s',
                                        page_id, region1.id, region2.id,
                                        '(removing)' if plausibilize else '')
                            mark_for_deletion.append(region1.id)
                        elif poly1.overlaps(poly2):
                            inter_poly = poly1.intersection(poly2)
                            union_poly = poly1.union(poly2)
                            LOG.debug('Page "%s" region "%s" overlaps "%s" by %f/%f',
                                      page_id, region1.id, region2.id, inter_poly.area/poly1.area, inter_poly.area/poly2.area)
                            if union_poly.convex_hull.area >= poly1.area + poly2.area:
                                # skip this pair -- combined polygon encloses previously free segments
                                pass
                            elif inter_poly.area / poly2.area > self.parameter['plausibilize_merge_min_overlap']:
                                LOG.warning('Page "%s" region "%s" is almost within "%s" %s',
                                            page_id, region2.id, region1.id,
                                            '(merging)' if plausibilize else '')
                                mark_for_merging[region2.id] = region1
                            elif inter_poly.area / poly1.area > self.parameter['plausibilize_merge_min_overlap']:
                                LOG.warning('Page "%s" region "%s" is almost within "%s" %s',
                                            page_id, region1.id, region2.id,
                                            '(merging)' if plausibilize else '')
                                mark_for_merging[region1.id] = region2

                        # TODO: more merging cases...
                        #LOG.info('Intersection %i', poly1.intersects(poly2))
                        #LOG.info('Containment %i', poly1.contains(poly2))
                        #if poly1.intersects(poly2):
                        #    LOG.info('Area 1 %d', poly1.area)
                        #    LOG.info('Area 2 %d', poly2.area)
                        #    LOG.info('Area intersect %d', poly1.intersection(poly2).area)

                if plausibilize:
                    # pass the regions sorted (see above)
                    _plausibilize_group(regionspolys, rogroup, mark_for_deletion, mark_for_merging)

            #
            # sanitize region segmentation (shrink to hull of lines)
            #
            if sanitize:
                self.sanitize_page(page, page_id)
                
            file_id = make_file_id(input_file, self.output_file_grp)
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(self.output_file_grp,
                                            file_id + '.xml'),
                content=to_xml(pcgts))
    
    def sanitize_page(self, page, page_id):
        LOG = getLogger('processor.RepairSegmentation')
        regions = page.get_AllRegions(classes=['Text'])
        page_image, page_coords, _ = self.workspace.image_from_page(
            page, page_id)
        for region in regions:
            LOG.info('Sanitizing region "%s"', region.id)
            lines = region.get_TextLine()
            if not lines:
                LOG.warning('Page "%s" region "%s" contains no textlines', page_id, region.id)
                continue
            heights = []
            tops = []
            # get labels:
            region_mask = np.zeros((page_image.height, page_image.width), dtype=np.uint8)
            for line in lines:
                line_polygon = coordinates_of_segment(line, page_image, page_coords)
                line_xywh = xywh_from_polygon(line_polygon)
                heights.append(line_xywh['h'])
                tops.append(line_xywh['y'])
                region_mask[draw.polygon(line_polygon[:, 1],
                                         line_polygon[:, 0],
                                         region_mask.shape)] = 1
                region_mask[draw.polygon_perimeter(line_polygon[:, 1],
                                                   line_polygon[:, 0],
                                                   region_mask.shape)] = 1
            # estimate scale:
            heights = np.array(heights)
            scale = int(np.max(heights))
            tops = np.array(tops)
            order = np.argsort(tops)
            heights = heights[order]
            tops = tops[order]
            if len(lines) > 1:
                # if interline spacing is larger than line height, use this
                bottoms = tops + heights
                deltas = tops[1:] - bottoms[:-1]
                scale = max(scale, int(np.max(deltas)))
            # close labels:
            region_mask = np.pad(region_mask, scale) # protect edges
            region_mask = np.array(morphology.binary_closing(region_mask, np.ones((scale, 1))), dtype=np.uint8)
            region_mask = region_mask[scale:-scale, scale:-scale] # unprotect
            # extend margins (to ensure simplified hull polygon is outside children):
            region_mask = filters.maximum_filter(region_mask, 3) # 1px in each direction
            # find outer contour (parts):
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # determine areas of parts:
            areas = [cv2.contourArea(contour) for contour in contours]
            total_area = sum(areas)
            if not total_area:
                # ignore if too small
                LOG.warning('Zero contour area in region "%s"', region.id)
                continue
            # pick contour and convert to absolute:
            region_polygon = None
            for i, contour in enumerate(contours):
                area = areas[i]
                if area / total_area < 0.1:
                    LOG.warning('Ignoring contour %d too small (%d/%d) in region "%s"',
                                i, area, total_area, region.id)
                    continue
                # simplify shape (until valid):
                # can produce invalid (self-intersecting) polygons:
                #polygon = cv2.approxPolyDP(contour, 2, False)[:, 0, ::] # already ordered x,y
                polygon = contour[:, 0, ::] # already ordered x,y
                polygon = Polygon(polygon).simplify(1)
                polygon = make_valid(polygon)
                polygon = polygon.exterior.coords[:-1] # keep open
                if len(polygon) < 4:
                    LOG.warning('Ignoring contour %d less than 4 points in region "%s"',
                                i, region.id)
                    continue
                if region_polygon is not None:
                    LOG.error('Skipping region "%s" due to non-contiguous contours',
                              region.id)
                    region_polygon = None
                    break
                region_polygon = coordinates_for_segment(polygon, page_image, page_coords)
            if region_polygon is not None:
                LOG.info('Using new coordinates for region "%s"', region.id)
                region.get_Coords().set_points(points_from_polygon(region_polygon))
    
def _plausibilize_group(regionspolys, rogroup, mark_for_deletion, mark_for_merging):
    LOG = getLogger('processor.RepairSegmentation')
    wait_for_deletion = list()
    reading_order = dict()
    regionrefs = list()
    ordered = False
    # the reading order does not have to include all regions
    # but it may include all types of regions!
    if isinstance(rogroup, (OrderedGroupType, OrderedGroupIndexedType)):
        regionrefs = (rogroup.get_RegionRefIndexed() +
                      rogroup.get_OrderedGroupIndexed() +
                      rogroup.get_UnorderedGroupIndexed())
        ordered = True
    if isinstance(rogroup, (UnorderedGroupType, UnorderedGroupIndexedType)):
        regionrefs = (rogroup.get_RegionRef() +
                      rogroup.get_OrderedGroup() +
                      rogroup.get_UnorderedGroup())
    for elem in regionrefs:
        reading_order[elem.get_regionRef()] = elem
        if not isinstance(elem, (RegionRefType, RegionRefIndexedType)):
            # recursive reading order element (un/ordered group):
            _plausibilize_group(regionspolys, elem, mark_for_deletion, mark_for_merging)
    for regionpoly in regionspolys:
        delete = regionpoly.region.id in mark_for_deletion
        merge = regionpoly.region.id in mark_for_merging
        if delete or merge:
            region = regionpoly.region
            poly = regionpoly.polygon
            if merge:
                # merge region with super region:
                superreg = mark_for_merging[region.id]
                # granularity will necessarily be lost here --
                # this is not for workflows/processors that already
                # provide good/correct segmentation and reading order
                # (in which case orientation, script and style detection
                #  can be expected as well), but rather as a postprocessor
                # for suboptimal segmentation (possibly before reading order
                # detection/correction); hence, all we now do here is
                # show warnings when granularity is lost; but there might
                # be good reasons to do more here when we have better processors
                # and use-cases in the future
                superpoly = Polygon(polygon_from_points(superreg.get_Coords().points))
                superpoly = superpoly.union(poly)
                if superpoly.type == 'MultiPolygon':
                    superpoly = superpoly.convex_hull
                if superpoly.minimum_clearance < 1.0:
                    superpoly = asPolygon(np.round(superpoly.exterior.coords))
                superpoly = make_valid(superpoly)
                superpoly = superpoly.exterior.coords[:-1] # keep open
                superreg.get_Coords().set_points(points_from_polygon(superpoly))
                # FIXME should we merge/mix attributes and features?
                if region.get_orientation() != superreg.get_orientation():
                    LOG.warning('Merging region "{}" with orientation {} into "{}" with {}'.format(
                        region.id, region.get_orientation(),
                        superreg.id, superreg.get_orientation()))
                if region.get_type() != superreg.get_type():
                    LOG.warning('Merging region "{}" with type {} into "{}" with {}'.format(
                        region.id, region.get_type(),
                        superreg.id, superreg.get_type()))
                if region.get_primaryScript() != superreg.get_primaryScript():
                    LOG.warning('Merging region "{}" with primaryScript {} into "{}" with {}'.format(
                        region.id, region.get_primaryScript(),
                        superreg.id, superreg.get_primaryScript()))
                if region.get_primaryLanguage() != superreg.get_primaryLanguage():
                    LOG.warning('Merging region "{}" with primaryLanguage {} into "{}" with {}'.format(
                        region.id, region.get_primaryLanguage(),
                        superreg.id, superreg.get_primaryLanguage()))
                if region.get_TextStyle():
                    LOG.warning('Merging region "{}" with TextStyle {} into "{}" with {}'.format(
                        region.id, region.get_TextStyle(), # FIXME needs repr...
                        superreg.id, superreg.get_TextStyle())) # ...to be informative
                if region.get_TextEquiv():
                    LOG.warning('Merging region "{}" with TextEquiv {} into "{}" with {}'.format(
                        region.id, region.get_TextEquiv(), # FIXME needs repr...
                        superreg.id, superreg.get_TextEquiv())) # ...to be informative
            wait_for_deletion.append(region)
            if region.id in reading_order:
                regionref = reading_order[region.id]
                # TODO: re-assign regionref.continuation and regionref.type to other?
                # could be any of the 6 types above:
                regionrefs = rogroup.__getattribute__(regionref.__class__.__name__.replace('Type', ''))
                # remove in-place
                regionrefs.remove(regionref)

    if ordered:
        # re-index the reading order!
        regionrefs.sort(key=RegionRefIndexedType.get_index)
        for i, regionref in enumerate(regionrefs):
            regionref.set_index(i)
        
    for region in wait_for_deletion:
        if region.parent_object_:
            # remove in-place
            region.parent_object_.get_TextRegion().remove(region)

def ensure_consistent(child):
    """Clip segment element polygon to parent polygon range."""
    points = child.get_Coords().points
    polygon = polygon_from_points(points)
    parent = child.parent_object_
    childp = Polygon(polygon)
    if isinstance(parent, PageType):
        if parent.get_Border():
            parentp = Polygon(polygon_from_points(parent.get_Border().get_Coords().points))
        else:
            parentp = Polygon([[0, 0], [0, parent.get_imageHeight()],
                               [parent.get_imageWidth(), parent.get_imageHeight()],
                               [parent.get_imageWidth(), 0]])
    else:
        parentp = Polygon(polygon_from_points(parent.get_Coords().points))
    # check if clipping is necessary
    if childp.within(parentp):
        return
    # ensure input coords have valid paths (without self-intersection)
    # (this can happen when shapes valid in floating point are rounded)
    childp = make_valid(childp)
    parentp = make_valid(parentp)
    # clip to parent
    interp = childp.intersection(parentp)
    if interp.is_empty or interp.area == 0.0:
        if hasattr(parent, 'pcGtsId'):
            parent_id = parent.pcGtsId
        elif hasattr(parent, 'imageFilename'):
            parent_id = parent.imageFilename
        else:
            parent_id = parent.id
        raise Exception("Segment '%s' does not intersect its parent '%s'" % (
            child.id, parent_id))
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
    polygon = interp.exterior.coords[:-1] # keep open
    points = points_from_polygon(polygon)
    child.get_Coords().set_points(points)

def ensure_valid(element):
    coords = element.get_Coords()
    points = coords.points
    polygon = polygon_from_points(points)
    poly = Polygon(polygon)
    if not poly.is_valid:
        poly = make_valid(poly)
        polygon = poly.exterior.coords[:-1]
        points = points_from_polygon(polygon)
        coords.set_points(points)

def make_valid(polygon):
    """Ensures shapely.geometry.Polygon object is valid by repeated simplification"""
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

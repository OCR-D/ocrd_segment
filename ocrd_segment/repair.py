from __future__ import absolute_import

import os.path
from skimage import draw
from scipy.ndimage import filters
import cv2
import numpy as np
from shapely.geometry import Polygon, LineString

from ocrd import Processor
from ocrd_utils import (
    getLogger, concat_padded,
    coordinates_for_segment,
    coordinates_of_segment,
    polygon_from_points,
    points_from_polygon,
    xywh_from_polygon,
    MIMETYPE_PAGE
)
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    CoordsType,
    LabelType, LabelsType,
    MetadataItemType,
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
from .config import OCRD_TOOL

TOOL = 'ocrd-segment-repair'
LOG = getLogger('processor.RepairSegmentation')

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
        sanitize = self.parameter['sanitize']
        plausibilize = self.parameter['plausibilize']
        
        for (n, input_file) in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page = pcgts.get_Page()
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

            #
            # validate segmentation (warn of children extending beyond their parents)
            #
            self.validate_coords(page, page_id)

            #
            # sanitize region segmentation (shrink to hull of lines)
            #
            if sanitize:
                self.sanitize_page(page, page_id)
                
            #
            # plausibilize region segmentation (remove redundant regions)
            #
            mark_for_deletion = list()
            mark_for_merging = list()

            regions = page.get_TextRegion()
            for i in range(0, len(regions)):
                for j in range(i+1, len(regions)):
                    region1 = regions[i]
                    region2 = regions[j]
                    LOG.debug('Comparing regions "%s" and "%s"', region1.id, region2.id)
                    region1_poly = Polygon(polygon_from_points(region1.get_Coords().points))
                    region2_poly = Polygon(polygon_from_points(region2.get_Coords().points))
                    
                    equality = region1_poly.almost_equals(region2_poly)
                    if equality:
                        LOG.warning('Page "%s" regions "%s" and "%s" cover the same area.',
                                    page_id, region1.id, region2.id)
                        mark_for_deletion.append(region2)

                    if region1_poly.contains(region2_poly):
                        LOG.warning('Page "%s" region "%s" contains "%s"',
                                    page_id, region1.id, region2.id)
                        mark_for_deletion.append(region2)
                    elif region2_poly.contains(region1_poly):
                        LOG.warning('Page "%s" region "%s" contains "%s"',
                                    page_id, region2.id, region1.id)
                        mark_for_deletion.append(region1)

                    #LOG.info('Intersection %i', region1_poly.intersects(region2_poly))
                    #LOG.info('Containment %i', region1_poly1.contains(region2_poly))
                    #if region1_poly.intersects(region2_poly):
                    #    LOG.info('Area 1 %d', region1_poly.area)
                    #    LOG.info('Area 2 %d', region2_poly.area)
                    #    LOG.info('Area intersect %d', region1_poly.intersection(region2_poly).area)
                        

            if plausibilize:
                # the reading order does not have to include all regions
                # but it may include all types of regions!
                ro = page.get_ReadingOrder()
                _plausibilize_group(regions, ro.get_OrderedGroup() or ro.get_UnorderedGroup(),
                                    mark_for_deletion)

            # Use input_file's basename for the new file -
            # this way the files retain the same basenames:
            file_id = input_file.ID.replace(self.input_file_grp, self.output_file_grp)
            if file_id == input_file.ID:
                file_id = concat_padded(self.output_file_grp, n)
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(self.output_file_grp,
                                            file_id + '.xml'),
                content=to_xml(pcgts))
    
    def sanitize_page(self, page, page_id):
        regions = page.get_TextRegion()
        page_image, page_coords, _ = self.workspace.image_from_page(
            page, page_id)
        for region in regions:
            LOG.info('Sanitizing region "%s"', region.id)
            region_image, region_coords = self.workspace.image_from_segment(
                region, page_image, page_coords)
            lines = region.get_TextLine()
            heights = []
            # get labels:
            region_mask = np.zeros((region_image.height, region_image.width), dtype=np.uint8)
            for line in lines:
                line_polygon = coordinates_of_segment(line, region_image, region_coords)
                heights.append(xywh_from_polygon(line_polygon)['h'])
                region_mask[draw.polygon(line_polygon[:, 1],
                                         line_polygon[:, 0],
                                         region_mask.shape)] = 1
                region_mask[draw.polygon_perimeter(line_polygon[:, 1],
                                                   line_polygon[:, 0],
                                                   region_mask.shape)] = 1
            # estimate scale:
            scale = int(np.median(np.array(heights)))
            # close labels:
            region_mask = np.pad(region_mask, scale) # protect edges
            region_mask = filters.maximum_filter(region_mask, (scale, 1), origin=0)
            region_mask = filters.minimum_filter(region_mask, (scale, 1), origin=0)
            region_mask = region_mask[scale:-scale, scale:-scale] # unprotect
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
                # simplify shape:
                polygon = cv2.approxPolyDP(contour, 2, False)[:, 0, ::] # already ordered x,y
                if len(polygon) < 4:
                    LOG.warning('Ignoring contour %d less than 4 points in region "%s"',
                                i, region.id)
                    continue
                if region_polygon is not None:
                    LOG.error('Skipping region "%s" due to non-contiguous contours',
                              region.id)
                    region_polygon = None
                    break
                region_polygon = coordinates_for_segment(polygon, region_image, region_coords)
            if region_polygon is not None:
                LOG.info('Using new coordinates for region "%s"', region.id)
                region.get_Coords().points = points_from_polygon(region_polygon)
    
    def validate_coords(self, page, page_id):
        valid = True
        regions = page.get_TextRegion()
        if page.get_Border():
            other_regions = (
                page.get_AdvertRegion() +
                page.get_ChartRegion() +
                page.get_ChemRegion() +
                page.get_GraphicRegion() +
                page.get_ImageRegion() +
                page.get_LineDrawingRegion() +
                page.get_MathsRegion() +
                page.get_MusicRegion() +
                page.get_NoiseRegion() +
                page.get_SeparatorRegion() +
                page.get_TableRegion() +
                page.get_UnknownRegion())
            for region in regions + other_regions:
                if not _child_within_parent(region, page.get_Border()):
                    LOG.warning('Region "%s" extends beyond Border of page "%s"',
                                region.id, page_id)
                    valid = False
        for region in regions:
            lines = region.get_TextLine()
            for line in lines:
                if not _child_within_parent(line, region):
                    LOG.warning('Line "%s" extends beyond region "%s" on page "%s"',
                                line.id, region.id, page_id)
                    valid = False
                if line.get_Baseline():
                    baseline = LineString(polygon_from_points(line.get_Baseline().points))
                    linepoly = Polygon(polygon_from_points(line.get_Coords().points))
                    if not baseline.within(linepoly):
                        LOG.warning('Baseline extends beyond line "%s" in region "%s" on page "%s"',
                                    line.id, region.id, page_id)
                        valid = False
                words = line.get_Word()
                for word in words:
                    if not _child_within_parent(word, line):
                        LOG.warning('Word "%s" extends beyond line "%s" in region "%s" on page "%s"',
                                    word.id, line.id, region.id, page_id)
                        valid = False
                    glyphs = word.get_Glyph()
                    for glyph in glyphs:
                        if not _child_within_parent(glyph, word):
                            LOG.warning('Glyph "%s" extends beyond word "%s" in line "%s" of region "%s" on page "%s"',
                                        glyph.id, word.id, line.id, region.id, page_id)
                            valid = False
        return valid

def _child_within_parent(child, parent):
    child_poly = Polygon(polygon_from_points(child.get_Coords().points))
    parent_poly = Polygon(polygon_from_points(parent.get_Coords().points))
    return child_poly.within(parent_poly)

def _plausibilize_group(regions, rogroup, mark_for_deletion):
    wait_for_deletion = list()
    reading_order = dict()
    ordered = False
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
            _plausibilize_group(regions, elem, mark_for_deletion)
    for region in regions:
        if (region in mark_for_deletion and
            region.get_id() in reading_order):
            wait_for_deletion.append(region)
            regionref = reading_order[region.get_id()]
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
        if region in regions:
            # remove in-place
            regions.remove(region)

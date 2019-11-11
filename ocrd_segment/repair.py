from __future__ import absolute_import

import os.path

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
    RegionRefIndexedType,
    to_xml
)

from .config import OCRD_TOOL

from skimage import draw
from scipy.ndimage import filters
import cv2
import numpy as np
from shapely.geometry import Polygon

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
            metadata = pcgts.get_Metadata() # ensured by from_file()
            metadata.add_MetadataItem(
                MetadataItemType(type_="processingStep",
                                 name=self.ocrd_tool['steps'][0],
                                 value=TOOL,
                                 # FIXME: externalRef is invalid by pagecontent.xsd, but ocrd does not reflect this
                                 # what we want here is `externalModel="ocrd-tool" externalId="parameters"`
                                 Labels=[LabelsType(#externalRef="parameters",
                                                    Label=[LabelType(type_=name,
                                                                     value=self.parameter[name])
                                                           for name in self.parameter.keys()])]))
            page = pcgts.get_Page()

            regions = page.get_TextRegion()

            #
            # sanitize regions
            #
            if sanitize:
                page_image, page_coords, _ = self.workspace.image_from_page(
                    page, page_id)
                for i, region in enumerate(regions):
                    LOG.info('Sanitizing region "%s"', region.id)
                    region_image, region_coords = self.workspace.image_from_segment(
                        region, page_image, page_coords)
                    lines = region.get_TextLine()
                    heights = []
                    # get labels:
                    region_mask = np.zeros((region_image.height, region_image.width), dtype=np.uint8)
                    for j, line in enumerate(lines):
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
                
            #
            # plausibilize segmentation
            #
            mark_for_deletion = set()
            mark_for_merging = set()

            for i in range(0,len(regions)):
                for j in range(i+1,len(regions)):
                    LOG.info('Comparing regions "%s" and "%s"', regions[i].id, regions[j].id)
                    region_poly1 = Polygon(polygon_from_points(regions[i].get_Coords().points))
                    region_poly2 = Polygon(polygon_from_points(regions[j].get_Coords().points))
                    
                    LOG.debug('Checking for equality ...')
                    equality = region_poly1.almost_equals(region_poly2)
                    if equality:
                        LOG.warn('Warning: regions %s and %s cover the same area.' % (regions[i].id, regions[j].id))
                        mark_for_deletion.add(j)

                    LOG.debug('Checking for containment ...')
                    containment_r = region_poly1.contains(region_poly2)
                    containment_l = region_poly2.contains(region_poly1)
                    if containment_r:
                        LOG.warn('Warning: %s contains %s' % (regions[i].id, regions[j].id))
                        mark_for_deletion.add(j)
                    if containment_l:
                        LOG.warn('Warning: %s contains %s' % (regions[j].id, regions[i].id))
                        mark_for_deletion.add(i)

            if plausibilize:
                new_regions = []
                reading_order = {}
                # the reading order does not have to include all regions
                # but it may include all types of regions!
                for elem in page.get_ReadingOrder().get_OrderedGroup().get_RegionRefIndexed():
                    reading_order[elem.get_regionRef()] = elem
                for i in range(0,len(regions)):
                    if not i in mark_for_deletion:
                        new_regions.append(regions[i])
                    else:
                        if regions[i].get_id() in reading_order:
                            page.get_ReadingOrder().get_OrderedGroup().get_RegionRefIndexed().remove(reading_order[regions[i].get_id()])
                page.get_ReadingOrder().get_OrderedGroup().get_RegionRefIndexed().sort(key=RegionRefIndexedType.get_index)

                # re-index the reading order!
                for i in range(0, len(page.get_ReadingOrder().get_OrderedGroup().get_RegionRefIndexed())):
                    page.get_ReadingOrder().get_OrderedGroup().get_RegionRefIndexed()[i].set_index(i)
                page.set_TextRegion(new_regions)


                    #LOG.info('Intersection %i', region_poly1.intersects(region_poly2))
                    #LOG.info('Containment %i', region_poly1.contains(region_poly2))
                    #if region_poly1.intersects(region_poly2):
                    #    LOG.info('Area 1 %d', region_poly1.area)
                    #    LOG.info('Area 2 %d', region_poly2.area)
                    #    LOG.info('Area intersect %d', region_poly1.intersection(region_poly2).area)
                        

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

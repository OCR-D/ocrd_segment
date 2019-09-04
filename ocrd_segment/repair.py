from __future__ import absolute_import

import os.path

from ocrd import Processor
from ocrd_utils import (
    getLogger, concat_padded,
    polygon_from_points,
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

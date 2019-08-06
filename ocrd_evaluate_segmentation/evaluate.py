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
    TextLineType,
    to_xml
)

from .config import OCRD_TOOL

from shapely.geometry import Polygon

TOOL = 'ocrd-evaluate-segmentation'
LOG = getLogger('processor.EvaluateSegmentation')

class EvaluateSegmentation(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(EvaluateSegmentation, self).__init__(*args, **kwargs)


    def process(self):
        """Performs segmentation evaluation with Shapely on the workspace.
        
        Open and deserialize PAGE input files and their respective images,
        then iterate over the element hierarchy down to the region level.
        
        Return information on the plausibility of the segmentation into
        regions on the logging level.
        """
        
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
            
            for region1 in page.get_TextRegion():
                for region2 in page.get_TextRegion():
                    if region1.id == region2.id:
                        continue
                    LOG.info('Comparing regions "%s" and "%s"', region1.id, region2.id)
                    region_poly1 = Polygon(polygon_from_points(region1.get_Coords().points))
                    region_poly2 = Polygon(polygon_from_points(region2.get_Coords().points))
                    LOG.info('Intersection %i', region_poly1.intersects(region_poly2))
                    LOG.info('Containment %i', region_poly1.contains(region_poly2))
                    if region_poly1.intersects(region_poly2):
                        LOG.info('Area 1 %d', region_poly1.area)
                        LOG.info('Area 2 %d', region_poly2.area)
                        LOG.info('Area intersect %d', region_poly1.intersection(region_poly2).area)
                        

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

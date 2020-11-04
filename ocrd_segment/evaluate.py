from __future__ import absolute_import

from shapely.geometry import Polygon

from ocrd import Processor
from ocrd_utils import (
    getLogger,
    assert_file_grp_cardinality,
    MIMETYPE_PAGE
)
from ocrd_modelfactory import page_from_file

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-evaluate'

class EvaluateSegmentation(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(EvaluateSegmentation, self).__init__(*args, **kwargs)

    def process(self):
        """Performs segmentation evaluation with Shapely on the workspace.
        
        Open and deserialize PAGE input files and their respective images,
        then iterate over the element hierarchy down to the region level.
        
        Compare region polygons with each other.
        """
        LOG = getLogger('processor.EvaluateSegmentation')

        # commented due to core#632
        #assert_file_grp_cardinality(self.output_file_grp, 0, 'no output files are written')
        # TODO assert_file_grp_cardinality only supports == check not <= or >=
        # assert_file_grp_cardinality(self.input_file_grp, 2, 'GT and evaluation data')
        ifgs = self.input_file_grp.split(",") # input file groups
        if len(ifgs) < 2:
            raise Exception("need multiple input file groups to compare")
        
        # get input file tuples:
        ifts = self.zip_input_files(mimetype=MIMETYPE_PAGE)
        for ift in ifts:
            pages = []
            for i, input_file in enumerate(ift):
                if not i:
                    LOG.info("processing page %s", input_file.pageId)
                if not input_file:
                    # file/page was not found in this group
                    continue
                LOG.info("INPUT FILE for '%s': '%s'", input_file.fileGrp, input_file.ID)
                pcgts = page_from_file(self.workspace.download_file(input_file))
                pages.append(pcgts.get_Page())
            if not len(pages) > 1:
                LOG.warning("Nothing to compare on page '%s'", ift[0].pageId)
                continue
            gt_page = pages[0]
            for pred_page in pages[1:]:
                #
                self._compare_segmentation(gt_page, pred_page, ift[0].pageId)
    
    def _compare_segmentation(self, gt_page, pred_page, page_id):
        LOG = getLogger('processor.EvaluateSegmentation')
        gt_regions = gt_page.get_TextRegion()
        pred_regions = pred_page.get_TextRegion()
        if len(gt_regions) != len(pred_regions):
            LOG.warning("page '%s': %d vs %d text regions",
                        page_id, len(gt_regions), len(pred_regions))
        # FIXME: add actual layout alignment and comparison

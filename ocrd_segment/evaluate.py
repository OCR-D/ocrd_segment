from __future__ import absolute_import

from shapely.geometry import Polygon

from ocrd import Processor
from ocrd_utils import getLogger, assert_file_grp_cardinality
from ocrd_modelfactory import page_from_file

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-evaluate'
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
        
        Compare region polygons with each other.
        """

        assert_file_grp_cardinality(self.output_file_grp, 0, 'no output files are written')
        # TODO assert_file_grp_cardinality only supports == check not <= or >=
        # assert_file_grp_cardinality(self.input_file_grp, 2, 'GT and evaluation data')
        ifgs = self.input_file_grp.split(",") # input file groups
        if len(ifgs) < 2:
            raise Exception("need multiple input file groups to compare")
        
        # get input files:
        ifts = self._zip_input_files(ifgs) # input file tuples
        for ift in ifts:
            pages = []
            for i, input_file in enumerate(ift):
                if not i:
                    LOG.info("processing page %s", input_file.pageId)
                if not input_file:
                    # file/page was not found in this group
                    continue
                LOG.info("INPUT FILE for '%s': '%s'", ifgs[i], input_file.ID)
                pcgts = page_from_file(self.workspace.download_file(input_file))
                pages.append(pcgts.get_Page())
            gt_page = pages[0]
            for pred_page in pages[1:]:
                #
                self._compare_segmentation(gt_page, pred_page, input_file.pageId)
    
    def _compare_segmentation(self, gt_page, pred_page, page_id):
        gt_regions = gt_page.get_TextRegion()
        pred_regions = pred_page.get_TextRegion()
        if len(gt_regions) != len(pred_regions):
            LOG.warning("page '%s': %d vs %d text regions",
                        page_id, len(gt_regions), len(pred_regions))

    def _zip_input_files(self, ifgs):
        ifts = list() # file tuples
        for page_id in ([self.page_id] if self.page_id else
                        self.workspace.mets.physical_pages):
            ifiles = list()
            for ifg in ifgs:
                LOG.debug("adding input file group %s to page %s", ifg, page_id)
                files = self.workspace.mets.find_files(pageId=page_id, fileGrp=ifg)
                if not files:
                    # fall back for missing pageId via Page imageFilename:
                    all_files = self.workspace.mets.find_files(fileGrp=ifg)
                    for file_ in all_files:
                        pcgts = page_from_file(self.workspace.download_file(file_))
                        image_url = pcgts.get_Page().get_imageFilename()
                        img_files = self.workspace.mets.find_files(url=image_url)
                        if img_files and img_files[0].pageId == page_id:
                            files = [file_]
                            break
                if not files:
                    # other fallback options?
                    LOG.error('found no page "%s" in file group %s',
                              page_id, ifg)
                    ifiles.append(None)
                else:
                    ifiles.append(files[0])
            if ifiles[0]:
                ifts.append(tuple(ifiles))
        return ifts
        

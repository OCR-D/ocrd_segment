from __future__ import absolute_import

import copy
import os.path

from ocrd_utils import (
    getLogger,
    make_file_id,
    concat_padded,
    assert_file_grp_cardinality,
    MIMETYPE_PAGE
)
from ocrd_models.ocrd_page import TextLineType, to_xml
from ocrd_modelfactory import page_from_file
from ocrd import Processor

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-classify-formdata-dummy'

class ClassifyFormDataDummy(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(ClassifyFormDataDummy, self).__init__(*args, **kwargs)

    def process(self):
        """Align form field regions (context and target) from GT annotations for multiple classes.
        
        Open and deserialize PAGE input files for all fileGrps (where the
        first group is the original segmentation and the subsequent groups
        contain the manual context/target annotation for one class each).
        Then iterate over the element hierarchy down to the text line level.
        
        Initialise the output annotation from the original input annotation.
        For each GT group, if the (region and) line ID does not exist in the
        original group, or has different coordinates, then add it to the output
        (appending the fileGrp name to the ID). Else re-use the existing ID.
        Regardless, if the line's ``TextRegion/@type`` equals ``context-type``,
        then append ``subtype:context=CLASS`` to the output line's ``@custom``,
        where CLASS is the value of ``categories`` for that GT group. 
        Likewise, if the line's ``TextRegion/@type`` equals ``target-type``,
        then append ``subtype:target=CLASS`` to the output line's ``@custom``,
        where CLASS is the value of ``categories`` for that GT group. 
        
        Produce a new output file by serialising the resulting hierarchy.
        
        (This is intended for training and evaluation of region segmentation models.)
        """
        LOG = getLogger('processor.ClassifyFormDataDummy')
        categories = self.parameter['categories']
        context_type = self.parameter['context-type']
        target_type = self.parameter['target-type']
        
        assert_file_grp_cardinality(self.input_file_grp, 1 + len(categories), \
                                    "original segmentation + class-specific GT groups")
        assert_file_grp_cardinality(self.output_file_grp, 1)
        ifgs = self.input_file_grp.split(',')

        # input file tuples
        ifts = self.zip_input_files(mimetype=MIMETYPE_PAGE, on_error='abort')
        for n, ift in enumerate(ifts):
            input_file, *gt_files = ift
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page = pcgts.get_Page()
            region_points = [region.get_Coords().points for region in page.get_AllRegions(order='document')]
            def find_region(ID, points):
                # When LAREX imports PAGE, it renames all segment IDs in a consecutive way.
                # So we need to compare either by ID or by points.
                return next((region
                             for region in page.get_AllRegions()
                             if ID == region.id or points == region.get_Coords().points), None)
            for gt_file, gt_group, category in zip(gt_files, ifgs[1:], categories):
                if not gt_file:
                    LOG.error("No GT for %s [%s] on page '%s'", gt_group, category, page_id)
                    continue
                LOG.info("Aligning GT for %s [%s]", gt_group, category)
                # iterate through all regions that could have lines
                gt_page = page_from_file(self.workspace.download_file(gt_file)).get_Page()
                for iregion in gt_page.get_AllRegions(classes=['Text']):
                    oregion = find_region(iregion.id, iregion.get_Coords().points)
                    if not oregion or oregion.get_Coords().points != iregion.get_Coords().points:
                        oregion = copy.deepcopy(iregion)
                        oregion.id += '_' + gt_group
                        for line in oregion.get_TextLine():
                            line.id += '_' + gt_group
                            for word in line.get_Word():
                                word.id += '_' + gt_group
                                for glyph in word.get_Glyph():
                                    glyph.id += '_' + gt_group
                        oregion.parent_object_ = page
                        page.add_TextRegion(oregion)
                        LOG.info("Adding new region '%s'", oregion.id)
                    itype = iregion.get_type()
                    if itype not in [context_type, target_type]:
                        continue # no mark (just another text line)
                    custom = 'subtype:%s=%s' % ({context_type: "context",
                                                 target_type: "target"}[itype],
                                                category)
                    oline = TextLineType(id=oregion.id + '_line',
                                         Coords=oregion.get_Coords(),
                                         custom=custom)
                    oregion.set_TextLine([oline])
                    oregion.set_type('other')
                    LOG.info("Marked region/line by '%s'", custom)
            
            # write regions to custom JSON for this page
            file_id = make_file_id(input_file, self.output_file_grp)
            file_path = os.path.join(self.output_file_grp, file_id + '.xml')
            pcgts.set_pcGtsId(file_id)
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                local_filename=file_path,
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts))

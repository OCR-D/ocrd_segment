from __future__ import absolute_import

import sys
import os
import json
import numpy as np
from skimage import draw
from shapely.geometry import Polygon

from ocrd import Processor
from ocrd_utils import (
    getLogger,
    assert_file_grp_cardinality,
    xywh_from_polygon,
    polygon_from_points,
    coordinates_of_segment,
    MIMETYPE_PAGE
)
from ocrd_modelfactory import page_from_file

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import encode as encodeMask

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-evaluate'

class EvaluateSegmentation(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(EvaluateSegmentation, self).__init__(*args, **kwargs)

    def process(self):
        """Performs segmentation evaluation with pycocotools on the workspace.
        
        Open and deserialize PAGE files from the first and second input file group
        (the first as ground truth, the second as prediction).
        Then iterate over the element hierarchy down to ``level-of-operation``.
        Aggregate and convert all pages' segmentation (coordinates and classes)
        to COCO:
        - On the region level, unless ``ignore-subtype``, differentiate segment
          classes by their `@type`, if applicable.
        - On the region level, unless ``for-categories`` is empty, select only
          segment classes in that (comma-separated) list.
        - If ``only-fg``, then use the foreground mask from the binarized
          image inside each segment for overlap calculations.
        
        Next, configure and run COCOEval for comparison of all pages. Show the matching
        pairs (GT segment ID, prediction segment ID, IoU) for every overlap on each page.
        Also, calculate per-class precision and recall (at the point of maximum recall).
        Finally, get the typical summary mean average precision / recall (but without
        restriction on the number of segments).
        
        Write a JSON report to the output file group.
        """
        LOG = getLogger('processor.EvaluateSegmentation')

        assert_file_grp_cardinality(self.output_file_grp, 1)
        assert_file_grp_cardinality(self.input_file_grp, 2, 'GT and evaluation data')
        # region or line level?
        level = self.parameter['level-of-operation']
        onlyfg = self.parameter['only-fg']
        typed = not self.parameter['ignore-subtype']
        selected = self.parameter['for-categories']
        if selected:
            selected = selected.split(',')
        # get input file groups
        ifgs = self.input_file_grp.split(",")
        # get input file tuples
        ifts = self.zip_input_files(mimetype=MIMETYPE_PAGE)
        # convert to 2 COCO datasets from all page pairs
        categories = ["bg"] # needed by cocoeval
        images = []
        annotations_gt = []
        annotations_dt = []
        for ift in ifts:
            file_gt, file_dt = ift
            if not file_gt:
                LOG.warning("skipping page %s missing from GT", file_gt.pageId)
                continue
            if not file_dt:
                LOG.warning("skipping page %s missing from prediction", file_dt.pageId)
                continue
            LOG.info("processing page %s", file_gt.pageId)
            pcgts_gt = page_from_file(self.workspace.download_file(file_gt))
            pcgts_dt = page_from_file(self.workspace.download_file(file_dt))
            page_gt = pcgts_gt.get_Page()
            page_dt = pcgts_dt.get_Page()
            if onlyfg:
                page_image, page_coords, _ = self.workspace.image_from_page(page_gt, file_gt.pageId,
                                                                            feature_selector='binarized')
                page_mask = ~ np.array(page_image)
            imgid = len(images)
            images.append({'file_name': file_gt.pageId,
                           'width': page_gt.get_imageWidth(),
                           'height': page_gt.get_imageHeight(),
                           })
            # read annotations from each page recursively (all categories including subtypes)
            # and merge GT and prediction categories
            for page, annotations in [(page_gt, annotations_gt), (page_dt, annotations_dt)]:
                for region in page.get_AllRegions(classes=None if level == 'region' else ['Text']):
                    if level == 'region':
                        cat = region.__class__.__name__[:-4]
                        if typed and hasattr(region, 'get_type') and region.get_type():
                            cat += '.' + region.get_type()
                        if cat not in categories:
                            categories.append(cat)
                        catid = categories.index(cat)
                        _add_annotation(annotations, region, imgid, catid,
                                        coords=page_coords if onlyfg else None,
                                        mask=page_mask if onlyfg else None)
                        continue
                    for line in region.get_TextLine():
                        _add_annotation(annotations, line, imgid, 1,
                                        coords=page_coords if onlyfg else None,
                                        mask=page_mask if onlyfg else None)

        if level == 'line':
            categories.append('textline')
        elif selected:
            selected = [categories.index(cat) for cat in selected if cat in categories]
        LOG.info(f"found {len(annotations_gt)} GT / {len(annotations_dt)} DT segments"
                 f" in {len(categories) - 1} categories for {len(images)} images")
        def add_ids(entries, start=0):
            for i, entry in enumerate(entries, start):
                if isinstance(entry, dict):
                    entry['id'] = i
                else:
                    entries[i] = {'id': i, 'name': entry}
        add_ids(categories)
        add_ids(images)
        add_ids(annotations_gt, 1) # cocoeval expects annotation IDs starting at 1
        add_ids(annotations_dt, 1) # cocoeval expects annotation IDs starting at 1
        coco_gt = COCO()
        coco_dt = COCO()
        coco_gt.dataset = {'categories': categories, 'images': images,
                           'annotations': annotations_gt}
        coco_dt.dataset = {'categories': categories, 'images': images,
                           'annotations': annotations_dt}
        with NoStdout():
            coco_gt.createIndex()
            coco_dt.createIndex()
        LOG.info("comparing segmentations")
        stats = dict(self.parameter)
        coco_eval = COCOeval(coco_gt, coco_dt, 'segm') # bbox
        if selected:
           coco_eval.params.catIds = selected
        #coco_eval.params.iouThrs = [.5:.05:.95]
        #coco_eval.params.iouThrs = np.linspace(.3, .95, 14)
        coco_eval.params.maxDets = [None] # unlimited nr of detections (requires pycocotools#559)
        #coco_eval.params.areaRng = [(0, np.inf)] # unlimited region size
        #coco_eval.params.areaRngLbl = ['all'] # unlimited region size
        # FIXME: cocoeval only allows/tracks 1-best (by confidence) GT match per DT,
        #        i.e. no way to detect undersegmentation
        # FIXME: somehow measure oversegmentation
        # FIXME: find way to get pixel-wise measures (IoU of matches, or IoGT-recall/IoDT-precision)
        coco_eval.evaluate()
        # get by-page alignment
        for img in coco_eval.evalImgs:
            if not img:
                continue
            if img['aRng'] != coco_eval.params.areaRng[0]:
                # ignore other restricted area ranges
                continue
            imgId = img['image_id']
            catId = img['category_id']
            image = images[imgId]
            pageId = image['file_name']
            cat = categories[catId]
            catName = cat['name']
            # get matches and ious and scores
            # (pick lowest overlap threshold iouThrs[0])
            gtMatches = img['gtMatches'][0].astype(np.int) # from gtind to matching DT annotation id
            dtMatches = img['dtMatches'][0].astype(np.int) # from dtind to matching GT annotation id
            dtScores = img['dtScores'] # from dtind to DT score
            gtIds = img['gtIds'] # from gtind to GT annotation id
            dtIds = img['dtIds'] # from dtind to DT annotation id
            gtIndices = np.zeros(max(gtIds, default=-1) + 1, np.int) # from GT annotation id to gtind
            for ind, id_ in enumerate(gtIds):
                gtIndices[id_] = ind
            dtIndices = np.zeros(max(dtIds, default=-1) + 1, np.int) # from DT annotation id to dtind
            for ind, id_ in enumerate(dtIds):
                dtIndices[id_] = ind
            ious = coco_eval.ious[imgId, catId] # each by dtind,gtind
            # record as dict by pageId / by category
            matches = stats.setdefault('matches', dict())
            imgMatches = matches.setdefault(pageId, dict())
            imgMatches[catName] = [(annotations_gt[gtIds[gtind] - 1]['segment_id'],
                                    annotations_dt[dtid - 1]['segment_id'],
                                    ious[dtIndices[dtid], gtind])
                                   for gtind, dtid in enumerate(gtMatches)
                                   if dtid > 0]
        coco_eval.accumulate()
        # get precision/recall at
        # T[0]=0.5 IoU
        # R[*] recall threshold equal to max recall
        # K[*] each class
        # A[0] all areas
        # M[-1]=100 max detections
        recalls = coco_eval.eval['recall'][0,:,0,-1]
        recallInds = np.searchsorted(np.linspace(0, 1, 101), recalls) - 1
        classInds = np.arange(len(recalls))
        precisions = coco_eval.eval['precision'][0, recallInds, classInds, 0, -1]
        classes = stats.setdefault('classes', dict())
        for cat in categories:
            classes[cat['name']] = {'precision': str(precisions[cat['id']]),
                                    'recall': str(recalls[cat['id']])}
        # FIXME: first row (AP for full IoU range) has fixed maxDets=100 in pycocotools, which we don't want
        coco_eval.summarize()
        statInds = np.ones(12, np.bool)
        statInds[6] = False # AR maxDet[0]
        statInds[7] = False # AR maxDet[1]
        coco_eval.stats = coco_eval.stats[statInds]
        stats['scores'] = dict(zip([
            'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all ]',
            'Average Precision  (AP) @[ IoU=0.50      | area=   all ]',
            'Average Precision  (AP) @[ IoU=0.75      | area=   all ]',
            'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small ]',
            'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium ]',
            'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large ]',
            'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all ]',
            'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small ]',
            'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium ]',
            'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large ]',
            ], coco_eval.stats.tolist()))

        # write regions to custom JSON for this page
        file_id = 'id' + self.output_file_grp + '_report'
        self.workspace.add_file(
            ID=file_id,
            file_grp=self.output_file_grp,
            pageId=None,
            local_filename=os.path.join(self.output_file_grp, file_id + '.json'),
            mimetype='application/json',
            content=json.dumps(stats, indent=2))
        # todo: also write report for each page

    def _compare_segmentation(self, coco_gt, coco_dt, page_id):
        LOG = getLogger('processor.EvaluateSegmentation')
        gt_regions = gt_page.get_TextRegion()
        pred_regions = pred_page.get_TextRegion()
        if len(gt_regions) != len(pred_regions):
            LOG.warning("page '%s': %d vs %d text regions",
                        page_id, len(gt_regions), len(pred_regions))
        # FIXME: add actual layout alignment and comparison

def _add_annotation(annotations, segment, imgid, catid, coords=None, mask=None):
    LOG = getLogger('processor.EvaluateSegmentation')
    score = segment.get_Coords().get_conf() or 1.0
    polygon = polygon_from_points(segment.get_Coords().points)
    if len(polygon) < 3:
        LOG.warning('ignoring segment "%s" with only %d points', segment.id, len(polygon))
        return
    if mask is None:
        segmentation = np.array(polygon).reshape(1, -1).tolist()
    else:
        polygon = coordinates_of_segment(segment, None, coords)
        py, px = draw.polygon(polygon[:,1], polygon[:,0], mask.shape)
        masked = np.zeros(mask.shape, dtype=np.uint8, order='F') # pycocotools.mask wants Fortran-contiguous arrays
        masked[py, px] = 1 * mask[py, px]
        segmentation = encodeMask(masked)
    xywh = xywh_from_polygon(polygon)
    annotations.append(
        {'segment_id': segment.id, # non-standard string-valued in addition to 'id'
         'image_id': imgid,
         'category_id': catid,
         'segmentation': segmentation,
         'area': Polygon(polygon).area,
         'bbox': [xywh['x'], xywh['y'], xywh['w'], xywh['h']],
         'score': score,
         'iscrowd': 0})

class NoStdout():
    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, etype, evalue, etrace):
        sys.stdout = self.stdout
        if etype is not None:
            return False # reraise

    def write(self, value):
        pass

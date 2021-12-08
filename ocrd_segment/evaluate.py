from __future__ import absolute_import

import sys
import os
import json
import click
import numpy as np
from skimage import draw
from PIL import Image
from shapely.geometry import Polygon

from ocrd import Processor
from ocrd_utils import (
    getLogger,
    initLogging,
    assert_file_grp_cardinality,
    xywh_from_polygon,
    polygon_from_points,
    coordinates_of_segment,
    MIMETYPE_PAGE
)
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import parse as parse_page

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import (
    encode as encodeMask,
    merge as mergeMasks,
    area as maskArea
)

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
            _add_annotations(annotations_gt, page_gt, imgid, categories,
                             level=level, typed=typed,
                             coords=page_coords if onlyfg else None,
                             mask=page_mask if onlyfg else None)
            _add_annotations(annotations_dt, page_dt, imgid, categories,
                             level=level, typed=typed,
                             coords=page_coords if onlyfg else None,
                             mask=page_mask if onlyfg else None)

        if level == 'line':
            categories.append('textline')
        elif selected:
            selected = [categories.index(cat) for cat in selected if cat in categories]
        _add_ids(categories)
        _add_ids(images)
        _add_ids(annotations_gt, 1) # cocoeval expects annotation IDs starting at 1
        _add_ids(annotations_dt, 1) # cocoeval expects annotation IDs starting at 1

        LOG.info(f"found {len(annotations_gt)} GT / {len(annotations_dt)} DT segments"
                 f" in {len(categories) - 1} categories for {len(images)} images")

        coco_gt = _create_coco(categories, images, annotations_gt)
        coco_dt = _create_coco(categories, images, annotations_dt)

        stats = evaluate_coco(coco_gt, coco_dt, self.parameter, selected)

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

@click.command()
@click.option('-G', '--gt-page-filelst', type=click.File('r'),
              help="list file of ground-truth page file paths")
@click.option('-D', '--dt-page-filelst', type=click.File('r'),
              help="list file of detection page file paths")
@click.option('-I', '--bin-img-filelst', type=click.File('r'),
              help="list file of binarized image file paths")
@click.option('-L', '--level-of-operation', type=click.Choice(['region', 'line']), default='region',
              help="hierarchy level of segments to compare")
@click.option('-T', '--ignore-subtype', is_flag=True,
              help="on region level, ignore @type distinction")
@click.option('-C', '--for-categories', default='', type=str,
              help="on region level, comma-separated list of category names to evaluate (empty for all)")
@click.option('-R', '--report-file', type=click.File('w'), default="eval.log",
              help="file name to write evaluation results to")
@click.argument('tabfile', type=click.File('r'), required=False)
def standalone_cli(gt_page_filelst,
                   dt_page_filelst,
                   bin_img_filelst,
                   level_of_operation,
                   ignore_subtype,
                   for_categories,
                   report_file,
                   tabfile):
    """Performs segmentation evaluation with pycocotools on the given PAGE-XML files.

    \b
    Open and deserialize PAGE files from the list files.
    Then iterate over the element hierarchy down to ``level-of-operation``.
    Aggregate and convert all pages' segmentation (coordinates and classes)
    to COCO:

    \b
    - On the region level, unless ``ignore-subtype``, differentiate segment
      classes by their `@type`, if applicable.
    - On the region level, unless ``for-categories`` is empty, select only
      segment classes in that (comma-separated) list.
    - If image files are given (as separate file list or in the 3rd column
      of the tab-separated list file), then for each PAGE file pair, use
      the foreground mask from the binarized image inside all segments for
      overlap calculations.

    \b
    Next, configure and run COCOEval for comparison of all pages. Show the 
    matching pairs (GT segment ID, prediction segment ID, IoU) for every
    overlap on each page.
    Also, calculate per-class precision and recall (at maximum recall).
    Finally, get the typical summary mean average precision / recall
    (but without restriction on the number of segments), and write all
    statistics to ``report-file``.

    \b
    Write a JSON report to the output file group.
    """
    assert (tabfile is None) == (gt_page_filelst is not None) == (dt_page_filelst is not None), \
        "pass file lists either as tab-separated single file or as separate files"
    if tabfile is None:
        gt_page_files = [line.strip() for line in gt_page_filelst.readlines()]
        dt_page_files = [line.strip() for line in dt_page_filelst.readlines()]
        assert len(gt_page_files) == len(dt_page_files), \
            "number of DT files must match number of GT files"
        if bin_img_filelst is not None:
            bin_img_files = [line.strip() for line in bin_img_filelst.readlines()]
            assert len(bin_img_files) == len(gt_page_files), \
                "number of image files must match number of GT files"
        else:
            bin_img_files = None
    else:
        files = [line.strip().split('\t') for line in tabfile.readlines()]
        assert len(files), "list of files is empty"
        len0 = len(files[0])
        assert 2 <= len0 <= 3, "list of files must be tab-separated (GT, DT[, bin-img])"
        assert all(map(lambda line: len(line) == len0, files)), \
            "number of DT files must match number of GT files"
        if len0 == 2:
            gt_page_files, dt_page_files = zip(*files)
            bin_img_files = None
        else:
            gt_page_files, dt_page_files, bin_img_files = zip(*files)
    stats = evaluate_files(gt_page_files,
                           dt_page_files,
                           bin_img_files,
                           level_of_operation,
                           not ignore_subtype,
                           for_categories)
    json.dump(stats, report_file, indent=2)

# standalone entry point
def evaluate_files(gt_files, dt_files, img_files=None, level='region', typed=True, selected=None):
    initLogging()
    LOG = getLogger('processor.EvaluateSegmentation')
    categories = ["bg"] # needed by cocoeval
    images = []
    annotations_gt = []
    annotations_dt = []
    for gt_file, dt_file, img_file in zip(gt_files, dt_files,
                                          img_files or [None] * len(gt_files)):
        pcgts_gt = parse_page(gt_file)
        pcgts_dt = parse_page(dt_file)
        page_id = pcgts_gt.pcGtsId or gt_file
        LOG.info("processing page %s", page_id)
        page_gt = pcgts_gt.get_Page()
        page_dt = pcgts_dt.get_Page()
        if img_file:
            page_image = Image.open(img_file)
            assert page_image.mode == '1', "input images must already be binarized"
            assert page_image.width - 2 < page_gt.get_imageWidth() < page_image.width + 2, \
                "mismatch between width of binary image and PAGE description"
            assert page_image.height - 2 < page_gt.get_imageHeight() < page_image.height + 2, \
                "mismatch between height of binary image and PAGE description"
            page_mask = ~ np.array(page_image)
            page_coords = {"transform": np.eye(3), "angle": 0, "features": "binarized"}
        imgid = len(images)
        images.append({'file_name': page_id,
                       'width': page_gt.get_imageWidth(),
                       'height': page_gt.get_imageHeight(),
        })
        # read annotations from each page recursively (all categories including subtypes)
        # and merge GT and prediction categories
        _add_annotations(annotations_gt, page_gt, imgid, categories,
                         level=level, typed=typed,
                         coords=page_coords if img_file else None,
                         mask=page_mask if img_file else None)
        _add_annotations(annotations_dt, page_dt, imgid, categories,
                         level=level, typed=typed,
                         coords=page_coords if img_file else None,
                         mask=page_mask if img_file else None)

    if level == 'line':
        categories.append('textline')
    elif selected:
        selected = [categories.index(cat) for cat in selected if cat in categories]
    _add_ids(categories)
    _add_ids(images)
    _add_ids(annotations_gt, 1) # cocoeval expects annotation IDs starting at 1
    _add_ids(annotations_dt, 1) # cocoeval expects annotation IDs starting at 1

    LOG.info(f"found {len(annotations_gt)} GT / {len(annotations_dt)} DT segments"
             f" in {len(categories) - 1} categories for {len(images)} images")

    coco_gt = _create_coco(categories, images, annotations_gt)
    coco_dt = _create_coco(categories, images, annotations_dt)

    parameters = {"level-of-operation": level,
                  "only-fg": bool(img_files),
                  "ignore-subtype": not typed,
                  "for-categories": selected}
    stats = evaluate_coco(coco_gt, coco_dt, parameters, selected)
    return stats

def evaluate_coco(coco_gt, coco_dt, parameters, catIds=None):
    LOG = getLogger('processor.EvaluateSegmentation')
    LOG.info("comparing segmentations")
    stats = dict(parameters)
    # cocoeval only allows/tracks 1-best (by confidence) GT match per DT,
    # therefore, to detect undersegmentation, we need to run inverse direction, too
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm') # bbox
    coco_lave = COCOeval(coco_dt, coco_gt, 'segm') # bbox
    if catIds:
       coco_eval.params.catIds = catIds
       coco_lave.params.catIds = catIds
    #coco_eval.params.iouThrs = [.5:.05:.95]
    #coco_eval.params.iouThrs = np.linspace(.3, .95, 14)
    coco_eval.params.maxDets = [None] # unlimited nr of detections (requires pycocotools#559)
    coco_lave.params.maxDets = [None] # unlimited nr of detections (requires pycocotools#559)
    #coco_eval.params.areaRng = [(0, np.inf)] # unlimited region size
    #coco_eval.params.areaRngLbl = ['all'] # unlimited region size
    # FIXME: the IoU threshold criterion is inadequate for flat segmentation, because over-/undersegmentation can become false negative/positive (but pixel-wise measures do not distinguish instances)
    #        (perhaps we can run with iouThrs=0.1, but then filter eval.ious w.r.t. GT and DT areas?)
    coco_eval.evaluate()
    coco_lave.evaluate()
    # get by-page alignment
    for imgind, img in enumerate(coco_eval.evalImgs):
        if not img:
            continue
        if img['aRng'] != coco_eval.params.areaRng[0]:
            # ignore other restricted area ranges
            continue
        if img['maxDet'] != None:
            # ignore restricted number of detections
            continue
        imgId = img['image_id']
        catId = img['category_id']
        image = coco_gt.imgs[imgId]
        pageId = image['file_name']
        cat = coco_gt.cats[catId]
        catName = cat['name']
        # get matches and ious and scores
        # (pick lowest overlap threshold iouThrs[0])
        gtMatches = img['gtMatches'][0].astype(np.int) # from gtind to matching DT annotation id
        dtMatches = img['dtMatches'][0].astype(np.int) # from dtind to matching GT annotation id
        dtScores = img['dtScores'] # from dtind to DT score
        gtIds = img['gtIds'] # from gtind to GT annotation id
        dtIds = img['dtIds'] # from dtind to DT annotation id
        # we can ignore gtIgnore here, because we only look at areaRng[0]=all
        # we can ignore dtIgnore here, because we only look at maxDet=None
        gtIndices = np.zeros(max(gtIds, default=-1) + 1, np.int) # from GT annotation id to gtind
        for ind, id_ in enumerate(gtIds):
            gtIndices[id_] = ind
        dtIndices = np.zeros(max(dtIds, default=-1) + 1, np.int) # from DT annotation id to dtind
        for ind, id_ in enumerate(dtIds):
            dtIndices[id_] = ind
        ious = coco_eval.ious[imgId, catId] # each by dtind,gtind
        # record as dict by pageId / by category
        imgstats = stats.setdefault('by-image', dict())
        pagestats = imgstats.setdefault(pageId, dict())
        pagestatsTP = pagestats.setdefault('true_positives', dict())
        pagestatsTP[catName] = list()
        # aggregate per-img/per-cat IoUs for microaveraging
        img['IoUs'] = list()
        img['IoGTs'] = list()
        img['IoDTs'] = list()
        for dtind, gtid in enumerate(dtMatches):
            if gtid <= 0:
                img['IoGTs'].append(0.0)
                img['IoDTs'].append(1.0)
                continue
            gtind = gtIndices[gtid]
            dtid = dtIds[dtind]
            gtann = coco_gt.anns[gtid]
            dtann = coco_dt.anns[dtid]
            iou = ious[dtind, gtind]
            union = maskArea(mergeMasks([gtann['segmentation'], dtann['segmentation']]))
            gtarea = maskArea(gtann['segmentation'])
            dtarea = maskArea(dtann['segmentation'])
            pagestatsTP[catName].append({'GT.ID': gtann['segment_id'],
                                         'DT.ID': dtann['segment_id'],
                                         'IoGT': iou * union / gtarea,
                                         'IoDT': iou * union / dtarea,
                                         'IoU': iou})
            img['IoGTs'].append(iou * union / gtarea)
            img['IoDTs'].append(iou * union / dtarea)
            img['IoUs'].append(iou)
        pagestatsFP = pagestats.setdefault('false_positives', dict())
        pagestatsFP[catName] = [coco_dt.anns[dtid]['segment_id']
                                for dtid in dtIds
                                if all(gtMatches != dtid)]
        pagestatsFN = pagestats.setdefault('false_negatives', dict())
        pagestatsFN[catName] = [coco_gt.anns[gtid]['segment_id']
                                for gtid in gtIds
                                if all(dtMatches != gtid)]
        # measure oversegmentation for this image and category
        # (follows Zhang et al 2021: Rethinking Semantic Segmentation Evaluation [arXiv:2101.08418])
        pagestatsOS = pagestats.setdefault('oversegmentation', dict())
        over_gt = set()
        over_dt = set()
        over_degree = 0
        unique_gtids, unique_dtinds = np.unique(dtMatches, return_inverse=True)
        for gtid in unique_gtids:
            dtinds = np.nonzero(unique_dtinds == gtid)[0]
            if len(dtinds) > 1:
                for dtind in dtinds:
                    over_dt.add(dtIds[dtind])
                over_gt.add(gtid)
                over_degree += len(dtinds) - 1
        if len(gtIds) and len(dtIds):
            oversegmentation = len(over_gt) * len(over_dt) / len(gtIds) / len(dtIds)
            # Zhang's idea of attenuating the oversegmentation ratio with a "penalty"
            # to account for the degree of further sub-segmentation is misguided IMHO:
            # - its degree term depends on the total number of segments
            # - our iouThr-based pairing does not find higher degrees anyway
            # oversegmentation = np.tanh(oversegmentation * over_degree)
        else:
            oversegmentation = 0
        pagestatsOS[catName] = oversegmentation
        # aggregate per-img/per-cat measures for microaveraging
        img['dtIdsOver'] = list(over_dt)
        img['gtIdsOver'] = list(over_gt)
    # inverse direction to measure undersegmentation for this image and category
    for imgind, img in enumerate(coco_lave.evalImgs):
        if not img:
            continue
        if img['aRng'] != coco_lave.params.areaRng[0]:
            # ignore other restricted area ranges
            continue
        if img['maxDet'] != None:
            # ignore restricted number of detections
            continue
        imgId = img['image_id']
        catId = img['category_id']
        image = coco_gt.imgs[imgId]
        pageId = image['file_name']
        cat = coco_gt.cats[catId]
        catName = cat['name']
        gtMatches = img['dtMatches'][0].astype(np.int) # from gtind to matching DT annotation id
        gtIds = img['dtIds'] # from gtind to GT annotation id
        pagestats = imgstats.setdefault(pageId, dict())
        pagestatsUS = pagestats.setdefault('undersegmentation', dict())
        under_gt = set()
        under_dt = set()
        under_degree = 0
        unique_dtids, unique_gtinds = np.unique(gtMatches, return_inverse=True)
        for dtid in unique_dtids:
            gtinds = np.nonzero(unique_gtinds == dtid)[0]
            if len(gtinds) > 1:
                for gtind in gtinds:
                    under_gt.add(gtIds[gtind])
                under_dt.add(dtid)
                under_degree += len(gtinds) - 1
        if len(dtIds) and len(gtIds):
            undersegmentation = len(under_gt) * len(under_dt) / len(gtIds) / len(dtIds)
            # Zhang's idea of attenuating the undersegmentation ratio with a "penalty"
            # to account for the degree of further sub-segmentation is misguided IMHO:
            # - its degree term depends on the total number of segments
            # - our iouThr-based pairing does not find higher degrees anyway
            # undersegmentation = np.tanh(undersegmentation * under_degree)
        else:
            undersegmentation = 0
        pagestatsUS[catName] = undersegmentation
        # aggregate per-img/per-cat measures for microaveraging
        img = coco_eval.evalImgs[imgind] # we throw away coco_lave here
        img['dtIdsUnder'] = list(under_dt)
        img['gtIdsUnder'] = list(under_gt)

    coco_eval.accumulate()
    # get precision/recall at
    # T[0]=0.5 IoU
    # R[*] recall threshold equal to max recall
    # K[*] each class
    # A[0] all areas
    # M[-1]=all detections
    recalls = coco_eval.eval['recall'][0,:,0,-1]
    recallInds = np.searchsorted(np.linspace(0, 1, 101), recalls) - 1
    classInds = np.arange(len(recalls))
    precisions = coco_eval.eval['precision'][0, recallInds, classInds, 0, -1]
    catstats = stats.setdefault('by-category', dict())
    for cat in coco_gt.cats.values():
        catstats[cat['name']] = {'precision': str(precisions[cat['id']]),
                                 'recall': str(recalls[cat['id']])}
    # accumulate our over-/undersegmentation and IoU ratios
    numImgs = len(coco_eval.params.imgIds)
    numAreas = len(coco_eval.params.areaRng)
    for catind, catId in enumerate(coco_eval.params.catIds):
        cat = coco_gt.cats[catId]
        start = catind * numImgs * numAreas
        # again, we stay at areaRng[0]=all and maxDets[0]=all
        evalimgs = [coco_eval.evalImgs[start + imgind] for imgind in range(numImgs)]
        evalimgs = [img for img in evalimgs if img is not None]
        assert all(img['category_id'] == catId for img in evalimgs)
        assert all(img['maxDet'] is None for img in evalimgs)
        assert all(img['aRng'] == coco_eval.params.areaRng[0] for img in evalimgs)
        if not len(evalimgs):
            continue
        # again, we can ignore gtIgnore here, because we only look at areaRng[0]=all
        # again, we can ignore dtIgnore here, because we only look at maxDet=None
        numDTs = sum(len(img['dtIds']) for img in evalimgs)
        numGTs = sum(len(img['gtIds']) for img in evalimgs)
        overDTs = sum(len(img['dtIdsOver']) for img in evalimgs)
        overGTs = sum(len(img['gtIdsOver']) for img in evalimgs)
        underDTs = sum(len(img['dtIdsUnder']) for img in evalimgs)
        underGTs = sum(len(img['gtIdsUnder']) for img in evalimgs)
        sumIoUs = sum(sum(img['IoUs']) for img in evalimgs)
        sumIoGTs = sum(sum(img['IoGTs']) for img in evalimgs)
        sumIoDTs = sum(sum(img['IoDTs']) for img in evalimgs)
        if numDTs and numGTs:
            oversegmentation = overDTs * overGTs / numDTs / numGTs
            undersegmentation = underDTs * underGTs / numDTs / numGTs
            iou = sumIoUs / numDTs
            iogt = sumIoGTs / numDTs
            iodt = sumIoDTs / numDTs
        else:
            oversegmentation = undersegmentation = 0
            iou = iogt = iodt = 0
        catstats[cat['name']]['oversegmentation'] = oversegmentation
        catstats[cat['name']]['undersegmentation'] = undersegmentation
        catstats[cat['name']]['IoGT'] = iogt
        catstats[cat['name']]['IoDT'] = iodt
        catstats[cat['name']]['IoU'] = iou

    coco_eval.summarize()
    statInds = np.ones(12, np.bool)
    statInds[7] = False # AR maxDet[1]
    statInds[8] = False # AR maxDet[2]
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
    return stats

def _create_coco(categories, images, annotations):
    coco = COCO()
    coco.dataset = {'categories': categories, 'images': images, 'annotations': annotations}
    with NoStdout():
        coco.createIndex()
    return coco

def _add_annotations(annotations, page, imgid, categories,
                     level='region', typed=True, coords=None, mask=None):
    for region in page.get_AllRegions(classes=None if level == 'region' else ['Text']):
        if level == 'region':
            cat = region.__class__.__name__[:-4]
            if typed and hasattr(region, 'get_type') and region.get_type():
                cat += '.' + region.get_type()
            if cat not in categories:
                categories.append(cat)
            catid = categories.index(cat)
            _add_annotation(annotations, region, imgid, catid,
                            coords=coords, mask=mask)
            continue
        for line in region.get_TextLine():
            _add_annotation(annotations, line, imgid, 1,
                            coords=coords, mask=mask)

def _add_annotation(annotations, segment, imgid, catid, coords=None, mask=None):
    LOG = getLogger('processor.EvaluateSegmentation')
    score = segment.get_Coords().get_conf() or 1.0
    polygon = polygon_from_points(segment.get_Coords().points)
    if len(polygon) < 3:
        LOG.warning('ignoring segment "%s" with only %d points', segment.id, len(polygon))
        return
    xywh = xywh_from_polygon(polygon)
    if mask is None:
        segmentation = np.array(polygon).reshape(1, -1).tolist()
    else:
        polygon = coordinates_of_segment(segment, None, coords)
        py, px = draw.polygon(polygon[:,1], polygon[:,0], mask.shape)
        masked = np.zeros(mask.shape, dtype=np.uint8, order='F') # pycocotools.mask wants Fortran-contiguous arrays
        masked[py, px] = 1 * mask[py, px]
        segmentation = encodeMask(masked)
    annotations.append(
        {'segment_id': segment.id, # non-standard string-valued in addition to 'id'
         'image_id': imgid,
         'category_id': catid,
         'segmentation': segmentation,
         'area': Polygon(polygon).area,
         'bbox': [xywh['x'], xywh['y'], xywh['w'], xywh['h']],
         'score': score,
         'iscrowd': 0})

def _add_ids(entries, start=0):
    for i, entry in enumerate(entries, start):
        if isinstance(entry, dict):
            entry['id'] = i
        else:
            entries[i] = {'id': i, 'name': entry}

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

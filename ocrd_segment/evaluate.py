from __future__ import absolute_import

import sys
import os
from typing import List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import OrderedDict as odict
import json
from itertools import chain, combinations
import click
import numpy as np
import networkx as nx
from skimage import draw
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components
from scipy.stats import gmean
from PIL import Image
from shapely.geometry import Polygon

from ocrd import Workspace, Processor
from ocrd_utils import (
    getLogger,
    initLogging,
    xywh_from_polygon,
    polygon_from_points,
    points_from_polygon,
    coordinates_of_segment,
    MIMETYPE_PAGE,
    pushd_popd,
    make_file_id
)
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    parse as parse_page,
    to_xml,
    CoordsType,
    OrderedGroupType,
    OrderedGroupIndexedType,
    UnorderedGroupType,
    UnorderedGroupIndexedType,
    RegionRefType,
    RegionRefIndexedType,
    ReadingOrderType,
)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import (
    encode as encodeMask,
    decode as decodeMask,
    merge as mergeMasks,
    iou as iouMasks,
    area as maskArea
)

from .project import make_intersection, make_valid

# PRImA constants
ALLOWANCE = 10 # pixels bboxes may overlap in direction of merge/split to still be allowable

class EvaluateSegmentation(Processor):

    @property
    def executable(self):
        return 'ocrd-segment-evaluate'

    def process_workspace(self, workspace: Workspace) -> None:
        """Performs segmentation evaluation with pycocotools on the workspace.

        Open and deserialize PAGE files from the first and second input file group
        (the first as ground truth, the second as prediction).
        Then iterate over the element hierarchy down to ``level-of-operation``.
        Aggregate and convert all pages' segmentation (coordinates and classes)
        to COCO:

        \b
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
        # region or line level?
        level = self.parameter['level-of-operation']
        onlyfg = self.parameter['only-fg']
        typed = not self.parameter['ignore-subtype']
        selected = self.parameter['for-categories']
        if selected:
            selected = selected.split(',')
        self.workspace = workspace
        self.verify()
        # FIXME: add configurable error handling as in super().process_workspace()
        # get input file groups
        ifgs = self.input_file_grp.split(",")
        # get input file tuples
        ifts = self.zip_input_files(mimetype=MIMETYPE_PAGE, require_first=False)
        # convert to 2 COCO datasets from all page pairs
        categories = ["bg"] # needed by cocoeval
        images = []
        annotations_gt = []
        annotations_dt = []
        for ift in ifts:
            file_gt, file_dt = ift
            if not file_gt:
                self.logger.warning("skipping page %s missing from GT", file_gt.pageId)
                continue
            if not file_dt:
                self.logger.warning("skipping page %s missing from prediction", file_gt.pageId)
                continue
            self.logger.info("processing page %s", file_gt.pageId)
            if self.download:
                file_gt = self.workspace.download_file(file_gt)
                file_dt = self.workspace.download_file(file_dt)
            with pushd_popd(self.workspace.directory):
                pcgts_gt = page_from_file(file_gt)
                pcgts_dt = page_from_file(file_dt)
            page_gt = pcgts_gt.get_Page()
            page_dt = pcgts_dt.get_Page()
            if onlyfg:
                page_image, page_coords, _ = self.workspace.image_from_page(
                    page_gt, file_gt.pageId,
                    feature_selector='binarized',
                    feature_filter='clipped')
                page_mask = ~ np.array(page_image.convert('L'))
            imgid = len(images)
            images.append({'file_name': file_gt.pageId,
                           'pcgts': (pcgts_gt, pcgts_dt),
                           'width': page_gt.get_imageWidth(),
                           'height': page_gt.get_imageHeight(),
                           })
            # read annotations from each page recursively (all categories including subtypes)
            # and merge GT and prediction categories
            _add_annotations(annotations_gt, page_gt, imgid, categories,
                             level=level, typed=typed,
                             coords=page_coords if onlyfg else None,
                             mask=page_mask if onlyfg else None,
                             log=self.logger)
            _add_annotations(annotations_dt, page_dt, imgid, categories,
                             level=level, typed=typed,
                             coords=page_coords if onlyfg else None,
                             mask=page_mask if onlyfg else None,
                             log=self.logger)

        if level == 'line':
            categories.append('textline')
        elif selected:
            selected = [categories.index(cat) for cat in selected if cat in categories]
        _add_ids(categories)
        _add_ids(images)
        _add_ids(annotations_gt, 1) # cocoeval expects annotation IDs starting at 1
        _add_ids(annotations_dt, 1) # cocoeval expects annotation IDs starting at 1

        self.logger.info(f"found {len(annotations_gt)} GT / {len(annotations_dt)} DT segments"
                         f" in {len(categories) - 1} categories for {len(images)} images")

        coco_gt = _create_coco(categories, images, annotations_gt)
        coco_dt = _create_coco(categories, images, annotations_dt)

        stats = evaluate_coco(coco_gt, coco_dt, self.parameter, selected, log=self.logger)
        for ift in ifts:
            file_gt, file_dt = ift
            pagestats = stats['by-image'][file_gt.pageId]
            pcgts = pagestats.pop('pcgts')
            self.add_metadata(pcgts)
            file_id = make_file_id(file_gt, self.output_file_grp)
            workspace.add_file(
                self.output_file_grp,
                ID=file_id,
                page_id=file_gt.pageId,
                local_filename=os.path.join(self.output_file_grp, file_id + '.xml'),
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts))

        # write regions to custom JSON for this page
        file_id = 'id' + self.output_file_grp + '_report'
        workspace.add_file(
            self.output_file_grp,
            ID=file_id,
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
    initLogging()
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
    log = getLogger('ocrd.processor.EvaluateSegmentation')
    categories = ["bg"] # needed by cocoeval
    images = []
    annotations_gt = []
    annotations_dt = []
    for gt_file, dt_file, img_file in zip(gt_files, dt_files,
                                          img_files or [None] * len(gt_files)):
        pcgts_gt = parse_page(gt_file)
        pcgts_dt = parse_page(dt_file)
        page_id = pcgts_gt.pcGtsId or gt_file
        log.info("processing page %s", page_id)
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
                         mask=page_mask if img_file else None,
                         log=log)
        _add_annotations(annotations_dt, page_dt, imgid, categories,
                         level=level, typed=typed,
                         coords=page_coords if img_file else None,
                         mask=page_mask if img_file else None,
                         log=log)

    if level == 'line':
        categories.append('textline')
    elif selected:
        selected = [categories.index(cat) for cat in selected if cat in categories]
    _add_ids(categories)
    _add_ids(images)
    _add_ids(annotations_gt, 1) # cocoeval expects annotation IDs starting at 1
    _add_ids(annotations_dt, 1) # cocoeval expects annotation IDs starting at 1

    log.info(f"found {len(annotations_gt)} GT / {len(annotations_dt)} DT segments"
             f" in {len(categories) - 1} categories for {len(images)} images")

    coco_gt = _create_coco(categories, images, annotations_gt)
    coco_dt = _create_coco(categories, images, annotations_dt)

    parameters = {"level-of-operation": level,
                  "only-fg": bool(img_files),
                  "ignore-subtype": not typed,
                  "for-categories": selected}
    stats = evaluate_coco(coco_gt, coco_dt, parameters, selected)
    return stats

def evaluate_coco(coco_gt, coco_dt, parameters, catIds=None, log=None, non_allowable_metric="max"):
    if log is None:
        log = getLogger('ocrd.processor.EvaluateSegmentation')
    log.setLevel(10)
    log.info("comparing segmentations")
    stats = dict(parameters)
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm') # bbox
    if catIds:
       coco_eval.params.catIds = catIds
    #coco_eval.params.iouThrs = [.5:.05:.95]
    #coco_eval.params.iouThrs = np.linspace(.3, .95, 14)
    coco_eval.params.maxDets = [None] # unlimited nr of detections (requires pycocotools#559)
    #coco_eval.params.areaRng = [(0, np.inf)] # unlimited region size
    #coco_eval.params.areaRngLbl = ['all'] # unlimited region size
    # Note: The IoU threshold criterion is inadequate for flat segmentation,
    #       because over-/undersegmentation can quickly become false negative/positive.
    #       The pycocotools implementation is especially inadequate, because
    #       it only counts 1:1 matches (and not even the largest or best-scoring, #564).
    #       On the other hand, purely pixel-wise measures do not distinguish instances,
    #       i.e. neighbours can quickly become merged or instances torn apart.
    #       Our approach therefore does not build on pycocotools for matching
    #       and aggregation, only for fast IoU calculation. All non-zero pairs
    #       are considered matches if their intersection over union > 0.5 _or_
    #       their intersection over either side > 0.5. Matches can thus be n:m.
    #       Non-matches are counted as well (false positives and false negatives).
    #       Aggregation uses microaveraging over images. Besides counting segments,
    #       the pixel areas are counted and averaged (as ratios).
    coco_eval.evaluate()
    # get by-page alignment (ignoring inadequate 1:1 matching by pycocotools)
    numImgs = len(coco_eval.params.imgIds)
    numAreas = len(coco_eval.params.areaRng)
    for imgind, imgId in enumerate(coco_eval.params.imgIds):
        img = coco_gt.imgs[imgId]
        pageId = img['file_name']
        pcgts_gt, pcgts_dt = img['pcgts']
        pcgts = page_from_file(pcgts_gt.Page.imageFilename)
        for catind, catId in enumerate(coco_eval.params.catIds):
            if not catId:
                continue
            cat = coco_gt.cats[catId]
            catName = cat['name']
            catClass = getattr(sys.modules['ocrd_models.ocrd_page'], catName + 'Type')
            catAdder = getattr(pcgts.Page, 'add_' + catName)
            # bypassing COCOeval.evaluateImg, hook onto its results
            # (again, we stay at areaRng[0]=all and maxDets[0]=all)
            start = catind * numImgs * numAreas
            evalimg = coco_eval.evalImgs[start + imgind]
            if evalimg is None:
                continue # no DT and GT here
            # record as dict by pageId / by category
            imgstats = stats.setdefault('by-image', odict())
            pagestats = imgstats.setdefault(pageId, odict())
            pagestats['pcgts'] = pcgts
            # create keys in well-defined order
            pagestats.setdefault('true_positives', odict())
            pagestats.setdefault('false_positives', odict())
            pagestats.setdefault('false_negatives', odict())
            pagestats.setdefault('precision', odict())
            pagestats.setdefault('recall', odict())
            pagestats.setdefault('merges', odict())
            pagestats.setdefault('splits', odict())
            pagestats.setdefault('undersegmentation', odict())
            pagestats.setdefault('oversegmentation', odict())
            pagestats.setdefault('pixel-precision', odict())
            pagestats.setdefault('pixel-recall', odict())
            pagestats.setdefault('pixel-nonallowable-merge-rate', odict())
            pagestats.setdefault('pixel-nonallowable-split-rate', odict())
            pagestats.setdefault('pixel-iou', odict())
            # get matches and ious and scores
            ious = coco_eval.ious[imgId, catId]
            if len(ious):
                overlaps_dt, overlaps_gt = ious.nonzero()
            else:
                overlaps_dt = overlaps_gt = []
            # reconstruct score sorting in computeIoU
            gt = coco_eval._gts[imgId, catId]
            dt = coco_eval._dts[imgId, catId]
            log.debug("page %s cat %s has %d overlaps between %d GT and %d detected segments",
                      pageId, catName, len(ious), len(gt), len(dt))
            dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
            dt = [dt[i] for i in dtind]
            matches = dict()
            gtmatches = dict()
            dtmatches = dict()
            for dtind, gtind in zip(overlaps_dt, overlaps_gt):
                d = dt[dtind]
                g = gt[gtind]
                iou = ious[dtind, gtind]
                union = maskArea(mergeMasks([g['segmentation'], d['segmentation']]))
                intersection = int(iou * union)
                # cannot use g or d['area'] here, because mask might be fractional (only-fg) instead of outline
                areag = int(maskArea(g['segmentation']))
                aread = int(maskArea(d['segmentation']))
                iogt = intersection / areag
                iodt = intersection / aread
                if iou < 0.5 and iogt < 0.5 and iodt < 0.5:
                    continue
                gseg = pcgts_gt.revmap[pcgts_gt.xpath(f"//*[@id='{g['segment_id']}']")[0]]
                dseg = pcgts_dt.revmap[pcgts_dt.xpath(f"//*[@id='{d['segment_id']}']")[0]]
                poly = make_intersection(make_valid(Polygon(polygon_from_points(gseg.Coords.points))),
                                         make_valid(Polygon(polygon_from_points(dseg.Coords.points))))
                catAdder(catClass(id='TP_' + str(g['id']) + '+' + str(d['id']),
                                  Coords=CoordsType(points=points_from_polygon(poly.exterior.coords[:-1])),
                                  comments=f"GT.ID={g['segment_id']},\n"
                                           f"DT.ID={d['segment_id']},\n"
                                           f"IoGT={iogt}, IoDT={iodt},\n"
                                           f"IoU={iou}, I.area={intersection}"))
                gtmatches.setdefault(gtind, list()).append(dtind)
                dtmatches.setdefault(dtind, list()).append(gtind)
                matches[gtind, dtind] = Match(g, d, iogt, iodt, iou, intersection)
                pagestats['true_positives'].setdefault(catName, list()).append(
                    {'GT.ID': g['segment_id'],
                     'DT.ID': d['segment_id'],
                     'GT.area': areag,
                     'DT.area': aread,
                     'I.area': intersection,
                     'IoGT': iogt,
                     'IoDT': iodt,
                     'IoU': iou})
            dtmisses = []
            for dtind, d in enumerate(dt):
                if dtind in dtmatches:
                    continue
                aread = int(maskArea(d['segmentation']))
                segd = pcgts_dt.revmap[pcgts_dt.xpath(f"//*[@id='{d['segment_id']}']")[0]]
                catAdder(catClass(id='FP_' + str(d['id']),
                                  Coords=CoordsType(points=segd.Coords.points),
                                  comments=f"DT.ID={d['segment_id']}"))
                dtmisses.append(Nonmatch(d['id'], aread))
                pagestats['false_positives'].setdefault(catName, list()).append(
                    {'DT.ID': d['segment_id'],
                     'area': aread})
            gtmisses = []
            for gtind, g in enumerate(gt):
                if gtind in gtmatches:
                    continue
                areag = int(maskArea(g['segmentation']))
                segg = pcgts_gt.revmap[pcgts_gt.xpath(f"//*[@id='{g['segment_id']}']")[0]]
                catAdder(catClass(id='FN_' + str(g['id']),
                                  Coords=CoordsType(points=segg.Coords.points),
                                  comments=f"GT.ID={g['segment_id']}"))
                gtmisses.append(Nonmatch(g['id'], areag))
                pagestats['false_negatives'].setdefault(catName, list()).append(
                    {'GT.ID': g['segment_id'],
                     'area': areag})
            # Differentiate between allowable and non-allowable over/under-segmentation (i.e. splits/merges)
            # (following Clausner et al 2011: Scenario Driven In-Depth Performance Evaluation
            #  of Document Layout Analysis Methods):
            #         A region's split is allowable iff it flows in the textLineOrder of the respective GT,
            #         i.e. lines are likely to be either on one side or the other, but not both.
            #         For top-to-bottom/bottom-to-top regions, vertical splits are allowable.
            #         For left-to-right/right-to-left regions, horizontal splits are allowable.
            #         FIXME: To be sure, we could also validate that explicitly – 
            #                i.e. evaluating both levels at the same time. Thus,
            #                a split would be allowable iff its constituent lines
            #                are preserved and follow up in the same order.
            #                (And if not, only the deviating lines should "count".)
            #         Analogously, a group of regions' merge is allowable iff it flows in the textLineOrder
            #         of them all, and the GT global reading order has no other regions in between.
            #         For top-to-bottom/bottom-to-top regions, vertical merges are allowable.
            #         For left-to-right/right-to-left regions, horizontal merges are allowable.
            #         FIXME: Again, we could also validate that the overall textline flow is equivalent.
            #         Crucially, this relation is transitive, so every merge or split yields a partition
            #         of in-order groups that are mutually incompatible.
            #
            #        This can in turn be used to weigh a match pair's score accordingly when aggregating:
            #        For precision-like scores, we rule out non-allowable merges (by counting them as FP),
            #        and for recall-like scores, we rule out non-allowable splits (by counting them as FN).
            #
            #        FIXME: We could also weigh these non-allowable cases by their share of height
            #        (in vertical textLineOrder and horizontal writing) or width
            #        (in horizontal textLineOrder and vertical writing) which is in disagreement,
            #        or the share of its textlines that have been split or lost.
            #        FIXME:
            #        Furthermore, we can weigh matches by the share of non-text regions or fg pixels involved.
            merges = {}
            for dtind in dtmatches:
                #areag = int(maskArea(g['segmentation']))
                merge = dtmatches[dtind]
                if len(merge) <= 1:
                    continue
                pairs = combinations(range(len(merge)), 2)
                graph = np.zeros((len(merge), len(merge)), dtype=bool)
                for indi, indj in pairs:
                    if indi == indj:
                        continue
                    gt1ind = merge[indi]
                    gt2ind = merge[indj]
                    d = dt[dtind]
                    g1 = gt[gt1ind]
                    g2 = gt[gt2ind]
                    # if catName in ['SeparatorRegion', 'GraphicRegion.decoration']:
                    #     # FIXME: others: we need all IOUs for non-sep vs. sep, then filter for this d
                    #     allowable = _allowable_merge_separator(g1, g2, d, others,
                    #                                            log=log)
                    # else:
                    allowable = _allowable_merge(g1, g2, d, log=log)
                    if allowable:
                        graph[indi, indj] = True
                n_allowed, labels_allowed = connected_components(csgraph=csr_array(graph),
                                                                 directed=False, return_labels=True)
                partition = []
                for label in range(n_allowed):
                    gtinds = {merge[ind] for ind in np.flatnonzero(labels_allowed == label)}
                    partition.append(tuple(gtinds))
                    if len(gtinds) > 1:
                        log.info("allowable merge: %s", '+'.join([gt[gtind]['segment_id']
                                                                  for gtind in gtinds]))
                if n_allowed > 1:
                    log.info("non-allowable merge: %s", '|'.join('+'.join([gt[gtind]['segment_id']
                                                                           for gtind in gtinds])
                                                                 for gtinds in partition))
                merges[dtind] = Mismatch(merge, [dtind], partition)
            splits = {}
            for gtind in gtmatches:
                #aread = int(maskArea(d['segmentation']))
                split = gtmatches[gtind]
                if len(split) <= 1:
                    continue
                pairs = combinations(range(len(split)), 2)
                graph = np.zeros((len(split), len(split)), dtype=bool)
                for indi, indj in pairs:
                    if indi == indj:
                        continue
                    dt1ind = split[indi]
                    dt2ind = split[indj]
                    g = gt[gtind]
                    d1 = dt[dt1ind]
                    d2 = dt[dt2ind]
                    # if catName in ['SeparatorRegion', 'GraphicRegion.decoration']:
                    #     # FIXME: others: we need all IOUs for non-sep vs. sep, then filter for this d
                    #     allowable = _allowable_merge_separator(d1, d2, g, others,
                    #                                            is_split=True, log=log)
                    # else:
                    allowable = _allowable_merge(d1, d2, g, is_split=True, log=log)
                    if allowable:
                        graph[indi, indj] = True
                n_allowed, labels_allowed = connected_components(csgraph=csr_array(graph),
                                                                 directed=False, return_labels=True)
                partition = []
                for label in range(n_allowed):
                    dtinds = {split[ind] for ind in np.flatnonzero(labels_allowed == label)}
                    partition.append(tuple(dtinds))
                    if len(dtinds) > 1:
                        log.info("allowable split: %s", '+'.join([dt[dtind]['segment_id']
                                                                  for dtind in dtinds]))
                if n_allowed > 1:
                    log.info("non-allowable split: %s", '|'.join('+'.join([dt[dtind]['segment_id']
                                                                           for dtind in dtinds])
                                                                 for dtinds in partition))
                splits[gtind] = Mismatch([gtind], split, partition)
            # create an artificial reading order to illustrate the non/allowable merge/splits:
            # - predictions transitively connected via allowable merges or allowable splits (of GTs)
            #   at the lower level: ordered group
            # - all such groups (i.e. unrelated or from unallowable merges or unallowable splits)
            #   at the higher level: unordered group
            groups = [] # will contain transitively allowable sets of matches
            for gtind, dtind in matches:
                match = matches[gtind, dtind]
                matchgroup = None
                for group in groups:
                    if match in group:
                        matchgroup = group
                if matchgroup is None:
                    # define a new group from this match alone
                    matchgroup = set([match])
                    groups.append(matchgroup)
                if gtind in splits:
                    for dtinds in splits[gtind].partition:
                        if dtind in dtinds:
                            partmatches = [matches[gtind, dtind2] for dtind2 in dtinds]
                            # connect this split partition to the group
                            matchgroup.update(partmatches)
                if dtind in merges:
                    for gtinds in merges[dtind].partition:
                        if gtind in gtinds:
                            partmatches = [matches[gtind2, dtind] for gtind2 in gtinds]
                            # connect this merge partition to the group
                            matchgroup.update(partmatches)
            reading_order = ReadingOrderType()
            pcgts.Page.set_ReadingOrder(reading_order)
            ana = UnorderedGroupType(id="all-non-allowable")
            reading_order.set_UnorderedGroup(ana)
            for group in groups:
                def TP_id(match):
                    return 'TP_' + str(match.gt['id']) + '+' + str(match.dt['id'])
                group = list(group)
                ag = OrderedGroupType()
                if len(group) == 1:
                    match = group[0]
                    ag.set_regionRef(TP_id(match))
                    continue
                # find path through adjacency matrix of GT side
                orders = [match.gt['order'] for match in group]
                assert all(o1 == o2 for o1, o2 in zip(orders[0:-1], orders[1:]))
                order = orders[0]
                gindex = {order.segments.index(match.gt['segment_id']): match
                          for match in group}
                graph = nx.from_numpy_array(np.triu(order.adjacency, 1) * 1,
                                            create_using=nx.DiGraph)
                subgraph = graph.subgraph(list(gindex))
                for idx, node in enumerate(bfs_nodes(subgraph)):
                    ag.add_RegionRefIndexed(
                        RegionRefIndexedType(index=idx, regionRef=TP_id(gindex[node])))
                ana.add_OrderedGroup(ag)
            # naive calculation of IoDt aggregate:
            # iodts = sum(sum(matches[gtind, dtind].iodt # sum(iodt)
            #                 for gtind in dtmatches[dtind])
            #             for dtind in dtmatches)
            # however, the GTs matching each DT may overlap each other:
            sum_iodt = 0.0
            sum_namrodt = 0.0 # non-allowable merge area over dt
            sum_namr = 0.0 # non-allowable merge rate
            for dtind in dtmatches:
                if dtind not in merges:
                    assert len(dtmatches[dtind]) == 1
                    sum_iodt += matches[dtmatches[dtind][0], dtind].iodt
                    continue
                # sum up all IoDts stored for each match pair
                #iodt = sum(matches[gtind, dtind].iodt for gtind in dtmatches[dtind])
                # sum of IoDt within each group of allowable merges
                iodts = [sum(matches[gtind, dtind].iodt for gtind in group)
                         for group in merges[dtind].partition]
                # penalty for partitions
                metric = {'max': max,
                          'gmean': gmean,
                          'amean': np.mean,
                }[non_allowable_metric]
                iodt = metric(iodts)
                # non-allowable error
                namrodt = sum(iodts) - iodt
                aread = int(maskArea(dt[dtind]['segmentation']))
                # superimpose GT intersection masks to count repeats
                gtoverlap = sum(decodeMask(mergeMasks([gt[gtind]['segmentation'],
                                                       dt[dtind]['segmentation']], intersect=True))
                                for gtind in dtmatches[dtind])
                # subtract union area of all matching GTs
                intersection = np.sum(gtoverlap > 0)
                gtoverlap = np.sum(gtoverlap - (gtoverlap > 0)) # total sum of overlapping px
                if gtoverlap:
                    # subtract this "overlap pseudo-area" from the aggregated IoDt
                    iodt = (iodt * aread - gtoverlap) / aread # corrected
                    namrodt = (namrodt * aread - gtoverlap) / aread # corrected
                pagestats['merges'].setdefault(catName, list()).append(
                    {'DT.ID': dt[dtind]['segment_id'],
                     'GT*.ID': [[gt[gtind]['segment_id'] for gtind in gtinds]
                                for gtinds in merges[dtind].partition],
                     'DT.area': aread,
                     'GT*.overlap-area': int(gtoverlap),
                     'I.area': int(intersection),
                     'IoDTs': list(map(float, iodts)),
                     'IoDT': float(iodt)})
                sum_iodt += iodt
                sum_namrodt += namrodt
                sum_namr += namrodt / iodt
            pixel_pr = sum_iodt / len(dtmatches) if len(dtmatches) else -1
            pixel_namr = sum_namrodt / len(dtmatches) if len(dtmatches) else -1
            precision = (len(dt) - len(dtmisses) - sum_namr) / len(dt) if len(dt) else -1
            # naive calculation of IoGt aggregate:
            # iogts = sum(sum(matches[gtind, dtind].iogt # sum(iogt)
            #                 for dtind in gtmatches[gtind])
            #             for gtind in gtmatches) / len(gtmatches) if len(gtmatches) else -1
            # however, the DTs matching each GT may overlap each other:
            sum_iogt = 0.0
            sum_nasrogt = 0.0 # non-allowable split area over gt
            sum_nasr = 0.0 # non-allowable split rate
            for gtind in gtmatches:
                if gtind not in splits:
                    assert len(gtmatches[gtind]) == 1
                    sum_iogt += matches[gtind, gtmatches[gtind][0]].iogt
                    continue
                # sum up all IoGts stored for each match pair
                #iogt = sum(matches[gtind, dtind].iogt for dtind in gtmatches[gtind])
                # sum of IoGt within each group of allowable merges
                iogts = [sum(matches[gtind, dtind].iogt for dtind in group)
                         for group in splits[gtind].partition]
                # penalty for partitions
                metric = {'max': max,
                          'gmean': gmean,
                          'amean': np.mean,
                }[non_allowable_metric]
                iogt = metric(iogts)
                # non-allowable error
                nasrogt = sum(iogts) - iogt
                areag = int(maskArea(gt[gtind]['segmentation']))
                # superimpose DT intersection masks to count repeats
                dtoverlap = sum(decodeMask(mergeMasks([gt[gtind]['segmentation'],
                                                       dt[dtind]['segmentation']], intersect=True))
                                for dtind in gtmatches[gtind])
                # subtract union area of all matching DTs
                intersection = np.sum(dtoverlap > 0)
                dtoverlap = np.sum(dtoverlap - (dtoverlap > 0)) # total sum of overlapping px
                if dtoverlap:
                    # subtract this "overlap pseudo-area" from the aggregated IoGt
                    iogt = (iogt * areag - dtoverlap) / areag # corrected
                    nasrogt = (nasrogt * areag - dtoverlap) / areag # corrected
                pagestats['splits'].setdefault(catName, list()).append(
                    {'GT.ID': gt[gtind]['segment_id'],
                     'DT*.ID': [[dt[dtind]['segment_id'] for dtind in dtinds]
                                for dtinds in splits[gtind].partition],
                     'GT.area': areag,
                     'DT*.overlap-area': int(dtoverlap),
                     'I.area': int(intersection),
                     'IoGTs': list(map(float, iogts)),
                     'IoGT': float(iogt)})
                sum_iogt += iogt
                sum_nasrogt += nasrogt
                sum_nasr += nasrogt / iogt
            pixel_rc = sum_iogt / len(gtmatches) if len(gtmatches) else -1
            pixel_nasr = sum_nasrogt / len(gtmatches) if len(gtmatches) else -1
            recall = (len(gt) - len(gtmisses) - sum_nasr) / len(gt) if len(gt) else -1
            pixel_iou = sum(match.iou for match in matches.values()) / len(matches) if len(matches) else -1
            tparea = sum(match.inter for match in matches.values())
            fparea = sum(nonmatch.area for nonmatch in dtmisses)
            fnarea = sum(nonmatch.area for nonmatch in gtmisses)
            if tparea or (fparea and fnarea):
                pagestats['pixel-precision'][catName] = pixel_pr #tparea / (tparea + fparea)
                pagestats['pixel-recall'][catName] =  pixel_rc # tparea / (tparea + fnarea)
                pagestats['pixel-nonallowable-merge-rate'][catName] = pixel_namr
                pagestats['pixel-nonallowable-split-rate'][catName] = pixel_nasr
                pagestats['pixel-iou'][catName] =  pixel_iou # tparea / (tparea + fparea + fnarea)
            # measure under/oversegmentation for this image and category
            # (follows Zhang et al 2021: Rethinking Semantic Segmentation Evaluation [arXiv:2101.08418])
            over_gt = set(gtind for gtind in splits)
            # without allowable/partitions:
            #over_dt = set(chain.from_iterable(splits[gtind].dtinds for gtind in splits))
            # count unique partition tuples instead of dtinds:
            over_dt = set(chain.from_iterable(splits[gtind].partition for gtind in splits))
            under_dt = set(dtind for dtind in merges)
            # without allowable/partitions:
            #under_gt = set(chain.from_iterable(merges[dtind].gtinds for dtind in merges))
            # count unique partition tuples instead of gtinds:
            under_gt = set(chain.from_iterable(merges[dtind].partition for dtind in merges))
            over_degree = sum(len(gtmatches[gtind]) - 1 for gtind in gtmatches)
            under_degree = sum(len(dtmatches[dtind]) - 1 for dtind in dtmatches)
            if len(dt) and len(gt):
                oversegmentation = len(over_gt) * len(over_dt) / len(gt) / len(dt)
                undersegmentation = len(under_gt) * len(under_dt) / len(gt) / len(dt)
                # Zhang's idea of attenuating the under/oversegmentation ratio with a "penalty"
                # to account for the degree of further sub-segmentation is misguided IMHO,
                # because its degree term depends on the total number of segments:
                # oversegmentation = np.tanh(oversegmentation * over_degree)
                # undersegmentation = np.tanh(undersegmentation * under_degree)
                pagestats['oversegmentation'][catName] = oversegmentation
                pagestats['undersegmentation'][catName] = undersegmentation
                pagestats['precision'][catName] = precision
                pagestats['recall'][catName] = recall
            # aggregate per-img/per-cat IoUs for microaveraging
            # FIXME: aggregation still ignores non/allowable...
            evalimg['matches'] = list(matches.values()) # TP
            evalimg['dtMisses'] = dtmisses # FP
            evalimg['gtMisses'] = gtmisses # FN
            evalimg['dtIdsOver'] = ['+'.join(str(dt[dtind]['id']) for dtind in group)
                                    for group in over_dt]
            evalimg['gtIdsOver'] = [str(gt[gtind]['id']) for gtind in over_gt]
            evalimg['gtIdsUnder'] = ['+'.join(str(gt[gtind]['id']) for gtind in group)
                                     for group in under_gt]
            evalimg['dtIdsUnder'] = [str(dt[dtind]['id']) for dtind in under_dt]

    catstats = stats.setdefault('by-category', odict())
    # accumulate our over-/undersegmentation and IoU ratios
    numImgs = len(coco_eval.params.imgIds)
    numAreas = len(coco_eval.params.areaRng)
    for catind, catId in enumerate(coco_eval.params.catIds):
        cat = coco_gt.cats[catId]
        catstats.setdefault(cat['name'], odict())
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
        numIoUs = sum(len(img['matches']) for img in evalimgs)
        numFPs = sum(len(img['dtMisses']) for img in evalimgs)
        numFNs = sum(len(img['gtMisses']) for img in evalimgs)
        sumIoUs = sum(sum(match.iou for match in img['matches']) for img in evalimgs)
        sumIoGTs = sum(sum(match.iogt for match in img['matches']) for img in evalimgs)
        sumIoDTs = sum(sum(match.iodt for match in img['matches']) for img in evalimgs)
        sumTParea = sum(sum(match.inter for match in img['matches']) for img in evalimgs)
        sumFParea = sum(sum(nonmatch.area for nonmatch in img['dtMisses']) for img in evalimgs) # sum(area)
        sumFNarea = sum(sum(nonmatch.area for nonmatch in img['gtMisses']) for img in evalimgs) # sum(area)
        if numDTs and numGTs:
            oversegmentation = overDTs * overGTs / numDTs / numGTs
            undersegmentation = underDTs * underGTs / numDTs / numGTs
            precision = (numDTs - numFPs) / numDTs
            recall = (numGTs - numFNs) / numGTs
        else:
            oversegmentation = undersegmentation = precision = recall = -1
        if numIoUs:
            iou = sumIoUs / numIoUs
            iogt = sumIoGTs / numIoUs
            iodt = sumIoDTs / numIoUs
        else:
            iou = iogt = iodt = -1
        if sumTParea or (sumFParea and sumFNarea):
            pixel_precision = sumTParea / (sumTParea + sumFParea)
            pixel_recall = sumTParea / (sumTParea + sumFNarea)
            pixel_iou = sumTParea / (sumTParea + sumFParea + sumFNarea)
        else:
            pixel_precision = pixel_recall = pixel_iou = -1
        catstats[cat['name']]['oversegmentation'] = oversegmentation
        catstats[cat['name']]['undersegmentation'] = undersegmentation
        catstats[cat['name']]['precision'] = precision
        catstats[cat['name']]['recall'] = recall
        catstats[cat['name']]['IoGT'] = iogt # i.e. per-match pixel-recall
        catstats[cat['name']]['IoDT'] = iodt # i.e. per-match pixel-precision
        catstats[cat['name']]['IoU'] = iou # i.e. per-match pixel-jaccardindex
        catstats[cat['name']]['pixel-precision'] = pixel_precision
        catstats[cat['name']]['pixel-recall'] = pixel_recall
        catstats[cat['name']]['pixel-iou'] = pixel_iou

    coco_eval.accumulate()
    coco_eval.summarize()
    statInds = np.ones(12, bool)
    statInds[7] = False # AR maxDet[1]
    statInds[8] = False # AR maxDet[2]
    coco_eval.stats = coco_eval.stats[statInds]
    stats['scores'] = odict(zip([
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
                     level='region', typed=True, coords=None, mask=None, log=None):
    if level == 'region':
        reading_order = dict()
        ro = page.get_ReadingOrder()
        if ro:
            page_get_reading_order(reading_order, ro.get_OrderedGroup() or ro.get_UnorderedGroup())
        if reading_order:
            reading_order = ReadingOrder(reading_order)
        else:
            reading_order = None
    for region in page.get_AllRegions(classes=None if level == 'region' else ['Text']):
        if level == 'region':
            # fixme: deal with recursion: prima compares within and across levels, selects minimal error
            cat = region.__class__.__name__[:-4]
            if typed and hasattr(region, 'get_type') and region.get_type():
                cat += '.' + region.get_type()
            if cat not in categories:
                categories.append(cat)
            catid = categories.index(cat)
            if cat == 'TextRegion':
                order = reading_order
                direction = (region.get_textLineOrder() or
                             page.get_textLineOrder() or
                             'top-to-bottom')
            else:
                order = None
                direction = None
            _add_annotation(annotations, region, imgid, catid, order, direction,
                            coords=coords, mask=mask, log=log)
            continue
        # generate pseudo-RO
        group = OrderedGroupType(id=region.id)
        reading_order = {}
        for idx, line in enumerate(region.TextLine):
            ref = RegionRefIndexedType(index=idx, regionRef=line.id)
            group.add_RegionRefIndexed(ref)
            ref.parent_object_ = group
            reading_order[line.id] = ref
        reading_order = ReadingOrder(reading_order)
        for line in region.get_TextLine():
            order = reading_order
            direction = (line.get_readingDirection() or
                         region.get_readingDirection() or
                         page.get_readingDirection() or
                         'left-to-right')
            _add_annotation(annotations, line, imgid, 1, order, direction,
                            coords=coords, mask=mask, log=log)

def _add_annotation(annotations, segment, imgid, catid, order, direction, coords=None, mask=None, log=None):
    if log is None:
        log = getLogger('ocrd.processor.EvaluateSegmentation')
    score = segment.get_Coords().get_conf() or 1.0
    polygon = polygon_from_points(segment.get_Coords().points)
    if len(polygon) < 3:
        log.warning('ignoring segment "%s" with only %d points', segment.id, len(polygon))
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
         'order': order, # non-standard, class ReadingOrder
         'direction': direction, # non-standard string-valued
         # fixme: what about @orientation and @readingOrientation (derotation before geometry for allowable check)?
         'iscrowd': 0})

def _add_ids(entries, start=0):
    for i, entry in enumerate(entries, start):
        if isinstance(entry, dict):
            entry['id'] = i
        else:
            entries[i] = {'id': i, 'name': entry}

def _allowable_merge_separator(seg1, seg2, link, others, is_split=False, log=None):
    """
    May separator segments `seg1` and `seg2` be merged (via alternative layout segment `link`)
    without breaking the overall reading order?

    Checks if any of the non-separator segments `others` lie between them:
    If so, the separators cannot be merged.

    If `is_split`, then the log message describes the inverse application
    (splitting `link` via `seg1` and `seg2` rather than 
    merging `seg1` and `seg2` via `link`).
    """
    if log is None:
        log = getLogger('ocrd.processor.EvaluateSegmentation')
    seg1ID = seg1['segment_id']
    seg2ID = seg2['segment_id']
    linkID = link['segment_id']
    # separators can be merged iff no other segments are between them
    # others: candidate list of non-separator segments overlapping link
    left1 = seg1['bbox'][0]
    left2 = seg2['bbox'][0]
    top1 = seg1['bbox'][1]
    top2 = seg2['bbox'][1]
    right1 = seg1['bbox'][0] + seg1['bbox'][2]
    right2 = seg2['bbox'][0] + seg2['bbox'][2]
    bottom1 = seg1['bbox'][1] + seg1['bbox'][3]
    bottom2 = seg2['bbox'][1] + seg2['bbox'][3]
    # define aliases
    hcenter1 = (left1 + right1) // 2
    hcenter2 = (left2 + right2) // 2
    vcenter1 = (top1 + bottom1) // 2
    vcenter2 = (top2 + bottom2) // 2
    if hcenter1 < hcenter2:
        leftL, leftR = left1, left2
        topL, topR = top1, top2
        rightL, rightR = right1, right2
        bottomL, bottomR = bottom1, bottom2
    else:
        leftR, leftL = left1, left2
        topR, topL = top1, top2
        rightR, rightL = right1, right2
        bottomR, bottomL = bottom1, bottom2
    if vcenter1 < vcenter2:
        leftT, leftB = left1, left2
        topT, topB = top1, top2
        rightT, rightB = right1, right2
        bottomT, bottomB = bottom1, bottom2
    else:
        leftB, leftT = left1, left2
        topB, topT = top1, top2
        rightB, rightT = right1, right2
        bottomB, bottomT = bottom1, bottom2
    # find intervening segment
    for other in others:
        otherID = other['segment_id']
        left0 = other['bbox'][0]
        top0 = other['bbox'][1]
        right0 = other['bbox'][0] + other['bbox'][2]
        bottom0 = other['bbox'][1] + other['bbox'][3]
        if left0 > rightL - ALLOWANCE and right0 < leftR + ALLOWANCE:
            # left-right division: o intervenes iff not completely above or below at least one side
            if bottom0 < topB + ALLOWANCE or top0 > bottomT - ALLOWANCE:
                continue
            if is_split:
                log.debug("sep split of %s (into %s and %s) is not allowable: %s intervenes horizontally", linkID, seg1ID, seg2ID, otherID)
            else:
                log.debug("sep merge of %s with %s (via %s) is not allowable: %s intervenes horizontally", seg1ID, seg2ID, linkID, otherID)
            return False
        if top0 > bottomT - ALLOWANCE and bottom0 < topB + ALLOWANCE:
            # top-down division: o intervenes iff not completely left or right at least one side
            if right0 < leftR + ALLOWANCE or left0 > rightL - ALLOWANCE:
                continue
            if is_split:
                log.debug("sep split of %s (into %s and %s) is not allowable: %s intervenes vertically", linkID, seg1ID, seg2ID, otherID)
            else:
                log.debug("sep merge of %s with %s (via %s) is not allowable: %s intervenes vertically", seg1ID, seg2ID, linkID, otherID)
            return False
        if (max(left0, left1, left2) < min(right0, right1, right2) and
            max(top0, top1, top2) < min(bottom0, bottom1, bottom2)):
            # 3-intersection non-empty
            if is_split:
                log.debug("sep split of %s (into %s and %s) is not allowable: %s intervenes directly", linkID, seg1ID, seg2ID, otherID)
            else:
                log.debug("sep merge of %s with %s (via %s) is not allowable: %s intervenes directly", seg1ID, seg2ID, linkID, otherID)
            return False
    if is_split:
        log.debug("sep split of %s (into %s and %s) is allowable", linkID, seg1ID, seg2ID)
    else:
        log.debug("sep merge of %s with %s (via %s) is allowable", seg1ID, seg2ID, linkID)
    return True

def _allowable_merge(seg1, seg2, link, is_split=False, log=None):
    """
    May segments `seg1` and `seg2` be merged (via alternative layout segment `link`)
    without breaking the overall reading order?

    Checks if both are known in ``order`` and adjacent (i.e. immediate neighbours),
    and their sub-constituent ``direction`` is the same and fits their relative position
    (e.g. `seg1` above `seg2` if ``top-to-bottom``, 
    or `seg2` left of `seg1` if ``right-to-left``).
    Otherwise, the segments cannot be merged.

    If `is_split`, then the log message describes the inverse application
    (splitting `link` via `seg1` and `seg2` rather than 
    merging `seg1` and `seg2` via `link`).
    """
    if log is None:
        log = getLogger('ocrd.processor.EvaluateSegmentation')
    seg1ID = seg1['segment_id']
    seg2ID = seg2['segment_id']
    linkID = link['segment_id']
    if seg1['order'] is None:
        if is_split:
            log.debug("split of %s (into %s and %s) is not allowable: former not in RO", linkID, seg1ID, seg2ID)
        else:
            log.debug("merge of %s with %s (via %s) is not allowable: former not in RO", seg1ID, seg2ID, linkID)
        return False
    if seg2['order'] is None:
        if is_split:
            log.debug("split of %s (into %s and %s) is not allowable: latter not in RO", linkID, seg1ID, seg2ID)
        else:
            log.debug("merge of %s with %s (via %s) is not allowable: latter not in RO", seg1ID, seg2ID, linkID)
        return False
    assert seg2['order'] == seg1['order']
    if not seg1['order'].adjacent(seg1ID, seg2ID):
        if is_split:
            log.debug("split of %s (into %s and %s) is not allowable: not adjacent in RO", linkID, seg1ID, seg2ID)
        else:
            log.debug("merge of %s with %s (via %s) is not allowable: not adjacent in RO", seg1ID, seg2ID, linkID)
        return False
    if seg1['order'].precedes(seg1ID, seg2ID):
        # swap for follow-up checks
        seg1, seg2 = seg2, seg1
    if not seg1['direction']:
        if is_split:
            log.debug("split of %s (into %s and %s) is not allowable: unknown direction", linkID, seg1ID, seg2ID)
        else:
            log.debug("merge of %s with %s (via %s) is not allowable: unknown direction", seg1ID, seg2ID, linkID)
        return False
    if seg1['direction'] != seg2['direction']:
        if is_split:
            log.debug("split of %s (into %s and %s) is not allowable: distinct directions", linkID, seg1ID, seg2ID)
        else:
            log.debug("merge of %s with %s (via %s) is not allowable: distinct directions", seg1ID, seg2ID, linkID)
        return False
    left1 = seg1['bbox'][0]
    left2 = seg2['bbox'][0]
    top1 = seg1['bbox'][1]
    top2 = seg2['bbox'][1]
    right1 = seg1['bbox'][0] + seg1['bbox'][2]
    right2 = seg2['bbox'][0] + seg2['bbox'][2]
    bottom1 = seg1['bbox'][1] + seg1['bbox'][3]
    bottom2 = seg2['bbox'][1] + seg2['bbox'][3]
    if (seg1['direction'] == 'top-to-bottom' and
        top2 < bottom1 - ALLOWANCE):
        # segment2 not completely underneath segment1
        if is_split:
            log.debug("split of %s (into %s and %s) is not allowable: not below", linkID, seg1ID, seg2ID)
        else:
            log.debug("merge of %s with %s (via %s) is not allowable: not below", seg1ID, seg2ID, linkID)
        return False
    if (seg1['direction'] == 'bottom-to-top' and
        top1 < bottom2 - ALLOWANCE):
        # segment2 not completely above segment1
        if is_split:
            log.debug("split of %s (into %s and %s) is not allowable: not above", linkID, seg1ID, seg2ID)
        else:
            log.debug("merge of %s with %s (via %s) is not allowable: not above", seg1ID, seg2ID, linkID)
        return False
    if (seg1['direction'] == 'left-to-right' and
        left2 < right1 - ALLOWANCE):
        # segment2 not completely right of segment1
        if is_split:
            log.debug("split of %s (into %s and %s) is not allowable: not to the right", linkID, seg1ID, seg2ID)
        else:
            log.debug("merge of %s with %s (via %s) is not allowable: not to the right", seg1ID, seg2ID, linkID)
        return False
    if (seg1['direction'] == 'right-to-left' and
        left1 < right2 - ALLOWANCE):
        # segment2 not completely left of segment1
        if is_split:
            log.debug("split of %s (into %s and %s) is not allowable: not to the left", linkID, seg1ID, seg2ID)
        else:
            log.debug("merge of %s with %s (via %s) is not allowable: not to the left", seg1ID, seg2ID, linkID)
        return False
    if is_split:
        log.debug("split of %s (into %s and %s) is allowable", linkID, seg1ID, seg2ID)
    else:
        log.debug("merge of %s with %s (via %s) is allowable", seg1ID, seg2ID, linkID)
    return True


class ReadingOrder:
    def __init__(self, reading_order):
        self.segments = list(reading_order)
        self.adjacency = np.zeros((len(reading_order), len(reading_order)), dtype=bool)
        groups = {}
        for ref in reading_order:
            elem = reading_order[ref]
            if hasattr(elem, 'index'):
                groups.setdefault(elem.parent_object_, list()).append(elem)
        for group in groups:
            it = iter(sorted(groups[group], key=lambda x: x.index))
            seg1 = next(it, None)
            for seg2 in it:
                idx1 = self.segments.index(seg1.regionRef)
                idx2 = self.segments.index(seg2.regionRef)
                self.adjacency[idx1, idx2] = True
                seg1 = seg2
    def adjacent(self, segment1, segment2):
        idx1 = self.segments.index(segment1)
        idx2 = self.segments.index(segment2)
        return self.adjacency[idx1, idx2] or self.adjacency[idx2, idx1]
    def follows(self, segment1, segment2):
        idx1 = self.segments.index(segment1)
        idx2 = self.segments.index(segment2)
        return self.adjacency[idx1, idx2]
    def precedes(self, segment1, segment2):
        idx1 = self.segments.index(segment1)
        idx2 = self.segments.index(segment2)
        return self.adjacency[idx2, idx1]

def page_get_reading_order(ro, rogroup):
    """Add all elements from the given reading order group to the given dictionary.

    Given a dict ``ro`` from layout element IDs to ReadingOrder element objects,
    and an object ``rogroup`` with additional ReadingOrder element objects,
    add all references to the dict, traversing the group recursively.
    """
    regionrefs = list()
    if isinstance(rogroup, (OrderedGroupType, OrderedGroupIndexedType)):
        regionrefs = (rogroup.get_RegionRefIndexed() +
                      rogroup.get_OrderedGroupIndexed() +
                      rogroup.get_UnorderedGroupIndexed())
    if isinstance(rogroup, (UnorderedGroupType, UnorderedGroupIndexedType)):
        regionrefs = (rogroup.get_RegionRef() +
                      rogroup.get_OrderedGroup() +
                      rogroup.get_UnorderedGroup())
    for elem in regionrefs:
        ref = elem.regionRef
        if ref is None:
            # pure group, no linked segment
            continue
        ro[ref] = elem
        if not isinstance(elem, (RegionRefType, RegionRefIndexedType)):
            page_get_reading_order(ro, elem)

def bfs_nodes(G, start=None):
    if start is None:
        for node in G.nodes:
            if G.in_degree(node) == 0:
                start = node
                break
    order = nx.topological_sort(G)
    nodes = [start]
    for out in order:
        for in_, _ in G.in_edges([out]):
            if in_ in nodes:
                yield in_
                nodes.append(out)
    yield out

@dataclass
class Match:
    """ground-truth segment object"""
    gt: dict
    """prediction segment object"""
    dt: dict
    """intersection over ground-truth segment area"""
    iogt: float
    """intersection over prediction segment area"""
    iodt: float
    """intersection over union"""
    iou: float
    """intersection area (number of pixels)"""
    inter: int
    def __hash__(self):
        return self.gt['id'] * 2**20 + self.dt['id']

@dataclass
class Nonmatch:
    """ground-truth (for FN) or prediction (for FP) segment object"""
    seg: dict
    """segment area (number of pixels)"""
    area: int

@dataclass
class Mismatch:
    """
    Either a:
    - Merge (multiple gts, just one dts), or a
    - Split (multiple dts, just one gts).
    """

    """ground-truth segment indices"""
    gtinds: Set[int]
    """prediction segment indices"""
    dtinds: Set[int]
    """
    merge/split of which segments is mutually (or transitively) _allowable_, 
    i.e. this correspondence would not disrupt the reading order of
    the textlines contained therein

    for merges, gt indices (i.e. partition of self.gtinds),
    for splits, dt indices (i.e. partition of self.dtinds)
    """
    partition: List[Tuple[int]]
    

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

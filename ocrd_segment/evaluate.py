from __future__ import absolute_import

import sys
import os
from typing import List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import OrderedDict as odict
from datetime import datetime
import json
from itertools import chain, combinations
import click
import numpy as np
import networkx as nx
from skimage import draw
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components
from scipy.stats import gmean, hmean
from PIL import Image
from shapely.geometry import Polygon

from ocrd import Workspace, Processor
from ocrd_utils import (
    getLogger,
    initLogging,
    xywh_from_polygon,
    polygon_from_points,
    points_from_polygon,
    points_from_xywh,
    coordinates_of_segment,
    MIMETYPE_PAGE,
    pushd_popd,
    make_file_id
)
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    to_xml,
    CoordsType,
    OrderedGroupType,
    OrderedGroupIndexedType,
    UnorderedGroupType,
    UnorderedGroupIndexedType,
    RegionRefType,
    RegionRefIndexedType,
    ReadingOrderType,
    TextRegionType,
)
from ocrd_models.ocrd_page_generateds import (
    GeneratedsSuper,
    PcGtsType,
    MetadataType,
    PageType,
)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import (
    encode as encodeMask,
    decode as decodeMask,
    merge as mergeMasks,
    iou as iouMasks,
    area as maskArea,
    toBbox as maskBbox,
)

from .project import make_intersection, make_valid, join_polygons
from .repair import page_get_reading_order

# algorithmic constants
MIN_OVERLAP = 0.03 # 0.5
METRICS = ['precision',
           'nonallowable-merge-rate',
           'recall',
           'nonallowable-split-rate',
           'f1',
           'undersegmentation',
           'oversegmentation',
           'pixel-precision',
           'pixel-nonallowable-merge-rate',
           'pixel-recall',
           'pixel-nonallowable-split-rate',
           'pixel-f1',
           'pixel-iou',
]

# PRImA constants
ALLOWANCE = 10 # pixels bboxes may overlap in direction of merge/split to still be allowable
ALLOWANCE = 15

# aesthetic constants (must be allowed in XML ID)
MATCHING_SYM = '·'
ALLOWABLE_SYM = '-'
NONALLOWABLE_SYM = ':'

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
                self.logger.warning("skipping page %s missing from GT", file_dt.pageId)
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
            images.append({'page_id': file_gt.pageId,
                           'file_name': (file_gt.local_filename,
                                         file_dt.local_filename),
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
            categories.append('TextLine')
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
        try:
            ofg1, ofg2 = self.output_file_grp.split(",")
        except ValueError:
            ofg1 = self.output_file_grp
            ofg2 = None
        for ift in ifts:
            file_gt, file_dt = ift
            pagestats = stats['by-image'][file_gt.pageId]
            pcgts1 = pagestats.pop('pcgts1')
            self.add_metadata(pcgts1)
            file_id = make_file_id(file_gt, ofg1)
            workspace.add_file(
                ofg1,
                ID=file_id,
                page_id=file_gt.pageId,
                local_filename=os.path.join(ofg1, file_id + '.xml'),
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts1))
            pcgts2 = pagestats.pop('pcgts2')
            if not ofg2:
                continue
            self.add_metadata(pcgts2)
            file_id = make_file_id(file_gt, ofg2)
            workspace.add_file(
                ofg2,
                ID=file_id,
                page_id=file_gt.pageId,
                local_filename=os.path.join(ofg2, file_id + '.xml'),
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts2))

        # write regions to custom JSON for this page
        file_id = 'id' + ofg1 + '_report'
        workspace.add_file(
            ofg1,
            ID=file_id,
            pageId=None,
            local_filename=os.path.join(ofg1, file_id + '.json'),
            mimetype='application/json',
            content=json.dumps(stats, indent=2))
        # todo: also write report for each page

@click.command(context_settings=dict(
    show_default=True,
    help_option_names=['-h', '--help']))
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
@click.option('-P', '--page-plots', type=click.Path(file_okay=False, writable=True),
              help="directory path to write diagnostic page files to")
@click.argument('tabfile', type=click.File('r'), required=False)
def standalone_cli(gt_page_filelst,
                   dt_page_filelst,
                   bin_img_filelst,
                   level_of_operation,
                   ignore_subtype,
                   for_categories,
                   report_file,
                   page_plots,
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
    for page_id, pagestats in stats['by-image'].items():
        pcgts1 = pagestats.pop('pcgts1')
        pcgts2 = pagestats.pop('pcgts2')
        if page_plots:
            os.makedirs(page_plots, exist_ok=True)
            with open(os.path.join(page_plots, os.path.basename(page_id) + ".grouped.xml"), 'wb') as file:
                file.write(bytes(to_xml(pcgts1), 'utf-8'))
            with open(os.path.join(page_plots, os.path.basename(page_id) + ".combined.xml"), 'wb') as file:
                file.write(bytes(to_xml(pcgts2), 'utf-8'))
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
        pcgts_gt = page_from_file(gt_file)
        pcgts_dt = page_from_file(dt_file)
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
        images.append({'page_id': page_id,
                       'file_name': (gt_file, dt_file),
                       'pcgts': (pcgts_gt, pcgts_dt),
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
        categories.append('TextLine')
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
    typed = not parameters["ignore-subtype"]
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
        pageId = img['page_id']
        file_gt, file_dt = img['file_name']
        # get parsed input PAGE files
        pcgts_gt, pcgts_dt = img['pcgts']
        # create trunk output PAGE files
        # would require access to the actual image file:
        #pcgts1 = page_from_file(pcgts_gt.Page.imageFilename)
        #pcgts2 = page_from_file(pcgts_gt.Page.imageFilename)
        # so instead, recreate:
        proc = EvaluateSegmentation(None)
        now = datetime.now()
        pcgts1 = PcGtsType(
            Metadata=MetadataType(
                Creator="%s %s" % (proc.executable, proc.version),
                Created=now,
                LastChange=now,
            ),
            Page=PageType(
                imageWidth=pcgts_gt.Page.imageWidth,
                imageHeight=pcgts_gt.Page.imageHeight,
                imageFilename=pcgts_gt.Page.imageFilename,
                ReadingOrder=ReadingOrderType(
                    UnorderedGroup=UnorderedGroupType(id="all-non-allowable")
                ),
            ),
            pcGtsId="grouped_" + (pcgts_gt.pcGtsId or "") + MATCHING_SYM + (pcgts_dt.pcGtsId or ""),
        )
        pcgts2 = PcGtsType(
            Metadata=MetadataType(
                Creator="%s %s" % (proc.executable, proc.version),
                Created=now,
                LastChange=now,
            ),
            Page=PageType(
                imageWidth=pcgts_gt.Page.imageWidth,
                imageHeight=pcgts_gt.Page.imageHeight,
                imageFilename=pcgts_gt.Page.imageFilename,
                ReadingOrder=ReadingOrderType(
                    OrderedGroup=OrderedGroupType(id="synthesis")
                ),
            ),
            pcGtsId="combined_" + (pcgts_gt.pcGtsId or "") + MATCHING_SYM + (pcgts_dt.pcGtsId or ""),
        )
        ana = pcgts1.Page.ReadingOrder.UnorderedGroup
        syn = pcgts2.Page.ReadingOrder.OrderedGroup
        for catind, catId in enumerate(coco_eval.params.catIds):
            if not catId:
                continue
            cat = coco_gt.cats[catId]
            catName = cat['name']
            catClass = getattr(sys.modules['ocrd_models.ocrd_page'], catName.split('.')[0] + 'Type')
            if parameters['level-of-operation'] == 'region':
                top1 = pcgts1.Page
                top2 = pcgts2.Page
            else:
                top1 = TextRegionType(
                    id="dummy_top",
                    Coords=CoordsType(
                        points=points_from_xywh(dict(
                            x=0,
                            y=0,
                            w=pcgts1.Page.imageWidth,
                            h=pcgts1.Page.imageHeight))))
                top2 = TextRegionType(
                    id="dummy_top",
                    Coords=CoordsType(
                        points=points_from_xywh(dict(
                            x=0,
                            y=0,
                            w=pcgts2.Page.imageWidth,
                            h=pcgts2.Page.imageHeight))))
                pcgts1.Page.add_TextRegion(top1)
                pcgts2.Page.add_TextRegion(top2)
            catAdder1 = getattr(top1, 'add_' + catName.split('.')[0])
            catAdder2 = getattr(top2, 'add_' + catName.split('.')[0])
            # bypassing COCOeval.evaluateImg, hook onto its results
            # (again, we stay at areaRng[0]=all and maxDets[0]=all)
            start = catind * numImgs * numAreas
            evalimg = coco_eval.evalImgs[start + imgind]
            if evalimg is None:
                continue # no DT and GT here
            # record as dict by pageId / by category
            imgstats = stats.setdefault('by-image', odict())
            pagestats = imgstats.setdefault(pageId, odict())
            pagestats['pcgts1'] = pcgts1
            pagestats['pcgts2'] = pcgts2
            # create keys in well-defined order
            pagestats.setdefault('true_positives', odict())
            pagestats.setdefault('false_positives', odict())
            pagestats.setdefault('false_negatives', odict())
            pagestats.setdefault('merges', odict())
            pagestats.setdefault('splits', odict())
            for key in METRICS:
                pagestats.setdefault(key, odict())
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
                if iou < MIN_OVERLAP and iogt < MIN_OVERLAP and iodt < MIN_OVERLAP:
                    continue
                gseg = pcgts_gt.revmap[pcgts_gt.xpath(f"//*[@id='{g['segment_id']}']")[0]]
                dseg = pcgts_dt.revmap[pcgts_dt.xpath(f"//*[@id='{d['segment_id']}']")[0]]
                poly = make_intersection(make_valid(Polygon(polygon_from_points(gseg.Coords.points))),
                                         make_valid(Polygon(polygon_from_points(dseg.Coords.points))))
                new_id = 'TP_' + str(g['id']) + MATCHING_SYM + str(d['id'])
                if poly is None:
                    log.warning("cannot construct intersection for overlap %s", new_id)
                    continue
                bbox = np.array(poly.bounds, dtype=int).tolist()
                newseg = catClass(id=new_id,
                                  Coords=CoordsType(points=points_from_polygon(
                                      poly.exterior.coords[:-1])),
                                  comments=(
                                      "\n"
                                      f"GT.ID={g['segment_id']},\n"
                                      f"DT.ID={d['segment_id']},\n"
                                      f"IoGT={iogt}, IoDT={iodt},\n"
                                      f"IoU={iou}, I.area={intersection}"
                                  ))
                gtmatches.setdefault(gtind, list()).append(dtind)
                dtmatches.setdefault(dtind, list()).append(gtind)
                matches[gtind, dtind] = Match(newseg, g, d, iogt, iodt, iou, areag, aread, intersection, bbox)
            dtmisses = []
            for dtind, d in enumerate(dt):
                if dtind in dtmatches:
                    continue
                aread = int(maskArea(d['segmentation']))
                bboxd = maskBbox(d['segmentation']).astype(int).tolist()
                segd = pcgts_dt.revmap[pcgts_dt.xpath(f"//*[@id='{d['segment_id']}']")[0]]
                newseg = catClass(id='FP_' + str(d['id']),
                                  Coords=CoordsType(points=segd.Coords.points),
                                  comments=f"DT.ID={d['segment_id']}")
                dtmisses.append(Nonmatch(newseg, d, aread, bboxd))
            gtmisses = []
            for gtind, g in enumerate(gt):
                if gtind in gtmatches:
                    continue
                areag = int(maskArea(g['segmentation']))
                bboxg = maskBbox(g['segmentation']).astype(int).tolist()
                segg = pcgts_gt.revmap[pcgts_gt.xpath(f"//*[@id='{g['segment_id']}']")[0]]
                newseg = catClass(id='FN_' + str(g['id']),
                                  Coords=CoordsType(points=segg.Coords.points),
                                  comments=f"GT.ID={g['segment_id']}")
                gtmisses.append(Nonmatch(newseg, g, areag, bboxg))
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
            #                E.g. a region split would remain allowable if each side
            #                had only 1 line.
            #         Analogously, a group of regions' merge is allowable iff it flows in the textLineOrder
            #         of them all, and the GT global reading order has no other regions in between.
            #         For top-to-bottom/bottom-to-top regions, vertical merges are allowable.
            #         For left-to-right/right-to-left regions, horizontal merges are allowable.
            #         (FIXME: Again, we could also validate that the overall textline flow is equivalent.)
            #
            #         Crucially, this relation is transitive, so every merge or split yields a partition
            #         of in-order groups that are mutually incompatible.
            #
            #        This must in turn be used to weigh a match pair's score accordingly when aggregating:
            #        For precision-like scores, we rule out non-allowable merges (by counting them as FP),
            #        and for recall-like scores, we rule out non-allowable splits (by counting them as FN).
            #
            #        FIXME: We could also weigh these non-allowable cases by their share of height
            #               (in vertical textLineOrder and horizontal writing) or width
            #               (in horizontal textLineOrder and vertical writing) in disagreement,
            #               or the share of its textlines that have been split or lost.
            #        FIXME: Furthermore, we could weigh matches by the share of non-text regions or
            #               non-text fg pixels involved.
            #
            # Beyond the non/allowable question of individual splits and merges,
            # when considering the relation between these matches (transitive partitioning),
            # there arises another question whenever multiple matches (TP) overlap each other,
            # which in turn can be caused by non-flat segmentation either via
            # - mutually overlapping DTs (i.e. unallowable split), or
            # - mutually overlapping GTs (i.e. unallowable merge).
            # These will necessarilly end up in separate partitions,
            # but to make all partitions truly disjoint, we have to
            # - if one side is contained within the other: drop the smaller one, i.e. for
            #   * one DT in another DT: drop the smaller one and count it as FP, unless
            #     the smaller one also matches another GT with higher precision: in which case
            #     only drop both cross-matches
            #   * one GT in another GT: drop the smaller one and count it as FN, unless
            #     the smaller one also matches another DT with higher recall: in which case
            #     only drop both cross-matches
            # - otherwise: sever the contested areas, assigning a "winning side" and
            #   removing it from the match/intersection on the "losing side", respectively.
            #   As a principle, here we try to maximise the share of usable text lines, i.e.
            #   * in top-to-bottom or bottom-to-top textline order,
            #     - prefer cutting horizontally over vertically, and
            #     - treat sloped cuts as vertical, but
            #   * in left-to-right or right-to-left textline order,
            #     - prefer cutting vertically over horizontally, and
            #     - treat sloped cuts as horizontal.
            #
            # Thus overall, in effect, a page with non-flat (overlapping) segmentation
            # should be 100% correct when comparing with itself.
            # 
            merges = {}
            for dtind in dtmatches:
                merge = dtmatches[dtind]
                if len(merge) <= 1:
                    continue
                d = dt[dtind]
                #aread = int(maskArea(d['segmentation']))
                pairs = combinations(range(len(merge)), 2)
                graph = np.zeros((len(merge), len(merge)), dtype=bool)
                prune = np.zeros(len(merge), dtype=bool) # to be deleted from merge and graph
                for ind1, ind2 in pairs:
                    if ind1 == ind2:
                        continue
                    if prune[ind1] or prune[ind2]:
                        continue
                    gt1ind = merge[ind1]
                    gt2ind = merge[ind2]
                    match1 = matches[gt1ind, dtind]
                    match2 = matches[gt2ind, dtind]
                    g1 = gt[gt1ind]
                    g2 = gt[gt2ind]
                    # if catName in ['SeparatorRegion', 'GraphicRegion.decoration']:
                    #     # FIXME: others: we need all IOUs for non-sep vs. sep, then filter for this d
                    #     allowable = _allowable_merge_separator(g1, g2, d, others,
                    #                                            log=log)
                    # else:
                    allowable = catName.startswith('Text') and _allowable_merge(g1, g2, d, log=log)
                    if allowable:
                        graph[ind1, ind2] = True
                        continue
                    g1mask = decodeMask(g1['segmentation'])
                    g2mask = decodeMask(g2['segmentation'])
                    g0mask = g1mask & g2mask
                    g0area = np.sum(g0mask)
                    g1area = np.sum(g1mask)
                    g2area = np.sum(g2mask)
                    worse, indworse = None, None
                    if g0area > (1 - MIN_OVERLAP) * min(g1area, g2area):
                        # one nearly fully contained in the other
                        if g1area < g2area:
                            smaller, larger = gt1ind, gt2ind
                            indsmaller, indlarger = ind1, ind2
                        else:
                            smaller, larger = gt2ind, gt1ind
                            indsmaller, indlarger = ind2, ind1
                        # is there a better match for either TP, so we don't need to cross-match?
                        if betterdtind := next((otherdtind
                                                for otherdtind in gtmatches[smaller]
                                                if (otherdtind != dtind and
                                                    matches[smaller, otherdtind].iodt >=
                                                    matches[smaller, dtind].iodt)),
                                               None):
                            # keep only smaller GT ←→ otherdt, larger GT ←→ dt
                            better = matches[smaller, betterdtind]
                            match = matches.pop((smaller, dtind))
                            log.info("found better DT match %s than %s for smaller TP", better, match)
                            worse, indworse = smaller, indsmaller
                        # convert the smaller TP to FN
                        else:
                            # convert smaller ←→ dt from TP to FN
                            match = matches.pop((smaller, dtind))
                            matchstr = str(match)
                            match.seg.id = 'FN_' + str(gt[smaller]['id'])
                            match.seg.comments = f"GT.ID={gt[smaller]['segment_id']}"
                            gtmisses.append(Nonmatch(match.seg, gt[smaller], match.areag, match.bbox))
                            log.info("converted %s for smaller TP to %s", matchstr, gtmisses[-1])
                            worse, indworse = smaller, indsmaller
                    elif g0area > MIN_OVERLAP *  min(g1area, g2area):
                        # is there a better match for either TP, so we don't need to cross-match?
                        if betterdtind := next((otherdtind
                                                for otherdtind in gtmatches[gt1ind]
                                                if (otherdtind != dtind and
                                                    matches[gt1ind, dtind].iodt < 
                                                    matches[gt2ind, dtind].iodt and
                                                    matches[gt1ind, otherdtind].iodt >=
                                                    matches[gt1ind, dtind].iodt and
                                                    matches[gt1ind, otherdtind].iogt >=
                                                    matches[gt1ind, dtind].iogt)),
                                               None):
                            # keep only GT1 ←→ otherdt, GT2 ←→ dt
                            better = matches[gt1ind, betterdtind]
                            match = matches.pop((gt1ind, dtind))
                            log.info("found better DT match %s than %s for first TP", better, match)
                            worse, indworse = gt1ind, ind1
                        elif betterdtind := next((otherdtind
                                                for otherdtind in gtmatches[gt2ind]
                                                if (otherdtind != dtind and
                                                    matches[gt2ind, dtind].iodt < 
                                                    matches[gt1ind, dtind].iodt and
                                                    matches[gt2ind, otherdtind].iodt >=
                                                    matches[gt2ind, dtind].iodt and
                                                    matches[gt2ind, otherdtind].iogt >=
                                                    matches[gt2ind, dtind].iogt)),
                                               None):
                            # keep only GT2 ←→ otherdt, GT1 ←→ dt
                            better = matches[gt2ind, betterdtind]
                            match = matches.pop((gt2ind, dtind))
                            log.info("found better DT match %s than %s for second TP", better, match)
                            worse, indworse = gt2ind, ind2
                    if worse is not None:
                        # remove the match from 'merge' and 'graph'
                        prune[indworse] = True
                        # remove the match from 'gtmatches' right away
                        gtmatches[worse].remove(dtind)
                        if not len(gtmatches[worse]):
                            del gtmatches[worse]
                # apply pruning from GT overlaps
                graph = graph[np.ix_(~prune, ~prune)]
                merge = list(np.array(merge)[~prune])
                dtmatches[dtind] = merge
                assert len(merge)
                # apply allowable transitively
                n_allowed, labels_allowed = connected_components(csgraph=csr_array(graph),
                                                                 directed=False, return_labels=True)
                partition = []
                for label in range(n_allowed):
                    gtinds = {merge[ind] for ind in np.flatnonzero(labels_allowed == label)}
                    partition.append(tuple(gtinds))
                    if len(gtinds) > 1:
                        log.info("allowable merge: %s",
                                 ALLOWABLE_SYM.join([gt[gtind]['segment_id']
                                                     for gtind in gtinds]))
                if n_allowed > 1:
                    log.info("non-allowable merge: %s",
                             NONALLOWABLE_SYM.join(
                                 ALLOWABLE_SYM.join([gt[gtind]['segment_id']
                                                     for gtind in gtinds])
                                 for gtinds in partition))
                merges[dtind] = Mismatch(merge, [dtind], partition)
            splits = {}
            for gtind in gtmatches:
                split = gtmatches[gtind]
                if len(split) <= 1:
                    continue
                g = gt[gtind]
                #areag = int(maskArea(g['segmentation']))
                pairs = combinations(range(len(split)), 2)
                graph = np.zeros((len(split), len(split)), dtype=bool)
                prune = np.zeros(len(split), dtype=bool) # to be deleted from split and graph
                for ind1, ind2 in pairs:
                    if ind1 == ind2:
                        continue
                    if prune[ind1] or prune[ind2]:
                        continue
                    dt1ind = split[ind1]
                    dt2ind = split[ind2]
                    match1 = matches[gtind, dt1ind]
                    match2 = matches[gtind, dt2ind]
                    d1 = dt[dt1ind]
                    d2 = dt[dt2ind]
                    # if catName in ['SeparatorRegion', 'GraphicRegion.decoration']:
                    #     # FIXME: others: we need all IOUs for non-sep vs. sep, then filter for this g
                    #     allowable = _allowable_split_separator(d1, d2, g, others,
                    #                                            log=log)
                    # else:
                    allowable = catName.startswith('Text') and _allowable_split(d1, d2, g, log=log)
                    if allowable:
                        graph[ind1, ind2] = True
                        continue
                    d1mask = decodeMask(d1['segmentation'])
                    d2mask = decodeMask(d2['segmentation'])
                    d0mask = d1mask & d2mask
                    d0area = np.sum(d0mask)
                    d1area = np.sum(d1mask)
                    d2area = np.sum(d2mask)
                    worse, indworse = None, None
                    if d0area > (1 - MIN_OVERLAP) * min(d1area, d2area):
                        # one nearly fully contained in the other
                        if d1area < d2area:
                            smaller, larger = dt1ind, dt2ind
                            indsmaller, indlarger = ind1, ind2
                        else:
                            smaller, larger = dt2ind, dt1ind
                            indsmaller, indlarger = ind2, ind1
                        # is there a better match for either TP, so we don't need to cross-match?
                        if bettergtind := next((othergtind
                                                for othergtind in dtmatches[smaller]
                                                if (othergtind != gtind and
                                                    matches[othergtind, smaller].iogt >=
                                                    matches[gtind, smaller].iogt)),
                                               None):
                            # keep only smaller DT ←→ othergt, larger DT ←→ gt
                            better = matches[bettergtind, smaller]
                            match = matches.pop((gtind, smaller))
                            log.info("found better GT match %s than %s for smaller TP", better, match)
                            worse, indworse = smaller, indsmaller
                        # convert the smaller TP to FP
                        else:
                            # convert smaller ←→ gt from TP to FP
                            match = matches.pop((gtind, smaller))
                            matchstr = str(match)
                            match.seg.id = 'FP_' + str(dt[smaller]['id'])
                            match.seg.comments = f"DT.ID={dt[smaller]['segment_id']}"
                            dtmisses.append(Nonmatch(match.seg, dt[smaller], match.aread, match.bbox))
                            log.info("converted %s for smaller TP to %s", matchstr, dtmisses[-1])
                            worse, indworse = smaller, indsmaller
                    elif d0area > MIN_OVERLAP * min(d1area, d2area):
                        # is there a better match for either TP, so we don't need to cross-match?
                        if bettergtind := next((othergtind
                                                for othergtind in dtmatches[dt1ind]
                                                if (othergtind != gtind and
                                                    matches[gtind, dt1ind].iogt < 
                                                    matches[gtind, dt2ind].iogt and
                                                    matches[othergtind, dt1ind].iodt >=
                                                    matches[gtind, dt1ind].iodt and
                                                    matches[othergtind, dt1ind].iogt >=
                                                    matches[gtind, dt1ind].iogt)),
                                               None):
                            # keep only DT1 ←→ othergt, DT2 ←→ gt
                            better = matches[bettergtind, dt1ind]
                            match = matches.pop((gtind, dt1ind))
                            log.info("found better GT match %s than %s for first TP", better, match)
                            worse, indworse = dt1ind, ind1
                        elif bettergtind := next((othergtind
                                                for othergtind in dtmatches[dt2ind]
                                                if (othergtind != gtind and
                                                    matches[gtind, dt2ind].iogt < 
                                                    matches[gtind, dt1ind].iogt and
                                                    matches[othergtind, dt2ind].iodt >=
                                                    matches[gtind, dt2ind].iodt and
                                                    matches[othergtind, dt2ind].iogt >=
                                                    matches[gtind, dt2ind].iogt)),
                                               None):
                            # keep only DT2 ←→ othergt, DT1 ←→ gt
                            better = matches[bettergtind, dt2ind]
                            match = matches.pop((gtind, dt2ind))
                            log.info("found better GT match %s than %s for second TP", better, match)
                            worse, indworse = dt2ind, ind2
                    if worse is not None:
                        # remove the smaller match from 'split' and 'graph'
                        prune[indworse] = True
                        # remove the match from 'dtmatches' right away
                        dtmatches[worse].remove(gtind)
                        if not len(dtmatches[worse]):
                            del dtmatches[worse]
                # apply pruning from DT overlaps
                graph = graph[np.ix_(~prune, ~prune)]
                split = list(np.array(split)[~prune])
                gtmatches[gtind] = split
                assert len(split)
                # apply allowable transitively
                n_allowed, labels_allowed = connected_components(csgraph=csr_array(graph),
                                                                 directed=False, return_labels=True)
                partition = []
                for label in range(n_allowed):
                    dtinds = {split[ind] for ind in np.flatnonzero(labels_allowed == label)}
                    partition.append(tuple(dtinds))
                    if len(dtinds) > 1:
                        log.info("allowable split: %s",
                                 ALLOWABLE_SYM.join([dt[dtind]['segment_id']
                                                     for dtind in dtinds]))
                if n_allowed > 1:
                    log.info("non-allowable split: %s",
                             NONALLOWABLE_SYM.join(
                                 ALLOWABLE_SYM.join([dt[dtind]['segment_id']
                                                     for dtind in dtinds])
                                 for dtinds in partition))
                splits[gtind] = Mismatch([gtind], split, partition)
            # create an artificial reading order to illustrate the non/allowable merge/splits:
            # - predictions transitively connected via allowable merges or allowable splits (of GTs)
            #   at the lower level: ordered group
            # - all such groups (i.e. unrelated or from unallowable merges or unallowable splits)
            #   at the higher level: unordered group
            # then in pcgts1, annotate this as 2-level hierarchy
            # moreover in pcgts2, annotate the lower (allowable) level as combined segments
            #                     (with joint polygons) and place them in a synthesis ordered group
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
                            partmatches = [matches[gtind, dtind2] for dtind2 in dtinds
                                           # ignore pruned matches due to self-merges (self-overlapping GT)
                                           if (gtind, dtind2) in matches]
                            # connect this split partition to the group
                            matchgroup.update(partmatches)
                if dtind in merges:
                    for gtinds in merges[dtind].partition:
                        if gtind in gtinds:
                            partmatches = [matches[gtind2, dtind] for gtind2 in gtinds
                                           # ignore pruned matches due to self-splits (self-overlapping DT)
                                           if (gtind2, dtind) in matches]
                            # connect this merge partition to the group
                            matchgroup.update(partmatches)
            for group_idx, group in enumerate(groups):
                group = list(group)
                for match in group:
                    catAdder1(match.seg)
                if catName.startswith('Text'):            
                    # represent as OrderedGroup in pcgts1
                    ag = OrderedGroupType(id="allowable-" + str(group_idx))
                    # find path through adjacency matrix of GT side, complemented by
                    # partial paths through adjacency matrix of DT side (for splits)
                    gorders = [match.gt['order'] for match in group]
                    dorders = [match.dt['order'] for match in group]
                    assert all(o1 == o2 for o1, o2 in zip(gorders[0:-1], gorders[1:])), "incomplete GT reading order"
                    assert all(o1 == o2 for o1, o2 in zip(dorders[0:-1], dorders[1:])), "incomplete DT reading order"
                    gorder = gorders[0]
                    dorder = dorders[0]
                    ggraph = nx.from_numpy_array(np.triu(gorder.adjacency, 1) * 1,
                                                 create_using=nx.DiGraph)
                    dgraph = nx.from_numpy_array(np.triu(dorder.adjacency, 1) * 1,
                                                 create_using=nx.DiGraph)
                    path = []
                    # outer path: GT
                    gindex = {gorder.segments.index(match.gt['segment_id']):
                              match for match in group
                              if match.gt['segment_id'] in gorder}
                    gsubgraph = ggraph.subgraph(list(gindex))
                    if not gsubgraph.edges:
                        toppath = group
                    else:
                        toppath = [gindex[node] for node in bfs_nodes(gsubgraph)]
                    # inner paths: DT
                    for topmatch in toppath:
                        subgroup = [match for match in group
                                    if match.gt is topmatch.gt]
                        if len(subgroup):
                            dindex = {dorder.segments.index(match.dt['segment_id']):
                                      match for match in subgroup
                                      if match.dt['segment_id'] in dorder}
                            dsubgraph = dgraph.subgraph(list(dindex))
                            if not dsubgraph.edges:
                                subpath = subgroup
                            else:
                                subpath = [dindex[node] for node in bfs_nodes(dsubgraph)]
                            path.extend(subpath)
                        else:
                            path.append(topmatch)
                    for idx, match in enumerate(path):
                        ag.add_RegionRefIndexed(
                            RegionRefIndexedType(index=idx, regionRef=(
                                'TP_' + str(match.gt['id']) + MATCHING_SYM + str(match.dt['id']))))
                    ana.add_OrderedGroup(ag)
                # represent as new segment in pcgts2
                poly = join_polygons(
                    [make_valid(Polygon(polygon_from_points(match.seg.Coords.points)))
                     for match in group])
                def prefix_join(lis, sym="."):
                    prefix = os.path.commonprefix(lis)
                    return prefix + '{' + sym.join([mem[len(prefix):] for mem in lis]) + '}'
                def fmt_join(lis, fmt="%.2f"):
                    return '[' + ','.join(fmt % mem for mem in lis) + ']'
                newseg = catClass(id='.'.join(match.seg.id for match in group),
                                  Coords=CoordsType(points=points_from_polygon(
                                      poly.exterior.coords[:-1])),
                                  comments=(
                                      "\n"
                                      f"GT.IDs={prefix_join([match.gt['segment_id'] for match in group])},\n"
                                      f"DT.IDs={prefix_join([match.dt['segment_id'] for match in group])},\n"
                                      f"IoGTs={fmt_join([match.iogt for match in group])},\n"
                                      f"IoDTs={fmt_join([match.iodt for match in group])},\n"
                                      f"IoUs={fmt_join([match.iou for match in group])},\n"
                                      f"IoU={sum(match.inter for match in group) / poly.area}"
                                  ))
                catAdder2(newseg)
            for nonmatch in dtmisses:
                catAdder1(nonmatch.seg)
                catAdder2(nonmatch.seg)
                if catName.startswith('TextRegion'):
                    ana.add_RegionRef(RegionRefType(regionRef='FP_' + str(nonmatch.obj['id'])))
                # FIXME: put in RO2 (syn)
            for match in matches.values():
                pagestats['true_positives'].setdefault(catName, list()).append(
                    {'GT.ID': match.gt['segment_id'],
                     'DT.ID': match.dt['segment_id'],
                     'GT.area': match.areag,
                     'DT.area': match.aread,
                     'I.area': match.inter,
                     'I.xywh': match.bbox,
                     'IoGT': match.iogt,
                     'IoDT': match.iodt,
                     'IoU': match.iou})
            for nonmatch in dtmisses:
                pagestats['false_positives'].setdefault(catName, list()).append(
                    {'DT.ID': nonmatch.obj['segment_id'],
                     'area': nonmatch.area,
                     'xywh': nonmatch.bbox})
            for nonmatch in gtmisses:
                pagestats['false_negatives'].setdefault(catName, list()).append(
                    {'GT.ID': nonmatch.obj['segment_id'],
                     'area': nonmatch.area,
                     'xywh': nonmatch.bbox})
            pixel_iou = sum(match.iou for match in matches.values()) / len(matches) if len(matches) else -1
            # naive calculation of IoDt aggregate:
            # iodts = sum(sum(matches[gtind, dtind].iodt # sum(iodt)
            #                 for gtind in dtmatches[dtind])
            #             for dtind in dtmatches)
            # however, the GTs matching each DT may overlap each other:
            sum_tparea = 0
            sum_parea = 0
            sum_namarea = 0 # non-allowable merge area (pixels)
            sum_namr = 0.0 # non-allowable merge rate (segments)
            for dtind in dtmatches:
                aread = int(maskArea(dt[dtind]['segmentation']))
                sum_parea += aread
                if dtind not in merges:
                    assert len(dtmatches[dtind]) == 1
                    sum_tparea += matches[dtmatches[dtind][0], dtind].iodt * aread
                    continue
                # sum up all IoDts stored for each match pair
                #iodt = sum(matches[gtind, dtind].iodt for gtind in dtmatches[dtind])
                # sum of IoDt within each group of allowable merges
                iodts = [sum(matches[gtind, dtind].iodt for gtind in group
                             # ignore pruned matches due to self-splits (self-overlapping DT)
                             if (gtind, dtind) in matches)
                         for group in merges[dtind].partition]
                # penalty for partitions
                metric = {'max': max,
                          'gmean': gmean,
                          'amean': np.mean,
                }[non_allowable_metric]
                iodt = metric(iodts)
                # non-allowable error
                namrodt = sum(iodts) - iodt
                # superimpose GT intersection masks to count repeats
                gtoverlap = sum(decodeMask(mergeMasks([gt[gtind]['segmentation'],
                                                       dt[dtind]['segmentation']], intersect=True))
                                for gtind in dtmatches[dtind])
                # subtract union area of all matching GTs
                intersection = np.sum(gtoverlap > 0)
                gtoverlap = np.sum(gtoverlap - (gtoverlap > 0)) # total sum of overlapping px
                if gtoverlap:
                    # subtract this "overlap pseudo-area" from the aggregated IoDt
                    iodt = max(0.0, iodt * aread - gtoverlap) / aread # corrected
                    namrodt = max(0.0, namrodt * aread - gtoverlap) / aread # corrected
                pagestats['merges'].setdefault(catName, list()).append(
                    {'DT.ID': dt[dtind]['segment_id'],
                     'GT*.ID': [[gt[gtind]['segment_id'] for gtind in gtinds]
                                for gtinds in merges[dtind].partition],
                     'DT.area': aread,
                     'GT*.overlap-area': int(gtoverlap),
                     'I.area': int(intersection),
                     'IoDTs': list(map(float, iodts)),
                     'IoDT': float(iodt)})
                sum_namr += namrodt / iodt if iodt else 0
                sum_tparea += iodt * aread
                sum_namarea += namrodt * aread
            for nonmatch in dtmisses:
                sum_parea += nonmatch.area
            namr = sum_namr / len(dt) if len(dt) else -1
            pixel_pr = sum_tparea / sum_parea if sum_parea else -1
            pixel_namr = sum_namarea / sum_parea if sum_parea else -1
            if catName.startswith('TextRegion'):
                log.debug("non-allowable merges count as %.1f in addition to %d FP", sum_namr, len(dtmisses))
                pagestats['nonallowable-merge-rate'][catName] = namr
                pagestats['pixel-nonallowable-merge-rate'][catName] = pixel_namr
            precision = max(0.0, len(dt) - len(dtmisses) - sum_namr) / len(dt) if len(dt) else -1
            # naive calculation of IoGt aggregate:
            # iogts = sum(sum(matches[gtind, dtind].iogt # sum(iogt)
            #                 for dtind in gtmatches[gtind])
            #             for gtind in gtmatches) / len(gtmatches) if len(gtmatches) else -1
            # however, the DTs matching each GT may overlap each other:
            sum_tparea = 0
            sum_tarea = 0
            sum_nasarea = 0 # non-allowable split area (pixels)
            sum_nasr = 0.0 # non-allowable split rate (segments)
            for gtind in gtmatches:
                areag = int(maskArea(gt[gtind]['segmentation']))
                sum_tarea += areag
                if gtind not in splits:
                    assert len(gtmatches[gtind]) == 1
                    sum_tparea += matches[gtind, gtmatches[gtind][0]].iogt * areag
                    continue
                # sum up all IoGts stored for each match pair
                #iogt = sum(matches[gtind, dtind].iogt for dtind in gtmatches[gtind])
                # sum of IoGt within each group of allowable merges
                iogts = [sum(matches[gtind, dtind].iogt for dtind in group
                             # ignore pruned matches due to self-merges (self-overlapping GT)
                             if (gtind, dtind) in matches)
                         for group in splits[gtind].partition]
                # penalty for partitions
                metric = {'max': max,
                          'gmean': gmean,
                          'amean': np.mean,
                }[non_allowable_metric]
                iogt = metric(iogts)
                # non-allowable error
                nasrogt = sum(iogts) - iogt
                # superimpose DT intersection masks to count repeats
                dtoverlap = sum(decodeMask(mergeMasks([gt[gtind]['segmentation'],
                                                       dt[dtind]['segmentation']], intersect=True))
                                for dtind in gtmatches[gtind])
                # subtract union area of all matching DTs
                intersection = np.sum(dtoverlap > 0)
                dtoverlap = np.sum(dtoverlap - (dtoverlap > 0)) # total sum of overlapping px
                if dtoverlap:
                    # subtract this "overlap pseudo-area" from the aggregated IoGt
                    iogt = max(0.0, iogt * areag - dtoverlap) / areag # corrected
                    nasrogt = max(0.0, nasrogt * areag - dtoverlap) / areag # corrected
                pagestats['splits'].setdefault(catName, list()).append(
                    {'GT.ID': gt[gtind]['segment_id'],
                     'DT*.ID': [[dt[dtind]['segment_id'] for dtind in dtinds]
                                for dtinds in splits[gtind].partition],
                     'GT.area': areag,
                     'DT*.overlap-area': int(dtoverlap),
                     'I.area': int(intersection),
                     'IoGTs': list(map(float, iogts)),
                     'IoGT': float(iogt)})
                sum_nasr += nasrogt / iogt if iogt else 0
                sum_tparea += iogt * areag
                sum_nasarea += nasrogt * areag
            for nonmatch in gtmisses:
                sum_tarea += nonmatch.area
            nasr = sum_nasr / len(gt) if len(gt) else -1
            pixel_rc = sum_tparea / sum_tarea if sum_tarea else -1
            pixel_nasr = sum_nasarea / sum_tarea if sum_tarea else -1
            if catName.startswith('TextRegion'):
                log.debug("non-allowable splits count as %.1f in addition to %d FN", sum_nasr, len(gtmisses))
                pagestats['nonallowable-split-rate'][catName] = nasr
                pagestats['pixel-nonallowable-split-rate'][catName] = pixel_nasr
            recall = max(0.0, len(gt) - len(gtmisses) - sum_nasr) / len(gt) if len(gt) else -1
            pagestats['pixel-precision'][catName] = pixel_pr
            pagestats['pixel-recall'][catName] =  pixel_rc
            pixel_f1 = hmean((pixel_pr, pixel_rc)) if pixel_pr >= 0 and pixel_rc >= 0 else -1
            pagestats['pixel-f1'][catName] = pixel_f1
            pagestats['pixel-iou'][catName] =  pixel_iou
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
                pagestats['f1'][catName] = hmean((precision, recall))
            # aggregate per-img/per-cat IoUs
            for key in pagestats:
                if key in METRICS:
                    evalimg[key] = pagestats[key]

    catstats = stats.setdefault('by-category', odict())
    # accumulate our over-/undersegmentation and IoU ratios
    numImgs = len(coco_eval.params.imgIds)
    numAreas = len(coco_eval.params.areaRng)
    for catind, catId in enumerate(coco_eval.params.catIds):
        cat = coco_gt.cats[catId]
        catName = cat['name']
        catstats.setdefault(catName, odict())
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
        for key in METRICS:
            try:
                vals = [val for img in evalimgs
                        if (val := img[key][catName]) != -1]
            except KeyError:
                continue
            if len(vals) > 0:
                catstats[catName][key] = {
                    "avg": np.mean(vals),
                    "med": np.median(vals),
                    "min": np.min(vals),
                    "max": np.max(vals),
                }

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
        reading_order = odict()
        ro = page.get_ReadingOrder()
        if ro:
            page_get_reading_order(reading_order, ro.get_OrderedGroup() or ro.get_UnorderedGroup())
        else:
            # generate artificial RO from element order
            log.warning("missing reading order, creating artificial one")
            group = OrderedGroupType(id="dummy-order")
            for idx, region in enumerate(page.get_AllRegions()):
                ref = RegionRefIndexedType(index=idx, regionRef=region.id)
                group.add_RegionRefIndexed(ref)
                ref.parent_object_ = group
                reading_order[region.id] = ref
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
            if cat.startswith('TextRegion'):
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
        # FIXME: create 2-level RO so the last/first line of adjacent regions become adjacent
        group = OrderedGroupType(id=region.id)
        reading_order = {}
        for idx, line in enumerate(sorted(region.TextLine, key=lambda x: x.index or 0)):
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

def _allowable_split_separator(seg1, seg2, link, others, log=None):
    return _allowable_merge_separator(seg1, seg2, link, others, log=log, is_split=True)

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

def _allowable_split(seg1, seg2, link, log=None):
    return _allowable_merge(seg1, seg2, link, log=log, is_split=True)

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
    if seg2['order'] != seg1['order']:
        if is_split:
            log.debug("split of %s (into %s and %s) is not allowable: not in same RO group", linkID, seg1ID, seg2ID)
        else:
            log.debug("merge of %s with %s (via %s) is not allowable: not in same RO group", seg1ID, seg2ID, linkID)
        return False
    if seg1ID not in seg1['order']:
        if is_split:
            log.debug("split of %s (into %s and %s) is not allowable: %s not in RO", linkID, seg1ID, seg2ID, seg1ID)
        else:
            log.debug("merge of %s with %s (via %s) is not allowable: %s not in RO", seg1ID, seg2ID, linkID, seg1ID)
        return False
    if seg2ID not in seg1['order']:
        if is_split:
            log.debug("split of %s (into %s and %s) is not allowable: %s not in RO", linkID, seg1ID, seg2ID, seg2ID)
        else:
            log.debug("merge of %s with %s (via %s) is not allowable: %s not in RO", seg1ID, seg2ID, linkID, seg2ID)
        return False
    if not seg1['order'].adjacent(seg1ID, seg2ID):
        if is_split:
            log.debug("split of %s (into %s and %s) is not allowable: not adjacent in RO", linkID, seg1ID, seg2ID)
        else:
            log.debug("merge of %s with %s (via %s) is not allowable: not adjacent in RO", seg1ID, seg2ID, linkID)
        return False
    if seg1['order'].follows(seg1ID, seg2ID):
        # swap for follow-up checks (so seg1 precedes seg2)
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
            it = iter(sorted(groups[group], key=lambda x: x.index or 0))
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
        return self.adjacency[idx2, idx1]
    def precedes(self, segment1, segment2):
        idx1 = self.segments.index(segment1)
        idx2 = self.segments.index(segment2)
        return self.adjacency[idx1, idx2]
    def __contains__(self, segment):
        return segment in self.segments
    def __repr__(self):
        prefix = os.path.commonprefix(self.segments)
        return prefix + '{' + ' '.join(
            (self.segments[idx1][len(prefix):] + "<" +
             self.segments[idx2][len(prefix):])
            for idx1, idx2 in zip(*self.adjacency.nonzero())
        ) + '}'

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
    """generated PAGE-XML segment"""
    seg: GeneratedsSuper
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
    """ground-truth segment area (number of pixels)"""
    areag: int
    """prediction segment area (number of pixels)"""
    aread: int
    """intersection area (number of pixels)"""
    inter: int
    """bounding-box (x1, y1, x2, y2)"""
    bbox: list
    def __hash__(self):
        return self.gt['id'] * 2**20 + self.dt['id']
    def __repr__(self):
        return ' '.join((self.seg.id,
                         "gt=%s" % self.gt['segment_id'],
                         "dt=%s" % self.dt['segment_id'],
                         "iogt=%.2f" % self.iogt,
                         "iodt=%.2f" % self.iodt,
                         "iou=%.2f" % self.iou,
                         "inter=%d" % self.inter,
                         ))

@dataclass
class Nonmatch:
    """generated PAGE-XML segment"""
    seg: GeneratedsSuper
    """ground-truth (for FN) or prediction (for FP) segment object"""
    obj: dict
    """segment area (number of pixels)"""
    area: int
    """bounding-box (x1, y1, x2, y2)"""
    bbox: list
    def __hash__(self):
        return self.obj['id']
    def __repr__(self):
        return ' '.join((self.seg.id,
                         "obj=%s" % self.obj['segment_id'],
                         "area=%d" % self.area,
                         ))

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

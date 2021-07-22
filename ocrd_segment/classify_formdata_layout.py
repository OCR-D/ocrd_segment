from __future__ import absolute_import

import os
import time
import ctypes
import multiprocessing as mp
import numpy as np
from shapely.geometry import Polygon, asPolygon
from shapely.ops import unary_union
from skimage import draw
import cv2
from PIL import Image

#pylint: disable=wrong-import-position
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # i.e. error
from mrcnn import model, utils
import tensorflow as tf
import keras.backend as K
#pylint: disable=wrong-import-position
tf.get_logger().setLevel('ERROR')

from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    coordinates_of_segment,
    coordinates_for_segment,
    crop_image,
    polygon_from_bbox,
    points_from_polygon,
    polygon_from_points,
    MIMETYPE_PAGE
)
from ocrd_models.ocrd_page import (
    to_xml,
    PageType,
    TextRegionType,
    TextLineType,
    CoordsType
)
from ocrd_modelfactory import page_from_file
from ocrd import Processor

from maskrcnn_cli.formdata import FIELDS, InferenceConfig

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-classify-formdata-layout'
# prefer Tensorflow (GPU/CPU) over Numpy (CPU)
# for morphological post-processing of NN predictions
TF_POSTPROCESSING = False
# when doing Numpy postprocessing, enlarge masks via
# outer (convex) instead of inner (concave) hull of
# corresponding connected components
NP_POSTPROCESSING_OUTER = False
# when pruning overlapping detections (in either mode),
# disregard up to this intersection over union
IOU_THRESHOLD = 0.2
# when finalizing contours of detections (in either mode),
# snap to connected components overlapping by this share
# (of component area), i.e. include if larger and exclude
# if smaller than this much
IOCC_THRESHOLD = 0.4
# when finalizing contours of detections (in either mode),
# add this many pixels in each direction
FINAL_DILATION = 4

class ClassifyFormDataLayout(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(ClassifyFormDataLayout, self).__init__(*args, **kwargs)
        if hasattr(self, 'output_file_grp'):
            # processing context
            self.setup()

    def setup(self):
        LOG = getLogger('processor.ClassifyFormDataLayout')
        self.categories = FIELDS
        def readable(path):
            return os.path.isfile(path) and os.access(path, os.R_OK)
        directories = ['', os.path.dirname(os.path.abspath(__file__))]
        if 'MRCNNDATA' in os.environ:
            directories = [os.environ['MRCNNDATA']] + directories
        model_path = ''
        for directory in directories:
            if readable(os.path.join(directory, self.parameter['model'])):
                model_path = os.path.join(directory, self.parameter['model'])
                break
        if not model_path:
            raise Exception("model file '%s' not found" % self.parameter['model'])
        LOG.info("Loading model '%s'", model_path)
        config = InferenceConfig()
        config.IMAGES_PER_GPU = self.parameter['images_per_gpu']
        config.BATCH_SIZE = config.IMAGES_PER_GPU * config.GPU_COUNT
        config.DETECTION_MIN_CONFIDENCE = self.parameter['min_confidence'] / 2 # will be raised after NMS
        config.DETECTION_MAX_INSTANCES = 4 # will be reduced to 1 after cross-class NMS
        config.PRE_NMS_LIMIT = 200
        config.POST_NMS_ROIS_INFERENCE = 100
        assert config.NUM_CLASSES == len(self.categories)
        proto = tf.compat.v1.ConfigProto()
        proto.gpu_options.allow_growth = True  # dynamically alloc GPU memory as needed
        # avoid over-allocation / OOM
        if 'MRCNNPROCS' in os.environ:
            # share GPU with nprocs others
            proto.gpu_options.per_process_gpu_memory_fraction = 1.0 / int(os.environ['MRCNNPROCS'])
        # fall-back to CPU / swap-out
        proto.gpu_options.experimental.use_unified_memory = True # allow swapping memory back to CPU instead of OOM
        self.sess = tf.compat.v1.Session(config=proto)
        #sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
        #    gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))
        K.tensorflow_backend.set_session(self.sess) # set this as default session for Keras / Mask-RCNN
        #config.display()
        self.model = model.MaskRCNN(
            mode="inference", config=config,
            # not really needed, but must be a path...
            model_dir=os.getcwd())
        self.model.load_weights(model_path, by_name=True)

    def process(self):
        """Detect form field target regions for multiple classes, each marked by context from text recognition results.
        
        Open and deserialize PAGE input files and their respective images,
        then iterate over the element hierarchy down to the text line level.
        
        Get the text classification result for each text line (i.e., which
        classes of form fields it marks context for, if any).
        
        Next, retrieve the page image according to the layout annotation (from
        the alternative image of the page, or by cropping at Border and deskewing
        according to @orientation) in raw RGB form. Represent it as an array with
        2 extra channels, one marking text lines, the other marking text lines
        that belong to a certain form class. That is, for each class with non-zero
        results, make a copy of the array and mark all non-zero lines as context.
        
        Pass these arrays to the visual address detector model (packing them
        efficiently into batches as large as possible, and setting only 
        one class as active respectively). For each array/class, retrieve region
        candidates as tuples of class, bounding box, and pixel mask.
        Postprocess the mask and bbox to ensure no words are cut off accidentally,
        and no two predictions overlap each other.
        
        Where the class confidence is high enough, annotate the resulting candidates
        as TextRegion and single TextLine (including the special form field type),
        and remove any overlapping regions or lines in the input.
        
        Produce a new output file by serialising the resulting hierarchy.
        """
        LOG = getLogger('processor.ClassifyFormDataLayout')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)
        
        # pylint: disable=attribute-defined-outside-init
        for n, input_file in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            
            page = pcgts.get_Page()
            page_image, page_coords, page_image_info = self.workspace.image_from_page(
                page, page_id,
                feature_filter='binarized',
                transparency=False)
            if page_image_info.resolution != 1:
                dpi = page_image_info.resolution
                if page_image_info.resolutionUnit == 'cm':
                    dpi = round(dpi * 2.54)
                zoom = 300.0 / dpi
            else:
                dpi = None
                zoom = 1.0
            if zoom < 0.7:
                LOG.info("scaling %dx%d image by %.2f", page_image.width, page_image.height, zoom)
                # actual resampling: see below
                zoomed = zoom
            else:
                zoomed = 1.0
            
            page_image_binarized, _, _ = self.workspace.image_from_page(
                page, page_id,
                feature_selector='binarized')
            # workaround for OCR-D/core#687:
            if 0 < abs(page_image.width - page_image_binarized.width) <= 2:
                diff = page_image.width - page_image_binarized.width
                if diff > 0:
                    page_image = crop_image(
                        page_image,
                        (int(np.floor(diff / 2)), 0,
                         page_image.width - int(np.ceil(diff / 2)),
                         page_image.height))
                else:
                    page_image_binarized = crop_image(
                        page_image_binarized,
                        (int(np.floor(-diff / 2)), 0,
                         page_image_binarized.width - int(np.ceil(-diff / 2)),
                         page_image_binarized.height))
            if 0 < abs(page_image.height - page_image_binarized.height) <= 2:
                diff = page_image.height - page_image_binarized.height
                if diff > 0:
                    page_image = crop_image(
                        page_image,
                        (0, int(np.floor(diff / 2)),
                         page_image.width,
                         page_image.height - int(np.ceil(diff / 2))))
                else:
                    page_image_binarized = crop_image(
                        page_image_binarized,
                        (0, int(np.floor(-diff / 2)),
                         page_image_binarized.width,
                         page_image_binarized.height - int(np.ceil(-diff / 2))))
            # ensure RGB (if raw was merely grayscale)
            if page_image.mode == '1':
                page_image = page_image.convert('L')
            page_image = page_image.convert(mode='RGB')
            # reduce resolution to 300 DPI max
            if zoomed != 1.0:
                page_image_binarized = page_image_binarized.resize(
                    (int(page_image.width * zoomed),
                     int(page_image.height * zoomed)),
                    resample=Image.BICUBIC)
                page_image = page_image.resize(
                    (int(page_image.width * zoomed),
                     int(page_image.height * zoomed)),
                    resample=Image.BICUBIC)
            
            page_array = np.array(page_image)
            # convert to RGB+Text+Context array
            page_array = np.dstack([page_array,
                                    np.zeros_like(page_array[:,:,:2], np.uint8)])
            # convert binarized to single-channel negative
            page_array_bin = np.array(page_image_binarized)
            if page_array_bin.ndim == 3:
                if page_array_bin.shape[-1] == 3:
                    page_array_bin = np.mean(page_array_bin, 2)
                elif page_array_bin.shape[-1] == 4:
                    dtype = page_array_bin.dtype
                    drange = np.iinfo(dtype).max
                    alpha = np.array(page_array_bin[:,:,3], np.float) / drange
                    color = page_array_bin[:,:,:3]
                    color = color * alpha + drange * (1.0 - alpha)
                    page_array_bin = np.array(np.mean(color, 2), dtype=dtype)
                else:
                    page_array_bin = page_array_bin[:,:,0]
            threshold = 0.5 * (page_array_bin.min() + page_array_bin.max())
            page_array_bin = np.array(page_array_bin <= threshold, np.bool)
            
            self._process_page(page, page_image, page_coords, page_id, page_array, page_array_bin, zoomed)
            
            file_id = make_file_id(input_file, self.output_file_grp)
            file_path = os.path.join(self.output_file_grp,
                                     file_id + '.xml')
            out = self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                local_filename=file_path,
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts))
            LOG.info('created file ID: %s, file_grp: %s, path: %s',
                     file_id, self.output_file_grp, out.local_filename)
    
    def _process_page(self, page, page_image, page_coords, page_id, page_array, page_array_bin, zoomed):
        # iterate through all regions that have lines,
        # look for @custom annotated context of any class,
        # derive active classes for this page, and for each class
        # mark a copy of the page array with all context lines
        # to pass to the detector; then post-process and annotate
        # results
        LOG = getLogger('processor.ClassifyFormDataLayout')
        def get_context(line):
            custom = line.get_custom()
            if not custom:
                return []
            return [cat.replace('subtype:context=', '')
                    for cat in custom.split(',')
                    if cat.startswith('subtype:context=')]
        time1 = time.time()
        allregions = page.get_AllRegions(classes=['Text'], depth=2)
        target_regions = []
        active_categories = []
        for region in allregions:
            for line in region.get_TextLine():
                if 'subtype:target=' in (line.get_custom() or ''):
                    target_regions.append(region)
                categories = set(get_context(line))
                for word in line.get_Word():
                    categories.update(get_context(word))
                if not categories:
                    continue
                for category in categories:
                    if category and category not in active_categories:
                        active_categories.append(category)
        # remove existing segmentation (have only detected targets survive)
        page.set_ReadingOrder(None)
        page.set_TextRegion(target_regions)
        page.set_custom('coords=%s' % page_coords['transform'])
        if active_categories:
            LOG.info("Page '%s' has context for: %s", page_id, str(active_categories))
        else:
            LOG.info("No active classes on page '%s'", page_id)
            return
        active_arrays = [np.copy(page_array) for _ in active_categories]
        for region in allregions:
            for line in region.get_TextLine():
                for segment in [line] + line.get_Word() or []:
                    polygon = coordinates_of_segment(segment, page_image, page_coords)
                    if zoomed != 1.0:
                        polygon = np.round(polygon * zoomed).astype(np.int32)
                    polygon_mask = draw.polygon(polygon[:,1], polygon[:,0], page_array.shape)
                    #polygon_hull = draw.polygon_perimeter(polygon[:,1], polygon[:,0], page_array.shape)
                    polygon_tmask = (polygon_mask[0], polygon_mask[1], 3 * np.ones_like(polygon_mask[0]))
                    polygon_amask = (polygon_mask[0], polygon_mask[1], 4 * np.ones_like(polygon_mask[0]))
                    #polygon_thull = (polygon_hull[0], polygon_hull[1], 3 * np.ones_like(polygon_hull[0]))
                    #polygon_ahull = (polygon_hull[0], polygon_hull[1], 4 * np.ones_like(polygon_hull[0]))
                    # mark text in all arrays
                    for array in active_arrays:
                        array[polygon_tmask] = 255
                        #array[polygon_thull] = 255
                    categories = get_context(segment)
                    # mark context, if any
                    for category in categories:
                        array = active_arrays[active_categories.index(category)]
                        array[polygon_amask] = 255
                        #array[polygon_ahull] = 255
        # prepare data generator
        class ArrayDataset(utils.Dataset):
            def __init__(self, arrays, categories):
                super().__init__()
                # Add classes
                for i, name in enumerate(FIELDS):
                    if name:
                        # use class name as source so we can train on each class dataset
                        # after another while only one class is active at a time
                        self.add_class(name, i, name)
                self.arrays = arrays
                for i, _ in enumerate(arrays):
                    self.add_image(categories[i], i, "")
            def load_image(self, image_id):
                return self.arrays[image_id]
        dataset = ArrayDataset(active_arrays, active_categories)
        dataset.prepare()
        generator = model.InferenceDataGenerator(dataset, self.model.config)
        time2 = time.time()
        # predict page image per-class as batch
        K.tensorflow_backend.set_session(self.sess)
        predictions = self.model.detect_generator(generator, verbose=0,
                                                  workers=max(3, self.model.config.BATCH_SIZE))
        time3 = time.time()
        # concatenate instances for all classes of this page image
        preds = dict()
        predictions = [pred for pred in predictions if pred['masks'].ndim > 2]
        if not predictions:
            LOG.warning("Detected no form fields on page '%s'", page_id)
            return
        for key in ['rois', 'class_ids', 'scores', 'masks']:
            vals = [pred[key] for pred in predictions]
            if key == 'masks':
                vals = [np.moveaxis(val, 2, 0) for val in vals]
            preds[key] = np.concatenate(vals)
        assert len(preds["rois"]) == len(preds["class_ids"]) == len(preds["scores"]) == len(preds["masks"])
        LOG.debug("Decoding %d ROIs for %d distinct classes (avg. score: %.2f)",
                  len(preds["class_ids"]),
                  len(np.unique(preds["class_ids"])),
                  np.mean(preds["scores"]) if all(preds["scores"].shape) else 0)
        # apply post-processing to detections:
        # - geometrically: remove overlapping candidates via non-maximum suppression across classes
        # - morphologically: extend masks/bboxes to avoid chopping off fg connected components
        if TF_POSTPROCESSING:
            postprocess = postprocess_graph
        else:
            postprocess = postprocess_numpy
        scale, boxes, scores, classes, masks = postprocess(
            preds['rois'], preds['scores'], preds['class_ids'],
            preds["masks"], page_array_bin, self.categories,
            min_confidence=self.parameter['min_confidence'],
            nproc=self.parameter['num_processes'])
        if len(boxes) == 0:
            LOG.warning("Detected no form fields on page '%s'", page_id)
            return
        scale = scale + (scale+1)%2 # odd
        LOG.debug("Estimated scale: %d", scale)
        time4 = time.time()
        for i, (bbox, score, class_id, mask) in enumerate(zip(boxes, scores, classes, masks)):
            category = self.categories[class_id]
            #region_polygon = polygon_from_bbox(bbox[1],bbox[0],bbox[3],bbox[2])
            # dilate until we have a single outer contour
            invalid = True
            for _ in range(10):
                contours, _ = cv2.findContours(mask.astype(np.uint8),
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 1 and len(contours[0]) > 3:
                    invalid = False
                    break
                mask = cv2.dilate(mask.astype(np.uint8),
                                  np.ones((scale,scale), np.uint8)) > 0
            if invalid:
                LOG.warning("Ignoring non-contiguous (%d) region for '%s'", len(contours), category)
                continue
            region_polygon = contours[0][:,0,:] # already in x,y order
            if zoomed != 1.0:
                region_polygon = region_polygon / zoomed
            # ensure consistent and valid polygon outline
            region_polygon = coordinates_for_segment(region_polygon,
                                                     page_image, page_coords)
            region_polygon = polygon_for_parent(region_polygon, page)
            if region_polygon is None:
                LOG.warning("Ignoring extant region for '%s'", category)
                continue
            # annotate new region/line
            region_coords = CoordsType(points_from_polygon(region_polygon), conf=score)
            region_id = 'region%02d_%s' % (i+1, category)
            region = TextRegionType(id=region_id, Coords=region_coords, type_='other')
            line = TextLineType(id=region_id + '_line', Coords=region_coords,
                                custom='subtype:target=' + category)
            region.add_TextLine(line)
            page.add_TextRegion(region)
            LOG.info("Detected %s region%02d (p=%.2f) on page '%s'",
                     category, i+1, score, page_id)
        time5 = time.time()
        LOG.debug("pre-processing time: %d", time2 - time1)
        LOG.debug("GPU detection time: %d", time3 - time2)
        LOG.debug("post-processing time: %d", time4 - time3)
        LOG.debug("contour finding time: %d", time5 - time4)

def postprocess_graph(boxes, scores, classes, masks, image, categories, min_confidence=0.5, nproc=1):
    """Apply post-processing to raw detections. Implement as a Tensorflow graph.
        
    - geometrically: remove overlapping candidates via non-maximum suppression across classes
    - morphologically: extend masks/bboxes to avoid chopping off fg connected components
    """
    # prepare NMS input
    boxesA = boxes.astype(np.float32)
    scoresA = scores
    classesA = classes
    masksA = masks
    imageA = image
    boxesV = tf.placeholder(tf.float32, name='boxes')
    scoresV = tf.placeholder(tf.float32, name='scores')
    classesV = tf.placeholder(tf.int64, name='classes')
    masksV = tf.placeholder(tf.bool, name='masks')
    imageV = tf.placeholder(tf.bool, name='image', shape=imageA.shape)
    feed_list = [boxesV, scoresV, classesV, masksV, imageV]
    # apply IoU-based NMS across classes via masks:
    # - rule out worse overlapping predictions
    # - rule out non-best predictions among classes
    # - combine both criteria best to worst:
    #   * worse instance of class is in conflict (with worse/better neighbour)
    #     → remove (so better classes do not always suppress worse)
    #   * better instance of class is in conflict with better neighbour
    #     → remove (so better instances do not always suppress worse)
    # (We need to do this in a parallel foldl instead of
    #  direct/full tensor calculations because the latter
    #  would cost too much memory: a bool N*N*H*W tensor;
    #  and SparseTensor does not allow tensordot() and other
    #  essential operations.
    #  Also, [combined_]non_max_suppression does not compute
    #  this correctly.)
    # FIXME this is SLOW! Takes 30% longer than Numpy on 300 DPI images, despite using twice CPU time.
    ninstances = tf.shape(masksV)[0]
    instances = tf.range(ninstances) # [N]
    instances_i, instances_j = tf.meshgrid(instances, instances, indexing='ij')
    combinations = tf.where(instances_i < instances_j) # [N*(N-1)/2,2]
    overlaps = tf.zeros((ninstances, ninstances), tf.bool)
    def compare_masks(overlaps, combination):
        i, j = combination[0], combination[1]
        imask = masksV[i]
        jmask = masksV[j]
        intersection = tf.count_nonzero(imask & jmask)
        union = tf.count_nonzero(imask | jmask)
        overlap = intersection / union > IOU_THRESHOLD
        # avoid division by zero:
        overlap = tf.cond(intersection > 0, lambda: overlap, lambda: False)
        # cannot use scatter_update on tensors in TF1
        return tf.cond(overlap,
                       lambda: tf.tensor_scatter_update(overlaps, [[i, j], [j, i]], [True, True]),
                       lambda: overlaps)
    overlaps = tf.foldl(compare_masks, combinations, initializer=overlaps)
    bad = tf.zeros(ninstances, tf.bool)
    def suppress(suppressed, i):
        sameclass = tf.equal(classesV[i], classesV)
        worse = tf.less(scoresV[i], scoresV)
        conflicts = worse & (overlaps[i] | sameclass)
        return tf.cond(tf.less(scoresV[i], min_confidence) | tf.reduce_any(conflicts),
                       lambda: tf.tensor_scatter_update(suppressed, [[i]], [True]),
                       lambda: suppressed)
    bad = tf.foldl(suppress, tf.argsort(scoresV, direction='DESCENDING'), initializer=bad)
    boxes = tf.boolean_mask(boxesV, ~bad)
    classes = tf.boolean_mask(classesV, ~bad)
    scores = tf.boolean_mask(scoresV, ~bad)
    masks = tf.boolean_mask(masksV, ~bad)
    # get connected components of binarized image
    components = tf.contrib.image.connected_components(imageV)
    complabels, _, compcounts = tf.unique_with_counts(tf.reshape(components, (-1,)))
    # estimate glyph scale (roughly)
    compcounts = tf.sqrt(3 * tf.cast(compcounts, tf.float32))
    compcounts = tf.gather(compcounts, tf.where((3 < compcounts) & (compcounts < 100))[:,0])
    def median(x):
        half = tf.shape(x)[0] // 2
        topk = tf.nn.top_k(x, half, sorted=False)
        return tf.reduce_min(topk.values)
    mediancount = median(compcounts)
    scale = tf.cond(tf.shape(compcounts)[0] > 1, lambda: mediancount, lambda: tf.cast(43, tf.float32))
    # FIXME: rewrite refinement rules from bbox to mask aggregation ...
    # get bboxes for each connected component
    def add_bbox(bboxes, idx):
        labelidx = tf.where(tf.equal(components, idx))
        bbox = tf.concat([tf.reduce_min(labelidx, axis=0),  # minx, miny
                          tf.reduce_max(labelidx, axis=0)], # maxx, maxy
                         0)
        # cannot use scatter_update on tensors in TF1
        return tf.tensor_scatter_update(bboxes, [[idx]], [bbox])
    compboxes = tf.zeros([tf.shape(complabels)[0], 4], dtype=tf.float32)
    compboxes = tf.foldl(add_bbox, complabels, initializer=compboxes)
    # get overlaps between component and ROI bboxes
    def get_overlaps(boxes1, boxes2, first=True):
        """Computes share ratios between two sets of boxes.
        For each, pair compute the intersection over:
        - the full area of the first box if `first`,
        - the full area of the second box otherwise.
        boxes1, boxes2: [N, (y1, x1, y2, x2)].
        """
        # 1. Tile boxes2 and repeat boxes1. This allows us to compare
        # every boxes1 against every boxes2 without loops.
        # TF doesn't have an equivalent to np.repeat() so simulate it
        # using tf.tile() and tf.reshape.
        b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                                [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
        b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
        # 2. Compute intersections
        b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
        b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
        y1 = tf.maximum(b1_y1, b2_y1)
        x1 = tf.maximum(b1_x1, b2_x1)
        y2 = tf.minimum(b1_y2, b2_y2)
        x2 = tf.minimum(b1_x2, b2_x2)
        intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
        # 3. Compute unions
        b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
        b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
        reference = b1_area if first else b2_area
        # 4. Compute IoU and reshape to [boxes1, boxes2]
        share = intersection / reference
        overlaps = tf.reshape(share, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
        return overlaps
    overlaps = tf.where(get_overlaps(compboxes, boxes) > IOCC_THRESHOLD)
    # enlarge each mask by all its overlapping component bboxes
    def extend_mask(masks, match):
        compidx, roiidx = match[0], match[1]
        mask = masks[roiidx]
        roimin = tf.gather_nd(boxes, [[roiidx, 0], [roiidx, 1]])
        roimax = tf.gather_nd(boxes, [[roiidx, 2], [roiidx, 3]])
        compmin = tf.gather_nd(compboxes, [[compidx, 0], [compidx, 1]])
        compmax = tf.gather_nd(compboxes, [[compidx, 2], [compidx, 3]])
        newmin = tf.minimum(roimin, compmin)
        newmax = tf.maximum(roimax, compmax)
        # ignore background (matches everywhere):
        background = tf.identity(complabels[compidx] == 0)
        # skip if wc > 2 * w or hc > 2 * h:
        toolarge = tf.reduce_any(compmax - compmin > 2 * (roimax - roimin))
        # skip if neww > 2 * w or newh > 1.5 * h:
        roimin = tf.cast(roimin, tf.float32)
        roimax = tf.cast(roimax, tf.float32)
        newmin = tf.cast(newmin, tf.float32)
        newmax = tf.cast(newmax, tf.float32)
        thresh = tf.constant([2, 1.5])
        excentric = tf.reduce_any(newmax - newmin > thresh * (roimax - roimin))
        # suppress or extend
        compmin = tf.cond(toolarge | excentric,
                          lambda: tf.minimum(tf.cast(tf.shape(mask), tf.float32), compmin + FINAL_DILATION),
                          lambda: tf.maximum(tf.cast(0, tf.float32), compmin - FINAL_DILATION))
        compmax = tf.cond(toolarge | excentric,
                          lambda: tf.maximum(tf.cast(0, tf.float32), compmax - FINAL_DILATION),
                          lambda: tf.minimum(tf.cast(tf.shape(mask), tf.float32), compmax + FINAL_DILATION))
        compmin = tf.cast(compmin, tf.int64)
        compmax = tf.cast(compmax, tf.int64)
        compmin = tf.minimum(compmax, compmin)
        compmax = tf.maximum(compmin, compmax)
        indexs = tf.transpose(tf.stack(tf.meshgrid(tf.range(compmin[0], compmax[0]),
                                                   tf.range(compmin[1], compmax[1]),
                                                   indexing='ij')),
                              perm=[1,2,0])
        update = tf.cond(toolarge | excentric,
                         lambda: tf.zeros(compmax - compmin, tf.bool),
                         lambda: tf.ones(compmax - compmin, tf.bool))
        newmask = tf.tensor_scatter_update(mask, indexs, update)
        newmasks = tf.tensor_scatter_update(masks, [[roiidx]], [newmask])
        return tf.cond(background, lambda: masks, lambda: newmasks)
    masks = tf.cond(tf.shape(overlaps)[0] > 0,
                    lambda: tf.foldl(extend_mask, overlaps, initializer=masks),
                    lambda: masks) # (unlikely) case of no overlaps
    # add some padding
    # complains about Dst not being initialized...
    # masks = tf.nn.dilation2d(tf.cast(masks, tf.float32),
    #                          filter=tf.ones((FINAL_DILATION//2 + 1,
    #                                          FINAL_DILATION//2 + 1,
    #                                          1)),
    #                          padding='SAME',
    #                          strides=(1,1,1,1), rates=(1,1,1,1))
    # masks = tf.cast(masks, tf.bool)
    # run kernel and return
    postprocess = K.get_session().make_callable([scale, boxes, scores, classes, masks], feed_list=feed_list)
    scale, boxes, scores, classes, masks = postprocess(boxesA, scoresA, classesA, masksA, imageA)
    return int(scale), boxes, scores, classes.astype(np.int32), masks.astype(np.bool)

def postprocess_numpy(boxes, scores, classes, masks, page_array_bin, categories, min_confidence=0.5, nproc=8):
    """Apply post-processing to raw detections. Implement via Numpy routines.
        
    - geometrically: remove overlapping candidates via non-maximum suppression across classes
    - morphologically: extend masks/bboxes to avoid chopping off fg connected components
    """
    LOG = getLogger('processor.ClassifyFormDataLayout')
    # apply IoU-based NMS across classes
    assert masks.dtype == np.bool
    instances = np.arange(len(masks))
    instances_i, instances_j = np.meshgrid(instances, instances, indexing='ij')
    combinations = list(zip(*np.where(instances_i < instances_j)))
    shared_masks = mp.sharedctypes.RawArray(ctypes.c_bool, masks.size)
    shared_masks_np = tonumpyarray_with_shape(shared_masks, masks.shape)
    np.copyto(shared_masks_np, masks)
    with mp.Pool(processes=nproc, # to be refined via param
                 initializer=overlapmasks_init,
                 initargs=(shared_masks, masks.shape)) as pool:
        # multiprocessing for different combinations of array slices (pure)
        overlapping_combinations = pool.starmap(overlapmasks, combinations)
    overlaps = np.zeros((len(masks), len(masks)), np.bool)
    for (i, j), overlapping in zip(combinations, overlapping_combinations):
        if overlapping:
            overlaps[i, j] = True
            overlaps[j, i] = True
    # find best-scoring instance per class
    bad = np.zeros_like(instances, np.bool)
    for i in np.argsort(-scores):
        class_id = classes[i]
        if not class_id:
            raise Exception('detected region for background class')
        category = categories[class_id]
        score = scores[i]
        bbox = boxes[i]
        mask = masks[i]
        assert mask.shape[:2] == page_array_bin.shape[:2]
        if scores[i] < min_confidence:
            LOG.debug("Ignoring instance for '%s' with too low score %.2f", category, score)
            bad[i] = True
            continue
        count = np.count_nonzero(mask)
        if count < 10:
            LOG.warning("Ignoring too small (%dpx) region for '%s'", count, category)
            bad[i] = True
            continue
        worse = score < scores
        if classes[i] in [7]:
            sameclass = False # only 1-best instance for most classes (7 can be multiple)
        else:
            sameclass = np.equal(class_id, classes)
        if np.any(worse & overlaps[i]):
            LOG.debug("Ignoring instance for '%s' with %.2f overlapping better neighbour",
                      category, score)
            bad[i] = True
        elif np.any(worse & sameclass):
            LOG.debug("Ignoring instance for '%s' with non-maximum score %.2f",
                      category, score)
            bad[i] = True
        else:
            LOG.debug("post-processing prediction for '%s' at %s area %d score %f",
                      category, str(bbox), count, score)
    # get connected components
    _, components = cv2.connectedComponents(page_array_bin.astype(np.uint8))
    # estimate glyph scale (roughly)
    _, counts = np.unique(components, return_counts=True)
    if counts.shape[0] > 1:
        counts = np.sqrt(3 * counts)
        counts = counts[(5 < counts) & (counts < 100)]
        scale = int(np.median(counts))
    else:
        scale = 43
    # post-process detections morphologically and decode to region polygons
    # does not compile (no OpenCV support):
    keep = np.where(~bad)[0]
    if not np.any(keep):
        return scale, [], [], [], []
    keep = sorted(keep, key=lambda i: scores[i], reverse=True)
    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]
    masks = masks[keep]
    cats = [categories[class_id] for class_id in classes]
    shared_masks = mp.sharedctypes.RawArray(ctypes.c_bool, masks.size)
    shared_components = mp.sharedctypes.RawArray(ctypes.c_int32, components.size)
    shared_masks_np = tonumpyarray_with_shape(shared_masks, masks.shape)
    shared_components_np = tonumpyarray_with_shape(shared_components, components.shape)
    np.copyto(shared_components_np, components, casting='equiv')
    np.copyto(shared_masks_np, masks)
    with mp.Pool(processes=nproc, # to be refined via param
                 initializer=morphmasks_init,
                 initargs=(shared_masks, masks.shape,
                           shared_components, components.shape)) as pool:
        # multiprocessing for different slices of array (in-place)
        pool.map(morphmasks, range(masks.shape[0]))
    masks = tonumpyarray_with_shape(shared_masks, masks.shape)
    return scale, boxes, scores, classes, masks

def polygon_for_parent(polygon, parent):
    """Clip polygon to parent polygon range.
    
    (Should be moved to ocrd_utils.coordinates_for_segment.)
    """
    childp = Polygon(polygon)
    if isinstance(parent, PageType):
        if parent.get_Border():
            parentp = Polygon(polygon_from_points(parent.get_Border().get_Coords().points))
        else:
            parentp = Polygon([[0,0], [0,parent.get_imageHeight()],
                               [parent.get_imageWidth(),parent.get_imageHeight()],
                               [parent.get_imageWidth(),0]])
    else:
        parentp = Polygon(polygon_from_points(parent.get_Coords().points))
    # ensure input coords have valid paths (without self-intersection)
    # (this can happen when shapes valid in floating point are rounded)
    childp = make_valid(childp)
    parentp = make_valid(parentp)
    if not childp.is_valid:
        return None
    if not parentp.is_valid:
        return None
    # check if clipping is necessary
    if childp.within(parentp):
        return childp.exterior.coords[:-1]
    # clip to parent
    interp = childp.intersection(parentp)
    # post-process
    if interp.is_empty or interp.area == 0.0:
        return None
    if interp.type == 'GeometryCollection':
        # heterogeneous result: filter zero-area shapes (LineString, Point)
        interp = unary_union([geom for geom in interp.geoms if geom.area > 0])
    if interp.type == 'MultiPolygon':
        # homogeneous result: construct convex hull to connect
        # FIXME: construct concave hull / alpha shape
        interp = interp.convex_hull
    if interp.minimum_clearance < 1.0:
        # follow-up calculations will necessarily be integer;
        # so anticipate rounding here and then ensure validity
        interp = asPolygon(np.round(interp.exterior.coords))
        interp = make_valid(interp)
    return interp.exterior.coords[:-1] # keep open

def make_valid(polygon):
    for split in range(1, len(polygon.exterior.coords)-1):
        if polygon.is_valid or polygon.simplify(polygon.area).is_valid:
            break
        # simplification may not be possible (at all) due to ordering
        # in that case, try another starting point
        polygon = Polygon(polygon.exterior.coords[-split:]+polygon.exterior.coords[:-split])
    for tolerance in range(1, int(polygon.area)):
        if polygon.is_valid:
            break
        # simplification may require a larger tolerance
        polygon = polygon.simplify(tolerance)
    return polygon

def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr, dtype=np.dtype(mp_arr))

def tonumpyarray_with_shape(mp_arr, shape):
    return np.frombuffer(mp_arr, dtype=np.dtype(mp_arr)).reshape(shape)

def overlapmasks_init(masks_array, masks_shape):
    global shared_masks
    global shared_masks_shape
    shared_masks = masks_array
    shared_masks_shape = masks_shape
    
def overlapmasks(i, j):
    masks = np.ctypeslib.as_array(shared_masks).reshape(shared_masks_shape)
    imask = masks[i]
    jmask = masks[j]
    intersection = np.count_nonzero(imask * jmask)
    if not intersection:
        return False
    union = np.count_nonzero(imask + jmask)
    if intersection / union > IOU_THRESHOLD:
        return True
    return False

def morphmasks_init(masks_array, masks_shape, components_array, components_shape):
    global shared_masks
    global shared_masks_shape
    global shared_components
    global shared_components_shape
    shared_masks = masks_array
    shared_masks_shape = masks_shape
    shared_components = components_array
    shared_components_shape = components_shape

def morphmasks(instance):
    masks = np.ctypeslib.as_array(shared_masks).reshape(shared_masks_shape)
    components = np.ctypeslib.as_array(shared_components).reshape(shared_components_shape)
    mask = masks[instance]
    # find closure in connected components
    complabels = np.unique(mask * components)
    left, top, w, h = cv2.boundingRect(mask.astype(np.uint8))
    right = left + w
    bottom = top + h
    if NP_POSTPROCESSING_OUTER:
        # overwrite pixel mask from (padded) outer bbox
        for label in complabels:
            if not label:
                continue # bg/white
            leftc, topc, wc, hc = cv2.boundingRect((components == label).astype(np.uint8))
            rightc = leftc + wc
            bottomc = topc + hc
            if wc > 2 * w or hc > 2 * h:
                continue # huge (non-text?) component
            # intersection over component too small?
            if (min(right, rightc) - max(left, leftc)) * \
                (min(bottom, bottomc) - max(top, topc)) < IOCC_THRESHOLD * wc * hc:
                continue # too little overlap
            newleft = min(left, leftc)
            newtop = min(top, topc)
            newright = max(right, rightc)
            newbottom = max(bottom, bottomc)
            if (newright - newleft) > 2 * w or (newbottom - newtop) > 1.5 * h:
                continue # 
            left = newleft
            top = newtop
            right = newright
            bottom = newbottom
            w = right - left
            h = bottom - top
        left = max(0, left - FINAL_DILATION)
        top = max(0, top - FINAL_DILATION)
        right = min(mask.shape[1], right + FINAL_DILATION)
        bottom = min(mask.shape[0], bottom + FINAL_DILATION)
        mask[top:bottom, left:right] = True
        
    else:
        # fill pixel mask from (padded) inner bboxes
        for label in complabels:
            if not label:
                continue # bg/white
            suppress = False
            leftc, topc, wc, hc = cv2.boundingRect((components == label).astype(np.uint8))
            rightc = leftc + wc
            bottomc = topc + hc
            if wc > 2 * w or hc > 2 * h:
                # huge (non-text?) component
                suppress = True
            if (min(right, rightc) - max(left, leftc)) * \
                (min(bottom, bottomc) - max(top, topc)) < IOCC_THRESHOLD * wc * hc:
                # intersection over component too small
                suppress = True
            newleft = min(left, leftc)
            newtop = min(top, topc)
            newright = max(right, rightc)
            newbottom = max(bottom, bottomc)
            if (newright - newleft) > 2 * w or (newbottom - newtop) > 1.5 * h:
                # huge (non-text?) component
                suppress = True
            elif (newright - newleft) < 1.1 * w and (newbottom - newtop) < 1.1 * h:
                suppress = False
            if suppress:
                leftc = min(mask.shape[1], leftc + FINAL_DILATION)
                topc = min(mask.shape[0], topc + FINAL_DILATION)
                rightc = max(0, rightc - FINAL_DILATION)
                bottomc = max(0, bottomc - FINAL_DILATION)
                mask[topc:bottomc, leftc:rightc] = False
            else:
                leftc = max(0, leftc - FINAL_DILATION)
                topc = max(0, topc - FINAL_DILATION)
                rightc = min(mask.shape[1], rightc + FINAL_DILATION)
                bottomc = min(mask.shape[0], bottomc + FINAL_DILATION)
                mask[topc:bottomc, leftc:rightc] = True
                left = newleft
                top = newtop
                right = newright
                bottom = newbottom
                w = right - left
                h = bottom - top

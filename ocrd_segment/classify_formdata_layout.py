from __future__ import absolute_import

import os.path
import os
import numpy as np
from shapely.geometry import Polygon, asPolygon
from shapely.prepared import prep
from shapely.ops import unary_union
from skimage import draw
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # i.e. error
from mrcnn import model
from mrcnn.config import Config
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    coordinates_of_segment,
    coordinates_for_segment,
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

from .config import OCRD_TOOL

from maskrcnn_cli.formdata import FIELDS

TOOL = 'ocrd-segment-classify-formdata-layout'

class FormDataConfig(Config):
    """Configuration for detection on formdata segmentation"""
    NAME = "formdata"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    BACKBONE = "resnet50"
    # Number of classes (including background)
    NUM_CLASSES = 21 + 1  # SmartHEC has bg + 21 classes
    DETECTION_MAX_INSTANCES = 4 # will be reduced to 1 after cross-class NMS
    DETECTION_MIN_CONFIDENCE = 0.5
    PRE_NMS_LIMIT = 200
    POST_NMS_ROIS_INFERENCE = 50
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 600
    IMAGE_MAX_DIM = 768
    IMAGE_CHANNEL_COUNT = 5
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 0, 0])

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
        FormDataConfig.IMAGES_PER_GPU = self.parameter['images_per_gpu']
        FormDataConfig.DETECTION_MIN_CONFIDENCE = self.parameter['min_confidence']
        config = FormDataConfig()
        assert config.NUM_CLASSES == len(self.categories)
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
                #feature_filter='binarized', # models will be trained on RGB soon
                feature_selector='binarized', # models are trained on binary now
                transparency=False)
            if page_image_info.resolution != 1:
                dpi = page_image_info.resolution
                if page_image_info.resolutionUnit == 'cm':
                    dpi = round(dpi * 2.54)
            else:
                dpi = None
            page_image_binarized, _, _ = self.workspace.image_from_page(
                page, page_id,
                feature_selector='binarized')
            
            # ensure RGB (if raw was merely grayscale)
            page_image = page_image.convert(mode='RGB')
            page_array = np.array(page_image)
            # convert to RGB+Text+Address array
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
            _, components = cv2.connectedComponents(page_array_bin.astype(np.uint8))

            self._process_page(page, page_image, page_coords, page_id, page_array, components)
            
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
    
    def _process_page(self, page, page_image, page_coords, page_id, page_array, components):
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
        allregions = page.get_AllRegions(classes=['Text'], depth=2)
        active_categories = []
        for region in allregions:
            for line in region.get_TextLine():
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
        page.set_TextRegion([])
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
                    polygon_mask = draw.polygon(polygon[:,1], polygon[:,0], page_array.shape)
                    polygon_hull = draw.polygon_perimeter(polygon[:,1], polygon[:,0], page_array.shape)
                    polygon_tmask = (polygon_mask[0], polygon_mask[1], 3 * np.ones_like(polygon_mask[0]))
                    polygon_amask = (polygon_mask[0], polygon_mask[1], 4 * np.ones_like(polygon_mask[0]))
                    polygon_thull = (polygon_hull[0], polygon_hull[1], 3 * np.ones_like(polygon_hull[0]))
                    polygon_ahull = (polygon_hull[0], polygon_hull[1], 4 * np.ones_like(polygon_hull[0]))
                    # mark text in all arrays
                    for array in active_arrays:
                        array[polygon_tmask] = 255
                        array[polygon_thull] = 255
                    categories = get_context(segment)
                    # mark context, if any
                    for category in categories:
                        array = active_arrays[active_categories.index(category)]
                        array[polygon_amask] = 255
                        array[polygon_ahull] = 255
        # predict page image per-class as batch
        predictions = []
        batch_no = 0
        batch_size = self.model.config.BATCH_SIZE
        while len(predictions) < len(active_arrays):
            left = batch_no * batch_size
            right = (batch_no + 1) * batch_size
            arrays = active_arrays[left:right]
            categories = active_categories[left:right]
            while len(arrays) < batch_size:
                # last batch: pad with zeros
                arrays.append(np.zeros(arrays[0].shape, dtype=arrays[0].dtype))
                categories.append('')
            # convert to incidence matrix
            class_ids = np.eye(len(self.categories), dtype=np.int32)[np.array(
                [self.categories.index(category) if category in self.categories else 0
                 for category in categories])]
            predictions.extend(self.model.detect(arrays, active_class_ids=class_ids))
            batch_no += 1
        # concatenate instances for all classes of this page image
        preds = dict()
        preds["rois"] = np.concatenate([pred["rois"] for pred in predictions])
        preds["class_ids"] = np.concatenate([pred["class_ids"] for pred in predictions])
        preds["scores"] = np.concatenate([pred["scores"] for pred in predictions])
        preds["masks"] = np.dstack([pred["masks"] for pred in predictions])
        LOG.debug("Decoding %d ROIs for %d distinct classes (avg. score: %.2f)",
                  len(preds["class_ids"]),
                  len(np.unique(preds["class_ids"])),
                  np.mean(preds["scores"]) if all(preds["scores"].shape) else 0)
        # apply IoU-based non-maximum suppression across classes
        worse = []
        for i in range(len(preds["class_ids"])):
            for j in range(i + 1, len(preds['class_ids'])):
                imask = preds['masks'][:,:,i]
                jmask = preds['masks'][:,:,j]
                if np.count_nonzero(imask * jmask) / np.count_nonzero(imask + jmask) > 0.5:
                    # LOG.debug("pred %d[%s] overlaps pred %d[%s]",
                    #           i, self.categories[preds["class_ids"][i]],
                    #           j, self.categories[preds["class_ids"][j]])
                    worse.append(i if preds['scores'][i] < preds['scores'][j] else j)
        # find best-scoring instance per class
        best = np.zeros(len(self.categories))
        for i in range(len(preds['class_ids'])):
            if i in worse:
                continue
            class_id = preds['class_ids'][i]
            score = preds['scores'][i]
            if class_id in [0, 7]:
                # only 1-best instance for most classes (7 can be multiple)
                continue
            if score > best[class_id]:
                best[class_id] = score
        if not np.any(best):
            LOG.warning("Detected no form fields on page '%s'", page_id)
            return
        # post-process detections morphologically and decode to regions
        for i in range(len(preds['class_ids'])):
            if i in worse:
                LOG.debug("Ignoring instance for class %d overlapping better neighbour",
                          preds['class_ids'][i])
                continue
            class_id = preds['class_ids'][i]
            score = preds['scores'][i]
            if not class_id:
                raise Exception('detected region for background class')
            if score < best[class_id]:
                LOG.debug("Ignoring instance for class %d with non-maximum score",
                          preds['class_ids'][i])
                continue
            category = self.categories[class_id]
            mask = preds['masks'][:,:,i]
            # estimate glyph scale (roughly)
            bbox = np.around(preds['rois'][i])
            area = np.count_nonzero(mask)
            scale = int(np.sqrt(area)//10)
            scale = scale + (scale+1)%2 # odd
            LOG.debug("post-processing prediction for '%s' at %s area %d score %f",
                      category, str(bbox), area, score)
            assert mask.shape[:2] == components.shape[:2]
            # find closure in connected components
            complabels = np.unique(mask * components)
            if False:
                # overwrite pixel mask from (padded) outer bbox
                left, top, w, h = cv2.boundingRect(mask.astype(np.uint8))
                right = left + w
                bottom = top + h
                for label in complabels:
                    if not label:
                        continue # bg/white
                    leftc, topc, wc, hc = cv2.boundingRect((components==label).astype(np.uint8))
                    rightc = leftc + wc
                    bottomc = topc + hc
                    left = min(left, leftc)
                    top = min(top, topc)
                    right = max(right, rightc)
                    bottom = max(bottom, bottomc)
                left = max(0, left - 4)
                top = max(0, top - 4)
                right = min(page_image.width, right + 4)
                bottom = min(page_image.height, bottom + 4)
                mask[top:bottom, left:right] = mask.max()
            else:
                # fill pixel mask from (padded) inner bboxes
                for label in complabels:
                    if not label:
                        continue # bg/white
                    left, top, w, h = cv2.boundingRect((components==label).astype(np.uint8))
                    right = left + w
                    bottom = top + h
                    left = max(0, left - 4)
                    top = max(0, top - 4)
                    right = min(page_image.width, right + 4)
                    bottom = min(page_image.height, bottom + 4)
                    mask[top:bottom, left:right] = mask.max()
            # dilate until we have a single outer contour
            contours = [None, None]
            for _ in range(10):
                if len(contours) == 1:
                    break
                mask = cv2.dilate(mask.astype(np.uint8),
                                  np.ones((scale,scale), np.uint8)) > 0
                contours, _ = cv2.findContours(mask.astype(np.uint8),
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
            region_polygon = contours[0][:,0,:] # already in x,y order
            #region_polygon = polygon_from_bbox(bbox[1],bbox[0],bbox[3],bbox[2])
            # ensure consistent and valid polygon outline
            region_polygon = coordinates_for_segment(region_polygon,
                                                     page_image, page_coords)
            region_polygon = polygon_for_parent(region_polygon, page)
            if region_polygon is None:
                LOG.warning('Ignoring extant region for class %s', category)
                continue
            # annotate new region/line
            region_coords = CoordsType(points_from_polygon(region_polygon), conf=score)
            region_id = 'region%02d_%s' % (i+1, category)
            region = TextRegionType(id=region_id, Coords=region_coords, type_='other')
            line = TextLineType(id=region_id + '_line', Coords=region_coords,
                                custom='subtype:target=' + category)
            region.add_TextLine(line)
            page.add_TextRegion(region)
            page.set_custom('coords=%s' % page_coords['transform'])
            LOG.info("Detected %s region '%s' on page '%s'",
                     category, region_id, page_id)

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

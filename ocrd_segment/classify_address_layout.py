from __future__ import absolute_import

import os
import numpy as np
from shapely.geometry import Polygon, asPolygon
from shapely.prepared import prep
from shapely.ops import unary_union
import cv2
from PIL import Image, ImageDraw

#pylint: disable=wrong-import-position
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # i.e. error
from mrcnn import model
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
    TextRegionType,
    PageType,
    CoordsType,
    RegionRefType,
    RegionRefIndexedType,
    OrderedGroupType,
    OrderedGroupIndexedType,
    UnorderedGroupType,
    UnorderedGroupIndexedType,
    TextEquivType
)
from ocrd_modelfactory import page_from_file
from ocrd import Processor

from maskrcnn_cli.address import InferenceConfig

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-classify-address-layout'

class ClassifyAddressLayout(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(ClassifyAddressLayout, self).__init__(*args, **kwargs)
        if hasattr(self, 'output_file_grp'):
            # processing context
            self.setup()

    def setup(self):
        LOG = getLogger('processor.ClassifyAddressLayout')
        self.categories = ['',
                           'address-rcpt',
                           'address-sndr',
                           'address-contact']
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
        config.DETECTION_MAX_INSTANCES = 10 # will be reduced to 5 after cross-class NMS
        assert config.NUM_CLASSES == len(self.categories)
        proto = tf.compat.v1.ConfigProto()
        proto.gpu_options.allow_growth = True  # dynamically alloc GPU memory as needed
        # avoid over-allocation / OOM
        if 'MRCNNPROCS' in os.environ:
            # share GPU with nprocs others
            proto.gpu_options.per_process_gpu_memory_fraction = 1.0 / int(os.environ['MRCNNPROCS'])
        # fall-back to CPU / swap-out
        #proto.gpu_options.experimental.use_unified_memory = True # allow swapping memory back to CPU instead of OOM
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
        """Detect and classify+resegment address regions from text recognition results.
        
        Open and deserialize PAGE input files and their respective images,
        then iterate over the element hierarchy down to the text line level.
        
        Then, get the text classification result for each text line (what
        parts of address descriptions it contains, if any).
        
        Next, retrieve the page image according to the layout annotation (from
        the alternative image of the page, or by cropping at Border and deskewing
        according to @orientation) in raw RGB form. Represent it as an array with
        2 extra channels, one marking text lines, the other marking text lines
        containing address components.
        
        Pass that array to the visual address detector model, and retrieve region
        candidates as tuples of region class, bounding box, and pixel mask.
        Postprocess the mask and bbox to ensure no words are cut off accidentally.
        
        Where the class confidence is high enough, annotate the resulting TextRegion
        (including the special address type), and remove any overlapping input regions
        (re-assigning its text lines).
        
        Produce a new output file by serialising the resulting hierarchy.
        """
        LOG = getLogger('processor.ClassifyAddressLayout')
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
            # get connected components
            _, components, cstats, _  = cv2.connectedComponentsWithStats(page_array_bin.astype(np.uint8))
            # estimate glyph scale (roughly)
            scalemap = np.zeros_like(components)
            for label in np.argsort(cstats[:,4]):
                if not label: continue
                left, top, width, height, area = cstats[label]
                right, bottom = left + width, top + height
                if np.max(scalemap[top:bottom, left:right]) > 0:
                    continue
                scalemap[top:bottom, left:right] = np.sqrt(area)
            scalemap = scalemap[(5 / zoom * zoomed < scalemap) & (scalemap < 100 / zoom * zoomed)]
            if np.any(scalemap):
                scale = int(np.median(scalemap))
            else:
                scale = int(43 / zoom * zoomed)
            LOG.debug("detected scale = %d", scale)

            # prepare mask image (alpha channel for input image)
            page_image_mask = Image.new(mode='L', size=page_image.size, color=0)
            def mark_line(line):
                text_class = line.get_custom() or ''
                text_class = text_class.replace('subtype: ', '')
                if not text_class.startswith('ADDRESS_'):
                    text_class = 'ADDRESS_NONE'
                # add to mask image (alpha channel for input image)
                polygon = coordinates_of_segment(line, page_image, page_coords)
                if zoomed != 1.0:
                    polygon = np.round(polygon * zoomed).astype(np.int32)
                # draw line mask:
                ImageDraw.Draw(page_image_mask).polygon(
                    list(map(tuple, polygon.tolist())),
                    fill=200 if text_class == 'ADDRESS_NONE' else 255)
            # iterate through all regions that could have text/context lines
            allregions = page.get_AllRegions(classes=['Text'], order='reading-order', depth=2)
            alllines = []
            for region in allregions:
                for line in region.get_TextLine():
                    alllines.append(line)
                    mark_line(line)
            
            # combine raw with aggregated mask to RGBA array
            page_image.putalpha(page_image_mask)
            page_array = np.array(page_image)
            # convert to RGB+Text+Context array
            tmask = page_array[:,:,3:4] > 0
            amask = page_array[:,:,3:4] == 255
            page_array = np.concatenate([page_array[:,:,:3],
                                         255 * tmask.astype(np.uint8),
                                         255 * amask.astype(np.uint8)],
                                        axis=2)
            
            # prepare reading order
            reading_order = dict()
            ro = page.get_ReadingOrder()
            if ro:
                rogroup = ro.get_OrderedGroup() or ro.get_UnorderedGroup()
                if rogroup:
                    page_get_reading_order(reading_order, rogroup)
            
            # predict
            K.tensorflow_backend.set_session(self.sess)
            preds = self.model.detect([page_array], verbose=0)[0]
            worse = []
            instances = np.arange(len(preds['class_ids']))
            instances_i, instances_j = np.meshgrid(instances, instances, indexing='ij')
            combinations = list(zip(*np.where(instances_i < instances_j)))
            for i, j in combinations:
                imask = preds['masks'][:,:,i]
                jmask = preds['masks'][:,:,j]
                intersection = np.count_nonzero(imask * jmask)
                if not intersection:
                    continue
                union = np.count_nonzero(imask + jmask)
                if intersection / union > 0.5:
                    worse.append(i if preds['scores'][i] < preds['scores'][j] else j)
            best = np.zeros(4)
            for i in range(len(preds['class_ids'])):
                if i in worse:
                    continue
                cat = preds['class_ids'][i]
                score = preds['scores'][i]
                if cat in [3] and score > best[cat]:
                    # only best probs for sndr and rcpt (contact can be many)
                    best[cat] = score
            if not np.any(best):
                LOG.warning("Detected no sndr/rcpt address on page '%s'", page_id)
            newregions = []
            newpolys = []
            for i in range(len(preds['class_ids'])):
                cat = preds['class_ids'][i]
                name = self.categories[cat]
                score = preds['scores'][i]
                if not cat:
                    raise Exception('detected region for background class')
                if i in worse:
                    LOG.debug("Ignoring instance for %s overlapping better neighbour", name)
                    continue
                if score < self.parameter['min_confidence']:
                    LOG.debug("Ignoring instance for %s with too low score (%.2f)", name, score)
                    continue
                if score < best[cat]:
                    LOG.debug("reassigning instance for %s with non-maximum score to address-contact",
                              name)
                    name = "address-contact"
                mask = preds['masks'][:,:,i]
                area = np.count_nonzero(mask)
                bbox = np.around(preds['rois'][i])
                top, left, bottom, right = bbox
                w, h = right - left, bottom - top
                LOG.debug("post-processing prediction for %s at %s area %d score %f",
                          name, str(bbox), area, score)
                # fill pixel mask from (padded) inner bboxes
                complabels = np.unique(mask * components)
                for label in complabels:
                    if not label:
                        continue # bg/white
                    suppress = False
                    leftc, topc, wc, hc = cv2.boundingRect((components==label).astype(np.uint8))
                    rightc = leftc + wc
                    bottomc = topc + hc
                    if wc > 2 * w or hc > 2 * h:
                        # huge (non-text?) component
                        suppress = True
                    if (min(right, rightc) - max(left, leftc)) * \
                       (min(bottom, bottomc) - max(top, topc)) < 0.4 * wc * hc:
                        # intersection over component too small
                        # (snap inverse)
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
                        leftc = min(mask.shape[1], leftc + 4)
                        topc = min(mask.shape[0], topc + 4)
                        rightc = max(0, rightc - 4)
                        bottomc = max(0, bottomc - 4)
                        mask[topc:bottomc, leftc:rightc] = False
                    else:
                        leftc = max(0, leftc - 4)
                        topc = max(0, topc - 4)
                        rightc = min(mask.shape[1], rightc + 4)
                        bottomc = min(mask.shape[0], bottomc + 4)
                        mask[topc:bottomc, leftc:rightc] = True
                        left = newleft
                        top = newtop
                        right = newright
                        bottom = newbottom
                        w = right - left
                        h = bottom - top
                # dilate and find (outer) contour
                contours = [None, None]
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
                    LOG.warning("Ignoring non-contiguous (%d) region for '%s'", len(contours), name)
                    continue
                region_polygon = contours[0][:,0,:] # already in x,y order
                #region_polygon = polygon_from_bbox(bbox[1],bbox[0],bbox[3],bbox[2])
                if zoomed != 1.0:
                    region_polygon = region_polygon / zoomed
                region_poly = prep(make_valid(Polygon(region_polygon)))
                region_polygon = coordinates_for_segment(region_polygon,
                                                         page_image, page_coords)
                region_polygon = polygon_for_parent(region_polygon, page)
                if region_polygon is None:
                    LOG.warning('Ignoring extant region for class %s', name)
                    continue
                # annotate new region
                region_coords = CoordsType(points_from_polygon(region_polygon), conf=score)
                region_id = 'addressregion%02d' % (i+1)
                region = TextRegionType(id=region_id,
                                        Coords=region_coords,
                                        type_='other',
                                        custom='subtype:' + name)
                LOG.info("Detected %s region '%s' on page '%s'",
                         name, region_id, page_id)
                newregions.append(region)
                newpolys.append(region_poly)
            # match / re-assign existing text lines to new regions
            # (each to its largest intersecting candidate)
            for line in alllines:
                polygon = coordinates_of_segment(line, page_image, page_coords)
                linepoly = make_valid(Polygon(polygon))
                best = None, 0
                for newreg, newpoly in zip(newregions, newpolys):
                    if newpoly.intersects(linepoly):
                        interp = newpoly.context.intersection(linepoly)
                        if interp.area > best[1]:
                            best = newreg, interp.area
                if not best[0] or best[1] < 0.5 * linepoly.area:
                    # no match for this line
                    custom = line.get_custom()
                    if custom and custom.startswith('subtype: ADDRESS_'):
                        LOG.warning("Keeping %s line '%s' without matching address region",
                                    custom[9:], line.id)
                    continue
                region = best[0]
                polygon = coordinates_of_segment(region, page_image, page_coords)
                regpoly = make_valid(Polygon(polygon))
                LOG.debug("stealing text line '%s' for '%s'", line.id, region.id)
                region.add_TextLine(line)
                if not linepoly.within(regpoly):
                    regpoly = linepoly.union(regpoly)
                if regpoly.type == 'MultiPolygon':
                    regpoly = regpoly.convex_hull
                polygon = coordinates_for_segment(regpoly.exterior.coords[:-1], page_image, page_coords)
                region.get_Coords().points = points_from_polygon(polygon)
            # validate new regions and persist the new assignments
            for region in newregions:
                # safe-guard against ghost detections:
                if not region.TextLine:
                    LOG.info("Ignoring region '%s' without any lines at all", region.id)
                    continue
                if not any(line.get_custom() and line.get_custom().startswith('subtype: ADDRESS_')
                           for line in region.TextLine):
                    LOG.info("Ignoring region '%s' without any address lines", region.id)
                    continue
                # persist the steal
                for line in region.get_TextLine():
                    oldreg = line.parent_object_
                    oldreg.TextLine.remove(line)
                    if not oldreg.TextLine and not oldreg.TextRegion:
                        # remove old region
                        LOG.debug("found redundant region '%s' for '%s'",
                                  oldreg.id, region.id)
                        oldreg.parent_object_.TextRegion.remove(oldreg)
                        if oldreg.id in reading_order:
                            roelem = reading_order[oldreg.id]
                            roelem.set_regionRef(region.id)
                            reading_order[region.id] = roelem
                            del reading_order[oldreg.id]
                    else:
                        # update old region's text content
                        oldreg.set_TextEquiv([TextEquivType(Unicode='\n'.join(
                            otherline.get_TextEquiv()[0].Unicode
                            for otherline in oldreg.get_TextLine()
                            if otherline.get_TextEquiv()))])
                # re-order new region's text lines by relative order in original segmentation
                region.TextLine = sorted(region.TextLine, key=lambda line: alllines.index(line))
                # update new region's text content
                region.set_TextEquiv([TextEquivType(Unicode='\n'.join(
                    line.get_TextEquiv()[0].Unicode
                    for line in region.get_TextLine()
                    if line.get_TextEquiv()))])
                # keep new region
                page.add_TextRegion(region)

            file_id = make_file_id(input_file, self.output_file_grp)
            file_path = os.path.join(self.output_file_grp,
                                     file_id + '.xml')
            pcgts.set_pcGtsId(file_id)
            out = self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                local_filename=file_path,
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts))
            LOG.info('created file ID: %s, file_grp: %s, path: %s',
                     file_id, self.output_file_grp, out.local_filename)

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
        ro[elem.get_regionRef()] = elem
        if not isinstance(elem, (RegionRefType, RegionRefIndexedType)):
            page_get_reading_order(ro, elem)

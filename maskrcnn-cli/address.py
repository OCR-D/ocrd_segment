"""
Mask R-CNN
Configurations and data loading code for address resegmentation
(textline alpha-masked input PNG images, address region output COCO JSON).

Based on coco.py in matterport/MaskRCNN.

------------------------------------------------------------

Usage: import the module, or run from the command line as such:

    # Train a new model starting from pre-trained COCO weights
    maskrcnn-address --model=/path/to/mask_rcnn_coco.h5 train --dataset=/path/to/coco*.json

    # Train a new model starting from ImageNet weights
    maskrcnn-address --model=imagenet train --dataset=/path/to/coco*.json

    # Continue training a model that you had trained earlier
    maskrcnn-address --model=/path/to/weights.h5 train --dataset=/path/to/coco*.json

    # Continue training the last model you trained
    maskrcnn-address --model=last train --dataset=/path/to/coco*.json

    # Run COCO prediction+evaluation on the last model you trained
    maskrcnn-address --model=last evaluate --dataset=/path/to/coco*.json

    # Run COCO prediction+evaluation on the last model you trained (only first 100 files, writing plot files)
    maskrcnn-address --model=last --limit 100 --plot pred evaluate --dataset=/path/to/coco*.json

    # Run COCO prediction on the last model you trained (creating new COCO along existing COCO for comparison)
    maskrcnn-address --model=last predict --dataset=/path/to/coco.json --dataset-pred=/path/to/coco-pred.json

    # Merge and sort multiple COCO datasets (for comparison)
    maskrcnn-address merge --dataset=/path/to/coco*.json --split 0 --dataset-merged=/path/to/coco-merged.json --sort --rle
    maskrcnn-address merge --dataset=/path/to/coco-pred*.json --split 0 --dataset-merged=/path/to/coco-pred-merged.json --sort --rle

    # Run COCO evaluation between ground truth and predicted dataset:
    maskrcnn-address compare --dataset=/path/to/coco-merged.json --dataset-pred=/path/to/coco-pred-merged.json

    # Run COCO prediction on the last model you trained (creating new COCO for arbitrary files)
    maskrcnn-address --model=last test --dataset=/path/to/files.json /path/to/files*

    # Run COCO prediction on the last model you trained (...also writing plot files)
    maskrcnn-address --model=last test --plot pred --dataset=/path/to/files.json /path/to/files*
"""

import os
import time
import pathlib
import json
import argparse

import numpy as np
import skimage.io
import skimage.color
import scipy.ndimage.measurements as measurements
import imgaug

from matplotlib import pyplot, cm
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection
from PIL import Image

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# Import Mask RCNN
#pylint: disable=wrong-import-position
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # i.e. error
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import tensorflow as tf
from keras.callbacks import Callback
#pylint: disable=wrong-import-position
tf.get_logger().setLevel('ERROR')

ALPHA_TEXT_CHANNEL = 200
ALPHA_CTXT_CHANNEL = 255

############################################################
#  Configurations
############################################################


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "address"

    # We use a GPU with 4GB memory, which can fit one images
    # (when training all layers).
    # Adjust up if you use a larger GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of training steps (batches) per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    # Mind that the number of samples presented during a single epoch
    # during training equals STEPS_PER_EPOCH * IMAGES_PER_GPU * GPU_COUNT.
    STEPS_PER_EPOCH = 1000

    # Number of classes (including background)
    NUM_CLASSES = 3 + 1  # address has 3 classes (rcpt/sndr/contact)

    # ...settings to reduce GPU memory requirements...

    # Use a smaller backbone network. The default is resnet101,
    # but you can use resnet50 to reduce memory load significantly
    # and it's sufficient for most applications. It also trains faster.
    BACKBONE = "resnet50"

    # Reduce the maximum number of instances per image if your images
    # don't have a lot of objects.
    MAX_GT_INSTANCES = 5
    # Max number of final detections
    DETECTION_MAX_INSTANCES = 5
    DETECTION_MIN_CONFIDENCE = 0.9

    # Use fewer ROIs in training the second stage. This setting
    # is like the batch size for the second stage of the model.
    # (includes subsampling of both positive and negative examples)
    TRAIN_ROIS_PER_IMAGE = 50

    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    #PRE_NMS_LIMIT = 6000
    PRE_NMS_LIMIT = 2000

    # ROIs kept after non-maximum suppression (training and inference)
    #POST_NMS_ROIS_TRAINING = 2000
    #POST_NMS_ROIS_INFERENCE = 1000
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 500

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 600
    IMAGE_MAX_DIM = 768

    # ...settings to accommodate alpha channel input...
    IMAGE_CHANNEL_COUNT = 5
    # don't touch the alpha channel
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 0, 0])

class InferenceConfig(CocoConfig):
    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    # We use a GPU with 4GB memory, which can fit 6 images
    # (during prediction), but yields highest throughput at 4.
    # (Full utilization of the GPU is only possible when
    #  using the detect_generator loop, which runs image
    #  pre-/postprocessing and GPU prediction in parallel.)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

############################################################
#  Image/Segmentation Augmentation
############################################################

class SegmapDropout(imgaug.augmenters.meta.Augmenter):
    """Augment by randomly dropping instances from GT and image alpha.

    Probabilistic augmenter that takes instance masks
    (in the form of segmentation maps with bg as index 0)
    and randomly drops a fraction of instances
    by setting its output segmap / mask to bg, and
    by setting its input 5th / address channel to 0 in that area.
    This is supposed to help not rely too much on certain instances
    always appearing (and also learn to cope with empty pages).
    """
    def __init__(self, p=0.3,
                 seed=None, name=None,
                 random_state="deprecated",
                 deterministic="deprecated"):
        super(SegmapDropout, self).__init__(
            seed=seed, name=name,
            random_state=random_state,
            deterministic=deterministic)
        self.p = imgaug.parameters.Binomial(1 - p)
    def _augment_batch_(self, batch, random_state, parents, hooks):
        for i in range(batch.nb_rows):
            image = batch.images[i]
            segmap = batch.segmentation_maps[i].arr
            assert segmap.shape[-1] == 1, "segmentation map is not in argmax form"
            ninstances = segmap.max()
            p = self.p.draw_samples((ninstances,), random_state=random_state)
            drop = p < 0.5
            drop = np.insert(drop, 0, [False]) # never "drop" background
            image[drop[segmap][:,:,0],4] = 0 # set amask=0
            segmap[drop[segmap]] = 0 # set to bg
            batch.images[i] = image
            batch.segmentation_maps[i].arr = segmap
        return batch
    def get_parameters(self):
        return [self.p]
    
class SegmapDropoutLines(imgaug.augmenters.meta.Augmenter):
    """Augment by randomly dropping instances' first line from image alpha.

    Probabilistic augmenter that takes instance masks
    (in the form of segmentation maps with bg as index 0),
    and randomly degrades a fraction of instances
    by setting its input 5th / address channel to 0
    in the top lines but keeping its output segmap / mask.
    This is supposed to help become robust against
    title/name part of address lines being undetected.
    """
    def __init__(self, p=0.3,
                 seed=None, name=None,
                 random_state="deprecated",
                 deterministic="deprecated"):
        super(SegmapDropoutLines, self).__init__(
            seed=seed, name=name,
            random_state=random_state,
            deterministic=deterministic)
        self.p = imgaug.parameters.Binomial(1 - p)
    def _augment_batch_(self, batch, random_state, parents, hooks):
        for i in range(batch.nb_rows):
            image = batch.images[i]
            segmap = batch.segmentation_maps[i].arr
            assert segmap.shape[-1] == 1, "segmentation map is not in argmax form"
            ninstances = segmap.max()
            p = self.p.draw_samples((ninstances,), random_state=random_state)
            drop = p < 0.5
            drop = np.insert(drop, 0, [False]) # never "drop" background
            labels, _ = measurements.label(image[:,:,3])
            # only instances with more than 1 label (line)
            # TODO: vectorize!
            for instance in drop.nonzero()[0]:
                masked = labels * (segmap[:,:,0] == instance)
                lines = np.unique(masked)
                # at least 3 lines (bg + 2 text lines)
                if len(lines) > 2:
                    top = np.argmin([np.nonzero(labels == line)[0].mean() for line in lines])
                    image[labels == lines[top],4] = 0 # set amask=0 of first line
            batch.images[i] = image
        return batch
    def get_parameters(self):
        return [self.p]

class SegmapBlackoutLines(imgaug.augmenters.meta.Augmenter):
    """Augment by randomly drawing black boxes over non-instances' lines in the image.

    Probabilistic augmenter that takes instance masks
    (in the form of segmentation maps with bg as index 0),
    and randomly degrades a fraction of non-instance lines
    (i.e. areas with 1 at their input 4th / text channel)
    by setting its input RGB channels to 0.
    This is supposed to help become robust against
    input images with anonymization.
    """
    def __init__(self, p=0.1, padding=5,
                 seed=None, name=None,
                 random_state="deprecated",
                 deterministic="deprecated"):
        super(SegmapBlackoutLines, self).__init__(
            seed=seed, name=name,
            random_state=random_state,
            deterministic=deterministic)
        self.p = imgaug.parameters.Binomial(1 - p)
        self.padding = padding
    def _augment_batch_(self, batch, random_state, parents, hooks):
        for i in range(batch.nb_rows):
            image = batch.images[i]
            segmap = batch.segmentation_maps[i].arr
            assert segmap.shape[-1] == 1, "segmentation map is not in argmax form"
            mask = image[:,:,3] > 0
            labels, nlabels = measurements.label(mask.astype(np.int32))
            p = self.p.draw_samples((nlabels,), random_state=random_state)
            drop = p < 0.5
            drop = np.insert(drop, 0, [False]) # never box out non-text
            for label in drop.nonzero()[0]:
                if not label:
                    continue
                mask = labels == label
                if np.any(mask & (segmap[:,:,0] > 0)):
                    # text line is part of GT RoI instance
                    continue
                y, x = np.nonzero(mask)
                ymin = np.maximum(0, y.min() - self.padding)
                ymax = np.minimum(mask.shape[0], y.max() + self.padding)
                xmin = np.maximum(0, x.min() - self.padding)
                xmax = np.minimum(mask.shape[1], x.max() + self.padding)
                y, x = np.indices((ymax-ymin, xmax-xmin))
                y += ymin
                x += xmin
                image[y, x, 0:3] = 0 # set RGB to black
            batch.images[i] = image
        return batch
    def get_parameters(self):
        return [self.p, self.padding]

class SaveDebugImage(imgaug.augmenters.meta.Augmenter):
    """Augment by randomly dropping instances' first line from image alpha.

    Augmenter that takes instance masks
    (in the form of segmentation maps with bg as index 0),
    and draws them on the image, reducing tmask/amask to alpha
    (by setting coloring output segmap / mask and
     by setting input alpha channel to 255 if amask else 200 if tmask else 0),
    and then blending the alpha onto RGB.
    These images are written as temporary files.
    Use them for debugging only (RGB, not RGBA as in CocoDataset).
    (The batch itself is not modified.)
    """
    def __init__(self, title='images',
                 seed=None, name=None,
                 random_state="deprecated",
                 deterministic="deprecated"):
        super(SaveDebugImage, self).__init__(
            seed=seed, name=name,
            random_state=random_state,
            deterministic=deterministic)
        self.title = title
    def _augment_batch_(self, batch, random_state, parents, hooks):
        images = []
        for i in range(batch.nb_rows):
            img = batch.images[i]
            # if tmask == 0, set RGB to 255 (white)
            # elif amask == 0, scale from 0..255 to 200..255 (gray)
            # else keep full contrast
            image = img[:,:,:3]
            tmask = img[:,:,3]
            amask = img[:,:,4]
            # print("image with %f%% text with %f%% address" % (
            #     100.0 * np.count_nonzero(tmask == 255)/np.prod(tmask.shape),
            #     100.0 * np.count_nonzero(amask == 255)/np.count_nonzero(tmask == 255)))
            image[amask < 255] = 200 + 55/255 * image[amask < 255]
            image[tmask < 255] = 255
            images.append(image)
        image = imgaug.augmenters.debug.draw_debug_image(
            images,
            segmentation_maps=batch.segmentation_maps)
        from imageio import imwrite
        from tempfile import mkstemp
        from os import close
        fd, fname = mkstemp(suffix=self.title + '.png')
        imwrite(fname, image)
        close(fd)
        return batch
    def get_parameters(self):
        return []
    
############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
    def __init__(self):
        super().__init__()
        # Add classes
        for i, name in enumerate([None,
                                  "address-rcpt",
                                  "address-sndr",
                                  "address-contact"]):
            if name:
                self.add_class("IAO", i, name)

    def load_coco(self, dataset_json, dataset_dir='.',
                  limit=None, class_ids=None,
                  class_map=None, return_coco=False):
        """Load a subset of the COCO dataset from a JSON file.
        dataset_json: JSON file path of the COCO (sub-) dataset.
        dataset_dir: parent directory of relative filenames in JSON
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object, and retains all its IDs.
        """
        if not class_ids:
            class_ids = []

        if isinstance(dataset_json, COCO):
            coco = dataset_json
        else:
            coco = COCO(dataset_json)

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id_ in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id_])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Limit to a subset
        if isinstance(limit, int):
            image_ids = image_ids[:limit]
        elif isinstance(limit, (list, np.ndarray)):
            image_ids = np.array(image_ids).take(limit)
        
        # Add images
        # we cannot keep the image_id refs, because Dataset can load multiple COCO files
        for i, id_ in enumerate(image_ids, len(self.image_info)):
            file_name = coco.imgs[id_]['file_name']
            ann_ids = coco.getAnnIds(
                imgIds=[id_], catIds=class_ids, iscrowd=None)
            self.add_image(
                "IAO", image_id=id_ if return_coco else i,
                path=os.path.join(dataset_dir, file_name),
                width=coco.imgs[id_]["width"],
                height=coco.imgs[id_]["height"],
                # still contains original/COCO image_id refs
                # and inconsistent/clashing id refs:
                annotations=coco.loadAnns(ann_ids))
        if return_coco:
            return coco
        return None

    def load_files(self, filenames, dataset_dir='.', limit=None):
        if isinstance(limit, int):
            filenames = filenames[:limit]
        elif isinstance(limit, (list, np.ndarray)):
            filenames = np.array(filenames).take(limit)
        for i, filename in enumerate(filenames, len(self.image_info)):
            filename = os.path.join(dataset_dir, filename)
            if not os.path.exists(filename):
                print('skipping image "%s" with non-existing filename "%s"' % (i, filename))
                continue
            with Image.open(filename) as image_pil:
                width = image_pil.width
                height = image_pil.height
            self.add_image(
                "IAO", image_id=i, path=filename,
                width=width, height=height)

    def image_reference(self, image_id):
        return self.image_info[image_id]['path']

    def dump_coco(self, dataset_dir='.'):
        """Dump dataset into an COCO JSON file."""
        result = { 'categories': self.class_info, 'images': list(), 'annotations': list() }
        i = 0
        for image_id in self.image_ids:
            image_info = self.image_info[image_id]
            result['images'].append({ 'id': image_info['id'],
                                      'width': image_info['width'],
                                      'height': image_info['height'],
                                      'file_name': str(pathlib.Path(image_info['path']).relative_to(dataset_dir)),
                                      })
            if 'annotations' in image_info:
                for ann in image_info['annotations']:
                    # ensure correct image_id and consistent id
                    ann = ann.copy()
                    ann['image_id'] = image_info['id']
                    ann['id'] = i
                    i += 1
                    result['annotations'].append(ann)
        return result

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "IAO":
            print('invalid source "%s" for image %d ("%s")' % (
                image_info['source'], image_id, image_info['path']))
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "IAO.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
        else:
            # print('no instance in image %d ("%s")' % (
            #     image_id, image_info['path']))
            mask = np.empty([0, 0, 0], dtype=np.bool)
            class_ids = np.empty([0], dtype=np.int32)
        return mask, class_ids

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,4] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has no alpha channel, complain
        if image.shape[-1] != 4:
            raise Exception('image %d ("%s") has no alpha channel' % (
                image_id, self.image_info[image_id]['path']))
        # Convert from RGBA to RGB+Text+Context
        tmask = image[:,:,3:4] > 0
        cmask = image[:,:,3:4] == ALPHA_CTXT_CHANNEL
        image = np.concatenate([image[:,:,:3],
                                255 * tmask.astype(np.uint8),
                                255 * cmask.astype(np.uint8)],
                               axis=2)
        return image
    
    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_id, rois, class_ids, scores, masks):
    """Arrange results to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    # Loop through detections
    for i in range(rois.shape[0]):
        class_id = class_ids[i]
        score = scores[i]
        bbox = np.around(rois[i], 1)
        mask = masks[:, :, i]

        result = {
            "image_id": image_id,
            "category_id": dataset.get_source_class_id(class_id, "IAO"),
            "iscrowd": 0,
            "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
            "score": score,
            "segmentation": maskUtils.encode(np.asfortranarray(mask))
        }
        results.append(result)
    return results

def detect_coco(model, dataset, verbose=False, limit=None, image_ids=None, plot=None):
    """Predict images
    dataset: A Dataset object with test data
    verbose: If not False, print summary of detection for each image
    limit: if not 0, it's the number of images to use for test
    image_ids: if not None, list or array of image IDs to use for test
    plot: if not None, write an image file showing the predictions color-coded.
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if isinstance(limit, int):
        image_ids = image_ids[:limit]
    elif isinstance(limit, (list, np.ndarray)):
        image_ids = np.array(image_ids).take(limit)
    if not len(image_ids):
        print("Ignoring empty dataset")
        return [], []

    t_prediction = 0
    t_start = time.time()
    class TimingCallback(Callback):
        def __init__(self):
            super(TimingCallback, self).__init__()
            self.time = 0
        def on_predict_batch_begin(self, batch, logs=None):
            self.time = time.time()
        def on_predict_batch_end(self, batch, logs=None):
            nonlocal t_prediction
            t_prediction += (time.time() - self.time)

    results = []
    cocoids = []
    generator = modellib.InferenceDataGenerator(dataset, model.config,
                                                image_ids=image_ids)
    # Run detection
    preds = model.detect_generator(generator, workers=3,
                                   verbose=1, callbacks=[TimingCallback()])
    for i, image_id in enumerate(image_ids):
        # Load image
        image_path = dataset.image_info[image_id]['path']
        image_cocoid = dataset.image_info[image_id]['id']
        image_source = dataset.image_info[image_id]['source']
        r = preds[i]
        assert image_cocoid == r['image_id'], "Generator queue failed to preserve image order"
        if verbose:
            print("image {} {} has {} rois with {} distinct classes".format(
                image_cocoid, image_path,
                r['masks'].shape[-1], len(np.unique(r['class_ids']))))

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, image_cocoid,
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        dataset.image_info[image_id].update({'annotations': image_results})
        results.extend(image_results)
        cocoids.append(image_cocoid)
        if plot:
            plot_result(dataset.load_image(image_id), image_results,
                        dataset.image_info[image_id]['width'],
                        dataset.image_info[image_id]['height'],
                        pathlib.Path(image_path).with_suffix('.' + plot + '.png'))

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids) if len(image_ids) else 0))
    print("Total time: ", time.time() - t_start)
    time.sleep(0.1) # avoid multithreading deadlocks (https://github.com/keras-team/keras/issues/11288)

    for i, ann in enumerate(results):
        ann['id'] = i
    return results, cocoids

def sort_coco(coco, image_ids=None):
    """Sort images in COCO dataset by their file_name alphabetically, reassigning image_ids"""
    images = coco.dataset['images']
    if image_ids is not None:
        images = [img for img in images if img['id'] in image_ids]
    def fname(img):
        return img['file_name']
    # redefine image ids by file_name sort order
    images2 = sorted(images, key=fname)
    mapping = dict((id_, i) for i, id_ in enumerate(img['id'] for img in images))
    coco.dataset['images'] = images2
    for img in images:
        img['id'] = mapping[img['id']]
    for ann in coco.dataset['annotations']:
        ann['image_id'] = mapping[ann['image_id']]
    coco.createIndex()
    return coco

def evaluate_coco(coco_gt, coco_res, limit=None, image_ids=None, eval_type='segm'):
    """Runs official COCO evaluation, printing additional statistics
    coco_gt: A COCO dataset with validation data
    coco_res: A COCO dataset with prediction data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    cocoEval = COCOeval(coco_gt, coco_res, eval_type)
    if isinstance(image_ids, list):
        cocoEval.params.imgIds = image_ids
    elif isinstance(limit, int):
        cocoEval.params.imgIds = sorted(coco_gt.getImgIds())[:limit]
    #cocoEval.params.catIds = [...]
    #cocoEval.params.iouThrs = [.5:.05:.95]
    ##cocoEval.params.iouThrs = np.linspace(.3, .95, 14)
    #cocoEval.params.maxDets = [10]
    cocoEval.params.useCats = 1
    print("Evaluating predictions against GT with classes")
    cocoEval.evaluate()
    def file_name(img):
        image_info = coco_gt.imgs[img['image_id']]
        return str(image_info['file_name'])
    def cat_name(img):
        class_info = coco_gt.cats[img['category_id']]
        return class_info['name']
    for img in sorted(cocoEval.evalImgs,
                      key=lambda img: file_name(img) if img else ""):
        if not img:
            continue
        if img['aRng'] != cocoEval.params.areaRng[0]:
            # ignore other restricted area ranges
            continue
        print((file_name(img) + '|' +
               cat_name(img) + ': ' +
               'GT matches=' + str(img['gtMatches'][0] > 0) + ' ' +
               'pred scores=' + str(img['dtScores'])))
    cocoEval.accumulate()
    # show precision/recall at:
    # T[0]=0.5 IoU
    # R[*] recall threshold equal to max recall
    # K[*] each class
    # A[0] all areas
    # M[-1]=100 max detections
    print("class precision/recall at IoU=0.5 max-recall all-area max-detections:")
    recalls = cocoEval.eval['recall'][0,:,0,-1]
    recallInds = np.searchsorted(np.linspace(0, 1, 101), recalls) - 1
    classInds = np.arange(len(recalls))
    precisions = cocoEval.eval['precision'][0,recallInds,classInds,0,-1]
    catIds = coco_gt.getCatIds()
    for id_, cat in coco_gt.cats.items():
        name = cat['name']
        i = catIds.index(id_)
        print(name + ' prc: ' + str(precisions[i]))
        print(name + ' rec: ' + str(recalls[i]))
    cocoEval.summarize()
    cocoEval.params.useCats = 0
    print("Evaluating predictions against GT ignoring classes")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

def showAnns(anns, height, width):
    """
    Display the specified annotations.
    :param anns (array of object): annotations to display
    :return: None
    """
    if len(anns) == 0:
        return
    ax = pyplot.gca()
    ax.set_autoscale_on(False)
    polygons = []
    colors = []
    for ann in anns:
        category = ann['category_id']
        color = cm.tab10(category, alpha=0.8)
        if isinstance(ann['segmentation'], list):
            # polygon
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((int(len(seg)/2), 2))
                polygons.append(Polygon(poly))
                colors.append(color)
        else:
            # mask
            if isinstance(ann['segmentation']['counts'], list):
                rle = maskUtils.frPyObjects([ann['segmentation']], height, width)
            else:
                rle = [ann['segmentation']]
            m = maskUtils.decode(rle)
            ax.imshow(m * color)
        color = cm.tab10(category, alpha=0.5)
        x, y, w, h = ann['bbox']
        ax.add_patch(Rectangle((x, y), w, h, fill=False, color=color))
    p = PatchCollection(polygons, facecolor=colors, linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=colors, linewidths=2)
    ax.add_collection(p)

def plot_result(image, anns, width, height, filename):
    # show result
    fig = pyplot.figure(frameon=False)
    pyplot.imshow(image[:,:,:3])
    ax = pyplot.gca()
    ax.set_axis_off()
    ax.set_xmargin(0)
    ax.set_ymargin(0)
    ax.set_frame_on(0)
    ax.set_position([0,0,1,1])
    #ax.title(image_cocoid)
    showAnns(anns, height, width)
    # make an extra effort to arrive at the same image size
    # (no frames, axes, margins):
    fig.set_size_inches((width/300, height/300))
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    pyplot.savefig(filename, dpi=300, pad_inches=0, bbox_inches=extent)
    fig.clf()
    pyplot.close()

def json_safe(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bytes):
        return obj.decode('UTF-8')
    else:
        return obj.__str__()

def store_coco(coco, filename, dataset_dir='.', anns_only=False):
    dataset_dir = os.path.normpath(dataset_dir)
    if dataset_dir != '.':
        for img in coco.dataset['images']:
            file_name = os.path.normpath(img['file_name'])
            prefix = os.path.commonprefix([dataset_dir, file_name])
            if prefix:
                img['file_name'] = file_name[len(prefix):]
            else:
                img['file_name'] = os.path.join(dataset_dir, file_name)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as outp:
        json.dump(coco.dataset['annotations'] if anns_only else coco.dataset,
                  outp, default=json_safe, indent=2)

############################################################
#  main
############################################################


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run Mask R-CNN for address region segmentation.')
    parser.add_argument('--model', required=False, default='last', metavar="PATH/TO/WEIGHTS.h5",
                        help="Path to weights .h5 file or 'imagenet'/'last' to load")
    parser.add_argument('--imgs-per-gpu', type=int, default=0, metavar="NUM",
                        help="Number of images to fit into one batch (depends on GPU memory size; 0 means %d|%d during training|inference)" % (CocoConfig.IMAGES_PER_GPU, InferenceConfig.IMAGES_PER_GPU))
    parser.add_argument('--logs', required=False, default="logs", metavar="PATH/TO/LOGS/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False, type=int, default=0, metavar="NUM",
                        help='Maximum number of images to use (default=all)')
    parser.add_argument('--cwd', action='store_true',
                        help='Interpret all "file_name" paths in COCO relative to the current working directory (instead of the directory of the COCO file). Use with care (and consider chdir), or image paths may not resolve now or next time.')
    subparsers = parser.add_subparsers(dest='command')
    train_parser = subparsers.add_parser('train', help="Train a model from images with COCO annotations")
    train_parser.add_argument('--dataset', required=True, metavar="PATH/TO/COCO.json", nargs='+',
                              help='File path(s) of the address annotations ground truth dataset(s) to be read (randomly split into training and validation)')
    train_parser.add_argument('--split', required=False, type=float, default=0.7, metavar="NUM",
                              help='ratio of train-set in random train/test split (default=0.7 equals 70%%)')
    train_parser.add_argument('--seed', required=False, type=int, default=42, metavar="NUM",
                              help='seed value for random train/test split')
    train_parser.add_argument('--exclude', required=False, default=None, metavar="<LAYER-LIST>",
                              help="Layer names to exclude when loading weights (comma-separated, or 'heads')")
    train_parser.add_argument('--depth', required=False, default=None, metavar="DEPTH-SPEC",
                              help='Layer depth to train on (heads, 3+, ..., all; default: multi-staged)')
    train_parser.add_argument('--epochs', required=False, type=int, default=100, metavar="NUM",
                              help='Number of iterations to train (unless multi-staged)')
    train_parser.add_argument('--rate', required=False, type=float, default=1e-3, metavar="NUM",
                              help='Base learning rate during training')
    evaluate_parser = subparsers.add_parser('evaluate', help="Evaluate a model on images with COCO annotations")
    evaluate_parser.add_argument('--dataset', required=True, metavar="PATH/TO/COCO.json", nargs='+',
                                 help='File path(s) of the address annotations ground truth dataset(s) to be read (randomly split into skip and evaluation)')
    evaluate_parser.add_argument('--split', required=False, type=float, default=0.7, metavar="NUM",
                                 help='ratio of (ignored) train-set in random train/test split (default=0.7 equals 70%%)')
    evaluate_parser.add_argument('--seed', required=False, type=int, default=42, metavar="NUM",
                                 help='seed value for random train/test split')
    evaluate_parser.add_argument('--plot', required=False, default=None, metavar="SUFFIX",
                                 help='Create plot files from prediction under *.SUFFIX.png')
    predict_parser = subparsers.add_parser('predict', help="Apply a model on images with COCO annotations, creating new annotations")
    predict_parser.add_argument('--dataset', required=True, metavar="PATH/TO/COCO.json",
                                help='File path of the address annotations ground truth dataset to be read')
    predict_parser.add_argument('--dataset-pred', required=True, metavar="PATH/TO/COCO.json",
                                help='File path of the address annotations prediction dataset to be written')
    predict_parser.add_argument('--plot', required=False, default=None, metavar="SUFFIX",
                                help='Create plot files from prediction under *.SUFFIX.png')
    test_parser = subparsers.add_parser('test', help="Apply a model on image files without COCO, creating new annotations")
    test_parser.add_argument('--dataset-pred', required=True, metavar="PATH/TO/COCO.json",
                             help='File path of the address annotations prediction dataset to be written')
    test_parser.add_argument('--plot', required=False, default=None, metavar="SUFFIX",
                             help='Create plot files from prediction under *.SUFFIX.png')
    test_parser.add_argument('files', nargs='+',
                             help='File path(s) of image file(s) to be read and annotated')
    merge_parser = subparsers.add_parser('merge', help="Join and sort COCO annotations from multiple files")
    merge_parser.add_argument('--dataset', required=True, metavar="PATH/TO/COCO.json", nargs='+',
                              help='File path(s) of the address annotations dataset(s) to be read (randomly split into skip and prediction)')
    merge_parser.add_argument('--split', required=False, type=float, default=0.7, metavar="NUM",
                              help='ratio of (ignored) train-set in random train/test split (default=0.7 equals 70%%)')
    merge_parser.add_argument('--seed', required=False, type=int, default=42, metavar="NUM",
                              help='seed value for random train/test split')
    merge_parser.add_argument('--dataset-merged', required=True, metavar="PATH/TO/COCO.json",
                              help='File path of the address annotations dataset to be written')
    merge_parser.add_argument('--replace-names', required=False, metavar="PATH/TO/PATHMAP.json",
                              help='File path of a JSON object mapping existing file_name paths to new ones')
    merge_parser.add_argument('--sort', action='store_true',
                              help='Sort images of result dataset by image path names')
    merge_parser.add_argument('--rle', action='store_true',
                              help='Convert segmentations of result dataset to RLE format (suitable for COCO explorer etc)')
    merge_parser.add_argument('--anns-only', action='store_true',
                              help='Keep only annotations in result dataset (suitable for COCO explorer etc)')
    compare_parser = subparsers.add_parser('compare', help="Evaluate COCO annotations from prediction vs ground truth")
    compare_parser.add_argument('--dataset-pred', required=True, metavar="PATH/TO/COCO.json",
                                help='File path of the address annotations prediction dataset to be read')
    compare_parser.add_argument('--dataset', required=True, metavar="PATH/TO/COCO.json",
                                help='File path of the address annotations ground truth dataset to be read')
    args = parser.parse_args()
    args.logs = os.path.abspath(args.logs)
    print("Command: ", args.command)
    if args.command not in ['merge', 'compare']:
        print("Model: ", args.model)
        print("Logs: ", args.logs)
    if args.command == 'train':
        print("Exclude: ", args.exclude)
        print("Depth: ", args.depth)
        if args.depth == 'all':
            print("Epochs: ", args.epochs)
        print("Rate: ", args.rate)
    if args.command in ['evaluate', 'predict', 'test']:
        print("Plot: ", args.plot)
    if args.command == 'test':
        print("Files: ", len(args.files))
    else:
        print("Dataset: ", args.dataset)
    if args.command in ['evaluate', 'train', 'merge']:
        print("Split: ", args.split)
    if args.command in ['predict', 'test', 'compare']:
        print("Prediction dataset: ", args.dataset_pred)
    elif args.command == 'merge':
        print("Merged dataset: ", args.dataset_merged)
    print("Limit: ", args.limit)

    # Configuration and model
    if args.command in ['merge', 'compare']:
        config = None
        model = None
    elif args.command == "train":
        config = CocoConfig()
        config.LEARNING_RATE = args.rate
        if args.imgs_per_gpu:
            config.IMAGES_PER_GPU = args.imgs_per_gpu
            config.BATCH_SIZE = config.IMAGES_PER_GPU * config.GPU_COUNT
        config.display()
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        config = InferenceConfig()
        if args.imgs_per_gpu:
            config.IMAGES_PER_GPU = args.imgs_per_gpu
            config.BATCH_SIZE = config.IMAGES_PER_GPU * config.GPU_COUNT
        config.display()
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if not model:
        model_path = None
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    if args.command == 'train':
        if args.exclude in ['final', 'heads']:
            exclude = ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
        elif args.exclude:
            exclude = args.exclude.split(',')
        else:
            exclude = list()
        if 'mask_rcnn_coco.h5' in model_path or args.model.lower() == 'imagenet':
            # shape of first Conv layer changed with 4-channel input,
            # so exclude from pre-trained weights
            exclude.append("conv1")
    else:
        exclude = list()
    if model and model_path:
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True, exclude=exclude)

    # Train or evaluate
    if args.command in ['train', 'evaluate']:
        dataset_train = CocoDataset()
        dataset_val = CocoDataset()
        for dataset in args.dataset:
            coco = COCO(dataset)
            np.random.seed(args.seed)
            limit = args.limit
            if not limit or limit > len(coco.imgs):
                limit = len(coco.imgs)
            indexes = np.random.permutation(limit)
            trainset = indexes[:int(args.split*limit)]
            valset = indexes[int(args.split*limit):]
            if args.command == "train":
                dataset_train.load_coco(coco,
                                        dataset_dir='.' if args.cwd
                                        else os.path.dirname(dataset),
                                        limit=trainset)
            dataset_val.load_coco(coco,
                                  dataset_dir='.' if args.cwd
                                  else os.path.dirname(dataset),
                                  limit=valset)
            del coco

        if args.command == "train":
            dataset_train.prepare()
            dataset_val.prepare()
            print("Running COCO training on {} train / {} val images.".format(
                dataset_train.num_images, dataset_val.num_images))
            #augmentation = None
            #augmentation = SegmapDropout(0.2)
            augmentation = imgaug.augmenters.Sequential([
                SegmapDropout(0.3),
                SegmapDropoutLines(0.3)])
            # augmentation = imgaug.augmenters.Sequential([
            #     SaveDebugImage('before-augmentation'),
            #     SegmapDropout(0.3),
            #     SegmapDropoutLines(0.3),
            #     SegmapBlackoutLines(0.1),
            #     SaveDebugImage('after-augmentation')])
            print("Augmenting with: {}".format(augmentation))

            # from MaskRCNN.train:
            def layers(depth, add='conv1'):
                layer_regex = {
                    # all layers but the backbone
                    "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
                    # From a specific Resnet stage and up
                    "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
                    "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
                    "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
                    # All layers
                    "all": ".*",
                }
                if depth in layer_regex.keys():
                    depth = layer_regex[depth]
                if add:
                    depth += '|' + add
                return depth
            if args.depth:
                print("Training %s" % args.depth)
                model.train(dataset_train, dataset_val,
                            learning_rate=config.LEARNING_RATE,
                            epochs=args.epochs,
                            layers=layers(args.depth),
                            augmentation=augmentation)
            else:
                # Training - Stage 1
                print("Training network heads")
                model.train(dataset_train, dataset_val,
                            learning_rate=config.LEARNING_RATE,
                            epochs=40,
                            layers=layers('heads'),
                            augmentation=augmentation)

                # Training - Stage 2
                # Finetune layers from ResNet stage 4 and up
                print("Fine tune Resnet stage 4 and up")
                model.train(dataset_train, dataset_val,
                            learning_rate=config.LEARNING_RATE,
                            epochs=120,
                            layers=layers('4+'),
                            augmentation=augmentation)

                # Training - Stage 3
                # Fine tune all layers
                print("Fine tune all layers")
                model.train(dataset_train, dataset_val,
                            learning_rate=config.LEARNING_RATE / 10,
                            epochs=160,
                            layers='all',
                            augmentation=augmentation)

                model_path = os.path.basename(os.getcwd()) + str(os.getpid()) + '.h5'
                print("Saving weights ", model_path)
                model.keras_model.save_weights(model_path, overwrite=True)
        else:
            dataset_val.prepare()
            print("Running COCO prediction on {} images.".format(dataset_val.num_images))
            coco = COCO()
            coco.dataset = dataset_val.dump_coco()
            coco.createIndex()
            results, _ = detect_coco(model, dataset_val, plot=args.plot)
            # Load results. This modifies results with additional attributes.
            if results:
                coco_results = coco.loadRes(results)
            else:
                coco_results = COCO()
                coco_results.dataset = coco.dataset.copy()
                coco_results.dataset['annotations'] = []
                coco_results.createIndex()
            # compare
            evaluate_coco(coco, coco_results)
    
    elif args.command == "predict":
        dataset = CocoDataset()
        coco = COCO(args.dataset)
        dataset.load_coco(coco,
                          dataset_dir='.' if args.cwd
                          else os.path.dirname(args.dataset),
                          limit=args.limit or None,
                          return_coco=True)
        dataset.prepare()
        print("Running COCO prediction on {} images.".format(dataset.num_images))
        results, _ = detect_coco(model, dataset, plot=args.plot)
        # Load results. This modifies results with additional attributes.
        if results:
            coco_results = coco.loadRes(results)
        else:
            coco_results = coco
            coco_results.dataset['annotations'] = []
            coco_results.createIndex()
        store_coco(coco_results, args.dataset_pred,
                   dataset_dir='.' if args.cwd
                   else os.path.dirname(args.dataset_pred))
    
    elif args.command == "test":
        # Test dataset (read images from args.files)
        dataset = CocoDataset()
        dataset.load_files(args.files, limit=args.limit or None)
        dataset.prepare()
        print("Running COCO prediction on {} images.".format(dataset.num_images))
        results, _ = detect_coco(model, dataset, verbose=True, plot=args.plot)
        coco = COCO()
        coco.dataset = dataset.dump_coco()
        coco.createIndex()
        if results:
            coco = coco.loadRes(results)
        store_coco(coco, args.dataset_pred,
                   dataset_dir='.' if args.cwd
                   else os.path.dirname(args.dataset_pred))

    elif args.command == "merge":
        dataset_merged = CocoDataset()
        for dataset in args.dataset:
            coco = COCO(dataset)
            np.random.seed(args.seed)
            limit = args.limit
            if not limit or limit > len(coco.imgs):
                limit = len(coco.imgs)
            indexes = np.random.permutation(limit)
            valset = indexes[int(args.split*limit):]
            # keep original order
            valset = np.sort(valset)
            dataset_merged.load_coco(coco,
                                     dataset_dir='.' if args.cwd
                                     else os.path.dirname(dataset),
                                     limit=valset)
            del coco
        dataset_merged.prepare()
        coco = COCO()
        coco.dataset = dataset_merged.dump_coco(os.path.dirname(args.dataset_merged))
        coco.createIndex()
        if args.replace_names:
            with open(args.replace_names, 'r') as replace_names:
                pathmap = json.load(replace_names)
                for img in coco.imgs.values():
                    if img['file_name'] in pathmap:
                        print('replacing file_name "%s" by "%s"' % (
                            img['file_name'], pathmap[img['file_name']]))
                        img['file_name'] = pathmap[img['file_name']]
        if args.rle:
            for ann in coco.anns.values():
                ann['segmentation'] = coco.annToRLE(ann)
        if args.sort:
            coco = sort_coco(coco)
        store_coco(coco, args.dataset_merged,
                   dataset_dir='.' if args.cwd
                   else os.path.dirname(args.dataset_merged),
                   anns_only=args.anns_only)

    elif args.command == "compare":
        coco = COCO(args.dataset_pred)
        coco_gt = COCO(args.dataset)
        # convert annotations from polygons to compressed format
        for ann in coco.anns.values():
            ann['segmentation'] = coco.annToRLE(ann)
        for ann in coco_gt.anns.values():
            ann['segmentation'] = coco_gt.annToRLE(ann)
        print("prediction dataset: %d imgs %d cats" % (
            len(coco.getImgIds()), len(coco.getCatIds())))
        print("ground-truth dataset: %d imgs %d cats" % (
            len(coco_gt.getImgIds()), len(coco_gt.getCatIds())))
        image_ids = [img['id'] for img in coco_gt.imgs.values()
                     if img['id'] in coco.imgs
                     and img['file_name'] == coco.imgs[img['id']]['file_name']]
        print("both: %d imgs with common file name" % len(image_ids))
        if image_ids:
            evaluate_coco(coco_gt, coco, image_ids=image_ids)
        else:
            raise SystemExit("Datasets '{}' and '{}' have no common image file names. "
                             "(consider running `merge --sort`)".format(args.dataset, args.dataset_pred))

if __name__ == '__main__':
    main()

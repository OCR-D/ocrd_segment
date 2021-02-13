"""
Mask R-CNN
Configurations and data loading code for formdata segmentation
(textline alpha-masked input PNG images for context, target region output COCO JSON).

Based on coco.py in matterport/MaskRCNN.

------------------------------------------------------------

Usage: import the module, or run from the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 formdata.py --model=/path/to/mask_rcnn_coco.h5 train --dataset=/path/to/coco*.json

    # Train a new model starting from ImageNet weights
    python3 formdata.py --model=imagenet train --dataset=/path/to/coco*.json

    # Continue training a model that you had trained earlier
    python3 formdata.py --model=/path/to/weights.h5 train --dataset=/path/to/coco*.json

    # Continue training the last model you trained
    python3 formdata.py --model=last train --dataset=/path/to/coco*.json

    # Run COCO prediction+evaluation on the last model you trained
    python3 formdata.py --model=last evaluate --dataset=/path/to/coco*.json

    # Run COCO prediction+evaluation on the last model you trained (only first 100 files, writing plot files)
    python3 formdata.py --model=last --limit 100 --plot pred evaluate --dataset=/path/to/coco*.json

    # Run COCO prediction on the last model you trained (creating new COCO along existing COCO for comparison)
    python3 formdata.py --model=last predict --dataset-gt=/path/to/coco.json --dataset=/path/to/coco-pred.json

    # Run COCO evaluation between original and predicted dataset
    python3 formdata.py compare --dataset-gt=/path/to/coco.json --dataset=/path/to/coco-pred.json

    # Run COCO prediction on the last model you trained (creating new COCO for arbitrary files)
    python3 formdata.py --model=last test --source abrechnungszeitraum --dataset=/path/to/files.json /path/to/files*

    # Run COCO prediction on the last model you trained (...also writing plot files)
    python3 formdata.py --model=last test --plot pred --source abrechnungszeitraum --dataset=/path/to/files.json /path/to/files*
"""

import os
import sys
import time
import numpy as np
import skimage.io
import skimage.color
import scipy.ndimage.measurements as measurements
import imgaug

import pathlib
from matplotlib import pyplot, cm
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection
from PIL import Image
import json
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import tarfile
import urllib.request
import shutil

# Import Mask RCNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # i.e. error
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.abspath("logs")

ALPHA_TEXT_CHANNEL = 200
ALPHA_CTXT_CHANNEL = 255

############################################################
#  Configurations
############################################################

FIELDS = [None,
          "abrechnungszeitraum",
          "nutzungszeitraum",
          "gebaeude_heizkosten_raumwaerme",
          "gebaeude_heizkosten_warmwasser",
          "anteil_grundkost_heizen", # "prozent_grundkosten_raumwaerme"
          "anteil_grundkost_warmwasser", # "prozent_grundkosten_warmwasser"
          "energietraeger", # not used
          "energietraeger_verbrauch",
          "energietraeger_einheit",
          "energietraeger_kosten",
          "gebaeude_flaeche",
          "wohnung_flaeche",
          "gebaeude_verbrauchseinheiten",
          "wohnung_verbrauchseinheiten",
          "gebaeude_warmwasser_verbrauch",
          "gebaeude_warmwasser_verbrauch_einheit",
          "kaltwasser_fuer_warmwasser",
          "wohnung_warmwasser_verbrauch",
          "wohnung_warmwasser_verbrauch_einheit",
          "gebaeude_grundkost_heizen", # "gebaeude_grundkosten_raumwaerme",
          "gebaeude_grundkost_warmwasser", # "gebaeude_grundkosten_warmwasser",
          # "gebaeude_heizkosten_gesamt",
          # "anteil_verbrauchskosten_heizen", # "prozent_verbrauchskosten_raumwaerme"
          # "anteil_verbrauchskosten_warmwasser", # "prozent_verbrauchskosten_warmwasser"
          # "gebaeude_verbrauchskosten_raumwaerme",
          # "gebaeude_verbrauchskosten_warmwasser",
          # "wohnung_heizkosten_gesamt",
          # "wohnung_grundkosten_raumwaerme",
          # "wohnung_verbrauchskosten_raumwaerme",
          # "wohnung_grundkosten_warmwasser",
          # "wohnung_verbrauchskosten_warmwasser",
          # "warmwasser_temperatur",
          #
          ## "nebenkosten_betriebsstrom",
          ## "nebenkosten_wartung_heizung",
          ## "nebenkosten_messgeraet_miete",
          ## "nebenkosten_messung_abrechnung",
]

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "formdata"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 21 + 1  # formdata has 21 classes

    # ...settings to reduce GPU memory requirements...
    
    # Use a smaller backbone network. The default is resnet101,
    # but you can use resnet50 to reduce memory load significantly
    # and it's sufficient for most applications. It also trains faster.
    BACKBONE = "resnet50"

    # Reduce the maximum number of instances per image if your images
    # don't have a lot of objects.
    MAX_GT_INSTANCES = 2
    # Max number of final detections
    DETECTION_MAX_INSTANCES = 1
    DETECTION_MIN_CONFIDENCE = 0.5

    # Use fewer ROIs in training the second stage. This setting
    # is like the batch size for the second stage of the model.
    # (includes subsampling of both positive and negative examples)
    TRAIN_ROIS_PER_IMAGE = 10

    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    #PRE_NMS_LIMIT = 6000
    PRE_NMS_LIMIT = 200

    # ROIs kept after non-maximum suppression (training and inference)
    #POST_NMS_ROIS_TRAINING = 2000
    #POST_NMS_ROIS_INFERENCE = 1000
    POST_NMS_ROIS_TRAINING = 100
    POST_NMS_ROIS_INFERENCE = 50

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 600
    IMAGE_MAX_DIM = 768

    # ...settings to accommodate alpha channel input...
    IMAGE_CHANNEL_COUNT = 5
    # don't touch the alpha channel
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 0, 0])

class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

############################################################
#  Image/Segmentation Augmentation
############################################################

class SegmapDropout(imgaug.augmenters.meta.Augmenter):
    """Augment by randomly dropping instances from GT and image alpha.

    Probabilistic augmenter that takes instance masks
    (in the form of segmentation maps with bg as index 0)
    and randomly drops all instances
    by setting its output segmap / mask to bg, and
    by setting its input 5th / context channel to 0 everywhere.
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
            p = self.p.draw_samples((1,), random_state=random_state)
            if np.any(p < 0.5):
                image[:,:,4] = 0 # set cmask=0
                segmap[:,:,:] = 0 # set to bg
            batch.images[i] = image
            batch.segmentation_maps[i].arr = segmap
        return batch
    def get_parameters(self):
        return [self.p]

# FIXME: instead of cmask dropout, here we need cmask multiplication
# (randomly extend cmask to tmask>0 areas that contain the same text)
class SegmapDropoutLines(imgaug.augmenters.meta.Augmenter):
    """Augment by randomly dropping instances' first line from image alpha.

    Probabilistic augmenter that takes instance masks
    (in the form of segmentation maps with bg as index 0),
    and randomly degrades a fraction of instances
    by setting its input 5th / context channel to 0
    in the top lines but keeping its output segmap / mask.
    This is supposed to help become robust against
    title/name part of context lines being undetected.
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
                    image[labels == lines[top],4] = 0 # set cmask=0 of first line
            batch.images[i] = image
        return batch
    def get_parameters(self):
        return [self.p]

class SegmapBlackoutLines(imgaug.augmenters.meta.Augmenter):
    """Augment by randomly drawing black boxes over non-instances'/non-context lines in the image.

    Probabilistic augmenter that takes instance masks
    (in the form of segmentation maps with bg as index 0),
    and randomly degrades a fraction of non-instance, non-context lines
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
            tmask = image[:,:,3] > 0
            labels, nlabels = measurements.label(tmask.astype(np.int32))
            cmask = image[:,:,4] > 0
            p = self.p.draw_samples((nlabels,), random_state=random_state)
            drop = p < 0.5
            drop = np.insert(drop, 0, [False]) # never box out non-text
            for label in drop.nonzero()[0]:
                if not label:
                    continue
                tmask = labels == label
                if np.any(tmask & (cmask | segmap[:,:,0] > 0)):
                    # text line is part of context (5th channel) or target (GT RoI instance)
                    continue
                y, x = np.nonzero(tmask)
                ymin = np.maximum(0, y.min() - self.padding)
                ymax = np.minimum(tmask.shape[0], y.max() + self.padding)
                xmin = np.maximum(0, x.min() - self.padding)
                xmax = np.minimum(tmask.shape[1], x.max() + self.padding)
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
    and draws them on the image, reducing tmask/cmask to alpha
    (by setting coloring output segmap / mask and
     by setting input alpha channel to 255 if cmask else 200 if tmask else 0).
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
            # elif cmask == 0, scale from 0..255 to 200..255 (gray)
            # else keep full contrast
            image = img[:,:,:3]
            tmask = img[:,:,3]
            cmask = img[:,:,4]
            # print("image with %f%% text with %f%% context" % (
            #     100.0 * np.count_nonzero(tmask == 255)/np.prod(tmask.shape),
            #     100.0 * np.count_nonzero(cmask == 255)/np.count_nonzero(tmask == 255)))
            #image[tmask < 255] = 255
            image[cmask < 255] = 200 + 55/255 * image[cmask < 255]
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
        for i, name in enumerate(FIELDS):
            if name:
                # use class name as source so we can train on each class dataset
                # after another while only one class is active at a time
                self.add_class(name, i, name)
        
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

        # use first annotated class as source
        source = coco.loadCats(coco.getCatIds())[0]['name']
        print('using class "%s" as source for all images' % source)
        # Add images
        # we cannot keep the image_id refs, because Dataset can load multiple COCO files
        for i, id_ in enumerate(image_ids, len(self.image_info)):
            file_name = coco.imgs[id_]['file_name']
            ann_ids = coco.getAnnIds(
                imgIds=[id_], catIds=class_ids, iscrowd=None)
            self.add_image(
                source, image_id=id_ if return_coco else i,
                path=os.path.join(dataset_dir, file_name),
                width=coco.imgs[id_]["width"],
                height=coco.imgs[id_]["height"],
                # still contains original/COCO image_id refs
                # and inconsistent/clashing id refs:
                annotations=coco.loadAnns(ann_ids))
        if return_coco:
            return coco
        return None
        
    def load_files(self, filenames, dataset_dir='.', limit=None, source=''):
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
                source, image_id=i, path=filename,
                width=width, height=height)

    def dump_coco(self, dataset_dir='.'):
        """Dump dataset into an COCO JSON file."""
        result = { 'categories': self.class_info, 'images': list(), 'annotations': list() }
        i = 0
        for image_id in self.image_ids:
            image_info = self.image_info[image_id]
            result['images'].append({ 'id': image_info['id'],
                                      'width': image_info['width'],
                                      'height': image_info['height'],
                                      'file_name': pathlib.Path(image_info['path']).relative_to(dataset_dir),
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
        if image_info["source"] not in self.sources:
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
                "{}.{}".format(image_info["source"], annotation['category_id']))
            if class_id is None:
                print('invalid category_id %d in source "%s" for image %d ("%s")' % (
                    annotation['category_id'], image_info["source"], image_id, image_info['path']))
                continue
            m = self.annToMask(annotation, image_info["height"], image_info["width"])
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

def build_coco_results(dataset, image_id, image_source, rois, class_ids, scores, masks):
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
            "category_id": dataset.get_source_class_id(class_id, image_source),
            "iscrowd": 0,
            "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
            "score": score,
            "segmentation": maskUtils.encode(np.asfortranarray(mask))
        }
        results.append(result)
    return results

def test_coco(model, dataset, verbose=False, limit=None, image_ids=None, plot=False):
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

    t_prediction = 0
    t_start = time.time()

    results = []
    cocoids = []
    for image_id in tqdm(image_ids):
        # Load image
        image = dataset.load_image(image_id)
        image_path = dataset.image_info[image_id]['path']
        image_cocoid = dataset.image_info[image_id]['id']
        image_source = dataset.image_info[image_id]['source']
        # Limit to a subset of classes
        active_class_ids = dataset.source_class_ids[image_source]

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0, active_class_ids=active_class_ids)[0]
        t_prediction += (time.time() - t)
        if verbose:
            print("image {} {} has {} rois with {} distinct classes".format(
                image_cocoid, image_path,
                r['masks'].shape[-1], len(np.unique(r['class_ids']))))

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, image_cocoid, image_source,
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        dataset.image_info[image_id].update({'annotations': image_results})
        results.extend(image_results)
        cocoids.append(image_cocoid)
        if plot:
            plot_result(image, image_results,
                        dataset.image_info[image_id]['width'],
                        dataset.image_info[image_id]['height'],
                        pathlib.Path(image_path).with_suffix('.' + plot + '.png'))

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)

    for i, ann in enumerate(results):
        ann['id'] = i
    return results, cocoids

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
    print("Evaluating predictions against GT")
    cocoEval.evaluate()
    def file_name(img):
        image_info = coco_gt.imgs[img['image_id']]
        return str(image_info['file_name'])
    def cat_name(img):
        class_info = coco_gt.cats[img['category_id']]
        return str(class_info['id']) + '_' + class_info['name']
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

def showAnns(anns, height, width):
    """
    Display the specified annotations.
    :param anns (array of object): annotations to display
    :return: None
    """
    if len(anns) == 0:
        return 0
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
    showAnns(anns, width, height)
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

def store_coco(coco, filename):        
    with open(filename, 'w') as outp:
        json.dump(coco.dataset, outp, default=json_safe, indent=2)

############################################################
#  main
############################################################


def main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run Mask R-CNN for formdata region segmentation.')
    parser.add_argument('--model', required=False, default='last', metavar="PATH/TO/WEIGHTS.h5",
                        help="Path to weights .h5 file or 'imagenet'/'last' to load")
    parser.add_argument('--logs', required=False, default=DEFAULT_LOGS_DIR, metavar="PATH/TO/LOGS/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False, type=int, default=0, metavar="NUM",
                        help='Maximum number of images to use (default=all)')
    subparsers = parser.add_subparsers(dest='command')
    train_parser = subparsers.add_parser('train', help="Train a model from images with COCO annotations")
    train_parser.add_argument('--dataset', required=True, metavar="PATH/TO/COCO.json", nargs='+',
                              help='File path of the formdata dataset annotations (randomly split into training and validation)')
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
                                 help='File path of the formdata dataset annotations (randomly split into skip and evaluation)')
    evaluate_parser.add_argument('--split', required=False, type=float, default=0.7, metavar="NUM",
                                 help='ratio of (ignored) train-set in random train/test split (default=0.7 equals 70%%)')
    evaluate_parser.add_argument('--seed', required=False, type=int, default=42, metavar="NUM",
                                 help='seed value for random train/test split')
    evaluate_parser.add_argument('--plot', required=False, default=None, metavar="SUFFIX",
                                 help='Create plot files from prediction under *.SUFFIX.png')
    predict_parser = subparsers.add_parser('predict', help="Apply a model on images with COCO annotations, creating new annotations")
    predict_parser.add_argument('--dataset-gt', required=True, metavar="PATH/TO/COCO.json",
                                help='File path of the formdata dataset annotations (ground truth)')
    predict_parser.add_argument('--dataset', required=True, metavar="PATH/TO/COCO.json",
                                help='File path of the formdata dataset annotations (to be filled)')
    test_parser = subparsers.add_parser('test', help="Apply a model on image files, creating COCO annotations")
    test_parser.add_argument('--source', required=True, metavar="CLASS",
                             help='Name of the unique (active) category all files are marked for')
    test_parser.add_argument('--dataset', required=True, metavar="PATH/TO/COCO.json",
                             help='File path of the formdata dataset annotations (to be filled)')
    test_parser.add_argument('--plot', required=False, default=None, metavar="SUFFIX",
                             help='Create plot files from prediction under *.SUFFIX.png')
    test_parser.add_argument('files', nargs='+',
                             help='Image files to annotate')
    compare_parser = subparsers.add_parser('compare', help="Evaluate COCO annotations w.r.t. GT COCO annotations")
    compare_parser.add_argument('--dataset-gt', required=True, metavar="PATH/TO/COCO.json",
                                help='File path of the formdata dataset annotations (from ground truth)')
    compare_parser.add_argument('--dataset', required=True, metavar="PATH/TO/COCO.json",
                                help='File path of the formdata dataset annotations (from prediction)')
    compare_parser.add_argument('--split', required=False, type=float, default=0.7, metavar="NUM",
                                help='ratio of (ignored) train-set in random train/test split (default=0.7 equals 70%%)')
    compare_parser.add_argument('--seed', required=False, type=int, default=42, metavar="NUM",
                                help='seed value for random train/test split')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Logs: ", args.logs)
    if args.command == 'train':
        print("Exclude: ", args.exclude)
        print("Depth: ", args.depth)
        if args.depth == 'all':
            print("Epochs: ", args.epochs)
        print("Rate: ", args.rate)
    if args.command in ['evaluate', 'test']:
        print("Plot: ", args.plot)
    print("Dataset: ", args.dataset)
    if args.command in ['evaluate', 'train', 'compare']:
        print("Split: ", args.split)
    if args.command == 'test':
        print("Source: ", len(args.source))
        print("Files: ", len(args.files))
    if args.command in ['predict', 'compare']:
        print("GT: ", args.dataset_gt)
    print("Limit: ", args.limit)

    # Configurations
    if args.command == "train":
        config = CocoConfig()
        config.LEARNING_RATE = args.rate
    else:
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.command == 'compare':
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
    if model_path:
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
                dataset_train.load_coco(coco, #os.path.dirname(args.dataset),
                                        limit=trainset)
            dataset_val.load_coco(coco, #os.path.dirname(args.dataset),
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
                SegmapBlackoutLines(0.1)])
            # augmentation = imgaug.augmenters.Sequential([
            #     SaveDebugImage('before-augmentation'),
            #     SegmapDropout(0.3),
            #     SegmapBlackoutLines(0.1),
            #     SaveDebugImage('after-augmentation')])

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
            print("Running COCO evaluation on {} images.".format(dataset_val.num_images))
            coco = COCO()
            coco.dataset = dataset_val.dump_coco()
            coco.createIndex()
            results, _ = test_coco(model, dataset_val, limit=limit, plot=args.plot)
            # Load results. This modifies results with additional attributes.
            if results:
                coco_results = coco.loadRes(results)
            else:
                coco_results = COCO()
                coco_results.dataset = coco.dataset.copy()
                coco_results.dataset['annotations'] = []
                coco_results.createIndex()
            # Evaluate
            evaluate_coco(coco, coco_results)
    
    elif args.command == "predict":
        dataset = CocoDataset()
        coco_gt = dataset.load_coco(args.dataset_gt, #os.path.dirname(args.dataset),
                                    limit=args.limit or None,
                                    return_coco=True)
        dataset.prepare()
        source = coco_gt.loadCats(coco_gt.getCatIds())[0]['name']
        print("Running COCO prediction for class {} on {} images.".format(source, dataset.num_images))
        results, _ = test_coco(model, dataset)
        if results:
            coco = coco_gt.loadRes(results)
        else:
            coco = coco_gt
            coco.dataset['annotations'] = []
            coco.createIndex()
        store_coco(coco, args.dataset)
        
    elif args.command == "test":
        # Test dataset (read images from args.files)
        dataset = CocoDataset()
        dataset.load_files(args.files, limit=args.limit or None, source=args.source)
        dataset.prepare()
        print("Running COCO test for class {} on {} images.".format(args.source, dataset.num_images))
        results, _ = test_coco(model, dataset, verbose=True, plot=args.plot)
        coco = COCO()
        coco.dataset = dataset.dump_coco() #os.path.dirname(args.dataset)
        coco.createIndex()
        if results:
            coco = coco.loadRes(results)
        store_coco(coco, args.dataset)
    
    elif args.command == "compare":
        dataset = CocoDataset()
        dataset_gt = CocoDataset()
        coco = COCO(args.dataset)
        coco_gt = COCO(args.dataset_gt)
        np.random.seed(args.seed)
        limit = args.limit
        if not limit or limit > len(coco_gt.imgs):
            limit = len(coco_gt.imgs)
        indexes = np.random.permutation(limit)
        trainset = indexes[:int(args.split*limit)]
        valset = indexes[int(args.split*limit):]
        image_ids = coco_gt.getImgIds()
        image_ids = np.array(image_ids).take(valset)
        # convert annotations from polygons to compressed format
        for ann in coco.anns.values():
            ann['segmentation'] = coco.annToRLE(ann)
        for ann in coco_gt.anns.values():
            ann['segmentation'] = coco.annToRLE(ann)
        print("prediction dataset: %d imgs %d cats" % (
            len(image_ids), len(set([ann['category_id']
                                     for ann in coco.loadAnns(coco.getAnnIds(imgIds=image_ids))]))))
        print("ground-truth dataset: %d imgs %d cats" % (
            len(image_ids), len(set([ann['category_id']
                                     for ann in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_ids))]))))
        evaluate_coco(coco_gt, coco, image_ids=image_ids)
    
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))

if __name__ == '__main__':
    main()
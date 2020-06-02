"""
Mask R-CNN
Configurations and data loading code for address resegmentation
(textline alpha-masked input PNG images, address region output COCO JSON).

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module, or run from the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 address.py --model=/path/to/mask_rcnn_coco.h5 train --dataset=/path/to/coco.json

    # Train a new model starting from ImageNet weights. Also auto download PubLayNet dataset
    python3 address.py --model=imagenet --download=True train --dataset=/path/to/coco.json

    # Continue training a model that you had trained earlier
    python3 address.py --model=/path/to/weights.h5 train --dataset=/path/to/coco.json

    # Continue training the last model you trained
    python3 address.py --model=last train --dataset=/path/to/coco.json

    # Run COCO evaluation on the last model you trained
    python3 address.py --model=last evaluate --dataset=/path/to/coco.json

    # Run COCO test on the last model you trained (only first 100 files)
    python3 address.py --model=last --limit 100 test --dataset=/path/to/coco.json

    # Run COCO prediction on the last model you trained
    python3 address.py --model=last predict --json /path/to/coco.json /path/to/files*
"""

import os
import sys
import time
import numpy as np
import skimage.io
import skimage.color
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

import pathlib
from matplotlib import pyplot, cm
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection
from PIL import Image
import json
from tqdm import tqdm

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
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

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

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
    DETECTION_MIN_CONFIDENCE = 0.7

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
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_json, dataset_dir='.',
                  limit=None, skip_empty=False,
                  class_ids=None,
                  class_map=None, return_coco=False):
        """Load a subset of the COCO dataset from a JSON file.
        dataset_json: JSON file path of the COCO (sub-) dataset.
        dataset_dir: parent directory of relative filenames in JSON
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
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
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids or sorted(coco.getCatIds()):
            self.add_class("IAO", i, coco.loadCats(i)[0]["name"])

        # Limit to a subset
        if isinstance(limit, int):
            image_ids = image_ids[:limit]
        elif isinstance(limit, (list, np.ndarray)):
            image_ids = np.array(image_ids).take(limit)
        
        # Add images
        for i in image_ids:
            file_name = coco.imgs[i]['file_name']
            ann_ids = coco.getAnnIds(
                imgIds=[i], catIds=class_ids, iscrowd=None)
            if skip_empty and not ann_ids:
                print('skipping image without annotations: %d ("%s")' % (
                    i, file_name))
                continue
            self.add_image(
                "IAO", image_id=i,
                path=os.path.join(dataset_dir, file_name),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(ann_ids))
        if return_coco:
            return coco
        
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

    def dump_coco(self, dataset_dir='.'):
        """Dump dataset into an COCO JSON file."""
        result = { 'categories': self.class_info, 'images': list(), 'annotations': list() }
        for image_id in self.image_ids:
            image_info = self.image_info[image_id]
            result['images'].append({ 'id': image_info['id'],
                                      'width': image_info['width'],
                                      'height': image_info['height'],
                                      'file_name': pathlib.Path(image_info['path']).relative_to(dataset_dir),
                                      })
            if 'annotations' in image_info:
                result['annotations'].extend(image_info['annotations'])
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
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            print('no instance in image %d ("%s")' % (
                image_id, image_info['path']))
            return super(CocoDataset, self).load_mask(image_id)

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
        # Convert from RGBA to RGB+Text+Address
        tmask = image[:,:,3:4] > 0
        amask = image[:,:,3:4] == 255
        image = np.concatenate([image[:,:,:3],
                                255 * tmask.astype(np.uint8),
                                255 * amask.astype(np.uint8)],
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

    def showAnns(self, anns, height, width):
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
            if type(ann['segmentation']) == list:
                # polygon
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((int(len(seg)/2), 2))
                    polygons.append(Polygon(poly))
                    colors.append(color)
            else:
                # mask
                if type(ann['segmentation']['counts']) == list:
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


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=None, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with validation data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if isinstance(limit, int):
        image_ids = image_ids[:limit]
    elif isinstance(limit, (list, np.ndarray)):
        image_ids = np.array(image_ids).take(limit)

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[image_id]["id"] for image_id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(tqdm(image_ids)):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        dataset.image_info[image_id].update({'annotations': image_results})
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)

def test_coco(model, dataset, coco, limit=0, image_ids=None, plot=False):
    """Predict images
    dataset: A Dataset object with test data
    limit: if not 0, it's the number of images to use for test
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if not limit is None:
        image_ids = image_ids[:limit]

    t_prediction = 0
    t_start = time.time()

    results = []
    for image_id in tqdm(image_ids):
        # Load image
        image = dataset.load_image(image_id)
        image_path = dataset.image_info[image_id]['path']
        image_cocoid = dataset.image_info[image_id]['id']

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)
        print("image %s %s has %d rois with %d distinct classes" % \
              (image_cocoid, image_path,
               r['masks'].shape[-1], len(np.unique(r['class_ids']))))

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, image_cocoid,
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        dataset.image_info[image_id].update({'annotations': image_results})
        results.extend(image_results)
        if plot:
            # show result
            pyplot.close() # clear axes from previous images
            fig = pyplot.figure(frameon=False)
            pyplot.imshow(image[:,:,:3])
            ax = pyplot.gca()
            ax.set_axis_off()
            ax.set_xmargin(0)
            ax.set_ymargin(0)
            ax.set_frame_on(0)
            ax.set_position([0,0,1,1])
            #ax.title(image_cocoid)
            width = dataset.image_info[image_id]['width']
            height = dataset.image_info[image_id]['height']
            dataset.showAnns(image_results, width, height)
            filename = pathlib.Path(image_path).with_suffix(".pred.png")
            # make an extra effort to arrive at the same image size
            # (no frames, axes, margins):
            fig.set_size_inches((width/300, height/300))
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            pyplot.savefig(filename, dpi=300, pad_inches=0, bbox_inches=extent)
            
    # Load results. This modifies results with additional attributes.
    if coco:
        coco = coco.loadRes(results)

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)

    return coco

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
        json.dump(coco.dataset, outp, default=json_safe)

############################################################
#  main
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run Mask R-CNN for address region segmentation.')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'imagenet'/'last' to load")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        type=int, default=0,
                        metavar="<image count>",
                        help='Maximum number of images to use (default=all)')
    subparsers = parser.add_subparsers(dest='command')
    train_parser = subparsers.add_parser('train', help="Train a model from images with COCO annotations")
    train_parser.add_argument('--dataset', required=True,
                              metavar="/path/to/coco.json",
                              help='Directory of the address dataset')
    train_parser.add_argument('--exclude', required=False,
                              default=None,
                              metavar="<layer-list>",
                              help="Layer names to exclude when loading weights (comma-separated, or 'heads')")
    train_parser.add_argument('--depth', required=False,
                              default=None,
                              metavar="depth-spec",
                              help='Layer depth to train on (heads, 3+, ..., all; default: multi-staged)'),
    evaluate_parser = subparsers.add_parser('evaluate', help="Evaluate a model on images with COCO annotations")
    evaluate_parser.add_argument('--dataset', required=True,
                                 metavar="/path/to/coco.json",
                                 help='Directory of the address dataset')
    test_parser = subparsers.add_parser('test', help="Apply a model on images, adding COCO annotations")
    test_parser.add_argument('--dataset', required=True,
                             metavar="/path/to/coco.json",
                             help='Directory of the address dataset')
    predict_parser = subparsers.add_parser('predict', help='Apply a model on image files, creating plots')
    predict_parser.add_argument('dataset', nargs='+',
                                help='Image files to predict')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Logs: ", args.logs)
    if args.command == 'train':
        print("Exclude: ", args.exclude)
        print("Depth: ", args.depth)
    print("Dataset: ", args.dataset)
    print("Limit: ", args.limit)

    # Configurations
    if args.command == "train":
        config = CocoConfig()
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
    if args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
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
    model.load_weights(model_path, by_name=True, exclude=exclude)

    # Train or evaluate
    if args.command in ['train', 'evaluate']:
        coco = COCO(args.dataset)
        np.random.seed(42)
        if not args.limit or args.limit > len(coco.imgs):
            args.limit = len(coco.imgs)
        indexes = np.random.permutation(args.limit)
        trainset = indexes[:int(0.7*args.limit)]
        valset = indexes[int(0.7*args.limit):]
        
        if args.command == "train":
            dataset_train = CocoDataset()
            dataset_train.load_coco(coco, os.path.dirname(args.dataset),
                                    limit=trainset, skip_empty=True) #False
            dataset_val = CocoDataset()
            dataset_val.load_coco(coco, os.path.dirname(args.dataset),
                                  limit=valset, skip_empty=True) #True
            del coco
            dataset_train.prepare()
            dataset_val.prepare()
            print("Running COCO training on {} train / {} val images.".format(
                dataset_train.num_images, dataset_val.num_images))
            # Image Augmentation
            # Right/Left flip 50% of the time
            #augmentation = imgaug.augmenters.Fliplr(0.5)
            augmentation = None

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
                            epochs=100,
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
                
        else:
            dataset = CocoDataset()
            coco_res = dataset.load_coco(coco, os.path.dirname(args.dataset), limit=valset,
                                         return_coco=True)
            del coco
            dataset.prepare()
            print("Running COCO evaluation on {} images.".format(dataset.num_images))
            evaluate_coco(model, dataset, coco_res, "bbox")
            #print(model.evaluate(dataset))
        
    elif args.command == "test":
        # Test dataset (read images from "test/" and categories from "val.json")
        image_path = os.path.join(args.dataset, "test")
        json_path = os.path.join(args.dataset, "val.json")
        dataset = CocoDataset()
        dataset.load_coco(json_path, limit=0, auto_download=args.download)
        dataset.load_files(os.listdir(image_path), image_path, limit=args.limit or None)
        dataset.prepare()
        print("Running COCO test on {} images.".format(dataset.num_images))
        coco = COCO()
        coco.dataset = dataset.dump_coco(image_path)
        coco.createIndex()
        coco = test_coco(model, dataset, coco, limit=args.limit or None)
        json_path = os.path.join(args.dataset, "test.json")
        store_coco(coco, json_path)
        
    elif args.command == "predict":
        # Other files (read images from CLI and categories from "val.json")
        dataset = CocoDataset()
        dataset.add_class("IAO", 0, "address-rcpt")
        dataset.add_class("IAO", 1, "address-sndr")
        dataset.add_class("IAO", 2, "address-contact")
        dataset.load_files(args.dataset, limit=args.limit or None)
        dataset.prepare()
        print("Running COCO predict on {} images.".format(dataset.num_images))
        test_coco(model, dataset, None, limit=args.limit or None, plot=True)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))

    

#! python
# -*- coding: utf-8 -*-

import pathlib
import json
import random
import click
from matplotlib import pyplot, cm
from matplotlib.patches import Rectangle

import numpy as np

from .layout_dataset import LayoutDataset, LayoutTrainConfig, LayoutPredictConfig
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn.model import MaskRCNN
from keras.callbacks import EarlyStopping

@click.group()
def cli():
    pass

@cli.command()
@click.argument('data', type=click.Path(exists=True, file_okay=False), required=True)
@click.option('-w', '--weights', type=click.Path(exists=True, dir_okay=False), required=True,
              help='initial weight/checkpoint file')
@click.option('-W', '--write-weights', type=click.Path(exists=False, dir_okay=False),
              help='save final weight file')
@click.option('-h', '--heads', is_flag=True, default=False,
              help='only train head layers')
@click.option('-d', '--depth', default=None,
              help='start training at what depth of layers (heads, 2+, ..., all)')
@click.option('-s', '--seed', type=int, default=0)
def train(data, weights, write_weights, heads, depth, seed):
    """Train model on DATA directory, loading from WEIGHTS."""
    # check parameters
    if heads:
        if depth and depth != 'heads':
            raise Exception("Inconsistency between options --heads and --depth")
        depth = 'heads'
    if not depth:
        depth = 'all'

    #
    # prepare data
    #

    # two data sets
    train_data = LayoutDataset()
    test_data = LayoutDataset()

    # set random seed
    random.seed(seed)

    # read GT
    gt = list(pathlib.Path(data).glob('*.json'))
    tr = set(random.sample(gt, k=len(gt)*80//100))
    for json_path in gt:
        if json_path in tr:
            dataset = train_data
        else:
            dataset = test_data
        json_path = pathlib.Path(json_path).resolve()
        annotation = None
        with json_path.open('r') as json_file:
            annotation = json.load(json_file)
        if not annotation.get('regions'):
            continue
        image_id = json_path.stem
        image_path = str(json_path.parent.joinpath(image_id + '.png')) # .bin.png, .dbg.png
        if pathlib.Path(image_path).exists():
            dataset.add_image('dataset', image_id=image_id, path=image_path, annotation=annotation)

    # compile data
    train_data.prepare()
    click.echo("Train %d" % len(train_data.image_ids), err=True)
    test_data.prepare()
    click.echo("Test %d" % len(test_data.image_ids), err=True)

    # load configuration
    config = LayoutTrainConfig()
    config.display()

    #
    # train model
    model = MaskRCNN(mode='training', model_dir='./', config=config)
    model.keras_model.metrics_tensors = []

    # load weights (mscoco) and exclude the output layers
    if weights:
        if 'coco' in weights.lower():
            # train head layers from scratch
            # this only makes sense for the very first transfer (from COCO etc):
            click.echo("Not loading final layers", err=True)
            exclude = ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
        else:
            exclude = None
        model.load_weights(weights, by_name=True, exclude=exclude)
    model.train(train_data, test_data,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                custom_callbacks=[EarlyStopping(patience=10, verbose=1, restore_best_weights=True)]
                if write_weights else None,
                layers=depth)
    if write_weights:
        model.keras_model.save_weights(write_weights)

@cli.command()
@click.argument('data', type=click.Path(exists=True, file_okay=False), required=True)
@click.option('-w', '--weights', type=click.Path(exists=True, dir_okay=False), required=True,
              help='weight/checkpoint file')
@click.option('-s', '--seed', type=int, default=0)
def evaluate(data, weights, seed):
    """Evaluate model on DATA directory, loading from WEIGHTS."""
    #
    # prepare data
    #

    # one data set
    test_data = LayoutDataset()

    # set random seed
    random.seed(seed)

    # read GT
    gt = list(pathlib.Path(data).glob('*.json'))
    tr = set(random.sample(gt, k=len(gt)*80//100))
    for json_path in gt:
        if json_path in tr:
            continue
        json_path = pathlib.Path(json_path) #.resolve()
        annotation = None
        with json_path.open('r') as json_file:
            annotation = json.load(json_file)
        if not annotation.get('regions'):
            continue
        image_id = json_path.stem
        image_path = str(json_path.parent.joinpath(image_id + '.png')) # .bin.png, .dbg.png
        if pathlib.Path(image_path).exists():
            test_data.add_image('dataset', image_id=image_id, path=image_path, annotation=annotation)

    # compile data
    test_data.prepare()
    click.echo("Test %d" % len(test_data.image_ids), err=True)

    #image_index = random.randint(0,len(gt))
    #image_id = pathlib.Path(gt[image_index]).stem
    #image = train_data.load_image(image_index)
    # load image mask
    #mask, class_ids = train_data.load_mask(image_index)
    # extract bounding boxes from the masks
    #bbox = extract_bboxes(mask)
    # display image with masks and bounding boxes
    #display_instances(image, bbox, mask, class_ids, train_data.class_names)

    # load configuration
    config = LayoutPredictConfig()
    config.display()

    #
    # evaluate model
    model = MaskRCNN(mode='inference', model_dir='./', config=config)
    #model.keras_model.metrics_tensors = []

    # load pretrained weights
    model.load_weights(weights, by_name=True)

    # evaluate
    aps = []
    for image_id in test_data.image_ids:
        image_path = pathlib.Path(test_data.image_reference(image_id)).stem
        click.echo("predicting image: %s" % image_path)
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(test_data, config, image_id, use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, config)
        # convert image into one sample
        sample = np.expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        ap, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        if not np.isnan(ap):
            aps.append(ap)
        click.echo("AP: %.3f image: %s" % (ap, image_path))
    click.echo("Test mAP: %.3f" % np.mean(aps))

@cli.command()
@click.argument('images', type=click.Path(exists=True, dir_okay=False), required=True, nargs=-1)
@click.option('-w', '--weights', type=click.Path(exists=True, dir_okay=False), required=True,
              help='weight/checkpoint file')
def predict(images, weights):
    """Predict model on IMAGES files, loading from WEIGHTS."""
    #
    # prepare data
    #

    # one data set
    predict_data = LayoutDataset()

    # read images
    for image in images:
        image_path = pathlib.Path(image) #.resolve()
        image_id = image_path.stem
        predict_data.add_image('dataset', image_id=image_id, path=image_path)

    # compile data
    predict_data.prepare()
    click.echo("Predict %d" % len(predict_data.image_ids), err=True)

    # load configuration
    config = LayoutPredictConfig()
    config.display()

    #
    # evaluate model
    model = MaskRCNN(mode='inference', model_dir='./', config=config)

    # load pretrained weights
    model.load_weights(weights, by_name=True)

    # evaluate
    aps = []
    for image_id in predict_data.image_ids:
        image_path = pathlib.Path(predict_data.image_reference(image_id)).stem
        click.echo("predicting image: %s" % image_path)
        # load image
        image = predict_data.load_image(image_id)
        # convert pixel values (e.g. center)
        # FIXME robert: why mold here?? 
        # (this step is already included in model.detect()'s mold_inputs)
        scaled_image = mold_image(image, config)
        # convert image into one sample
        sample = np.expand_dims(scaled_image, 0)
        # (also resizes according to config)
        # make prediction
        yhat = model.detect(sample, verbose=0)[0]
        masks = yhat['masks']
        click.echo("image has %d rois with %d distinct classes" % \
                   (masks.shape[-1], len(np.unique(yhat['class_ids']))))
        # show result
        pyplot.close() # clear axes from previous images
        pyplot.imshow(image)
        ax = pyplot.gca()
        ax.set_axis_off()
        ax.set_xmargin(0)
        ax.set_ymargin(0)
        ax.set_frame_on(0)
        ax.set_position([0,0,1,1])
        # label the image with the file name
        #ax.set_title(image_path)
        for i, box in enumerate(yhat['rois']):
            # get best class and score of best class
            class_ = yhat['class_ids'][i]
            score_ = yhat['scores'][i]
            # get binary pixel mask for region
            mask = masks[:,:,i]
            # get coordinates of bounding box
            y1, x1, y2, x2 = box
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            rect = Rectangle((x1, y1), width, height, fill=False,
                             color=cm.tab10(class_, alpha=0.8))
            # draw the box onto the image
            ax.add_patch(rect)
            # draw the pixel mask onto the image (semi-transparent)
            # but do not color-code zero (but make transparent)
            # todo: colormap does not work with masked arrays
            # (but a custom cmap with set_under('k',alpha=0) and vmin does not, either)
            ax.imshow(np.ma.masked_where(mask == 0, mask * class_), cmap=cm.tab10, alpha=0.5)
        # write to CWD (not input directory)
        pyplot.savefig(image_path + '.pred.png', dpi=300, bbox_inches='tight')
    # show the very last plot interactively
    pyplot.show()

cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)


if __name__ == '__main__':
    cli()

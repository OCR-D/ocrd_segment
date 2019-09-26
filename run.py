#! python
# -*- coding: utf-8 -*-

import click
import pathlib
import random
from matplotlib import pyplot
from matplotlib.patches import Rectangle

from numpy import expand_dims
from numpy import mean

from layout_dataset import LayoutDataset, LayoutTrainConfig, LayoutPredictConfig
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn.model import MaskRCNN

@click.group()
def cli():
    pass

@cli.command()
@click.argument('data', type=click.Path(dir_okay=True), required=True)
@click.option('-w', '--weights', type=click.Path(file_okay=True), required=True)
@click.option('-h', '--heads', is_flag=True, default=False)
@click.option('-s', '--seed', type=int, default=0)
def train(data,weights,heads,seed):

    #
    # prepare data
    #

    # two data sets
    train_data = LayoutDataset()
    test_data = LayoutDataset()

    # set random seed
    random.seed(seed)

    # read GT
    gt = list(pathlib.Path(data).glob('*/page/*.xml'))
    train = set(random.sample(gt, k=len(gt)*80//100))
    for page in gt:
        page_path = pathlib.Path(page).resolve()
        image_id = page_path.stem
        img = str(page_path.parents[1]) + '/jpg/' + image_id + '.jpg'
        if pathlib.Path(img).exists():
            if page in train:
                train_data.add_image('dataset', image_id=image_id, path=img, annotation=str(page_path))
            else:
                test_data.add_image('dataset', image_id=image_id, path=img, annotation=str(page_path))

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
        model.load_weights(weights, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
    if heads:
        model.train(train_data, test_data, learning_rate=config.LEARNING_RATE, epochs=10, layers='heads')
    else:
        model.train(train_data, test_data, learning_rate=config.LEARNING_RATE, epochs=10, layers='all')

@cli.command()
@click.argument('data', type=click.Path(dir_okay=True), required=True)
@click.option('-w', '--weights', type=click.Path(file_okay=True), required=True)
@click.option('-s', '--seed', type=int, default=0)
def evaluate(data,weights,seed):

    #
    # prepare data
    #

    # one data set
    test_data = LayoutDataset()

    # set random seed
    random.seed(seed)

    # read GT
    gt = list(pathlib.Path(data).glob('*/page/*.xml'))
    train = set(random.sample(gt, k=len(gt)*80//100))
    for page in gt:
        page_path = pathlib.Path(page).resolve()
        image_id = page_path.stem
        img = str(page_path.parents[1]) + '/jpg/' + image_id + '.jpg'
        if pathlib.Path(img).exists():
            if not page in train:
                test_data.add_image('dataset', image_id=image_id, path=img, annotation=str(page_path))

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
        click.echo("Image %s start" % image_id)
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(test_data, config, image_id, use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, config)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        ap, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        aps.append(ap)
        click.echo("Image %s done" % image_id)
    click.echo("Test mAP: %.3f" % mean(aps))

@cli.command()
@click.argument('images', type=click.Path(file_okay=True), required=True, nargs=-1)
@click.option('-w', '--weights', type=click.Path(file_okay=True), required=True)
def predict(images,weights):

    #
    # prepare data
    #

    # one data set
    predict_data = LayoutDataset()

    # read images
    for image in images:
        image_path = pathlib.Path(image).resolve()
        image_id = image_path.stem
        predict_data.add_image('dataset', image_id=image_id, path=str(image_path))

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
        # load image
        image = predict_data.load_image(image_id)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, config)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)[0]
        # show result
        pyplot.imshow(image)
        ax = pyplot.gca()
        for box in yhat['rois']:
            # get coordinates
            y1, x1, y2, x2 = box
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            # draw the box
            ax.add_patch(rect)
    pyplot.show()

cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)


if __name__ == '__main__':
    cli()

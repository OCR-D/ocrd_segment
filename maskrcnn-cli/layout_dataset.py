import sys
from numpy import zeros
from numpy import asarray
from skimage import draw
from mrcnn.utils import Dataset
from mrcnn.config import Config

class LayoutTrainConfig(Config):
    # Give the configuration a recognizable name
    NAME = "layout_train_cfg"
    # Number of classes (background + region types)
    NUM_CLASSES = 1 + 6
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 800
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class LayoutPredictConfig(Config):
    # Give the configuration a recognizable name
    NAME = "layout_predict_cfg"
    # Number of classes (background + region types)
    NUM_CLASSES = 1 + 6
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # default is square, but when trained on crop,
    # we should use the full-resolution image, too:
    #IMAGE_RESIZE_MODE = "none" # yields impatible tensor shapes
    IMAGE_MAX_DIM = 4096

class LayoutDataset(Dataset):

    def __init__(self):
        super().__init__()

        # define classes
        self.add_class("dataset", 1, "text")
        self.add_class("dataset", 2, "separator")
        self.add_class("dataset", 3, "table")
        self.add_class("dataset", 4, "graphic")
        self.add_class("dataset", 5, "noise")
        self.add_class("dataset", 6, "maths")

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_mask(self, image_id):
        """
        Represent region as mask
        """
        info = self.image_info[image_id]
        meta = info['annotation']
        height, width, colours = self.load_image(image_id).shape
        regions = meta.get('regions', [])

        # create one array for all masks, each on a different channel
        masks = zeros([height, width, len(regions)], dtype='uint8')
        # create masks
        class_ids = list()
        for i, region in enumerate(regions):
            class_ = region.get('type', 'text')
            subclass = region.get('subtype', '')
            polygon = region['coords']
            polygon = asarray(polygon, dtype='int32')
            rows, cols = draw.polygon(polygon[:,1], polygon[:,0], (height, width))
            masks[rows, cols, i] = 1
            rows, cols = draw.polygon_perimeter(polygon[:,1], polygon[:,0], (height, width))
            masks[rows, cols, i] = 1
            class_ids.append(self.class_names.index(class_))
        return masks, asarray(class_ids, dtype='int32')

def bbox_from_polygon(polygon,
                      minx=sys.maxsize,
                      miny=sys.maxsize,
                      maxx=-sys.maxsize,
                      maxy=-sys.maxsize):
    """Construct a numeric list representing a bounding box
    from polygon coordinates in numeric list representation."""
    for xy in polygon:
        if xy[0] < minx:
            minx = xy[0]
        if xy[0] > maxx:
            maxx = xy[0]
        if xy[1] < miny:
            miny = xy[1]
        if xy[1] > maxy:
            maxy = xy[1]
    return minx, miny, maxx, maxy

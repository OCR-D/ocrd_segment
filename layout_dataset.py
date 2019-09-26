from lxml import etree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.config import Config

ns = {
     'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15',
     'xlink' : "http://www.w3.org/1999/xlink",
     're' : "http://exslt.org/regular-expressions",
     }
PC = "{%s}" % ns['pc']
XLINK = "{%s}" % ns['xlink']

class LayoutTrainConfig(Config):
    # Give the configuration a recognizable name
    NAME = "layout_train_cfg"
    # Number of classes (background + text)
    NUM_CLASSES = 1 + 1
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 800
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class LayoutPredictConfig(Config):
    # Give the configuration a recognizable name
    NAME = "layout_predict_cfg"
    # Number of classes (background + text)
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class LayoutDataset(Dataset):

    def __init__(self):
        super().__init__()

        # define classes
        self.add_class("dataset", 1, "text")
        #self.add_class("dataset", 2, "separator")
        #self.add_class("dataset", 3, "table")
        #self.add_class("dataset", 4, "graphic")
        #self.add_class("dataset", 5, "noise")
        #self.add_class("dataset", 6, "maths")

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def extract_boxes(self, filename):
        """
        Get regions
        """
        boxes = list()
        page = etree.parse(filename).getroot().find("./" + PC + "Page")
        w = int(page.get('imageWidth'))
        h = int(page.get('imageHeight'))
        for struct in page.xpath(".//*[re:test(local-name(), '[A-Z][a-z]*Region')]", namespaces=ns):
            xys = [tuple([int(p) for p in pair.split(',')]) for pair in struct.find("./" + PC + "Coords").get("points").split(' ')]
            min_x = w
            min_y = h
            max_x = 0
            max_y = 0
            for xy in xys:
                if xy[0] < min_x:
                    min_x = xy[0]
                if xy[0] > max_x:
                    max_x = xy[0]
                if xy[1] < min_y:
                    min_y = xy[1]
                if xy[1] > max_y:
                    max_y = xy[1]
            boxes.append([min_x,min_y,max_x,max_y])
        return boxes, w, h

    def load_mask(self, image_id):
        """
        Represent region as mask
        """
        info = self.image_info[image_id]
        page = info['annotation']
        boxes, w, h = self.extract_boxes(page)

        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('text'))
        return masks, asarray(class_ids, dtype='int32')

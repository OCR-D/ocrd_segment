import json
import sys

coco = json.load(sys.stdin)
coco['images'] = list()
coco['annotations'] = list()
json.dump(coco, sys.stdout, indent=2)

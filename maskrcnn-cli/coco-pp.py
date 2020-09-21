import json
import sys

coco = json.load(sys.stdin)
json.dump(coco, sys.stdout, indent=2)

# ocrd_segment

This repository aims to provide a number of [OCR-D-compliant processors](https://ocr-d.github.io/cli) for layout analysis and evaluation.

[![image](https://img.shields.io/pypi/v/ocrd_segment.svg)](https://pypi.org/project/ocrd_segment/)

## Installation

In your virtual environment, run:
```bash
pip install .
```

## Usage

  - exporting page images (including results from preprocessing like cropping/masking, deskewing, dewarping or binarization) along with region polygon coordinates and metadata, also MS-COCO:
    - [ocrd-segment-extract-pages](ocrd_segment/extract_pages.py)
  - exporting region images (including results from preprocessing like cropping/masking, deskewing, dewarping or binarization) along with region polygon coordinates and metadata:
    - [ocrd-segment-extract-regions](ocrd_segment/extract_regions.py)
  - exporting line images (including results from preprocessing like cropping/masking, deskewing, dewarping or binarization) along with line polygon coordinates and metadata:
    - [ocrd-segment-extract-lines](ocrd_segment/extract_lines.py)
  - importing layout segmentations from other formats (mask images, MS-COCO JSON annotation):
    - [ocrd-segment-from-masks](ocrd_segment/import_image_segmentation.py)
    - [ocrd-segment-from-coco](ocrd_segment/import_coco_segmentation.py)
  - repairing layout segmentations (input file groups N >= 1, based on heuristics implemented using Shapely):
    - [ocrd-segment-repair](ocrd_segment/repair.py) :construction: (much to be done)
  - comparing different layout segmentations (input file groups N = 2, compute the distance between two segmentations, e.g. automatic vs. manual):
    - [ocrd-segment-evaluate](ocrd_segment/evaluate.py) :construction: (very early stage)
  - pattern-based segmentation (input file groups N=1, based on a PAGE template, e.g. from Aletheia, and some XSLT or Python to apply it to the input file group)
    - `ocrd-segment-via-template` :construction: (unpublished)
  - data-driven segmentation (input file groups N=1, based on a statistical model, e.g. Neural Network)  
    - `ocrd-segment-via-model` :construction: (unpublished)

For detailed description on input/output and parameters, see [ocrd-tool.json](ocrd_segment/ocrd-tool.json)

## Testing

None yet.

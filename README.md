# ocrd_segment

This repository aims to provide a number of [OCR-D](https://ocr-d.de) [compliant](https://ocr-d.de/en/spec) [processors](https://ocr-d.de/en/spec/cli) for layout analysis and evaluation.

[![image](https://img.shields.io/pypi/v/ocrd_segment.svg)](https://pypi.org/project/ocrd_segment/)

## Installation

In your [Python virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/), run:

    pip install ocrd_segment


## Usage

Contains processors for various tasks:

  - exporting segment images (including results from preprocessing like cropping/masking, deskewing, dewarping or binarization) along with polygon coordinates and metadata:
    - [ocrd-segment-extract-pages](ocrd_segment/extract_pages.py) (for pages, also exports [MS-COCO](https://cocodataset.org/) format)
    - [ocrd-segment-extract-regions](ocrd_segment/extract_regions.py) (for regions)
    - [ocrd-segment-extract-lines](ocrd_segment/extract_lines.py) (for lines, also exports text and .xlsx)
    - [ocrd-segment-extract-words](ocrd_segment/extract_words.py) (for words, also exports text)
    - [ocrd-segment-extract-glyphs](ocrd_segment/extract_glyphs.py) (for glyphs, also exports text)
  - importing layout segmentations from other formats:
    - [ocrd-segment-from-masks](ocrd_segment/import_image_segmentation.py) (for mask/label images, i.e. semantic segmentation)
    - [ocrd-segment-from-coco](ocrd_segment/import_coco_segmentation.py) (for [MS-COCO](https://cocodataset.org/) annotation)
  - post-processing or repairing layout segmentations:
    - [ocrd-segment-repair](ocrd_segment/repair.py) (validity and consistency of coordinates, reducing overlaps/redundancy between neighbours, shrinking regions to the alpha shape of their lines)
    - [ocrd-segment-project](ocrd_segment/project.py) (remake segment coordinates into the concave hull / alpha shape of their constituents)
    - [ocrd-segment-replace-original](ocrd_segment/replace_original.py) (rebase all segments on cropped+deskewed border frame as new full page)
    - [ocrd-segment-replace-page](ocrd_segment/replace_page.py) (2 input fileGrps; overwrite segmentation below page of first fileGrp by all segments of second fileGrp, rebasing all coordinates; "inverse" of `replace-original`)
  - comparing different layout segmentations:
    - [ocrd-segment-evaluate](ocrd_segment/evaluate.py) :construction: (2 input fileGrps; align, compare and evaluate page segmentations; early stage)
    - [page-segment-evaluate](ocrd_segment/evaluate.py) (same with standalone CLI)
  - pattern-based segmentation (input file groups N=1, based on a PAGE template, e.g. from Aletheia, and some XSLT or Python to apply it to the input file group)
    - `ocrd-segment-via-template` :construction: (unpublished)
  - data-driven segmentation (input file groups N=1, based on a statistical model, e.g. Neural Network)  
    - `ocrd-segment-via-model` :construction: (unpublished)

For detailed behaviour, see `--help` on each processor CLI.
For detailed description on input/output and parameters, see [ocrd-tool.json](ocrd_segment/ocrd-tool.json) or `--dump-json` on each processor CLI.

## Testing

None yet.

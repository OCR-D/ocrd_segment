# ocrd_segment

This repository aims to provide a number of OCR-D-compliant processors for layout analysis and evaluation.

  - pattern-based segmentation aka. `ocrd-segment-via-template` (input file groups N=1, based on a PAGE template, e.g. from Aletheia, and some XSLT or Python to apply it to the input file group)
  - data-driven segmentation aka. `ocrd-segment-via-model` (input file groups N=1, based on a statistical model, e.g. Neural Network)
  - comparing different layout segmentations aka. `ocrd-segment-evaluate` (input file groups N = 2, compute the distance between two segmentations, e.g. automatic vs. manual)
  - repairing of layout segmentations aka. `ocrd-segment-repair` (input file groups N >= 1, based on heuristics implemented using Shapely)

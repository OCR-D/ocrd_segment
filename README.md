# ocrd_segment

This repository aims to provide a number of OCR-D-compliant processors for layout analysis and evaluation.

  - pattern-based segmentation aka. `ocrd-segment-via-template` (input file groups N=1, based on a PAGE template, e.g. from Aletheia, and some XSLT or Python to apply it to the input file group)
  - data-driven segmentation aka. `ocrd-segment-via-model` (input file groups N=1, based on a statistical model, e.g. Neural Network)
  - comparing different layout segmentations aka. `ocrd-segment-evaluate` (input file groups N = 2, compute the distance between two segmentations, e.g. automatic vs. manual)
  - repairing of layout segmentations aka. `ocrd-segment-repair` (input file groups N >= 1, based on heuristics implemented using Shapely)


## Installation

### Requirements

    $ virtualenv env --python=python3.7  # 3.8 did not work yet for all needed libraries at 2019-08-21
    $ source env/bin/activate
    (env) pip install -r requirements.txt  # TODO(js): Pin requirements

### Jupyter Notebook Kernel preparation (to use the virtualenv)

    # XXX(js): Not sure whether we'll need it for a
    (env) ipython kernel install --user ocrd-segmentations

#### And who like VIM key bindings

    (env) jupyter nbextension install https://raw.githubusercontent.com/lambdalisue/jupyter-vim-binding/master/vim_binding.js --nbextensions=$(jupyter --data-dir)/nbextensions/vim_binding
    (env) jupyter nbextension enable vim_binding/vim_binding

#### Run the jupyter notebook

     (env) jupyter notebook

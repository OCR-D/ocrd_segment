Change Log
==========
Versioned according to [Semantic Versioning](http://semver.org/).

## Unreleased

## [0.1.12]

Changed:

 * evaluate: basic IoU matching, Pr/Rc and mAP/mAR stats via pycocotools

## [0.1.11]

Fixed:

 * extract-pages: `Border` has no `id`

## [0.1.10]

Fixed:

 * extract-regions: apply `feature_filter` param

Changed:

 * extract-pages: add `feature_filter` param
 * extract-pages: add `order` choice for `plot_segmasks`

## [0.1.9]

Changed:

 * extract-regions/lines/words/glyphs: add `feature_filter` param

## [0.1.8]

Fixed:

 * replace-page: `getLogger` context

Changed:

 * extract-words: new
 * extract-glyphs: new
 * extract-pages: expose `colordict` parameter (w/ same default)
 * extract-pages: multi-level mask output via `plot_segmasks`

## [0.1.7]

Fixed:

 * repair: also ensure polygons have at least 3 points
 * replace-page: allow non-PAGE input files, too

## [0.1.6]

Fixed:

 * repair: also fix negative coords, also on page level
 * replace-original: also remove page border/@orientation
 * replace-original: add new original as derived image, too

## [0.1.5]

Fixed:

 * evaluate: adapt to `zip_input_files` in core

Changed:

 * replace-original: delegate to `repair.ensure_consistent`
 * replace-page: new CLI (inverse or replace-original)

## [0.1.4]

Changed:

 * repair: fix coordinate consistency/validity errors

## [0.1.3] - 2020-09-24

Changed:

 * logging according to OCR-D/core#599

## [0.1.2] - 2020-09-18

Fixed:

  * repair: traverse all text regions recursively (typo)

## [0.1.1] - 2020-09-14

Changed:

  * repair: traverse all text regions recursively
  
Fixed:

  * repair: be robust against invalid input polygons
  * repair: be careful to make valid output polygons

## [0.1.0] - 2020-08-21

Changed:

  * adapt to 1-output-file-group convention, use `make_file_id` and `assert_file_grp_cardinality`, #41

Fixed:

  * typo in `extract_lines`, #40

## [0.0.2] - 2019-12-19

Changed:

  * further improve README

<!-- link-labels -->
[0.1.12]: ../../compare/v0.1.11...v0.1.12
[0.1.11]: ../../compare/v0.1.10...v0.1.11
[0.1.10]: ../../compare/v0.1.9...v0.1.10
[0.1.9]: ../../compare/v0.1.8...v0.1.9
[0.1.8]: ../../compare/v0.1.7...v0.1.8
[0.1.7]: ../../compare/v0.1.6...v0.1.7
[0.1.6]: ../../compare/v0.1.5...v0.1.6
[0.1.5]: ../../compare/v0.1.4...v0.1.5
[0.1.4]: ../../compare/v0.1.3...v0.1.4
[0.1.3]: ../../compare/v0.1.2...v0.1.3
[0.1.2]: ../../compare/v0.1.1...v0.1.2
[0.1.1]: ../../compare/v0.1.0...v0.1.1
[0.1.0]: ../../compare/v0.0.2...v0.1.0
[0.0.2]: ../../compare/HEAD...v0.0.2

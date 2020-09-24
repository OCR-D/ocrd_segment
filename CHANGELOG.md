Change Log
==========
Versioned according to [Semantic Versioning](http://semver.org/).

## Unreleased

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
[0.1.2]: ../../compare/v0.1.2...v0.1.1
[0.1.1]: ../../compare/v0.1.1...v0.1.0
[0.1.0]: ../../compare/v0.1.0...v0.0.2
[0.0.2]: ../../compare/HEAD...v0.0.2

Change Log
==========
Versioned according to [Semantic Versioning](http://semver.org/).

## Unreleased

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
[0.1.0]: ../../compare/v0.1.0...v0.0.2
[0.0.2]: ../../compare/HEAD...v0.0.2

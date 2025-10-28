Change Log
==========
Versioned according to [Semantic Versioning](http://semver.org/).

## [Unreleased]

## [0.2.2] - 2025-10-28

Fixed:
- extract_regions/page: broken identifier scope since 0.2.0

## [0.2.1] - 2025-05-13

Fixed:

- repair/plausibilize: typo introduced in 8d1cbc4a

## [0.2.0] - 2025-03-07

Changed:

- adapt to (and require) `ocrd>=3.0`
- switched to `pyproject.toml` build, tracking version via `ocrd-tool.json`
- `from_coco`: allow file path as basename (directory part inferred)
- `from_coco`: add parameter `categorydict` (instead of fixed `PubLayNet` mapping),
  but keep `source=PAGE` override in place

Fixed:

- `from_coco`: allow empty (unannotated) pages

## [0.1.24] - 2024-04-25

### Fixed

- project/repair `join_polygons`: fix rare case of adjacent rings
- from_masks: skip if no segmentation input file

### Added

- repair: add params `spread` / `spread_level`, update/improve docs

### Changed

- repair: validate/repair polygons with 0 px tolerance

## [0.1.23] - 2024-03-10

### Fixed

 * repair/project: adapt to Shapely deprecations,
 * repair/project: more robust `join_polygons`, `make_intersections`, `make_valid`

### Changed

 * :fire: require Shapely 2
 * project: clip coords to parent's parent instead of parent
 * repair (`sanitize`): shrink before attempting repair (hierarchical consistency)

## [0.1.22] - 2023-06-29

### Added

 * replace-text: new processor (inverse of extract-* for text)
 * extract-lines: new parameter `textequiv-index`

### Fixed

 * repair/project: use minimum spanning tree when joining polygons
 * repair/project: ever more robustness
 * all: set PAGE `@pcGtsId` from METS `file/@ID`

## [0.1.21] - 2022-05-27

### Changed

 * extract-regions: add parameter `classes` and output COCO, too
 * repair/project: join polygons directly instead of alphashape

## [0.1.20] - 2022-04-27

### Fixed

 * extract-pages: fix extraction of region JSON
 * repair/project: make alpha shape more robust

## [0.1.19] - 2022-04-15

### Changed

 * repair (`sanitize`): run on all region types
 * repair (`sanitize`): add parameter `sanitize_padding`
 * repair (`sanitize`): use binary foreground instead of text line coordinates
 * repair (`plausibilize`): use true alpha shape instead of convex hull
 * project: add `level-of-operation=table`
 * repair: add option `simplify`
 * ensure compatibility with Shapely 1.8

## [0.1.18] - 2022-03-30

### Fixed

 * extract-lines/words: move extra parameters where they belong
 * extract-lines: fix regressions in v0.1.15

## [0.1.17] - 2022-03-12

### Changed

 * project: use true alpha shape instead of convex hull

## [0.1.16] - 2022-02-21

### Fixed

 * repair: fix `plausibilize` scope of apply-list

### Changed

 * project: new processor for convex hull resegmentation

## [0.1.15] - 2022-02-17

### Changed

 * repair: `plausibilize`: both analyse & apply iff enabled
 * extract-lines: add parameters for output types and conditions for line extraction
 * extract-lines: add `xlsx` output option for GT editing

## [0.1.14] - 2022-02-17

### Changed

 * repair: for non-trivial region overlaps, recurse to line level
 * repair: for non-trivial line overlaps, merge (if centric) or subtract

## [0.1.13] - 2021-12-11

### Fixed

 * evaluate: multi-matching (without pycocotools)

### Changed

 * evaluate: improved report format (hierarchy and names)

### Added

 * evaluate: over-/undersegmentation metrics, pixel-wise metrics

## [0.1.12] - 2021-12-03

### Changed

 * evaluate: basic IoU matching, Pr/Rc and mAP/mAR stats via pycocotools

## [0.1.11] - 2021-03-23

### Fixed

 * extract-pages: `Border` has no `id`

## [0.1.10] - 2021-02-26

### Fixed

 * extract-regions: apply `feature_filter` param

### Changed

 * extract-pages: add `feature_filter` param
 * extract-pages: add `order` choice for `plot_segmasks`

## [0.1.9] - 2021-02-10

### Changed

 * extract-regions/lines/words/glyphs: add `feature_filter` param

## [0.1.8] - 2021-02-07

### Fixed

 * replace-page: `getLogger` context

### Changed

 * extract-words: new
 * extract-glyphs: new
 * extract-pages: expose `colordict` parameter (w/ same default)
 * extract-pages: multi-level mask output via `plot_segmasks`

## [0.1.7] - 2021-01-07

### Fixed

 * repair: also ensure polygons have at least 3 points
 * replace-page: allow non-PAGE input files, too

## [0.1.6] - 2020-11-25

### Fixed

 * repair: also fix negative coords, also on page level
 * replace-original: also remove page border/@orientation
 * replace-original: add new original as derived image, too

## [0.1.5] - 2020-11-04

### Fixed

 * evaluate: adapt to `zip_input_files` in core

### Changed

 * replace-original: delegate to `repair.ensure_consistent`
 * replace-page: new CLI (inverse or replace-original)

## [0.1.4] - 2020-11-04

### Changed

 * repair: fix coordinate consistency/validity errors

## [0.1.3] - 2020-09-24

### Changed

 * logging according to OCR-D/core#599

## [0.1.2] - 2020-09-18

### Fixed

  * repair: traverse all text regions recursively (typo)

## [0.1.1] - 2020-09-14

### Changed

  * repair: traverse all text regions recursively

### Fixed

  * repair: be robust against invalid input polygons
  * repair: be careful to make valid output polygons

## [0.1.0] - 2020-08-21

### Changed

  * adapt to 1-output-file-group convention, use `make_file_id` and `assert_file_grp_cardinality`, #41

### Fixed

  * typo in `extract_lines`, #40

## [0.0.2] - 2019-12-19

### Changed

  * further improve README

<!-- link-labels -->
[unreleased]: ../../compare/v0.2.0...master
[0.2.0]: ../../compare/v0.1.24...v0.2.0
[0.1.24]: ../../compare/v0.1.23...v0.1.24
[0.1.23]: ../../compare/v0.1.22...v0.1.23
[0.1.22]: ../../compare/v0.1.21...v0.1.22
[0.1.21]: ../../compare/v0.1.20...v0.1.21
[0.1.20]: ../../compare/v0.1.19...v0.1.20
[0.1.19]: ../../compare/v0.1.18...v0.1.19
[0.1.18]: ../../compare/v0.1.17...v0.1.18
[0.1.17]: ../../compare/v0.1.16...v0.1.17
[0.1.16]: ../../compare/v0.1.15...v0.1.16
[0.1.15]: ../../compare/v0.1.14...v0.1.15
[0.1.14]: ../../compare/v0.1.13...v0.1.14
[0.1.13]: ../../compare/v0.1.12...v0.1.13
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

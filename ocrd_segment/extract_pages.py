from __future__ import absolute_import

import json
from collections import namedtuple
import os.path
import numpy as np
import cv2
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from shapely.validation import explain_validity
from shapely.prepared import prep
import xlsxwriter

from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    coordinates_of_segment,
    xywh_from_polygon,
    polygon_from_bbox,
    MIME_TO_EXT
)
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    OrderedGroupType, OrderedGroupIndexedType,
    RegionRefType, RegionRefIndexedType,
)
from ocrd import Processor

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-extract-pages'
# region classes and their colours in mask (pseg) images:
# (from prima-page-viewer/src/org/primaresearch/page/viewer/ui/render/PageContentColors,
#  but added alpha channel to also discern subtype, if not visually;
#  ordered such that overlaps still allows maximum separation)
# (Not used any more; now as default to ocrd-tool.json parameter.)
# pragma pylint: disable=bad-whitespace
CLASSES = {
    '':                                     'FFFFFF00',
    'Glyph':                                '2E8B08FF',
    'Word':                                 'B22222FF',
    'TextLine':                             '32CD32FF',
    'Border':                               'FFFFFFFF',
    'TableRegion':                          '8B4513FF',
    'AdvertRegion':                         '4682B4FF',
    'ChemRegion':                           'FF8C00FF',
    'MusicRegion':                          '9400D3FF',
    'MapRegion':                            '9ACDD2FF',
    'TextRegion':                           '0000FFFF',
    'TextRegion:paragraph':                 '0000FFFA',
    'TextRegion:heading':                   '0000FFF5',
    'TextRegion:caption':                   '0000FFF0',
    'TextRegion:header':                    '0000FFEB',
    'TextRegion:footer':                    '0000FFE6',
    'TextRegion:page-number':               '0000FFE1',
    'TextRegion:drop-capital':              '0000FFDC',
    'TextRegion:credit':                    '0000FFD7',
    'TextRegion:floating':                  '0000FFD2',
    'TextRegion:signature-mark':            '0000FFCD',
    'TextRegion:catch-word':                '0000FFC8',
    'TextRegion:marginalia':                '0000FFC3',
    'TextRegion:footnote':                  '0000FFBE',
    'TextRegion:footnote-continued':        '0000FFB9',
    'TextRegion:endnote':                   '0000FFB4',
    'TextRegion:TOC-entry':                 '0000FFAF',
    'TextRegion:list-label':                '0000FFA5',
    'TextRegion:other':                     '0000FFA0',
    'ChartRegion':                          '800080FF',
    'ChartRegion:bar':                      '800080FA',
    'ChartRegion:line':                     '800080F5',
    'ChartRegion:pie':                      '800080F0',
    'ChartRegion:scatter':                  '800080EB',
    'ChartRegion:surface':                  '800080E6',
    'ChartRegion:other':                    '800080E1',
    'GraphicRegion':                        '008000FF',
    'GraphicRegion:logo':                   '008000FA',
    'GraphicRegion:letterhead':             '008000F0',
    'GraphicRegion:decoration':             '008000EB',
    'GraphicRegion:frame':                  '008000E6',
    'GraphicRegion:handwritten-annotation': '008000E1',
    'GraphicRegion:stamp':                  '008000DC',
    'GraphicRegion:signature':              '008000D7',
    'GraphicRegion:barcode':                '008000D2',
    'GraphicRegion:paper-grow':             '008000CD',
    'GraphicRegion:punch-hole':             '008000C8',
    'GraphicRegion:other':                  '008000C3',
    'ImageRegion':                          '00CED1FF',
    'LineDrawingRegion':                    'B8860BFF',
    'MathsRegion':                          '00BFFFFF',
    'NoiseRegion':                          'FF0000FF',
    'SeparatorRegion':                      'FF00FFFF',
    'UnknownRegion':                        '646464FF',
    'CustomRegion':                         '637C81FF',
    'ReadingOrderLevel0':                   'DC143CFF',
    'ReadingOrderLevel1':                   '9400D3FF',
    'ReadingOrderLevelN':                   '8B0000FF',
}
# pragma pylint: enable=bad-whitespace

class ExtractPages(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(ExtractPages, self).__init__(*args, **kwargs)

    def process(self):
        """Extract page images and region descriptions (type and coordinates) from the workspace.
        
        Open and deserialize PAGE input files and their respective images,
        then iterate over the element hierarchy down to the region level.
        
        Get all regions with their types (region element class), sub-types (@type)
        and coordinates relative to the page (which depending on the workflow could
        already be cropped, deskewed, dewarped, binarized etc). Extract the image of
        the (cropped, deskewed, dewarped) page, both in binarized form (if available)
        and raw form. For the latter, apply ``feature_filter`` (a comma-separated list
        of image features, cf. :py:func:`ocrd.workspace.Workspace.image_from_page`)
        to skip specific features when retrieving derived images. If ``transparency``
        is true, then also add an alpha channel which is fully transparent outside of
        the mask.
        
        In addition, create a new (third) image with masks for each segment type in
        ``plot_segmasks``, color-coded by class according to ``colordict``.
        
        Create two JSON files with region types and coordinates: one (page-wise) in
        our custom format and one (global) in MS-COCO.
        
        The output file group may be given as a comma-separated list to separate
        these 3 kinds of images. If fewer than 3 fileGrps are specified, they will
        share the same fileGrp (and directory). In particular, write files as follows:
        * in the first (or only) output file group (directory):
          - ID + '.png': raw image of the page (preprocessed, but with ``feature_filter``)
          - ID + '.json': region coordinates/classes (custom format)
        * in the second (or only) output file group (directory):
          - ID + '.bin.png': binarized image of the (preprocessed) page, if available
        * in the third (or second or only) output file group (directory):
          - ID + '.pseg.png': mask image of page; contents depend on ``plot_segmasks``:
            1. if it contains `page`, fill page frame,
            2. if it contains `region`, fill region segmentation/classification,
            3. if it contains `line`, fill text line segmentation,
            4. if it contains `word`, fill word segmentation,
            5. if it contains `glyph`, fill glyph segmentation,
            where each follow-up layer and segment draws over the previous state, starting
            with a blank (white) image - unless ``plot_overlay`` is true, in which case
            each layer and segment is superimposed (alpha blended) onto the previous one,
            starting with the above raw image.
        
        In addition, write a file for all pages at once:
        * in the third (or first) output file group (directory):
          - output_file_grp + '.coco.json': region coordinates/classes (MS-COCO format)
          - output_file_grp + '.colordict.json': the used ``colordict``
        
        (This is intended for training and evaluation of region segmentation models.)
        """
        LOG = getLogger('processor.ExtractPages')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        file_groups = self.output_file_grp.split(',')
        if len(file_groups) > 3:
            raise Exception("at most 3 output file grps allowed (raw, [binarized, [mask]] image)")
        if len(file_groups) > 2:
            mask_image_grp = file_groups[2]
        else:
            mask_image_grp = file_groups[0]
            LOG.info("No output file group for mask images specified, falling back to output filegrp '%s'", mask_image_grp)
        if len(file_groups) > 1:
            bin_image_grp = file_groups[1]
        else:
            bin_image_grp = file_groups[0]
            LOG.info("No output file group for binarized images specified, falling back to output filegrp '%s'", bin_image_grp)
        self.output_file_grp = file_groups[0]
        classes = self.parameter['colordict']

        # COCO: init data structures
        images = list()
        annotations = list()
        categories = list()
        i = 0
        for cat, color in classes.items():
            # COCO format does not allow alpha channel
            color = (int(color[0:2], 16),
                     int(color[2:4], 16),
                     int(color[4:6], 16))
            try:
                supercat, name = cat.split(':')
            except ValueError:
                name = cat
                supercat = ''
            categories.append(
                {'id': i, 'name': name, 'supercategory': supercat,
                 'source': 'PAGE', 'color': color})
            i += 1

        i = 0
        # pylint: disable=attribute-defined-outside-init
        for n, input_file in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            try:
                # separate non-numeric part of page ID to retain the numeric part
                num_page_id = int(page_id.strip(page_id.strip("0123456789")))
            except Exception:
                num_page_id = n
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            page = pcgts.get_Page()
            ptype = page.get_type()
            page_image, page_coords, page_image_info = self.workspace.image_from_page(
                page, page_id,
                feature_filter=self.parameter['feature_filter'],
                transparency=self.parameter['transparency'])
            if page_image_info.resolution != 1:
                dpi = page_image_info.resolution
                if page_image_info.resolutionUnit == 'cm':
                    dpi = round(dpi * 2.54)
            else:
                dpi = None
            file_id = make_file_id(input_file, self.output_file_grp)
            file_path = self.workspace.save_image_file(page_image,
                                                       file_id,
                                                       self.output_file_grp,
                                                       page_id=page_id,
                                                       mimetype=self.parameter['mimetype'])
            try:
                page_image_bin, _, _ = self.workspace.image_from_page(
                    page, page_id,
                    feature_selector='binarized',
                    transparency=self.parameter['transparency'])
                self.workspace.save_image_file(page_image_bin,
                                               file_id + '.bin',
                                               bin_image_grp,
                                               page_id=page_id)
            except Exception as err:
                if err.args[0].startswith('Found no AlternativeImage'):
                    LOG.warning('Page "%s" has no binarized images, skipping .bin', page_id)
                else:
                    raise
            # init multi-level mask output
            if self.parameter['plot_overlay']:
                page_image_segmask = page_image.convert('RGBA')
            else:
                page_image_segmask = Image.new(mode='RGBA',
                                               size=page_image.size,
                                               color='#FFFFFF00')
            neighbors = dict()
            for level in ['page', 'region', 'line', 'word', 'glyph']:
                neighbors[level] = list()
            # produce border mask plot, if necessary
            if page.get_Border():
                poly = segment_poly(page_id, page.get_Border(), page_coords)
            else:
                poly = Polygon(polygon_from_bbox(0, 0, page_image.width, page_image.height))
            if 'page' in self.parameter['plot_segmasks']:
                plot_segment(page_id, page.get_Border(), poly, 'Border', classes,
                             page_image_segmask, [], self.parameter['plot_overlay'])
            # get regions and aggregate masks on all hierarchy levels
            description = {'angle': page.get_orientation()}
            regions = dict()
            for name in classes.keys():
                if not name or not name.endswith('Region'):
                    # no region subtypes or non-region types here
                    continue
                #regions[name] = getattr(page, 'get_' + name)()
                regions[name] = page.get_AllRegions(classes=name[:-6], order='reading-order')
            for rtype, rlist in regions.items():
                for region in rlist:
                    if rtype in ['TextRegion', 'ChartRegion', 'GraphicRegion']:
                        subrtype = region.get_type()
                    else:
                        subrtype = None
                    if subrtype:
                        rtype0 = rtype + ':' + subrtype
                    else:
                        rtype0 = rtype
                    poly = segment_poly(page_id, region, page_coords)
                    # produce region mask plot, if necessary
                    if 'region' in self.parameter['plot_segmasks']:
                        plot_segment(page_id, region, poly, rtype0, classes,
                                     page_image_segmask, neighbors['region'],
                                     self.parameter['plot_overlay'])
                    if rtype == 'TextRegion':
                        lines = region.get_TextLine()
                        for line in lines:
                            # produce line mask plot, if necessary
                            poly = segment_poly(page_id, line, page_coords)
                            if 'line' in self.parameter['plot_segmasks']:
                                plot_segment(page_id, line, poly, 'TextLine', classes,
                                             page_image_segmask, neighbors['line'],
                                             self.parameter['plot_overlay'])
                            words = line.get_Word()
                            for word in words:
                                # produce line mask plot, if necessary
                                poly = segment_poly(page_id, word, page_coords)
                                if 'word' in self.parameter['plot_segmasks']:
                                    plot_segment(page_id, word, poly, 'Word', classes,
                                                 page_image_segmask, neighbors['word'],
                                                 self.parameter['plot_overlay'])
                                glyphs = word.get_Glyph()
                                for glyph in glyphs:
                                    # produce line mask plot, if necessary
                                    poly = segment_poly(page_id, glyph, page_coords)
                                    if 'glyph' in self.parameter['plot_segmasks']:
                                        plot_segment(page_id, glyph, poly, 'Glyph', classes,
                                                     page_image_segmask, neighbors['glyph'],
                                                     self.parameter['plot_overlay'])
                    if not poly:
                        continue
                    polygon = np.array(poly.exterior, np.int)[:-1].tolist()
                    xywh = xywh_from_polygon(polygon)
                    area = poly.area
                    description.setdefault('regions', []).append(
                        { 'type': rtype,
                          'subtype': subrtype,
                          'coords': polygon,
                          'area': area,
                          'features': page_coords['features'],
                          'DPI': dpi,
                          'region.ID': region.id,
                          'page.ID': page_id,
                          'page.type': ptype,
                          'file_grp': self.input_file_grp,
                          'METS.UID': self.workspace.mets.unique_identifier
                        })
                    # COCO: add annotations
                    i += 1
                    annotations.append(
                        {'id': i, 'image_id': num_page_id,
                         'category_id': next((cat['id'] for cat in categories if cat['name'] == subrtype),
                                             next((cat['id'] for cat in categories if cat['name'] == rtype))),
                         'segmentation': np.array(poly.exterior, np.int)[:-1].reshape(1, -1).tolist(),
                         'area': area,
                         'bbox': [xywh['x'], xywh['y'], xywh['w'], xywh['h']],
                         'iscrowd': 0})

            if 'order' in self.parameter['plot_segmasks']:
                plot_order(page.get_ReadingOrder(), classes, page_image_segmask,
                           neighbors['region'], self.parameter['plot_overlay'])
            if self.parameter['plot_segmasks']:
                self.workspace.save_image_file(page_image_segmask,
                                               file_id + '.pseg',
                                               mask_image_grp,
                                               page_id=page_id,
                                               mimetype=self.parameter['mimetype'])
            self.workspace.add_file(
                ID=file_id + '.json',
                file_grp=mask_image_grp,
                pageId=input_file.pageId,
                local_filename=file_path.replace(MIME_TO_EXT[self.parameter['mimetype']], '.json'),
                mimetype='application/json',
                content=json.dumps(description))

            # COCO: add image
            images.append({
                # COCO does not allow string identifiers:
                # -> use numerical part of page_id
                'id': num_page_id,
                # all exported coordinates are relative to the cropped page:
                # -> use that for reference (instead of original page.imageFilename)
                'file_name': file_path,
                # -> use its size (instead of original page.imageWidth/page.imageHeight)
                'width': page_image.width,
                'height': page_image.height})
        
        # COCO: write result
        file_id = mask_image_grp + '.coco.json'
        LOG.info('Writing COCO result file "%s" in "%s"', file_id, mask_image_grp)
        self.workspace.add_file(
            ID=file_id,
            file_grp=mask_image_grp,
            local_filename=os.path.join(mask_image_grp, file_id),
            mimetype='application/json',
            pageId=None,
            content=json.dumps(
                {'categories': categories,
                 'images': images,
                 'annotations': annotations}))

        # write inverse colordict (for ocrd-segment-from-masks)
        file_id = mask_image_grp + '.colordict.json'
        LOG.info('Writing colordict file "%s" in .', file_id)
        with open(file_id, 'w') as out:
            json.dump(dict((col, name)
                           for name, col in classes.items()
                           if name),
                      out)

def segment_poly(page_id, segment, coords):
    LOG = getLogger('processor.ExtractPages')
    polygon = coordinates_of_segment(segment, None, coords)
    # validate coordinates
    try:
        poly = Polygon(polygon)
        reason = ''
        if not poly.is_valid:
            reason = explain_validity(poly)
        elif poly.is_empty:
            reason = 'is empty'
        elif poly.bounds[0] < 0 or poly.bounds[1] < 0:
            reason = 'is negative'
        elif poly.length < 4:
            reason = 'has too few points'
    except ValueError as err:
        reason = err
    if reason:
        tag = segment.__class__.__name__.replace('Type', '')
        if tag != 'Border':
            tag += ' "%s"' % segment.id
        LOG.error('Page "%s" %s %s', page_id, tag, reason)
        return None
    return poly

def plot_order(readingorder, classes, image, regions, alpha=False):
    LOG = getLogger('processor.ExtractPages')
    regiondict = dict((region.id, region.poly) for region in regions)
    def get_points(rogroup, level):
        points = list()
        if isinstance(rogroup, (OrderedGroupType, OrderedGroupIndexedType)):
            # FIXME: @index is broken in prima-core-libs+prima-page-viewer and producers
            # (so we have to do ignore index here too to stay compatible)
            regionrefs = rogroup.get_AllIndexed(index_sort=False)
        else:
            # FIXME: PageViewer does not render these in-order via arrows,
            #        but creates a "star" plus circle for unordered groups
            regionrefs = rogroup.get_UnorderedGroupChildren()
        for regionref in regionrefs:
            morepoints = list()
            poly = regiondict.get(regionref.get_regionRef(), None)
            if poly:
                # we have seen this region
                morepoints.append((level, tuple(np.array(poly.centroid, np.int))))
            if not isinstance(regionref, (RegionRefType, RegionRefIndexedType)):
                # try to get subgroup regions instead
                morepoints = get_points(regionref, level + 1) or morepoints
            points.extend(morepoints)
        return points
    newimg = 255 * np.ones((image.height, image.width, 3), np.uint8)
    points = [(0, (0, 0))]
    if readingorder:
        readingorder = readingorder.get_OrderedGroup() or readingorder.get_UnorderedGroup()
    if readingorder:
        # use recursive group ordering
        points.extend(get_points(readingorder, 0))
    else:
        # use XML ordering
        points.extend([(0, tuple(np.array(region.poly.centroid, np.int))) for region in regions])
    for p1, p2 in zip(points[:-1], points[1:]):
        color = 'ReadingOrderLevel%s' % (str(p1[0]) if p1[0] < 2 else 'N')
        if color not in classes:
            LOG.error('mask plots requested, but "colordict" does not contain a "%s" mapping', color)
            return
        color = classes[color]
        color = (int(color[0:2], 16),
                 int(color[2:4], 16),
                 int(color[4:6], 16))
        cv2.arrowedLine(newimg, p1[1], p2[1], color, thickness=2, tipLength=0.01)
    layer = Image.fromarray(newimg)
    layer.putalpha(Image.fromarray(255 * np.any(newimg < 255, axis=2).astype(np.uint8), mode='L'))
    image.alpha_composite(layer)

def plot_segment(page_id, segment, poly, stype, classes, image, neighbors, alpha=False):
    LOG = getLogger('processor.ExtractPages')
    if not poly:
        return
    if stype not in classes:
        LOG.error('mask plots requested, but "colordict" does not contain a "%s" mapping', stype)
        return
    color = classes[stype]
    Neighbor = namedtuple('Neighbor', ['id', 'poly', 'type'])
    LOG = getLogger('processor.ExtractPages')
    # check intersection with neighbours
    # (which would melt into another in the mask image)
    if segment and hasattr(segment, 'id') and not alpha:
        poly_prep = prep(poly)
        for neighbor in neighbors:
            if (stype == neighbor.type and
                poly_prep.intersects(neighbor.poly) and
                poly.intersection(neighbor.poly).area > 0):
                inter = poly.intersection(neighbor.poly).area
                union = poly.union(neighbor.poly).area
                LOG.warning('Page "%s" segment "%s" intersects neighbour "%s" (IoU: %.3f)',
                            page_id, segment.id, neighbor.id, inter / union)
            elif (stype != neighbor.type and
                  poly_prep.within(neighbor.poly)):
                LOG.warning('Page "%s" segment "%s" within neighbour "%s" (IoU: %.3f)',
                            page_id, segment.id, neighbor.id,
                            poly.area / neighbor.poly.area)
    if segment and hasattr(segment, 'id'):
        neighbors.append(Neighbor(segment.id, poly, stype))
    # draw segment
    if alpha:
        layer = Image.new(mode='RGBA', size=image.size, color='#FFFFFF00')
        ImageDraw.Draw(layer).polygon(list(map(tuple, poly.exterior.coords[:-1])),
                                      fill='#' + color[:6] + '1E',
                                      outline='#' + color[:6] + '96')
        image.alpha_composite(layer)
    else:
        ImageDraw.Draw(image).polygon(list(map(tuple, poly.exterior.coords[:-1])),
                                      fill='#' + color)

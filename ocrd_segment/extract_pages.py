from __future__ import absolute_import

import json
from collections import namedtuple
import os.path
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from shapely.validation import explain_validity
from shapely.prepared import prep

from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    coordinates_of_segment,
    xywh_from_polygon,
    MIME_TO_EXT
)
from ocrd_modelfactory import page_from_file
from ocrd import Processor

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-extract-pages'
LOG = getLogger('processor.ExtractPages')
# region classes and their colours in mask (dbg) images:
# (from prima-page-viewer/src/org/primaresearch/page/viewer/ui/render/PageContentColors,
#  but added alpha channel to also discern subtype, if not visually;
#  ordered such that overlaps still allows maximum separation)
# pragma pylint: disable=bad-whitespace
CLASSES = {
    '':                                     'FFFFFF00',
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
    'CustomRegion':                         '637C81FF'}
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
        and non-binarized form. In addition, create a new image with masks for all
        regions, color-coded by type. Create two JSON files with region types and
        coordinates: one (page-wise) in our custom format and one (global) in MS-COCO.
        
        The output file group may be given as a comma-separated list to separate
        these 3 page-level images. Write files as follows:
        * in the first (or only) output file group (directory):
          - ID + '.png': raw image of the (preprocessed) page
          - ID + '.json': region coordinates/classes (custom format)
        * in the second (or first) output file group (directory):
          - ID + '.bin.png': binarized image of the (preprocessed) page, if available
        * in the third (or first) output file group (directory):
          - ID + '.dbg.png': debug image
        
        In addition, write a file for all pages at once:
        * in the third (or first) output file group (directory):
          - output_file_grp + '.coco.json': region coordinates/classes (MS-COCO format)
          - output_file_grp + '.colordict.json': color definitions (as in PAGE viewer)
        
        (This is intended for training and evaluation of region segmentation models.)
        """
        assert_file_grp_cardinality(self.input_file_grp, 1)
        file_groups = self.output_file_grp.split(',')
        if len(file_groups) > 3:
            raise Exception("at most 3 output file grps allowed (raw, [binarized, [mask]] image)")
        if len(file_groups) > 2:
            dbg_image_grp = file_groups[2]
        else:
            dbg_image_grp = file_groups[0]
            LOG.info("No output file group for debug images specified, falling back to output filegrp '%s'", dbg_image_grp)
        if len(file_groups) > 1:
            bin_image_grp = file_groups[1]
        else:
            bin_image_grp = file_groups[0]
            LOG.info("No output file group for binarized images specified, falling back to output filegrp '%s'", bin_image_grp)
        self.output_file_grp = file_groups[0]

        # COCO: init data structures
        images = list()
        annotations = list()
        categories = list()
        i = 0
        for cat, color in CLASSES.items():
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
            num_page_id = int(page_id.strip(page_id.strip("0123456789")))
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            page = pcgts.get_Page()
            ptype = page.get_type()
            page_image, page_coords, page_image_info = self.workspace.image_from_page(
                page, page_id,
                feature_filter='binarized',
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
            page_image_dbg = Image.new(mode='RGBA', size=page_image.size,
                                       color='#' + CLASSES[''])
            if page.get_Border():
                polygon = coordinates_of_segment(
                    page.get_Border(), page_image, page_coords).tolist()
                ImageDraw.Draw(page_image_dbg).polygon(
                    list(map(tuple, polygon)),
                    fill='#' + CLASSES['Border'])
            else:
                page_image_dbg.paste('#' + CLASSES['Border'],
                                     (0, 0, page_image.width, page_image.height))
            regions = dict()
            for name in CLASSES.keys():
                if not name or name == 'Border' or ':' in name:
                    # no subtypes here
                    continue
                regions[name] = getattr(page, 'get_' + name)()
            description = {'angle': page.get_orientation()}
            Neighbor = namedtuple('Neighbor', ['id', 'poly', 'type'])
            neighbors = []
            for rtype, rlist in regions.items():
                for region in rlist:
                    if rtype in ['TextRegion', 'ChartRegion', 'GraphicRegion']:
                        subrtype = region.get_type()
                    else:
                        subrtype = None
                    polygon = coordinates_of_segment(
                        region, page_image, page_coords)
                    polygon2 = polygon.reshape(1, -1).tolist()
                    polygon = polygon.tolist()
                    xywh = xywh_from_polygon(polygon)
                    # validate coordinates and check intersection with neighbours
                    # (which would melt into another in the mask image):
                    try:
                        poly = Polygon(polygon)
                        reason = ''
                    except ValueError as err:
                        reason = err
                    if not poly.is_valid:
                        reason = explain_validity(poly)
                    elif poly.is_empty:
                        reason = 'is empty'
                    elif poly.bounds[0] < 0 or poly.bounds[1] < 0:
                        reason = 'is negative'
                    elif poly.length < 4:
                        reason = 'has too few points'
                    if reason:
                        LOG.error('Page "%s" region "%s" %s',
                                  page_id, region.id, reason)
                        continue
                    poly_prep = prep(poly)
                    for neighbor in neighbors:
                        if (rtype == neighbor.type and
                            poly_prep.intersects(neighbor.poly) and
                            poly.intersection(neighbor.poly).area > 0):
                            LOG.warning('Page "%s" region "%s" intersects neighbour "%s" (IoU: %.3f)',
                                        page_id, region.id, neighbor.id,
                                        poly.intersection(neighbor.poly).area / \
                                        poly.union(neighbor.poly).area)
                        elif (rtype != neighbor.type and
                              poly_prep.within(neighbor.poly)):
                            LOG.warning('Page "%s" region "%s" within neighbour "%s" (IoU: %.3f)',
                                        page_id, region.id, neighbor.id,
                                        poly.area / neighbor.poly.area)
                    neighbors.append(Neighbor(region.id, poly, rtype))
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
                    # draw region:
                    ImageDraw.Draw(page_image_dbg).polygon(
                        list(map(tuple, polygon)),
                        fill='#' + CLASSES[(rtype + ':' + subrtype) if subrtype else rtype])
                    # COCO: add annotations
                    i += 1
                    annotations.append(
                        {'id': i, 'image_id': num_page_id,
                         'category_id': next((cat['id'] for cat in categories if cat['name'] == subrtype),
                                             next((cat['id'] for cat in categories if cat['name'] == rtype))),
                         'segmentation': polygon2,
                         'area': area,
                         'bbox': [xywh['x'], xywh['y'], xywh['w'], xywh['h']],
                         'iscrowd': 0})
            
            self.workspace.save_image_file(page_image_dbg,
                                           file_id + '.dbg',
                                           dbg_image_grp,
                                           page_id=page_id,
                                           mimetype=self.parameter['mimetype'])
            self.workspace.add_file(
                ID=file_id + '.json',
                file_grp=dbg_image_grp,
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
        file_id = dbg_image_grp + '.coco.json'
        LOG.info('Writing COCO result file "%s" in "%s"', file_id, dbg_image_grp)
        self.workspace.add_file(
            ID=file_id,
            file_grp=dbg_image_grp,
            local_filename=os.path.join(dbg_image_grp, file_id),
            mimetype='application/json',
            pageId=None,
            content=json.dumps(
                {'categories': categories,
                 'images': images,
                 'annotations': annotations}))

        # write inverse colordict (for ocrd-segment-from-masks)
        file_id = dbg_image_grp + '.colordict.json'
        LOG.info('Writing colordict file "%s" in .', file_id)
        with open(file_id, 'w') as out:
            json.dump(dict(('#' + col, name)
                           for name, col in CLASSES.items()
                           if name),
                      out)

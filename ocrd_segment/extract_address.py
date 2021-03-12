from __future__ import absolute_import

import json
from shapely.geometry import Polygon
from PIL import Image, ImageDraw

from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    coordinates_of_segment,
    xywh_from_polygon
)
from ocrd_models.ocrd_page import TextLineType
from ocrd_modelfactory import page_from_file
from ocrd import Processor

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-extract-address'

class ExtractAddress(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(ExtractAddress, self).__init__(*args, **kwargs)

    def process(self):
        """Extract alpha-masked page images and region descriptions (type and coordinates) from the workspace.
        
        Open and deserialize PAGE input files and their respective images,
        then iterate over the element hierarchy down to the region level.
        
        Extract the image of the (cropped, deskewed, dewarped) page in raw
        (non-binarized) form. Get all regions with their types (region class),
        sub-types (@type) and coordinates relative to the page (which could
        already be cropped, deskewed, dewarped, binarized etc, depending on
        the workflow).
        Create a mask for all text lines and another mask for all text lines
        which belong to a context region, marked by ``TextRegion/@type=other``
        with ``@custom=address``. Combine both masks into one alpha channel,
        using 200 for text and 255 for address. Blend that into the page image.
        Also, for all target regions, marked likewise, aggregate annotations
        (including type and coordinates) into two distinct JSON containers:
        one (page-wise) in our custom format and one (global) in COCO.
        
        The output file group must be given as a comma-separated list, in order
        to write output files:
        * in the first output file group (directory):
          - ID + '.png': raw image of the (preprocessed) page,
                         with color-coded text line masks as alpha channel
        * in the second output file group (directory):
          - ID + '.json': region coordinates/classes (custom format)
          - output_file_grp + '.coco.json': region coordinates/classes (COCO format)
        
        (This is intended for training and evaluation of region segmentation models.)
        """
        LOG = getLogger('processor.ExtractAddress')
        assert_file_grp_cardinality(self.output_file_grp, 2, msg="(masked image, JSON GT)")
        image_file_grp, json_file_grp = self.output_file_grp.split(',')

        # COCO: init data structures
        images = list()
        annotations = list()
        cat2id = dict()
        type2cat = dict()
        categories = self.parameter['categories']
        for cat in categories:
            cat2id[cat['name']] = cat['id']
            if "type" in cat:
                type2cat[cat['type']] = cat['name']
        
        # pylint: disable=attribute-defined-outside-init
        i = 0
        for n, input_file in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            num_page_id = n
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page = pcgts.get_Page()
            ptype = page.get_type()
            page_image, page_coords, page_image_info = self.workspace.image_from_page(
                page, page_id,
                feature_filter='binarized',
                transparency=False)
            if page_image_info.resolution != 1:
                dpi = page_image_info.resolution
                if page_image_info.resolutionUnit == 'cm':
                    dpi = round(dpi * 2.54)
            else:
                dpi = None
            # prepare mask image (alpha channel for input image)
            page_image_mask = Image.new(mode='L', size=page_image.size, color=0)
            # prepare region JSON (output segmentation)
            description = {'angle': page.get_orientation()}
            # iterate through all regions that could have lines
            for region in page.get_AllRegions(classes=['Text']):
                subtype = ''
                score = 1.0
                if region.get_type() in type2cat:
                    subtype = type2cat[region.get_type()]
                elif region.get_type() == 'other' and region.get_custom():
                    subtype = region.get_custom().replace('subtype:', '')
                    score = region.get_Coords().get_conf() or 1.0
                if subtype.startswith('address'):
                    fill = 255
                else:
                    fill = 200
                if not region.get_TextLine():
                    LOG.warning('text region "%s" does not contain text lines on page "%s"',
                                region.id, page_id)
                    # happens when annotating a new region in LAREX
                    region.add_TextLine(TextLineType(id=region.id + '_line',
                                                     Coords=region.get_Coords()))
                # add to mask image (alpha channel for input image)
                for line in region.get_TextLine():
                    if line.get_custom() and line.get_custom().startswith('subtype: ADDRESS'):
                        fill = 255
                    polygon = coordinates_of_segment(line, page_image, page_coords)
                    # draw line mask:
                    ImageDraw.Draw(page_image_mask).polygon(
                        list(map(tuple, polygon.tolist())),
                        fill=fill)
                if not subtype.startswith('address'):
                    continue
                # add to region JSON (output segmentation)
                polygon = coordinates_of_segment(
                    region, page_image, page_coords)
                polygon2 = polygon.reshape(1, -1).tolist()
                polygon = polygon.tolist()
                xywh = xywh_from_polygon(polygon)
                poly = Polygon(polygon)
                area = poly.area
                description.setdefault('regions', []).append(
                    { 'type': subtype,
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
                     'category_id': cat2id.get(subtype),
                     'segmentation': polygon2,
                     'area': area,
                     'bbox': [xywh['x'], xywh['y'], xywh['w'], xywh['h']],
                     'score': score,
                     'iscrowd': 0})
            # write raw+mask RGBA PNG
            if page_image.mode.startswith('I') or page_image.mode == 'F':
                # workaround for Pillow#4926
                page_image = page_image.convert('RGB')
            if page_image.mode == '1':
                page_image = page_image.convert('L')
            page_image.putalpha(page_image_mask)
            file_id = make_file_id(input_file, image_file_grp)
            file_path = self.workspace.save_image_file(page_image,
                                                       file_id,
                                                       image_file_grp,
                                                       page_id=page_id,
                                                       mimetype='image/png')
            # write regions to custom JSON for this page
            file_id = make_file_id(input_file, json_file_grp)
            self.workspace.add_file(
                ID=file_id + '.json',
                file_grp=json_file_grp,
                pageId=input_file.pageId,
                local_filename=file_path.replace('.png', '.json'),
                mimetype='application/json',
                content=json.dumps(description, indent=2))

            # add regions to COCO JSON for all pages
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
        
        # write COCO JSON for all pages
        file_id = json_file_grp + '.coco'
        LOG.info('Writing COCO result file "%s" in "%s"', file_id, json_file_grp)
        self.workspace.add_file(
            ID='id' + file_id,
            file_grp=json_file_grp,
            pageId=None,
            local_filename=file_id + '.json',
            mimetype='application/json',
            content=json.dumps(
                {'categories': categories,
                 'images': images,
                 'annotations': annotations},
                indent=2))

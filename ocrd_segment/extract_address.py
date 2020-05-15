from __future__ import absolute_import

import json
import itertools
import os.path
from shapely.geometry import Polygon
from PIL import Image, ImageDraw

from ocrd_utils import (
    getLogger, concat_padded,
    coordinates_of_segment,
    xywh_from_polygon
)
from ocrd_models.ocrd_page import (
    LabelsType, LabelType,
    MetadataItemType
)
from ocrd_modelfactory import page_from_file
from ocrd import Processor

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-extract-address'
LOG = getLogger('processor.ExtractAddress')

class ExtractAddress(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(ExtractAddress, self).__init__(*args, **kwargs)

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
        
        The output file group must be given as a comma-separated list:
        * in the first output file group (directory):
          - ID + '.png': raw image of the (preprocessed) page,
                         with color-coded text line masks as alpha channel
        * in the second output file group (directory):
          - ID + '.json': region coordinates/classes (custom format)
        
        In addition, write a file for all pages at once:
        * in the second output file group (directory):
          - output_file_grp + '.coco.json': region coordinates/classes (MS-COCO format)
        
        (This is intended for training and evaluation of region segmentation models.)
        """
        try:
            image_file_grp, json_file_grp = self.output_file_grp.split(',')
        except ValueError:
            raise Exception("requires 2 output file grps (output-image,output-JSON)")

        # COCO: init data structures
        images = list()
        annotations = list()
        categories = [
            {'id': 0,
             'name': 'address-rcpt',
             'source': 'IAO'},
            {'id': 1,
             'name': 'address-sndr',
             'source': 'IAO'},
            {'id': 2,
             'name': 'address-contact',
             'source': 'IAO'}
        ]
        
        # pylint: disable=attribute-defined-outside-init
        i = 0
        for n, input_file in enumerate(self.input_files):
            file_id = input_file.ID.replace(self.input_file_grp, image_file_grp)
            page_id = input_file.pageId or input_file.ID
            num_page_id = int(page_id.strip(page_id.strip("0123456789")))
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
            for region in itertools.chain.from_iterable(
                    [page.get_TextRegion()] +
                    [subregion.get_TextRegion() for subregion in page.get_TableRegion()]):
                subtype = ''
                if region.get_type() == 'other' and region.get_custom():
                    subtype = region.get_custom().replace('subtype:', '')
                if subtype.startswith('address'):
                    fill = 255
                else:
                    fill = 200
                # add to mask image (alpha channel for input image)
                for line in region.get_TextLine():
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
                     'category_id': {
                         "address-rcpt": 0,
                         "address-sndr": 1,
                         "address-contact": 2
                     }.get(subtype),
                     'segmentation': polygon2,
                     'area': area,
                     'bbox': [xywh['x'], xywh['y'], xywh['w'], xywh['h']],
                     'iscrowd': 0})
            # write raw+mask RGBA PNG
            page_image.putalpha(page_image_mask)
            file_path = self.workspace.save_image_file(page_image,
                                                       file_id,
                                                       image_file_grp,
                                                       page_id=page_id,
                                                       mimetype='image/png')
            # write regions to custom JSON for this page
            self.workspace.add_file(
                ID=file_id + '.json',
                file_grp=json_file_grp,
                pageId=page_id,
                local_filename=file_path.replace('.png', '.json'),
                mimetype='application/json',
                content=json.dumps(description))

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
        file_id = json_file_grp + '.coco.json'
        LOG.info('Writing COCO result file "%s" in "%s"', file_id, json_file_grp)
        self.workspace.add_file(
            ID=file_id,
            file_grp=json_file_grp,
            local_filename=os.path.join(json_file_grp, file_id),
            mimetype='application/json',
            content=json.dumps(
                {'categories': categories,
                 'images': images,
                 'annotations': annotations}))


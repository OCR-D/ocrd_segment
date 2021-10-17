from __future__ import absolute_import

import json
import os.path
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

from maskrcnn_cli.formdata import FIELDS, CTXT_CATEGORY, TEXT_CATEGORY
from .config import OCRD_TOOL

TOOL = 'ocrd-segment-extract-formdata'

class ExtractFormData(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(ExtractFormData, self).__init__(*args, **kwargs)

    def process(self):
        """Extract alpha-masked page images and region descriptions (type and coordinates) for a single class from the workspace.
        
        Open and deserialize PAGE input files and their respective images,
        then iterate over the element hierarchy down to the region level.
        
        Extract the image of the (cropped, deskewed, dewarped) page in raw
        (non-binarized) form. Get all regions with their types (region class),
        sub-types (@type) and coordinates relative to the page (which could
        already be cropped, deskewed, dewarped, binarized etc, depending on
        the workflow).
        
        Retrieve all input (i.e. context) and output (i.e. target) markers
        for the class given by ``categories``. For that:
        1. Create a mask for all text lines and another mask for all text lines
           or words which belong to a context region, marked by either
           ``TextRegion/@type`` (as given by ``context-type``) or marked by
           ``@custom="subtype:context=..."`` (for the given class).
        2. Combine both masks into one alpha channel, using 200 for text
           and 255 for context. Blend that channel into the page image.
        3. Create list of annotations for all target regions, marked by either
           ``TextRegion/@type`` (as given by ``target-type``) or marked by
           ``@custom="subtype:target=..."`` (for the given class).
        4. Aggregate annotations (including type and coordinates) into
           two distinct JSON containers: one (page-wise) in our custom format
           and one (global) in COCO.
        
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
        LOG = getLogger('processor.ExtractFormData')
        assert_file_grp_cardinality(self.output_file_grp, 2, msg="(masked image, JSON GT)")
        image_file_grp, json_file_grp = self.output_file_grp.split(',')
        source = self.parameter['source']

        # COCO: init data structures
        images = list()
        categories = dict()
        annotations = dict()
        for i, cat in enumerate(FIELDS[1:], 1):
            categories[i] = [{'id': CTXT_CATEGORY, 'name': 'CTXT'},
                             {'id': TEXT_CATEGORY, 'name': 'TEXT'},
                             {'id': 0, 'name': 'BG'},
                             {'id': i, 'name': cat, 'source': source}]
            annotations[i] = list()
        
        # pylint: disable=attribute-defined-outside-init
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
            if page_image.mode == '1':
                page_image = page_image.convert('L')
            if page_image_info.resolution != 1:
                dpi = page_image_info.resolution
                if page_image_info.resolutionUnit == 'cm':
                    dpi = round(dpi * 2.54)
            else:
                dpi = None
            # prepare region JSON (output segmentation)
            description = {'angle': page.get_orientation()}
            def get_context(segment):
                custom = segment.get_custom()
                if not custom:
                    return []
                return [cat.replace('subtype:context=', '')
                        for cat in custom.split(',')
                        if cat.startswith('subtype:context=')]
            def get_target(segment):
                custom = segment.get_custom()
                if not custom:
                    return []
                return [cat.replace('subtype:target=', '')
                        for cat in custom.split(',')
                        if cat.startswith('subtype:target=')]
            # iterate through all regions that could have lines
            for region in page.get_AllRegions(classes=['Text']):
                score = region.get_Coords().get_conf() or 1.0
                polygon = coordinates_of_segment(region, page_image, page_coords)
                if len(polygon) < 3:
                    LOG.warning('ignoring region "%s" with only %d points', region.id, len(polygon))
                    continue
                contexts = get_context(region)
                targets = get_target(region)
                for i, cat in enumerate(FIELDS[1:], 1):
                    if region.get_type() == 'other' and cat in contexts:
                        add_annotation(annotations, num_page_id, i, CTXT_CATEGORY, polygon, score)
                    else:
                        add_annotation(annotations, num_page_id, i, TEXT_CATEGORY, polygon, score)
                    if region.get_type() == 'other' and cat in targets:
                        add_annotation(annotations, num_page_id, i, i, polygon, score)
                description.setdefault('regions', []).append(
                    { 'type': region.get_type() + ':' + ','.join(targets),
                      'coords': polygon.tolist(),
                      'features': page_coords['features'],
                      'DPI': dpi,
                      'region.ID': region.id,
                      'page.ID': page_id,
                      'page.type': ptype,
                      'file_grp': self.input_file_grp,
                      'METS.UID': self.workspace.mets.unique_identifier
                    })
                if not region.get_TextLine():
                    LOG.warning('text region "%s" does not contain text lines on page "%s"',
                                region.id, page_id)
                    # happens when annotating a new region in LAREX
                    region.add_TextLine(TextLineType(id=region.id + '_line',
                                                     Coords=region.get_Coords()))
                for line in region.get_TextLine():
                    score = line.get_Coords().get_conf() or 1.0
                    polygon = coordinates_of_segment(line, page_image, page_coords)
                    if len(polygon) < 3:
                        LOG.warning('ignoring line "%s" with only %d points', line.id, len(polygon))
                        continue
                    contexts = get_context(line)
                    targets = get_target(line)
                    for i, cat in enumerate(FIELDS[1:], 1):
                        if cat in contexts:
                            add_annotation(annotations, num_page_id, i, CTXT_CATEGORY, polygon, score)
                        if cat in targets:
                            add_annotation(annotations, num_page_id, i, i, polygon, score)
                    for word in line.get_Word():
                        score = word.get_Coords().get_conf() or 1.0
                        polygon = coordinates_of_segment(word, page_image, page_coords)
                        if len(polygon) < 3:
                            LOG.warning('ignoring word "%s" with only %d points', word.id, len(polygon))
                            continue
                        contexts = get_context(word)
                        targets = get_target(word)
                        for i, cat in enumerate(FIELDS[1:], 1):
                            if cat in contexts:
                                add_annotation(annotations, num_page_id, i, CTXT_CATEGORY, polygon, score)
            # annotate cropped+deskewed base image
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
        
        # write COCO JSON for all pages and all categories
        for i, cat in enumerate(FIELDS[1:], 1):
            file_id = json_file_grp + f'_{i:02d}.' + cat + '.coco'
            LOG.info('Writing COCO result file "%s" in "%s"', file_id, json_file_grp)
            self.workspace.add_file(
                ID='id' + file_id,
                file_grp=json_file_grp,
                pageId=None,
                local_filename=os.path.join(json_file_grp, file_id + '.json'),
                mimetype='application/json',
                content=json.dumps(
                    {'categories': categories[i],
                     'images': images,
                     'annotations': annotations[i]},
                    indent=2))

def add_annotation(annotations, image_id, cat, anncat, polygon, score):
    xywh = xywh_from_polygon(polygon.tolist())
    # convert to COCO and add
    i = 1 + len(annotations[cat])
    annotations[cat].append(
        {'id': i, 'image_id': image_id,
         'category_id': anncat,
         'segmentation': polygon.reshape(1, -1).tolist(),
         'area': Polygon(polygon).area,
         'bbox': [xywh['x'], xywh['y'], xywh['w'], xywh['h']],
         'score': score,
         'iscrowd': 0})

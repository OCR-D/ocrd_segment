from __future__ import absolute_import

import os.path
import json
import logging
import numpy as np

from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    points_from_polygon,
    MIMETYPE_PAGE,
    membername
)
from ocrd_modelfactory import page_from_file
# pragma pylint: disable=unused-import
# (region types will be referenced indirectly via globals())
from ocrd_models.ocrd_page import (
    CoordsType,
    TextRegionType,
    ImageRegionType,
    MathsRegionType,
    SeparatorRegionType,
    NoiseRegionType,
    to_xml)
from ocrd_models.ocrd_page_generateds import (
    BorderType,
    TableRegionType,
    GraphicRegionType,
    ChartRegionType,
    ChemRegionType,
    LineDrawingRegionType,
    MusicRegionType,
    UnknownRegionType,
    AdvertRegionType,
    CustomRegionType,
    MapRegionType,
    TextTypeSimpleType,
    GraphicsTypeSimpleType,
    ChartTypeSimpleType
)
# pragma pylint: enable=unused-import
from ocrd import Processor

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-from-coco'

class ImportCOCOSegmentation(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(ImportCOCOSegmentation, self).__init__(*args, **kwargs)

    def process(self):
        """Performs region segmentation by reading from COCO annotations.
        
        Open and deserialize the COCO JSON file from the second input file group.
        (It lists region categories/subtypes, file names and segmentations for all pages.)
        
        Open and deserialize each PAGE input file (or generate from image input file)
        from the first input file group. Now find this page in COCO:
        - try to match the PAGE ``imageFilename`` or METS file path matches to some
          COCO ``file_name``, otherwise
        - try to match the numeric part of the METS physical page ID to some
          COCO ``id``, otherwise
        - skip with an error.
        
        Then create and add a region for each ``segmentation``, converting its polygon
        to coordinate points and its COCO category to a region type (and subtype),
        either for a PubLayNet classification or PAGE classification (as produced by
        ocrd-segment-extract-pages), as indicated by ``source``.
        
        Produce a new output file by serialising the resulting hierarchy.
        
        Afterwards, if there are still COCO images left unaccounted for (i.e. without
        corresponding input files), then show a warning.
        """
        LOG = getLogger('processor.ImportCOCOSegmentation')
        # Load JSON
        assert_file_grp_cardinality(self.input_file_grp, 2, 'base and COCO')
        # pylint: disable=attribute-defined-outside-init
        self.input_file_grp, coco_grp = self.input_file_grp.split(',')
        # pylint: disable=attribute-defined-outside-init
        if not self.input_files:
            LOG.warning('No input files to process')
            return
        if coco_grp in self.workspace.mets.file_groups:
            try:
                cocofile = next(f for f in self.workspace.mets.find_files(fileGrp=coco_grp)
                                # if f.mimetype == 'application/json' and not f.pageId
                                if not f.pageId)
            except StopIteration:
                raise Exception("no non-page-specific file in second file group (COCO file)", coco_grp)
            cocofile = self.workspace.download_file(cocofile).local_filename
        elif os.path.isfile(coco_grp):
            cocofile = coco_grp
        else:
            raise Exception("file not found in second file group (COCO file)", coco_grp)
        
        LOG.info('Loading COCO annotations from "%s" into memory...', cocofile)
        with open(cocofile, 'r') as inp:
            coco = json.load(inp)
        LOG.info('Loaded JSON for %d images with %d regions in %d categories',
                 len(coco['images']), len(coco['annotations']), len(coco['categories']))
        coco_source = 'PubLayNet'
        # Convert to usable dicts
        # classes:
        categories = dict()
        subcategories = dict()
        for cat in coco['categories']:
            if cat['source'] == 'PAGE':
                coco_source = 'PAGE'
            if 'supercategory' in cat and cat['supercategory']:
                categories[cat['id']] = cat['supercategory']
                subcategories[cat['id']] = cat['name']
            else:
                categories[cat['id']] = cat['name']
        # images and annotations:
        images_by_id = dict()
        images_by_filename = dict()
        for image in coco['images']:
            images_by_id[image['id']] = image
            images_by_filename[image['file_name']] = image
        for annotation in coco['annotations']:
            image = images_by_id[annotation['image_id']]
            regions = image.setdefault('regions', list())
            regions.append(annotation)
        del coco
        
        LOG.info('Converting %s annotations into PAGE-XML', coco_source)
        for n, input_file in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            num_page_id = int(page_id.strip(page_id.strip("0123456789")))
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            page = pcgts.get_Page()

            # find COCO image
            if page.imageFilename in images_by_filename:
                image = images_by_filename[page.imageFilename]
            elif num_page_id in images_by_id:
                image = images_by_id[num_page_id]
            else:
                LOG.error('Page "%s" / file "%s" not found in COCO',
                          page_id, page.imageFilename)
                # todo: maybe we should at least write the (unchanged) output PAGE?
                continue
            if image['width'] != page.imageWidth:
                LOG.error('Page "%s" width %d does not match annotated width %d',
                          page_id, page.imageWidth, image['width'])
            if image['height'] != page.imageHeight:
                LOG.error('Page "%s" height %d does not match annotated height %d',
                          page_id, page.imageHeight, image['height'])

            # todo: remove existing segmentation first?
            for region in image['regions']:
                assert isinstance(region['segmentation'], list), "importing RLE/mask segmentation not implemented"
                polygon = np.array(region['segmentation'])
                polygon = np.reshape(polygon, (polygon.shape[1]//2, 2))
                coords = CoordsType(points=points_from_polygon(polygon))
                category = categories[region['category_id']]
                if region['category_id'] in subcategories:
                    subcategory = subcategories[region['category_id']]
                else:
                    subcategory = None
                region_id = 'r' + str(region['id'])
                LOG.info('Adding region %s:%s [area %d]', category, subcategory or '', region['area'])
                if coco_source == 'PubLayNet':
                    if category == 'text':
                        region_obj = TextRegionType(id=region_id, Coords=coords,
                                                    type_=TextTypeSimpleType.PARAGRAPH)
                        page.add_TextRegion(region_obj)
                    elif category == 'title':
                        region_obj = TextRegionType(id=region_id, Coords=coords,
                                                    type_=TextTypeSimpleType.HEADING) # CAPTION?
                        page.add_TextRegion(region_obj)
                    elif category == 'list':
                        region_obj = TextRegionType(id=region_id, Coords=coords,
                                                    type_=TextTypeSimpleType.LISTLABEL) # OTHER?
                        page.add_TextRegion(region_obj)
                    elif category == 'table':
                        region_obj = TableRegionType(id=region_id, Coords=coords)
                        page.add_TableRegion(region_obj)
                    elif category == 'figure':
                        region_obj = ImageRegionType(id=region_id, Coords=coords)
                        page.add_ImageRegion(region_obj)
                    else:
                        raise Exception('unknown region category: %s' % category)
                else: # 'PAGE'
                    args = {'id': region_id,
                            'Coords': coords}
                    if subcategory:
                        typedict = {"TextRegion": TextTypeSimpleType,
                                    "GraphicRegion": GraphicsTypeSimpleType,
                                    "ChartType": ChartTypeSimpleType}
                        if category in typedict:
                            subtype = membername(typedict[category], subcategory)
                            if subtype == subcategory:
                                # not predefined in PAGE: use other + custom
                                args['custom'] = "subtype:%s" % subcategory
                                args['type_'] = "other"
                            else:
                                args['type_'] = subcategory
                        else:
                            args['custom'] = "subtype:%s" % subcategory
                    if category + 'Type' not in globals():
                        raise Exception('unknown region category: %s' % category)
                    region_type = globals()[category + 'Type']
                    if region_type is BorderType:
                        page.set_Border(BorderType(Coords=coords))
                    else:
                        region_obj = region_type(**args)
                        getattr(page, 'add_%s' % category)(region_obj)
            # remove image from dicts
            images_by_id.pop(num_page_id, None)
            images_by_filename.pop(page.imageFilename, None)

            file_id = make_file_id(input_file, self.output_file_grp)
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(self.output_file_grp, file_id + '.xml'),
                content=to_xml(pcgts))
        
        # warn of remaining COCO images
        if images_by_filename and not self.page_id:
            LOG.warning('%d images remain unaccounted for after processing', len(images_by_filename))
            if LOG.isEnabledFor(logging.DEBUG):
                for filename in images_by_filename:
                    LOG.debug('not found in workspace: "%s"', filename)

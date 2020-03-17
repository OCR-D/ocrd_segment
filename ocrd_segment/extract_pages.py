from __future__ import absolute_import

import json
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from ocrd_utils import (
    getLogger, concat_padded,
    coordinates_of_segment,
    xywh_from_polygon,
    MIME_TO_EXT
)
from ocrd_models.ocrd_page import (
    LabelsType, LabelType,
    MetadataItemType
)
from ocrd_models.ocrd_page_generateds import (
    TextTypeSimpleType,
    GraphicsTypeSimpleType,
    ChartTypeSimpleType
)    
from ocrd_modelfactory import page_from_file
from ocrd import Processor

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-extract-pages'
LOG = getLogger('processor.ExtractPages')
# region classes and their colours in debug images:
CLASSES = {'border': (255, 255, 255),
           'text': (200, 0, 0),
           'table': (0, 100, 20),
           'chart': (0, 120, 0),
           'chem': (0, 140, 0),
           'graphic': (0, 0, 200),
           'image': (0, 20, 180),
           'linedrawing': (20, 0, 180),
           'maths': (20, 120, 0),
           'music': (20, 120, 40),
           'noise': (50, 50, 50),
           'separator': (0, 0, 100),
           'unknown': (0, 0, 0)
}

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
        * in the directory of the first (or only) output file group:
          - ID + '.png': raw image of the (preprocessed) page
          - ID + '.json': region coordinates/classes (custom format)
        * in the directory of the second (or first) output file group:
          - ID + '.bin.png': binarized image of the (preprocessed) page, if available
        * in the directory of the third (or first) output file group:
          - ID + '.dbg.png': debug image
        
        In addition, write a file for all pages in the parent directory, named like so:
        * output_file_grp + '.coco.json': region coordinates/classes (MS-COCO)
        
        (This is intended for training and evaluation of region segmentation models.)
        """
        file_groups = self.output_file_grp.split(',')
        if len(file_groups) > 3:
            raise Exception("at most 3 output file grps allowed (raw, [binarized, [debug]] image)")
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
            categories.append(
                {'id': i, 'name': cat, 'supercategory': '',
                 'source': 'PAGE', 'color': color})
            i += 1
        typedict = {"text": TextTypeSimpleType,
                    "graphic": GraphicsTypeSimpleType,
                    "chart": ChartTypeSimpleType}
        i = len(categories)
        SUPERCLASSES = dict()
        for supercat, class_ in typedict.items():
            j = i
            for name, cat in vars(class_).items():
                if name[0] != '_':
                    color = list(CLASSES[supercat])
                    for c in range(3):
                        if not color[c]:
                            color[c] = (i-j+1) * 5
                    SUPERCLASSES[(cat, supercat)] = tuple(color)
                    categories.append(
                        {'id': i, 'name': cat, 'supercategory': supercat,
                         'source': 'PAGE', 'color': color})
                    i += 1

        i = 0
        # pylint: disable=attribute-defined-outside-init
        for n, input_file in enumerate(self.input_files):
            file_id = input_file.ID.replace(self.input_file_grp, self.output_file_grp)
            page_id = input_file.pageId or input_file.ID
            num_page_id = int(page_id.strip(page_id.strip("0123456789")))
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page = pcgts.get_Page()
            ptype = page.get_type()
            metadata = pcgts.get_Metadata() # ensured by from_file()
            metadata.add_MetadataItem(
                MetadataItemType(type_="processingStep",
                                 name=self.ocrd_tool['steps'][0],
                                 value=TOOL,
                                 Labels=[LabelsType(
                                     externalModel="ocrd-tool",
                                     externalId="parameters",
                                     Label=[LabelType(type_=name,
                                                      value=self.parameter[name])
                                            for name in self.parameter.keys()])]))
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
            page_image_dbg = Image.new(mode='RGB', size=page_image.size, color=0)
            regions = { 'text': page.get_TextRegion(),
                        'table': page.get_TableRegion(),
                        'chart': page.get_ChartRegion(),
                        'chem': page.get_ChemRegion(),
                        'graphic': page.get_GraphicRegion(),
                        'image': page.get_ImageRegion(),
                        'linedrawing': page.get_LineDrawingRegion(),
                        'maths': page.get_MathsRegion(),
                        'music': page.get_MusicRegion(),
                        'noise': page.get_NoiseRegion(),
                        'separator': page.get_SeparatorRegion(),
                        'unknown': page.get_UnknownRegion()
            }
            description = { 'angle': page.get_orientation() }
            for rtype, rlist in regions.items():
                for region in rlist:
                    subrtype = region.get_type() if rtype in ['text', 'chart', 'graphic'] else None
                    polygon = coordinates_of_segment(
                        region, page_image, page_coords)
                    polygon2 = polygon.reshape(1,-1).tolist()
                    polygon = polygon.tolist()
                    xywh = xywh_from_polygon(polygon)
                    area = Polygon(polygon).area
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
                        fill=SUPERCLASSES.get((subrtype, rtype), CLASSES.get(rtype)))
                    # draw hull:
                    #ImageDraw.Draw(page_image_dbg).line(
                    #    list(map(tuple, polygon + [polygon[0]])), 
                    #    fill=CLASSES['border'], width=3)
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
            file_path = file_path.replace(MIME_TO_EXT[self.parameter['mimetype']], '.json')
            with open(file_path, 'w') as out:
                json.dump(description, out)

            # COCO: add image
            images.append(
                {'id': num_page_id, 'file_name': page.imageFilename,
                 'width': page_image.width, 'height': page_image.height})
        
        # COCO: write result
        with open(self.output_file_grp + '.coco.json', 'w') as coco:
            json.dump({'categories': categories,
                       'images': images,
                       'annotations': annotations},
                      coco)

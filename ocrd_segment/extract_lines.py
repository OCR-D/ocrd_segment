from __future__ import absolute_import

import json
import itertools

from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    coordinates_of_segment,
    polygon_from_points,
    MIME_TO_EXT
)
from ocrd_modelfactory import page_from_file
from ocrd import Processor

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-extract-lines'
LOG = getLogger('processor.ExtractLines')

class ExtractLines(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(ExtractLines, self).__init__(*args, **kwargs)

    def process(self):
        """Extract textline images and texts from the workspace.
        
        Open and deserialize PAGE input files and their respective images,
        then iterate over the element hierarchy down to the line level.
        
        Extract an image for each textline (which depending on the workflow
        can already be deskewed, dewarped, binarized etc.), cropped to its
        minimal bounding box, and masked by the coordinate polygon outline.
        If ``transparency`` is true, then also add an alpha channel which is
        fully transparent outside of the mask.
        
        Create a JSON file with:
        * the IDs of the textline and its parents,
        * the textline's text content,
        * the textline's coordinates relative to the line image,
        * the textline's absolute coordinates,
        * the textline's TextStyle (if any),
        * the textline's @production (if any),
        * the textline's @readingDirection (if any),
        * the textline's @primaryScript (if any),
        * the textline's @primaryLanguage (if any),
        * the textline's AlternativeImage/@comments (features),
        * the parent textregion's @type,
        * the page's @type,
        * the page's DPI value.
        
        Create a plain text file for the text content, too.
        
        Write all files in the directory of the output file group, named like so:
        * ID + '.raw.png': line image (if the workflow provides raw images)
        * ID + '.bin.png': line image (if the workflow provides binarized images)
        * ID + '.nrm.png': line image (if the workflow provides grayscale-normalized images)
        * ID + '.json': line metadata.
        * ID + '.gt.txt': line text.
        
        (This is intended for training and evaluation of OCR models.)
        """
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)
        # pylint: disable=attribute-defined-outside-init
        for n, input_file in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            page = pcgts.get_Page()
            page_image, page_coords, page_image_info = self.workspace.image_from_page(
                page, page_id,
                transparency=self.parameter['transparency'])
            if page_image_info.resolution != 1:
                dpi = page_image_info.resolution
                if page_image_info.resolutionUnit == 'cm':
                    dpi = round(dpi * 2.54)
            else:
                dpi = None
            ptype = page.get_type()
            
            regions = itertools.chain.from_iterable(
                [page.get_TextRegion()] +
                [subregion.get_TextRegion() for subregion in page.get_TableRegion()])
            if not regions:
                LOG.warning("Page '%s' contains no text regions", page_id)
            for region in regions:
                region_image, region_coords = self.workspace.image_from_segment(
                    region, page_image, page_coords,
                    transparency=self.parameter['transparency'])
                rtype = region.get_type()
                
                lines = region.get_TextLine()
                if not lines:
                    LOG.warning("Region '%s' contains no text lines", region.id)
                for line in lines:
                    line_image, line_coords = self.workspace.image_from_segment(
                        line, region_image, region_coords,
                        transparency=self.parameter['transparency'])
                    lpolygon_rel = coordinates_of_segment(
                        line, line_image, line_coords).tolist()
                    lpolygon_abs = polygon_from_points(line.get_Coords().points)
                    ltext = line.get_TextEquiv()
                    if not ltext:
                        LOG.warning("Line '%s' contains no text content", line.id)
                        ltext = ''
                    else:
                        ltext = ltext[0].Unicode
                    lstyle = line.get_TextStyle() or region.get_TextStyle()
                    if lstyle:
                        lstyle = {
                            'fontFamily': lstyle.fontFamily,
                            'fontSize': lstyle.fontSize,
                            'xHeight': lstyle.xHeight,
                            'kerning': lstyle.kerning,
                            'serif': lstyle.serif,
                            'monospace': lstyle.monospace,
                            'bold': lstyle.bold,
                            'italic': lstyle.italic,
                            'smallCaps': lstyle.smallCaps,
                            'letterSpaced': lstyle.letterSpaced,
                            'strikethrough': lstyle.strikethrough,
                            'underlined': lstyle.underlined,
                            'underlineStyle': lstyle.underlineStyle,
                            'subscript': lstyle.subscript,
                            'superscript': lstyle.superscript
                        }
                    lfeatures = line_coords['features']
                    description = { 'line.ID': line.id,
                                    'text': ltext,
                                    'style': lstyle,
                                    'production': (
                                        line.get_production() or
                                        region.get_production()),
                                    'readingDirection': (
                                        line.get_readingDirection() or
                                        region.get_readingDirection() or
                                        page.get_readingDirection()),
                                    'primaryScript': (
                                        line.get_primaryScript() or
                                        region.get_primaryScript() or
                                        page.get_primaryScript()),
                                    'primaryLanguage': (
                                        line.get_primaryLanguage() or
                                        region.get_primaryLanguage() or
                                        page.get_primaryLanguage()),
                                    'features': lfeatures,
                                    'DPI': dpi,
                                    'coords_rel': lpolygon_rel,
                                    'coords_abs': lpolygon_abs,
                                    'region.ID': region.id,
                                    'region.type': rtype,
                                    'page.ID': page_id,
                                    'page.type': ptype,
                                    'file_grp': self.input_file_grp,
                                    'METS.UID': self.workspace.mets.unique_identifier
                    }
                    if 'binarized' in lfeatures:
                        extension = '.bin'
                    elif 'grayscale_normalized' in lfeatures:
                        extension = '.nrm'
                    else:
                        extension = '.raw'

                    file_id = make_file_id(input_file, self.output_file_grp)
                    file_path = self.workspace.save_image_file(
                        line_image,
                        file_id + '_' + region.id + '_' + line.id + extension,
                        self.output_file_grp,
                        page_id=page_id,
                        mimetype=self.parameter['mimetype'])
                    file_path = file_path.replace(extension + MIME_TO_EXT[self.parameter['mimetype']], '.json')
                    json.dump(description, open(file_path, 'w'))
                    file_path = file_path.replace('.json', '.gt.txt')
                    with open(file_path, 'wb') as f:
                        f.write((ltext + '\n').encode('utf-8'))

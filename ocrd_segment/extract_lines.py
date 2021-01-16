from __future__ import absolute_import

import os
import json
import itertools
import xlsxwriter

from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    coordinates_of_segment,
    polygon_from_points,
    MIME_TO_EXT
)
from ocrd_models.constants import NAMESPACES
from ocrd_modelfactory import page_from_file
from ocrd import Processor

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-extract-lines'

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
        LOG = getLogger('processor.ExtractLines')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        library_convention = self.parameter['library-convention']
        min_line_length = self.parameter['min-line-length']
        min_line_width = self.parameter['min-line-width']
        min_line_height = self.parameter['min-line-height']

        # pylint: disable=attribute-defined-outside-init
        for n, input_file in enumerate(self.input_files):
            file_id = make_file_id(input_file, self.output_file_grp)
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

            # add excel file
            LOG.info('Writing Excel result file "%s.xlsx" in "%s"', file_id, self.output_file_grp)
            excel_path = '%s.xlsx' % os.path.join(self.output_file_grp, file_id)
            if not os.path.isdir(self.output_file_grp):
                os.mkdir(self.output_file_grp)
            workbook = xlsxwriter.Workbook(excel_path)
            worksheet = workbook.add_worksheet()
            bold = workbook.add_format({'bold': True})
            normal = workbook.add_format({'valign': 'top'})
            worksheet.set_default_row(height=40)
            worksheet.freeze_panes(1, 0)
            worksheet.write('A1', 'ID', bold)
            worksheet.write('B1', 'Text', bold)
            worksheet.write('C1', 'Status', bold)
            worksheet.write('D1', 'Image', bold)
            symbols = 'ſ ꝛ aͤ oͤ uͤ æ œ Æ Œ ℳ  ç ę ë - ⸗ = Α α Β β ϐ Γ γ Δ δ Ε ε ϵ Ζ ζ Η η Θ θ ϑ Ι ι ' \
                      'Κ κ ϰ Λ λ Μ μ Ν ν Ξ ξ Ο ο Π π ϖ Ρ ρ ϱ Σ σ ς ϲ Τ τ Υ υ ϒ Φ φ ϕ Χ χ Ψ ψ Ω ω'.split(' ')
            for i, s in enumerate(symbols):
                col_idx = 3 + i
                worksheet.write_string(0, col_idx, s)
                worksheet.set_column(col_idx, col_idx, 2)
            self.workspace.add_file(
                ID=file_id,
                mimetype='application/vnd.ms-excel',
                pageId=page_id,
                url=excel_path,
                file_grp=self.output_file_grp,
            )
            url = self._get_presentation_image(input_file, library_convention)
            i = 2
            max_text_length = 0
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

                    file_path = self.workspace.save_image_file(
                        line_image,
                        file_id + '_' + region.id + '_' + line.id + extension,
                        self.output_file_grp,
                        page_id=page_id,
                        mimetype=self.parameter['mimetype'])

                    # plausibilize and modify excel
                    if (min_line_length < 0 or len(ltext) > min_line_length) and \
                       (min_line_width < 0 or line_image.width > min_line_width) and \
                       (min_line_height < 0 or line_image.height > min_line_height):
                        scale = 40.0 / line_image.height
                        worksheet.write('A%d' % i, file_id + '_' + region.id + '_' + line.id, normal)
                        if len(ltext) > max_text_length:
                            max_text_length = len(ltext)
                            worksheet.set_column('B:B', max_text_length)
                        worksheet.write('B%d' % i, ltext, normal)
                        worksheet.data_validation('C%d' %i, {'validate': 'list', 'source': ['ToDo', 'Done', 'Error']})
                        worksheet.insert_image('D%d' % i, file_path, {
                            'object_position': 1, 'url': url, 'y_scale': scale, 'x_scale': scale})

                    file_path = file_path.replace(extension + MIME_TO_EXT[self.parameter['mimetype']], '.json')
                    json.dump(description, open(file_path, 'w'))
                    file_path = file_path.replace('.json', '.gt.txt')
                    with open(file_path, 'wb') as f:
                        f.write((ltext + '\n').encode('utf-8'))
                    i += 1

            workbook.close()

    def _get_presentation_image(self, input_file, library_convention):
        if library_convention == 'slub':
            return self._get_presentation_image_slub(input_file)
        elif library_convention == 'sbb':
            return self._get_presentation_image_sbb(input_file)
        raise NotImplementedError("Unsupported library convention '%s'" % library_convention)

    def _get_presentation_image_sbb(self, input_file):
        ppn = self.workspace.mets._tree.getroot().xpath('//mods:recordIdentifier[@source="gbv-ppn"]', namespaces=NAMESPACES)[0].text
        pageId = input_file.pageId
        return f'https://digital.staatsbibliothek-berlin.de/werkansicht?PPN={ppn}&PHYSID={pageId}'

    def _get_presentation_image_slub(self, input_file):
        # get Kitodo.Presentation image URL
        url = self.workspace.mets._tree.getroot().xpath(
            '//mets:structMap[@TYPE="LOGICAL"]/mets:div/mets:mptr/@xlink:href',
            namespaces=NAMESPACES)
        NAMESPACES.update({'slub': 'http://slub-dresden.de/'})
        slub = self.workspace.mets._tree.getroot().xpath(
            '//mods:mods/mods:extension/slub:slub',
            namespaces=NAMESPACES)
        if input_file.pageId.startswith('PHYS_'):
            base = '_tif/jpegs/0000' + input_file.pageId[5:] + '.tif.original.jpg'
            if url and url[0].endswith('_anchor.xml'):
                url = url[0]
                url = url[:url.rindex('_anchor.xml')]
                url += base
            elif slub:
                digital = slub[0].xpath('slub:id[@type="digital"]', namespaces=NAMESPACES)
                ats = slub[0].xpath('slub:id[@type="tsl-ats"]', namespaces=NAMESPACES)
                if digital and ats:
                    url = ats[0].text + '_' + digital[0].text
                    url = 'https://digital.slub-dresden.de/data/kitodo/' + url + '/' + url + base
                else:
                    url = ''
        else:
            url = ''
        return url

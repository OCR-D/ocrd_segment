from __future__ import absolute_import

import json

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
from .extract_pages import CLASSES

TOOL = 'ocrd-segment-extract-regions'

class ExtractRegions(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(ExtractRegions, self).__init__(*args, **kwargs)

    def process(self):
        """Extract region images from the workspace.
        
        Open and deserialize PAGE input files and their respective images,
        then iterate over the element hierarchy down to the region level.
        
        Extract an image for each region (which depending on the workflow
        can already be deskewed, dewarped, binarized etc.), cropped to its
        minimal bounding box, and masked by the coordinate polygon outline.
        Apply ``feature_filter`` (a comma-separated list of image features,
        cf. :py:func:`ocrd.workspace.Workspace.image_from_page`) to skip
        specific features when retrieving derived images.
        If ``transparency`` is true, then also add an alpha channel which is
        fully transparent outside of the mask.
        
        Create a JSON file with:
        * the IDs of the region and its parents,
        * the region's coordinates relative to the region image,
        * the region's absolute coordinates,
        * the (text) region's text content (if any),
        * the (text) region's TextStyle (if any),
        * the (text) region's @production (if any),
        * the (text) region's @readingDirection (if any),
        * the (text) region's @textLineOrder (if any),
        * the (text) region's @primaryScript (if any),
        * the (text) region's @primaryLanguage (if any),
        * the region's AlternativeImage/@comments (features),
        * the region's element class,
        * the region's @type,
        * the page's @type,
        * the page's DPI value.
        
        Write all files in the directory of the output file group, named like so:
        * ID + '.raw.png': region image (if the workflow provides raw images)
        * ID + '.bin.png': region image (if the workflow provides binarized images)
        * ID + '.nrm.png': region image (if the workflow provides grayscale-normalized images)
        * ID + '.json': region metadata.
        """
        LOG = getLogger('processor.ExtractRegions')
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
                feature_filter=self.parameter['feature_filter'],
                transparency=self.parameter['transparency'])
            if page_image_info.resolution != 1:
                dpi = page_image_info.resolution
                if page_image_info.resolutionUnit == 'cm':
                    dpi = round(dpi * 2.54)
            else:
                dpi = None
            ptype = page.get_type()

            regions = dict()
            for name in CLASSES.keys():
                if not name or name == 'Border' or ':' in name:
                    # no subtypes here
                    continue
                regions[name] = getattr(page, 'get_' + name)()
            for rtype, rlist in regions.items():
                for region in rlist:
                    description = {'region.ID': region.id, 'region.type': rtype}
                    region_image, region_coords = self.workspace.image_from_segment(
                        region, page_image, page_coords,
                        transparency=self.parameter['transparency'])
                    if not region_image.width or not region_image.height:
                        LOG.error("ignoring zero-size region '%s'", region.id)
                        continue
                    if rtype in ['TextRegion', 'ChartRegion', 'GraphicRegion']:
                        subrtype = region.get_type()
                    else:
                        subrtype = None
                    description['subtype'] = subrtype
                    description['coords_rel'] = coordinates_of_segment(
                        region, region_image, region_coords).tolist()
                    description['coords_abs'] = polygon_from_points(region.get_Coords().points)
                    if rtype == 'text':
                        rtext = region.get_TextEquiv()
                        if rtext:
                            description['region.text'] = rtext[0].Unicode
                        else:
                            description['region.text'] = ''
                        rstyle = region.get_TextStyle() or page.get_TextStyle()
                        if rstyle:
                            description['region.style'] = {
                                'fontFamily': rstyle.fontFamily,
                                'fontSize': rstyle.fontSize,
                                'xHeight': rstyle.xHeight,
                                'kerning': rstyle.kerning,
                                'serif': rstyle.serif,
                                'monospace': rstyle.monospace,
                                'bold': rstyle.bold,
                                'italic': rstyle.italic,
                                'smallCaps': rstyle.smallCaps,
                                'letterSpaced': rstyle.letterSpaced,
                                'strikethrough': rstyle.strikethrough,
                                'underlined': rstyle.underlined,
                                'underlineStyle': rstyle.underlineStyle,
                                'subscript': rstyle.subscript,
                                'superscript': rstyle.superscript
                            }
                        description['production'] = region.get_production()
                        description['readingDirection'] = (
                            region.get_readingDirection() or
                            page.get_readingDirection())
                        description['textLineOrder'] = (
                            region.get_textLineOrder() or
                            page.get_textLineOrder())
                        description['primaryScript'] = (
                            region.get_primaryScript() or
                            page.get_primaryScript())
                        description['primaryLanguage'] = (
                            region.get_primaryLanguage() or
                            page.get_primaryLanguage())
                    description['features'] = region_coords['features']
                    description['DPI'] = dpi
                    description['page.ID'] = page_id
                    description['page.type'] = ptype
                    description['file_grp'] = self.input_file_grp
                    description['METS.UID'] = self.workspace.mets.unique_identifier
                    if 'binarized' in region_coords['features']:
                        extension = '.bin'
                    elif 'grayscale_normalized' in region_coords['features']:
                        extension = '.nrm'
                    else:
                        extension = '.raw'

                    file_id = make_file_id(input_file, self.output_file_grp) + '_' + region.id + extension
                    file_path = self.workspace.save_image_file(
                        region_image,
                        file_id,
                        self.output_file_grp,
                        pageId=input_file.pageId,
                        mimetype=self.parameter['mimetype'])
                    self.workspace.add_file(
                        ID=file_id + '.json',
                        file_grp=self.output_file_grp,
                        local_filename=file_path.replace(extension + MIME_TO_EXT[self.parameter['mimetype']], '.json'),
                        pageId=input_file.pageId,
                        mimetype='application/json',
                        content=json.dumps(description))

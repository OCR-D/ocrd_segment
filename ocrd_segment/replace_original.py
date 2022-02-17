from __future__ import absolute_import

import os.path

from ocrd_utils import (
    getLogger, concat_padded,
    coordinates_of_segment,
    points_from_polygon,
    make_file_id,
    assert_file_grp_cardinality,
    MIMETYPE_PAGE
)
from ocrd_models.ocrd_page import (
    AlternativeImageType,
    TextRegionType,
    to_xml
)
from ocrd_modelfactory import page_from_file
from ocrd import Processor

from .config import OCRD_TOOL
from .repair import ensure_valid

TOOL = 'ocrd-segment-replace-original'

class ReplaceOriginal(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(ReplaceOriginal, self).__init__(*args, **kwargs)

    def process(self):
        """Extract page image and replace original with it.
        
        Open and deserialize PAGE input files and their respective images,
        then go to the page hierarchy level.
        
        Retrieve the image of the (cropped, deskewed, dewarped) page, preferring
        the last annotated form (which, depending on the workflow, could be
        binarized or raw). Add that image file to the workspace with the fileGrp
        USE given in the output fileGrp.
        Reference that file in the page (not as AlternativeImage but) as original
        image. Adjust all segment coordinates accordingly.
        
        Produce a new output file by serialising the resulting hierarchy.
        """
        LOG = getLogger('processor.ReplaceOriginal')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)
        feature_selector = self.parameter['feature_selector']
        feature_filter = self.parameter['feature_filter']
        adapt_coords = self.parameter['transform_coordinates']
        
        # pylint: disable=attribute-defined-outside-init
        for n, input_file in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            page = pcgts.get_Page()
            page_image, page_coords, page_image_info = self.workspace.image_from_page(
                page, page_id,
                feature_filter=feature_filter,
                feature_selector=feature_selector)
            if page_image_info.resolution != 1:
                dpi = page_image_info.resolution
                if page_image_info.resolutionUnit == 'cm':
                    dpi = round(dpi * 2.54)
            else:
                dpi = None
            # annotate extracted image
            file_id = make_file_id(input_file, self.output_file_grp)
            file_path = self.workspace.save_image_file(page_image,
                                                       file_id + '-IMG',
                                                       self.output_file_grp,
                                                       page_id=input_file.pageId,
                                                       mimetype='image/png')
            # replace original image
            page.set_imageFilename(file_path)
            # remove all coordinate-sensitive page-level annotations
            page.set_imageWidth(page_image.width)
            page.set_imageHeight(page_image.height)
            page.set_Border(None) # also removes all derived images
            page.set_orientation(None)
            # also add image as derived image (in order to preserve image features)
            # (but exclude coordinate-sensitive features that have already been applied over the "original")
            features = ','.join(filter(lambda f: f not in [
                "cropped", "deskewed", "rotated-90", "rotated-180", "rotated-270"],
                                       page_coords['features'].split(",")))
            page.add_AlternativeImage(AlternativeImageType(
                filename=file_path, comments=features))
            # adjust all coordinates
            if adapt_coords:
                for region in page.get_AllRegions():
                    region_polygon = coordinates_of_segment(region, page_image, page_coords)
                    region.get_Coords().set_points(points_from_polygon(region_polygon))
                    ensure_valid(region)
                    if isinstance(region, TextRegionType):
                        for line in region.get_TextLine():
                            line_polygon = coordinates_of_segment(line, page_image, page_coords)
                            line.get_Coords().set_points(points_from_polygon(line_polygon))
                            ensure_valid(line)
                            for word in line.get_Word():
                                word_polygon = coordinates_of_segment(word, page_image, page_coords)
                                word.get_Coords().set_points(points_from_polygon(word_polygon))
                                ensure_valid(word)
                                for glyph in word.get_Glyph():
                                    glyph_polygon = coordinates_of_segment(glyph, page_image, page_coords)
                                    glyph.get_Coords().set_points(points_from_polygon(glyph_polygon))
                                    ensure_valid(glyph)

            # update METS (add the PAGE file):
            out = self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                local_filename=os.path.join(self.output_file_grp, file_id + '.xml'),
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts))
            LOG.info('created file ID: %s, file_grp: %s, path: %s',
                     file_id, self.output_file_grp, out.local_filename)

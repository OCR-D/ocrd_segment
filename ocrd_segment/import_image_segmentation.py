from __future__ import absolute_import

import os.path
from PIL import Image
import numpy as np
import cv2

from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    points_from_polygon,
    MIMETYPE_PAGE,
    pushd_popd,
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
    TextTypeSimpleType,
    GraphicsTypeSimpleType,
    ChartTypeSimpleType
)
# pragma pylint: enable=unused-import
from ocrd import Processor

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-from-masks'

class ImportImageSegmentation(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(ImportImageSegmentation, self).__init__(*args, **kwargs)

    def process(self):
        """Performs region segmentation by reading mask images in pseudo-colour.
        
        Open and deserialize each PAGE input file (or generate from image input file)
        from the first input file group, as well as mask image file from the second.
        
        Then iterate over all connected (equally colored) mask segments and compute
        convex hull contours for them. Convert them to polygons, and look up their
        color value in ``colordict`` to instantiate the appropriate region types
        (optionally with subtype). Instantiate and annotate regions accordingly.
        
        Produce a new output file by serialising the resulting hierarchy.
        """
        LOG = getLogger('processor.ImportImageSegmentation')
        assert_file_grp_cardinality(self.input_file_grp, 2, 'base and mask')
        assert_file_grp_cardinality(self.output_file_grp, 1)

        colordict = self.parameter['colordict']
        typedict = {"TextRegion": TextTypeSimpleType,
                    "GraphicRegion": GraphicsTypeSimpleType,
                    "ChartType": ChartTypeSimpleType}
        # collect input file tuples
        ifts = self.zip_input_files() # input file tuples
        # process input file tuples
        for ift in ifts:
            input_file, segmentation_file = ift
            LOG.info("processing page %s", input_file.pageId)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            page = pcgts.get_Page()

            # import mask image
            segmentation_filename = self.workspace.download_file(segmentation_file).local_filename
            with pushd_popd(self.workspace.directory):
                segmentation_pil = Image.open(segmentation_filename)
            has_alpha = segmentation_pil.mode == 'RGBA'
            if has_alpha:
                colorformat = "%08X"
            else:
                colorformat = "%06X"
                if segmentation_pil.mode != 'RGB':
                    segmentation_pil = segmentation_pil.convert('RGB')
            # convert to array
            segmentation_array = np.array(segmentation_pil)
            # collapse 3 color channels
            segmentation_array = segmentation_array.dot(
                np.array([2**24, 2**16, 2**8, 1], np.uint32)[0 if has_alpha else 1:])
            # partition mapped colors vs background
            colors = np.unique(segmentation_array)
            bgcolors = []
            for i, color in enumerate(colors):
                colorname = colorformat % color
                if (colorname not in colordict or
                    not colordict[colorname]):
                    #raise Exception("Unknown color %s (not in colordict)" % colorname)
                    LOG.info("Ignoring background color %s", colorname)
                    bgcolors.append(i)
            background = np.zeros_like(segmentation_array, np.uint8)
            if bgcolors:
                for i in bgcolors:
                    background += np.array(segmentation_array == colors[i], np.uint8)
                colors = np.delete(colors, bgcolors, 0)
            # iterate over mask for each mapped color/class
            regionno = 0
            for color in colors:
                # get region (sub)type
                colorname = colorformat % color
                classname = colordict[colorname]
                regiontype = None
                custom = None
                if ":" in classname:
                    classname, regiontype = classname.split(":")
                    if classname in typedict:
                        typename = membername(typedict[classname], regiontype)
                        if typename == regiontype:
                            # not predefined in PAGE: use other + custom
                            custom = "subtype:%s" % regiontype
                            regiontype = "other"
                    else:
                        custom = "subtype:%s" % regiontype
                if classname + "Type" not in globals():
                    raise Exception("Unknown class '%s' for color %s in colordict" % (classname, colorname))
                classtype = globals()[classname + "Type"]
                if classtype is BorderType:
                    # mask from all non-background regions
                    classmask = 1 - background
                else:
                    # mask from current color/class
                    classmask = np.array(segmentation_array == color, np.uint8)
                if not np.count_nonzero(classmask):
                    continue
                # now get the contours and make polygons for them
                contours, _ = cv2.findContours(classmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    # (could also just take bounding boxes to avoid islands/inclusions...)
                    area = cv2.contourArea(contour)
                    # filter too small regions
                    area_pct = area / np.prod(segmentation_array.shape) * 100
                    if area < 100 and area_pct < 0.1:
                        LOG.warning('ignoring contour of only %.1f%% area for %s',
                                    area_pct, classname)
                        continue
                    LOG.info('found region %s:%s:%s with area %.1f%%',
                             classname, regiontype or '', custom or '', area_pct)
                    # simplify shape
                    poly = cv2.approxPolyDP(contour, 2, False)[:, 0, ::] # already ordered x,y
                    if len(poly) < 4:
                        LOG.warning('ignoring contour of only %d points (area %.1f%%) for %s',
                                    len(poly), area_pct, classname)
                        continue
                    if classtype is BorderType:
                        # add Border
                        page.set_Border(BorderType(Coords=CoordsType(points=points_from_polygon(poly))))
                        break
                    else:
                        # instantiate region
                        regionno += 1
                        region = classtype(id="region_%d" % regionno, type_=regiontype, custom=custom,
                                           Coords=CoordsType(points=points_from_polygon(poly)))
                        # add region
                        getattr(page, 'add_%s' % classname)(region)

            file_id = make_file_id(input_file, self.output_file_grp)
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(self.output_file_grp,
                                            file_id + '.xml'),
                content=to_xml(pcgts))

from __future__ import absolute_import

import os.path
from PIL import Image
import numpy as np
import cv2

from ocrd_utils import (
    getLogger,
    concat_padded,
    points_from_polygon,
    MIMETYPE_PAGE,
    pushd_popd,
    membername
)
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    MetadataItemType,
    LabelsType, LabelType,
    CoordsType, AlternativeImageType,
    TextRegionType,
    ImageRegionType,
    MathsRegionType,
    SeparatorRegionType,
    NoiseRegionType,
    to_xml)
from ocrd_models.ocrd_page_generateds import (
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
from ocrd import Processor

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-from-masks'
LOG = getLogger('processor.ImportImageSegmentation')

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
        colordict = self.parameter['colordict']
        typedict = {"TextRegion": TextTypeSimpleType,
                    "GraphicRegion": GraphicsTypeSimpleType,
                    "ChartType": ChartTypeSimpleType}
        ifgs = self.input_file_grp.split(",") # input file groups
        if len(ifgs) != 2:
            raise Exception("need 2 input file groups (base and mask)")
        # collect input file tuples
        ifts = self.zip_input_files(ifgs) # input file tuples
        # process input file tuples
        for n, ift in enumerate(ifts):
            input_file, segmentation_file = ift
            LOG.info("processing page %s", input_file.pageId)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page = pcgts.get_Page()

            # add metadata about this operation and its runtime parameters:
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
            
            segmentation_filename = self.workspace.download_file(segmentation_file).local_filename
            with pushd_popd(self.workspace.directory):
                segmentation_pil = Image.open(segmentation_filename)
            if segmentation_pil.mode != 'RGB':
                segmentation_pil = segmentation_pil.convert('RGB')
            # convert to array
            segmentation_array = np.array(segmentation_pil)
            # collapse 3 color channels
            segmentation_array = segmentation_array.dot(np.array([2**16, 2**8, 1], np.uint32))
            # iterate over mask for each color/class
            regionno = 0
            colors = np.unique(segmentation_array)
            for color in colors:
                if not "#%06X" % color in colordict:
                    #raise Exception("Unknown color #%06X (not in colordict)" % color)
                    LOG.info("Ignoring background color #%06X", color)
                    continue
                # get region (sub)type
                classname = colordict["#%06X" % color]
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
                    raise Exception("Unknown class '%s' for color #%06X in colordict" % (classname, color))
                classtype = globals()[classname + "Type"]
                # mask image by current color/class
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
                        LOG.warning('ignoring contour of only %d%% area for color #%06X',
                                    area_pct, color)
                        continue
                    LOG.info('found region %s:%s:%s with area %d%%',
                             classname, regiontype or 'none', custom or 'none', area_pct)
                    # simplify shape
                    poly = cv2.approxPolyDP(contour, 2, False)[:, 0, ::] # already ordered x,y
                    if len(poly) < 4:
                        LOG.warning('ignoring contour of only %d points (area %d%%) for color #%06X',
                                    len(poly), area_pct, color)
                        continue
                    # instantiate region
                    regionno += 1
                    region = classtype(id="region_%d" % regionno, type_=regiontype, custom=custom,
                                       Coords=CoordsType(points=points_from_polygon(poly)))
                    # add region
                    getattr(page, 'add_%s' % classname)(region)
                    
            # Use input_file's basename for the new file -
            # this way the files retain the same basenames:
            file_id = input_file.ID.replace(ifgs[0], self.output_file_grp)
            if file_id == input_file.ID:
                file_id = concat_padded(self.output_file_grp, n)
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(self.output_file_grp,
                                            file_id + '.xml'),
                content=to_xml(pcgts))
            
    def zip_input_files(self, ifgs):
        """Get a list (for each physical page) of tuples (for each input file group) of METS files."""
        ifts = list() # file tuples
        if self.page_id:
            pages = [self.page_id]
        else:
            pages = self.workspace.mets.physical_pages
        for page_id in pages:
            ifiles = list()
            for ifg in ifgs:
                LOG.debug("adding input file group %s to page %s", ifg, page_id)
                files = self.workspace.mets.find_files(pageId=page_id, fileGrp=ifg)
                if not files:
                    # fall back for missing pageId via Page imageFilename:
                    all_files = self.workspace.mets.find_files(fileGrp=ifg)
                    for file_ in all_files:
                        pcgts = page_from_file(self.workspace.download_file(file_))
                        image_url = pcgts.get_Page().get_imageFilename()
                        img_files = self.workspace.mets.find_files(url=image_url)
                        if img_files and img_files[0].pageId == page_id:
                            files = [file_]
                            break
                if not files:
                    # other fallback options?
                    LOG.error('found no page %s in file group %s',
                              page_id, ifg)
                    ifiles.append(None)
                else:
                    ifiles.append(files[0])
            if ifiles[0]:
                ifts.append(tuple(ifiles))
        return ifts
            

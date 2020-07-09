from __future__ import absolute_import

import json
import os.path
import os

import requests

from ocrd_utils import (
    getLogger, concat_padded,
    MIMETYPE_PAGE
)
from ocrd_models.ocrd_page import (
    PageType,
    MetadataItemType,
    LabelsType, LabelType,
    to_xml
)
from ocrd_models.ocrd_page_generateds import (
    RegionRefType,
    RegionRefIndexedType,
    OrderedGroupType,
    OrderedGroupIndexedType,
    UnorderedGroupType,
    UnorderedGroupIndexedType
)
from ocrd_modelfactory import page_from_file
from ocrd import Processor

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-classify-address-text'
LOG = getLogger('processor.ClassifyAddressText')

# FIXME: rid of this switch, convert GT instead (from region to line level annotation)
# set True if input is GT, False to use classifier
ALREADY_CLASSIFIED = False

# text classification for address snippets
def classify_address(text):
    # TODO more simple heuristics to avoid API call
    # when no chance to be an address text
    if 8 > len(text) or len(text) > 80:
        return 'ADDRESS_NONE'
    # reduce allcaps to titlecase
    words = [word.title() if word.isupper() else word for word in text.split(' ')]
    text = ' '.join(words)
    result = requests.post(
        os.environ['SERVICE_URL'], json={'text': text},
        auth=requests.auth.HTTPBasicAuth(
            os.environ['SERVICE_LGN'],
            os.environ['SERVICE_PWD']))
    # should have result ADDRESS_ZIP_CITY
    # "Irgendwas 50667 Köln"
    # should have result ADDRESS_STREET_HOUSENUMBER_ZIP_CITY
    # "Bahnhofstrasse 12, 50667 Köln"
    # should have result ADDRESS_ADRESSEE_ZIP_CITY
    # "Matthias Maier , 50667 Köln"
    # should have result ADDRESS_FULL
    # "Matthias Maier - Bahnhofstrasse 12 - 50667 Köln"
    # should have result ADDRESS_NONE
    # "Hier ist keine Adresse sondern Rechnungsnummer 12312234:"
    # FIXME: train visual models for multi-class input and use multi-line text
    # TODO: check result network status
    LOG.debug("text classification result for '%s' is: %s", text, result.text)
    result = json.loads(result.text)
    # TODO: train visual models for soft input and use result['confidence']
    result = result['resultClass']
    if result != 'ADDRESS_NONE':
        return result
    # try a few other fallbacks
    if '·' in text:
        return classify_address(text.replace('·', ','))
    if ' - ' in text:
        return classify_address(text.replace(' - ', ' '))
    if ' | ' in text:
        return classify_address(text.replace(' | ', ' '))
    return result

class ClassifyAddressText(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(ClassifyAddressText, self).__init__(*args, **kwargs)

    def process(self):
        """Classify text lines belonging to addresses from text recognition results.
        
        Open and deserialize PAGE input files and their respective images,
        then iterate over the element hierarchy down to the text line level.
        
        Then, get the text results of each line and classify them into
        text belonging to address descriptions and other.
        
        Annotate the class results (name, street, zip, none) via `@custom` descriptor.
        
        Produce a new output file by serialising the resulting hierarchy.
        """
        
        # pylint: disable=attribute-defined-outside-init
        for n, input_file in enumerate(self.input_files):
            file_id = input_file.ID.replace(self.input_file_grp, self.output_file_grp)
            if file_id == input_file.ID:
                file_id = concat_padded(self.output_file_grp, n)
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            
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
            
            page = pcgts.get_Page()
            def mark_line(line, text_class):
                if text_class != 'ADDRESS_NONE':
                    line.set_custom('subtype: %s' % text_class)

            # iterate through all regions that could have lines,
            # but along annotated reading order to better connect
            # ((name+)street+)zip parts split across lines
            allregions = page_get_all_regions(page, classes='Text', order='reading-order', depth=2)
            if not allregions:
                allregions = page_get_all_regions(page, classes='Text', order='document', depth=2)
            prev_line = None
            last_line = None
            for region in allregions:
                for line in region.get_TextLine():
                    if ALREADY_CLASSIFIED:
                        # use GT classification
                        subtype = ''
                        if region.get_type() == 'other' and region.get_custom():
                            subtype = region.get_custom().replace('subtype:', '')
                        if subtype.startswith('address'):
                            mark_line(line, 'ADDRESS_FULL')
                        else:
                            mark_line(line, 'ADDRESS_NONE')
                        continue
                    # run text classification
                    textequivs = line.get_TextEquiv()
                    if not textequivs:
                        LOG.error("Line '%s' in region '%s' of page '%s' contains no text results",
                                  line.id, region.id, page_id)
                        continue
                    this_line = line
                    this_text = textequivs[0].Unicode
                    this_result = classify_address(this_text)
                    mark_line(this_line, this_result)
                    if this_result != 'ADDRESS_NONE':
                        if this_result != 'ADDRESS_FULL' and last_line:
                            last_text = last_line.get_TextEquiv()[0].Unicode
                            last_result = classify_address(', '.join([last_text, this_text]))
                            if last_result != 'ADDRESS_NONE':
                                mark_line(last_line, last_result)
                                if last_result != 'ADDRESS_FULL' and prev_line:
                                    prev_text = prev_line.get_TextEquiv()[0].Unicode
                                    prev_result = classify_address(', '.join([prev_text, last_text, this_text]))
                                    if prev_result != 'ADDRESS_NONE':
                                        mark_line(prev_line, prev_result)
                        prev_line, last_line = None, None
                    else:
                        prev_line, last_line = last_line, this_line
            
            file_path = os.path.join(self.output_file_grp,
                                     file_id + '.xml')
            out = self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                local_filename=file_path,
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts))
            LOG.info('created file ID: %s, file_grp: %s, path: %s',
                     file_id, self.output_file_grp, out.local_filename)

# FIXME: belongs into OCR-D/core
def page_get_all_regions(page, classes=None, order='document', depth=1):
    """
    Get all *Region elements below ``page``,
    or only those provided by ``classes``,
    returned in the order specified by ``reading_order``,
    and up to ``depth`` levels of recursion.
    Arguments:
       * ``classes`` (list) Classes of regions that shall be returned, e.g. ['Text', 'Image']
       * ``order`` ('document'|'reading-order') Whether to return regions sorted by document order (default) or by reading order
       * ``depth`` (integer) Maximum recursion level. Use 0 for arbitrary (i.e. unbounded) depth.
   
   For example, to get all text anywhere on the page in reading order, use:
   ::
       '\n'.join(line.get_TextEquiv()[0].Unicode
                 for region in page_get_all_regions(page, classes='Text', depth=0, order='reading-order')
                 for line in region.get_TextLine())
    """
    def region_class(x):
        return x.__class__.__name__.replace('RegionType', '')
    
    def get_recursive_regions(regions, level):
        if level == 1:
            # stop recursion, filter classes
            if classes:
                return [r for r in regions if region_class(r) in classes]
            else:
                return regions
        # find more regions recursively
        more_regions = []
        for region in regions:
            more_regions.append([])
            for class_ in ['Advert', 'Chart', 'Chem', 'Custom', 'Graphic', 'Image', 'LineDrawing', 'Map', 'Maths', 'Music', 'Noise', 'Separator', 'Table', 'Text', 'Unknown']:
                if class_ == 'Map' and not isinstance(region, PageType):
                    # 'Map' is not recursive in 2019 schema
                    continue
                more_regions[-1] += getattr(region, 'get_{}Region'.format(class_))()
        if not any(more_regions):
            return get_recursive_regions(regions, 1)
        regions = [region for r, more in zip(regions, more_regions) for region in [r] + more]
        return get_recursive_regions(regions, level - 1 if level else 0)
    ret = get_recursive_regions([page], depth + 1 if depth else 0)
    if order == 'reading-order':
        reading_order = page.get_ReadingOrder()
        if reading_order:
            reading_order = reading_order.get_OrderedGroup() or reading_order.get_UnorderedGroup()
        if reading_order:
            def get_recursive_reading_order(rogroup):
                if isinstance(rogroup, (OrderedGroupType, OrderedGroupIndexedType)):
                    elements = sorted(rogroup.get_RegionRefIndexed() +
                                      rogroup.get_OrderedGroupIndexed() + rogroup.get_UnorderedGroupIndexed(),
                                      key=lambda x : x.index)
                if isinstance(rogroup, (UnorderedGroupType, UnorderedGroupIndexedType)):
                    elements = (rogroup.get_RegionRef() + rogroup.get_OrderedGroup() + rogroup.get_UnorderedGroup())
                regionrefs = list()
                for elem in elements:
                    regionrefs.append(elem.get_regionRef())
                    if not isinstance(elem, (RegionRefType, RegionRefIndexedType)):
                        regionrefs.extend(get_recursive_reading_order(elem))
                return regionrefs
            reading_order = get_recursive_reading_order(reading_order)
        if reading_order:
            id2region = dict([(region.id, region) for region in ret])
            ret = [id2region[region_id] for region_id in reading_order if region_id in id2region]
    ret = [r for r in ret if region_class(r) in classes]
    return ret

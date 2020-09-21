from __future__ import absolute_import

import json
import os.path
import os

import requests

from ocrd_utils import (
    getLogger,
    make_file_id,
    bbox_from_points,
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
    if 8 > len(text) or len(text) > 100:
        return 'ADDRESS_NONE'
    # reduce allcaps to titlecase
    words = [word.title() if word.isupper() else word for word in text.split(' ')]
    text = ' '.join(words)
    # workaround for bad OCR:
    text = text.replace('ı', 'i')
    text = text.replace(']', 'I')
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
        return classify_address(text.replace(' - ', ', '))
    if ' | ' in text:
        return classify_address(text.replace(' | ', ', '))
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
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            
            page = pcgts.get_Page()
            def mark_line(line, text_class):
                if text_class != 'ADDRESS_NONE':
                    line.set_custom('subtype: %s' % text_class)
            def left_of(rseg, lseg):
                r_x1, r_y1, r_x2, r_y2 = bbox_from_points(rseg.get_Coords().points)
                l_x1, l_y1, l_x2, l_y2 = bbox_from_points(lseg.get_Coords().points)
                return (r_y1 < l_y2 and l_y1 < r_y2 and l_x2 < r_x1)

            # iterate through all regions that could have lines,
            # but along annotated reading order to better connect
            # ((name+)street+)zip parts split across lines
            allregions = page.get_AllRegions(classes=['Text'], order='reading-order', depth=2)
            last_lines = [None, None, None]
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
                    # If the current line is part of an address, then
                    # also try it in concatenation with the previous line etc.
                    # When concatenating, try to separate parts by comma,
                    # except if they are already written next to each other
                    # (horizontally).
                    # The top-most part (ADDRESS_FULL) is the freest, i.e.
                    # it may contain more than one line (without comma),
                    # but the text classifier is too liberal here, so
                    # we stop short at the last line of the name.
                    last_lines += [line] # expand
                    text = ''
                    for this_line, prev_line in zip(reversed(last_lines),
                                                    reversed([None] + last_lines[:-1])):
                        text = this_line.get_TextEquiv()[0].Unicode + text
                        result = classify_address(text)
                        if result == 'ADDRESS_NONE':
                            break
                        mark_line(this_line, result)
                        last_lines = [None] * len(last_lines) # reset
                        if not prev_line:
                            break
                        text = ' ' + text
                        if result == 'ADDRESS_FULL':
                            break # avoid false positives
                        if result != 'ADDRESS_FULL' and not left_of(prev_line, this_line):
                            text = ',' + text
                    last_lines = last_lines[1:] # advance

            file_id = make_file_id(input_file, self.output_file_grp)
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

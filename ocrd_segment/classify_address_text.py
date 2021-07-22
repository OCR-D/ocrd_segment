from __future__ import absolute_import

import json
import os
import math
import atexit
import itertools
from multiprocessing import Process, SimpleQueue
from queue import Empty
import requests

from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    bbox_from_points,
    MIMETYPE_PAGE
)
from ocrd_models.ocrd_page import to_xml, TextEquivType
from ocrd_modelfactory import page_from_file
from ocrd import Processor

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-classify-address-text'

# upper time limit to web API requests for textual address classification:
SERVICE_TIMEOUT = os.environ.get('SERVICE_TIMEOUT', 3.0)

NUMERIC = str.maketrans('', '', '-., €%')

# FIXME: rid of this switch, convert GT instead (from region to line level annotation)
# set True if input is GT, False to use classifier
ALREADY_CLASSIFIED = False

def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.zip_longest(a, b)

# text classification for address snippets
def classify(inq, outq):
    # Queue.get() blocks, Queue.put() too (but maxsize is infinite)
    # loop forever (or until receiving None)
    for text in iter(inq.get, None):
        outq.put(match(text))

def match(text):
    LOG = getLogger('processor.ClassifyAddressText')
    # TODO more simple heuristics to avoid API call
    # when no chance to be an address text
    if not 8 <= len(text) <= 100:
        # too short or too long for an address line (part)
        return text, 'ADDRESS_NONE', 1.0
    if (text.translate(NUMERIC).isdigit() or
        text.isalpha()):
        # must be mixed alpha _and_ numeric
        return text, 'ADDRESS_NONE', 1.0
    # normalize white-space
    text = ' '.join(text.strip().split())
    # reduce allcaps to titlecase
    if text.isupper():
        text = text.title()
    # workaround for bad OCR:
    #text = text.replace('ı', 'i')
    #text = text.replace(']', 'I')
    try:
        result = requests.post(
            os.environ['SERVICE_URL'], json={'text': text},
            timeout=SERVICE_TIMEOUT,
            auth=requests.auth.HTTPBasicAuth(
                os.environ['SERVICE_LGN'],
                os.environ['SERVICE_PWD']))
    except requests.exceptions.Timeout:
        LOG.warning("timeout for request: %s", text)
        return text, 'ADDRESS_NONE', 1.0
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
    #LOG.debug("text classification result for '%s' is: %s", text, result.text)
    result = json.loads(result.text)
    # TODO: train visual models for soft input and use result['confidence']
    class_ = result['resultClass']
    conf = result['confidence']
    # try a few other variants
    text2 = text
    for pattern, replacement in [
            (' » ', ', '),
            (' « ', ', '),
            ('·', ','),
            (' : ', ', '),
            (' - ', ', '),
            (' | ', ', '),
    ]:
        if pattern in text2:
            text2 = text2.replace(pattern, replacement)
    if text2 != text:
        text2, class2, conf2 = match(text2)
        if isbetter(class2, class_):
            text, class_, conf = text2, class2, conf2
    LOG.debug("text classification result for '%s' is: %s [%.1f]", text, class_, conf)
    return text, class_, conf

def isbetter(class1, class2):
    """Is class1 strictly more informative than class2?"""
    if class1 == 'ADDRESS_NONE':
        return False # (worst is always worse)
    if class2 == 'ADDRESS_NONE':
        return True # (anything better than nothing)
    if class2 == 'ADDRESS_FULL':
        return False # (nothing better than best)
    if class1 == 'ADDRESS_FULL':
        return True # (best is always better)
    if class1 == 'ADDRESS_ZIP_CITY':
        return False # (nothing but worst worse than 2nd-worst)
    if class2 == 'ADDRESS_ZIP_CITY':
        return True # (anything but worst better than 2nd-worst)
    # undecided:
    # - 'ADDRESS_STREET_HOUSENUMBER_ZIP_CITY' vs 'ADDRESS_ADRESSEE_ZIP_CITY'
    return False

class ClassifyAddressText(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(ClassifyAddressText, self).__init__(*args, **kwargs)
        if hasattr(self, 'output_file_grp'):
            # processing context
            self.setup()

    def setup(self):
        self.taskq = SimpleQueue() # from arbiter to workers
        self.doneq = SimpleQueue() # from workers to arbiter
        self.nproc = self.parameter['num_processes']
        for _ in range(self.nproc):
            Process(target=classify,
                    args=(self.taskq, self.doneq),
                    # ensure automatic termination at exit
                    daemon=True).start()
        def stopq():
            for _ in range(self.nproc):
                self.taskq.put(None)
        atexit.register(stopq)

    def cancelq(self):
        # we cannot just clear queues,
        # because currently active workers might still produce results:
        while not self.taskq.empty():
            try:
                self.taskq.get()
                self.doneq.put(None)
            except Empty:
                pass

    def process(self):
        """Classify text lines belonging to addresses from text recognition results.
        
        Open and deserialize PAGE input files and their respective images,
        then iterate over the element hierarchy down to the text line level.
        
        Then, get the text results of each line and classify them into
        text belonging to address descriptions and other.
        
        Annotate the class results (name, street, zip, none) via `@custom` descriptor.
        
        Produce a new output file by serialising the resulting hierarchy.
        """
        LOG = getLogger('processor.ClassifyAddressText')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)
        
        # pylint: disable=attribute-defined-outside-init
        for n, input_file in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            
            page = pcgts.get_Page()
            def mark_line(line, class_name, text=None, conf=None):
                if class_name != 'ADDRESS_NONE':
                    line.set_custom('subtype: %s' % class_name)
                    if text:
                        line.insert_TextEquiv_at(0, TextEquivType(
                            Unicode=text, conf=conf or 1.0))
            def left_of(lseg, rseg):
                r_x1, r_y1, r_x2, r_y2 = bbox_from_points(rseg.get_Coords().points)
                l_x1, l_y1, l_x2, l_y2 = bbox_from_points(lseg.get_Coords().points)
                return (r_y1 < l_y2 and l_y1 < r_y2 and l_x2 < r_x1)

            # iterate through all regions that could have lines,
            # but along annotated reading order to better connect
            # ((name+)street+)zip parts split across lines
            allregions = page.get_AllRegions(classes=['Text'], order='reading-order', depth=2)
            numlines, nummatches = 0, 0
            last_lines = list()
            for region in allregions:
                for line in region.get_TextLine():
                    numlines += 1
                    if ALREADY_CLASSIFIED:
                        # use GT classification
                        subtype = ''
                        if region.get_type() == 'other' and region.get_custom():
                            subtype = region.get_custom().replace('subtype:', '')
                        if subtype.startswith('address'):
                            nummatches += 1
                            mark_line(line, 'ADDRESS_FULL')
                        continue
                    # run text classification
                    line.texts = list()
                    line.confs = list()
                    textequivs = line.get_TextEquiv()
                    for textequiv in textequivs:
                        line.texts.append(textequiv.Unicode)
                        line.confs.append(textequiv.conf)
                    # now go looking for OCR hypotheses at the glyph level
                    def cutoff(textequiv):
                        return (textequiv.conf or 1) > self.parameter['glyph_conf_cutoff']
                    topn = self.parameter['glyph_topn_cutoff']
                    glyphs = [filter(cutoff, glyph.TextEquiv[:topn])
                              for word in line.Word
                              for glyph in word.Glyph]
                    topn = self.parameter['line_topn_cutoff']
                    # get up to n best hypotheses (without exhaustively expanding)
                    def aggconf(textequivs):
                        return sum(-math.log(te.conf or 1e-30) for te in textequivs)
                    def nbestproduct(*groups, n=1, key=id):
                        # FIXME this is just an approximation (for true breadth-first n-best outer product)
                        def add(values, prefixes):
                            sequences = []
                            for prefix in prefixes:
                                for value in values:
                                    sequences.append(prefix + (value,))
                            return sorted(sequences, key=key)[:n]
                        stack = iter(((),))
                        for group in map(tuple,groups):
                            stack = add(group, stack)
                        return stack
                    sequences = nbestproduct(*glyphs, key=aggconf, n=topn)
                    def glyphword(textequiv):
                        return textequiv.parent_object_.parent_object_
                    # regroup the line's flat glyph sequence into words
                    # then join glyphs into words and words into a text:
                    for seq in sequences:
                        if not any(seq):
                            continue
                        text = ' '.join(''.join(te.Unicode for te in word
                                                if te.Unicode)
                                        for _, word in itertools.groupby(seq, glyphword))
                        conf = sum(te.conf for te in seq) / (len(seq) or 1)
                        if text in line.texts:
                            continue
                        line.texts.append(text)
                        line.confs.append(conf)
                    if not line.texts:
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
                    # This uses multiprocessing queues for the alternative
                    # OCR texts; as soon as an address-like alternative is
                    # found, the queues are flushed.
                    last_lines.append(line) # shift in
                    text, class_ = '', 'ADDRESS_NONE'
                    for this_line, prev_line in pairwise(reversed(last_lines)):
                        for this_text in this_line.texts:
                            self.taskq.put(this_text + text)
                        this_text, this_class, this_conf = None, 'ADDRESS_NONE', None
                        cancelled = False
                        for _ in this_line.texts:
                            if cancelled:
                                self.doneq.get()
                                continue
                            # get textual prediction
                            alt_text, alt_class, alt_conf = self.doneq.get()
                            if isbetter(alt_class, this_class):
                                this_text, this_class, this_conf = alt_text, alt_class, alt_conf
                            if isbetter(this_class, class_):
                                # ignore other alternatives - by canceling additional tasks and
                                # consuming all follow-up results
                                self.cancelq()
                                cancelled = True
                        if not isbetter(this_class, class_) and this_class != 'ADDRESS_FULL':
                            # no improvement or no address at all - stop trying with more lines
                            # (but full address may still grow to "fuller" address)
                            break
                        LOG.info("Line '%s' ['%s'] is an %s", this_line.id, this_text, this_class)
                        nummatches += 1
                        mark_line(this_line, this_class,
                                  text=this_text[:len(this_text)-len(text)], conf=this_conf)
                        last_lines = list() # reset
                        if not prev_line:
                            break
                        if this_class == 'ADDRESS_FULL':
                            break # avoid false positives, already best
                        text = ' ' + this_text
                        if not left_of(prev_line, this_line):
                            text = ',' + text
                        class_ = this_class
                    last_lines = last_lines[-3:] # shift out
            LOG.info("Found %d lines and %d matches", numlines, nummatches)

            file_id = make_file_id(input_file, self.output_file_grp)
            file_path = os.path.join(self.output_file_grp,
                                     file_id + '.xml')
            pcgts.set_pcGtsId(file_id)
            out = self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                local_filename=file_path,
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts))
            LOG.info('created file ID: %s, file_grp: %s, path: %s',
                     file_id, self.output_file_grp, out.local_filename)

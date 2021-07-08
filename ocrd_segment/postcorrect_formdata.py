from __future__ import absolute_import

import os.path
import os
import math
import itertools
import re

from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    MIMETYPE_PAGE
)
from ocrd_models.ocrd_page import to_xml, TextEquivType
from ocrd_modelfactory import page_from_file
from ocrd import Processor

from .config import OCRD_TOOL
from .classify_formdata_text import KEYS

from maskrcnn_cli.formdata import FIELDS

TOOL = 'ocrd-segment-postcorrect-formdata'

TYPES = {
    'date': re.compile(r'^([0-9]{1,2})\.?([0-9]{1,2})\.?([0-9]{2}){1,2}( *- *([0-9]{1,2})\.?([0-9]{1,2})\.?([0-9]{2}){1,2})?$'),
    'percentage': re.compile(r'^((0|100|[1-9][0-9]{0,1})%?)$'),
    'integer': re.compile(r'^(0|[1-9][0-9]*(( *|\.)[0-9]{3})*)$'),
    'integer2': re.compile(r'^[1-9][0-9]$'),
    'float': re.compile(r'^(0|[1-9][0-9]*(( *|\.)[0-9]{3})*)(,[0-9]{1,3})?$'),
    'float6': re.compile(r'^(0|[1-9][0-9]*(( *|\.)[0-9]{3}){0,1})(,[0-9]{1,3})?$'),
    'costs': re.compile(r'^(0|[1-9][0-9]*(( *|\.)[0-9]{3})*)(,[0-9]{2})?( *€?)$'),
    # already matched/classified by context classifier:
    'energy': re.compile(r'^(erdgas|fluessiggas|fernwaerme|heizoel|holz|strom)$'),
    'energyunit': re.compile(r'^(liter|cbm|kg|kwh|mwh|gj)$'),
    'waterunit': re.compile(r'^(cbm|kwh|einheiten|personen|zapfstellen|qm)$'),
}

PATTERNS = {
    "abrechnungszeitraum": TYPES['date'],
    "nutzungszeitraum": TYPES['date'],
    "gebaeude_heizkosten_raumwaerme": TYPES['costs'],
    "gebaeude_heizkosten_warmwasser": TYPES['costs'],
    "anteil_grundkost_heizen": TYPES['percentage'], # "prozent_grundkosten_raumwaerme"
    "anteil_grundkost_warmwasser": TYPES['percentage'], # "prozent_grundkosten_warmwasser"
    "energietraeger": TYPES['energy'], # not used
    "energietraeger_verbrauch": TYPES['float'],
    "energietraeger_einheit": TYPES['energyunit'],
    "energietraeger_kosten": TYPES['costs'],
    "gebaeude_flaeche": TYPES['float6'],
    "wohnung_flaeche": TYPES['float6'],
    "gebaeude_verbrauchseinheiten": TYPES['float'],
    "wohnung_verbrauchseinheiten": TYPES['float'],
    "gebaeude_warmwasser_verbrauch": TYPES['float'],
    "gebaeude_warmwasser_verbrauch_einheit": TYPES['waterunit'],
    "kaltwasser_fuer_warmwasser": None,
    "wohnung_warmwasser_verbrauch": TYPES['float'],
    "wohnung_warmwasser_verbrauch_einheit": TYPES['waterunit'],
    "gebaeude_grundkost_heizen": TYPES['costs'], # "gebaeude_grundkosten_raumwaerme",
    "gebaeude_grundkost_warmwasser": TYPES['costs'], # "gebaeude_grundkosten_warmwasser",
    "gebaeude_heizkosten_gesamt": TYPES['costs'],
    "anteil_verbrauchskosten_heizen": TYPES['percentage'], # "prozent_verbrauchskosten_raumwaerme"
    "anteil_verbrauchskosten_warmwasser": TYPES['percentage'], # "prozent_verbrauchskosten_warmwasser"
    "gebaeude_verbrauchskosten_raumwaerme": TYPES['costs'],
    "gebaeude_verbrauchskosten_warmwasser": TYPES['costs'],
    "wohnung_heizkosten_gesamt": TYPES['costs'],
    "wohnung_grundkosten_raumwaerme": TYPES['costs'],
    "wohnung_verbrauchskosten_raumwaerme": TYPES['costs'],
    "wohnung_grundkosten_warmwasser": TYPES['costs'],
    "wohnung_verbrauchskosten_warmwasser": TYPES['costs'],
    "warmwasser_temperatur": TYPES['integer2'],
    "nebenkosten_betriebsstrom": TYPES['costs'],
    "nebenkosten_wartung_heizung": TYPES['costs'],
    "nebenkosten_messgeraet_miete": TYPES['costs'],
    "nebenkosten_messung_abrechnung": TYPES['costs'],
}

RENAME = {
    "anteil_grundkost_heizen": "prozent_grundkosten_raumwaerme",
    "anteil_grundkost_warmwasser": "prozent_grundkosten_warmwasser",
    "gebaeude_grundkost_heizen": "gebaeude_grundkosten_raumwaerme",
    "gebaeude_grundkost_warmwasser": "gebaeude_grundkosten_warmwasser",
    "anteil_verbrauchskosten_heizen": "prozent_verbrauchskosten_raumwaerme",
    "anteil_verbrauchskosten_warmwasser": "prozent_verbrauchskosten_warmwasser",
}
       

# normalize spelling
def normalize(text):
    text = text.strip()
    # try reducing allcaps to titlecase
    text = ' '.join(word.title() if word.isupper() else word
                    for word in text.split())
    return text

def match(category, texts, segment=''):
    LOG = getLogger('processor.PostCorrectFormData')
    if category not in FIELDS:
        raise Exception("Unknown category '%s'" % category)
    pattern = PATTERNS[category]
    keys = KEYS.get(category, [])
    text0 = ''
    textn = 0
    for text, conf in texts:
        if not text:
            LOG.warning("ignoring empty text at '%s'", segment)
            continue
        if not text0:
            text0 = text
        else:
            textn += 1
        text = normalize(text)
        if text in keys:
            if text == text0:
                LOG.debug("direct match for '%s' in %s: %s", text, category, keys[text])
            else:
                LOG.debug("direct match for '%s' over '%s' in %s: %s", text, text0, category, keys[text])
            return keys[text], conf, text
        if pattern.fullmatch(text):
            if text == text0:
                LOG.debug("pattern match for '%s' in %s: %s", text, category, pattern.pattern)
            else:
                LOG.debug("pattern match for '%s' over '%s' in %s: %s", text, text0, category, pattern.pattern)
            return text, conf, pattern.pattern
    LOG.warning("no match for '%s' (or %d alternatives) in %s at '%s'", text0, textn, category, segment)
    return None, 0, ''

class PostCorrectFormData(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(PostCorrectFormData, self).__init__(*args, **kwargs)

    def process(self):
        """Post-correct form field target lines from text recognition results.
        
        Open and deserialize PAGE input files and their respective images,
        then iterate over the element hierarchy down to the text line level.
        
        Get the text results (including glyph-level alternatives) of each line
        that is marked as form field target of any category via `@custom` descriptor.
        Search them for predefined patterns of the respective category. For each
        match, annotate that alternative as primary text result.
        
        Produce a new output file by serialising the resulting hierarchy.
        """
        LOG = getLogger('processor.PostCorrectFormData')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)
        
        # pylint: disable=attribute-defined-outside-init
        for n, input_file in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            
            page = pcgts.get_Page()
            allregions = page.get_AllRegions(classes=['Text'], depth=2)
            numsegments = 0
            nummatches = 0
            for region in allregions:
                for line in region.get_TextLine():
                    custom = line.get_custom() or ''
                    category = next((cat.replace('subtype:target=', '')
                                     for cat in custom.split(',')
                                     if cat.startswith('subtype:target=')),
                                     '')
                    if not category:
                        LOG.warning("Line '%s' on page '%s' contains no target category", line.id, page_id)
                        continue
                    line.set_custom('subtype:target=%s' % RENAME.get(category, category))
                    # get OCR results (best on line/word level, n-best concatenated from glyph level)
                    line.texts = list()
                    line.confs = list()
                    textequivs = line.get_TextEquiv()
                    for textequiv in textequivs:
                        line.texts.append(textequiv.Unicode)
                        line.confs.append(textequiv.conf)
                    # now go looking for OCR hypotheses at the glyph level
                    def glyphword(textequiv):
                        return textequiv.parent_object_.parent_object_
                    def cutoff(textequiv):
                        return (textequiv.conf or 1) > self.parameter['glyph_conf_cutoff']
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
                    def length0(segment):
                        if segment.TextEquiv:
                            return len(segment.TextEquiv[0].Unicode or '')
                        return 0
                    # first try full line
                    topn = self.parameter['glyph_topn_cutoff']
                    glyphs = [filter(cutoff, glyph.TextEquiv[:topn])
                              for word in line.Word
                              for glyph in word.Glyph]
                    topn = self.parameter['line_topn_cutoff']
                    # get up to n best hypotheses (without exhaustively expanding)
                    textequivs = nbestproduct(*glyphs, key=aggconf, n=topn)
                    # regroup the line's flat glyph sequence into words
                    # then join glyphs into words and words into a text:
                    line.texts.extend([' '.join(''.join(te.Unicode for te in word
                                                        if te.Unicode)
                                                for _, word in itertools.groupby(seq, glyphword))
                                       for seq in textequivs])
                    line.confs.extend([sum(te.conf for te in seq) / (len(seq) or 1)
                                       for seq in textequivs])
                    # second try single words only
                    # (but sort longest words first to prevent "'0' over 'Q = 7.063'" matches)
                    for word in sorted(line.Word, key=length0, reverse=True):
                        topn = self.parameter['glyph_topn_cutoff']
                        glyphs = [filter(cutoff, glyph.TextEquiv[:topn])
                                  for glyph in word.Glyph]
                        topn = self.parameter['word_topn_cutoff']
                        # get up to n best hypotheses (without exhaustively expanding)
                        textequivs = nbestproduct(*glyphs, key=aggconf, n=topn)
                        # join glyphs into words:
                        line.texts.extend([''.join(te.Unicode for te in seq
                                                   if te.Unicode)
                                           for seq in textequivs])
                        line.confs.extend([sum(te.conf for te in seq) / (len(seq) or 1)
                                           for seq in textequivs])
                    # match results against keywords of all classes
                    if not line.texts:
                        LOG.error("Line '%s' on page '%s' contains no text results",
                                  line.id, page_id)
                        continue
                    numsegments += 1
                    # run regex decoding
                    # FIXME: decode alternatives with global constraints between targets:
                    # - anteil_grundkost_heizen+anteil_grundkost_warmwasser == 100
                    # - 150 <= gebaeude_flaeche < 100000
                    # - 15 <= wohnung_flaeche < gebaeude_flaeche
                    # - wohnung_verbrauchseinheiten < gebaeude_verbrauchseinheiten
                    # - wohnung_warmwasser_verbrauch < gebaeude_warmwasser_verbrauch
                    # - gebaeude_grundkost_heizen =~ gebaeude_heizkosten_raumwaerme * anteil_grundkost_heizen / 100
                    # - gebaeude_grundkost_warmwasser =~ gebaeude_heizkosten_warmwasser * anteil_grundkost_warmwasser / 100
                    # - gebaeude_heizkosten_gesamt =~ gebaeude_heizkosten_raumwaerme + gebaeude_heizkosten_warmwasser
                    # - anteil_verbrauchskosten_heizen+anteil_verbrauchskosten_warmwasser == 100
                    # - gebaeude_verbrauchskosten_raumwaerme =~ gebaeude_heizkosten_raumwaerme * anteil_verbrauchskosten_heizen / 100
                    # - gebaeude_verbrauchskosten_warmwasser =~ gebaeude_heizkosten_warmwasser * anteil_verbrauchskosten_warmwasser / 100
                    # - wohnung_heizkosten_gesamt =~ wohnung_grundkosten_raumwaerme + wohnung_verbrauchskosten_raumwaerme + wohnung_grundkosten_warmwasser + wohnung_verbrauchskosten_warmwasser < gebaeude_heizkosten_gesamt
                    # - 35 <= warmwasser_temperatur <= 70
                    text, conf, pattern = match(category, zip(line.texts, line.confs), line.id)
                    if text:
                        nummatches += 1
                        # annotate correct text value for target
                        line.insert_TextEquiv_at(0, TextEquivType(
                            Unicode=text, conf=conf, comments=pattern))
                # remove region-level text results
                region.set_TextEquiv([])
            LOG.info("Found %d lines and %d matches",
                     numsegments, nummatches)

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

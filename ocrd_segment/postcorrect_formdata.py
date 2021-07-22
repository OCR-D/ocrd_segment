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

from maskrcnn_cli.formdata import FIELDS

from .config import OCRD_TOOL
from .classify_formdata_text import KEYS

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
    "prozent_grundkosten_raumwaerme": TYPES['percentage'],
    "prozent_grundkosten_warmwasser": TYPES['percentage'],
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
    "gebaeude_grundkosten_raumwaerme": TYPES['costs'],
    "gebaeude_grundkosten_warmwasser": TYPES['costs'],
    "gebaeude_heizkosten_gesamt": TYPES['costs'],
    "prozent_verbrauchskosten_raumwaerme": TYPES['percentage'],
    "prozent_verbrauchskosten_warmwasser": TYPES['percentage'],
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

TODECIMAL = str.maketrans(',', '.', '. €%')

NUM_CONSTRAINTS = {
    "gebaeude_flaeche": lambda x: 150 <= x < 100000,
    "wohnung_flaeche": lambda x: 15 <= x,
    "warmwasser_temperatur": lambda x: 35 <= x <= 70,
}

RENAME_FIELDS = {
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
    """Generator for locally correct alternatives.
    
    Iterate over `texts`, and yield results that match the
    regular expression for the respective `category`.
    
    Takes tuples of text and confidence.
    
    Yields tuples of text, confidence, and the pattern applied.
    """
    LOG = getLogger('processor.PostCorrectFormData')
    if category not in FIELDS:
        raise Exception("Unknown category '%s'" % category)
    keys = KEYS.get(category, [])
    pattern = PATTERNS[category]
    num_constraint = NUM_CONSTRAINTS.get(category, None)
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
            yield keys[text], conf, text
        if pattern.fullmatch(text) and (
                num_constraint is None or
                num_constraint(float(text.translate(TODECIMAL)))):
            if text == text0:
                LOG.debug("pattern match for '%s' in %s: %s", text, category, pattern.pattern)
            else:
                LOG.debug("pattern match for '%s' over '%s' in %s: %s", text, text0, category, pattern.pattern)
            yield text, conf, pattern.pattern
    LOG.warning("no match for '%s' (or %d alternatives) in %s at '%s'", text0, textn, category, segment)

def prepend(value, iterator):
    return itertools.chain([value], iterator)

def consist(targets):
    """Try to make results globally consistent by looking at all alternatives.
    
    Check line results in `targets` for pairs/groups known to be semantically
    related. For each pair/group that is fully present, iterate through all
    locally correct alternatives and check whether the respective constraints
    can be fulfilled in any combination. If that combination equals the first
    (i.e. was already annotated before iterating), then do nothing - the result
    is already consistent. Otherwise override the annotation with that result
    and stop. If no consistent combination can be found, show a warning.
    
    (Returns/yields nothing. By side effect, ad-hoc attributes of the line
    segments will be evaluated, and possibly their TextEquiv results changed.)
    """
    LOG = getLogger('processor.PostCorrectFormData')
    # Note: the order of the checks is significant here, because we will
    # exhaust the generator of local alternatives upon first use. This is
    # by choice: If we were to copy the .iter each time, we would override
    # any previous solutions. Since we cannot know in advance, what combination
    # of detections are in `targets`, we have to guess by importance.
    # TODO (But it would be great if there _was_ a way to copy the iterator...)
    
    # - prozent_grundkosten_raumwaerme+prozent_verbrauchskosten_raumwaerme == 100
    # - prozent_grundkosten_warmwasser+prozent_verbrauchskosten_warmwasser == 100
    for term1, term2 in [
            ('prozent_grundkosten_raumwaerme', 'prozent_verbrauchskosten_raumwaerme'),
            ('prozent_grundkosten_warmwasser', 'prozent_verbrauchskosten_warmwasser')]:
        value1 = targets.get(term1, None)
        value2 = targets.get(term2, None)
        if value1 and value2:
            # search for consistent solution (needs to exhaust the locally valid alternatives)
            for (text1, conf1, pattern1), (text2, conf2, pattern2) \
            in itertools.product(prepend((value1.text, value1.conf, value1.pattern),
                                         value1.iter),
                                 prepend((value2.text, value2.conf, value2.pattern),
                                         value2.iter)):
                num1 = int(text1.strip('%'))
                num2 = int(text2.strip('%'))
                if 99 <= num1 + num2 <= 101:
                    if (text1 != value1.text or text2 != value2.text):
                        LOG.info("Solved inconsistency between %s and %s", term1, term2)
                        value1.TextEquiv[0].Unicode = text1
                        value1.TextEquiv[0].conf = conf1
                        value1.TextEquiv[0].comments = pattern1
                        value2.TextEquiv[0].Unicode = text2
                        value2.TextEquiv[0].conf = conf2
                        value2.TextEquiv[0].comments = pattern2
                    else:
                        LOG.info("No inconsistency between %s and %s", term1, term2)
                    break
                LOG.warning("Unresolved inconsistency between %s and %s", term1, term2)
    
    # - wohnung_flaeche < gebaeude_flaeche
    # - wohnung_verbrauchseinheiten < gebaeude_verbrauchseinheiten
    # - wohnung_warmwasser_verbrauch < gebaeude_warmwasser_verbrauch
    # - wohnung_heizkosten_gesamt < gebaeude_heizkosten_gesamt
    for term1, term2 in [
            ('wohnung_flaeche', 'gebaeude_flaeche'),
            ('wohnung_verbrauchseinheiten', 'gebaeude_verbrauchseinheiten'),
            ('wohnung_warmwasser_verbrauch', 'gebaeude_warmwasser_verbrauch'),
            ('wohnung_heizkosten_gesamt', 'gebaeude_heizkosten_gesamt')]:
        value1 = targets.get(term1, None)
        value2 = targets.get(term2, None)
        if value1 and value2:
            # search for consistent solution (needs to exhaust the locally valid alternatives)
            for (text1, conf1, pattern1), (text2, conf2, pattern2) \
            in itertools.product(prepend((value1.text, value1.conf, value1.pattern),
                                         value1.iter),
                                 prepend((value2.text, value2.conf, value2.pattern),
                                         value2.iter)):
                num1 = float(text1.translate(TODECIMAL))
                num2 = float(text2.translate(TODECIMAL))
                if num1 < num2:
                    if (text1 != value1.text or text2 != value2.text):
                        LOG.info("Solved inconsistency between %s and %s", term1, term2)
                        value1.TextEquiv[0].Unicode = text1
                        value1.TextEquiv[0].conf = conf1
                        value1.TextEquiv[0].comments = pattern1
                        value2.TextEquiv[0].Unicode = text2
                        value2.TextEquiv[0].conf = conf2
                        value2.TextEquiv[0].comments = pattern2
                    else:
                        LOG.info("No inconsistency between %s and %s", term1, term2)
                    break
                LOG.warning("Unresolved inconsistency between %s and %s", term1, term2)
                
    # - gebaeude_grundkost_heizen =~ gebaeude_heizkosten_raumwaerme * prozent_grundkosten_raumwaerme / 100
    # - gebaeude_grundkost_warmwasser =~ gebaeude_heizkosten_warmwasser * prozent_grundkosten_warmwasser / 100
    # - gebaeude_verbrauchskosten_raumwaerme =~ gebaeude_heizkosten_raumwaerme * prozent_verbrauchskosten_raumwaerme / 100
    # - gebaeude_verbrauchskosten_warmwasser =~ gebaeude_heizkosten_warmwasser * prozent_verbrauchskosten_warmwasser / 100
    for prod, term, prct in [
            ('gebaeude_grundkost_heizen', 'gebaeude_heizkosten_raumwaerme', 'prozent_grundkosten_raumwaerme'),
            ('gebaeude_grundkost_warmwasser', 'gebaeude_heizkosten_warmwasser', 'prozent_grundkosten_warmwasser'),
            ('gebaeude_verbrauchskosten_raumwaerme', 'gebaeude_heizkosten_raumwaerme', 'prozent_verbrauchskosten_raumwaerme'),
            ('gebaeude_verbrauchskosten_warmwasser', 'gebaeude_heizkosten_warmwasser', 'prozent_verbrauchskosten_warmwasser')]:
        value_prod = targets.get(prod, None)
        value_term = targets.get(term, None)
        value_prct = targets.get(prct, None)
        if value_prod and value_term and value_prct:
            # search for consistent solution (needs to exhaust the locally valid alternatives)
            for ((text_prod, conf_prod, pattern_prod),
                 (text_term, conf_term, pattern_term),
                 (text_prct, conf_prct, pattern_prct)) \
            in itertools.product(prepend((value_prod.text, value_prod.conf, value_prod.pattern),
                                         value_prod.iter),
                                 prepend((value_term.text, value_term.conf, value_term.pattern),
                                         value_term.iter),
                                 prepend((value_prct.text, value_prct.conf, value_prct.pattern),
                                         value_prct.iter)):
                num_prod = float(text_prod.translate(TODECIMAL))
                num_term = float(text_term.translate(TODECIMAL))
                num_prct = float(text_prct.translate(TODECIMAL))
                if 99 <= num_term * num_prct / (num_prod or 1e-9) <= 101:
                    if (text_prod != value_prod.text or
                        text_term != value_term.text or
                        text_prct != value_prct.text):
                        LOG.info("Solved inconsistency between %s and %s and %s", prod, term, prct)
                        value_prod.TextEquiv[0].Unicode = text_prod
                        value_prod.TextEquiv[0].conf = conf_prod
                        value_prod.TextEquiv[0].comments = pattern_prod
                        value_term.TextEquiv[0].Unicode = text_term
                        value_term.TextEquiv[0].conf = conf_term
                        value_term.TextEquiv[0].comments = pattern_term
                        value_prct.TextEquiv[0].Unicode = text_prct
                        value_prct.TextEquiv[0].conf = conf_prct
                        value_prct.TextEquiv[0].comments = pattern_prct
                    else:
                        LOG.info("No inconsistency between %s and %s and %s", prod, term, prct)
                    break
                LOG.warning("Unresolved inconsistency between %s and %s and %s", prod, term, prct)
        
    # - gebaeude_heizkosten_gesamt =~ gebaeude_heizkosten_raumwaerme + gebaeude_heizkosten_warmwasser
    # - wohnung_heizkosten_gesamt =~ wohnung_grundkosten_raumwaerme + wohnung_verbrauchskosten_raumwaerme + wohnung_grundkosten_warmwasser + wohnung_verbrauchskosten_warmwasser
    for total, *addends in [
            ('gebaeude_heizkosten_gesamt', 'gebaeude_heizkosten_raumwaerme', 'gebaeude_heizkosten_warmwasser'),
            ('wohnung_heizkosten_gesamt', 'wohnung_grundkosten_raumwaerme', 'wohnung_verbrauchskosten_raumwaerme', 'wohnung_grundkosten_warmwasser', 'wohnung_verbrauchskosten_warmwasser')]:
        value_total = targets.get(total, None)
        value_addends = [targets.get(addend, None) for addend in addends]
        if value_total and all(value_addends):
            # search for consistent solution (needs to exhaust the locally valid alternatives)
            for (text_total, conf_total, pattern_total), *alternative_addends \
            in itertools.product(*prepend(prepend((value_total.text, value_total.conf, value_total.pattern),
                                                  value_total.iter),
                                          map(lambda value_addend:
                                              prepend((value_addend.text, value_addend.conf, value_addend.pattern),
                                                      value_addend.iter),
                                              value_addends))):
                num_total = float(text_total.translate(TODECIMAL))
                num_addends = [float(text_addend.translate(TODECIMAL)) for text_addend, _, _ in alternative_addends]
                if 0.98 <= sum(num_addends) / (num_total or 1e-9) <= 1.02:
                    if (text_total != value_total.text or
                        any(text_addend != value_addend.text
                            for (text_addend, _, _), value_addend in zip(alternative_addends, value_addends))):
                        LOG.info("Solved inconsistency between %s and %s", total, str(addends))
                        value_total.TextEquiv[0].Unicode = text_total
                        value_total.TextEquiv[0].conf = conf_total
                        value_total.TextEquiv[0].comments = pattern_total
                        for (text_addend, conf_addend, pattern_addend), value_addend \
                        in zip(alternative_addends, value_addends):
                            value_addend.TextEquiv[0].Unicode = text_addend
                            value_addend.TextEquiv[0].conf = conf_addend
                            value_addend.TextEquiv[0].comments = pattern_addend
                    else:
                        LOG.info("No inconsistency between %s and %s", total, str(addends))
                    break
                LOG.warning("Unresolved inconsistency between %s and %s", total, str(addends))


class PostCorrectFormData(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(PostCorrectFormData, self).__init__(*args, **kwargs)
        if hasattr(self, 'output_file_grp'):
            # processing context
            self.setup()

    def setup(self):
        for i, category in enumerate(FIELDS):
            if category in RENAME_FIELDS:
                FIELDS[i] = RENAME_FIELDS[category]

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
            targets = dict()
            # decode alternatives with local constraints for each target:
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
                    category = RENAME_FIELDS.get(category, category)
                    line.set_custom('subtype:target=%s' % category)
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
                    # run regex decoding (1-best match, but keep continuation if any)
                    it = match(category, zip(line.texts, line.confs), line.id)
                    try:
                        line.text, line.conf, line.pattern = next(it)
                        line.iter = it
                        targets[category] = line
                        nummatches += 1
                        # annotate correct text value for target
                        line.insert_TextEquiv_at(0, TextEquivType(
                            Unicode=line.text, conf=line.conf, comments=line.pattern))
                    except StopIteration:
                        pass
                # remove region-level text results
                region.set_TextEquiv([])
            LOG.info("Found %d lines and %d matches",
                     numsegments, nummatches)

            # (re)decode alternatives with global constraints between targets:
            if self.parameter['make_consistent']:
                consist(targets)

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

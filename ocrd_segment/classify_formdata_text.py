from __future__ import absolute_import

import json
import os.path
import os
from fuzzywuzzy import fuzz, process as fuzz_process

from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    bbox_from_points,
    MIMETYPE_PAGE
)
from ocrd_models.ocrd_page import to_xml, TextEquivType, WordType
from ocrd_modelfactory import page_from_file
from ocrd import Processor

from .config import OCRD_TOOL

from maskrcnn_cli.formdata import FIELDS

TOOL = 'ocrd-segment-classify-formdata-text'

KEYWORDS = { # FIXME We need a data-driven model for this (including confidence).
    # (derived intellectually from most frequent OCR results on GT
    #  for each context category, mixed across providers/models)
    None: [],
    "abrechnungszeitraum": ["Abrechnungszeitraum",
                            "Abrechnungszeit",
                            "Ablesetage",
                            "Ab",
                            "Abrechnung für",
                            # extra context (only unsystematic/initially?):
                            "Einzelabrechnung",
                            "Einzelabrechnung der Heizkosten",
                            "Einzelabrechnung der Heiz- und Warmwasserkosten",
                            "Einzelabrechnung der Kaltwasserkosten",
                            "Gesamtabrechnung",
                            "Heizkostenabrechnung",
                            "Heizung, Warmwasser, Kaltwasser",
                            "Heiz-, Warm- und Kaltwasser-",
                            "Heiz-, Warmwasser- und Haus-",
                            "kostenabrechnung"],
    "nutzungszeitraum": ["Nutzungszeit",
                         "Nutzungszeitraum",
                         "anteiliger Zeitraum",
                         "Kosten für den Zeitraum",
                         # extra context (only unsystematic/initially?):
                         "Einzelabrechnung",
                         "Einzelabrechnung der Heizkosten",
                         "Einzelabrechnung der Heiz- und Warmwasserkosten",
                         "Einzelabrechnung der Kaltwasserkosten",
                         "Gesamtabrechnung",
                         "Heizkostenabrechnung",
                         "Heizung, Warmwasser, Kaltwasser",
                         "Heiz-, Warm- und Kaltwasser-",
                         "Heiz-, Warmwasser- und Haus-",
                         "kostenabrechnung"],
    "gebaeude_heizkosten_raumwaerme": ["Aufteilung der Gesamtkosten",
                                       "Aufteilung der Kosten",
                                       "Aufteilung der Kosten von",
                                       "Verteilung der Gesamtkosten",
                                       "Kostenart",
                                       "Gesamt",
                                       "Gesamtkosten",
                                       "Heizkosten",
                                       "Heizung",
                                       "Summe Kosten für Heizung"],
    "gebaeude_heizkosten_warmwasser": ["Aufteilung der Gesamtkosten",
                                       "Aufteilung der Kosten",
                                       "Aufteilung der Kosten von",
                                       "Verteilung der Gesamtkosten",
                                       "Kostenart",
                                       "Gesamt",
                                       "Gesamtkosten",
                                       "Warmwasserkosten",
                                       "Warmwasser",
                                       "Kosten für Warmwasser"],
    # "prozent_grundkosten_raumwaerme"
    "anteil_grundkost_heizen": ["Anteil an den Gesamtkosten",
                                "Aufteilung der Gesamtkosten",
                                "Aufteilung der Kosten",
                                "Aufteilung der Kosten von",
                                "Verteilung der Gesamtkosten",
                                "Betrag", "in EUR",
                                "30%", "40%", "50%",
                                "Grundkosten",
                                "Heizkosten",
                                "Heizung",
                                "Kosten für Heizung"],
    # "prozent_grundkosten_warmwasser"
    "anteil_grundkost_warmwasser": ["Anteil an den Gesamtkosten",
                                "Aufteilung der Gesamtkosten",
                                "Aufteilung der Kosten",
                                "Aufteilung der Kosten von",
                                "Verteilung der Gesamtkosten",
                                "Betrag", "in EUR",
                                "30%", "40%", "50%",
                                "Grundkosten Warmwasser",
                                "Grundk. Warmwasser",
                                "Warmwasserkosten",
                                "Warmwasser",
                                "Kosten für Warmwasser"],
    # not used in layout model (ONLY textually), cf. KEYS
    "energietraeger": ["Gas",
                       "bwGas",
                       "Erdgas",
                       "Stadtgas",
                       "Fl.-Gas",
                       "Flüssiggas",
                       "Fernw."
                       "Fernwärme",
                       "Wärmelieferung",
                       "Heizöl",
                       "Öl",
                       "Strom"],
    "energietraeger_verbrauch": ["Aufstellung der Gesamtkosten",
                                 "Heizungsanlage",
                                 "Bezug",
                                 "Bezüge",
                                 "Verbrauch",
                                 "Summe",
                                 "Menge",
                                 "Gas",
                                 "bwGas",
                                 "Erdgas",
                                 "Stadtgas",
                                 "Fl.-Gas",
                                 "Flüssiggas",
                                 "Fernwärme",
                                 "Wärmelieferung",
                                 "Fernw.",
                                 "Heizöl",
                                 "Öl"],
    "energietraeger_einheit": ["Kostenaufstellung des gesamten Objektes",
                               "Einheit",
                               "Position",
                               "Wärmelieferung",
                               "Menge",
                               "bwGas",
                               "Gas",
                               "Erdgas",
                               "Stadtgas",
                               "Fl.-Gas",
                               "Flüssiggas",
                               "Fernwärme",
                               "Wärmelieferung",
                               "Fernw.",
                               "Heizöl",
                               "Öl"],
    "energietraeger_kosten": ["Kostenaufstellung des gesamten Objektes",
                              "Betrag",
                              "Betrag EUR",
                              "Gesamt",
                              "in EUR",
                              "EUR",
                              "Summe",
                              "Summe Brennstoffkosten",
                              "Summe Verbrauch",
                              "Summe Wärmekosten",
                              "Kostenart",
                              "Brennstoffkosten",
                              "Energiekosten"],
    "gebaeude_flaeche": ["qm", "m2", "Quadratmeter",
                         "Gesamteinheiten",
                         "der Liegenschaft"],
    "wohnung_flaeche": ["qm", "m2", "Quadratmeter",
                        "Einheiten",
                        "Wohnfläche"],
    "gebaeude_verbrauchseinheiten": ["Gesamteinheiten",
                                     "der Liegenschaft",
                                     "Einheiten",
                                     "HKV-Einheiten",
                                     "kWh", "MWh",
                                     "Kilowatt-Stunden",
                                     "Megawattstunden",
                                     "Striche",
                                     "Stricheinheiten",
                                     "Verbrauchswerte"],
    "wohnung_verbrauchseinheiten": ["Einheiten",
                                    "Striche"],
    "gebaeude_warmwasser_verbrauch": ["Gesamteinheiten",
                                      "der Liegenschaft",
                                      "Warmwasser",
                                      "Warmwasserkosten"],
    "gebaeude_warmwasser_verbrauch_einheit": ["Gesamteinheiten",
                                              "Warmwasser"],
    "kaltwasser_fuer_warmwasser": [],
    "wohnung_warmwasser_verbrauch": ["Einheiten",
                                     "Striche"],
    "wohnung_warmwasser_verbrauch_einheit": ["Einheiten"],
    # "gebaeude_grundkosten_raumwaerme",
    "gebaeude_grundkost_heizen": ["Betrag",
                                  "in EUR",
                                  "20%", "30%", "40%", "50%",
                                  "Heizung",
                                  "Heizkosten",
                                  "Grundkosten",
                                  "Anteil an den Gesamtkosten",
                                  "Aufteilung der Kosten von",
                                  "Aufteilung der Gesamtkosten",
                                  "Verteilung der Grundkosten"],
    # "gebaeude_grundkosten_warmwasser",
    "gebaeude_grundkost_warmwasser": ["Betrag",
                                      "in EUR",
                                      "20%", "30%", "40%", "50%", "100%",
                                      "Warmwasser",
                                      "Warmwasserkosten",
                                      "Kosten für Warmwasser",
                                      "Verbrauchsk. Warmw.",
                                      "Grundkosten",
                                      "Grundk. Warmwasser",
                                      "Anteil an den Gesamtkosten",
                                      "Aufteilung der Kosten von",
                                      "Aufteilung der Gesamtkosten",
                                      "Verteilung der Grundkosten"],
}

KEYS = {"energietraeger": {"Gas": "erdgas",
                           "bwGas": "erdgas",
                           "Erdgas": "erdgas",
                           "Stadtgas": "erdgas",
                           "Fl.-Gas": "fluessiggas",
                           "Flüssiggas": "fluessiggas",
                           "Fernw.": "fernwaerme",
                           "Fernwärme": "fernwaerme",
                           "Wärmelieferung": "fernwaerme",
                           "Heizöl": "heizoel",
                           "Öl": "heizoel",
                           "Strom": "strom"}
}

# normalize spelling
def normalize(text):
    # workaround for bad OCR:
    text = text.replace('ı', 'i')
    text = text.replace(']', 'I')
    # TODO more simple heuristics
    if text.endswith(':') or text.endswith(';'):
        text = text[:-1]
    if text.startswith('Ihr '):
        text = text[4:]
    if text.startswith('Ihre '):
        text = text[5:]
    text = text.strip()
    # try reducing allcaps to titlecase
    text = ' '.join(word.title() if word.isupper() else word
                    for word in text.split())
    return text

def classify(text, threshold=95, tag=''):
    LOG = getLogger('processor.ClassifyFormDataText')
    classes = []
    for class_id, category in enumerate(FIELDS):
        if category not in KEYWORDS:
            LOG.warning("No keywords known for category '%s'", category)
            continue
        if text in KEYWORDS[category]:
            classes.append(class_id)
            LOG.debug("direct %s match: %s in %s", tag, text, category)
            continue
        result = fuzz_process.extractOne(text, KEYWORDS[category],
                                         processor=normalize,
                                         scorer=fuzz.UQRatio, #UWRatio
                                         score_cutoff=threshold)
        if result:
            keyword, score = result
            classes.append(class_id)
            LOG.debug("fuzzy %s match: %s~%s in %s", tag, text, keyword, category)
    return classes

class ClassifyFormDataText(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(ClassifyFormDataText, self).__init__(*args, **kwargs)

    def process(self):
        """Classify form field context lines from text recognition results.
        
        Open and deserialize PAGE input files and their respective images,
        then iterate over the element hierarchy down to the text line level.
        
        Get the text results of each line and match them against keywords
        of all categories. For each match close enough, annotate the class
        via `@custom` descriptor.
        
        Produce a new output file by serialising the resulting hierarchy.
        """
        LOG = getLogger('processor.ClassifyFormDataText')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)
        
        # pylint: disable=attribute-defined-outside-init
        for n, input_file in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            
            page = pcgts.get_Page()
            def mark_segment(segment, category, subtype='context'):
                custom = segment.get_custom() or ''
                if custom:
                    custom += ','
                custom += 'subtype:%s=%s' % (subtype, category)
                segment.set_custom(custom)

            allregions = page.get_AllRegions(classes=['Text'], depth=2)
            numsegments = 0
            nummatches = 0
            for region in allregions:
                for line in region.get_TextLine():
                    for segment in [line] + line.get_Word() or []:
                        # run text classification
                        textequivs = segment.get_TextEquiv()
                        if not textequivs:
                            LOG.error("Segment '%s' on page '%s' contains no text results",
                                  segment.id, page_id)
                            continue
                        numsegments += 1
                        text = textequivs[0].Unicode
                        for class_id in classify(text, self.parameter['threshold'],
                                                 'word' if isinstance(segment, WordType) else 'line'):
                            nummatches += 1
                            mark_segment(segment, FIELDS[class_id])
                            if FIELDS[class_id] in ['energietraeger']:
                                # classified purely textually
                                mark_segment(segment, FIELDS[class_id], subtype='target')
                                # annotate nearest text value for target
                                keyword, _ = fuzz_process.extractOne(text, KEYWORDS[FIELDS[class_id]],
                                                                     processor=normalize,
                                                                     scorer=fuzz.UQRatio) #UWRatio
                                segment.insert_TextEquiv_at(0, TextEquivType(
                                    Unicode=KEYS[FIELDS[class_id]].get(keyword),
                                    conf=textequivs[0].conf))
            LOG.info("Found %d lines/words and %d matches across classes",
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

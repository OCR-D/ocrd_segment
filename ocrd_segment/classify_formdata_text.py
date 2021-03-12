from __future__ import absolute_import

import os.path
import os
import math
from fuzzywuzzy import fuzz, process as fuzz_process

from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    MIMETYPE_PAGE
)
from ocrd_models.ocrd_page import to_xml, TextEquivType, WordType
from ocrd_modelfactory import page_from_file
from ocrd import Processor

from .config import OCRD_TOOL

from maskrcnn_cli.formdata import FIELDS

TOOL = 'ocrd-segment-classify-formdata-text'

NUMBER = "0123456789,."
KEYWORDS = { # FIXME We need a data-driven model for this (including confidence).
    # (derived intellectually from most frequent OCR results on GT
    #  for each context category, mixed across providers/models --
    #  see gt-larex-prepare.mk OCR-D-OCR-TESS-deu-SEG-tesseract-sparse-CROP-FORM-LAREX-OCR
    #  and subsequent context-target-text.sh)
    None: [],
    "abrechnungszeitraum": [
        "Abrechnungszeitraum",
        "Abrechnungszeit",
        "Abrechnungstage",
        "Ablesetage",
        "Ab",
        "Abrechnung für",
        # extra context (only unsystematic/initially?):
        "Einzelabrechnung",
        "Einzelabrechnung der Energiekosten",
        "Einzelabrechnung der Energie- und Betriebskosten",
        "Einzelabrechnung der Heizkosten",
        "Einzelabrechnung der Heiz- und Warmwasserkosten",
        "Einzelabrechnung der Kaltwasserkosten",
        "Einzelabrechnung pro Nutzer",
        "Gesamtabrechnung",
        "Energieabrechnung",
        "Heizkostenabrechnung",
        "Heizkosten- und Warmwasserkosten-Abrechnung",
        "Heiz- und Warmwasserkostenabrechnung",
        "Heizung, Warmwasser, Kaltwasser",
        "Heiz-, Warm- und Kaltwasser-",
        "Heiz-, Warmwasser- und Haus-",
        "nebenkostenabrechnung",
        "kostenabrechnung",
    ],
    "nutzungszeitraum": [
        "Nutzungszeit",
        "Nutzungszeitraum",
        "Ihr Nutzungszeitraum",
        "Zeitraum",
        "anteiliger Zeitraum",
        "Kosten für den Zeitraum",
        "Ihre Kosten für den Zeitraum",
        "Ihr anteiliger Zeitraum",
        "Ihre Abrechnung für",
        "Ihre BRUNATA-Abrechnung für",
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
        "kostenabrechnung",
    ],
    "gebaeude_heizkosten_raumwaerme": [
        "Aufteilung der Gesamtkosten",
        "Aufteilung der Kosten",
        "Aufteilung der Kosten von",
        "Aufteilung der Kosten der Heizanlage",
        "Aufteilung der Kosten für Wärmeversorgung",
        "Insgesamt zu verteilende Kosten",
        "Verteilung der Gesamtkosten",
        "Berechnung und Aufteilung der Kosten für Heizung",
        "Abrechnungsgrundlage für die Kostenverteilung auf die Nutzergruppen",
        "Kostenart",
        "Kostenaufstellung",
        "Kosten in EUR",
        "Kosten für Heizung",
        "Kosten für Heizung gesamt",
        "Kosten nur für Heizung",
        "Ihre Kosten",
        "Gesamt",
        "Gesamtkosten",
        "Heizungs Anteile",
        "Heizkosten",
        "Heizung",
        "Betrag",
        "in EUR",
        "€",
        "EUR",
        "Euro",
        "Summe Heizung",
        "Summe Kosten für Heizung",
    ],
    "gebaeude_heizkosten_warmwasser": [
        "Anteil an den Gesamtkosten",
        "Aufteilung der Gesamtkosten",
        "Aufteilung der Heiz- und Warmwasserkosten",
        "Aufteilung der Kosten",
        "Aufteilung der Kosten von",
        "Aufstellung der Kosten",
        "Verteilung der Kosten",
        "Verteilung der Gesamtkosten",
        "Berechnung und Verteilung der Kosten für Warmwasser",
        "Insgesamt zu verteilende Kosten",
        "zu verteilende Kosten",
        "Ermittlung Ihres Kostenanteils",
        "Kostenart",
        "Kostenaufstellung",
        "Gesamt",
        "Gesamtkosten",
        "Warmwasser Anteile",
        "Warmwasserkosten",
        "Warmwasserkosten (Wassererwärmungskosten)",
        "Warmwasser",
        "Betrag",
        "in EUR",
        "€",
        "EUR",
        "Euro",
        "Kosten für Warmwasser",
        "Kosten nur für Warmwasser",
        "Kosten für Warmwasser-Erwärmung",
        "Kosten Wassererwärmung",
        "Summe Warmwasser",
    ],
    # "prozent_grundkosten_raumwaerme"
    "anteil_grundkost_heizen": [
        "Anteil an den Gesamtkosten",
        "Aufteilung der Gesamtkosten",
        "Aufteilung der Kosten",
        "Aufteilung der Kosten von",
        "Verteilung der Gesamtkosten",
        "Verteilung der Kosten",
        "Ermittlung Ihres Kostenanteils",
        "Aufstellung der Kosten",
        "Berechnung Ihres Kostenanteils",
        "Berechnung und Verteilung",
        "Berechnung und Verteilung der Kosten für Heizung",
        "Betrag", "in EUR",
        #"30%", "40%", "50%",
        "Festkosten",
        "Grundkosten",
        "Grundkosten Heizung",
        "Heizkosten",
        "Heizung",
        "Heizungskosten",
        "Kosten",
        "Kosten für Heizung",
    ],
    # "prozent_grundkosten_warmwasser"
    "anteil_grundkost_warmwasser": [
        "Anteil an den Gesamtkosten",
        "Aufteilung der Gesamtkosten",
        "Aufteilung der Kosten",
        "Aufteilung der Kosten von",
        "Verteilung der Gesamtkosten",
        "Verteilung der Kosten",
        "Ermittlung Ihres Kostenanteils",
        "Berechnung Ihres Kostenanteils",
        "Berechnung und Verteilung",
        "Berechnung und Verteilung der Kosten für Warmwasser",
        "Betrag", "in EUR",
        #"30%", "40%", "50%",
        "Festkosten",
        "Grundkosten Warmwasser",
        "Grundk.",
        "Grundk. Warmwasser",
        "Warmwasserkosten",
        "Warmwasserkosten (Wassererwärmungskosten)",
        "Warmwasser",
        "Kosten für Warmwasser",
    ],
    # not used in layout model (ONLY textually), cf. KEYS
    "energietraeger": [
        "Gas",
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
        "Strom",
    ],
    "energietraeger_verbrauch": [
        "Aufstellung der Gesamtkosten",
        "Heizungsanlage",
        "Bezug",
        "Bezüge",
        "Verbrauch",
        "Gesamtverbrauch",
        "Brennstoff",
        "Summe",
        "Summe Brennstoffe",
        "Summe Verbrauch",
        "Menge",
        "Menge in kWh",
        "Position",
        "kwh",
        "kWh",
        "MWh",
        "Gas",
        "bwGas",
        "Erdgas",
        "Stadtgas",
        "Fl.-Gas",
        "Flüssiggas",
        "Fernw.",
        "Fernwärme",
        "Nahwärme",
        "Wärmelieferung",
        "Heizöl",
        "Öl",
    ],
    "energietraeger_einheit": [
        "Kostenaufstellung des gesamten Objektes",
        "Kostenaufstellung",
        "Fortsetzung der Kostenaufstellung",
        "Aufstellung der Gesamtkosten",
        "Einheit",
        "Position",
        "Brennstofflieferungen",
        "Wärmelieferung",
        "Menge",
        "bwGas",
        "Gas",
        "Erdgas",
        "Erdgas H",
        "Erdgas L",
        "Stadtgas",
        "Fl.-Gas",
        "Flüssiggas",
        "Fernw.",
        "Fernwärme",
        "Wärme",
        "Wärmelieferung",
        "Fernw.",
        "Heizöl",
        "Öl",
    ],
    "energietraeger_kosten": [
        "Aufstellung der Gesamtkosten",
        "Aufstellung der Kosten",
        "Kostenaufstellung",
        "Kostenaufstellung des gesamten Objektes",
        "Betrag",
        "Betrag EUR",
        "Kosten EUR",
        "Gesamt",
        "Gesamtverbrauch",
        "in EUR",
        "€",
        "EUR",
        "Euro",
        "Position",
        "Summe",
        "Summe Brennstoffkosten",
        "Summe Brennstoffverbrauch/-kosten",
        "Summe Verbrauch",
        "Summe Wärmekosten",
        "Verbrauch",
        "Kostenart",
        "Brennstoff",
        "Brennstoffkosten",
        "Brennstoffkosten Summe",
        "Brennstoff- /Energiekosten",
        "Energiekosten",
    ],
    "gebaeude_flaeche": [
        "qm", "m2", "m²", "Quadratmeter",
        "Nutzfläche",
        "Fläche",
        "Wohnfläche",
        "Gesamteinheiten",
        "der Liegenschaft",
    ],
    "wohnung_flaeche": [
        "qm", "m2", "m²", "Quadratmeter",
        "anteilige Einheiten",
        "Einheiten",
        "Ihre Einheiten",
        "Ihre Fläche",
        "Ihr Flächenanteil",
        "beheizb. Wohnfl.",
        "Nutzfläche",
        "Wohnfläche",
    ],
    "gebaeude_verbrauchseinheiten": [
        "Gesamteinheiten",
        "der Liegenschaft",
        "Einheiten",
        "Aufteilung der Kosten", #really?
        "Berechnung und Verteilung der Kosten für Heizung", #really?
        "HKV-Einheiten",
        "kwh", "kWh", "MWh",
        "Kilowatt-Stunden",
        "Megawattstunden",
        "Striche",
        "Stricheinheiten",
        "Verbrauchskosten",
        "Verbrauchswerte",
    ],
    "wohnung_verbrauchseinheiten": [
        "Einheiten",
        "Ihre Einheiten",
        #"Ihre Kosten", # really?
        "Verbrauchskosten",
        "kwh", "kWh", "MWh",
        "Striche",
    ],
    "gebaeude_warmwasser_verbrauch": [
        "Gesamteinheiten",
        "der Liegenschaft",
        "Kosten für Warmwasser",
        "Kosten für Warmwasser-Erwärmung",
        "verteilt über Verbrauch Warmwasserzähler",
        "Wasser",
        "Warmwasser",
        "Warmwasserkosten",
    ],
    "gebaeude_warmwasser_verbrauch_einheit": [
        "Gesamteinheiten",
        "gesamte Einheiten",
        NUMBER,
        "Warmwasser",
    ],
    "kaltwasser_fuer_warmwasser": [
    ],
    "wohnung_warmwasser_verbrauch": [
        "Einheiten",
        "Ihre Einheiten",
        "verteilt über Verbrauch Warmwasserzähler",
        "Striche",
    ],
    "wohnung_warmwasser_verbrauch_einheit": [
        "Einheiten",
        "Ihre Einheiten",
        NUMBER,
    ],
    # "gebaeude_grundkosten_raumwaerme",
    "gebaeude_grundkost_heizen": [
        "Betrag",
        "€",
        "EUR",
        "Euro",
        "in EUR",
        "20%", "30%", "40%", "50%",
        "Heizung",
        "Heizkosten",
        "Kosten in EUR",
        "Kosten für Heizung",
        "Ihre Kosten",
        "Kostenart",
        "Festkosten",
        "Grundkosten",
        "Grundkosten Heizung",
        "Gesamtbetrag",
        "Gesamtsumme",
        "Insgesamt zu verteilende Kosten",
        "Ermittlung Ihres Kostenanteils",
        "Berechnung Ihres Kostenanteils",
        "Berechnungsweg für Ihren Kostenanteil",
        "Berechnung und Verteilung der Kosten für Heizung",
        "Anteil an den Gesamtkosten",
        "Aufteilung der Kosten",
        "Aufteilung der Kosten von",
        "Aufteilung der Gesamtkosten",
        "Verteilung der Gesamtkosten",
        "Verteilung der Grundkosten",
        "Verteilung der Kosten",
        "Verteilung der Kosten auf die Nutzer für den Abrechnungsbereich Heizung",
    ],
    # "gebaeude_grundkosten_warmwasser",
    "gebaeude_grundkost_warmwasser": [
        "Betrag",
        "€",
        "EUR",
        "Euro",
        "in EUR",
        "20%", "30%", "40%", "50%", "100%",
        "Warmwasser",
        "Warmwasserkosten",
        "Warmwasserkosten (Wassererwärmungskosten)",
        "Kosten in EUR",
        "Kosten für Warmwasser",
        "Kosten für Warmwasser-Erwärmung",
        "Verbrauchsk. Warmw.",
        "Verbrauchskosten",
        "Festkosten",
        "Grundk.",
        "Grundk. Warmwasser",
        "Grundkosten",
        "Gesamtbetrag",
        "Gesamtsumme",
        "Insgesamt zu verteilende Kosten",
        "Ermittlung Ihres Kostenanteils",
        "Berechnung Ihres Kostenanteils",
        "Berechnung Ihres Kostenanteils",
        "Berechnungsweg für Ihren Kostenanteil",
        "Berechnung und Aufteilung der Kosten für Warmwasser-Erwärmung",
        "Berechnung und Verteilung der Kosten für Warmwasser",
        "Anteil an den Gesamtkosten",
        "Aufteilung der Kosten",
        "Aufteilung der Kosten von",
        "Aufteilung der Gesamtkosten",
        "Verteilung der Gesamtkosten",
        "Verteilung der Grundkosten",
    ],
    "gebaeude_heizkosten_gesamt": [
        "€",
        "EUR",
        "Euro",
        "Betrag",
        "Betrag EUR",
        "Aufstellung der Gesamtkosten",
        "Aufstellung der Kosten",
        "Kostenaufstellung",
        "Kostenaufstellung des gesamten Objektes",
        "Heizungsanlage",
        "Gesamtkosten Heizungsanlage",
        "Gesamtkosten der Liegenschaft",
        "Gesamtkosten Netto",
        "Gesamtkosten",
        "Heiz- und Warmwasserkosten",
        "Insgesamt zu verteilende Kosten",
        "Kosten der Heizanlage",
        "Kosten für Heizung",
        "Kosten für Heizung und Wassererwärmung",
        "Summe der Kosten",
        "Summe Energie- und Heiznebenkosten",
        "Summe Energie- und Heizungsbetriebskosten",
        "Summe Heizanlage",
        "Summe Heizanlage (Brennstoff- und Heiznebenkosten)",
        "Weitere Heizungsbetriebskosten",
    ],
    "anteil_verbrauchskosten_heizen": [
        "Aufstellung der Kosten",
        "Aufteilung der Gesamtkosten",
        "Aufteilung der Kosten",
        "Aufteilung der Kosten von",
        "Berechnung und Aufteilung der Kosten für Heizung",
        "Berechnung und Verteilung",
        "Gesamtkosten",
        "Heizkosten",
        "Heizung",
        "Ihr Anteil an den Gesamtkosten",
        "Insgesamt zu verteilende Kosten",
        "Kostenart",
        "Kosten für Heizung",
        "Verbrauchskosten",
        "Verbrauchsk. Heizung",
        "Verteilung der Gesamtkosten",
        "Verteilung der Kosten",
        "Verteilung der Kosten auf die Nutzer für den Abrechnungsbereich Heizung",
    ],
    "anteil_verbrauchskosten_warmwasser": [
        "Aufstellung der Kosten",
        "Aufteilung der Gesamtkosten",
        "Aufteilung der Kosten",
        "Aufteilung der Kosten von",
        "Berechnung und Verteilung",
        "Gesamtkosten",
        "Ihr Anteil an den Gesamtkosten",
        "Insgesamt zu verteilende Kosten",
        "Kostenart",
        "Kosten für Warmwasser",
        "Kosten für Warmwasser-Erwärmung",
        "Kosten für Erwärmung Warmwasser",
        "Verbrauchskosten",
        "Verbrauchsk. Warmw.",
        "Verteilung der Gesamtkosten",
        "Verteilung der Kosten",
        "Warmwasser",
        "Warmwasserkosten",
    ],
    "gebaeude_verbrauchskosten_raumwaerme": [
        "€",
        "EUR",
        "Euro",
        "in EUR",
        "Betrag",
        "Betrag EUR",
        "Aufstellung der Kosten",
        "Aufteilung der Gesamtkosten",
        "Aufteilung der Kosten",
        "Aufteilung der Kosten von",
        "Heizkosten",
        "Heizung",
        "Heizungs Anteile",
        "Ihr Anteil an den Gesamtkosten",
        "Insgesamt zu verteilende Kosten",
        "Kosten für Heizung",
        "Verbrauchskosten",
        "Verbrauchsk. Heizung",
        "Verteilung der Kosten",
    ],
    "gebaeude_verbrauchskosten_warmwasser": [
        "€",
        "EUR",
        "Euro",
        "in EUR",
        "Betrag",
        "Betrag EUR",
        "Aufstellung der Kosten",
        "Aufteilung der Gesamtkosten",
        "Aufteilung der Kosten",
        "Aufteilung der Kosten von",
        "Ihr Anteil an den Gesamtkosten",
        "Insgesamt zu verteilende Kosten",
        "Kosten für Warmwasser",
        "Kosten für Warmwasser-Erwärmung",
        "Kosten für Erwärmung Warmwasser",
        "Verbrauchskosten",
        "Verbrauchsk. Warmw.",
        "Verteilung der Kosten",
        "Warmwasser",
        "Warmwasserkosten",
    ],
    "wohnung_heizkosten_gesamt": [
        "€",
        "EUR",
        "Euro",
        "in EUR",
        "Abrechnung",
        "Ihre Abrechnung",
        "Aufteilung der Gesamtkosten",
        "Gesamtkosten",
        "Heizkosten",
        "Heiz- und Warmwasserkosten",
        "Heiz- u. Warmwasserkosten",
        "Ihr Anteil an den Gesamtkosten",
        "Ihre Gesamtkosten",
        "Ihre Gesamtkosten Heizung/Warmwasser",
        "Ihre Heizkosten",
        "Ihre Heiz- und Warmwasserkosten",
        "Ihre Heizungs Anteile",
        "Ihre Kosten",
        "Ihre Kosten (alle Beträge in brutto)",
        "Kosten",
        "Kosten EUR"
        "Rechnungsbetrag",
        "Summe",
        "Summe Heizung",
        "Summe Kosten für Heizung",
        "Summe Kosten für Heizung und Warmwasser",
        "Verteilung der Kosten",
        "Verteilung der Kosten auf die Nutzer für den Abrechnungsbereich Heizung",
        "Verteilung der Kosten auf die Nutzer für die Abrechnungsbereiche Heizung und Warmwasser",
        "Verteilung der Kosten für Heizung",
        "Übertrag",
    ],
    "wohnung_grundkosten_raumwaerme": [
        "€",
        "EUR",
        "Euro",
        "in EUR",
        "Ihre Abrechnung",
        "Aufteilung der Gesamtkosten",
        "Heizkosten",
        "Heizung",
        "Heizungs Anteile",
        "Festkosten",
        "Grundkosten",
        "Grundkosten Heizung",
        "Ihr Anteil an den Gesamtkosten",
        "Ihre Kosten",
        "Ihre Kosten (alle Beträge in brutto)",
        "Kosten",
        "Kosten EUR"
        "Kosten für Heizung",
        "Verteilung der Gesamtkosten",
        "Verteilung der Kosten",
        "Verteilung der Kosten auf die Nutzer für den Abrechnungsbereich Heizung",
        "Verteilung der Kosten auf die Nutzer für die Abrechnungsbereiche Heizung und Warmwasser",
        "Verteilung der Kosten für Heizung",
    ],
    "wohnung_verbrauchskosten_raumwaerme": [
        "€",
        "EUR",
        "in EUR",
        "Euro",
        "Heizkosten",
        "Heizung",
        "Heizungs Anteile",
        "Aufteilung der Gesamtkosten",
        "Ihr Anteil an den Gesamtkosten",
        "Ihre Abrechnung",
        "Ihre Kosten (alle Beträge in brutto)",
        "Ihre Kosten",
        "Kosten",
        "Kosten EUR"
        "Kosten für Heizung",
        "Verbrauchsk. Heizung",
        "Verbrk. Heizung",
        "Verbrauchskosten",
        "Verteilung der Kosten auf die Nutzer für den Abrechnungsbereich Heizung",
        "Verteilung der Kosten auf die Nutzer für die Abrechnungsbereiche Heizung und Warmwasser",
        "Verteilung der Kosten für Heizung",
        "Verteilung der Kosten",
    ],
    "wohnung_grundkosten_warmwasser": [
        "€",
        "EUR",
        "Euro",
        "in EUR",
        "Ihre Abrechnung",
        "Aufteilung der Gesamtkosten",
        "Festkosten",
        "Grundkosten",
        "Grundk. Warmwasser",
        "Ihr Anteil an den Gesamtkosten",
        "Ihre Kosten",
        "Ihre Kosten (alle Beträge in brutto)",
        "Kosten",
        "Kosten EUR"
        "Kosten für Warmwasser"
        "Verteilung der Kosten",
        "Verteilung der Kosten für Warmwasser",
        "Warmwasser",
        "Warmwasserkosten",
    ],
    "wohnung_verbrauchskosten_warmwasser": [
        "€",
        "EUR",
        "in EUR",
        "Euro",
        "Aufteilung der Gesamtkosten",
        "Ihr Anteil an den Gesamtkosten",
        "Ihre Abrechnung",
        "Ihre Kosten (alle Beträge in brutto)",
        "Ihre Kosten",
        "Kosten",
        "Kosten EUR"
        "Kosten für Warmwasser"
        "Verbrauchsk. Warmwasser",
        "Verbrauchsk. Warmw.",
        "Verbrauchskosten",
        "Verteilung der Kosten für Warmwasser",
        "Warmwasser",
        "Warmwasserkosten",
    ],
    "warmwasser_temperatur": [
        "Brauchwassertemperatur",
        "°C",
        "Erwärmung Warmwasser",
        "Grad",
        "mittlere Warmwassertemp. in °C",
        "Wassertemperatur",
        "Warmwasser",
        "Warmwassertemperatur",
        "Warmwasserversorgung",
        "Wassererwärmung",
    ],
    "nebenkosten_betriebsstrom": [
        "€",
        "EUR",
        "in EUR",
        "Euro",
        "Aufstellung der Kosten",
        "Betrag",
        "Betrag in EUR",
        "Betrag EUR",
        "Betriebsstrom",
        "Energiekosten Heizung",
        "Heiznebenkosten",
        "Heizungsstrom",
        "Kostenart",
        "Nebenkosten",
        "Strom",
        "Strom für Brenner + Pumpe",
        "Strom für Brenner und Pumpe",
        "Strom für Heizung",
        "Strom für Umwälzpumpe",
        "Strom Heizung",
        "Stromkosten",
        "Stromkosten Heizung",
        "Teilbetrag",
        "Weitere Kosten der Heizungsanlage",
        "Weitere Heizungsbetriebskosten",
        "Weitere Kosten",
        "Weitere Kosten der Heizungsanlage",
        "Zusatzkosten Heizung",
    ],
    "nebenkosten_wartung_heizung": [
        "Art der Aufwendung",
        "Aufstellung der Kosten",
        "Betrag",
        "Betrag in EUR",
        "Betrag EUR",
        "€"
        "EUR",
        "in EUR",
        "Brennerservice/Kesselreinigung",
        "Brennerwartung",
        "Brennerwartung + Kessel",
        "Energiekosten Heizung",
        "Gerätemiete Heizung",
        "Gerätewartung",
        "Gerätewartung Heizung",
        "Heiznebenkosten",
        "Nebenkosten",
        "Teilbetrag",
        "Wartung",
        "Wartung/Reinigung",
        "Wartung Brenner",
        "Wartung Enthärt. Heiz.",
        "Wartung Gas-Kessel",
        "Wartung Gastherme",
        "Wartung Heizanlage",
        "Wartung Heizung",
        "Wartung Heizzentrale",
        "Wartung (Kesselreinigung)",
        "Wartung (Lohn)",
        "Wartung (Mat.)",
        "Wartungskosten",
        "Wartungskosten Heizung",
        "Wartungsservice",
        "Weitere Heizungsbetriebskosten",
        "Weitere Kosten",
        "Weitere Kosten der Heizungsanlage",
        "Zusatzkosten Heizung",
    ],
    "nebenkosten_messgeraet_miete": [
        "Aufstellung der Kosten",
        "Ausgaben zur gesonderten Verteilung",
        "Betrag",
        "Betrag in EUR",
        "Betrag EUR",
        "€"
        "EUR",
        "in EUR",
        "Energiekosten Heizung",
        "Geb. Verbrauchserfsg.",
        "Gerätemiete",
        "Gerätemiete Heizung",
        "Gerätemiete Warmwasser",
        "Gerätemiete HKV",
        "Heiznebenkosten",
        "Kostenart",
        "Kosten für Heizung",
        "Kosten Geräte Hzg",
        "Kosten nur für Heizung",
        "Miete für Hkve",
        "Miete für Wärmemengenzähler",
        "Miete Heizkostenverteiler",
        "Miete Messanlage",
        "Miete Warmwasserzähler",
        "Miete Wärmezähler",
        "Miete/Wartung Geräte HZ",
        "Mietservicegebühr Wärmezähler",
        "Mietservice Wärmezähler",
        "Miet/Wart.Geräte Hzg",
        "Miet/Wart.Geräte WW",
        "Summe EUR",
        "Teilbetrag",
        "Weitere Heizungsbetriebskosten",
        "Weitere Kosten",
        "Weitere Kosten der Heizungsanlage",
        "Zusatzkosten Heizung",
        "Zusatzkosten Warmwasser",
    ],
    "nebenkosten_messung_abrechnung": [
        "Ablesegebühr",
        "Abrechnungsdienst",
        "Abrechnungskosten",
        "Abrechnungsservice",
        "Aufstellung der Kosten",
        "Energiekosten Heizung",
        "Betrag",
        "Betrag in EUR",
        "Betrag EUR",
        "€"
        "EUR",
        "in EUR",
        "BFW-Kundendienstgebühr",
        "Gebühr Verbrauchserfassung",
        "Geb.Verbrauchserfsg.",
        "Heiznebenkosten",
        "Kosten für Ablesung + Abrechnung",
        "Nebenkosten",
        "Teilbetrag",
        "Verbrauchsabr./-analyse",
        "Verbrauchsabrechnung",
        "Verbrauchserfassung",
        "Weitere Heizungsbetriebskosten",
        "Weitere Kosten",
        "Weitere Kosten der Heizungsanlage",
    ],
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
                           "Strom": "strom"},
        "energietraeger_einheit": {"Ltr.": "liter",
                                   "Liter": "liter",
                                   "cbm": "cbm",
                                   "m³": "cbm",
                                   "m3": "cbm",
                                   "Kubikmeter": "cbm",
                                   "kWh": "kwh",
                                   "MWh": "mwh"}
        # "gebaeude_warmwasser_verbrauch_einheit"
        # "wohnung_warmwasser_verbrauch_einheit"
}

# normalize spelling
def normalize(text):
    # workaround for bad OCR:
    text = text.replace('ı', 'i')
    text = text.replace(']', 'I')
    # TODO more simple heuristics
    if text.endswith(':') or text.endswith(';'):
        text = text[:-1]
    # if text.startswith('Ihr '):
    #     text = text[4:]
    # if text.startswith('Ihre '):
    #     text = text[5:]
    if text.startswith('ihre '):
        text = text.title()
    text = text.strip()
    # try reducing allcaps to titlecase
    text = ' '.join(word.title() if word.isupper() else word
                    for word in text.split())
    return text

def classify(text, threshold=95, tag=''):
    LOG = getLogger('processor.ClassifyFormDataText')
    classes = []
    if not text:
        return classes
    for class_id, category in enumerate(FIELDS):
        if category not in KEYWORDS:
            raise Exception("Unknown category '%s'" % category)
        if (NUMBER in KEYWORDS[category] and
            text.translate(str.maketrans('', '', ',. ')).isnumeric()):
            classes.append(class_id)
            LOG.debug("numeric %s match: %s in %s", tag, text, category)
            continue
        if text in KEYWORDS[category]:
            classes.append(class_id)
            LOG.debug("direct %s match: %s in %s", tag, text, category)
            continue
        # FIXME: Fuzzy string search is very different from robust OCR search:
        #        We don't want to tolerate arbitrary differences, but only
        #        graphemically likely ones; e.g.
        #        - `Ihr Kosten-` vs `Ihre Kosten`
        #        - `Verbrauchs-` vs `Verbrauch`
        #        - `100` or `100€` vs `100%`
        #        - `ab` vs `Ab-`
        #        But OCR with alternative hypotheses is hard to get by...
        #
        if (text.startswith('m') and
            ('m²' in KEYWORDS[category] or 'm³' in KEYWORDS[category]) and
            len(text) <= 2):
            classes.append(class_id)
            LOG.debug("exception %s match: %s~%s in %s", tag, text, KEYWORDS[category][0], category)
            continue
        if ('C' in text and
            '°C' in KEYWORDS[category] and
            len(text) <= 2):
            classes.append(class_id)
            LOG.debug("exception %s match: %s~%s in %s", tag, text, KEYWORDS[category][0], category)
            continue
        # fuzz scores are relative to length, but we actually
        # want to have a measure of the edit distance, or a
        # mix of both; so here we just attenuate short strings
        min_score = (1-math.exp(-len(text))) * threshold
        result = fuzz_process.extractOne(text, KEYWORDS[category],
                                         processor=normalize,
                                         scorer=fuzz.UQRatio, #UWRatio
                                         score_cutoff=min_score)
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

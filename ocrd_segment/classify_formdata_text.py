from __future__ import absolute_import

import os.path
import os
import math
import itertools
import multiprocessing as mp
from rapidfuzz import fuzz, process as fuzz_process

from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    MIMETYPE_PAGE
)
from ocrd_models.ocrd_page import (
    TextRegionType,
    TextLineType,
    WordType,
    TextEquivType,
    to_xml
)
from ocrd_modelfactory import page_from_file
from ocrd import Processor

from maskrcnn_cli.formdata import FIELDS

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-classify-formdata-text'

NUMBER = "0123456789,."
KEYWORDS = { # FIXME We need a data-driven model for this (including confidence).
    # (derived intellectually from most frequent OCR results on GT
    #  for each context category, mixed across providers/models --
    #  see gt-larex-prepare.mk OCR-D-OCR-TESS-deu-SEG-tesseract-sparse-CROP-FORM-LAREX-OCR
    #  and subsequent context-target-text.sh)
    None: [],
    "abrechnungszeitraum": {
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
    },
    "nutzungszeitraum": {
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
    },
    "gebaeude_heizkosten_raumwaerme": {
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
    },
    "gebaeude_heizkosten_warmwasser": {
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
    },
    # "prozent_grundkosten_raumwaerme"
    "anteil_grundkost_heizen": {
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
    },
    # "prozent_grundkosten_warmwasser"
    "anteil_grundkost_warmwasser": {
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
        "Grundkosten",
        "Grundkosten Warmwasser",
        "Grundk.",
        "Grundk. Warmwasser",
        "Warmwasserkosten",
        "Warmwasserkosten (Wassererwärmungskosten)",
        "Warmwasser",
        "Kosten für Warmwasser",
    },
    # not used in layout model (ONLY textually), cf. KEYS
    "energietraeger": {
        "Gas",
        "bwGas",
        "Erdgas",
        "Stadtgas",
        "Fl.-Gas",
        "Flüssiggas",
        "Nahwärme",
        "Fernw.",
        "Fernwärme",
        "Fernwaerme",
        "Wärmelieferung",
        "Waermelieferung",
        "Wärmel.",
        "Waermel.",
        "Wärme",
        "Waerme",
        "Menge Wärme",
        "Heizöl",
        "Heizoel",
        "Öl",
        "Oel",
        "Strom",
        # Kohle?
        "Holzpellets",
        "Holzbriketts",
        "Pellets",
        "Scheitholz",
        "Wärmepumpe",
        "Waermepumpe",
    },
    "energietraeger_verbrauch": {
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
    },
    "energietraeger_einheit": {
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
    },
    "energietraeger_kosten": {
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
    },
    "gebaeude_flaeche": {
        "qm", "m2", "m²", "Quadratmeter",
        "Nutzfläche",
        "Fläche",
        "Wohnfläche",
        "Gesamteinheiten",
        "der Liegenschaft",
    },
    "wohnung_flaeche": {
        "qm", "m2", "m²", "Quadratmeter",
        "anteilige Einheiten",
        "Einheiten",
        "Ihre Einheiten",
        "Ihre Fläche",
        "Ihr Flächenanteil",
        "beheizb. Wohnfl.",
        "Nutzfläche",
        "Wohnfläche",
    },
    "gebaeude_verbrauchseinheiten": {
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
    },
    "wohnung_verbrauchseinheiten": {
        "Einheiten",
        "Ihre Einheiten",
        "Ihre",
        #"Ihre Kosten", # really?
        "Verbrauchskosten",
        "kwh", "kWh", "MWh",
        "Striche",
    },
    "gebaeude_warmwasser_verbrauch": {
        "Gesamteinheiten",
        "der Liegenschaft",
        "Kosten für Warmwasser",
        "Kosten für Warmwasser-Erwärmung",
        "verteilt über Verbrauch Warmwasserzähler",
        "Wasser",
        "Warmwasser",
        "Warmwasserkosten",
        "m³",
    },
    "gebaeude_warmwasser_verbrauch_einheit": {
        "Gesamteinheiten",
        "gesamte Einheiten",
        NUMBER,
        "Verbrauchskosten",
        "Warmwasser",
    },
    "kaltwasser_fuer_warmwasser": {
    },
    "wohnung_warmwasser_verbrauch": {
        "Einheiten",
        "Ihre Einheiten",
        "verteilt über Verbrauch Warmwasserzähler",
        "Striche",
    },
    "wohnung_warmwasser_verbrauch_einheit": {
        "Einheiten",
        "Ihre Einheiten",
        NUMBER,
    },
    # "gebaeude_grundkosten_raumwaerme",
    "gebaeude_grundkost_heizen": {
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
    },
    # "gebaeude_grundkosten_warmwasser",
    "gebaeude_grundkost_warmwasser": {
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
        #"Verbrauchsk. Warmw.",
        #"Verbrauchskosten",
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
    },
    "gebaeude_heizkosten_gesamt": {
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
    },
    "anteil_verbrauchskosten_heizen": {
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
    },
    "anteil_verbrauchskosten_warmwasser": {
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
    },
    "gebaeude_verbrauchskosten_raumwaerme": {
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
    },
    "gebaeude_verbrauchskosten_warmwasser": {
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
    },
    "wohnung_heizkosten_gesamt": {
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
        "Kosten EUR",
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
    },
    "wohnung_grundkosten_raumwaerme": {
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
        "Kosten EUR",
        "Kosten für Heizung",
        "Verteilung der Gesamtkosten",
        "Verteilung der Kosten",
        "Verteilung der Kosten auf die Nutzer für den Abrechnungsbereich Heizung",
        "Verteilung der Kosten auf die Nutzer für die Abrechnungsbereiche Heizung und Warmwasser",
        "Verteilung der Kosten für Heizung",
    },
    "wohnung_verbrauchskosten_raumwaerme": {
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
        "Kosten EUR",
        "Kosten für Heizung",
        "Verbrauchsk. Heizung",
        "Verbrk. Heizung",
        "Verbrauchskosten",
        "Verteilung der Kosten auf die Nutzer für den Abrechnungsbereich Heizung",
        "Verteilung der Kosten auf die Nutzer für die Abrechnungsbereiche Heizung und Warmwasser",
        "Verteilung der Kosten für Heizung",
        "Verteilung der Kosten",
    },
    "wohnung_grundkosten_warmwasser": {
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
        "Kosten EUR",
        "Kosten für Warmwasser",
        "Verteilung der Kosten",
        "Verteilung der Kosten für Warmwasser",
        "Warmwasser",
        "Warmwasserkosten",
    },
    "wohnung_verbrauchskosten_warmwasser": {
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
        "Kosten EUR",
        "Kosten für Warmwasser",
        "Verbrauchsk. Warmwasser",
        "Verbrauchsk. Warmw.",
        "Verbrauchskosten",
        "Verteilung der Kosten für Warmwasser",
        "Warmwasser",
        "Warmwasserkosten",
    },
    "warmwasser_temperatur": {
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
    },
    "nebenkosten_betriebsstrom": {
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
    },
    "nebenkosten_wartung_heizung": {
        "Art der Aufwendung",
        "Aufstellung der Kosten",
        "Betrag",
        "Betrag in EUR",
        "Betrag EUR",
        "€",
        "EUR",
        "in EUR",
        "Brennerservice/Kesselreinigung",
        "Brennerwartung",
        "Brennerwartung + Kessel",
        "Energiekosten Heizung",
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
    },
    "nebenkosten_messgeraet_miete": {
        "Aufstellung der Kosten",
        "Ausgaben zur gesonderten Verteilung",
        "Betrag",
        "Betrag in EUR",
        "Betrag EUR",
        "€",
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
    },
    "nebenkosten_messung_abrechnung": {
        "Ablesegebühr",
        "Abrechnungsdienst",
        "Abrechnungskosten",
        "Abrechnungsservice",
        "Aufstellung der Kosten",
        "Energiekosten Heizung",
        "Betrag",
        "Betrag in EUR",
        "Betrag EUR",
        "€",
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
    },
}

KEYS = {
    # applied here:
    "energietraeger": {
        "Gas": "erdgas",
        "bwGas": "erdgas",
        "Erdgas": "erdgas",
        "Stadtgas": "erdgas",
        "Fl.-Gas": "fluessiggas",
        "Flüssiggas": "fluessiggas",
        "Nahwärme": "fernwaerme",
        "Fernw.": "fernwaerme",
        "Fernwärme": "fernwaerme",
        "Fernwaerme": "fernwaerme",
        "Wärmelieferung": "fernwaerme",
        "Waermelieferung": "fernwaerme",
        "Wärmel.": "fernwaerme",
        "Waermel.": "fernwaerme",
        "Wärme": "fernwaerme",
        "Waerme": "fernwaerme",
        "Menge Wärme": "fernwaerme",
        "Heizöl": "heizoel",
        "Heizoel": "heizoel",
        "Öl": "heizoel",
        "Oel": "heizoel",
        # Kohle?
        "Holzpellets": "holz",
        "Holzbriketts": "holz",
        "Pellets": "holz",
        "Scheitholz": "holz",
        "Wärmepumpe": "strom",
        "Waermepumpe": "strom",
        "Strom": "strom",
    },
    # applied by postcorrect_formdata:
    "energietraeger_einheit": {
        "l": "liter",
        "Ltr.": "liter",
        "Liter": "liter",
        "cbm": "cbm",
        "m³": "cbm",
        "m3": "cbm",
        "Kubikmeter": "cbm",
        "Kilogramm": "kg",
        "kg": "kg",
        "GJ": "gj",
        "Kilowattstunde": "kwh",
        "Kilowattstunden": "kwh",
        "kWh": "kwh",
        "MWh": "mwh",
    },
    # applied by postcorrect_formdata:
    "gebaeude_warmwasser_verbrauch_einheit": {
        "cbm": "cbm",
        "m³": "cbm",
        "m3": "cbm",
        "Kubikmeter": "cbm",
        "kWh": "kwh",
        "Kilowattstunde": "kwh",
        "Kilowattstunden": "kwh",
        # ???
        "Einheiten": "einheiten",
        "Personen": "personen",
        "Zapfstellen": "zapfstellen",
        "qm": "qm",
        "m²": "qm",
        "m2": "qm",
        "Quadratmeter": "qm",
    },
    # applied by postcorrect_formdata:
    "wohnung_warmwasser_verbrauch_einheit": {
        "cbm": "cbm",
        "m³": "cbm",
        "m3": "cbm",
        "Kubikmeter": "cbm",
        "kWh": "kwh",
        "Kilowattstunde": "kwh",
        "Kilowattstunden": "kwh",
        # ???
        "Einheiten": "einheiten",
        "Personen": "personen",
        "Zapfstellen": "zapfstellen",
        "qm": "qm",
        "m²": "qm",
        "m2": "qm",
        "Quadratmeter": "qm",
    },
}

# normalize spelling
def normalize(text):
    if os.environ.get('FORMDATA_UMLAUTS', None) == '0':
        text = text.translate(str.maketrans('äöüÄÖÜß€¹²³', 'aouAOUBC123'))
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

def classify(inq, outq, threshold=95, slice=(0,None)):
    for texts, tag, segment in iter(inq.get, 'QUIT'):
        LOG = getLogger('processor.ClassifyFormDataText')
        LOG.debug("matching %s '%s' against %d candidate inputs [%d:%d]",
                  tag, segment, len(texts), slice.start, slice.stop)
        if not texts or not any(texts):
            outq.put([])
            continue
        classes = list()
        for class_id, category in enumerate(FIELDS[slice], slice.start):
            result = match(class_id, category, texts, tag=tag, segment=segment, threshold=threshold)
            if result:
                classes.append(result)
        outq.put(classes)

def match(class_id, category, texts, tag='line', segment='', threshold=95):
    LOG = getLogger('processor.ClassifyFormDataText')
    if category not in KEYWORDS:
        raise Exception("Unknown category '%s'" % category)
    for text in texts:
        if not text:
            LOG.warning("ignoring empty text in '%s'", segment)
            continue
        if NUMBER in KEYWORDS[category]:
            matches = text.translate(str.maketrans('', '', ',. ')).isnumeric()
            matcher = 'numeric'
        else:
            matches = text in KEYWORDS[category]
            matcher = 'direct'
        if matches:
            LOG.debug("%s %s match: %s in %s", matcher, tag, text, category)
            return class_id, 100, text
        # FIXME: Fuzzy string search is very different from robust OCR search:
        #        We don't want to tolerate arbitrary differences, but only
        #        graphemically likely ones; e.g.
        #        - `Ihr Kosten-` vs `Ihre Kosten`
        #        - `Verbrauchs-` vs `Verbrauch`
        #        - `100` or `100€` vs `100%`
        #        - `ab` vs `Ab-`
        if text.startswith('m') and len(text) <= 2 and (
                'm²' in KEYWORDS[category] or 'm2' in KEYWORDS[category] or
                'm³' in KEYWORDS[category] or 'm3' in KEYWORDS[category]):
            LOG.debug("exception %s match: %s~%s in %s", tag, text, "m²", category)
            return class_id, 100, text
        if text.endswith('C') and len(text) <= 2 and '°C' in KEYWORDS[category]:
            LOG.debug("exception %s match: %s~%s in %s", tag, text, "°C", category)
            return class_id, 100, text
        # fuzz scores are relative to length, but we actually
        # want to have a measure of the edit distance, or a
        # mix of both; so here we just attenuate short strings
        min_score = (1-math.exp(-len(text)-1)) * threshold
        result = fuzz_process.extractOne(text, KEYWORDS[category],
                                         processor=normalize,
                                         scorer=fuzz.QRatio, #WRatio
                                         score_cutoff=min_score)
        if result:
            key, score, _ = result
            LOG.debug("fuzzy %s match: %s~%s in %s", tag, text, key, category)
            return class_id, score, key
    return None

class ClassifyFormDataText(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(ClassifyFormDataText, self).__init__(*args, **kwargs)
        if hasattr(self, 'output_file_grp'):
            # processing context
            self.setup()

    def setup(self):
        self.taskq = mp.Queue()
        self.doneq = mp.Queue()
        self.nproc = self.parameter['num_processes']
        for i in range(self.nproc):
            # exclude bg = 0
            chunksize = math.ceil((len(FIELDS) - 1) / self.nproc)
            mp.Process(target=classify,
                       args=(self.taskq, self.doneq),
                       kwargs={'slice': slice(1 + i * chunksize,
                                              min(len(FIELDS), 1 + (i + 1) * chunksize)),
                               'threshold': self.parameter['threshold']},
                       daemon=True).start()

    def process(self):
        """Classify form field context lines from text recognition results.
        
        Open and deserialize PAGE input files and their respective images,
        then iterate over the element hierarchy down to the text line level.
        
        Get the text results (including glyph-level alternatives) of each line
        and word and search them for keywords of all categories. For each
        match close enough, annotate the class via `@custom` descriptor.
        
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
                    words = line.get_Word() or []
                    for segment in [line] + words:
                        # get OCR results (best on line/word level, n-best concatenated from glyph level)
                        segment.texts = list()
                        segment.confs = list()
                        textequivs = segment.get_TextEquiv()
                        for textequiv in textequivs:
                            segment.texts.append(textequiv.Unicode)
                            segment.confs.append(textequiv.conf)
                        # now go looking for OCR hypotheses at the glyph level
                        def cutoff(textequiv):
                            return (textequiv.conf or 1) > self.parameter['glyph_conf_cutoff']
                        topn = self.parameter['glyph_topn_cutoff']
                        if isinstance(segment, WordType):
                            glyphs = [filter(cutoff, glyph.TextEquiv[:topn])
                                      for glyph in segment.Glyph]
                            topn = self.parameter['word_topn_cutoff']
                        else:
                            glyphs = [filter(cutoff, glyph.TextEquiv[:topn])
                                      for word in segment.Word
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
                        textequivs = nbestproduct(*glyphs, key=aggconf, n=topn)
                        def glyphword(textequiv):
                            return textequiv.parent_object_.parent_object_
                        # regroup the line's (or word's) flat glyph sequence into words
                        # then join glyphs into words and words into a text:
                        segment.texts.extend([' '.join(''.join(te.Unicode for te in word
                                                               if te.Unicode)
                                                       for _, word in itertools.groupby(seq, glyphword))
                                              for seq in textequivs
                                              if any(seq)])
                        segment.confs.extend([sum(te.conf for te in seq) / (len(seq) or 1)
                                              for seq in textequivs
                                              if any(seq)])
                    # only allow sub-line level matching if there is a free (i.e. fully numeric) token
                    if not any(True for word in words
                               if any(True for text in word.texts
                                      if text.translate(str.maketrans('', '', ',.% ')).isnumeric())):
                        words = []
                    # match results against keywords of all classes
                    for segment in [line] + words:
                        if not segment.texts:
                            LOG.error("Segment '%s' on page '%s' contains no text results",
                                  segment.id, page_id)
                            continue
                        numsegments += 1
                        # run (fuzzy, deep) text classification
                        # FIXME: pass confs as well, use to weight matches somehow
                        for _ in range(self.nproc):
                            self.taskq.put((segment.texts,
                                            'word' if isinstance(segment, WordType) else
                                            'line',
                                            segment.id))
                        for class_id, score, text in itertools.chain.from_iterable(
                                self.doneq.get() for i in range(self.nproc)):
                            nummatches += 1
                            mark_segment(segment, FIELDS[class_id])
                            if FIELDS[class_id] in ['energietraeger']:
                                # classified purely textually (i.e. target segment = context segment)
                                category = FIELDS[class_id]
                                region_id = region.id + '_' + category
                                region_new = TextRegionType(id=region_id, Coords=segment.Coords, type_='other')
                                line_new = TextLineType(id=region_id + '_line', Coords=segment.Coords,
                                                        custom='subtype:target=' + category)
                                equiv_new = TextEquivType(Unicode=KEYS[category].get(text),
                                                          conf=segment.confs[0])
                                line_new.add_TextEquiv(equiv_new)
                                region_new.add_TextLine(line_new)
                                page.add_TextRegion(region_new)
                                LOG.info("Detected %s %s (p=%.2f) on page '%s'",
                                         category, region.id, score, page_id)
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

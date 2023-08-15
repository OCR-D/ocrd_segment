from __future__ import absolute_import

import os.path
from itertools import chain
from glob import glob

from ocrd_utils import (
    getLogger, concat_padded,
    make_file_id,
    assert_file_grp_cardinality,
    MIMETYPE_PAGE
)
from ocrd_models.ocrd_page import (
    TextEquivType,
    to_xml
)
from ocrd_modelfactory import page_from_file
from ocrd import Processor

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-replace-text'

class ReplaceText(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super().__init__(*args, **kwargs)

    def process(self):
        """Add TextEquiv anywhere below the page level from external text files named by segments.

        Open and deserialize PAGE input files. For each page, try to find text files matching
        ``file_glob`` which have both the page ID and some segment ID in their path name.

        For every match, insert the content of the text file as first TextEquiv of that very
        segment.

        Produce a new output file by serialising the resulting hierarchy.
        """
        LOG = getLogger('processor.ReplaceText')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)
        file_glob = self.parameter['file_glob']

        input_text_files = glob(file_glob)
        assert len(input_text_files), "file_glob '%s' does not match any path names" % file_glob
        for n, input_file in enumerate(self.input_files):
            file_id = make_file_id(input_file, self.output_file_grp)
            page_id = input_file.pageId
            LOG.info("INPUT FILE %i / %s", n, input_file.pageId or input_file.ID)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            pcgts.set_pcGtsId(file_id)
            self.add_metadata(pcgts)
            page = pcgts.get_Page()
            regions = page.get_AllRegions(classes=['Text'])
            lines = list(chain.from_iterable(
                [region.get_TextLine() for region in regions]))
            words = list(chain.from_iterable(
                [line.get_Word() for line in lines]))
            glyphs = list(chain.from_iterable(
                [word.get_Glyph() for word in words]))
            segids = { seg.id: seg for seg in glyphs + words + lines + regions }
            text_files = ([path for path in input_text_files
                           if page_id in path] or
                          [path for path in input_text_files
                           if input_file.ID in path])
            if not len(text_files):
                LOG.warning("no text file input for page %s", page_id)
            segments = set()
            for text_file in text_files:
                basename = os.path.splitext(text_file)[0]
                basename2 = os.path.splitext(basename)[0]
                segment = None
                for id_ in segids:
                    if basename.endswith(id_) or basename2.endswith(id_):
                        segment = segids[id_]
                        break
                if not segment:
                    LOG.error("no segment for text file '%s' on page '%s'", text_file, page_id)
                    continue
                with open(text_file, 'r') as text_fd:
                    text = text_fd.read().strip()
                LOG.debug("adding '%s' to '%s'", text, segment.id)
                segment.insert_TextEquiv_at(0, TextEquivType(Unicode=text))
                segments.add(segment)
            if not segments.isdisjoint(glyphs):
                nonglyphs = segments.difference(glyphs)
                LOG.info("updated %d of %d glyphs", len(segments) - len(nonglyphs), len(glyphs))
                segments.difference_update(glyphs)
            if not segments.isdisjoint(words):
                nonwords = segments.difference(words)
                LOG.info("updated %d of %d words", len(segments) - len(nonwords), len(words))
                segments.difference_update(words)
            if not segments.isdisjoint(lines):
                nonlines = segments.difference(lines)
                LOG.info("updated %d of %d lines", len(segments) - len(nonlines), len(lines))
                segments.difference_update(lines)
            if not segments.isdisjoint(regions):
                nonregions = segments.difference(regions)
                LOG.info("updated %d of %d regions", len(segments) - len(nonregions), len(regions))
                segments.difference_update(regions)
            assert len(segments) == 0

            # update METS (add the PAGE file):
            out = self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=page_id,
                local_filename=os.path.join(self.output_file_grp, file_id + '.xml'),
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts))
            LOG.info('created file ID: %s, file_grp: %s, path: %s',
                     file_id, self.output_file_grp, out.local_filename)

"""
Sentence span injection into XHTML blocks for synchronized Media Overlay highlighting.
"""

from __future__ import annotations

import html
import logging
import re
from typing import List, Optional, Tuple

from tools.audiobookgen.text.splitter import RegexSentenceSplitter, SentenceSplitter

logger = logging.getLogger(__name__)


class SentenceSpanInjector:
    """
    Injects <span id="chN_sM"> elements around sentences in EPUB XHTML content documents.
    Preserves inline markup (e.g. <em>, <strong>, <a>) whenever possible and falls back
    to safe paragraph flattening on complex cross-boundary formatting.
    """

    def __init__(self, splitter: Optional[SentenceSplitter] = None):
        self.splitter = splitter or RegexSentenceSplitter()

    def _clean_plain_text(self, inner_html: str) -> str:
        # Strip all HTML tags
        plain = re.sub(r"<[^>]+>", " ", inner_html)
        # Unescape HTML entities
        plain = html.unescape(plain)
        # Collapse whitespace
        plain = re.sub(r"\s+", " ", plain).strip()
        return plain

    def _escape_html_text(self, text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def inject_spans(
        self,
        xhtml: str,
        chapter_index: int = 1,
        lang: str = "en",
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Inject sentence spans into XHTML block elements.

        Returns:
            (marked_xhtml, [(span_id, sentence_text), ...])
        """
        if not xhtml:
            return xhtml, []

        sentence_counter = 0
        paragraph_counter = 0
        extracted_sentences: List[Tuple[str, str]] = []
        prefix = f"ch{chapter_index}"

        def replace_block(match: re.Match) -> str:
            nonlocal sentence_counter, paragraph_counter
            paragraph_counter += 1

            tag = match.group(1)
            attrs = match.group(2)
            inner_html = match.group(3)

            # Idempotency check: if already has span with id="...s..."
            if re.search(r'<span\s+[^>]*id=["\'](?:ch\d+_)?s\d+["\']', inner_html, re.IGNORECASE):
                # Extract existing spans if re-processing
                for m in re.finditer(r'<span\s+[^>]*id=["\']([^"\']+)["\'][^>]*>(.*?)</span>', inner_html, re.DOTALL | re.IGNORECASE):
                    sid = m.group(1)
                    stext = self._clean_plain_text(m.group(2))
                    if stext:
                        extracted_sentences.append((sid, stext))
                return match.group(0)

            plain_text = self._clean_plain_text(inner_html)
            if not plain_text:
                return match.group(0)

            sentences = self.splitter.split(plain_text, lang=lang)
            if not sentences:
                return match.group(0)

            # Case 1: Simple block with no inner HTML tags
            if "<" not in inner_html:
                span_parts = []
                for s_text in sentences:
                    sentence_counter += 1
                    span_id = f"{prefix}_s{sentence_counter}"
                    escaped = self._escape_html_text(s_text)
                    span_parts.append(f'<span id="{span_id}">{escaped}</span>')
                    extracted_sentences.append((span_id, s_text))
                return f"<{tag}{attrs}>{' '.join(span_parts)}</{tag}>"

            # Case 2: Exactly one sentence in the block with inline markup
            if len(sentences) == 1:
                sentence_counter += 1
                span_id = f"{prefix}_s{sentence_counter}"
                extracted_sentences.append((span_id, sentences[0]))
                return f'<{tag}{attrs}><span id="{span_id}">{inner_html}</span></{tag}>'

            # Case 3: Multiple sentences with inline markup
            # Attempt inline structure preservation
            transformed = self._try_wrap_inline_sentences(
                inner_html, sentences, prefix, sentence_counter, chapter_index, paragraph_counter
            )
            if transformed is not None:
                new_inner, new_counter, new_extracted = transformed
                sentence_counter = new_counter
                extracted_sentences.extend(new_extracted)
                return f"<{tag}{attrs}>{new_inner}</{tag}>"

            # Fallback: Paragraph flattening
            logger.warning(
                f"Falling back to flattened markup for chapter {chapter_index}, paragraph {paragraph_counter} "
                f"due to cross-boundary inline elements."
            )
            span_parts = []
            for s_text in sentences:
                sentence_counter += 1
                span_id = f"{prefix}_s{sentence_counter}"
                escaped = self._escape_html_text(s_text)
                span_parts.append(f'<span id="{span_id}">{escaped}</span>')
                extracted_sentences.append((span_id, s_text))

            return f"<{tag}{attrs}>{' '.join(span_parts)}</{tag}>"

        block_pattern = re.compile(
            r"<(p|h[1-6]|li|blockquote|div)([^>]*)>(.*?)</\1>",
            flags=re.DOTALL | re.IGNORECASE,
        )
        marked_xhtml = block_pattern.sub(replace_block, xhtml)

        return marked_xhtml, extracted_sentences

    def _try_wrap_inline_sentences(
        self,
        inner_html: str,
        sentences: List[str],
        prefix: str,
        start_counter: int,
        chapter_idx: int,
        para_idx: int,
    ) -> Optional[Tuple[str, int, List[Tuple[str, str]]]]:
        """
        Attempt to wrap sentences around or inside inline tags cleanly.
        If inline elements cross sentence boundaries in an irregular manner, returns None to trigger fallback.
        """
        counter = start_counter
        extracted: List[Tuple[str, str]] = []

        # Check for simple inline containers (e.g. <em>Stop. Now.</em>)
        m_single_tag = re.fullmatch(r"\s*<([a-zA-Z0-9]+)([^>]*)>(.*?)</\1>\s*", inner_html, re.DOTALL)
        if m_single_tag:
            itag, iattrs, icontent = m_single_tag.groups()
            if "<" not in icontent:
                # Inside is pure text with multiple sentences
                span_parts = []
                for s_text in sentences:
                    counter += 1
                    span_id = f"{prefix}_s{counter}"
                    escaped = self._escape_html_text(s_text)
                    span_parts.append(f'<span id="{span_id}">{escaped}</span>')
                    extracted.append((span_id, s_text))
                return f"<{itag}{iattrs}>{' '.join(span_parts)}</{itag}>", counter, extracted

        # If more complex, return None to trigger safe flattening fallback
        return None

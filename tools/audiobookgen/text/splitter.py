"""
Sentence splitting layer for natural TTS pacing and Media Overlay alignment.
"""

from __future__ import annotations

import re
from typing import List, Optional, Protocol, runtime_checkable

from tools.audiobookgen.text.normalizer import MultilingualNormalizer, TextNormalizer


@runtime_checkable
class SentenceSplitter(Protocol):
    """Protocol for sentence splitting implementations."""

    def split(self, text: str, lang: str = "en") -> List[str]:
        """Split text into distinct sentences preserving punctuation."""
        ...


class RegexSentenceSplitter:
    """
    Sentence splitter with abbreviation awareness, multilingual regex patterns,
    and a safety cap on maximum sentence/chunk length to avoid TTS model degeneration.
    """

    def __init__(
        self,
        normalizer: Optional[TextNormalizer] = None,
        max_chunk_chars: int = 300,
    ):
        self.normalizer = normalizer or MultilingualNormalizer()
        self.max_chunk_chars = max_chunk_chars

    def split(self, text: str, lang: str = "en") -> List[str]:
        if not text:
            return []

        # Speaker tag preservation mode (raw turn lines)
        if "<|speaker:" in text:
            turns = [t.strip() for t in text.split("\n") if t.strip()]
            return turns

        lang_clean = lang.lower().strip().replace("-", "_").split("_")[0]

        # Asian full-width sentence punctuation
        if lang_clean in ("ja", "zh"):
            parts = re.split(r"([。！？\n]+)", text.strip())
            sentences = []
            for i in range(0, len(parts) - 1, 2):
                s = (parts[i] + parts[i + 1]).strip()
                if s:
                    sentences.append(s)
            if len(parts) % 2 == 1 and parts[-1].strip():
                sentences.append(parts[-1].strip())
            return sentences if sentences else [text.strip()]

        # Normalize text to protect abbreviations (e.g. "f.eks." -> "for eksempel", "Dr." -> "Doctor")
        normalized = self.normalizer.normalize(text, lang=lang)
        # Collapse excessive internal whitespace but keep paragraphs
        normalized = re.sub(r"[ \t]+", " ", normalized).strip()
        if not normalized:
            return []

        # Split on sentence boundaries: (. ! ? … optionally followed by closing quote) followed by whitespace and capital letter, quote, or dash
        boundary_pattern = (
            r"(?:(?<=[.!?…])|(?<=[.!?…][\"'»”’]))\s+(?=[A-ZÁÉÍÓÖŐÚÜŰÄÖÜÀÈÉÌÒÙÂÊÎÔÛÇÑ„»\"'“‘—–\(\[\d])|\n+"
        )
        raw_sentences = [s.strip() for s in re.split(boundary_pattern, normalized) if s.strip()]

        if not raw_sentences:
            raw_sentences = [normalized]

        # Apply max_chunk_chars cap to prevent Fish Speech hallucination / truncation on giant runs
        final_sentences: List[str] = []
        for s in raw_sentences:
            if len(s) <= self.max_chunk_chars:
                final_sentences.append(s)
            else:
                chunks = self._split_long_chunk(s, self.max_chunk_chars)
                final_sentences.extend(chunks)

        return final_sentences

    def _split_long_chunk(self, text: str, max_chars: int) -> List[str]:
        """Split a long sentence on sub-clause boundaries (comma, semicolon, dash, space)."""
        if len(text) <= max_chars:
            return [text]

        results = []
        remaining = text
        while len(remaining) > max_chars:
            # Try to find a sub-clause break point: semicolon, colon, comma, or dash
            candidate_window = remaining[:max_chars]
            break_pos = -1

            # Check for punctuation break points in reverse order
            for sep in (";", ":", ",", "—", "–", "-", " "):
                pos = candidate_window.rfind(sep)
                if pos > max_chars // 3:  # Don't break on tiny prefix
                    break_pos = pos + (1 if sep in (";", ":", ",", "—", "–", "-") else 0)
                    break

            if break_pos == -1:
                break_pos = max_chars

            chunk = remaining[:break_pos].strip()
            if chunk:
                results.append(chunk)
            remaining = remaining[break_pos:].strip()

        if remaining:
            results.append(remaining)

        return results

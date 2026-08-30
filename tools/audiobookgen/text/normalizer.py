"""
Text normalization layer for multilingual TTS synthesis.

Divergence and resolution notes:
- voice_gen/client.py vs epub_to_audiobook_fishspeech.py:
  1. Danish (da): voice_gen included 'ca.', 'nr.', 'kl.', 'mio.', 'mia.', 'stk.', 'i.e.', 'e.g.'
     and decimal normalization for both comma and period ('12,5' and '12.5' -> '12 komma 5').
     epub_to_audiobook_fishspeech.py included 'm.fl.', 'd.v.s.', 'pga.', 'p.g.a.', 'hr.', 'Hr.', 'fr.', 'frk.'.
     Merged both: voice_gen definitions take precedence, and non-overlapping additions are preserved.
  2. Hungarian (hu): voice_gen handled currency 'Ft' -> 'forint' and percent '%' -> 'százalék',
     with decimal '12,5' -> '12 egész 5 tized'. Identical in core mappings.
  3. English (en) and European (de, fr, es, it): expands 'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'St.',
     'etc.', 'e.g.', 'i.e.', 'vs.', 'approx.', '%' -> ' percent', and '12.5' -> '12 point 5'.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class TextNormalizer(Protocol):
    """Protocol for text normalization components."""

    def normalize(self, text: str, lang: str = "en") -> str:
        """Expand abbreviations and format numbers/symbols for natural TTS."""
        ...


class MultilingualNormalizer:
    """
    Multilingual text normalizer supporting Danish (da), Hungarian (hu),
    English (en), German (de), French (fr), Spanish (es), Italian (it), etc.
    """

    def normalize(self, text: str, lang: str = "en") -> str:
        if not text:
            return ""

        lang_clean = lang.lower().strip().replace("-", "_").split("_")[0]

        if lang_clean in ("da", "danish", "dansk"):
            return self._normalize_danish(text)
        elif lang_clean in ("hu", "hungarian", "magyar"):
            return self._normalize_hungarian(text)
        elif lang_clean in ("en", "english", "de", "german", "fr", "french", "es", "spanish", "it", "italian"):
            return self._normalize_english_and_western(text)

        return text.strip()

    def _normalize_danish(self, text: str) -> str:
        da_abbrev_map = {
            r"\bf\.eks\.": "for eksempel",
            r"\bbl\.a\.": "blandt andet",
            r"\bdvs\.": "det vil sige",
            r"\bd\.v\.s\.": "det vil sige",
            r"\bosv\.": "og så videre",
            r"\bi\.e\.": "det vil sige",
            r"\be\.g\.": "for eksempel",
            r"\bca\.": "cirka",
            r"\bnr\.": "nummer",
            r"\bkl\.": "klokken",
            r"\bmio\.": "millioner",
            r"\bmia\.": "milliarder",
            r"\bm\.fl\.": "med flere",
            r"\bpga\.": "på grund af",
            r"\bp\.g\.a\.": "på grund af",
            r"\bdr\.": "doktor",
            r"\bDr\.": "Doktor",
            r"\bprof\.": "professor",
            r"\bProf\.": "Professor",
            r"\bhr\.": "herre",
            r"\bHr\.": "Herre",
            r"\bfr\.": "frue",
            r"\bfrk\.": "frøken",
            r"\bstk\.": "stykker",
            r"\bkr\.": "kroner",
            r"%": " procent",
        }
        for pattern, repl in da_abbrev_map.items():
            text = re.sub(pattern, repl, text)

        # Decimal numbers: 58,6 or 58.6 -> 58 komma 6
        text = re.sub(r"(\d+),(\d+)", r"\1 komma \2", text)
        text = re.sub(r"(\d+)\.(\d+)", r"\1 komma \2", text)
        return text.strip()

    def _normalize_hungarian(self, text: str) -> str:
        hu_abbrev_map = {
            r"\bZrt\.": "Zrt",
            r"\bKft\.": "Kft",
            r"\bNyrt\.": "Nyrt",
            r"\bBt\.": "Bt",
            r"\bdr\.": "doktor",
            r"\bDr\.": "Doktor",
            r"\bprof\.": "professzor",
            r"\bProf\.": "Professzor",
            r"\bpl\.": "például",
            r"\bkb\.": "körülbelül",
            r"\bstb\.": "és így tovább",
            r"\bvö\.": "vesd össze",
            r"\bún\.": "úgynevezett",
            r"\bFt\b": "forint",
            r"%": " százalék",
        }
        for pattern, repl in hu_abbrev_map.items():
            text = re.sub(pattern, repl, text)

        # Hungarian decimal numbers: 58,6 -> 58 egész 6 tized
        text = re.sub(r"(\d+),(\d+)", r"\1 egész \2 tized", text)
        return text.strip()

    def _normalize_english_and_western(self, text: str) -> str:
        en_abbrev_map = {
            r"\bMr\.": "Mister",
            r"\bMrs\.": "Missus",
            r"\bMs\.": "Miss",
            r"\bDr\.": "Doctor",
            r"\bProf\.": "Professor",
            r"\bSt\.": "Saint",
            r"\betc\.": "etcetera",
            r"\be\.g\.": "for example",
            r"\bi\.e\.": "that is",
            r"\bvs\.": "versus",
            r"\bapprox\.": "approximately",
            r"%": " percent",
        }
        for pattern, repl in en_abbrev_map.items():
            text = re.sub(pattern, repl, text)

        # English decimal numbers: 58.6 -> 58 point 6
        text = re.sub(r"(\d+)\.(\d+)", r"\1 point \2", text)
        return text.strip()

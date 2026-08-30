"""
EPUB 3 Media Overlays (.smil) generation.
"""

from __future__ import annotations

from typing import List

from tools.audiobookgen.models import SentenceTiming


def format_smil_clock(seconds: float) -> str:
    """Format seconds into SMIL standard clock format (HH:MM:SS.mmm)."""
    if seconds < 0:
        seconds = 0.0
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"


class SmilBuilder:
    """
    Constructs W3C EPUB 3 Media Overlay XML documents consumable by KOReader's
    epubmediaoverlay.lua parser.
    """

    @classmethod
    def build(
        cls,
        timings: List[SentenceTiming],
        xhtml_relative_path: str,
        audio_relative_path: str,
    ) -> str:
        """
        Build standard SMIL 3.0 XML document.

        Args:
            timings: List of SentenceTiming objects.
            xhtml_relative_path: Relative path from SMIL directory to the XHTML file (e.g. "../text/ch01.xhtml").
            audio_relative_path: Relative path from SMIL directory to the audio file (e.g. "../Audio/ch_001.mp3").

        Returns:
            Formatted SMIL XML string.
        """
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<smil xmlns="http://www.w3.org/ns/SMIL" xmlns:epub="http://www.idpf.org/2007/ops" version="3.0">',
            '  <body>',
            f'    <seq epub:textref="{xhtml_relative_path}">',
        ]

        for t in timings:
            par_id = f"par_{t.span_id}"
            t_src = f"{xhtml_relative_path}#{t.span_id}"
            clip_b = format_smil_clock(t.start)
            clip_e = format_smil_clock(t.end)

            lines.append(f'      <par id="{par_id}">')
            lines.append(f'        <text src="{t_src}"/>')
            lines.append(f'        <audio src="{audio_relative_path}" clipBegin="{clip_b}" clipEnd="{clip_e}"/>')
            lines.append('      </par>')

        lines.append('    </seq>')
        lines.append('  </body>')
        lines.append('</smil>')

        return "\n".join(lines)

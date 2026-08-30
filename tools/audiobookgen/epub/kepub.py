"""
KEpub unwrapper to strip Kobo-specific koboSpan, <st> elements, and style hacks.
"""

from __future__ import annotations

import re


class KepubUnwrapper:
    """
    Strips Kobo-specific reader markup (<st>, koboSpan) so clean standard XHTML
    can be processed and marked up for Media Overlays.
    """

    @classmethod
    def unwrap(cls, xhtml: str) -> str:
        if not xhtml:
            return ""

        result = xhtml

        # Remove kobostylehacks
        result = re.sub(
            r'<style\s+[^>]*class=["\']kobostylehacks["\'][^>]*>.*?</style>',
            "",
            result,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Iteratively unwrap <span class="koboSpan" id="kobo.X.Y">...</span>
        kobo_span_pattern = re.compile(
            r'<span\s+[^>]*class=["\']koboSpan["\'][^>]*>(.*?)</span>',
            flags=re.DOTALL | re.IGNORECASE,
        )
        prev = ""
        while prev != result:
            prev = result
            result = kobo_span_pattern.sub(r"\1", result)

        # Iteratively unwrap <st c="...">...</st> elements
        st_pattern = re.compile(
            r'<st\s+[^>]*>(.*?)</st>',
            flags=re.DOTALL | re.IGNORECASE,
        )
        prev = ""
        while prev != result:
            prev = result
            result = st_pattern.sub(r"\1", result)

        return result

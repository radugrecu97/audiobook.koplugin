"""
EPUB package parsing, kepub unwrapping, sentence span injection, SMIL building, and repacking.
"""

from tools.audiobookgen.epub.kepub import KepubUnwrapper
from tools.audiobookgen.epub.markup import SentenceSpanInjector
from tools.audiobookgen.epub.package import EpubPackage, ManifestItem
from tools.audiobookgen.epub.smil import SmilBuilder, format_smil_clock
from tools.audiobookgen.epub.writer import EpubWriter

__all__ = [
    "EpubPackage",
    "ManifestItem",
    "KepubUnwrapper",
    "SentenceSpanInjector",
    "SmilBuilder",
    "format_smil_clock",
    "EpubWriter",
]

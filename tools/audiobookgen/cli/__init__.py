"""
CLI entry points module.
"""

from tools.audiobookgen.cli.book import main as book_main
from tools.audiobookgen.cli.speak import main as speak_main

__all__ = [
    "speak_main",
    "book_main",
]

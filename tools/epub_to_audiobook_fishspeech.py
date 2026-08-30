#!/usr/bin/env python3
"""
EPUB to Aligned Audiobook Generator using Fish Speech & Media Overlays (SMIL).
Delegates to tools.audiobookgen.cli.book for full resumable pipeline with QC verification.
"""

from pathlib import Path
import sys

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from tools.audiobookgen.cli.book import main

if __name__ == "__main__":
    main()

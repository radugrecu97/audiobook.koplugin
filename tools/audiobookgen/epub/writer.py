"""
EPUB writer with Media Overlay manifest patching, metadata insertion, and spec-compliant repacking.
"""

from __future__ import annotations

import os
from pathlib import Path
import posixpath
import re
from typing import Dict, List, Optional
import zipfile

from tools.audiobookgen.epub.package import EpubPackage
from tools.audiobookgen.epub.smil import format_smil_clock
from tools.audiobookgen.models import ChapterResult, EpubStructureError


class EpubWriter:
    """
    Creates the final aligned EPUB 3 audiobook archive with Media Overlays,
    patching the OPF package document and storing mimetype uncompressed at offset 0.
    """

    def __init__(self, package: EpubPackage):
        self.package = package

    def write_audiobook_epub(
        self,
        output_path: Path | str,
        chapter_results: List[ChapterResult],
        total_duration: float,
    ) -> Path:
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        opf_xml = self.package.opf_xml
        opf_dir = self.package.opf_dir

        # Maps internal zip path -> (bytes, compress_type)
        files_to_write: Dict[str, tuple[bytes, int]] = {}

        # 1. Copy all non-overridden original files from source EPUB
        for name in self.package._zf.namelist():
            if name == "mimetype":
                continue
            data = self.package._zf.read(name)
            files_to_write[name] = (data, zipfile.ZIP_DEFLATED)

        # 2. Add modified XHTML, SMIL, and Audio files
        manifest_insertions: List[str] = []
        metadata_insertions: List[str] = []

        for res in chapter_results:
            ch_idx = res.chapter_index
            item_id = res.item_id
            smil_id = f"smil_{item_id}"
            audio_id = f"audio_{item_id}"

            # Determine file extensions and relative paths
            ext = ".mp3" if "mpeg" in res.audio_mime else ".ogg" if "ogg" in res.audio_mime or "opus" in res.audio_mime else ".wav"
            smil_filename = f"ch_{ch_idx:03d}.smil"
            audio_filename = f"ch_{ch_idx:03d}{ext}"

            smil_rel = f"MediaOverlays/{smil_filename}"
            audio_rel = f"Audio/{audio_filename}"

            if opf_dir:
                smil_internal = posixpath.normpath(posixpath.join(opf_dir, smil_rel))
                audio_internal = posixpath.normpath(posixpath.join(opf_dir, audio_rel))
                xhtml_internal = posixpath.normpath(posixpath.join(opf_dir, res.href))
            else:
                smil_internal = smil_rel
                audio_internal = audio_rel
                xhtml_internal = res.href

            # Store files in zip map
            files_to_write[xhtml_internal] = (res.marked_xhtml.encode("utf-8"), zipfile.ZIP_DEFLATED)
            files_to_write[smil_internal] = (res.smil_xml.encode("utf-8"), zipfile.ZIP_DEFLATED)
            # Audio is stored uncompressed to avoid CPU penalty on already compressed audio
            files_to_write[audio_internal] = (res.audio_bytes, zipfile.ZIP_STORED)

            # Patch OPF content item with media-overlay attribute
            pattern = rf'(<item\s+[^>]*\bid=["\']{re.escape(item_id)}["\'][^>]*)(/?>)'

            def add_mo(m: re.Match) -> str:
                tag_start = m.group(1)
                tag_end = m.group(2)
                if "media-overlay=" not in tag_start:
                    return f'{tag_start} media-overlay="{smil_id}"{tag_end}'
                return m.group(0)

            opf_xml = re.sub(pattern, add_mo, opf_xml, flags=re.IGNORECASE)

            # Manifest entries
            manifest_insertions.append(
                f'    <item id="{smil_id}" href="{smil_rel}" media-type="application/smil+xml"/>\n'
                f'    <item id="{audio_id}" href="{audio_rel}" media-type="{res.audio_mime}"/>\n'
            )

            # Chapter duration metadata refinement
            ch_dur_clock = format_smil_clock(res.duration)
            metadata_insertions.append(
                f'    <meta property="media:duration" refines="#{smil_id}">{ch_dur_clock}</meta>\n'
            )

        # Insert new items into <manifest>
        if manifest_insertions:
            all_manifest_entries = "".join(manifest_insertions)
            opf_xml = re.sub(
                r"(</manifest>)",
                rf"{all_manifest_entries}\1",
                opf_xml,
                count=1,
                flags=re.IGNORECASE,
            )

        # Insert media:duration and active-class into <metadata>
        book_dur_clock = format_smil_clock(total_duration)
        all_meta_entries = (
            f'    <meta property="media:duration">{book_dur_clock}</meta>\n'
            f'    <meta property="media:active-class">-epub-media-overlay-active</meta>\n'
            + "".join(metadata_insertions)
        )
        opf_xml = re.sub(
            r"(</metadata>)",
            rf"{all_meta_entries}\1",
            opf_xml,
            count=1,
            flags=re.IGNORECASE,
        )

        files_to_write[self.package.opf_internal_path] = (opf_xml.encode("utf-8"), zipfile.ZIP_DEFLATED)

        # 3. Write final EPUB zip with mimetype FIRST and STORED
        tmp_output = out_p.with_name(f".tmp_{out_p.name}_{os.getpid()}")
        with zipfile.ZipFile(tmp_output, "w") as zf:
            # 1. mimetype: uncompressed (ZIP_STORED) at offset 0
            mimetype_bytes = self.package._read_zip_bytes("mimetype") or b"application/epub+zip"
            zf.writestr("mimetype", mimetype_bytes, compress_type=zipfile.ZIP_STORED)

            # 2. Write all other files
            for internal_path, (content_bytes, compress_type) in files_to_write.items():
                zf.writestr(internal_path, content_bytes, compress_type=compress_type)

        os.replace(tmp_output, out_p)
        return out_p

"""
EPUB package inspection, container.xml parsing, OPF manifest and spine traversal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import posixpath
import re
from typing import Dict, Iterator, List, Optional, Tuple
import zipfile

from tools.audiobookgen.models import EpubStructureError


@dataclass(frozen=True)
class ManifestItem:
    """An item in the OPF <manifest>."""
    id: str
    href: str
    media_type: str
    media_overlay: Optional[str] = None


class EpubPackage:
    """
    Parses and navigates an EPUB / KEpub archive container, manifest, and spine.
    """

    def __init__(self, file_path: Path | str):
        self.file_path = Path(file_path).resolve()
        if not self.file_path.exists():
            raise EpubStructureError(f"EPUB file not found: {self.file_path}")

        self._zf = zipfile.ZipFile(self.file_path, "r")
        self.container_xml = self._read_zip_text("META-INF/container.xml")
        if not self.container_xml:
            raise EpubStructureError("Invalid EPUB: missing META-INF/container.xml")

        self.opf_internal_path = self._find_opf_path(self.container_xml)
        self.opf_dir = posixpath.dirname(self.opf_internal_path)
        self.opf_xml = self._read_zip_text(self.opf_internal_path)
        if not self.opf_xml:
            raise EpubStructureError(f"Could not read OPF file at {self.opf_internal_path}")

        self.manifest = self._parse_manifest(self.opf_xml)
        self.spine = self._parse_spine(self.opf_xml)

    def _read_zip_text(self, internal_path: str) -> Optional[str]:
        try:
            return self._zf.read(internal_path).decode("utf-8")
        except KeyError:
            return None
        except UnicodeDecodeError:
            return self._zf.read(internal_path).decode("latin-1")

    def _read_zip_bytes(self, internal_path: str) -> Optional[bytes]:
        try:
            return self._zf.read(internal_path)
        except KeyError:
            return None

    def _find_opf_path(self, container_xml: str) -> str:
        # Match full-path="path/to/content.opf"
        m = re.search(r'full-path\s*=\s*["\']([^"\']+)["\']', container_xml, re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # Fallback: look for any .opf in the archive namelist
        for name in self._zf.namelist():
            if name.lower().endswith(".opf"):
                return name

        raise EpubStructureError("Could not locate OPF package document in container.xml or archive.")

    def _parse_manifest(self, opf_xml: str) -> Dict[str, ManifestItem]:
        manifest_items: Dict[str, ManifestItem] = {}
        for m in re.finditer(r'<item\s+([^>]+)/?>', opf_xml, re.IGNORECASE):
            attrs = m.group(1)
            id_m = re.search(r'id\s*=\s*["\']([^"\']+)["\']', attrs, re.IGNORECASE)
            href_m = re.search(r'href\s*=\s*["\']([^"\']+)["\']', attrs, re.IGNORECASE)
            type_m = re.search(r'media-type\s*=\s*["\']([^"\']+)["\']', attrs, re.IGNORECASE)
            mo_m = re.search(r'media-overlay\s*=\s*["\']([^"\']+)["\']', attrs, re.IGNORECASE)

            if id_m and href_m and type_m:
                item_id = id_m.group(1)
                manifest_items[item_id] = ManifestItem(
                    id=item_id,
                    href=href_m.group(1),
                    media_type=type_m.group(1),
                    media_overlay=mo_m.group(1) if mo_m else None,
                )
        return manifest_items

    def _parse_spine(self, opf_xml: str) -> List[str]:
        spine_items: List[str] = []
        for m in re.finditer(r'<itemref\s+([^>]+)/?>', opf_xml, re.IGNORECASE):
            attrs = m.group(1)
            idref_m = re.search(r'idref\s*=\s*["\']([^"\']+)["\']', attrs, re.IGNORECASE)
            if idref_m:
                spine_items.append(idref_m.group(1))
        return spine_items

    def iter_content_documents(self) -> Iterator[Tuple[int, str, str, str, str]]:
        """
        Yields content documents in spine order.

        Yields:
            (spine_index (1-based), item_id, href, internal_zip_path, xhtml_content)
        """
        for idx, item_id in enumerate(self.spine, start=1):
            item = self.manifest.get(item_id)
            if not item:
                continue

            media_type = item.media_type.lower()
            if "html" not in media_type and "xml" not in media_type:
                continue

            if self.opf_dir:
                internal_path = posixpath.normpath(posixpath.join(self.opf_dir, item.href))
            else:
                internal_path = item.href

            content = self._read_zip_text(internal_path)
            if content is not None:
                yield idx, item.id, item.href, internal_path, content

    def close(self) -> None:
        self._zf.close()

    def __enter__(self) -> EpubPackage:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

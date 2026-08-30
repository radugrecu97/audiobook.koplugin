"""
Disk-based sentence audio cache and chapter state persistence for resumable generation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

from tools.audiobookgen.models import AudioClip, SentenceTiming, SynthesisRequest, VerificationResult


@runtime_checkable
class SentenceCache(Protocol):
    """Protocol for caching synthesized sentences and chapter generation state."""

    def compute_sentence_key(self, request: SynthesisRequest) -> str:
        """Compute unique cache key based on sentence text and all voice parameters."""
        ...

    def get_sentence(self, key: str) -> Optional[Tuple[AudioClip, Optional[Dict[str, Any]]]]:
        """Retrieve cached AudioClip and metadata if present."""
        ...

    def put_sentence(
        self,
        key: str,
        clip: AudioClip,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store synthesized AudioClip and metadata atomically."""
        ...

    def get_chapter_state(self, chapter_index: int) -> Optional[Dict[str, Any]]:
        """Retrieve chapter state dict if previously saved."""
        ...

    def save_chapter_state(
        self,
        chapter_index: int,
        state: Dict[str, Any],
        audio_bytes: bytes,
        audio_ext: str = ".mp3",
    ) -> Path:
        """Save chapter state and encoded audio file atomically."""
        ...

    def is_chapter_completed(self, chapter_index: int, fingerprint: str) -> bool:
        """Check if chapter was previously completed with matching fingerprint."""
        ...

    def load_completed_chapter_audio(self, chapter_index: int, audio_ext: str = ".mp3") -> Optional[bytes]:
        """Load encoded audio bytes for completed chapter."""
        ...


class DiskSentenceCache(SentenceCache):
    """
    Filesystem-backed sentence cache and chapter state manager.
    """

    def __init__(self, work_dir: Path | str):
        self.work_dir = Path(work_dir).resolve()
        self.cache_dir = self.work_dir / "cache"
        self.state_dir = self.work_dir / "state"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._ref_audio_hash_cache: Dict[Path, str] = {}

    def _hash_file(self, path: Optional[Path]) -> str:
        if not path:
            return "none"
        p = path.resolve()
        if p in self._ref_audio_hash_cache:
            return self._ref_audio_hash_cache[p]
        if not p.exists():
            return "missing"
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        self._ref_audio_hash_cache[p] = h
        return h

    def compute_sentence_key(self, request: SynthesisRequest) -> str:
        voice = request.voice
        ref_hash = self._hash_file(voice.ref_audio_path)
        seed = request.effective_seed

        key_components = [
            request.text.strip(),
            request.lang.strip().lower(),
            str(voice.model_id),
            ref_hash,
            str(voice.ref_text or ""),
            str(voice.style_tag),
            f"{voice.temperature:.4f}",
            f"{voice.top_p:.4f}",
            f"{voice.repetition_penalty:.4f}",
            str(seed),
            f"{voice.speed:.4f}",
            str(voice.chunk_length),
        ]
        key_raw = "|".join(key_components).encode("utf-8")
        return hashlib.sha256(key_raw).hexdigest()

    def get_sentence(self, key: str) -> Optional[Tuple[AudioClip, Optional[Dict[str, Any]]]]:
        prefix = key[:2]
        wav_path = self.cache_dir / prefix / f"{key}.wav"
        json_path = self.cache_dir / prefix / f"{key}.json"

        if not wav_path.exists():
            return None

        try:
            clip = AudioClip.from_wav_bytes(wav_path.read_bytes())
            metadata = None
            if json_path.exists():
                metadata = json.loads(json_path.read_text(encoding="utf-8"))
            return clip, metadata
        except Exception:
            return None

    def put_sentence(
        self,
        key: str,
        clip: AudioClip,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        prefix = key[:2]
        folder = self.cache_dir / prefix
        folder.mkdir(parents=True, exist_ok=True)

        wav_path = folder / f"{key}.wav"
        json_path = folder / f"{key}.json"

        # Atomic write for WAV
        wav_data = clip.to_wav_bytes()
        tmp_wav = folder / f".tmp_{key}_{os.getpid()}.wav"
        tmp_wav.write_bytes(wav_data)
        os.replace(tmp_wav, wav_path)

        # Atomic write for JSON metadata
        meta = metadata or {}
        meta["duration"] = clip.duration
        meta["sample_rate"] = clip.sample_rate
        meta["channels"] = clip.channels
        meta["sample_width"] = clip.sample_width

        tmp_json = folder / f".tmp_{key}_{os.getpid()}.json"
        tmp_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_json, json_path)

    def _chapter_stem(self, chapter_index: int) -> str:
        return f"ch_{chapter_index:04d}"

    def get_chapter_state(self, chapter_index: int) -> Optional[Dict[str, Any]]:
        state_file = self.state_dir / f"{self._chapter_stem(chapter_index)}.json"
        if not state_file.exists():
            return None
        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    def save_chapter_state(
        self,
        chapter_index: int,
        state: Dict[str, Any],
        audio_bytes: bytes,
        audio_ext: str = ".mp3",
    ) -> Path:
        stem = self._chapter_stem(chapter_index)
        ext = audio_ext if audio_ext.startswith(".") else f".{audio_ext}"
        audio_file = self.state_dir / f"{stem}{ext}"
        state_file = self.state_dir / f"{stem}.json"

        # Save audio atomically
        tmp_audio = self.state_dir / f".tmp_{stem}_{os.getpid()}{ext}"
        tmp_audio.write_bytes(audio_bytes)
        os.replace(tmp_audio, audio_file)

        # Update and save state JSON
        state["audio_path"] = str(audio_file.name)
        state["chapter_index"] = chapter_index
        state["status"] = "completed"

        tmp_state = self.state_dir / f".tmp_{stem}_{os.getpid()}.json"
        tmp_state.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_state, state_file)

        return audio_file

    def is_chapter_completed(self, chapter_index: int, fingerprint: str) -> bool:
        st = self.get_chapter_state(chapter_index)
        if not st or st.get("status") != "completed":
            return False
        if st.get("fingerprint") != fingerprint:
            return False
        audio_name = st.get("audio_path")
        if not audio_name or not (self.state_dir / audio_name).exists():
            return False
        return True

    def load_completed_chapter_audio(self, chapter_index: int, audio_ext: str = ".mp3") -> Optional[bytes]:
        st = self.get_chapter_state(chapter_index)
        if not st:
            return None
        audio_name = st.get("audio_path")
        if audio_name:
            p = self.state_dir / audio_name
            if p.exists():
                return p.read_bytes()
        stem = self._chapter_stem(chapter_index)
        ext = audio_ext if audio_ext.startswith(".") else f".{audio_ext}"
        p = self.state_dir / f"{stem}{ext}"
        if p.exists():
            return p.read_bytes()
        return None

    def clear(self) -> None:
        """Clear work directory cache and state."""
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        shutil.rmtree(self.state_dir, ignore_errors=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

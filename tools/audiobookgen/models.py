"""
Data models, frozen dataclasses, and exceptions for audiobookgen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
from pathlib import Path
from typing import Any, Dict, List, Optional
import wave


# ==============================================================================
# Exceptions
# ==============================================================================

class AudiobookGenError(Exception):
    """Base exception for all audiobookgen errors."""


class SynthesisError(AudiobookGenError):
    """Raised when TTS synthesis fails or returns invalid/empty audio."""


class EncodingError(AudiobookGenError):
    """Raised when audio encoding or conversion fails."""


class EpubStructureError(AudiobookGenError):
    """Raised when EPUB / Kepub structure is invalid or cannot be parsed."""


class VerificationError(AudiobookGenError):
    """Raised when QC verification fails and cannot be recovered."""


# ==============================================================================
# Frozen Data Models
# ==============================================================================

@dataclass(frozen=True)
class VoiceProfile:
    """Voice synthesis configuration and parameters."""
    ref_audio_path: Optional[Path] = None
    ref_text: Optional[str] = None
    style_tag: str = ""
    temperature: float = 0.68
    top_p: float = 0.85
    repetition_penalty: float = 1.05
    seed: int = 42
    speed: float = 1.0
    chunk_length: int = 700
    context_tail_seconds: float = 5.0
    model_id: str = "fishspeech-s2-pro"


@dataclass(frozen=True)
class SynthesisRequest:
    """A request to synthesize a single sentence or chunk."""
    text: str
    lang: str
    voice: VoiceProfile
    seed_override: Optional[int] = None

    @property
    def effective_seed(self) -> int:
        return self.seed_override if self.seed_override is not None else self.voice.seed


@dataclass(frozen=True)
class AudioClip:
    """Uncompressed raw PCM audio clip with audio format metadata."""
    pcm: bytes
    sample_rate: int = 44100
    channels: int = 1
    sample_width: int = 2  # 16-bit PCM = 2 bytes per sample

    @property
    def duration(self) -> float:
        bytes_per_frame = self.channels * self.sample_width
        if bytes_per_frame <= 0 or self.sample_rate <= 0:
            return 0.0
        return len(self.pcm) / float(self.sample_rate * bytes_per_frame)

    def to_wav_bytes(self) -> bytes:
        """Serialize PCM data to standard WAV container bytes."""
        bio = io.BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(self.pcm)
        return bio.getvalue()

    @classmethod
    def from_wav_bytes(cls, wav_bytes: bytes) -> AudioClip:
        """Create an AudioClip from standard WAV bytes."""
        bio = io.BytesIO(wav_bytes)
        with wave.open(bio, "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            pcm = wf.readframes(wf.getnframes())
        return cls(pcm=pcm, sample_rate=sample_rate, channels=channels, sample_width=sample_width)


@dataclass(frozen=True)
class SentenceTiming:
    """Synchronized timing for a sentence span."""
    span_id: str
    text: str
    start: float  # seconds, chapter-relative
    end: float    # seconds, chapter-relative (excludes trailing pause)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SentenceTiming:
        return cls(
            span_id=data["span_id"],
            text=data["text"],
            start=float(data["start"]),
            end=float(data["end"]),
        )


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of QC verification for a synthesized audio clip."""
    ok: bool
    score: float
    transcript: str
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "score": self.score,
            "transcript": self.transcript,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VerificationResult:
        return cls(
            ok=bool(data["ok"]),
            score=float(data["score"]),
            transcript=str(data["transcript"]),
            reason=str(data.get("reason", "")),
        )


@dataclass(frozen=True)
class ChapterResult:
    """Full synthesis result for an EPUB chapter document."""
    chapter_index: int
    item_id: str
    href: str
    marked_xhtml: str
    smil_xml: str
    audio_bytes: bytes
    audio_mime: str
    duration: float
    timings: List[SentenceTiming]

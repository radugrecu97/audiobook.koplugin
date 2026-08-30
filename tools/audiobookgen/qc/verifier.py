"""
Quality Control (QC) synthesis verification layer.
"""

from __future__ import annotations

import io
import re
import tempfile
from typing import List, Optional, Protocol, Tuple, runtime_checkable

from tools.audiobookgen.models import AudioClip, VerificationResult


@runtime_checkable
class SynthesisVerifier(Protocol):
    """Protocol for validating quality, completeness, and accuracy of synthesized audio."""

    def verify(self, clip: AudioClip, expected_text: str, lang: str = "en") -> VerificationResult:
        """
        Verify that synthesized audio matches expected text.

        Returns:
            VerificationResult with ok=True/False, CER score, transcript, and reason.
        """
        ...


class NullVerifier:
    """Pass-through verifier when QC is disabled."""

    def verify(self, clip: AudioClip, expected_text: str, lang: str = "en") -> VerificationResult:
        return VerificationResult(
            ok=True,
            score=0.0,
            transcript=expected_text,
            reason="QC disabled",
        )


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute character-level Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    v0 = list(range(len(s2) + 1))
    v1 = [0] * (len(s2) + 1)

    for i in range(len(s1)):
        v1[0] = i + 1
        for j in range(len(s2)):
            cost = 0 if s1[i] == s2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0 = list(v1)

    return v0[len(s2)]


def normalize_for_qc(text: str) -> str:
    """Normalize punctuation and whitespace for fair character comparison."""
    t = text.lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def check_ngram_repetition(text: str, n: int = 4, max_repeats: int = 3) -> Optional[str]:
    """Check if any n-gram word sequence repeats more than max_repeats times."""
    words = text.lower().split()
    if len(words) < n * 2:
        return None

    counts: dict[Tuple[str, ...], int] = {}
    for i in range(len(words) - n + 1):
        gram = tuple(words[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] > max_repeats:
            return " ".join(gram)
    return None


class WhisperVerifier:
    """
    QC verifier using faster-whisper (lazy loaded).
    Performs CER comparison, duration sanity checks, and hallucination / repetition detection.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "default",
        max_cer: float = 0.15,
        min_sec_per_char: float = 0.03,
        max_sec_per_char: float = 0.20,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.max_cer = max_cer
        self.min_sec_per_char = min_sec_per_char
        self.max_sec_per_char = max_sec_per_char
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except ImportError as e:
                raise ImportError(
                    "faster-whisper is required for QC verification. Install with: pip install faster-whisper"
                ) from e
        return self._model

    def transcribe(self, clip: AudioClip, lang: str = "en") -> str:
        """Transcribe AudioClip using Whisper."""
        model = self._get_model()
        wav_bytes = clip.to_wav_bytes()
        bio = io.BytesIO(wav_bytes)

        lang_code = lang.lower().split("-")[0].split("_")[0]
        segments, info = model.transcribe(
            bio,
            language=lang_code if lang_code in ("en", "da", "hu", "de", "fr", "es", "it") else None,
            beam_size=5,
        )
        transcript = " ".join([seg.text for seg in segments]).strip()
        return transcript

    def verify(self, clip: AudioClip, expected_text: str, lang: str = "en") -> VerificationResult:
        dur = clip.duration
        char_count = max(1, len(expected_text.strip()))

        # Check (b): Duration sanity checks
        sec_per_char = dur / char_count
        if sec_per_char < self.min_sec_per_char:
            return VerificationResult(
                ok=False,
                score=1.0,
                transcript="",
                reason=f"Audio truncated or too short: {sec_per_char:.3f}s/char (< {self.min_sec_per_char}s/char)",
            )
        if sec_per_char > self.max_sec_per_char:
            return VerificationResult(
                ok=False,
                score=1.0,
                transcript="",
                reason=f"Audio runaway / stuck generation: {sec_per_char:.3f}s/char (> {self.max_sec_per_char}s/char)",
            )

        # Transcribe audio
        try:
            transcript = self.transcribe(clip, lang=lang)
        except Exception as e:
            return VerificationResult(
                ok=False,
                score=1.0,
                transcript="",
                reason=f"Transcription failed: {e}",
            )

        # Check (c): Repetition detection
        repeated_gram = check_ngram_repetition(transcript, n=4, max_repeats=3)
        if repeated_gram:
            return VerificationResult(
                ok=False,
                score=0.9,
                transcript=transcript,
                reason=f"Hallucinatory repetition detected: 4-gram '{repeated_gram}' repeated > 3 times",
            )

        # Check (a): CER computation
        norm_expected = normalize_for_qc(expected_text)
        norm_transcript = normalize_for_qc(transcript)
        dist = levenshtein_distance(norm_expected, norm_transcript)
        denom = max(1, len(norm_expected))
        cer = dist / float(denom)

        if cer > self.max_cer:
            return VerificationResult(
                ok=False,
                score=cer,
                transcript=transcript,
                reason=f"CER {cer:.3f} exceeded threshold {self.max_cer:.3f}",
            )

        return VerificationResult(
            ok=True,
            score=cer,
            transcript=transcript,
            reason="Passed QC",
        )

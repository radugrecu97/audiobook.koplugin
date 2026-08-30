"""
TTS Provider protocol definition.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tools.audiobookgen.models import AudioClip, SynthesisRequest


@runtime_checkable
class TTSProvider(Protocol):
    """Protocol for Text-to-Speech synthesis engines."""

    def synthesize(self, request: SynthesisRequest) -> AudioClip:
        """
        Synthesize text into raw uncompressed PCM AudioClip.

        Raises:
            SynthesisError: If synthesis fails or returns empty/invalid audio.
        """
        ...

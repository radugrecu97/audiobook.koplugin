"""
Audio encoding layer for MP3, WAV, and Opus.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import Optional, Protocol, Tuple, runtime_checkable

from tools.audiobookgen.models import EncodingError


@runtime_checkable
class AudioEncoder(Protocol):
    """Protocol for audio encoders converting raw WAV bytes into compressed audiobook audio."""

    def encode(self, wav_bytes: bytes) -> Tuple[bytes, str]:
        """
        Encode WAV bytes into final audio format.

        Returns:
            Tuple of (encoded_audio_bytes, mime_type).

        Raises:
            EncodingError: If encoding fails or required tool (ffmpeg) is missing.
        """
        ...


def run_ffmpeg_encode(cmd: list[str], input_bytes: bytes) -> bytes:
    """Shared helper to run ffmpeg subprocess and handle errors cleanly."""
    ffmpeg_bin = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    if not os.path.exists(ffmpeg_bin):
        raise EncodingError("ffmpeg binary not found. Please install ffmpeg to encode audio.")

    cmd = [ffmpeg_bin] + cmd[1:]
    try:
        proc = subprocess.run(
            cmd,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        if not proc.stdout:
            raise EncodingError("ffmpeg produced 0 bytes of output.")
        return proc.stdout
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="ignore") if e.stderr else str(e)
        raise EncodingError(f"ffmpeg encoding failed ({e.returncode}): {err}") from e
    except Exception as e:
        raise EncodingError(f"Failed to execute ffmpeg: {e}") from e


class Mp3Encoder:
    """Encodes WAV to MP3 using ffmpeg libmp3lame."""

    def __init__(
        self,
        bitrate: str = "64k",
        channels: int = 1,
        sample_rate: int = 22050,
        quality: Optional[int] = None,
    ):
        self.bitrate = bitrate
        self.channels = channels
        self.sample_rate = sample_rate
        self.quality = quality

    def encode(self, wav_bytes: bytes) -> Tuple[bytes, str]:
        if not wav_bytes:
            return b"", "audio/mpeg"

        cmd = ["ffmpeg", "-y", "-i", "pipe:0", "-codec:a", "libmp3lame"]
        if self.quality is not None:
            cmd.extend(["-q:a", str(self.quality)])
        elif self.bitrate:
            cmd.extend(["-b:a", self.bitrate])

        if self.sample_rate:
            cmd.extend(["-ar", str(self.sample_rate)])
        if self.channels:
            cmd.extend(["-ac", str(self.channels)])

        cmd.extend(["-f", "mp3", "pipe:1"])

        out_bytes = run_ffmpeg_encode(cmd, wav_bytes)
        return out_bytes, "audio/mpeg"


class OpusEncoder:
    """Encodes WAV to Opus using ffmpeg libopus."""

    def __init__(
        self,
        bitrate: str = "32k",
        channels: int = 1,
        sample_rate: int = 24000,
    ):
        self.bitrate = bitrate
        self.channels = channels
        self.sample_rate = sample_rate

    def encode(self, wav_bytes: bytes) -> Tuple[bytes, str]:
        if not wav_bytes:
            return b"", "audio/ogg"

        cmd = [
            "ffmpeg", "-y", "-i", "pipe:0",
            "-codec:a", "libopus",
            "-b:a", self.bitrate,
            "-ar", str(self.sample_rate),
            "-ac", str(self.channels),
            "-f", "ogg", "pipe:1",
        ]
        out_bytes = run_ffmpeg_encode(cmd, wav_bytes)
        return out_bytes, "audio/ogg"


class WavEncoder:
    """Passes through WAV bytes or normalizes sample rate / channels."""

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels

    def encode(self, wav_bytes: bytes) -> Tuple[bytes, str]:
        if not wav_bytes:
            return b"", "audio/wav"

        if not self.sample_rate and not self.channels:
            return wav_bytes, "audio/wav"

        cmd = ["ffmpeg", "-y", "-i", "pipe:0"]
        if self.sample_rate:
            cmd.extend(["-ar", str(self.sample_rate)])
        if self.channels:
            cmd.extend(["-ac", str(self.channels)])
        cmd.extend(["-f", "wav", "pipe:1"])

        out_bytes = run_ffmpeg_encode(cmd, wav_bytes)
        return out_bytes, "audio/wav"

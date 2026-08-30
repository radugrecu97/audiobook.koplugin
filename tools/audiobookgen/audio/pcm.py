"""
PCM audio manipulation: buffer accumulation, silence generation, and ffmpeg speed scaling.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import shutil
import subprocess
import wave
from typing import List, Optional

from tools.audiobookgen.models import AudioClip, EncodingError


def silence(
    duration: float,
    sample_rate: int = 44100,
    channels: int = 1,
    sample_width: int = 2,
) -> AudioClip:
    """Generate exact silence PCM clip."""
    if duration <= 0:
        return AudioClip(pcm=b"", sample_rate=sample_rate, channels=channels, sample_width=sample_width)
    frame_count = int(round(sample_rate * duration))
    num_bytes = frame_count * channels * sample_width
    return AudioClip(
        pcm=b"\x00" * num_bytes,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
    )


def apply_speed_scale(clip: AudioClip, speed: float) -> AudioClip:
    """
    Adjust playback speed of AudioClip without changing pitch using ffmpeg atempo filters.
    """
    if abs(speed - 1.0) < 0.01 or speed <= 0 or not clip.pcm:
        return clip

    ffmpeg_bin = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    if not os.path.exists(ffmpeg_bin):
        raise EncodingError(f"ffmpeg is required for speed scaling ({speed}x), but was not found.")

    # Build atempo filter chain (atempo only accepts 0.5 <= rate <= 2.0 per filter)
    filters = []
    curr = float(speed)
    while curr < 0.5:
        filters.append("atempo=0.5")
        curr /= 0.5
    while curr > 2.0:
        filters.append("atempo=2.0")
        curr /= 2.0
    filters.append(f"atempo={curr:.5f}")
    filter_str = ",".join(filters)

    cmd = [
        ffmpeg_bin, "-y", "-i", "pipe:0",
        "-filter:a", filter_str,
        "-f", "wav", "pipe:1"
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=clip.to_wav_bytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return AudioClip.from_wav_bytes(proc.stdout)
    except subprocess.CalledProcessError as exc:
        err_msg = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
        raise EncodingError(f"ffmpeg speed scaling failed: {err_msg}") from exc


class PcmBuffer:
    """Accumulate PCM audio clips, enforcing format consistency."""

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        sample_width: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self._frames: List[bytes] = []

    def append(self, clip: AudioClip) -> None:
        """Append an AudioClip to the buffer, checking or setting format."""
        if not clip.pcm:
            return

        if self.sample_rate is None:
            self.sample_rate = clip.sample_rate
            self.channels = clip.channels
            self.sample_width = clip.sample_width
        elif (
            self.sample_rate != clip.sample_rate
            or self.channels != clip.channels
            or self.sample_width != clip.sample_width
        ):
            # If formats differ, resample via ffmpeg
            clip = self._resample_clip(clip, self.sample_rate, self.channels, self.sample_width)

        self._frames.append(clip.pcm)

    def _resample_clip(
        self, clip: AudioClip, target_sr: int, target_ch: int, target_sw: int
    ) -> AudioClip:
        ffmpeg_bin = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
        if not os.path.exists(ffmpeg_bin):
            raise EncodingError("ffmpeg required to resample mismatched audio clip formats.")

        sample_fmt = "s16le" if target_sw == 2 else "s32le" if target_sw == 4 else "u8"
        cmd = [
            ffmpeg_bin, "-y", "-i", "pipe:0",
            "-ar", str(target_sr),
            "-ac", str(target_ch),
            "-c:a", f"pcm_{sample_fmt}",
            "-f", "wav", "pipe:1",
        ]
        proc = subprocess.run(
            cmd,
            input=clip.to_wav_bytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return AudioClip.from_wav_bytes(proc.stdout)

    def to_clip(self) -> AudioClip:
        pcm_bytes = b"".join(self._frames)
        sr = self.sample_rate or 44100
        ch = self.channels or 1
        sw = self.sample_width or 2
        return AudioClip(pcm=pcm_bytes, sample_rate=sr, channels=ch, sample_width=sw)

    def to_wav_bytes(self) -> bytes:
        return self.to_clip().to_wav_bytes()

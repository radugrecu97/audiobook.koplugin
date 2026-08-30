"""
Chapter audio assembler combining individual sentence audio clips, silence pauses,
and calculating precise sentence timeline offsets.
"""

from __future__ import annotations

from typing import List, Tuple

from tools.audiobookgen.audio.pause import FixedPause, PausePolicy
from tools.audiobookgen.audio.pcm import PcmBuffer, silence
from tools.audiobookgen.models import AudioClip, SentenceTiming


class ChapterAudioAssembler:
    """
    Assembles synthesized sentence clips into a single chapter WAV audio track,
    inserting exact silence pauses and generating accurate SentenceTiming metadata.
    """

    def __init__(self, default_pause_policy: PausePolicy | None = None):
        self.default_pause_policy = default_pause_policy or FixedPause(0.45)

    def assemble(
        self,
        items: List[Tuple[str, str, AudioClip]],
        pause_policy: PausePolicy | None = None,
    ) -> Tuple[bytes, List[SentenceTiming]]:
        """
        Assemble sentence clips into full chapter WAV bytes and sentence timings.

        Args:
            items: List of (span_id, text, clip) tuples.
            pause_policy: Policy for pauses between sentences (or uses default).

        Returns:
            Tuple of (combined_wav_bytes, list_of_sentence_timings).
        """
        if not items:
            return b"", []

        policy = pause_policy or self.default_pause_policy
        total_items = len(items)

        # Detect audio format from first non-empty clip
        first_clip = items[0][2]
        sr = first_clip.sample_rate
        ch = first_clip.channels
        sw = first_clip.sample_width

        buffer = PcmBuffer(sample_rate=sr, channels=ch, sample_width=sw)
        timings: List[SentenceTiming] = []
        current_time = 0.0

        for idx, (span_id, text, clip) in enumerate(items):
            clip_dur = clip.duration
            start_time = current_time
            end_time = start_time + clip_dur

            # Append speech audio
            buffer.append(clip)

            # Record timing: start to end (excludes trailing pause so highlight clears during pause)
            timings.append(
                SentenceTiming(
                    span_id=span_id,
                    text=text,
                    start=start_time,
                    end=end_time,
                )
            )

            # Determine pause after this sentence
            pause_dur = policy.pause_after(text, idx, total_items)
            if pause_dur > 0:
                silence_clip = silence(
                    duration=pause_dur,
                    sample_rate=sr,
                    channels=ch,
                    sample_width=sw,
                )
                buffer.append(silence_clip)
                current_time = end_time + silence_clip.duration
            else:
                current_time = end_time

        return buffer.to_wav_bytes(), timings

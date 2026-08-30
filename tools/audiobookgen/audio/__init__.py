"""
Audio processing, buffering, pause policy, assembling, and encoding module.
"""

from tools.audiobookgen.audio.assembler import ChapterAudioAssembler
from tools.audiobookgen.audio.encoder import (
    AudioEncoder,
    Mp3Encoder,
    OpusEncoder,
    WavEncoder,
    run_ffmpeg_encode,
)
from tools.audiobookgen.audio.pause import (
    FixedPause,
    PausePolicy,
    PunctuationAwarePause,
    VariancePause,
)
from tools.audiobookgen.audio.pcm import PcmBuffer, apply_speed_scale, silence

__all__ = [
    "PcmBuffer",
    "apply_speed_scale",
    "silence",
    "PausePolicy",
    "FixedPause",
    "VariancePause",
    "PunctuationAwarePause",
    "ChapterAudioAssembler",
    "AudioEncoder",
    "Mp3Encoder",
    "OpusEncoder",
    "WavEncoder",
    "run_ffmpeg_encode",
]

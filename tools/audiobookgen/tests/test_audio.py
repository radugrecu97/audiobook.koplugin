"""
Unit tests for audio manipulation, buffering, pauses, assembler timing, and encoders.
"""

import pytest

from tools.audiobookgen.audio.assembler import ChapterAudioAssembler
from tools.audiobookgen.audio.encoder import Mp3Encoder, WavEncoder
from tools.audiobookgen.audio.pause import FixedPause, PunctuationAwarePause, VariancePause
from tools.audiobookgen.audio.pcm import PcmBuffer, apply_speed_scale, silence
from tools.audiobookgen.models import AudioClip, SentenceTiming


def test_audio_clip_properties():
    sr = 22050
    ch = 1
    sw = 2
    # 1 second of audio = 22050 frames * 2 bytes = 44100 bytes
    pcm = b"\x00" * (sr * ch * sw)
    clip = AudioClip(pcm=pcm, sample_rate=sr, channels=ch, sample_width=sw)
    assert abs(clip.duration - 1.0) < 1e-4

    # WAV serialization and deserialization
    wav_bytes = clip.to_wav_bytes()
    restored = AudioClip.from_wav_bytes(wav_bytes)
    assert restored.sample_rate == sr
    assert restored.channels == ch
    assert restored.sample_width == sw
    assert len(restored.pcm) == len(pcm)
    assert abs(restored.duration - 1.0) < 1e-4


def test_silence_generation():
    clip = silence(duration=0.5, sample_rate=44100, channels=1, sample_width=2)
    assert abs(clip.duration - 0.5) < 1e-4
    assert len(clip.pcm) == 44100  # 44100 * 0.5 * 1 * 2


def test_pcm_buffer():
    buf = PcmBuffer(sample_rate=22050, channels=1, sample_width=2)
    clip1 = silence(duration=1.0, sample_rate=22050, channels=1, sample_width=2)
    clip2 = silence(duration=2.0, sample_rate=22050, channels=1, sample_width=2)

    buf.append(clip1)
    buf.append(clip2)

    combined = buf.to_clip()
    assert abs(combined.duration - 3.0) < 1e-4


def test_pause_policies():
    # Fixed pause
    fixed = FixedPause(duration=0.4)
    assert fixed.pause_after("Sentence one.", index=0, total=2) == 0.4
    assert fixed.pause_after("Sentence two.", index=1, total=2) == 0.0

    # Variance pause (deterministic with seed)
    var_pause = VariancePause(base_duration=0.5, variance=0.1, seed=42)
    p1 = var_pause.pause_after("First.", 0, 3)
    p2 = var_pause.pause_after("Second.", 1, 3)
    assert 0.4 <= p1 <= 0.6
    assert 0.4 <= p2 <= 0.6
    assert var_pause.pause_after("Last.", 2, 3) == 0.0

    # Punctuation-aware pause
    punct_pause = PunctuationAwarePause(base_duration=0.5, comma_factor=0.5, question_excl_factor=1.5, variance=0.0)
    p_period = punct_pause.pause_after("Normal sentence.", 0, 4)
    p_comma = punct_pause.pause_after("Trailing comma,", 1, 4)
    p_excl = punct_pause.pause_after("Excited sentence!", 2, 4)

    assert abs(p_period - 0.5) < 1e-3
    assert abs(p_comma - 0.25) < 1e-3
    assert abs(p_excl - 0.75) < 1e-3


def test_assembler_timing_arithmetic():
    assembler = ChapterAudioAssembler(default_pause_policy=FixedPause(0.5))

    clip1 = silence(duration=2.0, sample_rate=22050, channels=1, sample_width=2)
    clip2 = silence(duration=3.0, sample_rate=22050, channels=1, sample_width=2)
    clip3 = silence(duration=1.5, sample_rate=22050, channels=1, sample_width=2)

    items = [
        ("s1", "First sentence.", clip1),
        ("s2", "Second sentence.", clip2),
        ("s3", "Third sentence.", clip3),
    ]

    wav_bytes, timings = assembler.assemble(items, pause_policy=FixedPause(0.5))

    assert len(timings) == 3

    # Timing rule verification:
    # Sentence 1: start = 0.0, end = 2.0 (pause 0.5 follows)
    assert abs(timings[0].start - 0.0) < 1e-3
    assert abs(timings[0].end - 2.0) < 1e-3

    # Sentence 2: start = 2.5, end = 5.5 (pause 0.5 follows)
    assert abs(timings[1].start - 2.5) < 1e-3
    assert abs(timings[1].end - 5.5) < 1e-3

    # Sentence 3: start = 6.0, end = 7.5 (no trailing pause)
    assert abs(timings[2].start - 6.0) < 1e-3
    assert abs(timings[2].end - 7.5) < 1e-3

    # Total WAV duration should match exactly 7.5 seconds
    total_clip = AudioClip.from_wav_bytes(wav_bytes)
    assert abs(total_clip.duration - 7.5) < 1e-3


def test_speed_scaling():
    clip = silence(duration=1.0, sample_rate=22050, channels=1, sample_width=2)
    # Scaled 2x should be ~0.5s
    scaled = apply_speed_scale(clip, 2.0)
    assert abs(scaled.duration - 0.5) < 0.05


def test_encoders():
    clip = silence(duration=1.0, sample_rate=22050, channels=1, sample_width=2)
    wav_bytes = clip.to_wav_bytes()

    # MP3 encoder
    mp3_enc = Mp3Encoder(bitrate="64k", channels=1, sample_rate=22050)
    mp3_bytes, mime = mp3_enc.encode(wav_bytes)
    assert mime == "audio/mpeg"
    assert len(mp3_bytes) > 0

    # WAV encoder
    wav_enc = WavEncoder()
    w_bytes, w_mime = wav_enc.encode(wav_bytes)
    assert w_mime == "audio/wav"
    assert len(w_bytes) > 0

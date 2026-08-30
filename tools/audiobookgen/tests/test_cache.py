"""
Unit tests for disk sentence cache, atomic writes, and chapter state persistence.
"""

from pathlib import Path
import tempfile

from tools.audiobookgen.models import AudioClip, SentenceTiming, SynthesisRequest, VoiceProfile
from tools.audiobookgen.pipeline.cache import DiskSentenceCache


def test_cache_key_generation_and_invalidation():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskSentenceCache(tmpdir)

        voice1 = VoiceProfile(temperature=0.65, seed=42, speed=1.0)
        req1 = SynthesisRequest(text="Hello world.", lang="en", voice=voice1)
        key1 = cache.compute_sentence_key(req1)

        # Same params should produce identical key
        req1_dup = SynthesisRequest(text="Hello world.", lang="en", voice=voice1)
        assert cache.compute_sentence_key(req1_dup) == key1

        # Changing seed should change key
        voice2 = VoiceProfile(temperature=0.65, seed=43, speed=1.0)
        req2 = SynthesisRequest(text="Hello world.", lang="en", voice=voice2)
        assert cache.compute_sentence_key(req2) != key1

        # Changing temperature should change key
        voice3 = VoiceProfile(temperature=0.75, seed=42, speed=1.0)
        req3 = SynthesisRequest(text="Hello world.", lang="en", voice=voice3)
        assert cache.compute_sentence_key(req3) != key1

        # Changing speed should change key
        voice4 = VoiceProfile(temperature=0.65, seed=42, speed=1.1)
        req4 = SynthesisRequest(text="Hello world.", lang="en", voice=voice4)
        assert cache.compute_sentence_key(req4) != key1


def test_sentence_cache_put_and_get():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskSentenceCache(tmpdir)
        voice = VoiceProfile(seed=42)
        req = SynthesisRequest(text="Test sentence.", lang="en", voice=voice)
        key = cache.compute_sentence_key(req)

        # Should be None before put
        assert cache.get_sentence(key) is None

        # Store clip
        clip = AudioClip(pcm=b"\x00" * 4410, sample_rate=22050, channels=1, sample_width=2)
        cache.put_sentence(key, clip, metadata={"notes": "test"})

        # Retrieve clip
        res = cache.get_sentence(key)
        assert res is not None
        cached_clip, meta = res
        assert cached_clip.sample_rate == 22050
        assert len(cached_clip.pcm) == 4410
        assert meta["notes"] == "test"


def test_chapter_state_and_resuming():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskSentenceCache(tmpdir)

        ch_idx = 1
        fingerprint = "fp_abc_123"
        assert not cache.is_chapter_completed(ch_idx, fingerprint)

        state = {
            "item_id": "ch1",
            "href": "text/ch1.xhtml",
            "marked_xhtml": "<p>done</p>",
            "smil_xml": "<smil></smil>",
            "duration": 15.0,
            "fingerprint": fingerprint,
            "timings": [{"span_id": "s1", "text": "t", "start": 0.0, "end": 1.0}],
        }
        audio_data = b"FAKE_AUDIO_DATA_FOR_CH1"
        cache.save_chapter_state(ch_idx, state, audio_data, audio_ext=".mp3")

        # Now is_chapter_completed should be True
        assert cache.is_chapter_completed(ch_idx, fingerprint)
        # With different fingerprint should be False
        assert not cache.is_chapter_completed(ch_idx, "different_fp")

        loaded_audio = cache.load_completed_chapter_audio(ch_idx, audio_ext=".mp3")
        assert loaded_audio == audio_data

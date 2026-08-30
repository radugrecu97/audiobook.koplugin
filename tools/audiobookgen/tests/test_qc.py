"""
Unit tests for QC verification, Levenshtein distance, repetition checks, and RetryPolicy.
"""

from unittest.mock import MagicMock

from tools.audiobookgen.audio.pcm import silence
from tools.audiobookgen.models import AudioClip, SynthesisRequest, VerificationResult, VoiceProfile
from tools.audiobookgen.qc.retry import RetryPolicy
from tools.audiobookgen.qc.verifier import (
    NullVerifier,
    WhisperVerifier,
    check_ngram_repetition,
    levenshtein_distance,
    normalize_for_qc,
)
from tools.audiobookgen.tts.provider import TTSProvider


def test_levenshtein_and_normalization():
    assert levenshtein_distance("hello", "hello") == 0
    assert levenshtein_distance("kitten", "sitting") == 3
    assert normalize_for_qc("Hello, World!") == "hello world"


def test_repetition_detection():
    normal = "This is a normal sentence without excessive repetition."
    assert check_ngram_repetition(normal, n=4, max_repeats=3) is None

    repeated = "and then he said and then he said and then he said and then he said finally."
    found = check_ngram_repetition(repeated, n=4, max_repeats=3)
    assert found is not None
    assert "and then he said" in found


def test_whisper_verifier_scoring():
    verifier = WhisperVerifier(max_cer=0.15, min_sec_per_char=0.03, max_sec_per_char=0.20)
    # Mock the internal transcribe method so no neural model is downloaded in unit tests
    verifier.transcribe = MagicMock(return_value="The quick brown fox jumps over the lazy dog.")

    # 1. Good audio duration & matching transcript
    clip_good = silence(duration=2.5, sample_rate=22050)
    expected = "The quick brown fox jumps over the lazy dog."
    res = verifier.verify(clip_good, expected, lang="en")
    assert res.ok
    assert res.score == 0.0

    # 2. Too short audio (truncated)
    clip_short = silence(duration=0.1, sample_rate=22050)
    res_short = verifier.verify(clip_short, expected, lang="en")
    assert not res_short.ok
    assert "too short" in res_short.reason

    # 3. Too long audio (runaway)
    clip_long = silence(duration=20.0, sample_rate=22050)
    res_long = verifier.verify(clip_long, expected, lang="en")
    assert not res_long.ok
    assert "runaway" in res_long.reason

    # 4. Bad transcript exceeding CER
    verifier.transcribe = MagicMock(return_value="Completely different words that do not match.")
    res_bad = verifier.verify(clip_good, expected, lang="en")
    assert not res_bad.ok
    assert "exceeded threshold" in res_bad.reason


def test_retry_policy():
    class MockTTSProvider(TTSProvider):
        def __init__(self):
            self.calls = []

        def synthesize(self, request: SynthesisRequest) -> AudioClip:
            self.calls.append(request.effective_seed)
            # 1 second clip for 15 chars = 0.067 s/char (valid)
            return silence(duration=1.0, sample_rate=22050)

    class MockFlakyVerifier:
        def __init__(self):
            self.attempts = 0

        def verify(self, clip: AudioClip, expected_text: str, lang: str = "en") -> VerificationResult:
            self.attempts += 1
            if self.attempts < 2:
                return VerificationResult(ok=False, score=0.4, transcript="bad", reason="CER too high")
            return VerificationResult(ok=True, score=0.05, transcript=expected_text, reason="Passed")

    provider = MockTTSProvider()
    verifier = MockFlakyVerifier()
    retry_policy = RetryPolicy(provider=provider, verifier=verifier, max_retries=2)

    voice = VoiceProfile(seed=42)
    req = SynthesisRequest(text="Hello from test", lang="en", voice=voice)

    clip, qc, retries = retry_policy.synthesize_with_retry(req, chapter_index=1, span_id="s1")
    assert qc.ok
    assert retries == 1
    assert len(provider.calls) == 2
    # Verify second call used adjusted seed
    assert provider.calls[0] == 42
    assert provider.calls[1] == 42 + 1000 + 1

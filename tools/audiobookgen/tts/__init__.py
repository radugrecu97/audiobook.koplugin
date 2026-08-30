"""
TTS provider module.
"""

from tools.audiobookgen.tts.fishspeech import FishSpeechProvider, resolve_api_key
from tools.audiobookgen.tts.provider import TTSProvider

__all__ = [
    "TTSProvider",
    "FishSpeechProvider",
    "resolve_api_key",
]

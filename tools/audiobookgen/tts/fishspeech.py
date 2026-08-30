"""
Fish Speech HTTP API client and TTS provider.
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Dict, Optional
import urllib.error
import urllib.request
import wave

from tools.audiobookgen.models import AudioClip, SynthesisError, SynthesisRequest
from tools.audiobookgen.tts.provider import TTSProvider


def resolve_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    """Retrieve API key / token from explicit argument or environment variables."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    for var_name in ("FISH_API_KEY", "FISH_AUDIO_API_KEY", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        val = os.environ.get(var_name)
        if val and val.strip():
            return val.strip()
    return None


class FishSpeechProvider(TTSProvider):
    """
    TTS Provider for Fish Speech HTTP API (S2 Pro, v1.4, v1.5, Cloud API).
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8080/v1/tts",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.api_key = resolve_api_key(api_key)
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._ref_audio_cache: Dict[Path, str] = {}

    def _get_ref_audio_base64(self, ref_path: Optional[Path]) -> Optional[str]:
        if not ref_path:
            return None
        p = ref_path.resolve()
        if p in self._ref_audio_cache:
            return self._ref_audio_cache[p]
        if not p.exists():
            raise SynthesisError(f"Reference audio file not found: {p}")
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        self._ref_audio_cache[p] = b64
        return b64

    def synthesize(self, request: SynthesisRequest) -> AudioClip:
        text = request.text.strip()
        if not text:
            raise SynthesisError("Cannot synthesize empty text.")

        voice = request.voice
        seed = request.effective_seed

        # Format turn with speaker and style tags
        if "<|speaker:" in text:
            tagged_text = text
        elif voice.style_tag and not text.startswith("["):
            tagged_text = f"<|speaker:0|>{voice.style_tag} {text}"
        else:
            tagged_text = f"<|speaker:0|>{text}"

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "audiobookgen-fishspeech/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        audio_b64 = self._get_ref_audio_base64(voice.ref_audio_path)

        if self.server_url.endswith("/audio/speech"):
            # OpenAI-compatible endpoint
            payload: dict = {
                "model": voice.model_id,
                "input": tagged_text,
                "response_format": "wav",
                "speed": voice.speed,
            }
        else:
            # Native Fish Speech REST format
            payload = {
                "text": tagged_text,
                "format": "wav",
                "normalize": True,
                "streaming": False,
                "temperature": float(voice.temperature),
                "top_p": float(voice.top_p),
                "repetition_penalty": float(voice.repetition_penalty),
                "seed": int(seed),
                "chunk_length": int(voice.chunk_length),
            }
            if audio_b64:
                payload["references"] = [
                    {
                        "audio": audio_b64,
                        "text": voice.ref_text or "",
                    }
                ]

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.server_url,
            data=req_data,
            headers=headers,
            method="POST",
        )

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    resp_bytes = resp.read()
                    if not resp_bytes:
                        raise SynthesisError("Fish Speech server returned an empty response body.")
                    wav_bytes = self._ensure_valid_wav(resp_bytes)
                    return self._wav_bytes_to_clip(wav_bytes)
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="ignore")
                last_error = SynthesisError(f"Fish Speech API HTTP error {e.code}: {err_body}")
                # Retry on 5xx server errors
                if 500 <= e.code < 600 and attempt < self.max_retries:
                    time.sleep(self.backoff_factor * (2 ** (attempt - 1)))
                    continue
                raise last_error from e
            except urllib.error.URLError as e:
                last_error = SynthesisError(
                    f"Cannot connect to Fish Speech server at {self.server_url}: {e.reason}\n"
                    f"Ensure server is running (e.g. `python -m fish_speech.api.server --listen 127.0.0.1:8080`)"
                )
                if attempt < self.max_retries:
                    time.sleep(self.backoff_factor * (2 ** (attempt - 1)))
                    continue
                raise last_error from e
            except Exception as e:
                last_error = SynthesisError(f"Synthesis failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.backoff_factor * (2 ** (attempt - 1)))
                    continue
                raise last_error from e

        raise last_error or SynthesisError("Synthesis failed after maximum retries.")

    def _ensure_valid_wav(self, audio_bytes: bytes) -> bytes:
        """Ensure audio bytes form a valid WAV; convert via ffmpeg if server returned MP3/OGG."""
        if audio_bytes.startswith(b"RIFF"):
            return audio_bytes

        ffmpeg_bin = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
        if not os.path.exists(ffmpeg_bin):
            raise SynthesisError("Received non-WAV audio from server and ffmpeg is not available to convert it.")

        try:
            proc = subprocess.run(
                [ffmpeg_bin, "-y", "-i", "pipe:0", "-f", "wav", "pipe:1"],
                input=audio_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return proc.stdout
        except Exception as exc:
            raise SynthesisError(f"Failed to convert server audio response to WAV: {exc}") from exc

    def _wav_bytes_to_clip(self, wav_bytes: bytes) -> AudioClip:
        try:
            bio = io.BytesIO(wav_bytes)
            with wave.open(bio, "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                pcm = wf.readframes(wf.getnframes())

            if not pcm:
                raise SynthesisError("Decoded WAV contains 0 audio frames.")

            return AudioClip(
                pcm=pcm,
                sample_rate=sample_rate,
                channels=channels,
                sample_width=sample_width,
            )
        except Exception as e:
            raise SynthesisError(f"Failed to decode WAV audio clip: {e}") from e

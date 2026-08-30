"""
Composition root for EPUB aligned audiobook generation CLI.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Optional

from tools.audiobookgen.audio.assembler import ChapterAudioAssembler
from tools.audiobookgen.audio.encoder import Mp3Encoder, OpusEncoder, WavEncoder
from tools.audiobookgen.audio.pause import FixedPause, PunctuationAwarePause, VariancePause
from tools.audiobookgen.epub.kepub import KepubUnwrapper
from tools.audiobookgen.epub.markup import SentenceSpanInjector
from tools.audiobookgen.models import VoiceProfile
from tools.audiobookgen.pipeline.book import BookGenerationService
from tools.audiobookgen.pipeline.cache import DiskSentenceCache
from tools.audiobookgen.pipeline.chapter import ChapterSynthesisService
from tools.audiobookgen.pipeline.progress import ConsoleProgressReporter
from tools.audiobookgen.qc.retry import RetryPolicy
from tools.audiobookgen.qc.verifier import NullVerifier, WhisperVerifier
from tools.audiobookgen.text.normalizer import MultilingualNormalizer
from tools.audiobookgen.text.splitter import RegexSentenceSplitter
from tools.audiobookgen.tts.fishspeech import FishSpeechProvider, resolve_api_key

DEFAULT_FISH_API_URL = "http://127.0.0.1:8080/v1/tts"


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert standard EPUB / KEpub into an aligned EPUB 3 audiobook with Media Overlays."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input .epub or .kepub file.")
    parser.add_argument("-o", "--output", default=None, help="Path to output .epub file (default: <name>_audiobook.epub).")
    parser.add_argument("--api-url", default=os.environ.get("SERVER_URL", DEFAULT_FISH_API_URL), help=f"Fish Speech server URL (default: {DEFAULT_FISH_API_URL}).")
    parser.add_argument("--api-key", default=None, help="Fish Speech / HF API key (or set FISH_API_KEY / HF_TOKEN).")
    parser.add_argument("--prompt-audio", "--ref-audio", "-r", dest="prompt_audio", default=None, help="Reference voice sample (.wav) for voice cloning.")
    parser.add_argument("--prompt-text", "--ref-text", dest="prompt_text", default=None, help="Transcript text of the reference voice sample.")
    parser.add_argument("--lang", "-l", default="en", help="Language code (e.g. en, da, hu, de, fr, es, it, ja, zh).")
    parser.add_argument("--speed", "-S", type=float, default=float(os.environ.get("SPEED", "1.0")), help="Speech speed rate (e.g. 1.0 = normal, 0.9 = slower, 1.15 = faster).")
    parser.add_argument("--temperature", "-T", type=float, default=float(os.environ.get("TEMPERATURE", "0.68")), help="Sampling temperature (default: 0.68).")
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", "0.85")), help="Top-p nucleus sampling (default: 0.85).")
    parser.add_argument("--repetition-penalty", type=float, default=float(os.environ.get("REPETITION_PENALTY", "1.05")), help="Repetition penalty (default: 1.05).")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")), help="Fixed random seed for consistent voice timbre (default: 42).")
    parser.add_argument("--chunk-length", type=int, default=int(os.environ.get("CHUNK_LENGTH", "300")), help="Max chunk byte size in batch (default: 300).")
    parser.add_argument("--style-tag", "--style", "-s", dest="style_tag", default=os.environ.get("STYLE_TAG", os.environ.get("STYLE", "")), help="Style/narrator tag prepended to sentences.")
    parser.add_argument("--pacing", "-p", default=os.environ.get("PACING", "none"), choices=["none", "slow", "relaxed", "dots", "break"], help="Pacing preset.")
    parser.add_argument("--pause-duration", "-P", type=float, default=float(os.environ.get("PAUSE_DURATION", "0.45")), help="Silence duration between sentences in seconds (default: 0.45).")
    parser.add_argument("--pause-variance", "-V", type=float, default=float(os.environ.get("PAUSE_VARIANCE", "0.10")), help="Random variance around pause duration in seconds (default: 0.10).")

    # Chapter range selection
    parser.add_argument("--start-chapter", type=int, default=1, help="First chapter to process (1-based index, default: 1).")
    parser.add_argument("--end-chapter", type=int, default=None, help="Last chapter to process (optional).")

    # Resume and caching
    parser.add_argument("--resume", action="store_true", default=True, help="Resume previously interrupted run from work-dir state (default: on).")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not resume from previous state.")
    parser.add_argument("--fresh", action="store_true", help="Clear work-dir and cache before starting.")
    parser.add_argument("--work-dir", default=None, help="Working directory for sentence cache and chapter states (default: .audiobookgen/<stem>/).")

    # Quality Control
    parser.add_argument("--qc", default="off", choices=["off", "fast", "strict"], help="QC verification mode: off (disabled), fast (Whisper small), strict (Whisper large-v3).")
    parser.add_argument("--qc-model", default=None, help="Override Whisper model size for QC (e.g. tiny, small, medium, large-v3).")
    parser.add_argument("--qc-retries", type=int, default=2, help="Max synthesis retries when QC flags a sentence (default: 2).")

    # Audio Encoding
    parser.add_argument("--audio-format", default="mp3", choices=["mp3", "wav", "opus"], help="Output audio format (default: mp3).")
    parser.add_argument("--audio-bitrate", default="64k", help="Audio bitrate for MP3/Opus (default: 64k).")
    parser.add_argument("--audio-sample-rate", type=int, default=22050, help="Output audio sample rate (default: 22050 Hz).")
    parser.add_argument("--audio-channels", type=int, default=1, help="Output audio channels (default: 1 for mono).")

    # Dry-run
    parser.add_argument("--dry-run", action="store_true", help="Analyze EPUB, split sentences, estimate character counts and generation duration without calling TTS.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_cli(argv)

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = input_path.with_name(f"{input_path.stem}_audiobook.epub")

    # Determine work dir
    if args.work_dir:
        work_dir = Path(args.work_dir).resolve()
    else:
        work_dir = output_path.parent / ".audiobookgen" / input_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    cache = DiskSentenceCache(work_dir=work_dir)
    if args.fresh:
        print(f"Clearing work-dir and cache: {work_dir}")
        cache.clear()

    # Voice Profile
    ref_audio_p = Path(args.prompt_audio).resolve() if args.prompt_audio else None
    voice = VoiceProfile(
        ref_audio_path=ref_audio_p,
        ref_text=args.prompt_text,
        style_tag=args.style_tag or "",
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        speed=args.speed,
        chunk_length=args.chunk_length,
    )

    # Audio Encoder
    if args.audio_format == "opus":
        encoder = OpusEncoder(bitrate=args.audio_bitrate, channels=args.audio_channels, sample_rate=args.audio_sample_rate)
        audio_ext = ".ogg"
    elif args.audio_format == "wav":
        encoder = WavEncoder(sample_rate=args.audio_sample_rate, channels=args.audio_channels)
        audio_ext = ".wav"
    else:
        encoder = Mp3Encoder(bitrate=args.audio_bitrate, channels=args.audio_channels, sample_rate=args.audio_sample_rate)
        audio_ext = ".mp3"

    # Pause Policy
    pause_policy = VariancePause(
        base_duration=args.pause_duration,
        variance=args.pause_variance,
        seed=args.seed,
    )

    # QC Verifier
    if args.qc == "off":
        verifier = NullVerifier()
    else:
        model_name = args.qc_model or ("small" if args.qc == "fast" else "large-v3")
        try:
            verifier = WhisperVerifier(model_size=model_name)
        except Exception as e:
            print(f"Warning: Could not initialize WhisperVerifier ({e}). Falling back to NullVerifier.", file=sys.stderr)
            verifier = NullVerifier()

    # TTS Provider & Retry Policy
    provider = FishSpeechProvider(
        server_url=args.api_url,
        api_key=args.api_key,
    )
    retry_policy = RetryPolicy(
        provider=provider,
        verifier=verifier,
        max_retries=args.qc_retries if args.qc != "off" else 0,
    )

    # Text layer
    normalizer = MultilingualNormalizer()
    splitter = RegexSentenceSplitter(normalizer=normalizer)
    injector = SentenceSpanInjector(splitter=splitter)
    unwrapper = KepubUnwrapper()
    assembler = ChapterAudioAssembler(default_pause_policy=pause_policy)

    # Chapter and Book Services
    chapter_service = ChapterSynthesisService(
        tts_provider=provider,
        retry_policy=retry_policy,
        cache=cache,
        encoder=encoder,
        pause_policy=pause_policy,
        normalizer=normalizer,
        splitter=splitter,
        injector=injector,
        unwrapper=unwrapper,
        assembler=assembler,
    )

    progress_log = work_dir / "progress.log"
    reporter = ConsoleProgressReporter(log_path=progress_log)

    book_service = BookGenerationService(
        chapter_service=chapter_service,
        cache=cache,
        progress_reporter=reporter,
    )

    out = book_service.run(
        input_epub=input_path,
        output_epub=output_path,
        voice=voice,
        lang=args.lang,
        start_chapter=args.start_chapter,
        end_chapter=args.end_chapter,
        resume=args.resume,
        dry_run=args.dry_run,
        audio_ext=audio_ext,
        audio_bitrate=args.audio_bitrate,
    )

    if not args.dry_run and out:
        print(f"\nAligned audiobook ready: {out}")


if __name__ == "__main__":
    main()

"""
Composition root for single-utterance voice synthesis CLI (replaces voice_gen/client.py).
"""

from __future__ import annotations

import argparse
import base64
import io
import os
from pathlib import Path
import random
import sys
import urllib.error
import wave

from tools.audiobookgen.audio.assembler import ChapterAudioAssembler
from tools.audiobookgen.audio.pause import VariancePause
from tools.audiobookgen.audio.pcm import apply_speed_scale
from tools.audiobookgen.models import AudioClip, SynthesisRequest, VoiceProfile
from tools.audiobookgen.text.normalizer import MultilingualNormalizer
from tools.audiobookgen.text.splitter import RegexSentenceSplitter
from tools.audiobookgen.tts.fishspeech import FishSpeechProvider


def synthesize_utterance(
    text: str,
    ref_audio: str,
    ref_text: str,
    output: str,
    style: str = "",
    lang: str = "en",
    pacing: str = "none",
    pause_duration: float = 0.45,
    pause_variance: float = 0.10,
    speed: float = 1.0,
    seed: int = 42,
    temperature: float = 0.65,
    top_p: float = 0.85,
    repetition_penalty: float = 1.05,
    chunk_length: int = 300,
    server_url: str = "http://127.0.0.1:8080/v1/tts",
) -> None:
    ref_path = Path(ref_audio).resolve()
    if not ref_path.exists():
        print(f"Error: Reference audio not found at '{ref_path}'", file=sys.stderr)
        sys.exit(1)

    out_path = Path(output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    normalizer = MultilingualNormalizer()
    splitter = RegexSentenceSplitter(normalizer=normalizer)

    normalized_text = normalizer.normalize(text, lang=lang)
    sentences = splitter.split(normalized_text, lang=lang)

    pause_desc = f"{pause_duration:.2f}s"
    if pause_variance > 0:
        min_p = max(0.05, pause_duration - pause_variance)
        max_p = pause_duration + pause_variance
        pause_desc += f" (±{pause_variance:.2f}s -> range {min_p:.2f}s - {max_p:.2f}s)"

    print("==========================================================")
    print(f"Fish Speech Voice Client ({lang.upper()})")
    print(f"Voice Reference:    {ref_path}")
    print(f"Temperature:        {temperature} (timing stability)")
    print(f"Top-P:              {top_p} | Rep-Penalty: {repetition_penalty}")
    print(f"Style Tag:          {style if style else '(none - natural speaker cadence)'}")
    print(f"Speed Rate:         {speed:.2f}x" + (" (slower speech)" if speed < 1.0 else " (faster speech)" if speed > 1.0 else " (normal)"))
    print(f"Sentence Pause:     {pause_desc} [exact silence, unaffected by speed rate]")
    print(f"Seed:               {seed}")
    print(f"Synthesizing:       {normalized_text[:120]}..." if len(normalized_text) > 120 else f"Synthesizing:       {normalized_text}")
    print(f"Output File:        {out_path}")
    print("==========================================================")

    voice_profile = VoiceProfile(
        ref_audio_path=ref_path,
        ref_text=ref_text,
        style_tag=style,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
        speed=speed,
        chunk_length=chunk_length,
    )

    provider = FishSpeechProvider(server_url=server_url)
    pause_policy = VariancePause(base_duration=pause_duration, variance=pause_variance, seed=seed)
    assembler = ChapterAudioAssembler(default_pause_policy=pause_policy)

    try:
        if len(sentences) > 1 and pause_duration > 0:
            assembled_items = []
            for idx, sentence in enumerate(sentences):
                sent_seed = seed + idx if seed is not None else None
                req = SynthesisRequest(
                    text=sentence,
                    lang=lang,
                    voice=voice_profile,
                    seed_override=sent_seed,
                )
                clip = provider.synthesize(req)
                if speed != 1.0:
                    clip = apply_speed_scale(clip, speed=speed)
                assembled_items.append((f"s{idx+1}", sentence, clip))

            final_wav_bytes, _ = assembler.assemble(assembled_items, pause_policy=pause_policy)
            out_path.write_bytes(final_wav_bytes)

        else:
            # Single sentence or single batch request
            req = SynthesisRequest(
                text=normalized_text,
                lang=lang,
                voice=voice_profile,
            )
            clip = provider.synthesize(req)
            if speed != 1.0:
                clip = apply_speed_scale(clip, speed=speed)
            out_path.write_bytes(clip.to_wav_bytes())

    except urllib.error.URLError as exc:
        print(f"\nError: Unable to connect to Fish Speech API server at {server_url}", file=sys.stderr)
        print(f"Details: {exc}", file=sys.stderr)
        print("Please ensure the server is started using: ./run_server.sh\n", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\nError during synthesis: {exc}", file=sys.stderr)
        sys.exit(1)

    file_size_kb = out_path.stat().st_size / 1024
    print(f"Success! Audio saved to: {out_path} ({file_size_kb:.1f} KB)")
    print(f"To listen: aplay {out_path} (or mpv {out_path})\n")


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fish Speech API voice synthesis client.")
    parser.add_argument("positional_args", nargs="*", help="Text prompt to speak, optionally followed by output .wav path.")
    parser.add_argument("--text", "-t", type=str, default=None, help="Text to speak.")
    parser.add_argument("--file", "-f", type=str, default=None, help="Text file path to read from (or '-' for stdin).")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output .wav path.")
    parser.add_argument("--ref-audio", "-r", type=str, required=True, help="Reference audio .wav path.")
    parser.add_argument("--ref-text", type=str, required=True, help="Reference transcript.")
    parser.add_argument("--style", "-s", type=str, default=os.environ.get("STYLE", os.environ.get("STYLE_TAG", "")), help="Narrator/style expression tag.")
    parser.add_argument("--pacing", "-p", type=str, default=os.environ.get("PACING", "none"), choices=["none", "slow", "relaxed", "dots", "break"], help="Pacing preset.")
    parser.add_argument("--pause-duration", "-P", type=float, default=float(os.environ.get("PAUSE_DURATION", "0.45")), help="Base silence duration between sentences in seconds (default: 0.45).")
    parser.add_argument("--pause-variance", "-V", type=float, default=float(os.environ.get("PAUSE_VARIANCE", "0.10")), help="Random variance (+/-) around pause duration in seconds (default: 0.10).")
    parser.add_argument("--speed", "-S", type=float, default=float(os.environ.get("SPEED", "1.0")), help="Speech speed rate (e.g. 0.88 = 12%% slower, 1.0 = normal, 1.15 = faster).")
    parser.add_argument("--lang", "-l", type=str, default="en", help="Language code (en, da, hu).")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")), help="Random seed (default: 42).")
    parser.add_argument("--temperature", "-T", type=float, default=float(os.environ.get("TEMPERATURE", "0.65")), help="Sampling temperature / timing stability (default: 0.65).")
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", "0.85")), help="Top-p sampling (default: 0.85).")
    parser.add_argument("--repetition-penalty", type=float, default=float(os.environ.get("REPETITION_PENALTY", "1.05")), help="Repetition penalty (default: 1.05).")
    parser.add_argument("--chunk-length", type=int, default=int(os.environ.get("CHUNK_LENGTH", "300")), help="Max batch chunk size in bytes.")
    parser.add_argument("--server-url", type=str, default=os.environ.get("SERVER_URL", "http://127.0.0.1:8080/v1/tts"))
    parser.add_argument("--default-text", type=str, default="Hello, this is a test.")
    parser.add_argument("--default-output", type=str, default="output.wav")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_cli(argv)

    output_path = args.output
    text_parts = list(args.positional_args)

    if not output_path:
        if text_parts and text_parts[-1].endswith(".wav"):
            output_path = text_parts.pop()
        else:
            output_path = args.default_output

    if args.file:
        if args.file == "-":
            text = sys.stdin.read().strip()
        else:
            text = Path(args.file).read_text(encoding="utf-8").strip()
    elif args.text:
        text = args.text
    elif text_parts:
        text = " ".join(text_parts).strip()
    else:
        text = args.default_text

    synthesize_utterance(
        text=text,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        output=output_path,
        style=args.style,
        lang=args.lang,
        pacing=args.pacing,
        pause_duration=args.pause_duration,
        pause_variance=args.pause_variance,
        speed=args.speed,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        chunk_length=args.chunk_length,
        server_url=args.server_url,
    )


if __name__ == "__main__":
    main()

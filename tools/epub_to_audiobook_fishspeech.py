#!/usr/bin/env python3
"""
EPUB to Aligned Audiobook Generator using Fish Speech (S2 Pro / v1.4 / v1.5) & EPUB 3 Media Overlays (SMIL).

Supports:
- Local Fish Speech API server (http://127.0.0.1:8080/v1/tts or /v1/audio/speech)
- Fish Audio Cloud API (https://api.fish.audio/v1/tts)
- Zero-shot voice cloning with reference audio (--prompt-audio) and reference text (--prompt-text)
- Multi-language sentence tokenization (English, German, French, Spanish, Italian, Hungarian, Japanese, Chinese, etc.)
- W3C EPUB 3 Media Overlays (.smil) generation with <span id="sN"> markup for KOReader tap-to-seek and synchronized highlighting.

Usage:
    # 1. Start your local Fish Speech server:
    #    python -m fish_speech.api.server --listen 127.0.0.1:8080
    #
    # 2. Run the generator:
    python tools/epub_to_audiobook_fishspeech.py -i mybook.epub -o mybook_audiobook.epub --lang en
    python tools/epub_to_audiobook_fishspeech.py -i mybook.epub --prompt-audio voice_sample.wav --prompt-text "Sample transcript"
"""

import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import wave
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional

DEFAULT_FISH_API_URL = "http://127.0.0.1:8080/v1/tts"
DEFAULT_OPENAI_COMPAT_URL = "http://127.0.0.1:8080/v1/audio/speech"


def get_api_key(cli_key: Optional[str] = None) -> Optional[str]:
    """Retrieve API key / HF token from CLI or environment."""
    if cli_key and cli_key.strip():
        return cli_key.strip()
    for var in ("FISH_API_KEY", "FISH_AUDIO_API_KEY", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        val = os.environ.get(var)
        if val and val.strip():
            return val.strip()
    return None


class FishSpeechSynthesizer:
    """Communicates with a local Fish Speech API server or Fish Audio Cloud endpoint."""

    def __init__(
        self,
        api_url: str = DEFAULT_FISH_API_URL,
        api_key: Optional[str] = None,
        prompt_audio_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        speed: float = 1.0,
        model_name: str = "fishspeech-s2-pro",
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = get_api_key(api_key)
        self.speed = speed
        self.model_name = model_name
        self.prompt_audio_base64: Optional[str] = None
        self.prompt_text = prompt_text or ""

        if prompt_audio_path:
            p_path = Path(prompt_audio_path)
            if p_path.exists():
                audio_raw = p_path.read_bytes()
                self.prompt_audio_base64 = base64.b64encode(audio_raw).decode("utf-8")
                print(f"Loaded reference audio for voice cloning: {p_path} ({len(audio_raw) / 1024:.1f} KB)")
            else:
                print(f"Warning: prompt audio file not found: {prompt_audio_path}")

        print(f"Fish Speech endpoint: {self.api_url}")
        if self.api_key:
            print("Using authorization token.")

    def synthesize_to_wav_bytes(self, text: str) -> bytes:
        """Send text to Fish Speech server and return WAV audio bytes."""
        text = text.strip()
        if not text:
            return b""

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "fishspeech-epub-generator/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Detect endpoint style: OpenAI-compatible or Native Fish Speech API
        if self.api_url.endswith("/audio/speech"):
            payload = {
                "model": self.model_name,
                "input": text,
                "response_format": "wav",
                "speed": self.speed,
            }
        else:
            # Native Fish Speech REST format
            payload = {
                "text": text,
                "format": "wav",
                "normalize": True,
                "streaming": False,
                "speed": self.speed,
            }
            if self.prompt_audio_base64:
                payload["references"] = [
                    {
                        "audio": self.prompt_audio_base64,
                        "text": self.prompt_text,
                    }
                ]

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.api_url, data=req_data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                audio_bytes = resp.read()
                return self._ensure_valid_wav(audio_bytes)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Fish Speech API HTTP error {e.code}: {err_body}")
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot connect to Fish Speech server at {self.api_url}: {e.reason}\n"
                f"Make sure your Fish Speech server is running (e.g. `python -m fish_speech.api.server --listen 127.0.0.1:8080`)"
            )

    def _ensure_valid_wav(self, audio_bytes: bytes) -> bytes:
        """Ensure audio bytes form a valid WAV; convert if MP3/OGG returned."""
        if audio_bytes.startswith(b"RIFF"):
            return audio_bytes

        # If server returned MP3 or another format, convert to WAV using ffmpeg if available
        ffmpeg = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
        if ffmpeg:
            proc = subprocess.Popen(
                [ffmpeg, "-y", "-i", "pipe:0", "-f", "wav", "pipe:1"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            wav_out, _ = proc.communicate(input=audio_bytes)
            if proc.returncode == 0 and wav_out:
                return wav_out

        return audio_bytes


def split_sentences_multilingual(text: str, lang: str = "en") -> List[str]:
    """
    Split text into sentences supporting multiple languages
    (en, de, fr, es, it, hu, ja, zh, etc.).
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Japanese / Chinese full-width sentence splitters
    if lang.lower() in ("ja", "zh", "ja_jp", "zh_cn", "zh_tw"):
        parts = re.split(r"([。！？\n]+)", text)
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            s = (parts[i] + parts[i + 1]).strip()
            if s:
                sentences.append(s)
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1].strip())
        return sentences

    # European languages regex splitting
    # Split on sentence boundaries: (. ! ? …) followed by space and capital letter, quote, or dash
    pattern = r"(?<=[.!?…])\s+(?=[A-ZÁÉÍÓÖŐÚÜŰÄÖÜÀÈÉÌÒÙÂÊÎÔÛÇÑ„»\"'—–\d])"
    sents = re.split(pattern, text)
    cleaned = []
    for s in sents:
        s_clean = s.strip()
        if s_clean:
            cleaned.append(s_clean)
    return cleaned if cleaned else [text]


def format_smil_clock(seconds: float) -> str:
    """Format seconds into SMIL clock format (HH:MM:SS.mmm)."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"


def convert_wav_to_mp3(wav_bytes: bytes) -> Tuple[bytes, str]:
    """Convert WAV to MP3 if ffmpeg is available; otherwise return WAV."""
    ffmpeg = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if not ffmpeg:
        return wav_bytes, "audio/wav"

    try:
        proc = subprocess.Popen(
            [ffmpeg, "-y", "-i", "pipe:0", "-codec:a", "libmp3lame", "-q:a", "4", "-f", "mp3", "pipe:1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        mp3_bytes, stderr = proc.communicate(input=wav_bytes)
        if proc.returncode == 0 and mp3_bytes:
            return mp3_bytes, "audio/mpeg"
    except Exception:
        pass

    return wav_bytes, "audio/wav"


class ChapterProcessor:
    """Processes an XHTML chapter: splits text, calls Fish Speech, builds SMIL & marked XHTML."""

    def __init__(self, chapter_id: str, html_path: str, html_content: str, synthesizer: FishSpeechSynthesizer, lang: str = "en"):
        self.chapter_id = chapter_id
        self.html_path = html_path
        self.html_content = html_content
        self.synthesizer = synthesizer
        self.lang = lang
        self.sentences: List[Dict] = []
        self.total_duration = 0.0

    def process(self, audio_rel_path: str, chapter_num: int) -> Tuple[str, str, bytes, float]:
        sentence_counter = 0
        timing_entries = []

        def replace_block_content(match):
            nonlocal sentence_counter
            tag = match.group(1)
            attrs = match.group(2)
            inner_html = match.group(3)

            plain_text = re.sub(r"<[^>]+>", " ", inner_html)
            plain_text = re.sub(r"&nbsp;|&#160;", " ", plain_text)
            plain_text = re.sub(r"&amp;", "&", plain_text)
            plain_text = re.sub(r"&lt;", "<", plain_text)
            plain_text = re.sub(r"&gt;", ">", plain_text)
            plain_text = re.sub(r"&quot;", '"', plain_text)
            plain_text = re.sub(r"&#39;|&apos;", "'", plain_text)
            plain_text = re.sub(r"\s+", " ", plain_text).strip()

            if not plain_text:
                return match.group(0)

            sents = split_sentences_multilingual(plain_text, self.lang)
            if not sents:
                return match.group(0)

            span_parts = []
            for sent_text in sents:
                sentence_counter += 1
                span_id = f"s{sentence_counter}"
                escaped_text = (
                    sent_text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                )
                span_parts.append(f'<span id="{span_id}">{escaped_text}</span>')
                self.sentences.append({"id": span_id, "text": sent_text})

            return f"<{tag}{attrs}>{' '.join(span_parts)}</{tag}>"

        pattern = r"<(p|h[1-6]|li|blockquote|div)([^>]*)>(.*?)</\1>"
        marked_html = re.sub(pattern, replace_block_content, self.html_content, flags=re.DOTALL | re.IGNORECASE)

        if not self.sentences:
            return self.html_content, "", b"", 0.0

        current_time = 0.0
        sample_rate = 44100
        sample_width = 2
        channels = 1
        raw_frames_list = []

        print(f"  Synthesizing {len(self.sentences)} sentences with Fish Speech for chapter {chapter_num} ({self.html_path})...")
        for idx, sent in enumerate(self.sentences, 1):
            text = sent["text"]
            wav_bytes = self.synthesizer.synthesize_to_wav_bytes(text)
            if not wav_bytes:
                continue

            try:
                with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                    channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                    sample_rate = wf.getframerate()
                    n_frames = wf.getnframes()
                    raw_frames = wf.readframes(n_frames)
                    dur = n_frames / float(sample_rate)
            except Exception:
                # If non-WAV returned, estimate duration from size
                dur = max(1.0, len(text) * 0.07)
                raw_frames = b""

            if raw_frames:
                raw_frames_list.append(raw_frames)
                pause_frames = int(sample_rate * 0.25)
                raw_frames_list.append(b"\x00" * (pause_frames * sample_width * channels))

            total_sent_dur = dur + 0.25
            start_t = current_time
            end_t = current_time + dur
            current_time += total_sent_dur

            timing_entries.append({
                "id": sent["id"],
                "start": start_t,
                "end": end_t,
            })

            if idx % 5 == 0 or idx == len(self.sentences):
                sys.stdout.write(f"\r    [{idx}/{len(self.sentences)}] sentences synthesized...")
                sys.stdout.flush()
        sys.stdout.write("\n")

        self.total_duration = current_time

        combined_wav = b""
        if raw_frames_list:
            combined_wav_io = io.BytesIO()
            with wave.open(combined_wav_io, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                for f in raw_frames_list:
                    wf.writeframes(f)
            combined_wav = combined_wav_io.getvalue()

        # Build SMIL XML
        html_filename = Path(self.html_path).name
        smil_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<smil xmlns="http://www.w3.org/ns/SMIL" xmlns:epub="http://www.idpf.org/2007/ops" version="3.0">',
            '  <body>',
            f'    <seq epub:textref="../text/{html_filename}">',
        ]
        for t in timing_entries:
            par_id = f"par_{t['id']}"
            t_src = f"../text/{html_filename}#{t['id']}"
            clip_b = format_smil_clock(t["start"])
            clip_e = format_smil_clock(t["end"])
            smil_lines.append(f'      <par id="{par_id}">')
            smil_lines.append(f'        <text src="{t_src}"/>')
            smil_lines.append(f'        <audio src="{audio_rel_path}" clipBegin="{clip_b}" clipEnd="{clip_e}"/>')
            smil_lines.append('      </par>')
        smil_lines.append('    </seq>')
        smil_lines.append('  </body>')
        smil_lines.append('</smil>')
        smil_xml = "\n".join(smil_lines)

        return marked_html, smil_xml, combined_wav, self.total_duration


def process_epub_with_fishspeech(
    input_epub: Path,
    output_epub: Path,
    synthesizer: FishSpeechSynthesizer,
    lang: str = "en",
    start_chapter: int = 1,
    end_chapter: Optional[int] = None,
):
    """Main pipeline to convert EPUB using Fish Speech into EPUB 3 with Media Overlays."""
    print(f"Reading input EPUB: {input_epub}")
    temp_dir = Path(tempfile.mkdtemp(prefix="fish_epub_"))

    try:
        with zipfile.ZipFile(input_epub, "r") as zf:
            zf.extractall(temp_dir)

        container_xml = temp_dir / "META-INF" / "container.xml"
        if not container_xml.exists():
            raise RuntimeError("Invalid EPUB: missing META-INF/container.xml")

        container_content = container_xml.read_text(encoding="utf-8")
        opf_match = re.search(r'full-path\s*=\s*["\']([^"\']+)["\']', container_content)
        if not opf_match:
            raise RuntimeError("Could not find OPF path in container.xml")

        opf_rel_path = opf_match.group(1)
        opf_path = temp_dir / opf_rel_path
        opf_dir = opf_path.parent
        opf_content = opf_path.read_text(encoding="utf-8")

        manifest_items = {}
        for m in re.finditer(r'<item\s+([^>]+)/>', opf_content):
            attrs = m.group(1)
            id_m = re.search(r'id\s*=\s*["\']([^"\']+)["\']', attrs)
            href_m = re.search(r'href\s*=\s*["\']([^"\']+)["\']', attrs)
            type_m = re.search(r'media-type\s*=\s*["\']([^"\']+)["\']', attrs)
            if id_m and href_m and type_m:
                manifest_items[id_m.group(1)] = {
                    "id": id_m.group(1),
                    "href": href_m.group(1),
                    "media_type": type_m.group(1),
                }

        spine_items = []
        for m in re.finditer(r'<itemref\s+([^>]+)/>', opf_content):
            attrs = m.group(1)
            idref_m = re.search(r'idref\s*=\s*["\']([^"\']+)["\']', attrs)
            if idref_m:
                spine_items.append(idref_m.group(1))

        print(f"Found {len(spine_items)} spine documents in OPF.")

        smil_dir = opf_dir / "MediaOverlays"
        audio_dir = opf_dir / "Audio"
        smil_dir.mkdir(exist_ok=True)
        audio_dir.mkdir(exist_ok=True)

        new_manifest_items = []
        total_book_duration = 0.0
        processed_count = 0

        for ch_idx, item_id in enumerate(spine_items, 1):
            if ch_idx < start_chapter:
                continue
            if end_chapter and ch_idx > end_chapter:
                break

            item_info = manifest_items.get(item_id)
            if not item_info:
                continue

            media_type = item_info["media_type"].lower()
            if "html" not in media_type and "xml" not in media_type:
                continue

            ch_href = item_info["href"]
            ch_file_path = (opf_dir / ch_href).resolve()
            if not ch_file_path.exists():
                continue

            html_text = ch_file_path.read_text(encoding="utf-8", errors="ignore")
            processor = ChapterProcessor(item_id, ch_href, html_text, synthesizer, lang=lang)

            audio_base_name = f"ch_{ch_idx:03d}"
            marked_html, smil_xml, wav_bytes, ch_dur = processor.process(
                audio_rel_path=f"../Audio/{audio_base_name}.mp3",
                chapter_num=ch_idx,
            )

            if not smil_xml or not wav_bytes:
                continue

            final_audio_bytes, mime_type = convert_wav_to_mp3(wav_bytes)
            ext = ".mp3" if mime_type == "audio/mpeg" else ".wav"
            audio_filename = f"{audio_base_name}{ext}"
            audio_dest = audio_dir / audio_filename
            audio_dest.write_bytes(final_audio_bytes)

            if ext == ".wav":
                smil_xml = smil_xml.replace(f"{audio_base_name}.mp3", f"{audio_base_name}.wav")

            smil_filename = f"ch_{ch_idx:03d}.smil"
            smil_dest = smil_dir / smil_filename
            smil_dest.write_text(smil_xml, encoding="utf-8")

            ch_file_path.write_text(marked_html, encoding="utf-8")

            smil_item_id = f"smil_{item_id}"
            audio_item_id = f"audio_{item_id}"
            new_manifest_items.append((
                item_id,
                smil_item_id,
                f"MediaOverlays/{smil_filename}",
                audio_item_id,
                f"Audio/{audio_filename}",
                mime_type,
                ch_dur,
            ))

            total_book_duration += ch_dur
            processed_count += 1

        print(f"\nProcessed {processed_count} chapters. Total narration duration: {format_smil_clock(total_book_duration)}")

        print("Updating OPF manifest and metadata with Media Overlays...")
        for orig_id, smil_id, smil_href, audio_id, audio_href, audio_mime, ch_dur in new_manifest_items:
            pattern = rf'(<item\s+[^>]*\bid=["\']{orig_id}["\'][^>]*)(/?>)'

            def add_mo(match):
                tag_start = match.group(1)
                tag_end = match.group(2)
                if "media-overlay=" not in tag_start:
                    return f'{tag_start} media-overlay="{smil_id}"{tag_end}'
                return match.group(0)

            opf_content = re.sub(pattern, add_mo, opf_content)

            smil_entry = f'    <item id="{smil_id}" href="{smil_href}" media-type="application/smil+xml"/>\n'
            audio_entry = f'    <item id="{audio_id}" href="{audio_href}" media-type="{audio_mime}"/>\n'
            manifest_insert = smil_entry + audio_entry

            opf_content = re.sub(r"(</manifest>)", rf"{manifest_insert}\1", opf_content, count=1)

        meta_duration = (
            f'    <meta property="media:duration">{format_smil_clock(total_book_duration)}</meta>\n'
            f'    <meta property="media:active-class">-epub-media-overlay-active</meta>\n'
        )
        opf_content = re.sub(r"(</metadata>)", rf"{meta_duration}\1", opf_content, count=1)

        opf_path.write_text(opf_content, encoding="utf-8")

        print(f"Creating output EPUB: {output_epub}")
        output_epub.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(output_epub, "w") as out_zf:
            mimetype_path = temp_dir / "mimetype"
            if mimetype_path.exists():
                out_zf.write(mimetype_path, "mimetype", compress_type=zipfile.ZIP_STORED)
            else:
                out_zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)

            for root, dirs, files in os.walk(temp_dir):
                for f in files:
                    full_p = Path(root) / f
                    rel_p = full_p.relative_to(temp_dir).as_posix()
                    if rel_p == "mimetype":
                        continue
                    out_zf.write(full_p, rel_p, compress_type=zipfile.ZIP_DEFLATED)

        size_mb = output_epub.stat().st_size / (1024 * 1024)
        print(f"\nSUCCESS! Created aligned Fish Speech audiobook EPUB: {output_epub} ({size_mb:.1f} MB)")
        print("Ready to copy to your Kobo e-reader for use with audiobook.koplugin!")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Convert standard EPUB into an aligned EPUB 3 audiobook using Fish Speech (S2 Pro) API."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input .epub file")
    parser.add_argument("-o", "--output", help="Path to output .epub file (default: <name>_audiobook.epub)")
    parser.add_argument("--api-url", default=DEFAULT_FISH_API_URL, help=f"Fish Speech server URL (default: {DEFAULT_FISH_API_URL})")
    parser.add_argument("--api-key", help="Fish Speech / HF API key (or set FISH_API_KEY / HF_TOKEN in env)")
    parser.add_argument("--prompt-audio", help="Reference voice sample (.wav/.mp3) for zero-shot voice cloning")
    parser.add_argument("--prompt-text", help="Transcript text of the reference voice sample")
    parser.add_argument("--lang", default="en", help="Language code for sentence tokenization (e.g. en, de, fr, es, it, hu, ja, zh)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (e.g. 1.0 = normal, 1.2 = faster)")
    parser.add_argument("--start-chapter", type=int, default=1, help="First chapter to process (default: 1)")
    parser.add_argument("--end-chapter", type=int, default=None, help="Last chapter to process (optional)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_audiobook.epub")

    synthesizer = FishSpeechSynthesizer(
        api_url=args.api_url,
        api_key=args.api_key,
        prompt_audio_path=args.prompt_audio,
        prompt_text=args.prompt_text,
        speed=args.speed,
    )

    process_epub_with_fishspeech(
        input_path,
        output_path,
        synthesizer,
        lang=args.lang,
        start_chapter=args.start_chapter,
        end_chapter=args.end_chapter,
    )


if __name__ == "__main__":
    main()

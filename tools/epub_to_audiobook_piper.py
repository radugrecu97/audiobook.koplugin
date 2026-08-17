#!/usr/bin/env python3
"""
EPUB to Aligned Audiobook Generator using Piper TTS & EPUB 3 Media Overlays (SMIL).

Converts any standard EPUB ebook into an EPUB 3 enriched with:
- Neural TTS narration generated locally with Piper (hu_HU-anna-medium by default)
- W3C EPUB 3 Media Overlays (.smil)
- HTML sentence marking (<span id="sN">) for synchronized reading and word/sentence tap-to-seek

Usage:
    python tools/epub_to_audiobook_piper.py -i mybook.epub -o mybook_audiobook.epub
    python tools/epub_to_audiobook_piper.py -i mybook.epub --model hu_HU-anna-medium.onnx
"""

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import wave
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Default Piper Hungarian voice URLs
DEFAULT_VOICE_NAME = "hu_HU-anna-medium"
DEFAULT_VOICE_ONNX_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/hu/hu_HU/anna/medium/hu_HU-anna-medium.onnx"
)
DEFAULT_VOICE_JSON_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/hu/hu_HU/anna/medium/hu_HU-anna-medium.onnx.json"
)

# Common Hungarian abbreviations to avoid false sentence splitting
HU_ABBREVIATIONS = {
    "kb", "pl", "stb", "dr", "prof", "sz", "vö", "m", "u", "kft", "bt", "nyrt",
    "zrt", "kkt", "kép", "ld", "ill", "ún", "sk", "özv", "ifj", "id", "str",
    "vol", "cap", "art", "jan", "feb", "márc", "ápr", "máj", "jún", "júl",
    "aug", "szept", "okt", "nov", "dec", "fsz", "em", "krt", "út", "u", "tér"
}


def get_hf_token(cli_token: Optional[str] = None) -> Optional[str]:
    """Retrieve Hugging Face token from CLI arg or environment variables."""
    if cli_token:
        return cli_token.strip()
    for env_var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(env_var)
        if token and token.strip():
            return token.strip()
    return None


def download_file(url: str, dest_path: Path, description: str, hf_token: Optional[str] = None):
    """Download a file with progress reporting and optional Hugging Face auth token."""
    print(f"Downloading {description} from {url}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "piper-epub-generator/1.0"}
    token = get_hf_token(hf_token)
    if token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {token}"
        print("  (Using Hugging Face authorization token)")

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            total_size = resp.headers.get("Content-Length")
            total_size = int(total_size) if total_size else 0

            downloaded = 0
            block_size = 64 * 1024
            temp_dest = dest_path.with_suffix(dest_path.suffix + ".tmp")

            with open(temp_dest, "wb") as out_f:
                while True:
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = min(100, int(downloaded * 100 / total_size))
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        sys.stdout.write(f"\r  Progress: {percent}% ({mb_done:.1f} MB / {mb_total:.1f} MB)")
                    else:
                        mb_done = downloaded / (1024 * 1024)
                        sys.stdout.write(f"\r  Downloaded: {mb_done:.1f} MB")
                    sys.stdout.flush()

            temp_dest.replace(dest_path)
            sys.stdout.write("\n")
            print(f"Saved to {dest_path}")

    except urllib.error.HTTPError as e:
        sys.stdout.write("\n")
        if e.code in (401, 403):
            print(f"HTTP Error {e.code}: Access denied while downloading from Hugging Face.")
            print("If this is a gated model, set the HF_TOKEN environment variable:")
            print("  export HF_TOKEN=your_token_here    (Linux/macOS)")
            print("  $env:HF_TOKEN=\"your_token_here\"     (PowerShell)")
            print("  set HF_TOKEN=your_token_here       (Windows CMD)")
        raise


def ensure_voice_files(model_path: Optional[str], hf_token: Optional[str] = None) -> Tuple[Path, Path]:
    """Ensure ONNX model and config JSON are available."""
    if model_path:
        onnx_file = Path(model_path)
        json_file = Path(str(model_path) + ".json")
        if not json_file.exists():
            # Try replacing .onnx with .onnx.json or .json
            alt_json = onnx_file.with_suffix(".onnx.json")
            if alt_json.exists():
                json_file = alt_json
            else:
                alt_json2 = onnx_file.with_suffix(".json")
                if alt_json2.exists():
                    json_file = alt_json2
        if not onnx_file.exists():
            raise FileNotFoundError(f"Model file not found: {onnx_file}")
        if not json_file.exists():
            raise FileNotFoundError(f"Model config JSON not found: {json_file}")
        return onnx_file, json_file

    # Default voice in ./piper_models/
    models_dir = Path("./piper_models")
    models_dir.mkdir(exist_ok=True)
    onnx_file = models_dir / f"{DEFAULT_VOICE_NAME}.onnx"
    json_file = models_dir / f"{DEFAULT_VOICE_NAME}.onnx.json"

    if not onnx_file.exists():
        download_file(DEFAULT_VOICE_ONNX_URL, onnx_file, f"{DEFAULT_VOICE_NAME} model", hf_token=hf_token)
    if not json_file.exists():
        download_file(DEFAULT_VOICE_JSON_URL, json_file, f"{DEFAULT_VOICE_NAME} config", hf_token=hf_token)

    return onnx_file, json_file


class PiperSynthesizer:
    """Wrapper to synthesize speech using Piper Python API or CLI binary."""

    def __init__(self, model_path: Path, config_path: Path, piper_binary: Optional[str] = None, length_scale: float = 1.0):
        self.model_path = model_path
        self.config_path = config_path
        self.length_scale = length_scale
        self.piper_binary = piper_binary or shutil.which("piper") or shutil.which("piper.exe")
        self.use_python_module = False
        self.voice = None

        # Try importing piper python module first
        try:
            from piper.voice import PiperVoice
            self.voice = PiperVoice.load(str(model_path), config_path=str(config_path))
            self.use_python_module = True
            print("Using Piper Python module.")
        except (ImportError, Exception) as e:
            if self.piper_binary and os.path.exists(self.piper_binary):
                print(f"Using Piper binary: {self.piper_binary}")
            else:
                print("Notice: 'piper' Python package not installed and binary not found on PATH.")
                print("Attempting to run 'piper' directly...")
                self.piper_binary = "piper"

    def synthesize_to_wav_bytes(self, text: str) -> bytes:
        """Synthesize text and return raw WAV file bytes."""
        text = text.strip()
        if not text:
            return b""

        if self.use_python_module and self.voice:
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wav_file:
                self.voice.synthesize(text, wav_file, length_scale=self.length_scale)
            return wav_io.getvalue()
        else:
            # Fallback to CLI
            cmd = [
                self.piper_binary or "piper",
                "--model", str(self.model_path),
                "--config", str(self.config_path),
                "--output-raw"
            ]
            if self.length_scale != 1.0:
                cmd.extend(["--length-scale", str(self.length_scale)])

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                raw_audio, stderr = proc.communicate(input=text.encode("utf-8"))
                if proc.returncode != 0:
                    raise RuntimeError(f"Piper error: {stderr.decode('utf-8', errors='ignore')}")

                # Read sample rate from config
                sample_rate = 22050
                try:
                    with open(self.config_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                        sample_rate = cfg.get("audio", {}).get("sample_rate", 22050)
                except Exception:
                    pass

                # Wrap raw PCM (16-bit mono) into WAV container
                wav_io = io.BytesIO()
                with wave.open(wav_io, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(raw_audio)
                return wav_io.getvalue()

            except FileNotFoundError:
                raise RuntimeError(
                    "Piper is not installed. Install via `pip install piper-tts` "
                    "or download the binary from https://github.com/rhasspy/piper/releases"
                )


def split_into_sentences_hungarian(text: str) -> List[str]:
    """
    Split text into sentences while respecting Hungarian abbreviations,
    dialogue dashes, quotation marks, and numbers.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    raw_sentences = []
    pos = 0
    length = len(text)
    current_sent = []

    while pos < length:
        # Match word / token
        m = re.match(r"(\S+)(\s*)", text[pos:])
        if not m:
            break
        token = m.group(1)
        space = m.group(2)
        pos += len(token) + len(space)

        current_sent.append(token)

        # Check if token ends with sentence terminator
        m_term = re.search(r"([.!?…]+)([\"”»'\)\]]*)$", token)
        if m_term:
            term = m_term.group(1)
            # Check if it's an abbreviation
            clean_word = re.sub(r"[^\wáéíóöőúüűÁÉÍÓÖŐÚÜŰ]", "", token.lower())
            is_abbr = clean_word in HU_ABBREVIATIONS

            # Check if it's an ordinal number in Hungarian (e.g. "1.", "2024.") followed by lowercase
            is_number = bool(re.match(r"^\d+\.$", token))
            next_is_upper = False
            if pos < length:
                next_char = text[pos]
                next_is_upper = next_char.isupper() or next_char in "„»\"'—–"

            if (not is_abbr and not (is_number and not next_is_upper)) and (pos >= length or next_is_upper):
                sent_str = " ".join(current_sent).strip()
                if sent_str:
                    raw_sentences.append(sent_str)
                current_sent = []

    if current_sent:
        sent_str = " ".join(current_sent).strip()
        if sent_str:
            raw_sentences.append(sent_str)

    return raw_sentences


def format_smil_clock(seconds: float) -> str:
    """Format seconds into SMIL clock format (HH:MM:SS.mmm)."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"


class ChapterProcessor:
    """Processes an XHTML chapter: splits text, synthesizes audio, builds SMIL and marked XHTML."""

    def __init__(self, chapter_id: str, html_path: str, html_content: str, synthesizer: PiperSynthesizer):
        self.chapter_id = chapter_id
        self.html_path = html_path
        self.html_content = html_content
        self.synthesizer = synthesizer
        self.sentences: List[Dict] = []
        self.total_duration = 0.0

    def process(self, audio_rel_path: str, chapter_num: int) -> Tuple[str, str, bytes, float]:
        """
        Process the chapter HTML.
        Returns:
            (marked_xhtml_content, smil_xml_content, combined_wav_bytes, duration_seconds)
        """
        sentence_counter = 0
        timing_entries = []

        def replace_block_content(match):
            nonlocal sentence_counter
            tag = match.group(1)
            attrs = match.group(2)
            inner_html = match.group(3)

            # Strip inner tags to get plain text for TTS
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

            # Split into sentences
            sents = split_into_sentences_hungarian(plain_text)
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
                self.sentences.append({
                    "id": span_id,
                    "text": sent_text,
                })

            return f"<{tag}{attrs}>{' '.join(span_parts)}</{tag}>"

        pattern = r"<(p|h[1-6]|li|blockquote|div)([^>]*)>(.*?)</\1>"
        marked_html = re.sub(pattern, replace_block_content, self.html_content, flags=re.DOTALL | re.IGNORECASE)

        if not self.sentences:
            return self.html_content, "", b"", 0.0

        current_time = 0.0
        sample_rate = 22050
        sample_width = 2
        channels = 1
        raw_frames_list = []

        print(f"  Synthesizing {len(self.sentences)} sentences for chapter {chapter_num} ({self.html_path})...")
        for idx, sent in enumerate(self.sentences, 1):
            text = sent["text"]
            wav_bytes = self.synthesizer.synthesize_to_wav_bytes(text)
            if not wav_bytes:
                continue

            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_frames = wf.readframes(n_frames)
                dur = n_frames / float(sample_rate)

            raw_frames_list.append(raw_frames)
            
            # Short silence pause between sentences (~250ms)
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

            if idx % 10 == 0 or idx == len(self.sentences):
                sys.stdout.write(f"\r    [{idx}/{len(self.sentences)}] sentences synthesized...")
                sys.stdout.flush()
        sys.stdout.write("\n")

        self.total_duration = current_time

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
            stderr=subprocess.PIPE
        )
        mp3_bytes, stderr = proc.communicate(input=wav_bytes)
        if proc.returncode == 0 and mp3_bytes:
            return mp3_bytes, "audio/mpeg"
    except Exception:
        pass

    return wav_bytes, "audio/wav"


def process_epub(
    input_epub: Path,
    output_epub: Path,
    synthesizer: PiperSynthesizer,
    start_chapter: int = 1,
    end_chapter: Optional[int] = None
):
    """Main pipeline to convert EPUB to EPUB 3 with Media Overlays."""
    print(f"Reading input EPUB: {input_epub}")
    temp_dir = Path(tempfile.mkdtemp(prefix="epub_audio_"))

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
            processor = ChapterProcessor(item_id, ch_href, html_text, synthesizer)

            audio_base_name = f"ch_{ch_idx:03d}"
            marked_html, smil_xml, wav_bytes, ch_dur = processor.process(
                audio_rel_path=f"../Audio/{audio_base_name}.mp3",
                chapter_num=ch_idx
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
            new_manifest_items.append((item_id, smil_item_id, f"MediaOverlays/{smil_filename}", audio_item_id, f"Audio/{audio_filename}", mime_type, ch_dur))

            total_book_duration += ch_dur
            processed_count += 1

        print(f"\nProcessed {processed_count} chapters. Total narration duration: {format_smil_clock(total_book_duration)}")

        print("Updating OPF manifest and metadata with Media Overlays...")
        for orig_id, smil_id, smil_href, audio_id, audio_href, audio_mime, ch_dur in new_manifest_items:
            pattern = rf'(<item\s+[^>]*\bid=["\']{orig_id}["\'][^>]*)(/?>)'
            
            def add_mo(match):
                tag_start = match.group(1)
                tag_end = match.group(2)
                if 'media-overlay=' not in tag_start:
                    return f'{tag_start} media-overlay="{smil_id}"{tag_end}'
                return match.group(0)

            opf_content = re.sub(pattern, add_mo, opf_content)

            smil_entry = f'    <item id="{smil_id}" href="{smil_href}" media-type="application/smil+xml"/>\n'
            audio_entry = f'    <item id="{audio_id}" href="{audio_href}" media-type="{audio_mime}"/>\n'
            manifest_insert = smil_entry + audio_entry

            opf_content = re.sub(r'(</manifest>)', rf'{manifest_insert}\1', opf_content, count=1)

        meta_duration = (
            f'    <meta property="media:duration">{format_smil_clock(total_book_duration)}</meta>\n'
            f'    <meta property="media:active-class">-epub-media-overlay-active</meta>\n'
        )
        opf_content = re.sub(r'(</metadata>)', rf'{meta_duration}\1', opf_content, count=1)

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
        print(f"\nSUCCESS! Created aligned audiobook EPUB: {output_epub} ({size_mb:.1f} MB)")
        print("Ready to copy to your Kobo e-reader for use with audiobook.koplugin!")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Convert standard EPUB into an aligned EPUB 3 audiobook using Piper neural TTS."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input .epub file")
    parser.add_argument("-o", "--output", help="Path to output .epub file (default: <name>_audiobook.epub)")
    parser.add_argument("-m", "--model", help=f"Path to Piper .onnx model file (default: downloads {DEFAULT_VOICE_NAME})")
    parser.add_argument("--piper-path", help="Path to piper executable")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (e.g. 1.0 = normal, 1.2 = faster)")
    parser.add_argument("--hf-token", help="Hugging Face API token (or set HF_TOKEN env variable)")
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

    length_scale = 1.0 / args.speed if args.speed > 0 else 1.0

    onnx_file, json_file = ensure_voice_files(args.model, hf_token=args.hf_token)
    synthesizer = PiperSynthesizer(
        onnx_file,
        json_file,
        piper_binary=args.piper_path,
        length_scale=length_scale
    )

    process_epub(
        input_path,
        output_path,
        synthesizer,
        start_chapter=args.start_chapter,
        end_chapter=args.end_chapter
    )


if __name__ == "__main__":
    main()

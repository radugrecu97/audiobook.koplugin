#!/usr/bin/env python3
"""
Bitrate A/B Comparison Tool for Audiobooks.

Encodes sample audio at multiple MP3 and Opus bitrates, displays size/bandwidth projections,
and optionally generates a multi-chapter EPUB test harness to audition different bitrates
directly on the Kobo device.
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import posixpath
import shutil
import subprocess
import sys
import tempfile
import wave
import zipfile

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from tools.audiobookgen.audio.encoder import Mp3Encoder, OpusEncoder, WavEncoder
from tools.audiobookgen.epub.package import EpubPackage
from tools.audiobookgen.epub.smil import format_smil_clock
from tools.audiobookgen.models import AudioClip, SentenceTiming


def get_audio_duration_seconds(wav_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        return 0.0


def extract_audio_from_epub(epub_path: Path, chapter_idx: int) -> tuple[bytes, str]:
    with EpubPackage(epub_path) as pkg:
        # Look for audio item in manifest
        audio_items = [
            item for item in pkg.manifest.values()
            if "audio" in item.media_type or item.href.lower().endswith((".mp3", ".wav", ".ogg", ".opus"))
        ]
        if not audio_items:
            raise RuntimeError("No audio tracks found inside EPUB.")

        target_item = None
        if chapter_idx <= len(audio_items):
            target_item = audio_items[chapter_idx - 1]
        else:
            target_item = audio_items[0]

        internal_path = posixpath.join(pkg.opf_dir, target_item.href) if pkg.opf_dir else target_item.href
        raw_audio = pkg._read_zip_bytes(internal_path)
        if not raw_audio:
            raise RuntimeError(f"Could not extract audio at {internal_path}")

        # Convert to WAV if needed
        if raw_audio.startswith(b"RIFF"):
            return raw_audio, target_item.href

        ffmpeg_bin = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
        proc = subprocess.run(
            [ffmpeg_bin, "-y", "-i", "pipe:0", "-f", "wav", "pipe:1"],
            input=raw_audio,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return proc.stdout, target_item.href


def generate_bitrate_test_epub(
    output_path: Path,
    sample_wav: bytes,
    sample_text: str,
    encodings: list[tuple[str, str, bytes, str]],  # (name, ext, audio_bytes, mime)
) -> Path:
    """
    Creates a small test EPUB where each bitrate is presented as a separate chapter.
    """
    duration = get_audio_duration_seconds(sample_wav)
    dur_clock = format_smil_clock(duration)

    tmp_out = output_path.with_name(f".tmp_{output_path.name}")
    with zipfile.ZipFile(tmp_out, "w") as zf:
        zf.writestr("mimetype", b"application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?>\n<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
            '  <rootfiles>\n    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>\n  </rootfiles>\n</container>',
            compress_type=zipfile.ZIP_DEFLATED,
        )

        manifest_items = []
        spine_items = []
        metadata_items = [
            f'    <meta property="media:duration">{format_smil_clock(duration * len(encodings))}</meta>',
            '    <meta property="media:active-class">-epub-media-overlay-active</meta>',
        ]

        for idx, (label, ext, a_bytes, a_mime) in enumerate(encodings, start=1):
            ch_id = f"ch_{idx:02d}"
            xhtml_name = f"chapter_{idx:02d}.xhtml"
            smil_name = f"chapter_{idx:02d}.smil"
            audio_name = f"audio_{idx:02d}{ext}"

            # 1. XHTML
            escaped_text = sample_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            xhtml_content = (
                f'<?xml version="1.0" encoding="utf-8"?>\n'
                f'<html xmlns="http://www.w3.org/1999/xhtml">\n<head>\n'
                f'  <title>{label}</title>\n'
                f'</head>\n<body>\n'
                f'  <h1>{label}</h1>\n'
                f'  <p><span id="s1">{escaped_text}</span></p>\n'
                f'</body>\n</html>'
            )
            zf.writestr(f"OEBPS/text/{xhtml_name}", xhtml_content, compress_type=zipfile.ZIP_DEFLATED)

            # 2. SMIL
            smil_content = (
                f'<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<smil xmlns="http://www.w3.org/ns/SMIL" xmlns:epub="http://www.idpf.org/2007/ops" version="3.0">\n'
                f'  <body>\n'
                f'    <seq epub:textref="../text/{xhtml_name}">\n'
                f'      <par id="par_s1">\n'
                f'        <text src="../text/{xhtml_name}#s1"/>\n'
                f'        <audio src="../Audio/{audio_name}" clipBegin="00:00:00.000" clipEnd="{dur_clock}"/>\n'
                f'      </par>\n'
                f'    </seq>\n'
                f'  </body>\n</smil>'
            )
            zf.writestr(f"OEBPS/MediaOverlays/{smil_name}", smil_content, compress_type=zipfile.ZIP_DEFLATED)

            # 3. Audio (uncompressed stored)
            zf.writestr(f"OEBPS/Audio/{audio_name}", a_bytes, compress_type=zipfile.ZIP_STORED)

            # OPF items
            manifest_items.append(
                f'    <item id="{ch_id}" href="text/{xhtml_name}" media-type="application/xhtml+xml" media-overlay="smil_{ch_id}"/>\n'
                f'    <item id="smil_{ch_id}" href="MediaOverlays/{smil_name}" media-type="application/smil+xml"/>\n'
                f'    <item id="audio_{ch_id}" href="Audio/{audio_name}" media-type="{a_mime}"/>'
            )
            spine_items.append(f'    <itemref idref="{ch_id}"/>')
            metadata_items.append(f'    <meta property="media:duration" refines="#smil_{ch_id}">{dur_clock}</meta>')

        opf_content = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="pub-id">\n'
            '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
            '    <dc:title>Bitrate A/B Audio Comparison</dc:title>\n'
            '    <dc:identifier id="pub-id">bitrate-test-epub</dc:identifier>\n'
            '    <dc:language>en</dc:language>\n'
            + "\n".join(metadata_items) + "\n"
            '  </metadata>\n'
            '  <manifest>\n'
            + "\n".join(manifest_items) + "\n"
            '  </manifest>\n'
            '  <spine>\n'
            + "\n".join(spine_items) + "\n"
            '  </spine>\n'
            '</package>'
        )
        zf.writestr("OEBPS/content.opf", opf_content, compress_type=zipfile.ZIP_DEFLATED)

    os.replace(tmp_out, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="A/B audio bitrate comparison and test EPUB generator.")
    parser.add_argument("--wav", "-w", default=None, help="Input WAV file path.")
    parser.add_argument("--from-epub", default=None, help="Extract sample audio from existing EPUB file.")
    parser.add_argument("--chapter", type=int, default=1, help="Chapter index to extract from EPUB (default: 1).")
    parser.add_argument("--out-dir", "-o", default="bitrate_comparison", help="Output directory for test audio files.")
    parser.add_argument("--bitrates", default="32k,48k,64k,96k,128k", help="Comma-separated MP3 bitrates (default: 32k,48k,64k,96k,128k).")
    parser.add_argument("--opus-bitrates", default="16k,24k,32k,48k", help="Comma-separated Opus bitrates (default: 16k,24k,32k,48k).")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Audio sample rate (default: 22050 Hz).")
    parser.add_argument("--book-hours", type=float, default=10.0, help="Projected full-book duration in hours for size calculation (default: 10.0 hrs).")
    parser.add_argument("--emit-test-epub", default=None, help="Optionally generate bitrate_test.epub to audition on Kobo.")

    args = parser.parse_args()

    wav_bytes = None
    if args.wav:
        wav_path = Path(args.wav).resolve()
        if not wav_path.exists():
            print(f"Error: WAV file not found: {wav_path}", file=sys.stderr)
            sys.exit(1)
        wav_bytes = wav_path.read_bytes()
    elif args.from_epub:
        epub_p = Path(args.from_epub).resolve()
        if not epub_p.exists():
            print(f"Error: EPUB file not found: {epub_p}", file=sys.stderr)
            sys.exit(1)
        wav_bytes, extracted_name = extract_audio_from_epub(epub_p, args.chapter)
        print(f"Extracted sample audio '{extracted_name}' from EPUB ({len(wav_bytes)/1024:.1f} KB WAV)")
    else:
        print("Error: Specify either --wav <file.wav> or --from-epub <book.epub>", file=sys.stderr)
        sys.exit(1)

    duration = get_audio_duration_seconds(wav_bytes)
    if duration <= 0:
        print("Error: Input audio has 0s duration.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mp3_rates = [b.strip() for b in args.bitrates.split(",") if b.strip()]
    opus_rates = [b.strip() for b in args.opus_bitrates.split(",") if b.strip()]

    encodings_for_epub: list[tuple[str, str, bytes, str]] = []

    print("=" * 80)
    print(f"AUDIO BITRATE COMPARISON (Sample Duration: {duration:.2f}s | Projected Book: {args.book_hours:.1f} hrs)")
    print("=" * 80)
    header = f"{'Format':<8} | {'Bitrate':<8} | {'File Size':<10} | {'MB / Hour':<10} | {'Projected Book':<15}"
    print(header)
    print("-" * 80)

    # 1. MP3 Encodings
    for br in mp3_rates:
        encoder = Mp3Encoder(bitrate=br, channels=1, sample_rate=args.sample_rate)
        data, mime = encoder.encode(wav_bytes)
        out_f = out_dir / f"sample_{br}.mp3"
        out_f.write_bytes(data)

        size_kb = len(data) / 1024.0
        mb_per_hour = (len(data) / duration * 3600.0) / (1024.0 * 1024.0)
        proj_book_mb = mb_per_hour * args.book_hours

        print(f"{'MP3':<8} | {br:<8} | {size_kb:>7.1f} KB | {mb_per_hour:>7.2f} MB/h | {proj_book_mb:>10.1f} MB")
        encodings_for_epub.append((f"MP3 ({br})", ".mp3", data, mime))

    # 2. Opus Encodings
    for br in opus_rates:
        encoder = OpusEncoder(bitrate=br, channels=1, sample_rate=24000)
        data, mime = encoder.encode(wav_bytes)
        out_f = out_dir / f"sample_{br}.ogg"
        out_f.write_bytes(data)

        size_kb = len(data) / 1024.0
        mb_per_hour = (len(data) / duration * 3600.0) / (1024.0 * 1024.0)
        proj_book_mb = mb_per_hour * args.book_hours

        print(f"{'Opus':<8} | {br:<8} | {size_kb:>7.1f} KB | {mb_per_hour:>7.2f} MB/h | {proj_book_mb:>10.1f} MB")
        encodings_for_epub.append((f"Opus ({br})", ".ogg", data, mime))

    print("=" * 80)
    print(f"Sample audio files written to: {out_dir}")

    if args.emit_test_epub:
        test_epub_p = Path(args.emit_test_epub).resolve()
        sample_text = (
            "This is an audio test comparing multiple bitrates on your Kobo e-reader. "
            "Listen carefully to speech clarity, background noise, and compression artifacts."
        )
        generate_bitrate_test_epub(test_epub_p, wav_bytes, sample_text, encodings_for_epub)
        print(f"Test EPUB created: {test_epub_p}")
        print("Copy to your Kobo to compare bitrates directly with audiobook.koplugin!")


if __name__ == "__main__":
    main()

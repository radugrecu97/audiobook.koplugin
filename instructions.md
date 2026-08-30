# Aligned Audiobook Generation & KOReader Read-Along Guide

This guide covers everything needed to generate pre-computed neural TTS audiobooks with **synchronized sentence highlighting** and **tap-to-seek** for [KOReader](https://github.com/koreader/koreader) using the `audiobook.koplugin`.

---

## Table of Contents
1. [Overview & How It Works](#1-overview--how-it-works)
2. [Why Sentence Highlighting is Better for E-Ink](#2-why-sentence-highlighting-is-better-for-e-ink)
3. [Method 1: Piper Neural TTS (Fast, Lightweight, Offline)](#3-method-1-piper-neural-tts-fast-lightweight-offline)
4. [Method 2: Fish Speech (S2 Pro / v1.5 / Voice Cloning)](#4-method-2-fish-speech-s2-pro--v15--voice-cloning)
   - [Local Setup](#option-a-local-fish-speech-server)
   - [Cloud API Setup](#option-b-fish-audio-cloud-api)
   - [Zero-Shot Voice Cloning](#zero-shot-voice-cloning)
5. [Resumable Pipeline & Quality Control (QC)](#5-resumable-pipeline--quality-control-qc)
   - [Resuming Interrupted Runs](#resuming-interrupted-runs)
   - [Quality Control Verification](#quality-control-verification)
   - [What to do when QC flags a sentence](#what-to-do-when-qc-flags-a-sentence)
6. [Audio Bitrates & A/B Comparison](#6-audio-bitrates--ab-comparison)
7. [Installing & Using the Plugin on your Kobo](#7-installing--using-the-plugin-on-your-kobo)

---

## 1. Overview & How It Works

1. **PC Generation**: You convert any standard `.epub` or `.kepub` ebook on your PC using the generator pipeline (`tools/epub_to_audiobook_fishspeech.py` or `tools/epub_to_audiobook_piper.py`).
2. **Timestamps & SMIL Overlays**: The script synthesizes the audio and generates W3C EPUB 3 Media Overlays (`.smil` files) with `<span id="chN_sM">` markup that maps every sentence to its exact millisecond timestamp in the audio.
3. **Kobo Playback**: You transfer the output `.epub` to your Kobo. The `audiobook.koplugin` plays the audio through Bluetooth or speakers, highlights sentences smoothly, turns pages automatically, and jumps to the exact sentence whenever you tap or long-press on the screen.

---

## 2. Why Sentence Highlighting is Better for E-Ink

On E-Ink screens (Carta, Kaleido, etc.), screen refreshes take **150ms–300ms** per cycle.
* **Word-by-word visual highlighting**: Flashes the screen 3–5 times *every single second*, causing heavy visual distraction, ghosting artifacts, and high battery drain.
* **Sentence-level visual highlighting**: Updates only once every 3–8 seconds. It is completely smooth, clear, and battery-efficient with region-limited e-ink refresh.
* **Sentence-level seek precision**: Tapping or long-pressing *any* sentence immediately seeks playback to the exact start of that sentence.

---

## 3. Method 1: Piper Neural TTS (Fast, Lightweight, Offline)

Piper is very fast, requires no heavy PyTorch install, and works on standard Python environments.

### Requirements:
```bash
pip install piper-tts
```
*(Optional: install `ffmpeg` to compress audio to `.mp3`. If `ffmpeg` is missing, uncompressed `.wav` is used automatically).*

### Hungarian Anna (`hu_HU-anna-medium`):
```bash
python tools/epub_to_audiobook_piper.py -i konyv.epub -o konyv_audiobook.epub
```

---

## 4. Method 2: Fish Speech (S2 Pro / v1.5 / Voice Cloning)

Fish Speech provides zero-shot voice cloning and state-of-the-art multi-lingual synthesis.

### Option A: Local Fish Speech Server (Requires GPU + Conda)

1. **Create Python 3.11 Environment & Install**:
   ```bash
   conda create -n fish-speech python=3.11 -y
   conda activate fish-speech
   git clone https://github.com/fishaudio/fish-speech.git
   cd fish-speech
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
   pip install -e .
   ```

2. **Download Model Checkpoints (S2 Pro)**:
   ```bash
   pip install huggingface_hub
   huggingface-cli download drbaph/s2-pro-fp8 --local-dir checkpoints/s2-pro-fp8
   ```

3. **Start the API Server**:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   python tools/api_server.py \
     --llama-checkpoint-path checkpoints/s2-pro-fp8 \
     --decoder-checkpoint-path checkpoints/s2-pro-fp8/codec.pth \
     --decoder-config-name modded_dac_vq \
     --half \
     --compile \
     --listen 127.0.0.1:8080
   ```

4. **Generate the Aligned Audiobook**:
   ```bash
   python tools/epub_to_audiobook_fishspeech.py \
     -i book.epub \
     -o book_audiobook.epub \
     --prompt-audio voice_sample.wav \
     --prompt-text "Sample voice transcript" \
     --lang en \
     --audio-bitrate 64k \
     --resume
   ```

### Option B: Fish Audio Cloud API

```bash
python tools/epub_to_audiobook_fishspeech.py \
  -i book.epub \
  --api-url https://api.fish.audio/v1/tts \
  --api-key "your_fish_audio_key" \
  --lang en
```

---

## 5. Resumable Pipeline & Quality Control (QC)

The new pipeline in `tools/audiobookgen/` provides automatic crash-resilience, atomic caching, and automated Whisper-based Quality Control.

### Work Directory & Cache Structure
By default, intermediate files and sentence cache are stored in `.audiobookgen/<book-stem>/`:
```
.audiobookgen/mybook/
├── cache/            # Content-addressed sentence audio (.wav + .json metadata)
│   └── 3f/
│       ├── 3f1a2b...wav
│       └── 3f1a2b...json
├── state/            # Completed chapter states and encoded chapter audio
│   ├── ch_0001.json
│   ├── ch_0001.mp3
│   └── ...
├── progress.log      # Unattended run log with rolling ETA
└── qc_report.json    # QC flags, character error rates, and retry history
```

### Resuming Interrupted Runs
* `--resume` is **enabled by default**. If generation is interrupted at chapter 100 of 129, re-running the exact same command skips completed chapters instantly and resumes from the exact sentence where it stopped.
* `--fresh`: Wipes the cache directory and restarts from scratch.
* `--dry-run`: Quickly analyzes the EPUB, counts sentences and characters, and prints projected audio duration and GPU generation time without calling TTS.

### Quality Control Verification
Fish Speech can occasionally truncate, repeat words, or hallucinate. The built-in QC verifier automatically audits synthesized audio using `faster-whisper`:
* `--qc off`: Disable QC.
* `--qc fast`: Uses Whisper `small` (recommended for general verification).
* `--qc strict`: Uses Whisper `large-v3`.
* `--qc-retries 2`: Automatically retries flagged sentences with jittered seeds and selects the best attempt.

### What to do when QC flags a sentence
1. Review `.audiobookgen/<book-stem>/qc_report.json`.
2. Every flagged sentence includes the chapter number, span ID, source text, Whisper transcript, CER score, and final disposition.
3. To re-run a specific chapter, delete its entry `state/ch_<NNN>.json` and `state/ch_<NNN>.mp3` from the work directory, then run the generator with `--start-chapter N --end-chapter N`.

---

## 6. Audio Bitrates & A/B Comparison

Audio bitrate directly determines final book file size:
* **64 kbps MP3** (Default): ~27.6 MB / hour (~143 MB for a 5.2-hour book). Excellent speech clarity.
* **48 kbps MP3**: ~20.7 MB / hour (~108 MB for a 5.2-hour book). Great balance for limited storage.
* **32 kbps MP3**: ~13.8 MB / hour (~72 MB for a 5.2-hour book). Highly compressed.
* **32 kbps Opus**: ~13.9 MB / hour (~72 MB for a 5.2-hour book). High fidelity at low bitrate.

### Comparing Bitrates on your Kobo
Use the bitrate comparison tool to encode a sample and generate a test EPUB:
```bash
python tools/compare_bitrates.py --from-epub book.epub --emit-test-epub /tmp/bitrate_test.epub
```
Copy `bitrate_test.epub` to your Kobo to listen to the different bitrates through your headphones/Bluetooth.

---

## 7. Installing & Using the Plugin on your Kobo

1. **Install Plugin**:
   Copy the `audiobook.koplugin` folder from this repository into your Kobo at:
   ```
   .adds/koreader/plugins/audiobook.koplugin/
   ```
   *(Restart KOReader after copying).*

2. **Copy the Aligned EPUB**:
   Copy your generated `*_audiobook.epub` file into your books folder on the Kobo (e.g. `/mnt/onboard/books/`).

3. **Read with Audio & Synchronized Highlighting**:
   * Open the book in KOReader.
   * **Start from current page**: Go to top menu $\rightarrow$ **Tools** $\rightarrow$ **Audiobook Read-Along** $\rightarrow$ **Start reading from current page** (or tap the read-aloud icon).
   * **Jump from any sentence**: Long-press any word on the screen $\rightarrow$ tap **Play aligned audiobook from here** (or **Read aloud from here**).
   * Playback begins immediately from that sentence, highlighting sentences in sync and turning pages automatically.

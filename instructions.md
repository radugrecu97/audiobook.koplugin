# Aligned Audiobook Generation & KOReader Read-Along Guide

This guide covers everything needed to generate pre-computed neural TTS audiobooks with **synchronized text highlighting** and **word-level tap-to-seek** for [KOReader](https://github.com/koreader/koreader) using the `audiobook.koplugin`.

---

## Table of Contents
1. [Overview & How It Works](#1-overview--how-it-works)
2. [Why Sentence Highlighting is Better for E-Ink](#2-why-sentence-highlighting-is-better-for-e-ink)
3. [Method 1: Piper Neural TTS (Fast, Lightweight, Offline)](#3-method-1-piper-neural-tts-fast-lightweight-offline)
   - [Hungarian Anna (`hu_HU-anna-medium`)](#hungarian-anna-hu_hu-anna-medium)
   - [Testing Other Languages (English, German, French, etc.)](#testing-other-languages)
   - [Hugging Face Token Authentication](#hugging-face-token-authentication)
4. [Method 2: Fish Speech (S2 Pro / v1.5 / Voice Cloning)](#4-method-2-fish-speech-s2-pro--v15--voice-cloning)
   - [Local Setup (Python 3.10–3.12 + PyTorch)](#option-a-local-fish-speech-server)
   - [Cloud API Setup (No Local GPU Needed)](#option-b-fish-audio-cloud-api)
   - [Zero-Shot Voice Cloning](#zero-shot-voice-cloning)
5. [Installing & Using the Plugin on your Kobo](#5-installing--using-the-plugin-on-your-kobo)

---

## 1. Overview & How It Works

1. **PC Generation**: You convert any standard `.epub` ebook on your PC using one of the included generator scripts (`tools/epub_to_audiobook_piper.py` or `tools/epub_to_audiobook_fishspeech.py`).
2. **Timestamps & SMIL Overlays**: The script synthesizes the audio and generates W3C EPUB 3 Media Overlays (`.smil` files) with `<span id="sN">` markup that maps every sentence and word to its exact millisecond timestamp in the audio.
3. **Kobo Playback**: You transfer the output `.epub` to your Kobo. The `audiobook.koplugin` plays the audio through Bluetooth or speakers, highlights text smoothly, turns pages automatically, and jumps to the exact word whenever you tap on the screen.

---

## 2. Why Sentence Highlighting is Better for E-Ink

On E-Ink screens (Carta, Kaleido, etc.), screen refreshes take **150ms–300ms** per cycle.
* **Word-by-word visual highlighting**: Flashes the screen 3–5 times *every single second*, causing heavy visual distraction, ghosting artifacts, and high battery drain.
* **Sentence-level visual highlighting**: Updates only once every 3–8 seconds. It is completely smooth, clear, and battery-efficient.
* **Word-level seek precision**: You get the best of both worlds—sentence highlights while reading, but tapping *any* individual word immediately seeks the audiobook directly to that word.

---

## 3. Method 1: Piper Neural TTS (Fast, Lightweight, Offline)

Piper is very fast, requires no heavy PyTorch install, and works on standard Python environments.

### Requirements:
```bash
pip install piper-tts
```
*(Optional: install `ffmpeg` to compress audio to `.mp3`. If `ffmpeg` is missing, uncompressed `.wav` is used automatically).*

### Hungarian Anna (`hu_HU-anna-medium`):
Converts any Hungarian EPUB (automatically downloads `hu_HU-anna-medium.onnx` and config on first run):
```bash
python tools/epub_to_audiobook_piper.py -i konyv.epub -o konyv_audiobook.epub
```

### Testing Other Languages:
You can pass any Piper `.onnx` model or Hugging Face URL with `-m`:

* **English (Lessac)**:
  ```bash
  python tools/epub_to_audiobook_piper.py -i book.epub -m https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
  ```
* **German (Thorsten)**:
  ```bash
  python tools/epub_to_audiobook_piper.py -i book.epub -m https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx
  ```
* **French (Siwis)**:
  ```bash
  python tools/epub_to_audiobook_piper.py -i book.epub -m https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx
  ```
* **Spanish (CarlFM)**:
  ```bash
  python tools/epub_to_audiobook_piper.py -i book.epub -m https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/carlfm/medium/es_ES-carlfm-medium.onnx
  ```
* Browse all voice samples: [rhasspy.github.io/piper-samples](https://rhasspy.github.io/piper-samples/)

### Hugging Face Token Authentication:
If downloading gated or rate-limited models from Hugging Face:
```bash
# Via Environment Variable
export HF_TOKEN="hf_your_token_here"       # Linux / macOS
$env:HF_TOKEN="hf_your_token_here"        # PowerShell
set HF_TOKEN=hf_your_token_here           # Windows CMD

# Or via CLI argument
python tools/epub_to_audiobook_piper.py -i book.epub --hf-token "hf_your_token_here"
```

### Additional Options:
* `--speed 1.15`: Increase speech speed (e.g. 1.15x).
* `--start-chapter 1 --end-chapter 2`: Test on a small subset of chapters first.

---

## 4. Method 2: Fish Speech (S2 Pro / v1.5 / Voice Cloning)

Fish Speech provides zero-shot voice cloning and state-of-the-art multi-lingual synthesis.

> **Note on Python Version**: Fish Speech uses PyTorch and CUDA, which require **Python 3.10, 3.11, or 3.12** (Python 3.14 is currently not supported by PyTorch wheels).

### Option A: Local Fish Speech Server (Requires GPU + Conda)

1. **Create Python 3.11 Environment**:
   ```bash
   conda create -n fish-speech python=3.11 -y
   conda activate fish-speech
   ```

2. **Install Fish Speech**:
   ```bash
   git clone https://github.com/fishaudio/fish-speech.git
   cd fish-speech
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install -e .
   ```

3. **Download Model Checkpoints**:
   ```bash
   pip install huggingface_hub
   huggingface-cli download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5
   ```

4. **Start the API Server**:
   ```bash
   python -m fish_speech.api.server --listen 127.0.0.1:8080 --checkpoint-path checkpoints/fish-speech-1.5
   ```

5. **Generate the EPUB**:
   ```bash
   python tools/epub_to_audiobook_fishspeech.py -i book.epub -o book_audiobook.epub --lang en
   ```

### Option B: Fish Audio Cloud API (No Local GPU Required)

1. Get an API key from [fish.audio](https://fish.audio/).
2. Run with the Cloud API endpoint:
   ```bash
   python tools/epub_to_audiobook_fishspeech.py \
     -i book.epub \
     --api-url https://api.fish.audio/v1/tts \
     --api-key "your_fish_audio_key" \
     --lang en
   ```

### Zero-Shot Voice Cloning:
To clone any voice, supply a short 5–15 second clean `.wav` sample and its transcript:
```bash
python tools/epub_to_audiobook_fishspeech.py \
  -i book.epub \
  -o book_cloned_audiobook.epub \
  --prompt-audio narrator_sample.wav \
  --prompt-text "Sample transcript of the narrator's voice." \
  --lang en
```

---

## 5. Installing & Using the Plugin on your Kobo

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
   * **Jump from any word/sentence**: Long-press any word on the screen $\rightarrow$ tap **Play aligned audiobook from here** (or **Read aloud from here**).
   * Playback begins immediately from that word, highlighting sentences in sync and turning pages automatically.

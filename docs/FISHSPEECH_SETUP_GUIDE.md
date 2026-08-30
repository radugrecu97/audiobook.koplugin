# Fish Speech (S2 Pro) Aligned Audiobook Setup Guide

This guide explains how to set up **Fish Speech (S2 Pro / v1.4 / v1.5)** on your PC to generate high-fidelity, multilingual, voice-cloned EPUB 3 audiobooks with synchronized highlighting and tap-to-seek on your **Kobo e-reader**.

---

## 1. Running Fish Speech Locally on your PC

### Option A: Using Fish Speech Git & Python
1. Clone the Fish Speech repository:
   ```bash
   git clone https://github.com/fishaudio/fish-speech.git
   cd fish-speech
   ```
2. Install dependencies (PyTorch with CUDA recommended):
   ```bash
   pip install -e .
   ```
3. Start the API server on port 8080:
   ```bash
   python -m fish_speech.api.server --listen 127.0.0.1:8080
   ```

### Option B: Using Fish Speech WebUI / Docker
1. Start the Fish Speech Docker container or WebUI:
   ```bash
   docker run --gpus all -p 8080:8080 fishaudio/fish-speech:latest
   ```
2. The HTTP API endpoint will be live at `http://127.0.0.1:8080/v1/tts`.

---

## 2. Converting EPUB with Fish Speech

Once the Fish Speech server is running, generate the aligned EPUB:

### 1. English (or other languages) with Default Voice
```bash
python tools/epub_to_audiobook_fishspeech.py -i book.epub -o book_audiobook.epub --lang en
```

### 2. Zero-Shot Voice Cloning (Custom Voice Reference)
Provide a 5–15 second clean `.wav` sample and its transcript:
```bash
python tools/epub_to_audiobook_fishspeech.py \
  -i book.epub \
  -o book_cloned_audiobook.epub \
  --prompt-audio my_narrator_sample.wav \
  --prompt-text "This is a clean recording of the narrator's voice." \
  --lang en
```

### 3. Voice Pacing & Pause Tuning
Fine-tune speech rate and pauses between sentences:
```bash
python tools/epub_to_audiobook_fishspeech.py \
  -i book.epub \
  --prompt-audio sample.wav \
  --prompt-text "Transcript" \
  --speed 0.90 \
  --pause-duration 0.35 \
  --pause-variance 0.075 \
  --audio-bitrate 64k
```

### 4. Resuming Interrupted Generations
Generation is crash-resilient by default. Intermediate chapter state and sentence audio are cached:
```bash
# Automatically resumes from last finished chapter
python tools/epub_to_audiobook_fishspeech.py -i book.epub --resume
```

### 5. Quality Control (Whisper Verification)
Enable automated verification to detect truncations or hallucinations:
```bash
python tools/epub_to_audiobook_fishspeech.py -i book.epub --qc fast --qc-retries 2
```

### 6. Dry Run Estimation
Pre-flight check without calling the TTS server:
```bash
python tools/epub_to_audiobook_fishspeech.py -i book.epub --dry-run
```

---

## 3. Reading on Kobo with KOReader

1. Copy `book_audiobook.epub` to your Kobo storage (e.g. `/mnt/onboard/books/`).
2. Open the book in KOReader.
3. Long-press any word on the screen $\rightarrow$ tap **Play aligned audiobook from here**.
4. The Fish Speech neural narration begins immediately from that sentence, with synchronized screen highlights and automatic page turns.

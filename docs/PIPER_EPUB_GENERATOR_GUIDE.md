# Hungarian Piper Aligned Audiobook Generation Guide

This guide explains how to convert Hungarian EPUB ebooks into synchronized read-along EPUB 3 audiobooks on your PC using **Piper TTS (`hu_HU-anna-medium`)**, and how to read them on your **Kobo e-reader** with `audiobook.koplugin` in KOReader.

---

## 1. Requirements on your PC

1. **Python 3.8+**
2. **Piper TTS**:
   - Option A (Recommended): `pip install piper-tts`
   - Option B: Download the standalone `piper` binary from [Piper GitHub Releases](https://github.com/rhasspy/piper/releases) and place it in your PATH.
3. *(Optional but recommended)* **ffmpeg**:
   - If installed, the script compresses chapter audio to `.mp3` (smaller file size).
   - If ffmpeg is not installed, the script falls back to uncompressed `.wav` audio (which KOReader also supports natively).

---

## 2. Converting an EPUB on your PC

Run the script from the repository root:

```bash
# Basic conversion (automatically downloads hu_HU-anna-medium voice on first run)
python tools/epub_to_audiobook_piper.py -i /path/to/konyv.epub -o /path/to/konyv_audiobook.epub
```

### Advanced Options:

```bash
# Set Hugging Face token via environment variable (if downloading gated/rate-limited models)
export HF_TOKEN="your_hf_token_here"       # Linux / macOS
$env:HF_TOKEN="your_hf_token_here"        # PowerShell

# Or pass the token directly as an argument
python tools/epub_to_audiobook_piper.py -i konyv.epub --hf-token "your_hf_token_here"

# Adjust reading speed (e.g. 1.15x faster)
python tools/epub_to_audiobook_piper.py -i konyv.epub --speed 1.15

# Test on just the first 2 chapters
python tools/epub_to_audiobook_piper.py -i konyv.epub --start-chapter 1 --end-chapter 2

# Specify a custom Piper model path
python tools/epub_to_audiobook_piper.py -i konyv.epub -m /path/to/hu_HU-anna-medium.onnx
```

---

## 3. Reading on your Kobo with KOReader

1. **Ensure `audiobook.koplugin` is installed on your Kobo**:
   Copy the `audiobook.koplugin` directory to:
   `.adds/koreader/plugins/audiobook.koplugin/`

2. **Copy the generated `.epub` to your Kobo**:
   Copy `konyv_audiobook.epub` into your books folder on the Kobo (e.g. `/mnt/onboard/books/`).

3. **Open the book in KOReader**:
   - Open `konyv_audiobook.epub`.
   - **To start reading from the beginning / current page**:
     Go to the top menu $\rightarrow$ **Tools** $\rightarrow$ **Audiobook Read-Along** $\rightarrow$ **Start reading from current page** or tap the book icon.
   - **To jump and play from any specific word/sentence**:
     Long-press any word on the screen $\rightarrow$ tap **Play aligned audiobook from here** (or **Read aloud from here**).
   - The audio begins immediately from that sentence/word, and the text highlights smoothly on your Kobo screen as the narration progresses!

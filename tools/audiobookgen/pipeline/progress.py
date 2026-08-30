"""
Progress reporting for console output and line-oriented progress logging.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Optional, Protocol, runtime_checkable

from tools.audiobookgen.epub.smil import format_smil_clock
from tools.audiobookgen.models import ChapterResult


@runtime_checkable
class ProgressReporter(Protocol):
    """Protocol for tracking audiobook generation progress."""

    def on_book_start(self, total_chapters: int, total_sentences: int, total_chars: int) -> None: ...
    def on_chapter_start(self, chapter_index: int, total_chapters: int, title: str, sentence_count: int, char_count: int) -> None: ...
    def on_sentence_synthesized(self, chapter_index: int, sentence_index: int, total_sentences: int, cached: bool, retry_count: int, char_count: int) -> None: ...
    def on_chapter_complete(self, result: ChapterResult) -> None: ...
    def on_book_complete(self, total_chapters: int, total_duration: float, total_sentences: int, cache_hits: int, retries: int) -> None: ...


class ConsoleProgressReporter:
    """
    Real-time terminal progress reporter with rolling ETA estimation, throughput metrics,
    and work-dir progress.log file logging.
    """

    def __init__(self, log_path: Optional[Path | str] = None):
        self.log_file = Path(log_path).resolve() if log_path else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.start_time = time.time()
        self.total_book_chars = 0
        self.processed_chars = 0
        self.total_sentences = 0
        self.synthesized_sentences = 0
        self.cache_hits = 0
        self.retries = 0
        self.total_chapters = 0
        self.current_chapter = 0

        self._recent_rates: list[float] = []  # chars per second
        self._last_sentence_time = time.time()

    def _log_to_file(self, msg: str) -> None:
        if self.log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {msg}\n")

    def on_book_start(self, total_chapters: int, total_sentences: int, total_chars: int) -> None:
        self.start_time = time.time()
        self.total_chapters = total_chapters
        self.total_sentences = total_sentences
        self.total_book_chars = total_chars
        self.processed_chars = 0
        self.synthesized_sentences = 0
        self.cache_hits = 0
        self.retries = 0

        header = (
            f"Starting generation: {total_chapters} chapters | "
            f"~{total_sentences} sentences | ~{total_chars:,} characters"
        )
        print("=" * 65)
        print(header)
        print("=" * 65)
        self._log_to_file(header)

    def on_chapter_start(
        self,
        chapter_index: int,
        total_chapters: int,
        title: str,
        sentence_count: int,
        char_count: int,
    ) -> None:
        self.current_chapter = chapter_index
        msg = f"Chapter [{chapter_index}/{total_chapters}]: '{title}' ({sentence_count} sentences, {char_count:,} chars)"
        print(f"\n---> {msg}")
        self._log_to_file(msg)

    def on_sentence_synthesized(
        self,
        chapter_index: int,
        sentence_index: int,
        total_sentences: int,
        cached: bool,
        retry_count: int,
        char_count: int,
    ) -> None:
        now = time.time()
        elapsed = max(0.001, now - self.start_time)
        dt = max(0.001, now - self._last_sentence_time)
        self._last_sentence_time = now

        self.synthesized_sentences += 1
        self.processed_chars += char_count

        if cached:
            self.cache_hits += 1
        else:
            chars_per_sec = char_count / dt
            self._recent_rates.append(chars_per_sec)
            if len(self._recent_rates) > 20:
                self._recent_rates.pop(0)

        if retry_count > 0:
            self.retries += retry_count

        # Estimate remaining time
        remaining_chars = max(0, self.total_book_chars - self.processed_chars)
        avg_rate = sum(self._recent_rates) / len(self._recent_rates) if self._recent_rates else 30.0
        eta_seconds = remaining_chars / max(1.0, avg_rate)
        eta_str = format_smil_clock(eta_seconds)
        elapsed_str = format_smil_clock(elapsed)

        sents_per_sec = self.synthesized_sentences / elapsed
        status_tag = "CACHED" if cached else f"SYNTH" + (f" (retry x{retry_count})" if retry_count > 0 else "")

        line = (
            f"\r    [{sentence_index}/{total_sentences}] {status_tag} | "
            f"Speed: {sents_per_sec:.2f} sent/s | Elapsed: {elapsed_str} | ETA: {eta_str}  "
        )
        sys.stdout.write(line)
        sys.stdout.flush()

    def on_chapter_complete(self, result: ChapterResult) -> None:
        sys.stdout.write("\n")
        dur_clock = format_smil_clock(result.duration)
        msg = f"  Completed chapter {result.chapter_index} ({result.item_id}): audio duration = {dur_clock}"
        print(msg)
        self._log_to_file(msg)

    def on_book_complete(
        self,
        total_chapters: int,
        total_duration: float,
        total_sentences: int,
        cache_hits: int,
        retries: int,
    ) -> None:
        elapsed = time.time() - self.start_time
        msg = (
            f"BOOK GENERATION COMPLETE!\n"
            f"  Total chapters:    {total_chapters}\n"
            f"  Total audio:       {format_smil_clock(total_duration)}\n"
            f"  Total sentences:   {total_sentences} ({cache_hits} cache hits, {retries} retries)\n"
            f"  Elapsed wall time: {format_smil_clock(elapsed)}\n"
        )
        print("\n" + "=" * 65)
        print(msg.strip())
        print("=" * 65)
        self._log_to_file(msg.strip())

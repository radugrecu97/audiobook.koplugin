"""
Book generation service orchestrating the whole EPUB-to-audiobook pipeline,
resuming, dry-run estimation, and final packaging.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional

from tools.audiobookgen.epub.markup import SentenceSpanInjector
from tools.audiobookgen.epub.package import EpubPackage
from tools.audiobookgen.epub.smil import format_smil_clock
from tools.audiobookgen.epub.writer import EpubWriter
from tools.audiobookgen.models import ChapterResult, SentenceTiming, VoiceProfile
from tools.audiobookgen.pipeline.cache import DiskSentenceCache, SentenceCache
from tools.audiobookgen.pipeline.chapter import ChapterSynthesisService
from tools.audiobookgen.pipeline.progress import ConsoleProgressReporter, ProgressReporter

logger = logging.getLogger(__name__)


class BookGenerationService:
    """
    Coordinates end-to-end audiobook generation from input EPUB/KEpub to output EPUB 3 Media Overlays.
    """

    def __init__(
        self,
        chapter_service: ChapterSynthesisService,
        cache: SentenceCache,
        progress_reporter: Optional[ProgressReporter] = None,
    ):
        self.chapter_service = chapter_service
        self.cache = cache
        self.progress_reporter = progress_reporter or ConsoleProgressReporter()

    def _compute_chapter_fingerprint(
        self,
        voice: VoiceProfile,
        lang: str,
        audio_mime: str,
        audio_bitrate: str,
    ) -> str:
        items = [
            str(voice.model_id),
            str(voice.ref_text or ""),
            str(voice.style_tag),
            f"{voice.temperature:.4f}",
            f"{voice.top_p:.4f}",
            f"{voice.repetition_penalty:.4f}",
            str(voice.seed),
            f"{voice.speed:.4f}",
            str(voice.chunk_length),
            lang.lower(),
            audio_mime,
            audio_bitrate,
        ]
        return hashlib.sha256("|".join(items).encode("utf-8")).hexdigest()

    def run(
        self,
        input_epub: Path | str,
        output_epub: Path | str,
        voice: VoiceProfile,
        lang: str = "en",
        start_chapter: int = 1,
        end_chapter: Optional[int] = None,
        resume: bool = True,
        dry_run: bool = False,
        audio_ext: str = ".mp3",
        audio_bitrate: str = "64k",
    ) -> Optional[Path]:
        in_path = Path(input_epub).resolve()
        out_path = Path(output_epub).resolve()

        with EpubPackage(in_path) as package:
            # 1. Filter spine content documents
            all_docs = list(package.iter_content_documents())
            target_docs = [
                d for d in all_docs
                if d[0] >= start_chapter and (end_chapter is None or d[0] <= end_chapter)
            ]

            if not target_docs:
                print(f"No content documents found in chapter range {start_chapter}..{end_chapter or 'end'}.")
                return None

            # 2. Dry run mode
            if dry_run:
                self._run_dry_run(target_docs, lang)
                return None

            # 3. Pre-scan for total character and sentence count estimation
            total_est_sentences = 0
            total_est_chars = 0
            injector = SentenceSpanInjector()
            for ch_idx, item_id, href, full_p, xhtml in target_docs:
                clean_xhtml = self.chapter_service.unwrapper.unwrap(xhtml)
                _, sents = injector.inject_spans(clean_xhtml, chapter_index=ch_idx, lang=lang)
                total_est_sentences += len(sents)
                total_est_chars += sum(len(s[1]) for s in sents)

            self.progress_reporter.on_book_start(
                total_chapters=len(target_docs),
                total_sentences=total_est_sentences,
                total_chars=total_est_chars,
            )

            # 4. Process each chapter
            results: List[ChapterResult] = []
            total_duration = 0.0
            fingerprint = self._compute_chapter_fingerprint(
                voice=voice,
                lang=lang,
                audio_mime="audio/mpeg" if audio_ext == ".mp3" else "audio/ogg" if "ogg" in audio_ext or "opus" in audio_ext else "audio/wav",
                audio_bitrate=audio_bitrate,
            )

            for ch_idx, item_id, href, full_p, raw_xhtml in target_docs:
                # Check resume state
                if resume and self.cache.is_chapter_completed(ch_idx, fingerprint):
                    st = self.cache.get_chapter_state(ch_idx)
                    audio_data = self.cache.load_completed_chapter_audio(ch_idx, audio_ext=audio_ext)
                    if st and audio_data:
                        timings = [SentenceTiming.from_dict(t) for t in st.get("timings", [])]
                        res = ChapterResult(
                            chapter_index=ch_idx,
                            item_id=item_id,
                            href=href,
                            marked_xhtml=st["marked_xhtml"],
                            smil_xml=st["smil_xml"],
                            audio_bytes=audio_data,
                            audio_mime=st.get("audio_mime", "audio/mpeg"),
                            duration=float(st.get("duration", 0.0)),
                            timings=timings,
                        )
                        self.progress_reporter.on_chapter_start(
                            chapter_index=ch_idx,
                            total_chapters=len(target_docs),
                            title=href,
                            sentence_count=len(timings),
                            char_count=sum(len(t.text) for t in timings),
                        )
                        self.progress_reporter.on_chapter_complete(res)
                        results.append(res)
                        total_duration += res.duration
                        continue

                # Process fresh or partially-cached chapter
                clean_xhtml = self.chapter_service.unwrapper.unwrap(raw_xhtml)
                _, preview_sents = injector.inject_spans(clean_xhtml, chapter_index=ch_idx, lang=lang)
                self.progress_reporter.on_chapter_start(
                    chapter_index=ch_idx,
                    total_chapters=len(target_docs),
                    title=href,
                    sentence_count=len(preview_sents),
                    char_count=sum(len(s[1]) for s in preview_sents),
                )

                res = self.chapter_service.process_chapter(
                    chapter_index=ch_idx,
                    item_id=item_id,
                    href=href,
                    raw_xhtml=raw_xhtml,
                    voice=voice,
                    lang=lang,
                    progress_reporter=self.progress_reporter,
                )

                # Persist chapter state to cache
                if res.audio_bytes:
                    state_dict = {
                        "item_id": item_id,
                        "href": href,
                        "marked_xhtml": res.marked_xhtml,
                        "smil_xml": res.smil_xml,
                        "audio_mime": res.audio_mime,
                        "duration": res.duration,
                        "fingerprint": fingerprint,
                        "timings": [t.to_dict() for t in res.timings],
                    }
                    self.cache.save_chapter_state(
                        chapter_index=ch_idx,
                        state=state_dict,
                        audio_bytes=res.audio_bytes,
                        audio_ext=audio_ext,
                    )

                self.progress_reporter.on_chapter_complete(res)
                results.append(res)
                total_duration += res.duration

            # 5. Package final aligned EPUB
            print("\nPackaging final EPUB 3 with Media Overlays...")
            writer = EpubWriter(package)
            out_file = writer.write_audiobook_epub(
                output_path=out_path,
                chapter_results=results,
                total_duration=total_duration,
            )

            # 6. Save QC report if any retries occurred
            if hasattr(self.chapter_service.retry_policy, "reports") and self.chapter_service.retry_policy.reports:
                if isinstance(self.cache, DiskSentenceCache):
                    qc_report_path = self.cache.work_dir / "qc_report.json"
                    qc_report_path.write_text(
                        json.dumps(self.chapter_service.retry_policy.reports, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    print(f"QC report written to: {qc_report_path}")

            cache_hits = getattr(self.progress_reporter, "cache_hits", 0)
            retries = getattr(self.progress_reporter, "retries", 0)
            self.progress_reporter.on_book_complete(
                total_chapters=len(results),
                total_duration=total_duration,
                total_sentences=total_est_sentences,
                cache_hits=cache_hits,
                retries=retries,
            )

            file_size_mb = out_file.stat().st_size / (1024 * 1024)
            print(f"SUCCESS! Output aligned audiobook EPUB: {out_file} ({file_size_mb:.1f} MB)")
            return out_file

    def _run_dry_run(self, target_docs: list, lang: str) -> None:
        injector = SentenceSpanInjector()
        print("=" * 65)
        print("DRY RUN: EPUB Content Document & Audio Estimation")
        print("=" * 65)

        total_sentences = 0
        total_chars = 0

        for ch_idx, item_id, href, full_p, xhtml in target_docs:
            clean_xhtml = self.chapter_service.unwrapper.unwrap(xhtml)
            _, sents = injector.inject_spans(clean_xhtml, chapter_index=ch_idx, lang=lang)
            ch_chars = sum(len(s[1]) for s in sents)
            total_sentences += len(sents)
            total_chars += ch_chars
            est_audio_sec = ch_chars * 0.071  # ~14 chars per sec
            print(f"  [Ch {ch_idx:03d}] {href:<35} | {len(sents):>4} sentences | {ch_chars:>6,} chars | ~{format_smil_clock(est_audio_sec)}")

        total_audio_sec = total_chars * 0.071
        # Approx 0.25x RTF on RTX 5080 / Modern GPU
        est_synth_sec = total_audio_sec * 0.25

        print("-" * 65)
        print(f"TOTALS:")
        print(f"  Chapters to process:   {len(target_docs)}")
        print(f"  Total sentences:       {total_sentences:,}")
        print(f"  Total character count: {total_chars:,}")
        print(f"  Estimated Audio Time:  {format_smil_clock(total_audio_sec)} (~{total_audio_sec/3600:.1f} hours)")
        print(f"  Estimated GPU Time:    ~{format_smil_clock(est_synth_sec)} (at ~0.25x RTF)")
        print("=" * 65)

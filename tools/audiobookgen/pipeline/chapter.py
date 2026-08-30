"""
Chapter synthesis service orchestrating text extraction, TTS, QC retry, audio assembly, and SMIL creation.
"""

from __future__ import annotations

from pathlib import Path
import posixpath
from typing import List, Optional, Tuple

from tools.audiobookgen.audio.assembler import ChapterAudioAssembler
from tools.audiobookgen.audio.encoder import AudioEncoder, Mp3Encoder
from tools.audiobookgen.audio.pause import FixedPause, PausePolicy
from tools.audiobookgen.audio.pcm import apply_speed_scale
from tools.audiobookgen.epub.kepub import KepubUnwrapper
from tools.audiobookgen.epub.markup import SentenceSpanInjector
from tools.audiobookgen.epub.smil import SmilBuilder
from tools.audiobookgen.models import (
    AudioClip,
    ChapterResult,
    SentenceTiming,
    SynthesisRequest,
    VoiceProfile,
)
from tools.audiobookgen.pipeline.cache import DiskSentenceCache, SentenceCache
from tools.audiobookgen.pipeline.progress import ProgressReporter
from tools.audiobookgen.qc.retry import RetryPolicy
from tools.audiobookgen.text.normalizer import MultilingualNormalizer, TextNormalizer
from tools.audiobookgen.text.splitter import RegexSentenceSplitter, SentenceSplitter
from tools.audiobookgen.tts.fishspeech import FishSpeechProvider
from tools.audiobookgen.tts.provider import TTSProvider


class ChapterSynthesisService:
    """
    Synthesizes a single EPUB chapter content document into aligned Media Overlays and audio.
    """

    def __init__(
        self,
        tts_provider: TTSProvider,
        retry_policy: RetryPolicy,
        cache: SentenceCache,
        encoder: Optional[AudioEncoder] = None,
        pause_policy: Optional[PausePolicy] = None,
        normalizer: Optional[TextNormalizer] = None,
        splitter: Optional[SentenceSplitter] = None,
        injector: Optional[SentenceSpanInjector] = None,
        unwrapper: Optional[KepubUnwrapper] = None,
        assembler: Optional[ChapterAudioAssembler] = None,
    ):
        self.tts_provider = tts_provider
        self.retry_policy = retry_policy
        self.cache = cache
        self.encoder = encoder or Mp3Encoder()
        self.pause_policy = pause_policy or FixedPause(0.45)
        self.normalizer = normalizer or MultilingualNormalizer()
        self.splitter = splitter or RegexSentenceSplitter(normalizer=self.normalizer)
        self.injector = injector or SentenceSpanInjector(splitter=self.splitter)
        self.unwrapper = unwrapper or KepubUnwrapper()
        self.assembler = assembler or ChapterAudioAssembler(default_pause_policy=self.pause_policy)

    def process_chapter(
        self,
        chapter_index: int,
        item_id: str,
        href: str,
        raw_xhtml: str,
        voice: VoiceProfile,
        lang: str = "en",
        progress_reporter: Optional[ProgressReporter] = None,
    ) -> ChapterResult:
        # 1. Clean Kobo kepub markup
        clean_xhtml = self.unwrapper.unwrap(raw_xhtml)

        # 2. Inject sentence spans
        marked_xhtml, sentence_list = self.injector.inject_spans(
            clean_xhtml,
            chapter_index=chapter_index,
            lang=lang,
        )

        if not sentence_list:
            return ChapterResult(
                chapter_index=chapter_index,
                item_id=item_id,
                href=href,
                marked_xhtml=marked_xhtml,
                smil_xml="",
                audio_bytes=b"",
                audio_mime="audio/mpeg",
                duration=0.0,
                timings=[],
            )

        # 3. Synthesize or retrieve each sentence
        total_sentences = len(sentence_list)
        assembled_items: List[Tuple[str, str, AudioClip]] = []

        for sent_idx, (span_id, text) in enumerate(sentence_list, start=1):
            norm_text = self.normalizer.normalize(text, lang=lang)
            req = SynthesisRequest(text=norm_text, lang=lang, voice=voice)
            key = self.cache.compute_sentence_key(req)

            cached = self.cache.get_sentence(key)
            if cached is not None:
                clip, meta = cached
                retries = (meta or {}).get("retries", 0)
                is_hit = True
            else:
                clip, qc_result, retries = self.retry_policy.synthesize_with_retry(
                    req, chapter_index=chapter_index, span_id=span_id
                )
                if voice.speed != 1.0:
                    clip = apply_speed_scale(clip, voice.speed)

                self.cache.put_sentence(
                    key,
                    clip,
                    metadata={
                        "text": text,
                        "normalized": norm_text,
                        "qc": qc_result.to_dict(),
                        "retries": retries,
                    },
                )
                is_hit = False

            if progress_reporter:
                progress_reporter.on_sentence_synthesized(
                    chapter_index=chapter_index,
                    sentence_index=sent_idx,
                    total_sentences=total_sentences,
                    cached=is_hit,
                    retry_count=retries,
                    char_count=len(text),
                )

            assembled_items.append((span_id, text, clip))

        # 4. Assemble chapter audio and sentence timings
        wav_bytes, timings = self.assembler.assemble(
            assembled_items, pause_policy=self.pause_policy
        )

        # 5. Encode audio to target format (MP3 / WAV / Opus)
        audio_bytes, audio_mime = self.encoder.encode(wav_bytes)

        # 6. Calculate relative paths for SMIL
        ext = ".mp3" if "mpeg" in audio_mime else ".ogg" if "ogg" in audio_mime or "opus" in audio_mime else ".wav"
        audio_filename = f"ch_{chapter_index:03d}{ext}"
        audio_rel_path = posixpath.relpath(f"Audio/{audio_filename}", "MediaOverlays")
        xhtml_rel_path = posixpath.relpath(href, "MediaOverlays")

        # 7. Build SMIL document
        smil_xml = SmilBuilder.build(
            timings=timings,
            xhtml_relative_path=xhtml_rel_path,
            audio_relative_path=audio_rel_path,
        )

        chapter_duration = timings[-1].end if timings else 0.0

        return ChapterResult(
            chapter_index=chapter_index,
            item_id=item_id,
            href=href,
            marked_xhtml=marked_xhtml,
            smil_xml=smil_xml,
            audio_bytes=audio_bytes,
            audio_mime=audio_mime,
            duration=chapter_duration,
            timings=timings,
        )

"""
Retry policy coordinating TTS synthesis attempts, seeds, and QC verification.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from tools.audiobookgen.models import AudioClip, SynthesisRequest, VerificationResult
from tools.audiobookgen.qc.verifier import NullVerifier, SynthesisVerifier
from tools.audiobookgen.tts.provider import TTSProvider

logger = logging.getLogger(__name__)


class RetryPolicy:
    """
    Manages retrying synthesis with adjusted random seeds when QC verification fails.
    """

    def __init__(
        self,
        provider: TTSProvider,
        verifier: Optional[SynthesisVerifier] = None,
        max_retries: int = 2,
    ):
        self.provider = provider
        self.verifier = verifier or NullVerifier()
        self.max_retries = max(0, int(max_retries))
        self.reports: List[Dict[str, Any]] = []

    def synthesize_with_retry(
        self,
        request: SynthesisRequest,
        chapter_index: int = 0,
        span_id: str = "",
    ) -> Tuple[AudioClip, VerificationResult, int]:
        """
        Synthesize text with QC verification and automatic retries.

        Returns:
            Tuple of (AudioClip, VerificationResult, retry_count).
        """
        base_seed = request.voice.seed

        # Attempt 0 (initial attempt)
        clip = self.provider.synthesize(request)
        qc = self.verifier.verify(clip, request.text, lang=request.lang)

        if qc.ok or self.max_retries == 0:
            if not qc.ok:
                self._record_report(chapter_index, span_id, request.text, qc, 0, base_seed, "accepted_without_retry")
            return clip, qc, 0

        # Keep best attempt
        best_clip = clip
        best_qc = qc
        best_attempt = 0

        for attempt in range(1, self.max_retries + 1):
            seed_override = base_seed + (attempt * 1000) + attempt
            logger.warning(
                f"QC flagged sentence [{span_id}] in chapter {chapter_index} ({qc.reason}). "
                f"Retrying attempt {attempt}/{self.max_retries} with seed {seed_override}..."
            )

            retry_request = SynthesisRequest(
                text=request.text,
                lang=request.lang,
                voice=request.voice,
                seed_override=seed_override,
            )

            try:
                attempt_clip = self.provider.synthesize(retry_request)
                attempt_qc = self.verifier.verify(attempt_clip, request.text, lang=request.lang)

                if attempt_qc.score < best_qc.score:
                    best_clip = attempt_clip
                    best_qc = attempt_qc
                    best_attempt = attempt

                if attempt_qc.ok:
                    self._record_report(
                        chapter_index, span_id, request.text, attempt_qc, attempt, seed_override, "resolved_on_retry"
                    )
                    return attempt_clip, attempt_qc, attempt

            except Exception as e:
                logger.warning(f"Retry attempt {attempt} failed: {e}")

        # All retries exhausted: keep best scoring attempt
        self._record_report(
            chapter_index, span_id, request.text, best_qc, best_attempt, base_seed, "unresolved_best_effort"
        )
        return best_clip, best_qc, self.max_retries

    def _record_report(
        self,
        chapter_index: int,
        span_id: str,
        source_text: str,
        qc: VerificationResult,
        attempts: int,
        final_seed: int,
        disposition: str,
    ) -> None:
        self.reports.append({
            "chapter_index": chapter_index,
            "span_id": span_id,
            "source_text": source_text,
            "transcript": qc.transcript,
            "score": qc.score,
            "reason": qc.reason,
            "attempts": attempts,
            "final_seed": final_seed,
            "disposition": disposition,
        })

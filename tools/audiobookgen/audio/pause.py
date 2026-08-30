"""
Pause duration policies between sentences.
"""

from __future__ import annotations

import random
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class PausePolicy(Protocol):
    """Protocol for calculating silence pause duration after a sentence."""

    def pause_after(self, sentence_text: str, index: int, total: int) -> float:
        """
        Calculate pause duration in seconds after sentence at index (0-indexed).
        Returns 0.0 if no pause is needed (e.g. after the last sentence).
        """
        ...


class FixedPause:
    """Fixed pause duration between sentences."""

    def __init__(self, duration: float = 0.45):
        self.duration = max(0.0, float(duration))

    def pause_after(self, sentence_text: str, index: int, total: int) -> float:
        if index >= total - 1:
            return 0.0
        return self.duration


class VariancePause:
    """
    Pause duration with deterministic pseudo-random jitter around base duration.
    Uses seeded RNG for reproducible runs.
    """

    def __init__(
        self,
        base_duration: float = 0.45,
        variance: float = 0.10,
        seed: Optional[int] = 42,
    ):
        self.base_duration = max(0.0, float(base_duration))
        self.variance = max(0.0, float(variance))
        self.seed = seed
        self._rng = random.Random(seed) if seed is not None else random.Random()

    def pause_after(self, sentence_text: str, index: int, total: int) -> float:
        if index >= total - 1 or self.base_duration <= 0:
            return 0.0

        if self.variance > 0:
            val = self._rng.uniform(
                self.base_duration - self.variance,
                self.base_duration + self.variance,
            )
            return max(0.05, val)

        return self.base_duration


class PunctuationAwarePause:
    """
    Pause duration varying according to the ending punctuation mark of the sentence.
    """

    def __init__(
        self,
        base_duration: float = 0.45,
        comma_factor: float = 0.5,
        question_excl_factor: float = 1.2,
        ellipsis_factor: float = 1.4,
        variance: float = 0.05,
        seed: Optional[int] = 42,
    ):
        self.base_duration = max(0.0, float(base_duration))
        self.comma_factor = float(comma_factor)
        self.question_excl_factor = float(question_excl_factor)
        self.ellipsis_factor = float(ellipsis_factor)
        self.variance = max(0.0, float(variance))
        self._rng = random.Random(seed) if seed is not None else random.Random()

    def pause_after(self, sentence_text: str, index: int, total: int) -> float:
        if index >= total - 1 or self.base_duration <= 0:
            return 0.0

        text = sentence_text.strip()
        factor = 1.0

        if text.endswith(("...", "…")):
            factor = self.ellipsis_factor
        elif text.endswith(("?", "!")):
            factor = self.question_excl_factor
        elif text.endswith((",", ";", ":", "—", "–")):
            factor = self.comma_factor

        target = self.base_duration * factor
        if self.variance > 0:
            target = self._rng.uniform(
                max(0.05, target - self.variance),
                target + self.variance,
            )

        return max(0.05, target)

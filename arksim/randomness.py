"""Deterministic probability primitives used by the battle engine.

The game client's exact random-number generator and initial state are not
public.  This module therefore defines a stable simulator-owned random axis:
the same seed and the same sequence of probability requests always produce
the same result, independently of Python's process hash seed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


_MASK_32 = (1 << 32) - 1
_MASK_64 = (1 << 64) - 1
_UINT32_RANGE = 1 << 32


class DeterministicRng:
    """Small fixed-algorithm PCG32 stream with observable draw count."""

    def __init__(self, seed: int = 0, *, sequence: int = 0) -> None:
        self._state = 0
        self._increment = ((int(sequence) & _MASK_64) << 1 | 1) & _MASK_64
        self._next_raw()
        self._state = (self._state + (int(seed) & _MASK_64)) & _MASK_64
        self._next_raw()
        self.draw_count = 0

    def _next_raw(self) -> int:
        old_state = self._state
        self._state = (
            old_state * 6364136223846793005 + self._increment
        ) & _MASK_64
        xorshifted = (((old_state >> 18) ^ old_state) >> 27) & _MASK_32
        rotation = (old_state >> 59) & 31
        return (
            (xorshifted >> rotation)
            | (xorshifted << ((-rotation) & 31))
        ) & _MASK_32

    def next_uint32(self) -> int:
        self.draw_count += 1
        return self._next_raw()

    def random(self) -> float:
        """Return one value in the half-open interval [0, 1)."""
        return self.next_uint32() / _UINT32_RANGE

    def roll(self, probability: float) -> bool:
        """Perform one PRTS-style strict-less-than probability check.

        Zero and one are resolved without consuming the random axis, matching
        the documented Dice/DicePRD boundary behaviour.
        """
        probability = float(probability)
        if math.isnan(probability):
            raise ValueError("probability must not be NaN")
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        return self.random() < probability

    def integer(self, lower: int, upper: int) -> int:
        """Return a uniform integer in [lower, upper) without modulo bias."""
        lower = int(lower)
        upper = int(upper)
        width = upper - lower
        if width <= 0:
            raise ValueError("upper must be greater than lower")
        if width > _UINT32_RANGE:
            raise ValueError("integer range is wider than 32 bits")
        limit = _UINT32_RANGE - (_UINT32_RANGE % width)
        while True:
            value = self.next_uint32()
            if value < limit:
                return lower + value % width

    def choose_index(self, count: int) -> int:
        if count <= 0:
            raise ValueError("cannot choose from an empty collection")
        return self.integer(0, count)


@dataclass
class IncreasingPrd:
    """DicePRD with an initial chance, failure increment and optional pity."""

    initial_probability: float
    failure_increment: float
    guarantee_at: int | None = None
    failures: int = 0

    @property
    def current_probability(self) -> float:
        return min(
            1.0,
            max(
                0.0,
                float(self.initial_probability)
                + self.failures * float(self.failure_increment),
            ),
        )

    def roll(self, rng: DeterministicRng) -> bool:
        if self.guarantee_at is not None and self.guarantee_at <= 0:
            raise ValueError("guarantee_at must be positive")
        guaranteed = (
            self.guarantee_at is not None
            and self.failures + 1 >= self.guarantee_at
        )
        success = guaranteed or rng.roll(self.current_probability)
        self.failures = 0 if success else self.failures + 1
        return success


# PRTS table for the global-blackboard DicePRD variant, indexed by the
# expected probability rounded to the nearest whole percent.
EXPECTED_PRD_BASE = (
    0.0, 0.00016, 0.00062, 0.00139, 0.00245, 0.0038, 0.00544,
    0.00736, 0.00955, 0.01202, 0.01475, 0.01774, 0.02098, 0.02448,
    0.02823, 0.03222, 0.03645, 0.04092, 0.04562, 0.05055, 0.0557,
    0.06108, 0.06668, 0.07249, 0.07851, 0.08474, 0.09118, 0.09783,
    0.10467, 0.11171, 0.11895, 0.12638, 0.134, 0.14181, 0.14981,
    0.15798, 0.16633, 0.17491, 0.18362, 0.19249, 0.20155, 0.21092,
    0.22037, 0.2299, 0.23954, 0.24931, 0.25987, 0.27045, 0.28101,
    0.29155, 0.3021, 0.31268, 0.32329, 0.33412, 0.34737, 0.3604,
    0.37322, 0.38584, 0.39828, 0.41054, 0.42265, 0.4346, 0.44642,
    0.4581, 0.46967, 0.48113, 0.49248, 0.50746, 0.52941, 0.55072,
    0.57143, 0.59155, 0.61111, 0.63014, 0.64865, 0.66667, 0.68421,
    0.7013, 0.71795, 0.73418, 0.75, 0.76543, 0.78049, 0.79518,
    0.80952, 0.82353, 0.83721, 0.85058, 0.86364, 0.8764, 0.88889,
    0.9011, 0.91304, 0.92473, 0.93617, 0.94737, 0.95833, 0.96907,
    0.97959, 0.9899, 1.0,
)


@dataclass
class ExpectedProbabilityPrd:
    """DicePRD whose long-run target chance is converted through the table."""

    expected_probability: float
    failures: int = 0

    @property
    def table_index(self) -> int:
        probability = max(0.0, min(float(self.expected_probability), 1.0))
        return min(100, math.floor(probability * 100.0 + 0.5))

    @property
    def base_probability(self) -> float:
        return EXPECTED_PRD_BASE[self.table_index]

    @property
    def current_probability(self) -> float:
        return min(1.0, (self.failures + 1) * self.base_probability)

    def roll(self, rng: DeterministicRng) -> bool:
        success = rng.roll(self.current_probability)
        self.failures = 0 if success else self.failures + 1
        return success

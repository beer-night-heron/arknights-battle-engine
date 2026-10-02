"""Batch-simulation statistics independent of battle content."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import NormalDist, fmean
from typing import Any, Iterable


@dataclass(frozen=True)
class BatchSummary:
    runs: int
    wins: int
    win_rate: float
    confidence: float
    win_rate_low: float
    win_rate_high: float
    mean_time: float
    p50_time: float
    p95_time: float
    mean_life_points: float
    mean_enemies_leaked: float
    behavior_warning_runs: int
    mechanic_warning_runs: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def wilson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between zero and trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    proportion = successes / trials
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (proportion + z2 / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z2 / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of no values")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    ratio = position - lower
    return ordered[lower] * (1.0 - ratio) + ordered[upper] * ratio


def summarize_results(
    results: Iterable[Any], *, confidence: float = 0.95
) -> BatchSummary:
    values = list(results)
    if not values:
        raise ValueError("cannot summarize an empty batch")
    wins = sum(bool(result.win) for result in values)
    low, high = wilson_interval(wins, len(values), confidence)
    times = [float(result.time) for result in values]
    return BatchSummary(
        runs=len(values),
        wins=wins,
        win_rate=wins / len(values),
        confidence=confidence,
        win_rate_low=low,
        win_rate_high=high,
        mean_time=fmean(times),
        p50_time=percentile(times, 0.50),
        p95_time=percentile(times, 0.95),
        mean_life_points=fmean(float(result.life_points) for result in values),
        mean_enemies_leaked=fmean(
            float(result.enemies_leaked) for result in values
        ),
        behavior_warning_runs=sum(
            int(getattr(result, "behavior_warnings", 0) > 0)
            for result in values
        ),
        mechanic_warning_runs=sum(
            int(getattr(result, "mechanic_warnings", 0) > 0)
            for result in values
        ),
    )

"""Optional semantic decision records and strict outcome replay.

Outcome replay still advances the simulator's PCG stream normally. It is a
controlled experiment, not a reconstruction of a client's random generator.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any, Callable

from .randomness import DeterministicRng


FORMAT = "arksim-random-decisions"
VERSION = 1


class RandomReplayError(ValueError):
    """The next decision does not match the supplied experiment."""


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


class RandomDecisionRng(DeterministicRng):
    """PCG32 with decision context, raw draws and optional outcome overrides."""

    def __init__(self, seed: int = 0, *, tape: dict | None = None) -> None:
        super().__init__(seed)
        self.seed = int(seed)
        self.entries: list[dict] = []
        self._raw: list[int] | None = None
        self._occurrences: dict[str, int] = {}
        self._tape: list[dict] | None = None
        if tape is not None:
            if (not isinstance(tape, dict) or tape.get("format") != FORMAT
                    or type(tape.get("version")) is not int or tape["version"] != VERSION
                    or not isinstance(tape.get("entries"), list)):
                raise RandomReplayError("invalid random decision tape format/version/entries")
            self._tape = _json_copy(tape["entries"])
            if not all(isinstance(entry, dict) for entry in self._tape):
                raise RandomReplayError("each random decision tape entry must be an object")

    def next_uint32(self) -> int:
        value = super().next_uint32()
        if self._raw is not None:
            self._raw.append(value)
        return value

    def _decide(self, kind: str, parameters: dict, context: dict | None,
                generate: Callable[[], Any], valid: Callable[[Any], bool]) -> Any:
        if self._raw is not None:
            raise RuntimeError("nested random decisions require a separate decision boundary")
        context = _json_copy(context or {})
        if not isinstance(context, dict):
            raise ValueError("random decision context must be an object")
        semantic = json.dumps([kind, {k: v for k, v in context.items() if k != "frame"}],
                              sort_keys=True, separators=(",", ":"))
        occurrence = self._occurrences.get(semantic, 0) + 1
        expected = {"kind": kind, "parameters": _json_copy(parameters),
                    "context": context, "occurrence": occurrence}
        position = len(self.entries)
        forced = None
        if self._tape is not None:
            if position >= len(self._tape):
                raise RandomReplayError(f"random decision #{position + 1}: tape exhausted; {expected}")
            entry = self._tape[position]
            for key, value in expected.items():
                # Canonical JSON also distinguishes booleans from integer IDs.
                if json.dumps(entry.get(key), sort_keys=True) != json.dumps(value, sort_keys=True):
                    raise RandomReplayError(f"random decision #{position + 1}: {key} mismatch; "
                                            f"expected {entry.get(key)!r}, actual {value!r}")
            forced = entry.get("result")
            if not valid(forced):
                raise RandomReplayError(f"random decision #{position + 1}: invalid result {forced!r}")
        before = self.draw_count
        raw: list[int] = []
        self._raw = raw
        try:
            natural = generate()
        finally:
            self._raw = None
        result = natural if self._tape is None else forced
        self._occurrences[semantic] = occurrence
        self.entries.append({"index": position + 1, **expected, "result": result,
                             "natural_result": natural, "overridden": result != natural,
                             "draw_before": before, "draw_after": self.draw_count,
                             "raw_uint32": raw})
        return result

    def roll(self, probability: float, *, context: dict | None = None) -> bool:
        probability = float(probability)
        if not math.isfinite(probability):
            raise ValueError("recorded probability must be finite")
        boundary = False if probability <= 0 else True if probability >= 1 else None
        return self._decide("roll", {"probability": probability}, context,
                            lambda: super(RandomDecisionRng, self).roll(probability),
                            lambda value: type(value) is bool and (boundary is None or value is boundary))

    def choose_index(self, count: int, *, context: dict | None = None) -> int:
        if type(count) is not int or not 0 < count <= 1 << 32:
            raise ValueError("choice count must be a positive 32-bit range")
        return self._decide("choice", {"count": count}, context,
                            lambda: super(RandomDecisionRng, self).choose_index(count),
                            lambda value: type(value) is int and 0 <= value < count)

    def uniform(self, lower: float, upper: float, *, context: dict | None = None) -> float:
        lower, upper = uniform_bounds(lower, upper)
        return self._decide("uniform", {"lower": lower, "upper": upper}, context,
                            lambda: uniform_value(self, lower, upper),
                            lambda value: type(value) in (int, float) and math.isfinite(value)
                            and (value == lower if lower == upper else lower <= value < upper))

    def assert_consumed(self) -> None:
        if self._tape is not None and len(self.entries) != len(self._tape):
            raise RandomReplayError(f"random decision tape has {len(self._tape) - len(self.entries)} unused entries")

    def export(self) -> dict:
        return {"format": FORMAT, "version": VERSION, "algorithm": "pcg32",
                "seed": self.seed, "outcome_replay": self._tape is not None,
                "draw_count": self.draw_count, "entries": deepcopy(self.entries)}


def uniform_bounds(lower: float, upper: float) -> tuple[float, float]:
    lower, upper = float(lower), float(upper)
    if not math.isfinite(lower) or not math.isfinite(upper) or not math.isfinite(upper - lower) or upper < lower:
        raise ValueError("uniform bounds and width must be finite and ordered")
    return lower, upper


def uniform_value(rng: DeterministicRng, lower: float, upper: float) -> float:
    if lower == upper:
        return lower
    value = lower + (upper - lower) * rng.random()
    # Floating-point addition may round a half-open sample to the upper bound.
    return min(value, math.nextafter(upper, lower))

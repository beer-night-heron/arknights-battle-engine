"""Small generic interfaces for deterministic strategy-space enumeration."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class SimulationCase:
    case_id: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class SeededCase:
    case: SimulationCase
    seed: int


class CaseEnumerator(Protocol):
    def __iter__(self) -> Iterator[SimulationCase]: ...


class CartesianEnumerator:
    """Enumerate the Cartesian product of named, finite parameter axes."""

    def __init__(
        self,
        axes: Mapping[str, Sequence[Any]],
        *,
        prefix: str = "case",
    ) -> None:
        self._names = tuple(axes)
        self._values = tuple(tuple(axes[name]) for name in self._names)
        self.prefix = prefix
        empty = [name for name, values in zip(self._names, self._values) if not values]
        if empty:
            raise ValueError(f"enumeration axes must not be empty: {empty}")

    def __iter__(self) -> Iterator[SimulationCase]:
        combinations = product(*self._values) if self._values else [()]
        for index, values in enumerate(combinations):
            yield SimulationCase(
                case_id=f"{self.prefix}_{index:06d}",
                parameters=dict(zip(self._names, values)),
            )


def expand_seeds(
    cases: Iterable[SimulationCase], seeds: Iterable[int]
) -> Iterator[SeededCase]:
    seed_values = tuple(int(seed) for seed in seeds)
    for case in cases:
        for seed in seed_values:
            yield SeededCase(case=case, seed=seed)

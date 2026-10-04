"""Battle-dependent gates over the existing static fragment timelines.

The one-frame departure handoff is a candidate model. Special branches and
condition actions require their own executors, not fabricated clear frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SpawnFragment:
    wave: int
    index: int
    end_time: float
    wave_start_time: float
    wave_end_time: float = 0.0
    post_delay: float = 0.0
    max_wait: float = -1.0
    events: list[Any] = field(default_factory=list)
    last: bool = False
    cancelled: bool = False


class DynamicWaveScheduler:
    def __init__(self, fragments: list[SpawnFragment]) -> None:
        self.fragments = fragments
        self.cursor = 0
        self.offset = 0.0
        self.wave_offsets: dict[int, float] = {}
        self.wave_departures: dict[int, float] = {}
        self.fragment_departures: dict[tuple[int, int], float] = {}
        self.last_births: dict[int, float] = {}
        self.forced_ends: dict[int, float] = {}
        self.events: list[dict[str, Any]] = []
        if fragments:
            self._activate(0.0, 0)

    @property
    def finished(self) -> bool:
        return self.cursor >= len(self.fragments)

    def _activate(self, time: float, frame: int) -> None:
        group = self.fragments[self.cursor]
        if group.wave not in self.wave_offsets:
            self.wave_offsets[group.wave] = self.offset
            self.events.append({"t": round(time, 3), "fixedFrame": frame,
                                "kind": "wave_ready", "wave": group.wave + 1,
                                "startTime": group.wave_start_time + self.offset})
        for event in group.events:
            event.time += self.offset
            event.nominal_time += self.offset
            event.wave_start_time += self.wave_offsets[group.wave]
            if event.logic_frame is not None:
                event.logic_frame += round(self.offset * 30)

    def departed(self, enemy: Any, time: float) -> None:
        if not enemy.dont_block_wave:
            self.wave_departures[enemy.wave_index] = time
        if enemy.block_fragment:
            self.fragment_departures[(enemy.wave_index, enemy.fragment_index)] = time

    def allowed(self, event: Any) -> bool:
        if event.cancelled:
            return True
        return not self.finished and (event.wave_index, event.fragment_index) == (
            self.fragments[self.cursor].wave, self.fragments[self.cursor].index)

    def advance(self, time: float, frame: int, dt: float, enemies: list[Any]) -> None:
        active = [enemy for enemy in enemies if not enemy.dead and not enemy.leaked]
        while not self.finished:
            group = self.fragments[self.cursor]
            if any(not event.dispatched for event in group.events):
                return
            end = group.end_time + self.offset
            if time + 1e-9 < end:
                return
            wave_blockers = [e for e in active if e.wave_index == group.wave and not e.dont_block_wave]
            fragment_blockers = [e for e in active if e.wave_index == group.wave
                                 and e.fragment_index == group.index and e.block_fragment]
            deadline = group.wave_start_time + self.wave_offsets[group.wave] + group.max_wait
            expired = group.max_wait > 0 and time + 1e-9 >= deadline
            next_index = self.cursor + 1
            truncated = group.wave in self.forced_ends
            if expired and not truncated and (not group.last or wave_blockers or fragment_blockers):
                skipped = []
                while next_index < len(self.fragments) and self.fragments[next_index].wave == group.wave:
                    skipped_group = self.fragments[next_index]
                    skipped_group.cancelled = True
                    skipped.append(skipped_group.index + 1)
                    for event in skipped_group.events:
                        event.cancelled = True
                    next_index += 1
                self.forced_ends[group.wave] = max(end + group.post_delay, time)
                self.events.append({"t": round(time, 3), "fixedFrame": frame, "kind": "wave_timeout",
                                    "wave": group.wave + 1, "skippedFragments": skipped})
                truncated = True
            while next_index < len(self.fragments) and self.fragments[next_index].cancelled:
                next_index += 1
            wave_boundary = group.last or truncated
            has_next_wave = next_index < len(self.fragments) and self.fragments[next_index].wave != group.wave
            if not expired and fragment_blockers and next_index < len(self.fragments):
                return
            if wave_boundary:
                release = self.forced_ends.get(group.wave, group.wave_end_time + self.offset)
                if has_next_wave and not expired:
                    if wave_blockers:
                        return
                    release = max(release, self.wave_departures.get(group.wave, -dt) + dt,
                                  self.fragment_departures.get((group.wave, group.index), -dt) + dt,
                                  self.last_births.get(group.wave, -dt) + dt)
                anchor = group.wave_end_time
            else:
                if fragment_blockers:
                    return
                release = max(end, self.fragment_departures.get((group.wave, group.index), -dt) + dt)
                anchor = group.end_time
            if time + 1e-9 < release:
                return
            self.cursor = next_index
            if not self.finished:
                self.offset = release - anchor
                self._activate(time, frame)

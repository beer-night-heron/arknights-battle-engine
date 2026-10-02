"""Fixed-frame target selector runtimes.

The client selector is a small state machine rather than a property of the
attack cooldown.  This module intentionally knows nothing about Battle or
enemy/operator types: callers provide the search and validation callbacks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


Target = object
TargetValidator = Callable[[Target], bool]
TargetSearcher = Callable[[], Target | None]


@dataclass
class FramePeriodicTicker:
    """A traditional countdown trigger advanced at most once per frame.

    ``tick`` only decrements the current countdown.  It deliberately does not
    schedule another period when the countdown reaches zero: the owner must
    call ``next`` after it actually performs the corresponding Search/Next.
    This is the observable behaviour of the traditional client ticker.
    """

    period_frames: int = 1
    wait_first_period: bool = False
    initialized: bool = False
    tick_count: int = 0
    next_ready_frame: int | None = None
    last_ticked_frame: int | None = None
    last_next_frame: int | None = None

    def __post_init__(self) -> None:
        self.period_frames = int(self.period_frames)
        if self.period_frames < 1:
            raise ValueError("period_frames must be at least 1")
        self.wait_first_period = bool(self.wait_first_period)

    @property
    def is_ready(self) -> bool:
        return self.initialized and self.tick_count <= 0

    def reset(
        self,
        frame_index: int = 0,
        *,
        wait_first_period: bool | None = None,
    ) -> None:
        frame = int(frame_index)
        if wait_first_period is not None:
            self.wait_first_period = bool(wait_first_period)
        self.initialized = True
        self.tick_count = (
            self.period_frames if self.wait_first_period else 0
        )
        self.last_ticked_frame = None
        self.next_ready_frame = frame + self.tick_count
        self.last_next_frame = None

    def tick(self, frame_index: int | None = None) -> bool:
        """Decrement once and return whether the ticker is ready.

        ``frame_index`` remains optional for small isolated callers; omitted
        calls advance an implicit integer frame.  Repeated calls for the same
        fixed frame do not consume another tick.
        """

        if frame_index is None:
            frame = (
                0
                if self.last_ticked_frame is None
                else self.last_ticked_frame + 1
            )
        else:
            frame = int(frame_index)
        if not self.initialized:
            self.reset(frame)
        if (
            self.last_ticked_frame is not None
            and frame <= self.last_ticked_frame
        ):
            return False
        self.last_ticked_frame = frame
        if self.tick_count > 0:
            self.tick_count -= 1
        return self.is_ready

    def next(self, frame_index: int | None = None) -> bool:
        """Consume a Search/Next and arm the next countdown period."""

        if not self.initialized:
            self.reset(0)
        if frame_index is None:
            frame = (
                0
                if self.last_ticked_frame is None
                else self.last_ticked_frame
            )
        else:
            frame = int(frame_index)
        if self.last_next_frame == frame:
            return False
        self.tick_count = self.period_frames
        self.next_ready_frame = frame + self.period_frames
        self.last_next_frame = frame
        return True

    @property
    def remaining_frames(self) -> int | None:
        if not self.initialized:
            return None
        return max(self.tick_count, 0)


@dataclass
class TargetSelectorRuntime:
    """Periodic target search with an explicitly validated cache."""

    ticker: FramePeriodicTicker
    cached_target: Target | None = None
    initialized: bool = False
    dirty: bool = True
    last_tick_ready: bool = False
    last_search_frame: int | None = None
    schedule_mode: str = "character_state"
    search_phase: str = "post_enemy_move"

    def reset(self, frame_index: int = 0) -> None:
        self.ticker.reset(frame_index)
        self.cached_target = None
        self.initialized = True
        self.dirty = True
        self.last_tick_ready = False
        self.last_search_frame = None

    def dispose(self) -> None:
        self.cached_target = None
        self.initialized = False
        self.dirty = True
        self.last_tick_ready = False
        self.last_search_frame = None

    def clear_cached_target(self) -> Target | None:
        """Clear the target without changing ticker phase."""

        previous = self.cached_target
        self.cached_target = None
        self.dirty = True
        return previous

    def tick(self, frame_index: int) -> bool:
        if not self.initialized:
            raise RuntimeError("target selector must be reset before ticking")
        self.last_tick_ready = self.ticker.tick(frame_index)
        return self.last_tick_ready

    def invalidate_if_needed(self, validate: TargetValidator) -> bool:
        """Clear an invalid cache immediately without starting a search."""

        target = self.cached_target
        if target is None or validate(target):
            return False
        self.cached_target = None
        self.dirty = True
        return True

    def refresh(
        self,
        search: TargetSearcher,
        validate: TargetValidator,
        *,
        force: bool = False,
        frame_index: int | None = None,
    ) -> bool:
        """Perform one Search and arm the next period.

        The method is usable both before and after ``tick`` in a fixed-frame
        pipeline.  A caller may pass the frame explicitly for the pre-Tick
        case; otherwise the most recently ticked frame is used.
        """

        if not self.initialized:
            raise RuntimeError("target selector must be reset before refresh")
        if frame_index is None:
            frame = self.ticker.last_ticked_frame
        else:
            frame = int(frame_index)
        if frame is None:
            frame = 0
        if self.last_search_frame == frame:
            return False
        if not force and not self.ticker.is_ready:
            return False

        candidate = search()
        if candidate is not None and not validate(candidate):
            candidate = None
        self.cached_target = candidate
        self.dirty = False
        self.last_search_frame = frame
        self.ticker.next(frame)
        # A search event is consumed by exactly one refresh, even if a caller
        # invokes refresh more than once during the same fixed frame.
        self.last_tick_ready = False
        return True

    def current_target(self, validate: TargetValidator) -> Target | None:
        self.invalidate_if_needed(validate)
        return self.cached_target

    def mark_dirty(self) -> None:
        self.dirty = True

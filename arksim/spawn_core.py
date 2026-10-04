"""Frame-based spawn scheduling ported from SusieGlitter's SpawnCore.

This implements static action-queue timing for the frame_core scheduling mode.
Runtime wave-clear gates remain optional inputs because they are not part of
the stage data itself.
"""

from __future__ import annotations

import math
from typing import Any


HZ = 30
MILLI_FRAMES = 1000
FRAGMENT_HANDOFF_FRAMES = 2
PREVIEW_CURSOR_PRE_DELAY = 3.0
PREVIEW_CURSOR_INTERVAL = 0.3
PREVIEW_CURSOR_COUNT = 2
ACTION_TYPES = [
    "SPAWN", "PREVIEW_CURSOR", "STORY", "TUTORIAL", "PLAY_BGM",
    "DISPLAY_ENEMY_INFO", "ACTIVATE_PREDEFINED", "PLAY_OPERA",
    "TRIGGER_PREDEFINED", "BATTLE_EVENTS", "WITHDRAW_PREDEFINED",
    "DIALOG", "SHOW_ALL_HIDDEN_CARDS", "EMPTY",
]
MULTI_ACTIONS = {"SPAWN", "PREVIEW_CURSOR"}


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, dict):
        value = value.get("m_value", default)
    if value is None:
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _js_round(value: float) -> int:
    return math.floor(value + 0.5)


def _milli_frames(seconds: Any) -> int:
    return _js_round(_number(seconds) * HZ * MILLI_FRAMES)


def _frames(seconds: Any) -> int:
    return _js_round(_number(seconds) * HZ)


def _to_frames(milli_frames: int) -> int:
    return max(0, math.floor((int(milli_frames) + MILLI_FRAMES / 2) / MILLI_FRAMES))


def _wait_frames(delta: int) -> int:
    return 0 if delta <= 0 else max(1, _to_frames(delta))


def _action_type(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("name", value.get("value", "SPAWN"))
    if isinstance(value, bool):
        return "EMPTY"
    if isinstance(value, (int, float)):
        index = int(value)
        return ACTION_TYPES[index] if 0 <= index < len(ACTION_TYPES) else "EMPTY"
    text = str(value or "").strip()
    if text.lstrip("+-").isdigit():
        index = int(text)
        return ACTION_TYPES[index] if 0 <= index < len(ACTION_TYPES) else "EMPTY"
    return text or "EMPTY"


def _mono_quicksort(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match Mono's in-place quicksort behavior for equal-time queue entries."""
    result = list(items)

    def compare(left: dict[str, Any], right: dict[str, Any]) -> int:
        a, b = left["time_mt"], right["time_mt"]
        return -1 if a < b else (1 if a > b else 0)

    def sort(low: int, high: int) -> None:
        while low < high:
            i, j = low, high
            pivot = result[(low + high) >> 1]
            while True:
                while compare(result[i], pivot) < 0:
                    i += 1
                while compare(pivot, result[j]) < 0:
                    j -= 1
                if i <= j:
                    result[i], result[j] = result[j], result[i]
                    i += 1
                    j -= 1
                if i > j:
                    break
            if low < j:
                sort(low, j)
            low = i

    if len(result) > 1:
        sort(0, len(result) - 1)
    return result


def _build_fragment_queue(
    fragment: dict[str, Any],
    *,
    enemy_delay_mt: dict[str, int] | None = None,
    enabled_hidden_groups: set[str] | None = None,
) -> list[dict[str, Any]]:
    enemy_delay_mt = enemy_delay_mt or {}
    items: list[dict[str, Any]] = []
    fragment_pre = _milli_frames(fragment.get("preDelay"))
    for action_index, action in enumerate(fragment.get("actions") or []):
        if not isinstance(action, dict):
            continue
        hidden_group = action.get("hiddenGroup")
        if hidden_group and (enabled_hidden_groups is None or
                             str(hidden_group) not in enabled_hidden_groups):
            continue
        kind = _action_type(action.get("actionType"))
        base = fragment_pre + _milli_frames(action.get("preDelay"))
        count = max(0, int(_number(action.get("count"), 0)))
        interval = _milli_frames(action.get("interval"))
        if kind == "PREVIEW_CURSOR":
            count = PREVIEW_CURSOR_COUNT
            interval = _milli_frames(PREVIEW_CURSOR_INTERVAL)
        key = str(action.get("key") or "")
        route = int(_number(action.get("routeIndex"), 0))
        if kind == "SPAWN":
            delay = max(0, int(enemy_delay_mt.get(key, 0)))
            base = max(base - delay, 0)
        if kind in MULTI_ACTIONS:
            for seq in range(count):
                items.append({
                    "time_mt": base + interval * seq,
                    "action": action_index,
                    "seq": seq,
                    "kind": kind,
                    "key": key,
                    "route": route,
                    "synthetic": False,
                    "hidden_group": hidden_group,
                    "no_frame": False,
                    "source_action": action,
                })
        else:
            items.append({
                "time_mt": base,
                "action": action_index,
                "seq": 0,
                "kind": kind,
                "key": key,
                "route": route,
                "synthetic": False,
                "hidden_group": hidden_group,
                "no_frame": kind == "EMPTY",
                "source_action": action,
            })
        if kind == "SPAWN" and action.get("autoPreviewRoute"):
            start = base - _milli_frames(PREVIEW_CURSOR_PRE_DELAY)
            step = _milli_frames(PREVIEW_CURSOR_INTERVAL)
            for seq in range(PREVIEW_CURSOR_COUNT):
                items.append({
                    "time_mt": start + step * seq,
                    "action": action_index,
                    "seq": count + seq,
                    "kind": "PREVIEW_CURSOR",
                    "key": key,
                    "route": route,
                    "synthetic": True,
                    "hidden_group": hidden_group,
                    "no_frame": False,
                    "source_action": action,
                })
        if kind == "SPAWN" and action.get("autoDisplayEnemyInfo"):
            items.append({
                "time_mt": max(base, 0),
                "action": action_index,
                "seq": count + PREVIEW_CURSOR_COUNT,
                "kind": "DISPLAY_ENEMY_INFO",
                "key": key,
                "route": route,
                "synthetic": True,
                "hidden_group": hidden_group,
                "no_frame": False,
                "source_action": action,
            })
    return _mono_quicksort(items)


def _drain_queue(
    items: list[dict[str, Any]],
    process_start: int,
    *,
    tail_overlap: bool,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    last_actual = process_start - 1
    previous_time = 0
    clock = process_start
    step_free_index = -1
    if tail_overlap:
        step_free_index = next(
            (index for index in range(len(items) - 1, -1, -1)
             if items[index]["synthetic"]), -1
        )
    for queue_index, item in enumerate(items):
        ideal = process_start + _to_frames(item["time_mt"])
        delta = item["time_mt"] - previous_time
        if queue_index == 0:
            clock = process_start + _wait_frames(item["time_mt"])
        elif delta > 0:
            clock += _wait_frames(delta)
        actual = clock
        if not item["no_frame"] and queue_index != step_free_index:
            clock += 1
        previous_time = item["time_mt"]
        last_actual = actual
        rows.append({
            "item": item,
            "queue_index": queue_index,
            "ideal_frame": ideal,
            "actual_frame": actual,
        })
    return rows, last_actual


def build_spawn_frame_schedule(
    waves: list[dict[str, Any]],
    *,
    enemy_delay_mt: dict[str, int] | None = None,
    enabled_hidden_groups: set[str] | None = None,
    wave_gates: dict[int, int] | None = None,
    wave_clear_frames: list[int | None] | None = None,
    fragment_timings: list[dict[str, Any]] | None = None,
    truncate_on_timeout: bool = True,
) -> list[dict[str, Any]]:
    """Return client-style action rows with absolute logic frames."""
    rows: list[dict[str, Any]] = []
    cursor = 0
    last_spawn_frame: int | None = None
    for wave_index, wave in enumerate(waves or []):
        timing_begin = len(fragment_timings) if fragment_timings is not None else 0
        if wave_index:
            candidates = [cursor]
            if last_spawn_frame is not None:
                candidates.append(last_spawn_frame + 1)
            if wave_clear_frames and wave_index - 1 < len(wave_clear_frames):
                clear_frame = wave_clear_frames[wave_index - 1]
                if clear_frame is not None:
                    candidates.append(int(clear_frame) + 1)
            default_gate = max(candidates)
            requested_gate = (wave_gates or {}).get(wave_index, default_gate)
            cursor = max(cursor, int(requested_gate))
        wave_start = cursor + _frames(wave.get("preDelay"))
        cursor = wave_start
        for fragment_index, fragment in enumerate(wave.get("fragments") or [{}]):
            process_start = cursor
            items = _build_fragment_queue(
                fragment,
                enemy_delay_mt=enemy_delay_mt,
                enabled_hidden_groups=enabled_hidden_groups,
            )
            drained, last_actual = _drain_queue(
                items,
                process_start,
                tail_overlap=(fragment_index == 0 and
                              _number(wave.get("preDelay")) > 0),
            )
            for row in drained:
                item = row["item"]
                event = {
                    **row,
                    "wave": wave_index,
                    "fragment": fragment_index,
                    "action": item["action"],
                    "seq": item["seq"],
                    "wave_start_frame": wave_start,
                    "actionType": item["kind"],
                    "key": item["key"],
                    "routeIndex": item["route"],
                    "hiddenGroup": item["hidden_group"],
                    "synthetic": item["synthetic"],
                    "managedByScheduler": True,
                }
                action = item["source_action"]
                event.update({
                    "managedByScheduler": action.get("managedByScheduler", True),
                    "dontBlockWave": bool(action.get("dontBlockWave")),
                    "blockFragment": bool(action.get("blockFragment")),
                    "forceBlockWaveInBranch": bool(
                        action.get("forceBlockWaveInBranch")),
                    "isUnharmfulAndAlwaysCountAsKilled": bool(
                        action.get("isUnharmfulAndAlwaysCountAsKilled")),
                    "notCountInTotal": bool(action.get("notCountInTotal")),
                    "groupKey": action.get("randomSpawnGroupKey"),
                    "packKey": action.get("randomSpawnGroupPackKey"),
                    "weight": action.get("weight"),
                })
                rows.append(event)
                if item["kind"] == "SPAWN":
                    last_spawn_frame = row["actual_frame"]
            completion = (last_actual + FRAGMENT_HANDOFF_FRAMES
                          if drained else process_start)
            if fragment_timings is not None:
                fragment_timings.append({"wave": wave_index, "fragment": fragment_index,
                                         "wave_start": wave_start, "completion": completion})
            max_wait = _number(wave.get("maxTimeWaitingForNextWave"), -1.0)
            if truncate_on_timeout and max_wait > 0 and completion - wave_start > _frames(max_wait):
                break
            cursor = completion
        cursor += _frames(wave.get("postDelay"))
        if fragment_timings is not None:
            for timing in fragment_timings[timing_begin:]:
                timing["last"] = timing is fragment_timings[-1]
                timing["wave_end"] = cursor
    return rows

"""Validated future-plan replacement on a reconstructed live battle."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .operations import OperationLog
from .assumptions import DEFAULT_ASSUMPTIONS


def runtime_signature(data_dir: Path, level_file: str) -> str:
    """Identify local runtime inputs without embedding paths or game tables."""
    files = {level_file, "character_table.json", "skill_table.json", "range_table.json", "stage_table.json",
             "enemy_database.json", "enemy_attack_timing.json", "operator_attack_timing.json",
             "projectile_data.json", "buff_template_data.json", "enemy_graphics.json", "operator_forms.json", "operator_graphics.json",
             "enemy_buff_abilities.json", "battle_equip_table.json", "uniequip_table.json", "favor_table.json"}
    digest = hashlib.sha256()
    for name in sorted(files):
        path = data_dir / name
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"<absent>")
        digest.update(b"\0")
    digest.update(json.dumps(asdict(DEFAULT_ASSUMPTIONS), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def frame_number(snapshot: dict) -> int:
    value = snapshot.get("fixedFrame")
    time = snapshot.get("t")
    if type(time) not in (int, float) or not math.isfinite(time) or time < 0:
        raise ValueError("回放帧的时间无效")
    if value is None:
        value = round(time * 30)
    if type(value) is not int or value < 0 or abs(value / 30 - time) > .001:
        raise ValueError("回放不是受支持的30 Hz逻辑帧")
    return value


def ordered(operations: list[dict]) -> list[dict]:
    return sorted(operations, key=lambda item: item["time"])


def validate_edit(original: dict, edited: dict, frame: int) -> None:
    if type(frame) is not int or frame < 0:
        raise ValueError("切入帧无效")
    for key in ("stage", "squad", "options"):
        if edited[key] != original[key]:
            raise ValueError("回放续编不能更换关卡、队伍练度或模拟选项")
    time = frame / 30
    prefix = lambda project: [op for op in ordered(project["operations"]) if op["time"] <= time + 1e-9]
    if prefix(original) != prefix(edited):
        raise ValueError("切入帧及之前的操作已锁定，只能修改之后的操作")


def replace_future_plan(battle: Any, plans: list, frame: int) -> None:
    """Keep live entities, timers, RNG and even an unfinished WAIT intact."""
    if battle.frame_index != frame or battle.end_reason:
        raise ValueError("切入点已经结束或尚未重建到对应逻辑帧")
    time = frame / 30
    prefix = [item for item in battle.plan if item.time <= time + 1e-9]
    plans = sorted(plans, key=lambda item: item.time)
    if prefix != [item for item in plans if item.time <= time + 1e-9]:
        raise ValueError("重建后的历史操作不一致")
    if battle.plan_index > len(prefix):
        raise ValueError("未来操作已经执行，不能替换")
    old_log = battle.operation_log
    # Historical feedback keeps its IDs; new feedback gets unused IDs.
    history_ids = [item["index"] for item in old_log.records[:len(prefix)]]
    start = max(history_ids, default=0) + 1
    log = OperationLog(plans, history_ids + list(range(start, start + len(plans) - len(prefix))))
    log.records[:len(prefix)] = deepcopy(old_log.records[:len(prefix)])
    log.events = deepcopy(old_log.events)
    battle.plan = prefix + plans[len(prefix):]
    battle.operation_log = log


def verify_prefix(source: dict, generated: dict, frame: int) -> None:
    expected = [item for item in source["frames"] if frame_number(item) <= frame]
    actual = {frame_number(item): item for item in generated["frames"]}
    for item in expected:
        number = frame_number(item)
        if actual.get(number) != item:
            raise ValueError(f"历史状态在f{number}不一致；原回放保留，请按当前输入重新模拟")
    if not expected or frame_number(expected[-1]) != frame:
        raise ValueError("找不到切入帧")
    boundary = frame / 30 + .0005
    events = lambda case: [item for item in case["events"] if item["t"] <= boundary]
    if events(source) != events(generated):
        raise ValueError("切入点之前的事件不一致；原回放保留")

"""Loading and indexing Arknights game data (JSON tables + level files)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .loadout import module_phase, validate_attributes


DATA_DIR: Path | None = None


def set_data_dir(path: str | Path) -> None:
    global DATA_DIR
    DATA_DIR = Path(path)


def _load(name: str) -> Any:
    assert DATA_DIR is not None, "call set_data_dir() first"
    with open(DATA_DIR / name, encoding="utf-8") as fh:
        return json.load(fh)


def load_characters() -> dict[str, Any]:
    return _load("character_table.json")


def load_skills() -> dict[str, Any]:
    return _load("skill_table.json")


def load_ranges() -> dict[str, Any]:
    return _load("range_table.json")


def load_stages() -> dict[str, Any]:
    return _load("stage_table.json")


def load_enemy_database() -> dict[str, Any]:
    return _load("enemy_database.json")


def load_attack_timing() -> dict[str, Any]:
    try:
        timing = _load("enemy_attack_timing.json")
    except FileNotFoundError:
        timing = {}
    try:
        timing.update(_load("operator_attack_timing.json"))
    except FileNotFoundError:
        pass
    return timing


def load_projectile_data() -> dict[str, Any]:
    """Load optional logical projectile speeds and unit mappings."""
    try:
        return _load("projectile_data.json")
    except FileNotFoundError:
        return {"speeds": {}, "enemies": {}, "operators": {}}


def load_behavior_templates() -> dict[str, Any]:
    """Load client behaviour trees when they were exported with the data set."""
    try:
        return _load("buff_template_data.json")
    except FileNotFoundError:
        return {}


def load_enemy_graphics() -> dict[str, Any]:
    """Load optional, preconverted map-space enemy attachment profiles."""
    try:
        return _load("enemy_graphics.json")
    except FileNotFoundError:
        return {}


def load_enemy_buff_abilities() -> dict[str, Any]:
    """Load the normalized enemy Buff definitions used by the simulator."""
    try:
        return _load("enemy_buff_abilities.json")
    except FileNotFoundError:
        return {}


def load_battle_equips() -> dict[str, Any]:
    try:
        return _load("battle_equip_table.json")
    except FileNotFoundError:
        return {}


def load_module_index() -> dict[str, Any]:
    try:
        return _load("uniequip_table.json")
    except FileNotFoundError:
        return {}


def load_favor_table() -> dict[str, Any]:
    try:
        return _load("favor_table.json")
    except FileNotFoundError:
        return {}


def load_level(filename: str) -> dict[str, Any]:
    return _load(filename)


def build_enemy_index(enemy_db: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return enemyId -> sorted list of {level, enemyData} variants."""
    index: dict[str, list[dict[str, Any]]] = {}
    for entry in enemy_db.get("enemies", []):
        variants = sorted(entry.get("Value", []), key=lambda v: v.get("level", 0))
        index[entry["Key"]] = variants
    return index


def enemy_data_at(
    index: dict[str, list[dict[str, Any]]], enemy_id: str, level: int = 0
) -> dict[str, Any]:
    variants = index.get(enemy_id)
    if not variants:
        raise KeyError(f"unknown enemy id: {enemy_id}")
    exact = [v for v in variants if v.get("level") == level]
    variant = exact[0] if exact else min(
        variants, key=lambda v: abs(v.get("level", 0) - level)
    )
    return variant["enemyData"]


def mdef(value: Any, default: Any) -> Any:
    """Decode an Arknights `{m_defined, m_value}` wrapper or plain value."""
    if isinstance(value, dict):
        if value.get("m_defined"):
            return value.get("m_value", default)
        return default
    return default if value is None else value


def _interpolate_attributes(
    frames: list[dict[str, Any]], level: float, maximum: float
) -> dict[str, Any]:
    if not frames:
        return {}
    ordered = sorted(frames, key=lambda frame: frame.get("level", 0))
    lo_level = float(ordered[0].get("level", 0))
    hi_level = max(float(maximum), float(ordered[-1].get("level", 0)))
    level = max(lo_level, min(float(level), hi_level))
    lower = ordered[0]
    upper = ordered[-1]
    for frame in ordered:
        frame_level = float(frame.get("level", 0))
        if frame_level <= level:
            lower = frame
        if frame_level >= level:
            upper = frame
            break
    keys = set(lower.get("data", {})) | set(upper.get("data", {}))
    out: dict[str, Any] = {}
    for key in keys:
        lower_value = lower.get("data", {}).get(key)
        upper_value = upper.get("data", {}).get(key)
        if (
            isinstance(lower_value, (int, float))
            and not isinstance(lower_value, bool)
            and isinstance(upper_value, (int, float))
            and not isinstance(upper_value, bool)
        ):
            lower_level = float(lower.get("level", 0))
            upper_level = float(upper.get("level", 0))
            if upper_level == lower_level:
                value = lower_value
            else:
                ratio = (level - lower_level) / (upper_level - lower_level)
                value = lower_value + (upper_value - lower_value) * ratio
            out[key] = value
        else:
            out[key] = upper_value if upper_value is not None else lower_value
    return out


_ATTRIBUTE_KEY = {
    "MAX_HP": "maxHp",
    "ATK": "atk",
    "DEF": "def",
    "MAGIC_RESISTANCE": "magicResistance",
    "COST": "cost",
    "BLOCK_CNT": "blockCnt",
    "ATTACK_SPEED": "attackSpeed",
    "RESPAWN_TIME": "respawnTime",
}


def char_attributes(
    char: dict[str, Any],
    level: int,
    elite: int | None = None,
    *,
    trust: float = 0.0,
    potential_rank: int = 0,
    module: dict[str, Any] | None = None,
    module_level: int = 0,
) -> dict[str, Any]:
    """Attributes for a character at `level` in the given elite phase.

    Follows the game's level-to-stat model: each elite phase has a
    `maxLevel` cap and a set of attribute keyframes; stats are linearly
    interpolated between keyframes after validating the phase's level range.
    """
    elite = validate_attributes(char, level, elite, trust, potential_rank)
    phase = char["phases"][elite]

    out = _interpolate_attributes(
        phase.get("attributesKeyFrames", []), level, phase.get("maxLevel", 1)
    )

    # Trust stats reach their cap at 100% even though trust itself can reach 200%.
    trust_level = min(max(float(trust), 0.0), 100.0) * 0.5
    favor_frames = char.get("favorKeyFrames", [])
    favor = _interpolate_attributes(favor_frames, trust_level, 50.0)
    for key, value in favor.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = float(out.get(key, 0.0)) + float(value)

    ranks = char.get("potentialRanks", [])
    for rank in ranks[: max(0, min(int(potential_rank), len(ranks)))]:
        modifiers = (
            ((rank.get("buff") or {}).get("attributes") or {}).get(
                "attributeModifiers"
            )
            or []
        )
        for modifier in modifiers:
            key = _ATTRIBUTE_KEY.get(str(modifier.get("attributeType", "")))
            if key is None:
                continue
            value = float(modifier.get("value", 0.0) or 0.0)
            formula = str(modifier.get("formulaItem", "ADDITION"))
            if formula == "ADDITION":
                out[key] = float(out.get(key, 0.0)) + value
            elif formula == "MULTIPLIER":
                out[key] = float(out.get(key, 0.0)) * (1.0 + value)

    if module is None and module_level != 0:
        raise ValueError("module_level requires module data")
    if module is not None:
        chosen = module_phase(module, module_level)
        for item in (chosen or {}).get("attributeBlackboard", []):
            key = _ATTRIBUTE_KEY.get(str(item.get("key", "")).upper())
            if key is not None:
                out[key] = float(out.get(key, 0.0)) + float(
                    item.get("value", 0.0) or 0.0
                )

    for key in (
        "blockCnt",
        "cost",
        "respawnTime",
        "maxDeployCount",
        "maxDeckStackCnt",
        "baseForceLevel",
    ):
        if key in out:
            out[key] = round(out[key])
    return out

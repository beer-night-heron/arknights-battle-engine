"""Select difficulty and optional stage runes without changing source data."""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any


def select_stage_runes(stages: dict[str, Any], stage: str, level: dict[str, Any],
                       optional_runes: list[str] | None = None, difficulty: str | None = None):
    meta = stages["stages"][stage]
    groups = [meta.get("advancedRuneIdList1") or [], meta.get("advancedRuneIdList2") or []]
    valid_ids = {key for group in groups for key in group}
    keys = []
    for key in optional_runes or []:
        if key in stages.get("sixStarRuneData", {}):
            if key not in valid_ids:
                raise ValueError(f"optional rune {key} does not belong to {stage}")
            key = stages["sixStarRuneData"][key]["runeKey"]
        keys.append(key)
    for group in groups:
        group_keys = {stages["sixStarRuneData"][key]["runeKey"] for key in group}
        if sum(key in group_keys for key in keys) > 1:
            raise ValueError("choose at most one optional rune per difficulty group")
    selected = difficulty or (meta["difficulty"] if meta.get("difficulty") in
                              {"NORMAL", "FOUR_STAR", "SIX_STAR"} else "NORMAL")
    return select_runes(level, selected, keys)


def select_runes(level: dict[str, Any], difficulty: str = "NORMAL",
                 optional_keys: list[str] | None = None) -> dict[str, Any]:
    difficulty = difficulty.upper()
    if difficulty not in {"NORMAL", "FOUR_STAR", "SIX_STAR"}:
        raise ValueError("difficulty must be NORMAL, FOUR_STAR or SIX_STAR")
    result = deepcopy(level)
    active = [r for r in (result.get("runes") or [])
              if r.get("difficultyMask", "ALL") in ("ALL", difficulty)]
    optional = result.get("optionalRunes") or {}
    keys = optional_keys or []
    if len(keys) != len(set(keys)):
        raise ValueError("optional rune keys must be unique")
    for key in keys:
        if key not in optional:
            raise ValueError(f"unknown optional rune key: {key}")
        selected = optional[key]
        if any(r.get("difficultyMask", "ALL") not in ("ALL", difficulty) for r in selected):
            raise ValueError(f"optional rune {key} is unavailable in {difficulty}")
        active.extend(selected)
    # Additions modify an explicitly named rune, rather than adding another
    # multiplicative attribute layer per option.
    for rune in active:
        if rune.get("key") != "add_other_rune_blackb":
            continue
        board = blackboard(rune)
        alias = board.get("rune_alias")
        targets = [r for r in active if r.get("key") != "add_other_rune_blackb"
                   and blackboard(r).get("rune_alias") == alias]
        if len(targets) != 1:
            raise ValueError(f"rune alias must identify one target: {alias}")
        for entry in targets[0].get("blackboard", []):
            name = entry["key"]
            if name in board and isinstance(board[name], (int, float)):
                entry["value"] += board[name]
    result["runes"] = [r for r in active if r.get("key") != "add_other_rune_blackb"]
    result["selectedDifficulty"] = difficulty
    result["selectedOptionalRunes"] = list(keys)
    for rune in active:
        if rune.get("key") == "global_lifepoint":
            result.setdefault("options", {})["maxLifePoint"] = int(blackboard(rune)["value"])
    hidden = {blackboard(r).get("key") for r in active if r.get("key") == "level_hidden_group_enable"}
    for wave in result.get("waves", []):
        for fragment in wave.get("fragments", []):
            fragment["actions"] = [a for a in fragment.get("actions", [])
                                   if not a.get("hiddenGroup") or a["hiddenGroup"] in hidden]
    return result


def blackboard(rune: dict[str, Any]) -> dict[str, Any]:
    return {entry["key"]: entry.get("valueStr") or entry.get("value", 0)
            for entry in rune.get("blackboard", [])}


def forbidden_locations(level: dict[str, Any]) -> set[tuple[int, int]]:
    cells = {(int(p["row"]), int(p["col"])) for p in level.get("tilesDisallowToLocate") or []}
    for rune in level.get("runes", []):
        if rune.get("key") == "global_forbid_location":
            value = str(blackboard(rune).get("location", ""))
            cells.update((int(r), int(c)) for r, c in re.findall(r"\((-?\d+),\s*(-?\d+)\)", value))
    return cells

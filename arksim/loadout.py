"""Resolve legal operator builds and select unlocked ability data.

Selection does not execute arbitrary talents or module effects. The battle
engine must explicitly support each effect and diagnose the remaining ones.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .battle import OperatorPlan


def phase_number(value: Any) -> int:
    if isinstance(value, str) and value.startswith("PHASE_"):
        value = int(value[6:])
    if type(value) is not int or value < 0:
        raise ValueError(f"invalid phase {value!r}")
    return value


def condition_met(condition: dict[str, Any] | None, elite: int, level: int) -> bool:
    condition = condition or {}
    phase = phase_number(condition.get("phase", 0))
    return (elite, level) >= (phase, int(condition.get("level", 1)))


def validate_attributes(
    char: dict[str, Any], level: int, elite: int | None,
    trust: float, potential_rank: int,
) -> int:
    phases = char.get("phases", [])
    elite = len(phases) - 1 if elite is None else elite
    if type(elite) is not int or not 0 <= elite < len(phases):
        raise ValueError(f"elite must identify an existing phase (0..{len(phases)-1})")
    maximum = int(phases[elite].get("maxLevel", 1))
    if type(level) is not int or not 1 <= level <= maximum:
        raise ValueError(f"level must be 1..{maximum} at elite {elite}")
    if type(trust) not in (int, float) or not math.isfinite(trust) or not 0 <= trust <= 200:
        raise ValueError("trust must be a finite percentage in 0..200")
    maximum_potential = int(char.get("maxPotentialLevel", len(char.get("potentialRanks", []))))
    if type(potential_rank) is not int or not 0 <= potential_rank <= maximum_potential:
        raise ValueError(f"potential_rank must be 0..{maximum_potential} (0 means first potential)")
    return elite


def module_phase(module: dict[str, Any], level: int) -> dict[str, Any]:
    if type(level) is not int or level <= 0:
        raise ValueError("module_level must be positive when module_id is set")
    phase = next((p for p in module.get("phases", []) if p.get("equipLevel") == level), None)
    if phase is None:
        raise ValueError(f"module_level {level} is absent from module data")
    return phase


def _skill_rank_unlocked(char: dict[str, Any], entry: dict[str, Any], rank: int,
                         elite: int, level: int) -> bool:
    normal = char.get("allSkillLvlup", [])
    normal_count = len(normal) + 1
    costs = normal[:min(rank - 1, len(normal))]
    if rank > normal_count:
        mastery = entry.get("levelUpCostCond", [])
        count = rank - normal_count
        if count > len(mastery):
            return False
        costs = costs + mastery[:count]
    return all(condition_met(c.get("unlockCond"), elite, level) for c in costs)


def resolve_plan(plan: "OperatorPlan", characters: dict, skills: dict,
                 modules: dict, module_index: dict, favor_table: dict) -> "OperatorPlan":
    char = characters.get(plan.char_id)
    if char is None:
        raise ValueError(f"unknown char_id {plan.char_id!r}")
    if type(plan.auto_skill) is not bool:
        raise ValueError("auto_skill must be a boolean")
    if plan.on_failure not in ("WAIT", "SKIP", "STOP"):
        raise ValueError("on_failure must be WAIT, SKIP or STOP")
    if plan.mode is not None and (plan.action != "SWITCH_MODE" or type(plan.mode) is not int or plan.mode not in (0, 1)):
        raise ValueError("mode must be 0 or 1, only for SWITCH_MODE")
    if plan.action in ("RETREAT", "SKILL", "SKILL_END", "SWITCH_MODE"):
        if plan.tile is not None:
            raise ValueError(f"{plan.action} must omit tile or set it to null")
        return plan
    if plan.action != "DEPLOY":
        raise ValueError(f"unsupported action {plan.action!r}")
    for key in ("skill_id", "module_id"):
        value = getattr(plan, key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{key} must be a nonempty string or null")
    if type(plan.module_level) is not int or plan.module_level < 0:
        raise ValueError("module_level must be a nonnegative integer")
    elite = validate_attributes(char, plan.level, plan.elite, plan.trust, plan.potential_rank)
    rank = plan.skill_level
    if plan.skill_id:
        entry = next((s for s in char.get("skills", []) if s.get("skillId") == plan.skill_id), None)
        if entry is None:
            raise ValueError(f"skill_id {plan.skill_id!r} does not belong to {plan.char_id}")
        if not condition_met(entry.get("unlockCond"), elite, plan.level):
            raise ValueError(f"skill_id {plan.skill_id!r} is not unlocked at this elite/level")
        levels = (skills.get(plan.skill_id) or {}).get("levels", [])
        if not levels:
            raise ValueError(f"skill_id {plan.skill_id!r} has no level data")
        legal = [i for i in range(1, len(levels) + 1)
                 if _skill_rank_unlocked(char, entry, i, elite, plan.level)]
        rank = max(legal) if rank is None else rank
        if type(rank) is not int or rank not in legal:
            raise ValueError(f"skill_level must be an unlocked rank for this skill; allowed: {legal}")
    elif rank is not None:
        raise ValueError("skill_level requires skill_id")
    if plan.module_id:
        metadata = module_index.get("equipDict", {}).get(plan.module_id)
        if not metadata:
            raise ValueError(f"module_id {plan.module_id!r} has no uniequip_table metadata")
        if metadata.get("charId") != plan.char_id:
            raise ValueError(f"module_id {plan.module_id!r} does not belong to {plan.char_id}")
        module = modules.get(plan.module_id)
        if module is None:
            raise ValueError(f"module_id {plan.module_id!r} has no battle_equip_table data")
        module_phase(module, plan.module_level)
        condition = {"phase": metadata.get("unlockEvolvePhase", 0),
                     "level": metadata.get("unlockLevel", 1)}
        if not condition_met(condition, elite, plan.level):
            raise ValueError(f"module_id {plan.module_id!r} is not unlocked at this elite/level")
        raw_favor = (metadata.get("unlockFavors") or {}).get(str(plan.module_level), 0)
        if raw_favor:
            points = [f["data"]["percent"] for f in favor_table.get("favorFrames", [])
                      if f["data"]["favorPoint"] >= raw_favor]
            if not points:
                raise ValueError("favor_table data is required to verify module trust condition")
            minimum = min(points)
            if plan.trust < minimum:
                raise ValueError(f"module level {plan.module_level} requires trust >= {minimum}%")
    elif plan.module_level != 0:
        raise ValueError("module_level requires module_id")
    return replace(plan, elite=elite, skill_level=rank)


def select_candidate(candidates: list[dict] | None, elite: int, level: int,
                     potential_rank: int) -> dict | None:
    eligible = [c for c in candidates or []
                if condition_met(c.get("unlockCondition"), elite, level)
                and int(c.get("requiredPotentialRank", 0)) <= potential_rank]
    if not eligible:
        return None
    return max(eligible, key=lambda c: (
        phase_number((c.get("unlockCondition") or {}).get("phase", 0)),
        int((c.get("unlockCondition") or {}).get("level", 1)),
        int(c.get("requiredPotentialRank", 0)),
    ))


def _merge_candidate(base: dict | None, override: dict) -> dict:
    merged = dict(base or {})
    merged.update(override)
    board = {b["key"]: b for b in (base or {}).get("blackboard") or []}
    board.update({b["key"]: b for b in override.get("blackboard") or []})
    merged["blackboard"] = list(board.values())
    return merged


def select_abilities(char: dict, elite: int, level: int, potential_rank: int,
                     module: dict | None) -> tuple[dict | None, dict[int, dict], list[str]]:
    trait = select_candidate((char.get("trait") or {}).get("candidates"), elite, level, potential_rank)
    talents = {}
    for index, bundle in enumerate(char.get("talents") or []):
        candidate = select_candidate(bundle.get("candidates"), elite, level, potential_rank)
        if candidate is not None:
            talents[index] = candidate
    deferred = []
    for index, part in enumerate((module or {}).get("parts", [])):
        if (part.get("isToken") or part.get("validInGameTag") is not None
                or part.get("validInMapTag") is not None):
            deferred.append(f"part:{index}:conditional_or_token")
            continue
        override = select_candidate((part.get("overrideTraitDataBundle") or {}).get("candidates"),
                                    elite, level, potential_rank)
        if override is not None:
            trait = _merge_candidate(trait, override)
        groups: dict[int, list[dict]] = {}
        for candidate in (part.get("addOrOverrideTalentDataBundle") or {}).get("candidates") or []:
            if candidate.get("validModeIndices") is not None:
                deferred.append(f"part:{index}:mode_condition")
                continue
            if "talentIndex" not in candidate:
                deferred.append(f"part:{index}:missing_talent_index")
                continue
            groups.setdefault(int(candidate["talentIndex"]), []).append(candidate)
        for talent_index, candidates in groups.items():
            override = select_candidate(candidates, elite, level, potential_rank)
            if override is not None:
                talents[talent_index] = _merge_candidate(talents.get(talent_index), override)
    return trait, talents, deferred

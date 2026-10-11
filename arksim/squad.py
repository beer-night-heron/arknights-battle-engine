"""Selected carried talents and field auras for the main battle engine.

Parameters come from the resolved talent/module candidates. Activation rules
are explicit; this does not interpret arbitrary descriptions or summon effects.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .loadout import module_phase, resolve_plan, select_abilities

BUILD_FIELDS = frozenset(("char_id", "elite", "level", "trust", "potential_rank",
                          "skill_id", "skill_level", "module_id", "module_level", "auto_skill"))
TALENT_INDICES = {
    "char_172_svrash": 0,
    "char_180_amgoat": 0, "char_340_shwaz": 1, "char_102_texas": 0,
    "char_4228_closur": 1, "char_197_poca": 1, "char_2014_nian": 0,
    "char_2025_shu": 1, "char_263_skadi": 0, "char_1023_ghost2": 1,
    "char_218_cuttle": 0, "char_249_mlyss": 1, "char_298_susuro": 0,
    "char_4141_marcil": 1,
    "char_1038_whitw2": 1,
    "char_1051_headb2": 1,
}


def resolve_squad(plan: list[Any], squad: list[Any] | None, characters: dict,
                  skills: dict, modules: dict, module_index: dict,
                  favor_table: dict) -> tuple[list[Any], set[str]]:
    from .operator_skills import TOKEN_SKILLS, token_builds
    carried, ambiguous = {}, set()
    if squad is None:
        for item in sorted(plan, key=lambda item: item.time):
            if item.action != "DEPLOY" or item.char_id in TOKEN_SKILLS:
                continue
            previous = carried.setdefault(item.char_id, item)
            if any(getattr(previous, key) != getattr(item, key) for key in BUILD_FIELDS):
                ambiguous.add(item.char_id)
    for index, item in enumerate(squad or [], 1):
        if item.action != "DEPLOY" or not item.char_id.startswith("char_"):
            raise ValueError(f"squad item {index}: must identify an operator build")
        if item.char_id in carried:
            raise ValueError(f"squad item {index}: duplicate operator {item.char_id}")
        try:
            carried[item.char_id] = resolve_plan(item, characters, skills, modules, module_index, favor_table)
        except ValueError as error:
            raise ValueError(f"squad item {index}: {error}") from error
    tokens = token_builds([vars(item) for item in carried.values()], characters)
    for item in plan:
        if item.char_id in TOKEN_SKILLS:
            expected = tokens.get(item.char_id)
            if expected is None:
                raise ValueError(f"token {item.char_id} has no carried owner with its required skill")
            if item.action == "DEPLOY" and any(getattr(item, key) != expected[key] for key in BUILD_FIELDS):
                raise ValueError(f"token build for {item.char_id} must inherit the owner's phase/level")
            continue
        if item.char_id not in carried:
            raise ValueError(f"plan operator {item.char_id} is not carried in squad")
        if squad is not None and item.action == "DEPLOY" and any(
            getattr(item, key) != getattr(carried[item.char_id], key) for key in BUILD_FIELDS
        ):
            raise ValueError(f"plan build for {item.char_id} differs from squad")
    return list(carried.values()), ambiguous


def board(candidate: dict | None) -> dict[str, float]:
    return {item["key"]: float(item["value"]) for item in (candidate or {}).get("blackboard") or []
            if item.get("value") is not None}


def skill_in_effect(op: Any) -> bool:
    """A permanent skill remains in effect after its charging lifecycle ends."""
    return op.skill_active or getattr(op, "skill_end_reason", None) == "PERMANENT"


@dataclass(frozen=True)
class SquadBonus:
    atk: float = 0.0
    defense: float = 0.0
    max_hp: float = 0.0
    attack_speed: float = 0.0
    heal: float = 0.0
    sp_recovery: float = 0.0


class SquadEffects:
    def __init__(self, builds: list[Any], characters: dict, modules: dict):
        self.characters = characters
        self.builds = {item.char_id: item for item in builds}
        self.boards: dict[str, dict[str, float]] = {}
        self.traits: dict[str, dict[str, float]] = {}
        self.extra: dict[str, dict[str, float]] = {}
        self.shu_extra_index: int | None = None
        for cid, item in self.builds.items():
            if cid not in TALENT_INDICES:
                continue
            phase = module_phase(modules[item.module_id], item.module_level) if item.module_id else None
            trait, talents, _ = select_abilities(characters[cid], item.elite, item.level,
                                                 item.potential_rank, phase)
            candidate = talents.get(TALENT_INDICES[cid])
            if candidate is None:
                continue
            self.boards[cid] = board(candidate)
            self.traits[cid] = board(trait)
            self.extra[cid] = board(talents.get(-1))
            if cid == "char_2025_shu" and item.module_id == "uniequip_003_shu":
                for index, auxiliary in talents.items():
                    if auxiliary.get("prefabKey") == "20_root":
                        self.extra[cid] = board(auxiliary)
                        self.shu_extra_index = index
                        break
        self.initial_cost = self.boards.get("char_102_texas", {}).get("cost", 0.0)
        self.cost_lower_bound = self.boards.get("char_4228_closur", {}).get("cost", 0.0)
        self.first_rhine_used = False
        self.skill_started: set[str] = set()
        self.respawn_modifier = self.boards.get("char_172_svrash", {}).get("respawn_time", 0)
        shu = self.boards.get("char_2025_shu", {})
        self.sui_active = bool(shu) and self.count("groupId", "sui") >= 4
        self.sp_interval = shu.get("interval", 0.0) if self.sui_active else 0.0
        self.sp_amount = shu.get("sp", 0.0) if self.sui_active else 0.0

    def count(self, key: str, value: str) -> int:
        return sum(self.characters[cid].get(key) == value for cid in self.builds)

    def upgraded(self, cid: str, mid: str) -> bool:
        item = self.builds[cid]
        return item.module_id == mid and item.module_level >= 2

    def handled_talent(self, cid: str, index: int) -> bool:
        return cid in self.boards and (index == TALENT_INDICES[cid]
            or (cid == "char_197_poca" and index == -1
                and self.upgraded(cid, "uniequip_003_poca"))
            or (cid == "char_2025_shu" and index == self.shu_extra_index
                and self.upgraded(cid, "uniequip_003_shu")))

    def partial_talent(self, cid: str) -> bool:
        # The team portion is supported, but these candidates also describe
        # mana, skill damage, or copying a summon, which are separate effects.
        return cid == "char_4141_marcil" or (
            cid == "char_102_texas" and self.upgraded(cid, "uniequip_002_texas")) or (
            cid == "char_249_mlyss" and self.upgraded(cid, "uniequip_003_mlyss"))

    def cost_modifier(self, cid: str) -> float:
        data = self.boards.get("char_249_mlyss", {})
        if self.characters[cid].get("groupId") != "rhine":
            return 0.0
        return data.get("cost", 0.0) + (0.0 if self.first_rhine_used else data.get("runtime_cost", 0.0))

    def deployed(self, cid: str) -> None:
        if "char_249_mlyss" in self.boards and self.characters[cid].get("groupId") == "rhine":
            self.first_rhine_used = True

    def initial_sp_bonus(self, cid: str) -> float:
        if self.characters[cid].get("nationId") == "siracusa":
            return self.boards.get("char_1038_whitw2", {}).get("sp", 0)
        return 0.0

    def skill_started_by(self, cid: str) -> None:
        self.skill_started.add(cid)

    def bonus(self, cid: str, initial_cost: float, field: list[Any]) -> SquadBonus:
        if not cid.startswith("char_"):
            return SquadBonus()
        field = [op for op in field if op.char_id.startswith("char_")]
        char = self.characters[cid]
        profession = char.get("profession")
        professions = Counter(self.characters[op.char_id].get("profession") for op in field)
        field_ids = {op.char_id for op in field}
        atk = hp = defense = speed = heal = recovery = 0.0
        for source, data in self.boards.items():
            if source == "char_172_svrash" and cid == source:
                atk += data.get("atk", 0.0)
            elif source == "char_180_amgoat" and profession == "CASTER":
                if self.upgraded(source, "uniequip_002_amgoat") or source in field_ids:
                    atk += data.get("atk", 0.0)
            elif source == "char_340_shwaz" and profession == "SNIPER":
                carried = self.upgraded(source, "uniequip_002_shwaz")
                enabled = self.count("profession", "SNIPER") >= 2 if carried else (
                    source in field_ids and professions["SNIPER"] >= 2)
                if enabled:
                    atk += data.get("atk", 0.0)
            elif source == "char_4228_closur" and char.get("nationId") == "rhodes":
                atk += data.get("atk", 0.0)
            elif source == "char_197_poca" and char.get("teamId") == "student":
                atk += data.get("atk", 0.0)
                extra = self.extra[source]
                active = sum(skill_in_effect(op) and self.characters[op.char_id].get("teamId") == "student"
                             for op in field)
                atk += min(extra.get("max_atk", 0.0), extra.get("init_atk", 0.0) * active)
            elif source == "char_1051_headb2":
                owner = next((op for op in field if op.char_id == source), None)
                receiver = next((op for op in field if op.char_id == cid), None)
                # S2 already stores its self share in the cast blackboard.
                self_in_cast = cid == source and owner is not None and owner.skill_prefab_id == "skchr_headb2_2"
                if (owner is not None and skill_in_effect(owner) and not self_in_cast
                        and receiver is not None and receiver.statuses.targetable):
                    scale = data.get("scale_bonus", 1.0) if char.get("teamId") == "student" else 1.0
                    atk += data.get("atk", 0.0) * scale
                    defense += data.get("def", 0.0) * scale
            elif source == "char_2014_nian":
                if profession == "TANK":
                    hp += data.get("max_hp", 0.0)
                if cid == source and self.upgraded(source, "uniequip_003_nian"):
                    trait = self.traits[source]
                    count = min(professions["TANK"], trait.get("max_stack_cnt", 0.0))
                    hp += trait.get("max_hp", 0.0) * count
                    heal += trait.get("heal_scale_addition", 0.0) * count
            elif source == "char_2025_shu":
                diverse = source in field_ids and len(professions) >= 3
                same = source in field_ids and max(professions.values(), default=0) >= 3
                if diverse:
                    hp += data.get("max_hp", 0.0)
                if same:
                    speed += data.get("attack_speed", 0.0)
                if self.sui_active:
                    atk += data.get("atk", 0.0)
                if sum((diverse, same, self.sui_active)) >= 2:
                    atk += self.extra[source].get("atk", 0.0)
            elif source in ("char_263_skadi", "char_1023_ghost2", "char_218_cuttle"):
                if char.get("groupId") == "abyssal":
                    atk += data.get("atk", 0.0)
                    hp += data.get("max_hp", 0.0)
                    speed += data.get("attack_speed", 0.0)
                    recovery = max(recovery, data.get("sp_recovery_per_sec", 0.0))
            elif source == "char_298_susuro" and initial_cost <= data.get("cost", -1.0):
                heal += data.get("heal_scale", 1.0) - 1.0
            elif source == "char_4141_marcil" and char.get("teamId") == "laios":
                if self.count("teamId", "laios") >= 4:
                    defense += data.get("def", 0.0)
                    speed += data.get("attack_speed", 0.0)
            elif source == "char_1038_whitw2" and char.get("nationId") == "siracusa" and cid in self.skill_started:
                speed += data.get("attack_speed", 0.0)
        return SquadBonus(atk, defense, hp, speed, heal, recovery)

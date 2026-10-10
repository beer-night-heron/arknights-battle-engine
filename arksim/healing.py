"""Shu's charged healing and sown-tile effects in the main battle engine.

Data establishes the effects; animation handoffs and tile listener phases are
still candidates. This runtime does not implement Shu's second/third skills.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from .mechanics import DamageRule, DamageRuleKind, DamageType, Status

SHU = "char_2025_shu"
SHU_S1 = "skchr_shu_1"


@dataclass
class ShuHealing:
    battle: Any
    owner: Any
    cells: set[tuple[int, int]] = field(default_factory=set)
    recipients: dict[int, tuple[Any, DamageRule]] = field(default_factory=dict)
    finish_frame: int | None = None
    spent_sp: float = 0.0

    @property
    def talent(self) -> dict[str, Any]:
        return self.owner.talent_blackboards.get("1", {})

    def born(self) -> None:
        range_id = self.talent.get("born_range_id")
        if range_id:
            self.sow(self.owner, str(range_id))

    def own_damage_resistance(self, kind: DamageType) -> float:
        # GUA-Y uses the nonstackable IncludePure resistance template.
        if (self.owner.module_id == "uniequip_003_shu"
                and kind in {DamageType.PHYSICAL, DamageType.ARTS, DamageType.TRUE, DamageType.ELEMENTAL}):
            return float(self.owner.trait_blackboard.get("damage_resistance", 0))
        return 0.0

    def sow(self, target: Any, range_id: str = "x-5") -> None:
        if not self.talent or self.owner.dead:
            return
        candidates = self.battle.range_cells_from_id(range_id, target.tile, 0)
        for cell in candidates:
            tile = self.battle.tile(*cell)
            # The tile selector accepts buildable tiles OR ground-passable
            # tiles; it does not exclude a buildable highland automatically.
            if tile and (tile.get("buildableType") in ("MELEE", "RANGED", "ALL")
                         or self.battle._ground_passable_cell(*cell)):
                self.cells.add(cell)
        self.refresh()

    def refresh(self) -> None:
        units = list(self.battle.operators)
        if not any(unit is self.owner for unit in units):
            units.append(self.owner)
        eligible = {unit.deployment_index: unit for unit in units
                    if not unit.dead and unit.tile in self.cells
                    and not unit.statuses.has(Status.HEAL_FREE)
                    and not unit.statuses.has(Status.UNTARGETABLE)}
        for identity, (unit, rule) in list(self.recipients.items()):
            if identity not in eligible:
                unit.damage_rules[:] = [item for item in unit.damage_rules if item is not rule]
                del self.recipients[identity]
        resistance = float(self.talent.get("damage_resistance", 0))
        for identity, unit in eligible.items():
            if identity in self.recipients:
                rule = self.recipients[identity][1]
                # Another lifecycle (e.g. a form reset) may have cleared rules.
                if not any(item is rule for item in unit.damage_rules):
                    unit.damage_rules.append(rule)
            else:
                rule = DamageRule(DamageRuleKind.RESISTANCE,
                                  damage_types=frozenset({DamageType.PHYSICAL, DamageType.ARTS}))
                unit.damage_rules.append(rule)
                self.recipients[identity] = (unit, rule)
            rule.value = max(0.0, resistance)

    def tick_tiles(self) -> None:
        if self.owner.dead:
            self.depart()
            return
        self.refresh()

    def passive_hp_recovery_for(self, target: Any) -> float:
        entry = self.recipients.get(target.deployment_index)
        if self.owner.dead or target.dead or entry is None or entry[0] is not target:
            return 0.0
        return max(0.0, float(self.talent.get("hp_recovery_per_sec", 0)))

    def start_s1(self) -> bool:
        op, battle = self.owner, self.battle
        if (op.pending_attack is not None or op.attack_cd > 0
                or not op.statuses.can_attack
                or battle.frame_index < (op.selector_first_search_frame or 0)):
            return False
        timing = op.skill_attack_timing.get(SHU_S1, {})
        cells = battle.range_cells_from_id(str(op.skill.get("rangeId") or "x-4"), op.tile, op.facing)
        targets = battle._operator_heal_targets(
            op, cells=cells, hp_ratio=float(timing.get("target_hp_ratio", .5)),
            include_equal=bool(timing.get("target_hp_inclusive", True)),
        )
        if not targets:
            return False
        self.spent_sp = battle._skill_cost(op)
        op.sp = max(0.0, op.sp - self.spent_sp)
        battle._register_ability_use(op)
        op.skill_cast_count += 1
        op.skill_active = True
        op.skill_end_reason = None
        interval = battle._effective_attack_interval(op)
        op.attack_cd = interval
        duration = battle._operator_action_duration(op)
        self.finish_frame = battle.frame_index + max(1, math.ceil(duration / battle.dt - 1e-9))
        battle._begin_operator_heal(op, targets[:1], float(op.skill_blackboard.get("heal_scale", 1)))
        if op.pending_attack is not None:
            self.finish_frame = max(self.finish_frame, op.pending_attack.execute_frame)
        battle._sync_squad_effects()
        return True

    def refund_dead_target(self) -> None:
        if self.owner.skill_prefab_id == SHU_S1 and self.finish_frame is not None:
            self.battle._store_sp_gain(self.owner, self.spent_sp)
            self.spent_sp = 0.0

    def tick_skill(self) -> None:
        if (self.finish_frame is not None and self.battle.frame_index >= self.finish_frame
                and self.owner.pending_attack is None):
            self.finish_frame = None
            self.spent_sp = 0.0
            if abs(self.owner.attack_cd) < 1e-9:
                self.owner.attack_cd = 0.0
            self.battle._finish_skill(self.owner, "HEAL_FINISHED")

    def depart(self) -> None:
        for unit, rule in self.recipients.values():
            unit.damage_rules[:] = [item for item in unit.damage_rules if item is not rule]
        self.recipients.clear()
        self.cells.clear()
        self.finish_frame = None
        self.spent_sp = 0.0

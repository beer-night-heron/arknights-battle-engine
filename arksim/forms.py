"""Character form lifecycle adapters used by the main battle engine.

Animation profiles are local inputs. Their static events establish parameters;
the fixed-frame handoff remains a candidate until compared with a live battle.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any

from .mechanics import DamagePacket, DamageRule, DamageRuleKind, DamageType, Status, apply_damage

MAKOTO = "char_4217_makoto"


@dataclass
class MakotoS3:
    battle: Any
    owner: Any
    profile: dict[str, Any]
    mode: str = "BODY"
    transition_target: str | None = None
    transition_until: float = -1.0
    expires_at: float = -1.0
    remaining: float = 0.0
    atk_bonus: float = 0.0
    hp_bonus: float = 0.0
    attack_speed: float = 0.0
    attack_time_add: float = 0.0
    heal_next_at: float = -1.0
    delayed_heals: list[tuple[float, Any]] = field(default_factory=list)
    allout_at: float = -1.0
    allout_stage_at: float = -1.0
    block_release_at: float = -1.0
    allout_members: list[Any] = field(default_factory=list)
    aura_rules: dict[int, tuple[Any, DamageRule]] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    component_rebuild: list[Any] = field(default_factory=list)

    def __post_init__(self):
        timing = self.profile.get("body_attack", {})
        if timing:
            self.owner.attack_duration = float(timing["attack_duration"])
            self.owner.attack_hit_time = float(timing["attack_hit_time"])
        self.base_interval = self.owner.attack_interval
        self.base_block = self.owner.block_count
        self.base_timing = self.owner.attack_duration, self.owner.attack_hit_time
        self.base_immunities = set(self.owner.statuses.immunities)

    @property
    def talent(self):
        return self.owner.talent_blackboards.get("1", {})

    @property
    def board(self):
        return self.owner.skill_blackboard

    def record(self, event: str):
        self.events.append({"event": event, "mode": self.mode,
                            "time": self.battle.time, "frame": self.battle.frame_index,
                            "hp": self.owner.hp, "sp": self.owner.sp,
                            "expires_at": self.expires_at})

    def on_lethal(self, packet: DamagePacket) -> bool:
        if self.mode == "BODY":
            self.begin("THANATOS", "LETHAL")
            return True
        if self.mode == "THANATOS" and self.expires_at - self.battle.time > .01:
            self.begin("ORPHEUS", "LETHAL")
            return True
        return False

    def manual_switch(self) -> bool:
        if (self.mode != "THANATOS" or not self.owner.statuses.can_use_skill
                or self.battle.time + 1e-9 < self.transition_until + float(self.profile.get("manual_enable_delay", .07))):
            return False
        self.begin("ORPHEUS", "MANUAL")
        return True

    def _clear_aura(self):
        for target, rule in self.aura_rules.values():
            target.damage_rules[:] = [r for r in target.damage_rules if r is not rule]
        self.aura_rules.clear()

    def depart(self):
        self._clear_aura()
        self.delayed_heals.clear()
        self.allout_at = -1
        self.allout_stage_at = -1
        self.block_release_at = -1

    def _reset(self):
        op, battle = self.owner, self.battle
        self._clear_aura()
        if op.component_listeners:
            self.component_rebuild = list(op.component_listeners)
            battle._dispatch_component_lifecycle(op, "ON_BUFF_FINISH")
            battle.component_catalog.detach(op)
        # End real instances, including their precise cross-container children.
        # Intrinsic component definitions are rebuilt at the existing handoff;
        # external buffs must not return with them.
        for instance in (*op.buffs.active, *op.buffs.pending_add):
            op.buffs.finish(instance)
        op.buffs.commit()
        op.pending_attack = None
        op.attack_cd = 0
        op.attack_chain_active = False
        op.statuses.durations.clear()
        op.statuses.grants.clear()
        op.statuses.immunities = set(self.base_immunities)
        op.damage_rules.clear()
        op.element_burst = None
        op.palsy_stacks = 0
        op.buff_blackboards.clear()
        op.sp = 0
        self.atk_bonus = self.hp_bonus = self.attack_speed = self.attack_time_add = 0
        op.attack_interval = self.base_interval
        op.attack_duration, op.attack_hit_time = self.base_timing
        op.range_cells = set(op.base_range_cells)
        op.damage_type = "PHYSICAL"
        op.block_count = self.base_block
        battle._squad_field_signature = None
        battle._sync_squad_effects()
        op.hp = op.max_hp
        op.dead = False

    def begin(self, target: str, reason: str):
        battle, op = self.battle, self.owner
        old = self.mode
        was_unblocking = op.block_count <= 0
        self.remaining = max(0., self.expires_at - battle.time) if target == "ORPHEUS" else 0
        self._reset()
        self.mode = "TRANSITION"
        self.transition_target = target
        key = {"THANATOS": "enter_delay", "ORPHEUS": "change_delay", "BODY": "return_delay"}[target]
        handoff = float(self.profile.get("lethal_handoff_delay", 0)) if reason == "LETHAL" else 0.
        self.transition_until = battle.time + float(self.profile.get(key, 0)) + handoff
        self.block_release_at = -1
        op.block_count = self.base_block
        if (was_unblocking and op.block_count > 0
                and battle.assumptions.operator_block_scan_from_deploy_frame
                and battle.assumptions.form_block_resume_next_frame):
            # Resuming the form's block scan has its own next-frame boundary.
            op.block_scan_next_frame = battle.frame_index + 1
            battle.used_assumptions.add("FORM_BLOCK_RESUME_NEXT_FRAME")
        if target == "THANATOS":
            self.block_release_at = battle.time + float(self.profile.get("thanatos_block_release_delay", 0)) + handoff
            if self.block_release_at <= battle.time:
                self._release_blocks()
        for status in (Status.INVINCIBLE, Status.SP_BLOCKED,
                       Status.HEAL_FREE, Status.FORCE_DISARMED, Status.SKILL_NOT_ACTIVATABLE):
            op.statuses.apply(status, None, resistible=False)
        op.statuses.immunities.update({Status.STUNNED, Status.FROZEN, Status.SLEEP, Status.DISABLED})
        if old == "BODY" and target == "THANATOS":
            cells = battle.range_cells_from_id(op.skill.get("rangeId", "x-1"), op.tile, op.facing)
            for enemy in battle.enemies:
                if battle._enemy_is_attackable(enemy) and battle._enemy_intersects_range(enemy, cells):
                    battle.apply_status(enemy, Status.SLUGGISH, float(self.talent.get("sluggish", 0)))
        if target == "BODY":
            self.allout_members = [u for u in battle.operators if not u.dead
                                  and "sees" in {battle.characters[u.char_id].get(k)
                                                 for k in ("groupId", "teamId", "nationId")}]
            delay = float(self.profile.get("allout_damage_delay", 1.6))
            frames = delay / battle.dt
            # Float32 asset values close to an integer frame are quantized here;
            # this does not change the timing of ordinary damage packets.
            if math.isclose(frames, round(frames), abs_tol=1e-5, rel_tol=0):
                frames = float(round(frames))
            self.allout_at = battle.time + math.ceil(frames) * battle.dt
            stage_delay = float(self.profile.get("allout_stage_delay",
                max(0., float(self.profile.get("return_delay", 0)) - 1.4333332777023315)))
            self.allout_stage_at = battle.time + stage_delay
        self.record("BEGIN_" + target + ":" + reason)

    def _release_blocks(self):
        self.block_release_at = -1
        self.owner.block_count = 0
        for enemy in list(self.owner.blocked):
            self.battle._release_enemy_block(enemy)

    def finish_transition(self):
        battle, op = self.battle, self.owner
        target = self.transition_target
        remaining = self.remaining
        preserved_sp = op.sp if self.mode == "ALL_OUT" else 0.
        self._reset()
        op.sp = preserved_sp
        self.mode = str(target)
        self.transition_target = None
        if target in ("THANATOS", "ORPHEUS"):
            self.atk_bonus = float(self.talent.get("atk", 0))
            self.hp_bonus = (float(op.trait_blackboard.get("max_hp", 0))
                             + float(self.talent.get("max_hp_t1", 0)))
            self.attack_time_add = float(self.talent.get("base_attack_time", 0))
            self.expires_at = battle.time + (remaining if target == "ORPHEUS"
                                            else float(op.trait_blackboard.get("duration", 20)))
            op.range_cells = battle.range_cells_from_id(op.skill.get("rangeId", "x-1"), op.tile, op.facing)
            op.statuses.apply(Status.SP_BLOCKED, None, resistible=False)
            if target == "THANATOS":
                self._release_blocks()
                op.prioritize_blocked = False
                op.damage_type = "WEAKNESS"
                self.attack_speed = float(self.board.get("talent@attack_speed", 0))
                timing = self.profile.get("thanatos_attack", {})
                op.attack_duration = float(timing.get("attack_duration", 0))
                op.attack_hit_time = float(timing.get("attack_hit_time", 0))
            else:
                op.block_count = self.base_block - 2 + int(self.board.get("attack@block_cnt", 2))
                op.statuses.apply(Status.FORCE_DISARMED, None, resistible=False)
                op.statuses.apply(Status.SKILL_NOT_ACTIVATABLE, None, resistible=False)
                self.heal_next_at = battle.time
        else:
            op.prioritize_blocked = True
            self.expires_at = -1
        battle._squad_field_signature = None
        battle._sync_squad_effects()
        op.hp = op.max_hp
        if self.component_rebuild:
            for old in self.component_rebuild:
                record = {**old.record, "state": "attached", "recreated_after_form": self.mode}
                listener = replace(old, board=dict(old.board), record=record,
                                   instance=None, started=False, finished=False)
                op.component_listeners.append(listener)
                battle.component_status.append(record)
            self.component_rebuild.clear()
            battle._dispatch_component_lifecycle(op, "ON_BUFF_START")
        battle.initialize_operator_target_selector(op, first_search_frame=battle.frame_index + 1)
        self.record("ENTER_" + self.mode)

    def enter_allout_stage(self):
        op = self.owner
        self.allout_stage_at = -1
        self.mode = "ALL_OUT"
        for status in (Status.SP_BLOCKED, Status.HEAL_FREE, Status.UNTARGETABLE):
            op.statuses.remove(status)
        op.statuses.immunities = set(self.base_immunities)
        # The native mode has no attack ability, but remains invincible and
        # skill-disabled while SP can recover. Reuse the existing attack gate.
        op.block_count = self.base_block
        op.prioritize_blocked = True
        self.record("ENTER_ALL_OUT")

    def tick(self):
        battle, op = self.battle, self.owner
        if op.dead:
            self._clear_aura()
            return
        if self.block_release_at >= 0 and battle.time + 1e-9 >= self.block_release_at:
            self._release_blocks()
        if self.mode in ("TRANSITION", "ALL_OUT"):
            if (self.transition_target == "BODY" and self.allout_stage_at >= 0
                    and battle.time + 1e-9 >= self.allout_stage_at):
                self.enter_allout_stage()
            if battle.time + 1e-9 >= self.transition_until:
                self.finish_transition()
        elif self.mode in ("THANATOS", "ORPHEUS") and battle.time + 1e-9 >= self.expires_at:
            self.begin("BODY", "TIMEOUT")
        if self.allout_at >= 0 and battle.time + 1e-9 >= self.allout_at:
            self.allout_at = -1
            board = op.talent_blackboards.get("2", {})
            amount = battle.operator_attack_power(op) * float(board.get("atk_scale", 0))
            for member in self.allout_members:
                if member.dead: continue
                cells = battle.range_cells_from_id(op.skill.get("rangeId", "x-1"), member.tile, member.facing)
                for enemy in list(battle.enemies):
                    if battle._enemy_is_attackable(enemy) and battle._enemy_intersects_range(enemy, cells):
                        battle._apply_operator_damage(op, enemy, amount, attack_type="NORMAL",
                                                     apply_way="MELEE", damage_type="TRUE",
                                                     damage_source=op if member is op else None)
            self.record("ALL_OUT_ATTACK")
        if self.mode == "ORPHEUS":
            self._sync_aura()
            if battle.time + 1e-9 >= self.heal_next_at:
                self.heal_next_at += float(self.board.get("attack@interval", 1))
                targets = [u for u in battle.operators if not u.dead and u.hp < u.max_hp
                           and not u.statuses.has(Status.HEAL_FREE)
                           and u.tile in op.range_cells]
                targets.sort(key=lambda u: (u.hp/u.max_hp, u.creation_index))
                for target in targets[:int(self.board.get("attack@max_target_heal", 1))]:
                    self.delayed_heals.append((battle.time + float(self.profile.get("heal_delay", .5)), target))
        pending = self.delayed_heals
        self.delayed_heals = []
        for due, target in pending:
            if battle.time + 1e-9 < due:
                self.delayed_heals.append((due, target))
            elif not target.dead:
                apply_damage(DamagePacket(battle.operator_attack_power(op)*float(self.board.get("attack@heal_scale", 0)),
                                          DamageType.HEAL, source=op), target)

    def _sync_aura(self):
        live = {id(u): u for u in self.battle.operators if not u.dead
                and u.tile in self.owner.range_cells}
        for key, (target, rule) in list(self.aura_rules.items()):
            if key not in live:
                target.damage_rules[:] = [r for r in target.damage_rules if r is not rule]
                del self.aura_rules[key]
        for key, target in live.items():
            if key in self.aura_rules and not any(r is self.aura_rules[key][1] for r in target.damage_rules):
                del self.aura_rules[key]
            if key not in self.aura_rules:
                rule = self.battle.add_damage_rule(target, DamageRule(
                    DamageRuleKind.EVADE, float(self.board.get("attack@prob", 0)),
                    damage_types=frozenset({DamageType.PHYSICAL, DamageType.ARTS})))
                self.aura_rules[key] = target, rule

"""Selected operator abilities on the main engine's existing combat lifecycle.

Values come from the equipped skill/talent blackboards. Movement and animation
phases without runtime evidence remain explicitly diagnosed as candidates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import cmp_to_key
from typing import Any

from .mechanics import (ApplyWay, AttackType, DamagePacket, DamageRule,
                        DamageRuleKind, DamageType, ElementType, Status,
                        compare_operator_target_keys, operator_target_key)
from .projectile_motion import HoverMotion, movement_speed, steer_direction, stepped_sine_position


NEXT = frozenset({
    "skchr_wisdel_1", "skchr_hsgma2_2", "skchr_ulpia_1", "skchr_oblvns_1",
    "skchr_glady_1", "skchr_ascln_1", "skchr_phatm2_1", "skchr_ines_1",
    "skchr_horn_1", "skchr_cello_1", "skchr_svrash_1", "skchr_halo2_2",
    "skchr_etlchi_1", "skchr_angel_1", "skchr_skybx_1",
})
HEAL_ACTION = frozenset({"skchr_demkni_1", "skchr_demkni_2", "skchr_thorn2_1"})
PERSISTENT = frozenset({"skchr_sbell2_2", "skchr_hsgma2_1", "skchr_ulpia_2",
                        "skchr_logos_1", "skchr_phatm2_2", "skchr_agoat2_1",
                        "skchr_swire2_3"})
AMMO = frozenset({"skchr_lemuen_1", "skchr_angel2_1"})
TIMED = frozenset({"skchr_whitw2_3", "skchr_hsguma_3", "skchr_kalts2_3",
                   "skchr_angel_3"})
SUPPORTED = NEXT | HEAL_ACTION | PERSISTENT | AMMO | TIMED | {
    "skchr_svash2_1", "skchr_necras_3", "sktok_wisdel_wward", "skchr_whitw2_1",
    "sktok_necras_skeltn_3"}
SOUL_WARD = "token_10035_wisdel_wward"
NECRAS_SERVANT_SKILL = "sktok_necras_skeltn_3"
TOKEN_SKILLS = {
    "token_10054_phatm2_encdool": "skchr_phatm2_2",
    "token_10068_kalts2_mtship": "skchr_kalts2_3",
}
FACING = ((0, 1), (-1, 0), (0, -1), (1, 0))


def token_builds(squad, characters):
    """Tokens inherit owner phase/level and never occupy a squad entry."""
    output = {}
    for owner in squad:
        for key, sid in TOKEN_SKILLS.items():
            if owner.get("skill_id") != sid or key not in characters:
                continue
            char = characters[key]
            elite = min(owner["elite"], len(char["phases"]) - 1)
            output[key] = {"char_id": key, "elite": elite,
                           "level": min(owner["level"], char["phases"][elite]["maxLevel"]),
                           "trust": 0, "potential_rank": 0, "skill_id": None,
                           "skill_level": None, "module_id": None, "module_level": 0, "auto_skill": False}
    return output


def live_runtimes(battle):
    return [op.skill_runtime for op in battle.operators
            if not op.dead and getattr(op, "skill_runtime", None) is not None]


def effect_runtimes(battle):
    return [op.skill_runtime for op in battle.operators if getattr(op, "skill_runtime", None) is not None]


@dataclass
class AttackProfile:
    scale: float = 1.0
    hits: int = 1
    kind: str = "PHYSICAL"
    radius: float = 0.0
    target_count: int = 1
    all_range: bool = False
    blocked_only: bool = False
    skill: bool = False
    payload: dict[str, Any] = field(default_factory=dict)


class OperatorSkillRuntime:
    def __init__(self, battle, owner):
        self.b, self.op = battle, owner
        self.sid = owner.skill_prefab_id or ""
        self.bb = owner.skill_blackboard
        self.armed = False
        self.active = False
        self.action_end = -1.0
        self.heal_spent = 0.0
        self.alchemy_flights = []
        self.next_pulse = battle.time + 1.0
        self.next_cello_pulse = battle.time + float(battle.attack_timing.get(owner.char_id, {}).get(
            "talent1", {}).get("first_interval", .9)) if owner.char_id == "char_245_cello" else math.inf
        count_check = battle.attack_timing.get(owner.char_id, {}).get("module_enemy_count", {})
        self.enemy_count_interval = (float(count_check.get("interval", 0))
            if count_check.get("module_id") == owner.module_id else 0.0)
        self.next_enemy_count_check = battle.time + float(count_check.get("first_interval", 0))
        self.enemy_count_bonus = False
        low_hp_check = battle.attack_timing.get(owner.char_id, {}).get("module_low_hp", {})
        self.low_hp_interval = (float(low_hp_check.get("interval", 0))
            if low_hp_check.get("module_id") == owner.module_id else 0.0)
        self.next_low_hp_check = battle.time + float(low_hp_check.get("first_interval", 0))
        self.low_hp_bonus = False
        pre_reborn = battle.attack_timing.get(owner.char_id, {}).get("module_pre_reborn", {})
        self.pre_reborn_interval = (float(pre_reborn.get("interval", 0))
            if pre_reborn.get("module_id") == owner.module_id else 0.0)
        self.next_pre_reborn_check = battle.time + float(pre_reborn.get("first_interval", 0))
        self.pre_reborn_bonus = False
        high_hp_check = battle.attack_timing.get(owner.char_id, {}).get("module_high_hp", {})
        self.high_hp_interval = (float(high_hp_check.get("interval", 0))
            if high_hp_check.get("module_id") == owner.module_id else 0.0)
        self.next_high_hp_check = battle.time + float(high_hp_check.get("first_interval", 0))
        self.high_hp_bonus = False
        ep_check = battle.attack_timing.get(owner.char_id, {}).get("module_element_check", {})
        self.ep_check_interval = (float(ep_check.get("interval", 0))
            if ep_check.get("module_id") == owner.module_id else 0.0)
        self.ep_aura_ready_at = battle.time + float(ep_check.get("activation_delay", 0))
        self.ep_aura_members = {}
        self.dots = {}  # key -> target, expiry times, per-second amount/type
        self.dot_next = {}
        self.zones = []
        self.events = []
        self.snow = {}
        self.snow_occupants = {}
        self.next_snow = battle.time + float(self.talent(1).get("interval", 5.5))
        self.coins = 0
        self.end_volley = None
        self.ammo_bombs = []
        self.upkeep = battle.time + 3.0
        self.stacks = 0
        self.tokens = []
        self.special_token = None
        self.token_levels = {}
        self.servant_queue = []
        self.notes = []
        self.drones = []
        self.funnel_target = None
        self.funnel_hits = 0
        self.fever = getattr(battle, "mujica_fever_value", 0)
        self.fever_until = getattr(battle, "mujica_fever_until", -1.0)
        self.next_state = battle.time
        self.last_damage = battle.time
        self.revive_count = 0
        self.first_hits = set()
        self.remnants = {}
        self.ward_spawn_at = None
        self.ward_camouflage_sources = {}
        self.entered = {}
        self.medical_regen = {}
        self.medical_shields = []
        self.deck_discount = None
        self.deck_discounts = {}
        self.deck_affected_at = {}
        self.deck_shields = {}
        self.attack_speed_delta = 0.0
        self.block_delta = 0
        self.growth_block_delta = 0
        self.hp_bonus = 0.0
        self.interval_delta = 0.0
        self.steal_atk = {}
        self.stolen_hp = 0.0
        self.stolen_enemy_hp = {}
        self.reaper_pending_hits = 0
        self.reaper_batch_at = None
        self.reaper_heal_count = 0
        self.reaper_heal_at = None
        self.lemuen_bonus = False
        self.wanted_entered = {}
        self.wanted_targets = set()
        self.abyssal_shared = {}
        self.angel_blessings = {}
        self.angel_blessing_count = 0
        self.angel_blessing_started = False
        self.lure = None
        self.lure_ready_at = battle.time
        self.anchor_ready = False
        self.anchor_move = None
        self.relocation_reserve = {}
        self.cast_automatic = False
        self.skill_spent = 0.0
        self.next_continuous_hit = math.inf
        self.low_hp_recovered = False
        self.enemy_entered = {}
        weak_check = battle.attack_timing.get(owner.char_id, {}).get("talent_weakness", {})
        self.weakness_check_interval = float(weak_check.get("check_interval", 0))
        self.weakness_members = {}
        self.extra_upkeep_stacks = 0
        self.attack_bonuses = {}
        self.penetration_stacks = 0
        self.penetration_until = -1.0
        self.logos_execute_checks = {}
        self.shadow_cells = set()
        self.pull_damage = {}
        self.element_pulses = {}
        self.agoat_aura_checks = {}
        self.return_at = -1.0
        self.glady_flight_target = None
        self.glady_landed = False
        self.glady_post_until = -1.0
        self.glady_emitted = False
        self.glady_anim_end = -1.0
        self._init_traits()
        battle.mechanic_diagnostics[f"OPERATOR_SKILL_TIMING_CANDIDATE:{self.sid}"] = 1

    def talent(self, index):
        board = self.op.talent_blackboards.get(str(index))
        if board is None:
            board = self.op.talent_blackboards.get(f"{index}_root")
        if board is None and index in (1, 2):
            talents = self.b.characters.get(self.op.char_id, {}).get("talents") or []
            data_index = next((i for i, talent in enumerate(talents) if any(
                candidate.get("prefabKey") in (str(index), f"{index}_root")
                for candidate in talent.get("candidates", []))), index - 1)
            board = self.op.talent_blackboards_by_index.get(data_index)
        return board or {}

    @property
    def fever(self):
        return getattr(self.b, "mujica_fever_value", 0)

    @fever.setter
    def fever(self, value):
        self.b.mujica_fever_value = value

    @property
    def fever_until(self):
        return getattr(self.b, "mujica_fever_until", -1.0)

    @fever_until.setter
    def fever_until(self, value):
        self.b.mujica_fever_until = value

    def value(self, *keys, default=0.0):
        for key in keys:
            value = self.bb.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return default

    def _init_traits(self):
        op = self.op
        if op.subprofession in {"stalker", "reaper", "crusher"}:
            op.multi_target = True
        if op.char_id == "char_4132_ascln":
            op.taunt_level -= 1
            op.damage_rules.append(DamageRule(DamageRuleKind.EVADE,
                value=float(op.trait_blackboard.get("prob", .5)),
                damage_types=frozenset({DamageType.PHYSICAL, DamageType.ARTS})))
            self.b.mechanic_diagnostics["ASCLN_EVASION_ORDER_CANDIDATE"] = 1
        if op.subprofession in {"artsprotector", "phalanx", "ritualist", "funnel", "slower", "soulcaster"}:
            op.damage_type = "ARTS"
        if self.sid == "skchr_oblvns_1":
            op.damage_type = "ARTS"
            op.lethal_handler = self.on_lethal
            self.b.mechanic_diagnostics["SAKIKO_NOTE_MOVEMENT_CANDIDATE"] = 1
            self.b.mechanic_diagnostics["SAKIKO_SHARED_FEVER_RANGE_EXTENSION_INCOMPLETE"] = 1
        if self.sid.startswith("skchr_demkni"):
            op.damage_type = "PHYSICAL"  # guardian attacks between heal actions
        if self.sid == "skchr_kalts2_3":
            op.damage_type = "HEAL"
            op.air_blocking = True
            op.block_radius += float(self.talent(1).get("block_radius_scale", 0))
            op.block_count += int(self.talent(1).get("block_cnt", 0))
            op.base_max_hp *= 1 + float(self.talent(1).get("max_hp", 0))
            op.def_scale += float(self.talent(1).get("def", 0))
            op.max_hp = op.hp = op.base_max_hp
            op.elements.max_value = op.max_hp
        if self.sid == "skchr_skybx_1":
            op.air_blocking = True
            op.block_radius = math.sqrt(.8)
        if op.char_id == "char_103_angel":
            self.attack_speed_delta += float(self.talent(1).get("attack_speed", 0))
        if op.char_id == "char_4087_ines":
            if self.b.redeploy_penalty.get(op.char_id, 0) == 0:
                op.respawn_time *= max(0, 1 + float(op.trait_blackboard.get("respawn_time", 0)))
        if op.char_id == "char_4132_ascln":
            self.attack_speed_delta += float(self.talent(2).get("attack_speed", 0))
            if any(self.b.tile(*cell).get("heightType") == "HIGHLAND" for cell in
                   ((op.tile[0] + 1, op.tile[1]), (op.tile[0] - 1, op.tile[1]),
                    (op.tile[0], op.tile[1] + 1), (op.tile[0], op.tile[1] - 1))):
                self.attack_speed_delta += float(self.talent(2).get("attack_speed_add", 0))
        if op.char_id in {"char_1044_hsgma2", "char_4039_horn", "char_1033_swire2", "char_1046_sbell2",
                          "char_4010_etlchi"}:
            op.lethal_handler = self.on_lethal

    def grant_attack_bonus(self, key, amount, duration):
        if amount and duration > 0:
            self.attack_bonuses[key] = (float(amount), self.b.time + float(duration))

    def coin_limit(self):
        return 10 + int(self.talent(11).get("sp", 0))

    def stack_limit(self):
        return int(self.talent(1).get("max_stack_cnt", 0)) + min(
            self.b.module_context.get("ingots", 0), int(self.talent(12).get("max", 0)))

    def add_coins(self, amount, *, upkeep=False):
        extra = int(self.talent(11).get("sp_add", 0)) if upkeep else 0
        self.coins = min(self.coin_limit(), self.coins + amount + extra)
        if upkeep:
            self.stacks = min(self.stack_limit(), self.stacks + 1 + int(self.talent(11).get("atk_per_coin", 0)))

    def cost_spent(self, amount):
        seconds = amount * float(self.op.trait_blackboard.get("respawn_time_ratio", 0))
        if not seconds:
            return
        ready = [(t, cid) for cid, t in self.b.redeploy_ready_at.items() if t > self.b.time and cid.startswith("char_")]
        if ready:
            t, cid = min(ready, key=lambda item: item[0])
            self.b.redeploy_ready_at[cid] = max(self.b.time, t-seconds)
            self.b.mechanic_diagnostics["SWIRE_ROGUE_COST_NOTIFICATION_ORDER_CANDIDATE"] = 1

    def calculate_damage_modifier(self, packet, target):
        if (self.op.char_id == "char_4133_logos" and packet.source is self.op and not self.op.dead
                and target in self.b.enemies and target.level_type in {"ELITE", "BOSS"}
                and packet.normalized_damage_type() not in {DamageType.HEAL, DamageType.LIFE_LOSS}):
            amount = float(self.op.trait_blackboard.get("sp", 0))
            if amount:
                self.b._gain_sp(self.op, amount)
                self.b.mechanic_diagnostics["LOGOS_MODULE_CALCULATE_SP_PHASE_CANDIDATE"] = 1

    def output_modifier(self, packet, target):
        op, b = self.op, self.b
        if packet.normalized_damage_type() is DamageType.HEAL:
            if (op.char_id == "char_202_demkni" and not op.dead and packet.source is op
                    and target in b.operators and not target.dead and self.talent(2)):
                t = self.talent(2)
                key = "demkni_e_sp[rhine].sp" if self.has_group_tag(target, "rhine") else "demkni_e_sp.sp"
                b._gain_sp(target, float(t.get(key, t.get("sp", 0))))
                b.mechanic_diagnostics["DEMKNI_HEAL_OUTPUT_PHASE_CANDIDATE"] = 1
            return
        self.output_damage(packet, target)

    def output_damage(self, packet, target):
        op, b = self.op, self.b
        if packet.normalized_damage_type() in {DamageType.HEAL, DamageType.LIFE_LOSS}:
            return
        if (op.char_id == "char_4145_ulpia" and not op.dead
                and op.module_id == "uniequip_003_ulpia" and packet.source is op
                and target in b.enemies and target.blocked_by is not None):
            # ulpia_e_003_tr is ON_OUTPUT_DAMAGE, with no damage/attack mask
            # and no requirement that this particular source be the blocker.
            packet.output_scale *= float(op.trait_blackboard.get("damage_scale", 1))
            b.mechanic_diagnostics["ULPIA_Y_BLOCKED_OUTPUT_PHASE_CANDIDATE"] = 1
        normal = packet.normalized_attack_type() is AttackType.NORMAL
        if op.char_id == "char_103_angel" and not op.dead and packet.source is op and target in b.enemies:
            t = self.talent(1)
            if "def_penetrate_fixed" in t:
                if b.time >= self.penetration_until:
                    self.penetration_stacks = 0
                self.penetration_stacks = min(int(t.get("max_stack_cnt", 0)), self.penetration_stacks + 1)
                self.penetration_until = b.time + float(t.get("duration", 5))
                b.mechanic_diagnostics["ANGEL_PENETRATION_EVENT_PHASE_CANDIDATE"] = 1
        if (op.char_id == "char_1044_hsgma2" and not op.dead and self.active
                and packet.source is op and target in b.enemies and normal
                and packet.ability_owner is op and packet.modifier_key != "artsprotector_e_002_tr"):
            scale = float(op.trait_blackboard.get("atk_scale", 0))
            if scale:
                self.damage(target, b.operator_attack_power(op) * scale, attack_type=AttackType.ADDITION,
                            modifier_key="artsprotector_e_002_tr", ability_owner=None)
                b.mechanic_diagnostics["HSGMA_MODULE_ADDITIONAL_DAMAGE_ORDER_CANDIDATE"] = 1
        if (op.char_id == "char_4132_ascln" and not op.dead and packet.source is op and normal
                and packet.normalized_damage_type() is not DamageType.HEAL
                and target in b.enemies and self.talent(1)):
            t = self.talent(1)
            self.add_dot(target, "ASCLN", float(t.get("debuff_duration", 25)), float(t.get("atk_ratio", .1)),
                         stacks=int(t.get("max_stack_cnt", 3)), current_atk=True)
            b.mechanic_diagnostics["ASCLN_OUTPUT_DAMAGE_DEATH_PHASE_CANDIDATE"] = 1
        if (op.char_id == "char_4010_etlchi" and not op.dead and packet.source is op
                and packet.normalized_attack_type() is not AttackType.BUFF
                and target in b.enemies and not target.dead and self.talent(1)):
            t = self.talent(1)
            self.add_dot(target, "ETLCHI", float(t.get("dot_duration", 5)), float(t.get("magic_value", 0)))
            b.mechanic_diagnostics["ETLCHI_DOT_OUTPUT_PHASE_CANDIDATE"] = 1

    def damage_landed(self, packet, target, result):
        if (result.cancelled or result.evaded or packet.source is not self.op
                or packet.normalized_damage_type() in {DamageType.HEAL, DamageType.LIFE_LOSS}):
            return
        op, b = self.op, self.b
        if (op.char_id == "char_4010_etlchi" and not op.dead
                and packet.normalized_damage_type() is not DamageType.HEAL
                and packet.normalized_attack_type() is not AttackType.BUFF):
            if result.final_amount > 0:
                self.queue_reaper_heal()

    def element_output_scale(self, target):
        return (float(self.op.trait_blackboard.get("ep_damage_scale", 1))
                if target.level_type in {"ELITE", "BOSS"} else 1.0)

    def element_vulnerability_scale(self, target):
        if self.op.char_id != "char_245_cello":
            return 1.0
        scale = float(self.talent(10).get("damage_scale", 1)) if self.b._enemy_intersects_range(target, self.op.range_cells) else 1.0
        if target.elements.active_break is ElementType.DARK and self.b._enemy_intersects_range(target, self.op.range_cells):
            scale = max(scale, float(self.talent(2).get("damage_scale", 1)))
        return scale

    def ally_damage_resistance(self, target, source=None):
        if self.op.char_id == "char_474_glady" and self.b.characters.get(target.char_id, {}).get("groupId") == "abyssal":
            upgraded = self.op.module_id == "uniequip_002_glady" and self.op.module_level >= 2
            if source in self.b.enemies and (upgraded or "seamonster" in source.enemy_tags):
                return float(self.talent(1).get("damage_resistance", 0))
        return 0.0

    def own_damage_resistance(self, kind=DamageType.PHYSICAL):
        return (float(self.op.trait_blackboard.get("damage_resistance", 0))
                if self.op.char_id == "char_202_demkni"
                and kind in {DamageType.PHYSICAL, DamageType.ARTS, DamageType.TRUE, DamageType.ELEMENTAL} else 0.0)

    def reveals(self, enemy):
        if self.op.char_id == "char_4039_horn":
            return any(z["kind"] == "ILLUMINATE" and self.b.time < z["end"]
                       and self.b._enemy_intersects_circle(enemy, z["pos"], z["radius"])
                       for z in self.zones)
        if self.op.char_id != "char_4087_ines":
            return False
        cells = self.shadow_cells if self.op.dead else self.op.range_cells
        return bool(self.talent(2)) and self.b._enemy_intersects_range(enemy, cells)

    def candidates(self, *, cells=None, ground=False):
        own_range = cells is None
        cells = self.op.range_cells if cells is None else cells
        ground = ground or self.op.subprofession in {"stalker", "reaper"}
        values = [e for e in self.b.enemies if self.b._enemy_is_attackable(e)
                  and self.b._enemy_targetable(e) and (self.b._enemy_intersects_range(e, cells)
                      or (own_range and self.is_wanted(e)))
                  and (not ground or not self.b._is_airborne(e))
                  and not e.statuses.has(Status.CAMOUFLAGE)]
        values.sort(key=cmp_to_key(lambda a, z: compare_operator_target_keys(operator_target_key(a), operator_target_key(z))))
        if self.op.char_id == "char_4193_lemuen":
            values.sort(key=lambda e: (self.b._effective_enemy_defense(e), e.creation_index))
        return values

    def is_wanted(self, enemy):
        return (not self.op.dead and self.op.char_id == "char_4193_lemuen"
                and enemy.enemy_index in self.wanted_targets)

    def has_group_tag(self, unit, tag):
        char = self.b.characters.get(getattr(unit, "char_id", ""), {})
        return any(char.get(key) == tag for key in ("nationId", "groupId", "teamId"))

    def wanted_damage_scale(self, source, enemy):
        if not self.is_wanted(enemy):
            return 1.0
        return float(self.talent(1).get("damage_scale", 1)) if self.has_group_tag(source, "laterano") else 1.0

    def tick_wanted_targets(self):
        if self.op.char_id != "char_4193_lemuen" or not self.talent(1):
            return
        b = self.b
        # Any Laterano range sustains the timer; overlap does not speed it up.
        sources = [p for p in b.operators if not p.dead
                   and self.has_group_tag(p, "laterano")]
        present = {e.enemy_index for e in b.enemies if b._enemy_is_attackable(e)
                   and b._enemy_targetable(e) and not e.statuses.has(Status.CAMOUFLAGE)
                   and e.level_type in {"ELITE", "BOSS"}
                   and any(b._enemy_intersects_range(e, p.range_cells) for p in sources)}
        self.wanted_entered = {k: v for k, v in self.wanted_entered.items() if k in present}
        for ident in present - self.wanted_targets:
            started = self.wanted_entered.setdefault(ident, b.time)
            if b.time - started + 1e-9 >= float(self.talent(1).get("interval", math.inf)):
                self.wanted_targets.add(ident)
                if self.op.target_selector is not None:
                    self.op.target_selector.mark_dirty()
        b.mechanic_diagnostics["LEMUEN_WANTED_RANGE_TIMER_PHASE_CANDIDATE"] = 1

    def can_cast(self):
        op = self.op
        if self.end_volley is not None:
            return False
        if self.sid == "skchr_demkni_2":
            # This skill switches state rather than replacing the next attack.
            # An ordinary windup/CD does not defer the ready area-heal action.
            return op.statuses.can_attack and bool(self.heal_targets())
        if self.sid in NEXT | HEAL_ACTION:
            if op.pending_attack is not None or op.attack_cd > 1e-9:
                return False
            if self.b.frame_index < (op.selector_first_search_frame or 0):
                return False
            if not op.statuses.can_attack:
                return False
            if self.sid in HEAL_ACTION:
                return bool(self.heal_targets())
            if self.sid == "skchr_oblvns_1":
                return bool(self.candidates()) or self.front_passable()
            if self.op.char_id == "char_4039_horn":
                return self.b._search_operator_target_now(self.op) is not None
            return bool(self.candidates())
        return True

    def auto_ready(self):
        if self.sid == "skchr_oblvns_1":
            return self.free_cast() or self.op.auto or self.op.sp + 1e-9 >= self.op.sp_max
        return self.op.auto

    def free_cast(self):
        return self.sid == "skchr_oblvns_1" and self.b.time < self.fever_until

    def activate(self):
        op, b = self.op, self.b
        self.active = True
        op.skill_active = True
        op.active_until = -1.0
        op.behavior_active = False  # this runtime owns the selected skill effects
        rid = op.skill.get("rangeId")
        if rid:
            op.range_cells = b.range_cells_from_id(str(rid), op.tile, op.facing)
            if op.target_selector:
                op.target_selector.mark_dirty()
        if self.sid in NEXT:
            self.armed = True
            self.skill_spent = b._skill_cost(op)
            if self.sid == "skchr_hsgma2_2":
                timing = op.skill_attack_timing.get(self.sid, {})
                self.events.append((b.time + float(timing.get("shield_launch_delay", .13333334028720856)),
                                    "SHIELD_LAUNCH", None, 0))
            if (self.sid == "skchr_oblvns_1" and not self.cast_automatic
                    and self.fever >= 450 and not self.free_cast()):
                self.fever = 0
                self.fever_until = b.time + 20
            return True
        if self.sid in HEAL_ACTION:
            self.heal_spent = b._skill_cost(op)
            if self.sid == "skchr_thorn2_1":
                from .battle import _PendingOperatorAttack
                target = self.heal_targets()[0]
                extra = float(self.talent(1).get("projectile_extend", 0)) if any(p is not op and not p.dead and p.tile in op.range_cells for p in b.operators) else 0
                frame, time = b._operator_attack_deadline(op)
                op.pending_attack = _PendingOperatorAttack(time, frame, "ALCHEMY", target, [target],
                    1, op.atk_scale, True, True, profile=AttackProfile(payload={
                        "tile": target.tile, "duration": self.value("projectile_delay_time", default=8) + extra}))
                op.attack_cd = b._effective_attack_interval(op)
                self.action_end = max(b.time + b._operator_action_duration(op), time)
                b._mark_operator_attack_started(op, not op.attack_chain_active)
                b._land_zero_delay_operator_attack(op)
            else:
                b._begin_operator_heal(op, self.heal_targets(), self.value("heal_scale", default=1))
                op.attack_cd = b._effective_attack_interval(op)
                self.action_end = max(b.time + b._operator_action_duration(op),
                                      op.pending_attack.execute_at if op.pending_attack else b.time)
            return True
        if self.sid == "skchr_svash2_1":
            self.begin_state_action("DECK")
            timing = op.skill_attack_timing.get(self.sid, {})
            self.action_end = b.time + float(timing.get("attack_duration", 1))
            return True
        if self.sid == "skchr_necras_3":
            self.stacks = 0
            op.attack_cd = 0.0
            return True
        b._apply_stat_buff(op, self.bb, timed=True,
                           duration=float(op.skill.get("duration", 0)) if self.sid in TIMED else None)
        self.interval_delta = self.value("base_attack_time")
        self.attack_speed_delta += self.value("attack_speed")
        self.block_delta = int(self.value("block_cnt"))
        op.block_count += self.block_delta
        self.hp_bonus = self.value("max_hp")
        b._squad_field_signature = None
        b._sync_squad_effects()
        if self.sid in AMMO:
            capacity = self.value("attack@trigger_time")
            b.configure_skill_ammo(op, capacity)
        if self.sid == "skchr_swire2_3":
            self.add_coins(int(self.talent(1).get("sp", 0)))
        if self.sid == "skchr_whitw2_3":
            count = 1 + int(self.value("attack@cnt")) + (1 if b.time - op.deployed_at >= float(self.talent(1).get("interval", 16)) * 3 else 0)
            self.drones = []
            forward = FACING[op.facing % 4]
            for index in range(count):
                angle = math.atan2(forward[0], forward[1]) + index * 2 * math.pi / count
                b.projectile_counter += 1
                self.drones.append({"pos": tuple(map(float, op.tile)), "target": None,
                    "arrived": False, "id": b.projectile_counter, "angle": angle,
                    "speed": .1, "phase": "SCATTER", "born": b.time,
                    "next_attack": b.time, "hits": 0})
                self.initialize_drone_origin(self.drones[-1], index)
        if self.sid == "skchr_kalts2_3":
            self.anchor_ready = True
        if self.sid == "skchr_hsguma_3":
            self.next_continuous_hit = b.time + b._operator_attack_hit_delay(op)
            op.pending_attack = None
        if self.sid == "skchr_agoat2_1":
            self.tick_agoat_element_aura()
        return True

    def begin_state_action(self, kind, *, hits=1, interval=0):
        """An animation action can run without an enemy selector (notes/DP)."""
        from .battle import _PendingOperatorAttack
        b, op = self.b, self.op
        delay = b._operator_attack_hit_delay(op)
        frame, time = b._operator_attack_deadline(op, delay=delay)
        followups = tuple(b._operator_attack_deadline(op, delay=delay + interval * i)[0]
                          for i in range(1, hits))
        p = AttackProfile(skill=self.armed or self.active,
                          payload={"state_action": kind, "note_index": 0,
                                   "sequenced": hits > 1})
        op.pending_attack = _PendingOperatorAttack(time, frame, "ROSTER_STATE", op, [],
            1, op.atk_scale, op.skill_active, True, profile=p, followup_frames=followups)
        b._register_ability_use(op)
        b._mark_operator_attack_started(op, not op.attack_chain_active)
        b._land_zero_delay_operator_attack(op)

    def land_state_action(self, attack):
        kind = attack.profile.payload["state_action"]
        if kind == "NOTES":
            self.launch_notes(attack.profile.skill, index=attack.profile.payload["note_index"])
        elif kind == "NECRAS":
            if self.active:
                self.necras_action()
        elif kind == "DECK":
            self.b.dp = min(self.b.max_cost, self.b.dp + self.value("cost"))
            self.select_deck_discount()

    def heal_targets(self):
        if self.sid == "skchr_thorn2_1":
            return sorted([p for p in self.b.operators if not p.dead and p.tile in self.op.range_cells
                           and self.b.characters.get(p.char_id, {}).get("profession") != "TRAP"],
                          key=lambda p: (p.hp / p.max_hp, -p.creation_index))
        ratio = .5 if self.sid == "skchr_demkni_1" else 1.0
        rid = self.op.skill.get("rangeId") if self.op.skill else None
        cells = self.b.range_cells_from_id(rid, self.op.tile, self.op.facing) if rid else self.op.range_cells
        targets = self.b._operator_heal_targets(self.op, cells=cells, hp_ratio=ratio, include_equal=ratio < 1)
        return targets[:1] if self.sid == "skchr_demkni_1" else targets

    def override_action(self):
        """True means the runtime owns this action (including intentional idle)."""
        op, b = self.op, self.b
        if self.end_volley is not None:
            return True
        if self.active and not self.armed and self.sid in {"skchr_skybx_1", "skchr_glady_1"}:
            return True
        if self.sid == "skchr_oblvns_1":
            if b.frame_index < (op.selector_first_search_frame or 0):
                return True
            if b._current_operator_target(op) is None and not self.front_passable():
                return True
            timing = op.skill_attack_timing.get(self.sid, {})
            interval = float(timing.get("additional_attack_interval_fever" if self.free_cast()
                                         else "additional_attack_interval", .1))
            self.begin_state_action("NOTES", hits=8 if self.armed else 1, interval=interval)
            op.attack_cd = b._effective_attack_interval(op)
            return True
        if self.sid == "skchr_necras_3" and self.active:
            self.begin_state_action("NECRAS")
            op.attack_cd = b._effective_attack_interval(op)
            return True
        if self.sid == "skchr_hsgma2_2" and self.active and not self.armed:
            return True
        if self.sid == "skchr_cello_1" and not self.armed:
            return True
        if self.sid == "skchr_hsguma_3" and self.active:
            return True
        if self.sid == "skchr_svash2_1" and self.active:
            return True
        if self.sid == "skchr_sbell2_2" and not self.active:
            return True
        if self.sid == "skchr_thorn2_1":
            return True
        if self.anchor_move is not None:
            return True
        if self.sid in HEAL_ACTION and op.skill_active:
            return True
        if op.damage_type == "HEAL":
            targets = b._operator_heal_targets(op)
            if self.sid == "skchr_agoat2_1":
                targets = b._operator_heal_targets(op, include_element_damage=True)
            count = 2 if self.active and self.sid in {"skchr_agoat2_1", "skchr_kalts2_3"} else 1
            if targets:
                b._begin_operator_heal(op, targets[:count], 1.0)
                op.attack_cd = b._effective_attack_interval(op)
            return True
        return False

    def profile(self, primary=None):
        op = self.op
        horn_melee = (bool(op.blocked) if primary is None else
            any(e is primary for e in op.blocked) or
            not self.b._enemy_intersects_range(primary, op.range_cells)) if op.char_id == "char_4039_horn" else False
        p = AttackProfile(kind=op.damage_type, skill=self.armed or self.active)
        if op.subprofession in {"stalker", "reaper"}:
            p.all_range = True
        elif op.subprofession in {"crusher", "centurion"}:
            p.blocked_only = True
        elif op.subprofession == "bombarder":
            p.radius = .9
        elif op.subprofession == "skybreaker":
            p.radius = 1.0
        elif op.subprofession == "fortress":
            p.radius = 0 if horn_melee or op.blocked else 1.1
            p.payload["horn_melee"] = horn_melee
        if op.subprofession == "artsprotector":
            p.kind = "ARTS" if p.skill else "PHYSICAL"
        if not p.skill:
            return p
        p.scale = self.value("attack@atk_scale", "atk_scale", default=1)
        if self.sid in {"skchr_hsgma2_1", "skchr_phatm2_2"}:
            p.scale = 1.0  # Their multiplier belongs to reflection/the placed lure.
        if self.sid == "skchr_sbell2_2":
            p.scale = self.value("attack@atk_scale_s2"); p.all_range = True
        if self.sid == "skchr_wisdel_1":
            p.radius = 1.1
        if self.sid in {"skchr_ascln_1", "skchr_phatm2_1", "skchr_etlchi_1", "skchr_swire2_3"}:
            p.hits = 2
        if self.sid == "skchr_swire2_3":
            p.scale = 1.0
        if self.sid == "skchr_hsgma2_2":
            p.hits = 3; p.blocked_only = True
        if self.sid == "skchr_ulpia_1":
            p.target_count = int(self.value("max_target", default=2)); p.blocked_only = False
        if self.sid == "skchr_lemuen_1":
            p.target_count = 2
        if self.sid == "skchr_horn_1":
            p.radius = 0 if horn_melee else self.value("projectile_range", default=1.7)
        if self.sid == "skchr_skybx_1":
            p.radius = 2.0
        if self.sid == "skchr_ines_1":
            p.scale = 1.0
        if self.sid == "skchr_cello_1":
            p.kind = "ARTS"
        if self.sid == "skchr_halo2_2":
            p.target_count = int(self.value("max_target_attack", default=2))
        if self.sid == "skchr_hsguma_3":
            p.all_range = True
            p.payload["front_only"] = True
        if self.sid in {"skchr_angel_1", "skchr_angel_3"}:
            p.hits = int(self.value("attack@times", "times", default=1))
        p.payload["sid"] = self.sid if p.skill else ""
        return p

    def begin_attack(self, first_attack):
        from .battle import _PendingOperatorAttack
        b, op = self.b, self.op
        b._finish_depleted_ammo_before_attack(op)
        primary = b._current_operator_target(op)
        if primary is None:
            return False
        p = self.profile(primary)
        timing = (op.skill_attack_timing.get(self.sid, {}) if p.skill else
                  b.attack_timing.get(op.char_id, {}))
        if op.char_id == "char_4039_horn":
            timing = {**timing, **timing.get("melee_attack" if p.payload["horn_melee"] else "ranged_attack", {})}
        for key in ("projectile_key", "projectile_delay_after_reached", "keep_projectile_after_departure"):
            if key in timing:
                p.payload[key] = timing[key]
        if op.subprofession == "lord" and op.blocked:
            p.payload["projectile_key"] = None
        if op.char_id == "char_103_angel":
            p.payload["keep_projectile_after_departure"] = True
        candidates = self.candidates()
        if self.sid == "skchr_halo2_2" and p.skill:
            # The first selector limits by descending distance to the exit;
            # the group then applies its ordinary ordering to that subset.
            candidates = sorted(candidates, key=lambda e: (-e.remaining_path_distance, e.creation_index))[:p.target_count]
            candidates.sort(key=cmp_to_key(lambda a, z: compare_operator_target_keys(
                operator_target_key(a), operator_target_key(z))))
            if candidates:
                primary = candidates[0]
            b.mechanic_diagnostics["HALO2_SKILL_SELECTOR_GROUP_ORDER_CANDIDATE"] = 1
        if self.sid == "skchr_angel2_1" and self.active:
            candidates.sort(key=lambda e: not b._is_airborne(e))
            if candidates:
                primary = candidates[0]
        if self.sid == "skchr_cello_1":
            candidates = [e for e in candidates if e.elements.active_break is not ElementType.DARK]
            if not candidates:
                return False
            primary = candidates[0]
        if p.blocked_only:
            targets = [e for e in op.blocked if b._enemy_is_attackable(e)]
            targets = targets or [primary]
        elif p.all_range:
            targets = candidates
            if p.payload.get("front_only"):
                dr, dc = FACING[op.facing % 4]
                cells = {op.tile, (op.tile[0] + dr, op.tile[1] + dc)}
                targets = [e for e in targets if b._enemy_intersects_range(e, cells)]
        else:
            targets = [primary] + [e for e in candidates if e is not primary][:p.target_count - 1]
        if not targets:
            return False
        if p.skill and self.sid == "skchr_glady_1":
            self.glady_flight_target = primary
            self.glady_landed = False
            self.glady_post_until = -1.0
            self.glady_emitted = False
            self.glady_anim_end = -1.0
        delay = (b._operator_attack_hit_delay(op, timing=timing)
                 if op.char_id == "char_4039_horn" else b._operator_attack_hit_delay(op))
        frame, time = b._operator_attack_deadline(op, delay=delay)
        followups = ()
        event_times = timing.get("attack_hit_times", [])
        delta = float(timing.get("additional_attack_interval", 0))
        if p.hits > 1 and (len(event_times) == p.hits or delta > 0):
            first_delay = b._operator_attack_hit_delay(op)
            if len(event_times) == p.hits:
                scale = b._operator_animation_duration_scale(op, timing)
                delays = [float(event) * scale for event in event_times[1:]]
            else:
                delays = [first_delay + delta * i for i in range(1, p.hits)]
            followups = tuple(b._operator_attack_deadline(op, delay=delay)[0] for delay in delays)
            p = replace(p, hits=1, payload={**p.payload, "sequenced": True})
        op.pending_attack = _PendingOperatorAttack(time, frame, "ROSTER", primary, targets,
            p.scale, op.atk_scale, op.skill_active, b._attack_hits(op, p.kind), profile=p,
            followup_frames=followups)
        b._register_ability_use(op)
        b._mark_operator_attack_started(op, first_attack,
            timing=timing if op.char_id == "char_4039_horn" else None)
        if p.skill and self.sid == "skchr_horn_1":
            # This skill does not finish early at the emission event, nor wait
            # for its projectiles to disappear. Cache the action's own duration.
            self.action_end = b.time + b._operator_action_duration(op)
        b._land_zero_delay_operator_attack(op)
        return True

    def continue_attack_sequence(self, attack):
        if not attack.followup_frames or self.op.dead:
            return False
        frame = attack.followup_frames[0]
        self.op.pending_attack = replace(attack, execute_frame=frame,
            execute_at=self.b.time + max(0, frame - self.b.frame_index) * self.b.dt,
            followup_frames=attack.followup_frames[1:], profile=replace(attack.profile,
                payload={**attack.profile.payload,
                         "note_index": attack.profile.payload.get("note_index", 0) + 1}),
            animation_events=attack.animation_events[1:])
        self.b._land_zero_delay_operator_attack(self.op)
        return True

    def refresh_attack_targets(self, attack):
        p = attack.profile
        timing = (self.op.skill_attack_timing.get(self.sid, {}) if p.skill else
                  self.b.attack_timing.get(self.op.char_id, {}))
        if timing.get("select_target_timing") != 1:
            return attack
        candidates = self.candidates()
        targets = candidates if p.all_range else candidates[:p.target_count]
        if p.blocked_only:
            targets = [e for e in self.op.blocked if self.b._enemy_is_attackable(e)]
        return replace(attack, primary=targets[0] if targets else attack.primary, targets=targets)

    def refund_dead_attack_target(self, attack):
        timing = self.op.skill_attack_timing.get(self.sid, {})
        if (not self.armed or self.skill_spent <= 0 or not attack.primary.dead
                or not timing.get("recover_sp_if_target_dead")
                or any(self.b._enemy_is_attackable(e) for e in attack.targets)):
            return False
        self.b._store_sp_gain(self.op, self.skill_spent)
        self.skill_spent = 0.0
        self.b._finish_skill(self.op, "TARGET_DEAD")
        return True

    def cancel_attack_sequence(self, attack):
        if (attack.profile is not None and attack.profile.payload.get("sequenced")
                and self.armed):
            self.b._finish_skill(self.op, "ATTACK_INTERRUPTED")

    def wisdel_specs(self, skilled):
        skilled = skilled and self.sid == "skchr_wisdel_1"
        ordinary = self.b.attack_timing.get(self.op.char_id, {})
        timing = self.op.skill_attack_timing.get(self.sid, {}) if skilled else ordinary
        main = timing.get("wisdel_phases", dict(
            key="projectile_chr_wisdel_s1" if skilled else "projectile_chr_wisdel_default",
            radius=1.1 if skilled else .9, stop_delay=.15,
            stop_gates=["", "enable_third_attack"], hit_on_reach=True,
            keep_after_departure=True))
        children = timing.get("wisdel_children", []) if skilled else []
        return [("MAIN", main), *[("STUN" if item["key"].endswith("_bomb") else "SHOCK", item)
                                  for item in children]]

    def emit_horn_projectiles(self, attack):
        b, op, profile = self.b, self.op, attack.profile
        key = profile.payload.get("projectile_key", op.projectile_key)
        components = [("DAMAGE", key)]
        if profile.skill and self.sid == "skchr_horn_1":
            components.append(("ILLUMINATE", "projectile_chr_horn_s1_light"))
        for target in attack.targets:
            if not b._enemy_is_attackable(target):
                continue
            for component, projectile_key in components:
                payload = replace(profile, payload={**profile.payload, "horn_component": component,
                    "keep_projectile_after_departure": True})
                b._new_projectile(key=projectile_key, side="OPERATOR", source=op, target=target,
                    operator_attack=replace(attack, primary=target, targets=[target], profile=payload),
                    will_hit=attack.will_hit)
        b.mechanic_diagnostics["HORN_PROJECTILE_ARRIVAL_AND_REVEAL_PHASE_CANDIDATE"] = 1

    def land_horn_projectile(self, projectile, *, stopped=False):
        b, op, attack = self.b, self.op, projectile.operator_attack
        profile = attack.profile
        position = (projectile.position_row, projectile.position_col)
        if profile.payload["horn_component"] == "ILLUMINATE":
            if not stopped:
                self.add_zone(position, self.value("projectile_delay_time"), "ILLUMINATE", radius=profile.radius)
            return
        if not projectile.will_hit:
            return
        amount = b.operator_attack_power(op, attack.source_atk_scale) * profile.scale
        for enemy in b.enemies:
            if (not b._enemy_is_attackable(enemy) or b._is_airborne(enemy)
                    or any(enemy.statuses.has(s) for s in (Status.INVINCIBLE, Status.UNTARGETABLE, Status.SLEEP))
                    or not b._enemy_intersects_circle(enemy, position, profile.radius)):
                continue
            # The collision selector ignores camouflage and only the invisible
            # target-free case; losing the trace target does not remove the area.
            self.damage(enemy, amount, profile.kind, attack_type=AttackType.NORMAL, apply_way=ApplyWay.RANGED)

    def emit_skybx_skill_projectile(self, attack):
        b, op = self.b, self.op
        amount = b.operator_attack_power(op, attack.source_atk_scale) * attack.profile.scale
        for target in attack.targets:
            if b._enemy_is_attackable(target):
                b._new_projectile(key="projectile_chr_skybx_s1", side="OPERATOR", source=op,
                    target=target, operator_attack=attack, cached_atk=amount, will_hit=attack.will_hit)
        b.mechanic_diagnostics["SKYBX_SKILL_CACHED_ATTACK_AND_COLLISION_PHASE_CANDIDATE"] = 1

    def land_skybx_splash(self, attack, position, amount):
        if not attack.will_hit:
            return
        for enemy in self.b.enemies:
            if (not self.b._enemy_is_attackable(enemy) or not self.b._enemy_targetable(enemy)
                    or not self.b._enemy_intersects_circle(enemy, position, attack.profile.radius)):
                continue
            # The reached collision ignores camouflage and covers both motion
            # types; each receiver still runs its own damage judgement.
            result = self.damage(enemy, amount, attack.profile.kind,
                attack_type=AttackType.NORMAL, apply_way=ApplyWay.RANGED)
            self.hit_effects(enemy, enemy, amount, attack.profile, result)

    def emit_halo2_links(self, attack, position):
        b, op = self.b, self.op
        targets = [enemy for enemy in b.enemies
            if enemy is not attack.primary and b._enemy_is_attackable(enemy)
            and b._enemy_targetable(enemy) and not enemy.statuses.has(Status.CAMOUFLAGE)
            and b._enemy_intersects_circle(enemy, position[:2], self.value("ability_range_radius"))]
        targets.sort(key=cmp_to_key(lambda a, z: compare_operator_target_keys(
            operator_target_key(a), operator_target_key(z))))
        profile = replace(attack.profile, payload={**attack.profile.payload,
            "pull_origin": position[:2], "link_force": int(self.value("force")),
            "link_scale": self.value("atk_scale_link"), "keep_projectile_after_departure": True})
        key = "projectile_chr_halo2_s2_link"
        for target in targets[:int(self.value("max_target"))]:
            packet = replace(attack, kind="HALO2_LINK", primary=target, targets=[target],
                profile=profile, will_hit=True)
            if b._projectile_speed(key) is None:
                self.land_halo2_link(packet, target)
                b.mechanic_diagnostics["HALO2_LINK_NO_PROJECTILE_DATA_FALLBACK"] = 1
                continue
            b._new_projectile(key=key, side="OPERATOR", source=op, target=target,
                operator_attack=packet, origin_override=position)
            # New entities start on the following update in this candidate
            # schedule, rather than inheriting a full step from their parent.
            b.projectiles[-1].born_at = b.time + b.dt
        b.mechanic_diagnostics["HALO2_LINK_SELECTOR_AND_UPDATE_PHASE_CANDIDATE"] = 1

    def land_halo2_link(self, attack, target, *, source_key=None):
        if not self.b._enemy_is_attackable(target):
            return
        payload = attack.profile.payload
        amount = self.b.operator_attack_power(self.op) * payload["link_scale"]
        self.damage(target, amount, "ARTS", attack_type=AttackType.NORMAL, apply_way=ApplyWay.NONE)
        # The harpoon callback also runs when the damage packet is cancelled
        # or evaded. Its destination is the parent projectile's cached map pos.
        self.b.apply_pull(target, *payload["pull_origin"], payload["link_force"],
            interrupt_source=self.op, source_key=source_key,
            stop_radius=.05, stop_after_reached=True, stop_within_source_block_range=True)
        self.b.mechanic_diagnostics["HALO2_HARPOON_RETURN_AND_CANCEL_PHASE_CANDIDATE"] = 1

    def wisdel_packet_data(self, attack, component, spec, base):
        skilled = attack.profile.skill
        return dict(component=component, spec=spec, base=base, skilled=skilled,
            main_scale=attack.profile.scale,
            primary=attack.primary.enemy_index,
            trace_scale=float(self.talent(1).get("attack@main_atk_scale", 1)),
            append_scale=self.value("append_atk_scale") if skilled else
                float(self.op.trait_blackboard.get("attack@append_atk_scale", .5)),
            stop_gates=[key for key in spec.get("stop_gates", []) if not key
                        or self.op.trait_blackboard.get("attack@"+key, 0)])

    def emit_wisdel_projectiles(self, attack):
        base = self.b.operator_attack_power(self.op, attack.source_atk_scale)
        for target in attack.targets:
            if not self.b._enemy_is_attackable(target):
                continue
            for component, spec in self.wisdel_specs(attack.profile.skill):
                packet = self.wisdel_packet_data(replace(attack, primary=target), component, spec, base)
                profile = replace(attack.profile, payload={**attack.profile.payload,
                    "wisdel_component": packet,
                    "projectile_delay_after_reached": float(spec["stop_delay"]),
                    "keep_projectile_after_departure": spec["keep_after_departure"]})
                self.b._new_projectile(key=spec["key"], side="OPERATOR", source=self.op,
                    target=target, operator_attack=replace(attack, primary=target, targets=[target],
                        profile=profile), cached_atk=base, will_hit=attack.will_hit)
        self.b.mechanic_diagnostics["WISDEL_PROJECTILE_PHASE_AND_CACHE_CANDIDATE"] = 1

    def wisdel_projectile_reached(self, projectile):
        if projectile.will_hit:
            self.wisdel_reach_packet((projectile.position_row, projectile.position_col),
                projectile.operator_attack.profile.payload["wisdel_component"])

    def wisdel_projectile_stopped(self, projectile):
        self.wisdel_stop_packet((projectile.position_row, projectile.position_col),
            projectile.operator_attack.profile.payload["wisdel_component"])

    def wisdel_targets(self, position, radius, *, ground=True):
        return [enemy for enemy in self.b.enemies if self.b._enemy_is_attackable(enemy)
                and self.b._enemy_targetable(enemy)
                and (not ground or not self.b._is_airborne(enemy))
                and self.b._enemy_intersects_circle(enemy, position, radius)]

    def clear_invalid_remnants(self, *, force=False):
        for key, enemy in list(self.remnants.items()):
            if force or self.op.dead or not self.b._enemy_is_attackable(enemy):
                if enemy.wisdel_remnant_source is self.op:
                    enemy.wisdel_remnant_source = None
                self.remnants.pop(key, None)

    def attach_wisdel_remnant(self, enemy):
        if self.op.dead or not self.talent(1):
            return
        previous = enemy.wisdel_remnant_source
        if previous is not None and previous.skill_runtime is not None:
            previous.skill_runtime.remnants.pop(enemy.enemy_index, None)
        enemy.wisdel_remnant_source = self.op
        self.remnants[enemy.enemy_index] = enemy

    def trigger_wisdel_remnant(self, enemy):
        # The saved template rolls first, then checks the bomb buff. The
        # only-once flag gates regular buffs, not the stop-hit extra template.
        success = self.b._roll_probability(float(self.talent(1).get("attack@prob", 0)),
            source=self.op, target=enemy, mechanism="wisdel_remnant")
        owner = enemy.wisdel_remnant_source
        if not success or owner is None or owner.dead:
            return
        position = enemy.position()
        radius = float(self.talent(1).get("attack@range_radius",
            self.b.attack_timing.get(self.op.char_id, {}).get("wisdel_remnant_radius", 1.1)))
        amount = self.b.operator_attack_power(self.op) * float(self.talent(1).get("attack@bomb_atk_scale", 0))
        for target in self.wisdel_targets(position, radius, ground=False):
            self.damage(target, amount, "PHYSICAL", attack_type=AttackType.NORMAL)
            target.statuses.apply(Status.STUNNED, float(self.talent(1).get("attack@stun", 0)))
        if owner.skill_runtime is not None:
            owner.skill_runtime.remnants.pop(enemy.enemy_index, None)
        enemy.wisdel_remnant_source = None
        self.b.mechanic_diagnostics["WISDEL_REMNANT_DAMAGE_AND_DEATH_ORDER_CANDIDATE"] = 1

    def wisdel_reach_packet(self, position, packet):
        if not packet["spec"]["hit_on_reach"]:
            return
        for enemy in self.wisdel_targets(position, packet["spec"]["radius"]):
            if packet["component"] == "MAIN":
                if enemy.enemy_index == packet["primary"]:
                    self.attach_wisdel_remnant(enemy)
                scale = packet["main_scale"] * (packet["trace_scale"]
                    if enemy.enemy_index == packet["primary"] else 1)
            else:
                # The selected final carrier has last_atk_scale=0/missing and
                # carries the S1 stun. It is independent of the damage packet.
                enemy.statuses.apply(Status.STUNNED, self.value("stun_duration"))
                scale = self.value("last_atk_scale", default=0)
            self.damage(enemy, packet["base"]*scale, "PHYSICAL",
                attack_type=AttackType.NORMAL, apply_way=ApplyWay.RANGED)
        self.clear_invalid_remnants()

    def wisdel_stop_packet(self, position, packet):
        for _ in packet["stop_gates"]:
            for enemy in self.wisdel_targets(position, packet["spec"]["radius"]):
                self.trigger_wisdel_remnant(enemy)
                if not self.b._enemy_is_attackable(enemy):
                    continue
                scale = packet["append_scale"] * (packet["trace_scale"]
                    if enemy.enemy_index == packet["primary"] else 1)
                self.damage(enemy, packet["base"]*scale, "PHYSICAL",
                    attack_type=AttackType.NORMAL, apply_way=ApplyWay.RANGED)
        self.clear_invalid_remnants()

    def land_wisdel_without_projectiles(self, attack):
        # Data-free callers retain a bounded immediate-flight fallback, with
        # the same reached/stop effects and configured stop delays.
        base = self.b.operator_attack_power(self.op, attack.source_atk_scale)
        for primary in attack.targets:
            if not self.b._enemy_is_attackable(primary):
                continue
            specs = self.wisdel_specs(attack.profile.skill)
            if attack.profile.skill and len(specs) == 1:
                specs += [("SHOCK", dict(key=key, radius=1.1, stop_delay=delay,
                    stop_gates=[""], hit_on_reach=False, keep_after_departure=True))
                    for key, delay in [("projectile_chr_wisdel_s1_shock", .5),
                                       ("projectile_chr_wisdel_s1_shock2", .65)]]
                specs += [("STUN", dict(key="projectile_chr_wisdel_s1_bomb", radius=1.1,
                    stop_delay=.15, stop_gates=[], hit_on_reach=True, keep_after_departure=True))]
            for component, spec in specs:
                packet = self.wisdel_packet_data(replace(attack, primary=primary), component, spec, base)
                self.wisdel_reach_packet(primary.position(), packet)
                if packet["stop_gates"]:
                    self.events.append((self.b.time+spec["stop_delay"], "AFTERSHOCK", primary.position(), packet))
        self.b.mechanic_diagnostics["WISDEL_NO_PROJECTILE_DATA_FALLBACK"] = 1

    def land(self, attack, *, target=None, ranged=False, projectile_id=None):
        b, op, p = self.b, self.op, attack.profile
        if target is None and p.skill and self.sid == "skchr_halo2_2":
            for primary in attack.targets:
                if b._enemy_is_attackable(primary):
                    self.emit_halo2_links(replace(attack, primary=primary), (*primary.position(), 0.0))
        if not attack.will_hit:
            return
        if p.skill and self.sid == "skchr_skybx_1":
            for primary in ([target] if target is not None else attack.targets):
                self.land_skybx_splash(attack, primary.position(),
                    b.operator_attack_power(op, attack.source_atk_scale) * p.scale)
            b.mechanic_diagnostics["SKYBX_SKILL_NO_PROJECTILE_DATA_FALLBACK"] = 1
            return
        if p.payload.get("end_volley"):
            if target is not None and b._enemy_is_attackable(target):
                self.damage(target, p.payload["cached_amount"], "PHYSICAL",
                            attack_type=AttackType.NORMAL, apply_way=ApplyWay.RANGED)
                if not target.dead and p.payload["end_volley"] == "COIN":
                    b.apply_push(target, target.position_row - op.tile[0],
                                 target.position_col - op.tile[1], int(self.value("force")))
            return
        if op.char_id == "char_1035_wisdel":
            self.land_wisdel_without_projectiles(attack)
            return
        originals = [target] if target is not None else attack.targets
        if target is None and p.skill and self.sid == "skchr_etlchi_1":
            originals = self.candidates()  # selector runs just before spell start
        amount = b.operator_attack_power(op, attack.source_atk_scale)
        landed_hits = 0
        for primary in originals:
            if not b._enemy_is_attackable(primary):
                continue
            targets = self.near(primary.position(), p.radius) if p.radius else [primary]
            for _ in range(p.hits):
                if op.dead and not (ranged and p.payload.get("keep_projectile_after_departure")):
                    break
                for enemy in targets:
                    if not b._enemy_is_attackable(enemy):
                        continue
                    scale = p.scale
                    if (op.char_id == "char_103_angel" and op.module_id == "uniequip_002_angel"
                            and op.module_level >= 1 and b._is_airborne(enemy)):
                        scale *= float(op.trait_blackboard.get("atk_scale", 1))
                    if op.char_id == "char_474_glady" and enemy.mass_level <= float(self.talent(2).get("value", -1)):
                        scale *= float(self.talent(2).get("atk_scale", 1))
                    if op.char_id == "char_172_svrash" and op.trait_blackboard.get("atk_scale_m"):
                        elite_scale = float(self.talent(10).get("atk_scale", 1)) if enemy.level_type in {"ELITE", "BOSS"} else 1.0
                        extra_scale = self.value("atk_scale", default=1) if p.skill else 1.0
                        self.damage(enemy, b.operator_attack_power(op) * extra_scale * elite_scale * float(op.trait_blackboard["atk_scale_m"]),
                                    "ARTS", attack_type=AttackType.ADDITION, modifier_key="svrash_e_002_tr")
                        b.mechanic_diagnostics["SVRASH_MODULE_CALCULATE_ORDER_CANDIDATE"] = 1
                        if enemy.dead:
                            continue
                    result = self.damage(enemy, amount * scale, p.kind,
                                         attack_type=AttackType.NORMAL,
                                         apply_way=ApplyWay.RANGED if ranged else ApplyWay.MELEE)
                    landed_hits += int(not result.cancelled and not result.evaded)
                    self.hit_effects(enemy, primary, amount, p, result)
                    if op.char_id == "char_4193_lemuen":
                        # ATTACK/COMBAT attachments follow the projectile hit,
                        # independently of the damage packet's result flags.
                        b._queue_lemuen_survivor_sp(op, enemy, float(op.trait_blackboard.get("sp", 0)))
            if p.skill:
                hit_profile = (replace(p, payload={**p.payload, "pull_source_key": projectile_id})
                    if self.sid == "skchr_glady_1" and projectile_id is not None else p)
                self.skill_hit(primary, amount, hit_profile)
        if op.subprofession == "reaper" and op.char_id != "char_4010_etlchi":
            count = min(landed_hits, op.block_count * p.hits)
            self.regen(op, float(op.trait_blackboard.get("value", 50)) * count)

    def damage(self, target, amount, kind="ARTS", *, attack_type=AttackType.BUFF, apply_way=ApplyWay.NONE,
               modifier_key="", ability_owner=...):
        return self.b._apply_operator_damage(self.op, target, amount,
            attack_type=attack_type, apply_way=apply_way, damage_type=kind,
            modifier_key=modifier_key, ability_owner=ability_owner)

    def near(self, pos, radius):
        return [e for e in self.b.enemies if self.b._enemy_is_attackable(e)
                and e.statuses.targetable and math.dist(e.position(), pos) <= radius + 1e-9]

    def hit_effects(self, enemy, primary, amount, profile, result):
        op, b = self.op, self.b
        if op.char_id == "char_1042_phatm2" and self.talent(1):
            # PassiveAttachment runs separately from the main damage result.
            # Its circle buff excludes the attack target, including lethal hits.
            t = self.talent(1)
            radius = self.value("talent@range_radius", default=float(t.get("range_radius", 1.3))) if self.active else float(t.get("range_radius", 1.3))
            splash_atk = b.operator_attack_power(op) if not op.dead else amount
            for other in self.near(enemy.position(), radius):
                if other is not enemy:
                    b.apply_element_damage(other, ElementType.SANITY,
                        splash_atk * float(t.get("ep_damage_ratio", 0)), source=op)
            b.mechanic_diagnostics["PHATM2_ELEMENT_SPLASH_ATTRIBUTE_PHASE_CANDIDATE"] = 1
        if result.cancelled or result.evaded:
            return
        if self.sid == "skchr_whitw2_3" and not self.active and enemy is primary:
            if self.funnel_target is not enemy:
                self.funnel_target = enemy
                self.funnel_hits = 0
            interval = float(self.talent(1).get("interval", 20))
            elapsed = b.time - op.deployed_at
            for _ in range(1 + (elapsed >= interval * 3)):
                self.damage(enemy, amount * self.funnel_scale(self.funnel_hits), "ARTS", attack_type=AttackType.ADDITION)
                if elapsed >= interval * 2:
                    enemy.statuses.apply(Status.SILENCE, float(self.talent(1).get("attack@silence_duration", 2)))
            self.funnel_hits += 1
        if op.subprofession == "slower":
            if self.sid != "skchr_halo2_2" or not profile.skill:
                self.apply_sluggish(enemy, float(op.trait_blackboard.get("sluggish", .8)))
        if op.char_id == "char_1042_phatm2":
            t = self.talent(1)
            multiplier = self.value("ep_damage_scale", default=1) if profile.skill and self.sid.endswith("_1") else 1
            b.apply_element_damage(enemy, ElementType.SANITY, amount * float(t.get("attack@ep_damage_ratio", 0)) * multiplier, source=op)
        if result.killed and self.sid == "skchr_swire2_3" and self.active:
            self.add_coins(1)

    def attack_emitted(self, attack):
        if self.sid == "skchr_glady_1" and attack.profile.skill:
            timing = self.op.skill_attack_timing.get(self.sid, {})
            offset = float(timing.get("pull_source_offset", .5))
            direction = FACING[self.op.facing % 4]
            attack.profile.payload["pull_origin"] = tuple(
                self.op.tile[i] + offset * direction[i] for i in (0, 1))
            self.glady_emitted = True
            self.glady_anim_end = self.op.graphic_started_at + float(
                timing.get("attack_duration", self.op.attack_duration)) / max(self.op.graphic_rate, 1e-9)
        if self.op.char_id != "char_4133_logos":
            return
        for primary in attack.targets:
            if not self.b._enemy_is_attackable(primary):
                continue
            self.try_logos_talent(primary)

    def try_logos_talent(self, primary):
        from .battle import _PendingOperatorAttack
        b, op, talent = self.b, self.op, self.talent(1)
        if not talent or not b._roll_probability(float(talent.get("prob", 0)),
                mechanism="logos_extra_hit", source=op, target=primary):
            return
        base_atk = b.operator_attack_power(op)
        amount = base_atk * float(talent.get("atk_scale", 0))
        for _ in range(max(1, int(talent.get("emit_count", 1)))):
            extra = b.choose_random_target(self.candidates(), source=op, mechanism="logos_extra_target")
            if extra is None:
                break
            payload = _PendingOperatorAttack(b.time, b.frame_index, "LOGOS_TALENT", extra, [extra],
                1.0, op.atk_scale, True, True,
                profile=AttackProfile(payload={"keep_projectile_after_departure": True,
                    "sluggish": float(talent.get("sluggish", 0)),
                    "element_amount": base_atk * float(talent.get("element_atk_scale", 0))}))
            b._new_projectile(key="projectile_chr_logos_t", side="OPERATOR", source=op,
                              target=extra, operator_attack=payload, cached_atk=amount)
        b.mechanic_diagnostics["LOGOS_TALENT_EMISSION_PHASE_CANDIDATE"] = 1
        b.mechanic_diagnostics["LOGOS_TALENT_ARC_AND_BORN_DELAY_CANDIDATE"] = 1

    def land_talent_projectile(self, projectile):
        op, target = self.op, projectile.target
        amount = projectile.cached_atk
        if amount is None or not self.b._enemy_is_attackable(target):
            return
        payload = projectile.operator_attack.profile.payload
        self.damage(target, amount, attack_type=AttackType.NORMAL, apply_way=ApplyWay.RANGED)
        if (op.module_id == "uniequip_002_logos" and target.elements.active_break is ElementType.DARK
                and not target.dead):
            self.damage(target, payload["element_amount"], "ELEMENTAL", attack_type=AttackType.ADDITION)
        if not target.dead:
            target.statuses.apply(Status.SLUGGISH, payload["sluggish"])

    def skill_hit(self, enemy, amount, profile):
        b, op = self.b, self.op
        if self.sid in {"skchr_glady_1", "skchr_ulpia_1"}:
            if self.sid == "skchr_glady_1":
                self.glady_landed = True
            force = int(self.value("force"))
            if op.char_id == "char_474_glady":
                t = self.talent(10)
                if math.dist(op.tile, enemy.position()) > float(t.get("skill@range_radius", math.inf)):
                    force += int(t.get("skill@delta_force", 0))
            origin = profile.payload.get("pull_origin", op.tile)
            pulled = b.apply_pull(enemy, *origin, force,
                source_key=profile.payload.get("pull_source_key"),
                interrupt_source=op if op.char_id == "char_474_glady" else None,
                stop_radius=.55 if op.char_id == "char_474_glady" else None,
                stop_within_source_block_range=op.char_id == "char_474_glady")
            if pulled and op.char_id == "char_474_glady":
                if op.trait_blackboard.get("value"):
                    self.pull_damage[enemy.enemy_index] = {"unit":enemy, "pos":enemy.position(),
                        "distance":0.0, "next":b.time + float(op.trait_blackboard.get("interval", .066))}
        elif self.sid == "skchr_phatm2_1":
            enemy.statuses.apply(Status.BIND, self.value("unmove"))
        elif self.sid == "skchr_ines_1":
            b.dp = min(b.max_cost, b.dp + self.value("cost"))
            self.add_dot(enemy, "INES", self.value("bleed_duration"), amount * self.value("bleed_atk_scale"))
        elif self.sid == "skchr_cello_1":
            b.apply_element_damage(enemy, ElementType.DARK, amount * self.value("ep_damage_ratio"), source=op)
        elif self.sid == "skchr_halo2_2":
            self.apply_sluggish(enemy, self.value("sluggish"))
        elif self.sid == "skchr_wisdel_1":
            for target in self.near(enemy.position(), profile.radius):
                target.statuses.apply(Status.STUNNED, self.value("stun_duration"))

    def attack_complete(self):
        if self.armed and self.sid == "skchr_horn_1":
            self.armed = False
            return
        if self.sid == "skchr_hsgma2_2":
            self.armed = False
        if self.armed and self.sid in {"skchr_skybx_1", "skchr_glady_1"}:
            self.armed = False
            if self.sid == "skchr_skybx_1":
                timing = self.op.skill_attack_timing.get(self.sid, {})
                self.return_at = self.op.graphic_started_at + float(timing.get("attack_duration", 1))
                self.action_end = self.return_at + float(timing.get("return_duration", 1))
                self.b.mechanic_diagnostics["SKYBX_SEQUENTIAL_ACTION_HANDOFF_CANDIDATE"] = 1
            return
        if self.armed and self.sid != "skchr_hsgma2_2":
            self.b._finish_skill(self.op, "ATTACK_COMPLETE")
        if self.sid in AMMO and self.op.ammo is not None and self.op.ammo <= 0:
            self.b._finish_skill(self.op, "AMMO_EXHAUSTED")

    def finish(self, reason):
        op, b = self.op, self.b
        was_active = self.active
        manual = reason in {"MANUAL_END", "SKILL_END"}
        if self.sid == "skchr_swire2_3" and manual and was_active:
            self.begin_coin_end()
        if self.sid == "skchr_angel2_1" and manual and was_active:
            self.begin_ammo_end()
        self.armed = self.active = False
        self.action_end = -1.0
        self.return_at = -1.0
        self.glady_flight_target = None
        self.glady_post_until = -1.0
        self.glady_emitted = False
        self.glady_anim_end = -1.0
        self.heal_spent = 0.0
        self.skill_spent = 0.0
        self.next_continuous_hit = math.inf
        self.logos_execute_checks.clear()
        self.agoat_aura_checks.clear()
        self.interval_delta = 0.0
        self.attack_speed_delta -= self.value("attack_speed") if was_active and self.sid in PERSISTENT | TIMED | AMMO else 0
        op.block_count -= self.block_delta
        self.block_delta = 0
        self.hp_bonus = 0.0
        self.drones.clear()
        if self.sid == "skchr_hsgma2_2":
            self.zones = [zone for zone in self.zones if zone["kind"] != "SHIELD_ORBIT"]
            self.events = [event for event in self.events if event[1] != "SHIELD_LAUNCH"]
        if self.sid == "skchr_skybx_1":
            op.air_blocking = True
        if self.sid == "skchr_kalts2_3":
            self.anchor_ready = False
        b._squad_field_signature = None

    def add_dot(self, target, key, duration, amount, *, stacks=1, current_atk=False):
        ident = (key, target.enemy_index)
        old = self.dots.get(ident)
        expiries = [v for v in old[1] if v > self.b.time] if old else []
        if key == "ASCLN":
            if not expiries:
                self.dot_next.pop(ident, None)
            expiries = self.refreshed_stack_expiries(expiries, duration, stacks)
        else:
            expiries.append(self.b.time + duration)
            expiries = expiries[-stacks:]
        self.dots[ident] = (target, expiries, amount, current_atk)
        interval = float(self.talent(1).get("interval", 1)) if key == "ASCLN" else 1.0
        self.dot_next.setdefault(ident, self.b.time + max(self.b.dt, interval))

    def refreshed_stack_expiries(self, expiries, duration, maximum):
        valid = [v for v in expiries if v > self.b.time]
        until = max([self.b.time + duration, *valid])
        return [until] * min(len(valid) + 1, max(0, int(maximum)))

    def add_zone(self, pos, duration, kind, *, radius=0, cells=None):
        self.zones.append({"pos": tuple(pos), "end": self.b.time + duration,
                           "kind": kind, "radius": radius, "cells": cells,
                           "next": self.b.time + (self.value("interval", default=.5) if kind == "SHIELD_ORBIT" else 1),
                           "born": self.b.time})
        if kind == "ALCHEMY":
            self.zones[-1]["regen"] = self.b.operator_attack_power(self.op) * self.value("hp_recovery_per_sec_ratio")

    @staticmethod
    def regen(target, amount):
        if not target.dead:
            target.hp = min(target.max_hp, target.hp + max(0, amount))

    def tick(self):
        b, op = self.b, self.op
        if op.dead:
            return
        if op.char_id == "char_1035_wisdel":
            self.tick_soul_wards()
        self.check_low_hp_recovery()
        if op.char_id == "char_1044_hsgma2":
            op.negative_hp_heal_free = op.hp <= 0 and bool(self.talent(1))
        self.tick_wanted_targets()
        self.tick_abyssal_sharing()
        self.tick_servant_queue()
        self.tick_reaper_heals()
        self.tick_module_enemy_count()
        self.tick_module_low_hp()
        self.tick_module_element_count()
        self.tick_module_pre_reborn()
        if self.sid == "skchr_skybx_1" and self.return_at >= 0 and b.time+1e-9 >= self.return_at:
            op.graphic_animation = op.skill_attack_timing.get(self.sid, {}).get("return_anim", "Skill_1_End")
            op.graphic_started_at = self.return_at
            op.graphic_rate = 1.0
            self.return_at = -1.0
        if self.sid == "skchr_glady_1" and self.active and not self.armed:
            target = self.glady_flight_target
            in_flight = any(p.source is op and p.key == "projectile_chr_glady_s1" for p in b.projectiles)
            pull_finished = (target is None or target.dead or target.disappeared
                             or not self.glady_landed or not target.unbalanced)
            if self.glady_emitted and not in_flight and pull_finished:
                if self.glady_post_until < 0:
                    self.glady_post_until = b.time + float(op.skill_attack_timing.get(self.sid, {}).get("min_post_delay_after_pull", 1))
                if b.time+1e-9 >= max(self.glady_post_until, self.glady_anim_end):
                    b._finish_skill(op, "PULL_COMPLETE")
            b.mechanic_diagnostics["GLADY_CHAIN_FLIGHT_RETURN_HANDOFF_CANDIDATE"] = 1
        if self.sid == "skchr_halo2_2":
            self.tick_weakness_aura()
        if self.action_end >= 0 and b.time + 1e-9 >= self.action_end and op.pending_attack is None:
            b._finish_skill(op, "ATTACK_COMPLETE" if self.sid in NEXT else "HEAL_ACTION_COMPLETE")
        if self.sid == "skchr_sbell2_2":
            self.tick_snow()
        if self.sid == "skchr_whitw2_3":
            self.tick_drones()
        if self.sid == "skchr_swire2_3":
            while b.time + 1e-9 >= self.upkeep:
                self.upkeep += max(b.dt, float(op.trait_blackboard.get("interval", 3)))
                fee = abs(float(op.trait_blackboard.get("cost", -3))) * float(op.trait_blackboard.get("cost_scale", 1))
                if b.dp < fee:
                    op.dead = op.retreated = True
                    b._process_operator_departure(op, retreated=False)
                    return
                b.dp -= fee
                self.cost_spent(fee)
                t = op.talent_blackboards.get("10_root", {})
                self.extra_upkeep_stacks = min(int(t.get("max_stack_cnt", 0)), self.extra_upkeep_stacks + 1)
                if self.active:
                    self.add_coins(int(self.talent(1).get("trait_sp", 0)), upkeep=True)
        if self.sid == "skchr_lemuen_1" and not self.lemuen_bonus and b.time - op.deployed_at >= float(self.talent(2).get("interval", math.inf)):
            self.lemuen_bonus = True
        self.tick_events()
        self.tick_end_volley()
        self.tick_ammo_bombs()
        self.tick_pull_damage()
        self.tick_element_bonus()
        self.tick_cello_talent()
        self.tick_agoat_element_aura()
        self.tick_alchemy_flights()
        self.tick_zones()
        self.tick_dots()
        self.tick_lure()

        self.tick_anchor()
        if self.sid == "skchr_kalts2_3":
            self.tick_medical_field()
        if self.sid == "skchr_hsguma_3" and self.active:
            interval = float(op.skill_attack_timing.get(self.sid, {}).get("continuous_interval", 1))
            while b.time + 1e-9 >= self.next_continuous_hit:
                self.next_continuous_hit += max(interval, b.dt)
                if op.statuses.can_attack and not b._skill_starting(op):
                    dr, dc = FACING[op.facing % 4]
                    cells = {op.tile, (op.tile[0] + dr, op.tile[1] + dc)}
                    for enemy in self.candidates(cells=cells, ground=True):
                        self.damage(enemy, b.operator_attack_power(op) * self.value("atk_scale", default=1),
                                    "PHYSICAL", attack_type=AttackType.NORMAL)
        if b.time + 1e-9 >= self.next_pulse:
            self.next_pulse += 1.0
            self.pulse()
        self.tick_module_high_hp()
        self.tick_logos_execute()

    def tick_module_enemy_count(self):
        if (self.op.dead or self.op.module_id not in {"uniequip_003_etlchi", "uniequip_002_oblvns"}
                or self.enemy_count_interval <= 0):
            return
        if self.b.time + 1e-9 < self.next_enemy_count_check:
            return
        while self.b.time + 1e-9 >= self.next_enemy_count_check:
            self.next_enemy_count_check += max(self.b.dt, self.enemy_count_interval)
        count = sum(self.b._enemy_is_attackable(e)
                    and self.b._enemy_intersects_range(e, self.op.range_cells) for e in self.b.enemies)
        threshold = int(self.op.trait_blackboard.get("cnt", 2)) if self.op.module_id == "uniequip_003_etlchi" else 2
        self.enemy_count_bonus = count >= threshold
        key = "ETLCHI_ENEMY_COUNT_CHECK_PHASE_CANDIDATE" if self.op.module_id == "uniequip_003_etlchi" else "OBLVNS_MODULE_ENEMY_COUNT_CHECK_PHASE_CANDIDATE"
        self.b.mechanic_diagnostics[key] = 1

    def tick_module_low_hp(self):
        if (self.op.dead or self.op.char_id != "char_1044_hsgma2"
                or self.low_hp_interval <= 0 or not self.talent(2).get("atk")):
            return
        if self.b.time + 1e-9 < self.next_low_hp_check:
            return
        while self.b.time + 1e-9 >= self.next_low_hp_check:
            self.next_low_hp_check += max(self.b.dt, self.low_hp_interval)
        self.low_hp_bonus = self.op.hp / self.op.max_hp <= float(self.talent(2).get("min_hp_ratio", .3))
        self.b.mechanic_diagnostics["HSGMA_MODULE_LOW_HP_CHECK_PHASE_CANDIDATE"] = 1

    def tick_module_element_count(self):
        if (self.op.dead or self.op.char_id != "char_1016_agoat2"
                or self.ep_check_interval <= 0 or self.b.time + 1e-9 < self.ep_aura_ready_at):
            return
        units = [p for p in self.b.operators if not p.dead and p.tile in self.op.range_cells]
        present = {p.deployment_index for p in units}
        self.ep_aura_members = {i: entry for i, entry in self.ep_aura_members.items() if i in present}
        for unit in units:
            entry = self.ep_aura_members.get(unit.deployment_index)
            if entry is not None and self.b.time + 1e-9 < entry[1]:
                continue
            due = self.b.time if entry is None else entry[1]
            while self.b.time + 1e-9 >= due:
                due += max(self.b.dt, self.ep_check_interval)
            damaged = (unit.elements.cooldown_remaining > 0
                       or unit.elements.current_damaged_element is not None)
            # ON_BUFF_START forces an entry check; subsequent clocks belong to recipients.
            self.ep_aura_members[unit.deployment_index] = (unit, due, damaged)
        self.b.mechanic_diagnostics["AGOAT_MODULE_ELEMENT_AURA_PHASE_CANDIDATE"] = 1

    def tick_module_pre_reborn(self):
        if (self.op.dead or self.op.char_id != "char_4039_horn"
                or self.pre_reborn_interval <= 0 or not self.talent(11)):
            return
        if self.b.time + 1e-9 < self.next_pre_reborn_check:
            return
        if self.revive_count:
            # Finishing the checker also finishes its derived attribute buff.
            self.pre_reborn_bonus = False
            self.next_pre_reborn_check = math.inf
        else:
            self.pre_reborn_bonus = True
            while self.b.time + 1e-9 >= self.next_pre_reborn_check:
                self.next_pre_reborn_check += max(self.b.dt, self.pre_reborn_interval)
        self.b.mechanic_diagnostics["HORN_PRE_REBORN_CHECK_PHASE_CANDIDATE"] = 1

    def has_pre_reborn_bonus(self):
        if self.op.dead:
            return False
        if self.pre_reborn_interval > 0:
            return self.pre_reborn_bonus
        if self.talent(11):
            self.b.mechanic_diagnostics["HORN_PRE_REBORN_TIMING_MISSING"] = 1
        return not self.revive_count

    def tick_module_high_hp(self):
        if self.op.dead or self.op.char_id != "char_1041_angel2" or self.high_hp_interval <= 0:
            return
        if self.b.time + 1e-9 < self.next_high_hp_check:
            return
        while self.b.time + 1e-9 >= self.next_high_hp_check:
            self.next_high_hp_check += max(self.b.dt, self.high_hp_interval)
        self.high_hp_bonus = self.op.hp >= self.op.max_hp * float(
            self.op.trait_blackboard.get("angel2_tr[e].hp_ratio", math.inf))
        self.b.mechanic_diagnostics["ANGEL2_HIGH_HP_CHECK_PHASE_CANDIDATE"] = 1

    def remove_element_aura_member(self, unit):
        self.ep_aura_members.pop(unit.deployment_index, None)

    def tick_logos_execute(self):
        if self.sid != "skchr_logos_1" or not self.active or self.op.dead:
            return
        from .battle import _PendingOperatorAttack
        b, op = self.b, self.op
        targets = self.candidates()
        present = {enemy.enemy_index for enemy in targets}
        self.logos_execute_checks = {i: due for i, due in self.logos_execute_checks.items() if i in present}
        interval = max(b.dt, float(b.attack_timing.get(op.char_id, {}).get(
            "execute", {}).get("interval", .1)))
        for enemy in targets:
            due = self.logos_execute_checks.setdefault(enemy.enemy_index, b.time)
            if b.time+1e-9 < due:
                continue
            self.logos_execute_checks[enemy.enemy_index] = due+interval
            if enemy.hp >= b.operator_attack_power(op) * self.value("attack@kill_atk_scale"):
                continue
            hp = enemy.hp
            origin = b._unit_hit_position(enemy) or (*enemy.position(), 0.0)
            # NoSourceDamage has no character output modifier or kill ownership.
            result = b._apply_damage(DamagePacket(self.value("attack@kill_damage"), "TRUE",
                attack_type=AttackType.NORMAL, ignore_evasion=True), enemy)
            if not result.killed:
                continue
            self.try_logos_talent(enemy)  # ForceTrigger still executes the talent's Dice node.
            options = [e for e in b.enemies if e is not enemy and b._enemy_is_attackable(e)
                       and b._enemy_targetable(e) and not e.statuses.has(Status.CAMOUFLAGE)]
            target = b.choose_random_target(options, source=op, mechanism="logos_execute_transfer")
            if target is not None:
                payload = _PendingOperatorAttack(b.time, b.frame_index, "LOGOS_TRANSFER", target, [target],
                    1.0, 1.0, True, True,
                    profile=AttackProfile(payload={"keep_projectile_after_departure": True}))
                b._new_projectile(key="projectile_chr_logos_s1_2", side="OPERATOR", source=op,
                    target=target, operator_attack=payload, cached_atk=hp, mount_point="HIT",
                    origin_override=origin)
            b.mechanic_diagnostics["LOGOS_EXECUTE_NOTIFICATIONS_AND_SELECTOR_CANDIDATE"] = 1

    def passive_hp_recovery_for(self, target):
        op, b = self.op, self.b
        if target.dead:
            return 0.0
        # These finite recipient/tile buffs survive their source's departure.
        amount = sum(float(zone["regen"]) for zone in self.zones
                     if zone["kind"] == "ALCHEMY" and b.time <= zone["end"] + 1e-9
                     and target.tile in zone["cells"])
        medical = self.medical_regen.get(target.deployment_index)
        if medical is not None and medical[0] is target and b.time < medical[1]:
            amount += medical[2]
        if op.dead:
            return max(0.0, amount)
        ratio = 0.0
        if op.char_id == "char_474_glady" and self.has_group_tag(target, "abyssal"):
            ratio = float(self.talent(1).get("hp_recovery_per_sec_by_max_hp_ratio", 0))
        elif op.char_id == "char_1045_svash2" and self.sjerag_ally(target):
            t = self.talent(2)
            mult = 2 if b.time-op.deployed_at >= float(t.get("interval", math.inf)) else 1
            ratio = float(t.get("hp_recovery_per_sec_by_max_hp_ratio", 0)) * mult
        elif op.char_id == "char_1044_hsgma2" and target is op and op.hp <= 0:
            t = self.talent(1)
            if t and b.time-self.last_damage >= float(t.get("hsgma2_t_1[heal].interval", math.inf)):
                ratio = float(t.get("hsgma2_t_1[heal].hp_recovery_per_sec_by_max_hp_ratio", 0))
        elif op.char_id == "char_4010_etlchi" and target is op and any(
                key[0] == "ETLCHI" and not enemy.dead and any(t > b.time for t in expiries)
                for key, (enemy, expiries, _, _) in self.dots.items()):
            ratio = float(self.talent(1).get("hp_recovery_per_sec_by_max_hp_ratio", 0))
        return max(0.0, amount + target.max_hp * max(0.0, ratio))

    def pulse(self):
        b, op = self.b, self.op
        if op.char_id == "char_1041_angel2":
            b._apply_damage(DamagePacket(op.max_hp * float(op.trait_blackboard.get("hp_ratio", .01)),
                                        DamageType.LIFE_LOSS, source=op, lethal=False), op)

    def begin_coin_end(self):
        b, op = self.b, self.op
        timing = op.skill_attack_timing.get(self.sid, {}).get("coin_end", {})
        count = self.coins
        cells = b.range_cells_from_id(timing.get("range_id", "2-4"), op.tile, op.facing)
        marked = self.candidates(cells=cells, ground=True)
        marked += [e for e in op.blocked if e not in marked and b._enemy_is_attackable(e)
                   and not b._is_airborne(e)]
        hit = float(timing.get("hit_time", .4333333373069763))
        delta = float(timing.get("interval", .07000000029802322))
        duration = max(float(timing.get("duration", 1.3333333730697632)),
                       hit + delta * max(0, count - 1))
        self.end_volley = dict(kind="COIN", targets=marked, remaining=count, next=b.time + hit,
                             interval=delta, until=b.time + duration)
        self.coins = 0
        op.pending_attack = None
        b._finish_operator_attack_chain(op)
        op.graphic_animation = timing.get("anim_key", "Skill_3_End")
        op.graphic_started_at = b.time
        op.graphic_rate = 1.0
        b.mechanic_diagnostics["SWIRE_COIN_END_PHASE_AND_MARK_CANDIDATE"] = 1

    def begin_ammo_end(self):
        b, op = self.b, self.op
        timing = op.skill_attack_timing.get(self.sid, {}).get("ammo_end", {})
        count = int(max(0, op.ammo or 0))
        hit = float(timing.get("hit_time", .2666666805744171))
        delta = float(timing.get("interval", .05000000074505806))
        self.end_volley = dict(kind="AMMO", remaining=count, next=b.time + hit,
            interval=delta, until=b.time + max(float(timing.get("duration", 1.3333333730697632)),
                                              hit + delta * max(0, count - 1)))
        self.end_volley["attack_scale"] = op.atk_scale
        self.end_volley["cells"] = set(op.range_cells)
        op.pending_attack = None
        b._finish_operator_attack_chain(op)
        op.graphic_animation = timing.get("anim_key", "Skill_1_Close")
        op.graphic_started_at, op.graphic_rate = b.time, 1.0
        # Freeze the count before the public skill lifecycle drops its ammo.
        if op.ammo is not None:
            op.ammo = 0
        b.mechanic_diagnostics["ANGEL2_AMMO_END_PHASE_CANDIDATE"] = 1

    def finish_end_volley(self):
        if self.end_volley["kind"] == "COIN":
            self.op.sp = 0.0  # Also required when the end action is interrupted.
        self.op.graphic_animation = "Idle"
        self.op.graphic_started_at = self.b.time
        self.end_volley = None

    def tick_end_volley(self):
        end = self.end_volley
        if end is None:
            return
        b, op = self.b, self.op
        if op.dead or not op.statuses.can_attack:
            self.finish_end_volley()
            return
        from .battle import _PendingOperatorAttack
        while end["remaining"] > 0 and b.time + 1e-9 >= end["next"]:
            coin = end["kind"] == "COIN"
            targets = ([e for e in end["targets"] if b._enemy_is_attackable(e)
                        and b._enemy_targetable(e) and not e.statuses.has(Status.CAMOUFLAGE)]
                       if coin else self.candidates(cells=end["cells"]))
            target = b.choose_random_target(targets, source=op,
                mechanism="swire_coin_target" if coin else "angel2_remaining_ammo")
            end["remaining"] -= 1
            end["next"] += end["interval"]
            if target is not None:
                amount = (b.operator_attack_power(op) * self.value("atk_scale") if coin else
                          b.operator_attack_power(op, end["attack_scale"]) * self.value("attack@atk_scale"))
                attack = _PendingOperatorAttack(b.time, b.frame_index, "ROSTER", target, [target],
                    1, op.atk_scale, False, b._attack_hits(op, "PHYSICAL"),
                    profile=AttackProfile(payload={"end_volley": end["kind"],
                        "cached_amount": amount, "keep_projectile_after_departure": True}))
                key = "projectile_chr_swire2_s_3" if coin else (
                    "projectile_chr_angel2" if not end.get("emitted") else "projectile_chr_angel2_s1_close")
                b._new_projectile(key=key, side="OPERATOR", source=op,
                    target=target, operator_attack=attack, cached_atk=amount, will_hit=attack.will_hit)
            if not coin:
                end["emitted"] = True
                for runtime in live_runtimes(b):
                    runtime.ammo_used(op, 1)
        if b.time + 1e-9 >= end["until"] and end["remaining"] == 0:
            self.finish_end_volley()

    def tick_events(self):
        self.clear_invalid_remnants()
        pending = []
        for time, kind, target, amount in self.events:
            if time > self.b.time + 1e-9:
                pending.append((time, kind, target, amount)); continue
            if kind == "AFTERSHOCK":
                self.wisdel_stop_packet(target, amount)
            elif kind == "RESTORE_RES":
                if self.b.time + 1e-9 >= getattr(target, "logos_core_until", time):
                    target.roster_res_delta = 0.0
                    target.roster_arts_addition = 0.0
            elif kind == "AMMO_BOMB_EMIT" and not self.op.dead:
                candidates = [e for e in target if self.b._enemy_is_attackable(e)
                              and self.b._enemy_targetable(e) and not e.statuses.has(Status.CAMOUFLAGE)]
                chosen = self.b.choose_random_target(candidates, source=self.op,
                                                     mechanism="angel2_bomb_target")
                if chosen is not None:
                    origin, mount = self.b._operator_projectile_origin(self.op, "projectile_chr_angel2_t1", chosen)
                    self.b.projectile_counter += 1
                    cfg = self.b.attack_timing.get(self.op.char_id, {}).get("ammo_bomb", {})
                    self.ammo_bombs.append(dict(id=self.b.projectile_counter, origin=origin, mount=mount,
                        pos=chosen.position(), born=self.b.time,
                        stop=self.b.time + float(cfg.get("lifetime", .30000001192092896)),
                        radius=float(cfg.get("radius", 1.2999999523162842)),
                        amount=self.b.operator_attack_power(self.op) * amount))
            elif kind == "SHIELD_LAUNCH" and self.active:
                if not any(zone["kind"] == "SHIELD_ORBIT" for zone in self.zones):
                    timing = self.op.skill_attack_timing.get(self.sid, {})
                    radius = float(timing.get("shield_radius", 1.2))
                    self.add_zone(self.op.tile, math.inf, "SHIELD_ORBIT", radius=radius)
                    zone = self.zones[-1]
                    mount_origin, mount = self.b._operator_projectile_origin(self.op, "projectile_chr_hsgma2_s2", self.op)
                    center = tuple(map(float, self.op.tile))
                    phase = math.atan2(*FACING[self.op.facing % 4])
                    # AroundTargetRangedAttack supplies the initial radius;
                    # RotateAround's fixed-radius branch is disabled here.
                    origin = (center[0]+radius*math.sin(phase),
                              center[1]+radius*math.cos(phase), mount_origin[2])
                    self.b.projectile_counter += 1
                    zone.update(origin=origin, mount=mount, id=self.b.projectile_counter,
                                center=center, phase=phase)
                    self.b.mechanic_diagnostics["HSGMA2_SHIELD_ORBIT_PROJECTION_CANDIDATE"] = 1
        self.events = pending

    def tick_zones(self):
        b, op = self.b, self.op
        for zone in list(self.zones):
            if b.time > zone["end"] + 1e-9:
                self.zones.remove(zone); continue
            if zone["kind"] == "SHIELD_ORBIT":
                timing = op.skill_attack_timing.get(self.sid, {})
                if op.dead or not self.active:
                    self.zones.remove(zone)
                    continue
                angle = zone.setdefault("angle", 0.0)
                clockwise = -1 if timing.get("shield_clockwise", False) else 1
                phase = zone.get("phase", math.atan2(*FACING[op.facing % 4])) + clockwise * angle
                radius = float(zone.get("radius", timing.get("shield_radius", 1.2)))
                center = zone.get("center", op.tile)
                pos = (center[0]+radius*math.sin(phase), center[1]+radius*math.cos(phase))
                zone["pos"] = pos
                targets = [enemy for enemy in b.enemies if b._enemy_is_attackable(enemy)
                    and b._enemy_targetable(enemy) and not enemy.statuses.has(Status.CAMOUFLAGE)
                    and (not timing.get("shield_ground_only", True) or not b._is_airborne(enemy))
                    and b._enemy_intersects_circle(enemy, pos, float(timing.get("shield_collision_radius", 1)))]
                speed = zone.setdefault("angular_speed", float(timing.get("shield_angular_speed", 90)))
                slowed = zone.setdefault("slowed", False)
                if bool(targets) != slowed:
                    ratio = 1-float(timing.get("shield_slow_ratio", .8))
                    speed = speed * ratio if targets else speed / ratio
                    zone.update(angular_speed=speed, slowed=bool(targets))
                zone["angle"] += math.radians(speed) * b.dt
                if zone["angle"] > math.radians(float(timing.get("shield_max_angle",365))):
                    self.zones.remove(zone)
                    b._finish_skill(op, "SHIELD_RETURN")
                    continue
                # The projectile attaches a periodic buff independently on entry;
                # leaving detaches it, and re-entry starts a fresh interval.
                contacts = zone.setdefault("contacts", {})
                present = {enemy.enemy_index for enemy in targets}
                for key in list(contacts):
                    if key not in present:
                        del contacts[key]
                interval = max(b.dt, self.value("interval", default=.5))
                for enemy in targets:
                    due = contacts.setdefault(enemy.enemy_index, b.time + (
                        interval if timing.get("shield_wait_first_interval", False) else 0))
                    if b.time + 1e-9 >= due:
                        contacts[enemy.enemy_index] = due + interval
                        enemy.statuses.apply(Status.SLUGGISH, self.value("sluggish"))
                        result = self.damage(enemy, b.operator_attack_power(op) * self.value("shield_atk_scale"),
                                             attack_type=AttackType.NORMAL)
                        b._resolve_operator_heal(op, op, abs(result.hp_delta) * self.value("heal_ratio"),
                                                 secondary=True, ignore_heal_free=True)
                b.mechanic_diagnostics["HSGMA2_SHIELD_COLLISION_UPDATE_ORDER_CANDIDATE"] = 1
            elif zone["kind"] == "LURE_BURST":
                for target in self.near(zone["pos"], zone["radius"]):
                    target.statuses.apply(Status.SLUGGISH, b.dt * 2)
                while b.time + 1e-9 >= zone["next"] and zone["next"] <= zone["end"] + 1e-9:
                    zone["next"] += self.value("interval_damage", default=.5)
                    for target in self.near(zone["pos"], zone["radius"]):
                        self.damage(target, zone["atk"] * self.value("atk_scale"))
                        b.apply_element_damage(target, ElementType.SANITY, zone["atk"] * self.value("ep_damage_ratio_token"), source=op)

    def select_deck_discount(self):
        """Deterministic reserve order; ties retain squad order."""
        b = self.b
        deck = [p for p in b.squad if b._active_operator(p.char_id) is None and p.char_id != self.op.char_id]
        # The eye insertion/deck ordering is not observable in existing plans.
        # Keep the candidate explicit rather than silently claiming native order.
        b.mechanic_diagnostics["SVASH2_DECK_ORDER_CANDIDATE"] = 1
        if not deck:
            self.deck_discount = None; return
        def cost(plan):
            from . import data as D
            char = b.characters[plan.char_id]
            attrs = D.char_attributes(char, plan.level, plan.elite, trust=plan.trust,
                potential_rank=plan.potential_rank, module=b.battle_equips.get(plan.module_id), module_level=plan.module_level)
            return b.deployment_cost(plan.char_id, float(attrs.get("cost", 0)))
        professions = {name: index for index, name in enumerate(
            ("PIONEER", "WARRIOR", "TANK", "SNIPER", "CASTER", "MEDIC", "SUPPORT", "SPECIAL", "TRAP"))}
        deck.sort(key=lambda p: (cost(p), self.deck_affected_at.get(p.char_id, float("inf")),
            professions.get(b.characters[p.char_id].get("profession"), 99), p.char_id))
        eye_cost = 16  # S1 fixes the reserve token's cost, independent of ordinary DP modifiers.
        candidates = [p for p in deck if cost(p) > eye_cost and
            b.characters[p.char_id].get("profession") in {"WARRIOR", "CASTER", "SNIPER"}]
        plan = candidates[0] if candidates else min(deck, key=lambda p: abs(cost(p) - eye_cost))
        self.deck_discount = plan.char_id
        self.deck_discounts[plan.char_id] = self.deck_discounts.get(plan.char_id, 0) + self.value("svash2_s_1[deck].cost")
        self.deck_affected_at.setdefault(plan.char_id, b.time)
        self.deck_shields[plan.char_id] = self.deck_shields.get(plan.char_id, 0) + self.op.max_hp * self.value("svash2_s_1[deck].shield")

    def deployed(self, unit):
        self.register_abyssal_ally(unit)
        self.bless_angel_ally(unit)
        self.tick_module_element_count()
        if self.sid != "skchr_svash2_1" or not unit.char_id.startswith("char_"):
            return
        if unit is self.op or unit.deployed_cost <= 16:
            self.b._gain_sp(unit, float(self.talent(1).get("sp", 0)))
            unit.respawn_time *= float(self.talent(1).get("respawn_time", 1))
        self.deck_affected_at.pop(unit.char_id, None)

    def on_lethal(self, packet):
        op, b = self.op, self.b
        if op.char_id == "char_4010_etlchi" and self.talent(2) and not self.low_hp_recovered:
            # The talent's initial undead buff survives until its one-shot heal.
            op.hp = 1.0
            return True
        if self.sid == "skchr_oblvns_1" and self.free_cast():
            op.hp = 1.0
            return True
        if op.char_id == "char_1044_hsgma2":
            t = self.talent(1)
            survives = bool(t) and op.hp > -op.max_hp * float(t.get("max_minus_hp_ratio", 0))
            op.negative_hp_heal_free = survives
            return survives
        if op.char_id == "char_1033_swire2":
            t = self.talent(2)
            if not t:
                return False
            cost = abs(float(t.get("cost", -5)))
            cost = (cost + abs(float(t["cost_addition"])) * self.revive_count if "cost_addition" in t
                    else cost * float(t.get("cost_multi", 2)) ** self.revive_count)
            if b.dp >= cost:
                b.dp -= cost; self.cost_spent(cost); self.revive_count += 1
                op.hp = 0.0
                b._resolve_operator_heal(op, op, op.max_hp * float(t.get("hp_ratio", 0)),
                                         secondary=True, ignore_heal_free=True)
                return op.hp > 0
        if self.revive_count:
            return False
        if op.char_id == "char_4039_horn":
            t = self.talent(1)
            if not t:
                return False
            self.revive_count = 1
            # FINAL_SCALER applies after additive/multiplier HP attributes,
            # including attributes added after this one-shot revival.
            op.max_hp_final_scale *= float(t.get("max_hp", 1))
            b._squad_field_signature = None; b._sync_squad_effects()
            # This heal explicitly skips modifier events but respects HealFree.
            op.hp = 0.0 if op.statuses.has(Status.HEAL_FREE) else min(
                op.max_hp, op.max_hp * float(t.get("hp_ratio", 0)))
            self.attack_speed_delta += float(t.get("attack_speed", 0))
            op.def_scale += float(t.get("def", 0)); return op.hp > 0
        if op.char_id == "char_1046_sbell2":
            t = self.talent(2)
            if not t:
                return False
            self.revive_count = 1; op.hp = 0.0
            b._resolve_operator_heal(op, op, op.max_hp * float(t.get("hp_ratio", 0)),
                                     secondary=True, ignore_heal_free=True)
            op.statuses.apply(Status.FROZEN, float(t.get("freeze", 0)))
            for enemy in self.candidates():
                enemy.statuses.apply(Status.FROZEN, float(t.get("c2e_freeze", 0)))
            return op.hp > 0
        return False

    def taking_damage_modifier(self, packet, target):
        op, b = self.op, self.b
        if op.dead or target is not op:
            return
        if op.char_id == "char_4145_ulpia" and self.talent(2):
            t = self.talent(2)
            key = "value1" if op.hp / op.max_hp > float(t.get("hp_ratio", .5)) else "value2"
            b._resolve_operator_heal(op, op,
                float(t.get(key, 0)) * (self.value("talent_scale", default=1) if self.active else 1),
                secondary=True)
            b.mechanic_diagnostics["ULPIA_TAKE_DAMAGE_NOTIFICATION_PHASE_CANDIDATE"] = 1
        source = packet.source
        if source not in b.enemies or not b._enemy_is_attackable(source):
            return
        if op.char_id == "char_1044_hsgma2" and self.active:
            if op.trait_blackboard.get("atk_scale"):
                self.damage(source, b.operator_attack_power(op) * float(op.trait_blackboard["atk_scale"]),
                            attack_type=AttackType.NORMAL, modifier_key="artsprotector_e_002_tr",
                            ability_owner=None)
            if self.sid == "skchr_hsgma2_1" and b._enemy_is_attackable(source):
                self.damage(source, b.operator_attack_power(op) * self.value("atk_scale"),
                            attack_type=AttackType.NORMAL, ability_owner=None)
            b.mechanic_diagnostics["HSGMA_TAKE_DAMAGE_NOTIFICATION_PHASE_CANDIDATE"] = 1
        if (op.char_id == "char_1046_sbell2" and self.talent(2)
                and not any(source.statuses.has(s) for s in
                            (Status.UNTARGETABLE, Status.INVINCIBLE, Status.INVISIBLE))):
            source.statuses.apply(Status.COLD, float(self.talent(2).get("cold", 0)))
            b.mechanic_diagnostics["SBELL_TAKE_DAMAGE_NOTIFICATION_PHASE_CANDIDATE"] = 1

    def taken_damage(self, packet, result):
        op, b = self.op, self.b
        if (op.char_id == "char_1044_hsgma2" and self.talent(1)
                and not result.healed and not result.cancelled and not result.blocked):
            self.last_damage = b.time
        if result.final_amount <= 0 or result.healed:
            return
        self.last_damage = b.time
        self.check_low_hp_recovery()

    def heal_landed(self, target, amount, result):
        if result.cancelled:
            return
        if self.op.char_id == "char_1016_agoat2":
            element = target.elements.current_damaged_element
            if element is not None:
                target.elements.recover(element, amount * float(self.op.trait_blackboard.get("ep_heal_ratio", .5)))
        if self.op.char_id == "char_1016_agoat2" and self.talent(1):
            key = ("AGOAT", target.deployment_index)
            # EXTEND does not separate character sources. Keep the first
            # live buff's source/blackboard and refresh only its ATK snapshot.
            owner = next((r for r in effect_runtimes(self.b)
                          if key in r.dots and any(v > self.b.time for v in r.dots[key][1])), self)
            if owner is not self:
                self.b.mechanic_diagnostics["AGOAT_SHARED_HOT_HANDOFF_CANDIDATE"] = 1
            old = owner.dots.get(key)
            if old is not None and not any(v > self.b.time for v in old[1]):
                owner.dot_next.pop(key, None)
            values = owner.refreshed_stack_expiries(old[1] if old else [],
                float(self.talent(1).get("duration", 6)), int(owner.talent(1).get("max_stack_cnt", 3)))
            owner.dots[key] = (target, values,
                              self.b.operator_attack_power(self.op) * float(owner.talent(1).get("heal_scale", .1)), False)
            owner.dot_next.setdefault(key, self.b.time + 1.0)

    def refund_dead_heal_target(self, attack):
        # S1's skill component requests a single refund, unlike S2's area heal.
        if (self.sid == "skchr_demkni_1" and attack.sp_recovery_locked
                and self.heal_spent > 0):
            self.b._store_sp_gain(self.op, self.heal_spent)
            self.heal_spent = 0.0

    def launch_alchemy(self, attack):
        # The prefab selects just before firing; invalid input targets still
        # provide a root tile. Exact selector/event handoff remains a candidate.
        targets = self.heal_targets()
        tile = targets[0].tile if targets else attack.profile.payload["tile"]
        timing = self.op.skill_attack_timing.get(self.sid, {})
        origin, mount = self.b._operator_projectile_origin(self.op, "projectile_chr_thorn2_s1",
                                                          targets[0] if targets else None)
        self.b.projectile_counter += 1
        self.alchemy_flights.append({"id": self.b.projectile_counter,
            "pos": origin[:2], "origin": origin, "mount": mount, "tile": tuple(tile),
            "speed": float(timing.get("projectile_speed", 0)), "born": self.b.time,
            "last": self.b.time, "duration": attack.profile.payload["duration"],
            "regen": self.b.operator_attack_power(self.op) * self.value("hp_recovery_per_sec_ratio")})
        self.b.mechanic_diagnostics["THORN2_ALCHEMY_FLIGHT_CANDIDATE"] = 1

    def tick_alchemy_flights(self):
        for flight in list(self.alchemy_flights):
            distance = math.dist(flight["pos"], flight["tile"])
            elapsed = max(0.0, self.b.time - flight["last"])
            flight["last"] = self.b.time
            step = flight["speed"] * elapsed
            if distance > step + 1e-9 and flight["speed"] > 0:
                flight["pos"] = tuple(a + (z - a) * step / distance
                                      for a, z in zip(flight["pos"], flight["tile"]))
                continue
            self.add_zone(flight["tile"], flight["duration"], "ALCHEMY",
                cells=self.b.range_cells_from_id(str(self.bb.get("projectile_range", "x-4")), flight["tile"], 0))
            self.zones[-1]["regen"] = flight["regen"]
            self.alchemy_flights.remove(flight)

    def depart(self):
        self.op.negative_hp_heal_free = False
        self.enemy_count_bonus = False
        self.low_hp_bonus = False
        self.pre_reborn_bonus = False
        self.high_hp_bonus = False
        self.ep_aura_members.clear()
        self.weakness_members.clear()
        self.enemy_entered.clear()
        self.clear_abyssal_sharing()
        self.clear_angel_blessings()
        self.servant_queue.clear()
        self.clear_stolen_enemy_hp()
        self.reaper_pending_hits = self.reaper_heal_count = 0
        self.reaper_batch_at = self.reaper_heal_at = None
        self.end_volley = None
        self.ammo_bombs.clear()  # The bomb is explicitly managed by its source.
        self.events = [e for e in self.events if e[1] != "AMMO_BOMB_EMIT"]
        if self.op.char_id == "char_4087_ines":
            for runtime in effect_runtimes(self.b):
                if runtime is not self and runtime.op.char_id == self.op.char_id:
                    runtime.shadow_cells.clear()
            self.shadow_cells=set(self.op.base_range_cells)
        self.finish("DEPARTURE")
        self.clear_invalid_remnants(force=True)
        self.ward_spawn_at = None
        self.clear_ward_camouflage(force=True)
        # The note controller checks source validity on its next FREE update.
        # A preassigned target also survives departure before TRACE begins.
        for token in self.tokens:
            if not token.dead:
                token.dead = token.retreated = True
                self.b._process_operator_departure(token, retreated=False)
        if self.lure is not None:
            self.lure_arrived(None)
        self.anchor_move = None
        self.relocation_reserve.clear()
        self.zones = [z for z in self.zones if z["kind"] in {"ALCHEMY", "ILLUMINATE", "LURE_BURST"}]
        for enemy, amount in self.steal_atk.values():
            if not enemy.dead:
                enemy.atk += amount
        self.steal_atk.clear()
        self.first_hits.clear()
        # Ascalon's debuff is attached to its passive parent on BUFF_START.
        for key in [key for key in self.dots if key[0] == "ASCLN"]:
            self.dots.pop(key, None)
            self.dot_next.pop(key, None)

    def tick_detached(self):
        self.tick_events()
        self.tick_pull_damage()
        self.tick_alchemy_flights()
        self.tick_zones()
        self.tick_dots()
        if self.sid == "skchr_kalts2_3":
            self.tick_medical_field()

    def front_passable(self):
        dr, dc = FACING[self.op.facing % 4]
        return self.b._ground_passable_cell(self.op.tile[0] + dr, self.op.tile[1] + dc)

    def tick_dots(self):
        b = self.b
        for key, (unit, expiries, amount, current) in list(self.dots.items()):
            deadline = self.dot_next[key]
            while deadline <= b.time + 1e-9 and not unit.dead:
                count = sum(v + 1e-9 >= deadline for v in expiries)
                if count:
                    value = amount * count * (b.operator_attack_power(self.op) if current else 1)
                    if key[0] == "AGOAT":
                        b._resolve_operator_heal(self.op, unit, value, secondary=True)
                        element = unit.elements.current_damaged_element
                        if element is not None and not unit.statuses.has(Status.HEAL_FREE):
                            unit.elements.recover(element, value * float(self.op.trait_blackboard.get("ep_heal_ratio", .5)))
                    else:
                        self.damage(unit, value, apply_way=ApplyWay.MELEE if key[0] == "ASCLN" else ApplyWay.NONE)
                interval = float(self.talent(1).get("interval", 1)) if key[0] == "ASCLN" else 1.0
                deadline += max(b.dt, interval)
            self.dot_next[key] = deadline
            expiries = [v for v in expiries if v > b.time + 1e-9]
            if unit.dead or not expiries:
                self.dots.pop(key, None); self.dot_next.pop(key, None)

    def attack_bonus(self):
        op, b = self.op, self.b
        value = sum(amount for amount,until in self.attack_bonuses.values() if self.b.time < until)
        if op.char_id == "char_1039_thorn2":
            value += float(self.talent(1).get("atk", 0))
        if op.char_id == "char_202_demkni":
            count = min(int(self.talent(1).get("max_stack_cnt", 5)),
                        int((b.time - op.deployed_at) / float(self.talent(1).get("interval", 20))))
            value += count * float(self.talent(1).get("atk", 0))
        if op.char_id == "char_1033_swire2":
            value += self.stacks * float(self.talent(1).get("atk", 0))
            value += self.extra_upkeep_stacks * float(op.talent_blackboards.get("10_root", {}).get("atk", 0))
        if op.char_id == "char_1044_hsgma2":
            t = self.talent(2)
            enabled = self.low_hp_bonus
            if self.low_hp_interval <= 0:
                enabled = op.hp / op.max_hp <= float(t.get("min_hp_ratio", .3))
                if t.get("atk"):
                    b.mechanic_diagnostics["HSGMA_MODULE_LOW_HP_TIMING_MISSING"] = 1
            if enabled and not op.dead:
                value += float(t.get("atk", 0))
            ratio = min(1, max(0, (1 - op.hp / op.max_hp) / (1 - float(t.get("min_hp_ratio", .3)))))
            value += ratio * float(t.get("min_atk", 0))
        if op.char_id == "char_4193_lemuen" and self.lemuen_bonus:
            value += float(self.talent(2).get("atk", 0))
        if op.char_id == "char_1047_halo2" and self.stacks >= float(self.talent(2).get("max_stack_cnt", math.inf)):
            value += float(self.talent(10).get("atk", 0))
        return value

    def flat_attack(self):
        amount = sum(amount for enemy, amount in self.steal_atk.values() if not enemy.dead)
        if self.op.char_id == "char_4087_ines":
            return min(amount, float(self.talent(1).get("steal_atk_max", 0)))
        return amount

    def before_apply_damage(self, packet, target):
        if packet.normalized_damage_type() in {DamageType.HEAL, DamageType.LIFE_LOSS}:
            return
        if (self.op.char_id == "char_4010_etlchi" and target in self.b.enemies
                and packet.normalized_attack_type() is not AttackType.BUFF):
            self.steal_enemy_hp(target)
        self.apply_first_hit_talent(target)

    def apply_first_hit_talent(self, target):
        if (self.op.dead or self.op.char_id not in {"char_4087_ines", "char_4213_skybx"}
                or target not in self.b.enemies or target.dead):
            return
        t = self.talent(1)
        if not t or target.enemy_index in self.first_hits:
            return
        self.first_hits.add(target.enemy_index)
        self.b.used_assumptions.add("FIRST_HIT_NOTIFICATION_PHASE_CANDIDATE")
        if self.op.char_id == "char_4087_ines":
            # The steal ability has keepTargetAttributeAtLeastOneAfterSteal=0.
            # Its owner's aggregate cap does not stop new targets losing ATK.
            steal = max(0, float(t.get("steal_atk", 0)))
            self.steal_atk[target.enemy_index] = (target, steal)
            target.atk -= steal
            target.statuses.apply(Status.BIND, float(t.get("duration", 0)))
            self.b.used_assumptions.add("INES_ATTRIBUTE_COMPOSITION_CANDIDATE")
        else:
            target.statuses.apply(Status.SILENCE, float(t.get("silence", 0)))

    def defense_bonus(self):
        if self.op.char_id == "char_136_hsguma" and self.op.blocked:
            return float(self.op.trait_blackboard.get("def", 0)) + float(self.talent(10).get("def", 0))
        if self.op.char_id == "char_136_hsguma":
            return float(self.talent(10).get("def", 0))
        if self.op.char_id == "char_4039_horn" and self.has_pre_reborn_bonus():
            return float(self.talent(11).get("horn_e_003_t[attr].def", 0))
        if self.op.subprofession == "phalanx" and not self.active:
            return float(self.op.trait_blackboard.get("def", 0))
        if self.op.char_id == "char_202_demkni":
            count = min(int(self.talent(1).get("max_stack_cnt", 5)),
                        int((self.b.time - self.op.deployed_at) / float(self.talent(1).get("interval", 20))))
            return count * float(self.talent(1).get("def", 0))
        return 0.0

    def flat_defense_for(self, target):
        value = sum(self.value("def") for z in self.zones if z["kind"] == "ALCHEMY"
                    and target.tile in z["cells"])
        if not self.op.dead and self.op.char_id == "char_1045_svash2" and self.sjerag_ally(target):
            t = self.talent(2)
            value += float(t.get("def", 0)) * (2 if self.b.time - self.op.deployed_at >= float(t.get("interval", 15)) else 1)
        return value

    def attack_speed(self):
        value = self.attack_speed_delta
        if self.op.module_id == "uniequip_003_angel":
            # This ground aura disables advanced target-free checks, unlike attacks.
            count = sum(self.b._enemy_is_attackable(e) and not self.b._is_airborne(e)
                        and self.b._enemy_intersects_range(e, self.op.range_cells) for e in self.b.enemies)
            if count >= int(self.op.trait_blackboard.get("cnt", 1)) and not self.op.dead:
                value += float(self.op.trait_blackboard.get("attack_speed", 0))
            self.b.mechanic_diagnostics["ANGEL_MODULE_GROUND_AURA_PHASE_CANDIDATE"] = 1
        if self.op.module_id == "uniequip_003_etlchi":
            # CheckHasEnemyInRange counts living enemies rather than attack targets.
            enabled = self.enemy_count_bonus
            if self.enemy_count_interval <= 0:
                # Older data lacks the component clock; preserve its fallback.
                count = sum(self.b._enemy_is_attackable(e)
                            and self.b._enemy_intersects_range(e, self.op.range_cells) for e in self.b.enemies)
                enabled = count >= int(self.op.trait_blackboard.get("cnt", 2))
                self.b.mechanic_diagnostics["ETLCHI_ENEMY_COUNT_TIMING_MISSING"] = 1
            if enabled and not self.op.dead:
                value += float(self.op.trait_blackboard.get("attack_speed", 0))
            self.b.mechanic_diagnostics["ETLCHI_ENEMY_COUNT_CHECK_PHASE_CANDIDATE"] = 1
        if self.sid == "skchr_halo2_2":
            value += self.stacks * float(self.talent(2).get("attack_speed", 0))
        if self.sid == "skchr_oblvns_1" and self.op.module_id == "uniequip_002_oblvns":
            enabled = self.enemy_count_bonus
            if self.enemy_count_interval <= 0:
                enabled = len(self.candidates()) >= 2
                self.b.mechanic_diagnostics["OBLVNS_MODULE_ENEMY_COUNT_TIMING_MISSING"] = 1
            if enabled and not self.op.dead:
                value += float(self.op.trait_blackboard.get("attack_speed", 0))
        if self.op.char_id == "char_4039_horn":
            if not self.op.blocked:
                value += float(self.talent(10).get("attack_speed", 0))
            if self.has_pre_reborn_bonus():
                value += float(self.talent(11).get("horn_e_003_t[attr].attack_speed", 0))
        if self.op.char_id == "char_1016_agoat2" and self.op.module_id == "uniequip_003_agoat2":
            enabled = any(entry[2] for entry in self.ep_aura_members.values())
            if self.ep_check_interval <= 0:
                enabled = any(not p.dead and p.tile in self.op.range_cells
                              and p.elements.current_damaged_element is not None for p in self.b.operators)
                self.b.mechanic_diagnostics["AGOAT_MODULE_ELEMENT_COUNT_TIMING_MISSING"] = 1
            if enabled and not self.op.dead:
                value += float(self.op.trait_blackboard.get("attack_speed", 0))
        return value

    def sp_recovery_bonus(self):
        if self.op.char_id == "char_1039_thorn2" and any(
                r.alchemy_flights or any(z["kind"]=="ALCHEMY" and z["end"]>self.b.time for z in r.zones)
                for r in effect_runtimes(self.b)):
            return float(self.talent(10).get("sp_recovery_per_sec", 0))
        if self.op.char_id == "char_1041_angel2":
            t = self.op.trait_blackboard
            enabled = self.high_hp_bonus
            if self.high_hp_interval <= 0:
                enabled = self.op.hp >= self.op.max_hp * float(t.get("angel2_tr[e].hp_ratio", math.inf))
                if t.get("angel2_tr[e].sp_recovery_per_sec"):
                    self.b.mechanic_diagnostics["ANGEL2_HIGH_HP_TIMING_MISSING"] = 1
            if enabled and not self.op.dead:
                return float(t.get("angel2_tr[e].sp_recovery_per_sec", 0))
        return 0.0

    def res_bonus(self):
        if self.op.subprofession == "phalanx" and not self.active:
            return float(self.op.trait_blackboard.get("magic_resistance", 0))
        if self.op.char_id == "char_1044_hsgma2":
            t = self.talent(2)
            ratio = min(1, max(0, (1 - self.op.hp / self.op.max_hp) / (1 - float(t.get("min_hp_ratio", .3)))))
            return ratio * float(t.get("min_magic_resistance", 0))
        return 0.0

    def move_factor(self, enemy):
        value = 1.0
        if (self.op.char_id == "char_4132_ascln" and self.op.module_id == "uniequip_002_ascln"
                and not self.op.dead and self.b._enemy_is_attackable(enemy)
                and not self.b._is_airborne(enemy) and self.b._enemy_targetable(enemy)
                and self.b._enemy_intersects_range(enemy, self.op.range_cells)):
            value *= max(0, 1 + float(self.talent(10).get("move_speed", 0)))
            self.b.mechanic_diagnostics["ASCLN_MODULE_SLOW_AURA_PHASE_CANDIDATE"] = 1
        data = self.dots.get(("ASCLN", enemy.enemy_index))
        if data:
            value *= max(0, 1 + sum(t > self.b.time for t in data[1]) * float(self.talent(1).get("move_speed", 0)))
        if self.op.char_id == "char_4087_ines" and self.reveals(enemy):
            if not self.op.dead or not any(r.op.char_id == self.op.char_id and r.reveals(enemy)
                                          for r in live_runtimes(self.b)):
                value *= 1 + float(self.talent(2).get("move_speed", 0))
        if self.sid == "skchr_sbell2_2":
            value *= max(0, 1 + self.snow.get(enemy.tile(), 0) * float(self.talent(1).get("move_speed", 0)))
        if self.sid == "skchr_whitw2_3" and self.active and any(
                math.dist(d["pos"], enemy.position()) <= self.value("attack@range_radius") for d in self.drones):
            value *= 1 + self.value("attack@move_speed")
        return value

    def modify_packet(self, packet, target):
        if packet.normalized_damage_type() in {DamageType.HEAL, DamageType.LIFE_LOSS}:
            return packet
        if (self.op.char_id == "char_4133_logos" and target in self.b.enemies
                and packet.normalized_damage_type() not in {DamageType.HEAL, DamageType.LIFE_LOSS}):
            t = self.talent(2)
            if t:
                target.roster_res_delta = float(t.get("magic_resistance", 0))
                target.roster_arts_addition = float(t.get("atk_addition", 0))
                target.logos_core_until = max(getattr(target, "logos_core_until", 0),
                    self.b.time + float(t.get("duration", 5)))
                self.events.append((target.logos_core_until, "RESTORE_RES", target, 0))
                packet = replace(packet, target_resistance_override=self.b._effective_resistance(target))
                self.b.mechanic_diagnostics["LOGOS_T2_CALCULATE_FILTER_PHASE_CANDIDATE"] = 1
        if self.sid == "skchr_oblvns_1":
            if not self.free_cast():
                self.fever = min(450, self.fever + float(self.talent(2).get("cnt", 0)))
            count = min(sum(not note.dead for note in self.notes), int(self.talent(1).get("max_cnt", 10)))
            packet = replace(packet, physical_penetration_ratio=packet.physical_penetration_ratio + count * float(self.talent(1).get("def_penetrate_ratio", 0)),
                             arts_penetration_ratio=packet.arts_penetration_ratio + count * float(self.talent(1).get("magic_resist_penetrate_ratio", 0)))
        if self.op.char_id == "char_1046_sbell2":
            t = self.talent(10)
            per_enemy = float(t.get("damage_scale", 0))
            cap = float(self.op.trait_blackboard.get("damage_scale", 0))
            # The equip aura uses TargetOptions, not the combat camouflage filter.
            count = sum(self.b._enemy_is_attackable(e) and self.b._enemy_targetable(e)
                        and self.b._enemy_intersects_range(e, self.op.range_cells) for e in self.b.enemies)
            packet = replace(packet, output_scale=packet.output_scale * (1 + min(cap, count * per_enemy)))
            if per_enemy:
                self.b.mechanic_diagnostics["SBELL_MODULE_RANGE_AURA_PHASE_CANDIDATE"] = 1
        if self.op.char_id == "char_103_angel" and self.b.time < self.penetration_until:
            packet = replace(packet, physical_penetration=packet.physical_penetration +
                             self.penetration_stacks * float(self.talent(1).get("def_penetrate_fixed", 0)))
        if (self.op.char_id == "char_172_svrash" and target in self.b.enemies and not target.dead
                and target.level_type in {"ELITE", "BOSS"}
                and packet.normalized_damage_type() is not DamageType.LIFE_LOSS
                and packet.modifier_key != "svrash_e_002_tr"):
            # The trait's direct NoOnCalDmg packet already includes this scale.
            packet = replace(packet, amount=packet.amount * float(self.talent(10).get("atk_scale", 1)))
            if self.talent(10).get("atk_scale"):
                self.b.mechanic_diagnostics["SVRASH_ELITE_CALCULATE_FILTER_PHASE_CANDIDATE"] = 1
        if (self.op.char_id == "char_4039_horn" and target in self.b.enemies and target.blocked_by is not None
                and packet.normalized_damage_type() is not DamageType.LIFE_LOSS):
            packet = replace(packet, amount=packet.amount * float(self.op.trait_blackboard.get("atk_scale", 1)))
            if self.op.trait_blackboard.get("atk_scale"):
                self.b.mechanic_diagnostics["HORN_BLOCKED_CALCULATE_FILTER_PHASE_CANDIDATE"] = 1
        if self.op.char_id == "char_450_necras" and packet.source is self.op and any(target.blocked_by is token for token in self.tokens):
            packet = replace(packet, amount=packet.amount * float(self.op.trait_blackboard.get("atk_scale", 1)))
        if self.op.subprofession == "lord" and packet.normalized_attack_type() is AttackType.NORMAL and not any(e is target for e in self.op.blocked):
            if self.sid != "skchr_oblvns_1":
                packet = replace(packet, output_scale=packet.output_scale * .8)
        if self.op.char_id == "char_450_necras" and target.hp / target.max_hp < float(self.talent(2).get("hp_ratio", 0)):
            packet = replace(packet, output_scale=packet.output_scale * float(self.talent(2).get("damage_scale", 1)))
        return packet

    def modify_taken_packet(self, packet):
        if packet.normalized_damage_type() in {DamageType.HEAL, DamageType.LIFE_LOSS} or self.op.dead:
            return packet
        if (not packet.ignore_invincible
                and any(self.op.statuses.has(s) for s in (Status.INVINCIBLE, Status.SLEEP))):
            return packet
        if self.op.char_id == "char_136_hsguma" and self.b._roll_probability(float(self.talent(1).get("prob", 0)),
                mechanism="hsguma_damage_resist", source=self.op, target=packet.source):
            self.grant_attack_bonus("HSGUMA_RESIST", self.talent(1).get("atk", 0), self.talent(1).get("duration", 0))
            return replace(packet, amount=0)
        return packet

    def check_low_hp_recovery(self):
        op = self.op
        if op.char_id != "char_4010_etlchi" or self.low_hp_recovered or op.dead:
            return
        t = self.talent(2)
        if not t or op.hp > op.max_hp * float(t.get("hp_ratio", 0)):
            return
        self.low_hp_recovered = True
        self.b._resolve_operator_heal(op, op, op.max_hp * float(t.get("etlchi_t_2[heal].hp_ratio", 0)),
                                      secondary=True, ignore_heal_free=True)
        op.damage_rules.append(DamageRule(DamageRuleKind.RESISTANCE,
            value=max(0, float(t.get("damage_resistance", 0))),
            damage_types=frozenset({DamageType.PHYSICAL})))
        self.b.mechanic_diagnostics["ETLCHI_LOW_HP_NOTIFICATION_ORDER_CANDIDATE"] = 1

    @staticmethod
    def adjust_enemy_max_hp(enemy, delta):
        previous = enemy.max_hp
        enemy.max_hp = max(1.0, previous + delta)
        if not enemy.dead and previous > 0:
            enemy.hp *= enemy.max_hp / previous
        return enemy.max_hp - previous

    def steal_enemy_hp(self, enemy):
        t = self.talent(1)
        if (self.op.dead or enemy.dead or not t or getattr(enemy, "profession", None) == "TRAP"
                or getattr(enemy, "prop_like", False)
                or "enemy_entlec_self_mark" in getattr(enemy, "buff_blackboards", {})):
            return
        entry = self.stolen_enemy_hp.setdefault(enemy.enemy_index,
            {"unit": enemy, "stolen": 0.0, "applied": 0.0})
        cap = float(t.get("attack@steal_hp_max", 0))
        amount = min(float(t.get("attack@steal_hp", 0)), max(0, cap-entry["stolen"]))
        if amount <= 0:
            return
        entry["stolen"] += amount
        entry["applied"] -= self.adjust_enemy_max_hp(enemy, -amount)
        total = min(cap, sum(p["stolen"] for p in self.stolen_enemy_hp.values()))
        self.op.base_max_hp += total-self.stolen_hp
        self.stolen_hp = total
        self.b._squad_field_signature = None
        self.b._sync_squad_effects()
        self.b.mechanic_diagnostics["ETLCHI_STEAL_ATTRIBUTE_AND_EXCLUSION_CANDIDATE"] = 1

    def clear_stolen_enemy_hp(self):
        for entry in self.stolen_enemy_hp.values():
            self.adjust_enemy_max_hp(entry["unit"], entry["applied"])
        self.stolen_enemy_hp.clear()
        self.op.base_max_hp -= self.stolen_hp
        self.stolen_hp = 0.0

    def queue_reaper_heal(self):
        cfg = self.b.attack_timing.get(self.op.char_id, {}).get("reaper_heal", {})
        if self.reaper_batch_at is None:
            self.reaper_batch_at = self.b.time + float(cfg.get("first_delay", .001))
        self.reaper_pending_hits = min(self.op.block_count, self.reaper_pending_hits + 1)

    def tick_reaper_heals(self):
        if self.op.char_id != "char_4010_etlchi" or self.op.dead:
            return
        b = self.b
        cfg = b.attack_timing.get(self.op.char_id, {}).get("reaper_heal", {})
        if self.reaper_batch_at is not None and b.time+1e-9 >= self.reaper_batch_at:
            if not self.reaper_heal_count:
                self.reaper_heal_at = self.reaper_batch_at
            self.reaper_heal_count += self.reaper_pending_hits
            self.reaper_pending_hits = 0
            self.reaper_batch_at = None
        interval = max(b.dt, float(cfg.get("interval", .12)))
        while self.reaper_heal_count and b.time+1e-9 >= self.reaper_heal_at:
            b._resolve_operator_heal(self.op, self.op, float(self.op.trait_blackboard.get("value", 50)),
                                     secondary=True, ignore_heal_free=True)
            self.reaper_heal_count -= 1
            self.reaper_heal_at += interval
        if not self.reaper_heal_count:
            self.reaper_heal_at = None
        b.mechanic_diagnostics["ETLCHI_REAPER_HEAL_QUEUE_PHASE_CANDIDATE"] = 1

    def apply_sluggish(self, enemy, duration):
        applied = enemy.statuses.apply(Status.SLUGGISH, duration)
        if applied is not None and self.sid == "skchr_halo2_2":
            self.stacks = min(int(self.talent(2).get("max_stack_cnt", 0)), self.stacks + 1)

    def tick_weakness_aura(self):
        if self.sid != "skchr_halo2_2" or self.op.dead:
            return
        b, t = self.b, self.talent(1)
        if not t:
            self.weakness_members.clear()
            return
        # The aura's standard validator does not add the attack camouflage filter.
        targets = {e.enemy_index: e for e in b.enemies if b._enemy_is_attackable(e)
                   and b._enemy_targetable(e) and b._enemy_intersects_range(e, self.op.range_cells)}
        if self.weakness_check_interval <= 0:
            self.enemy_entered = {i: at for i, at in self.enemy_entered.items() if i in targets}
            for i in targets:
                self.enemy_entered.setdefault(i, b.time)
            b.mechanic_diagnostics["HALO2_WEAKNESS_CLOCK_MISSING"] = 1
            return
        interval = max(b.dt, self.weakness_check_interval)
        growth = float(t.get("halo2_t_1[weak].interval", math.inf))
        for key, entry in list(self.weakness_members.items()):
            enemy = entry["unit"]
            if not b._enemy_is_attackable(enemy):
                self.weakness_members.pop(key)
                continue
            if b.time + 1e-9 >= entry["check"]:
                if key not in targets:
                    self.weakness_members.pop(key)
                    continue
                entry["check"] += (math.floor(max(0, b.time - entry["check"]) / interval) + 1) * interval
            if growth > 0 and b.time + 1e-9 >= entry["grow"]:
                count = math.floor(max(0, b.time - entry["grow"]) / growth) + 1
                entry["grow"] += count * growth
                entry["scale"] = min(float(t.get("halo2_t_1[weak].damage_scale_max", entry["scale"])),
                                     entry["scale"] + count * float(t.get("halo2_t_1[weak].damage_scale_add", 0)))
        for key, enemy in targets.items():
            self.weakness_members.setdefault(key, dict(unit=enemy, check=b.time + interval,
                grow=b.time + growth, scale=float(t.get("halo2_t_1[weak].damage_scale", 1))))
        b.mechanic_diagnostics["HALO2_WEAKNESS_AURA_CHECK_PHASE_CANDIDATE"] = 1

    def weakness_scale(self, enemy):
        if self.sid != "skchr_halo2_2" or self.op.dead or not self.b._enemy_is_attackable(enemy):
            return 1.0
        t = self.talent(1)
        if self.weakness_check_interval > 0:
            entry = self.weakness_members.get(enemy.enemy_index)
            return entry["scale"] if entry is not None else 1.0
        if not self.b._enemy_intersects_range(enemy, self.op.range_cells):
            return 1.0
        base = float(t.get("halo2_t_1[weak].damage_scale", 1))
        entered = self.enemy_entered.setdefault(enemy.enemy_index, self.b.time)
        if self.b.time - entered >= float(t.get("halo2_t_1[weak].interval", math.inf)):
            base = min(float(t.get("halo2_t_1[weak].damage_scale_max", base)),
                       base + float(t.get("halo2_t_1[weak].damage_scale_add", 0)))
        return base

    def aura_hp(self, target):
        if self.op.char_id == "char_1016_agoat2" and target.tile in self.op.range_cells:
            return float(self.talent(2).get("max_hp", 0))
        return 0.0

    def element_incoming_scale(self, target, element):
        if self.op.char_id == "char_1016_agoat2" and target in self.b.operators and target.tile in self.op.range_cells:
            return max(0, 1 - float(self.talent(2).get("ep_damage_resistance", 0)))
        if (self.op.char_id == "char_245_cello" and target in self.b.enemies and element is ElementType.DARK
                and (self.op.module_id == "uniequip_003_cello" or self.b._enemy_intersects_range(target, self.op.range_cells))):
            return float(self.talent(2).get("ep_damage_scale", 1))
        return 1.0

    def sjerag_ally(self, target):
        return self.b.characters.get(target.char_id, {}).get("nationId") == "kjerag"

    def enemy_died(self, enemy, killer=None):
        if (self.op.char_id == "char_4145_ulpia" and self.op.module_id == "uniequip_003_ulpia"
                and self.op.module_level >= 2 and not self.op.dead
                and killer is not self.op and self.ulpia_kill_aura_eligible(killer)):
            # Own kill credit stays at the existing caller; the module aura
            # routes other Abyssal Hunters' kill notifications to this source.
            self.killed_enemy(enemy)
        if self.op.char_id == "char_4132_ascln":
            dot = self.dots.get(("ASCLN", enemy.enemy_index))
            if dot and any(t > self.b.time for t in dot[1]):
                self.b._resolve_operator_heal(self.op, self.op,
                    self.op.max_hp * float(self.talent(1).get("hp_ratio", 0)), secondary=True)
        if self.op.char_id == "char_450_necras" and self.talent(1) and any(
                self.b._enemy_intersects_range(enemy, unit.range_cells)
                for unit in [self.op,*self.tokens] if not unit.dead):
            if sum(not p.dead for p in self.tokens) >= int(self.talent(1).get("max_token_cnt", 0)):
                self.summon_servant(enemy.tile())
            else:
                cfg = self.b.attack_timing.get(self.op.char_id, {}).get("servant_spawn", {})
                interval = max(self.b.dt, float(cfg.get("interval", .1)))
                previous = self.servant_queue[-1][0] if self.servant_queue else self.b.time
                self.servant_queue.append((max(self.b.time, previous)+interval, enemy.tile()))
            t=self.talent(10)
            self.grant_attack_bonus("NECRAS_DEATH", t.get("atk", 0), t.get("atk_duration", 0))

    def ulpia_kill_aura_eligible(self, unit):
        # CRU-Y's tag validator uses ordinary ally target/heal eligibility.
        return (unit is not None and not unit.dead and unit.char_id.startswith("char_")
                and unit.statuses.targetable and not unit.statuses.has(Status.HEAL_FREE)
                and self.has_group_tag(unit, "abyssal"))

    def ammo_capacity_bonus(self, target):
        if self.op.char_id != "char_4193_lemuen" or self.b.time-self.op.deployed_at < float(self.talent(2).get("interval", math.inf)):
            return 0.0
        if target is self.op:
            return float(self.talent(2).get("add_count", 0))
        if self.has_group_tag(target, "laterano"):
            return float(self.talent(2).get("ex_add_count", 0))
        return 0.0

    def departure_respawn_scale(self, target):
        if self.op.char_id == "char_172_svrash" and max(abs(target.tile[0]-self.op.tile[0]),abs(target.tile[1]-self.op.tile[1]))<=1:
            return max(0,1+float(self.talent(11).get("respawn_time", 0)))
        return 1.0

    def tick_pull_damage(self):
        for key,entry in list(self.pull_damage.items()):
            enemy=entry["unit"]
            if enemy.dead or enemy.disappeared:
                self.pull_damage.pop(key);continue
            position=enemy.position()
            entry["distance"]+=math.dist(position,entry["pos"]);entry["pos"]=position
            if self.b.time+1e-9>=entry["next"] or not enemy.unbalanced:
                entry["next"]=self.b.time+float(self.op.trait_blackboard.get("interval",.066))
                distance=entry["distance"]
                amount=distance*float(self.op.trait_blackboard.get("value",0))/max(float(self.op.trait_blackboard.get("dist",1)),1e-9)
                entry["distance"]=0
                # DamageByDistance discards a sampled travel delta above four,
                # rather than capping or carrying it into the next sample.
                if 0 < amount and 0 < distance <= 4:
                    self.damage(enemy,amount,"ARTS")
            if not enemy.unbalanced:self.pull_damage.pop(key,None)
        if self.pull_damage:self.b.mechanic_diagnostics["GLADY_DISTANCE_DAMAGE_NOTIFY_CANDIDATE"]=1

    def tick_agoat_element_aura(self):
        b, op = self.b, self.op
        if self.sid != "skchr_agoat2_1" or not self.active or op.dead:
            self.agoat_aura_checks.clear()
            return
        targets = [p for p in b.operators if not p.dead and p.tile in op.range_cells]
        present = {p.deployment_index for p in targets}
        self.agoat_aura_checks = {i: due for i, due in self.agoat_aura_checks.items() if i in present}
        interval = max(b.dt, float(op.skill_attack_timing.get(self.sid, {}).get("element_aura_interval", 1)))
        for target in targets:
            due = self.agoat_aura_checks.setdefault(target.deployment_index, b.time + interval)
            if b.time + 1e-9 < due:
                continue
            self.agoat_aura_checks[target.deployment_index] = due + interval
            amount = b.operator_attack_power(op) * self.value("agoat2_s_1[aura].ep_heal_ratio")
            for element in ElementType:
                target.elements.recover(element, amount)
        b.mechanic_diagnostics["AGOAT_ELEMENT_AURA_REGISTRATION_PHASE_CANDIDATE"] = 1

    def tick_cello_talent(self):
        b, op = self.b, self.op
        if op.char_id != "char_245_cello" or op.dead or not self.talent(1):
            return
        if b.time + 1e-9 < self.next_cello_pulse:
            return
        self.next_cello_pulse += max(b.dt, float(b.attack_timing.get(op.char_id, {}).get(
            "talent1", {}).get("interval", 1)))
        for enemy in self.candidates():
            b.apply_element_damage(enemy, ElementType.DARK,
                b.operator_attack_power(op) * float(self.talent(1).get("ep_damage_ratio", 0)), source=op)
            # AlwaysNext keeps sluggish independent of elemental immunity.
            enemy.statuses.apply(Status.SLUGGISH, float(self.talent(1).get("sluggish", 0)))
        b.mechanic_diagnostics["CELLO_TALENT_TIMER_ATTACHMENT_PHASE_CANDIDATE"] = 1

    def element_break_started(self, enemy):
        if (self.op.char_id != "char_245_cello" or self.op.dead or not self.talent(2).get("damage_value")
                or enemy not in self.b.enemies or not self.b._enemy_is_attackable(enemy)
                or enemy.elements.active_break is not ElementType.DARK):
            return
        if enemy.enemy_index not in self.element_pulses:
            # The derived UNIQUE buff has waitFirstTriggerInterval=false.
            self._pulse_element_bonus(enemy)

    def element_break_finished(self, target):
        self.element_pulses.pop(getattr(target, "enemy_index", None), None)

    def _pulse_element_bonus(self, enemy):
        self.element_pulses[enemy.enemy_index] = self.b.time + max(
            self.b.dt, float(self.talent(2).get("interval", 1)))
        self.b._apply_element_burst_damage(
            enemy, float(self.talent(2)["damage_value"]), DamageType.ELEMENTAL, None)
        self.b.mechanic_diagnostics["CELLO_ELEMENT_BURST_PHASE_CANDIDATE"] = 1

    def tick_element_bonus(self):
        if self.op.char_id != "char_245_cello" or self.op.dead or not self.talent(2).get("damage_value"):
            return
        for enemy in self.b.enemies:
            if not self.b._enemy_is_attackable(enemy) or enemy.elements.active_break is not ElementType.DARK:
                self.element_pulses.pop(enemy.enemy_index,None);continue
            deadline=self.element_pulses.setdefault(enemy.enemy_index,self.b.time)
            if self.b.time+1e-9>=deadline:
                self._pulse_element_bonus(enemy)
        self.b.mechanic_diagnostics["CELLO_ELEMENT_BURST_PHASE_CANDIDATE"]=1

    def killed_enemy(self, enemy):
        if self.op.char_id != "char_4145_ulpia":
            return
        t = self.talent(1)
        if self.stacks >= int(t.get("max_stack_cnt", 0)):
            return
        self.stacks += 1
        self.op.base_max_hp += float(t.get("max_hp", 0))
        self.op.atk += float(t.get("atk", 0))
        threshold = int(t.get("add_block_cnt_stack_cnt", 0))
        if (self.op.module_id == "uniequip_003_ulpia" and self.op.module_level >= 2
                and threshold > 0 and self.stacks >= threshold and not self.growth_block_delta):
            self.growth_block_delta = int(t.get("block_cnt", 0))
            self.op.block_count += self.growth_block_delta
        self.b._squad_field_signature = None
        self.b._sync_squad_effects()

    def register_abyssal_ally(self, unit):
        if (self.op.char_id != "char_4145_ulpia" or self.op.dead or unit.dead
                or not self.talent(1) or unit.char_id == self.op.char_id
                or not self.has_group_tag(unit, "abyssal")):
            return
        cfg = self.b.attack_timing.get(self.op.char_id, {}).get("abyssal_sharing", {})
        self.abyssal_shared.setdefault(unit.deployment_index,
            {"unit": unit, "hp": 0.0, "atk": 0.0,
             "next": self.b.time + float(cfg.get("first_delay", .04))})

    def clear_abyssal_sharing(self):
        for entry in self.abyssal_shared.values():
            entry["unit"].base_max_hp -= entry["hp"]
            entry["unit"].atk -= entry["atk"]
        if self.abyssal_shared:
            self.b._squad_field_signature = None
            self.b._sync_squad_effects()
        self.abyssal_shared.clear()

    def tick_abyssal_sharing(self):
        if self.op.char_id != "char_4145_ulpia" or not self.talent(1):
            return
        for unit in self.b.operators:
            self.register_abyssal_ally(unit)
        t = self.talent(1)
        cfg = self.b.attack_timing.get(self.op.char_id, {}).get("abyssal_sharing", {})
        count = min(self.stacks, int(t.get("ulpia_t_1[abyssal].max_stack_cnt", 0)))
        changed = False
        for ident, entry in list(self.abyssal_shared.items()):
            unit = entry["unit"]
            if unit.dead:
                unit.base_max_hp -= entry["hp"]
                unit.atk -= entry["atk"]
                self.abyssal_shared.pop(ident)
                continue
            if self.b.time + 1e-9 < entry["next"]:
                continue
            interval = max(self.b.dt, float(cfg.get("interval", 1)))
            entry["next"] += (math.floor(max(0.0, self.b.time-entry["next"])/interval) + 1) * interval
            hp = count * float(t.get("ulpia_t_1[abyssal].max_hp", 0))
            atk = count * float(t.get("ulpia_t_1[abyssal].atk", 0))
            if (hp, atk) != (entry["hp"], entry["atk"]):
                unit.base_max_hp += hp - entry["hp"]
                unit.atk += atk - entry["atk"]
                entry["hp"], entry["atk"] = hp, atk
                changed = True
        if changed:
            self.b._squad_field_signature = None
            self.b._sync_squad_effects()
        self.b.mechanic_diagnostics["ULPIA_SHARED_GROWTH_TIMER_AND_ATTRIBUTE_CANDIDATE"] = 1

    def aura_atk(self, target):
        char = self.b.characters.get(target.char_id, {})
        if self.op.char_id == "char_4039_horn" and char.get("profession") == "TANK":
            return float(self.talent(2).get("atk", 0))
        if self.op.char_id == "char_1041_angel2" and target.skill_duration_type == "AMMO":
            bonus = float(self.talent(2).get("atk", 0))
            return bonus * (float(self.talent(2).get("mult", 1)) if char.get("nationId") == "laterano" else 1)
        return 0

    def aura_def(self, target):
        if self.op.char_id == "char_136_hsguma" and self.b.characters.get(target.char_id, {}).get("profession") == "TANK":
            return float(self.talent(2).get("def", 0))
        return 0

    def aura_speed(self, target):
        if self.op.char_id == "char_4182_oblvns" and target.tile in self.op.range_cells:
            return float(self.talent(2).get("attack_speed", 0))
        if self.op.char_id == "char_1039_thorn2":
            t = self.talent(2)
            value = float(t.get("attack_speed_ally", 0))
            if self.long_road(target.tile, int(t.get("cnt", 6))):
                value += float(t.get("attack_speed_ally_extra", 0))
            return value
        return 0

    def long_road(self, cell, minimum):
        for dr, dc in ((0, 1), (1, 0)):
            count = 1
            for direction in (-1, 1):
                r, c = cell
                while self.b._ground_passable_cell(r + dr * direction, c + dc * direction):
                    r += dr * direction; c += dc * direction; count += 1
                    if count >= minimum:
                        return True
        return False

    def enemy_speed_delta(self, target):
        if self.op.char_id == "char_1039_thorn2":
            t = self.talent(2)
            return float(t.get("attack_speed_enemy", 0)) + (float(t.get("attack_speed_enemy_extra", 0))
                if self.long_road(target.tile(), int(t.get("cnt", 6))) else 0)
        if self.op.char_id == "char_1042_phatm2" and target.elements.active_break is ElementType.SANITY:
            return float(self.talent(2).get("attack_speed", 0))
        return 0

    def before_enemy_normal_attack(self, enemy):
        if (self.op.char_id != "char_1042_phatm2" or self.op.dead or enemy.dead
                or not self.talent(2) or not self.b._enemy_intersects_range(enemy, self.op.range_cells)):
            return
        # The range validator explicitly ignores target-free flags. The
        # template fires before SPELL_ON, rather than on hit or projectile reach.
        self.b.apply_element_damage(enemy, ElementType.SANITY,
                                    float(self.talent(2).get("value", 0)), source=self.op)
        self.b.mechanic_diagnostics["PHATM2_ENEMY_CAST_AURA_PHASE_CANDIDATE"] = 1

    def born(self):
        self.tick_module_low_hp()
        self.tick_module_pre_reborn()
        self.tick_weakness_aura()
        if self.op.char_id == "char_245_cello":
            # ON_BUFF_START also creates the derived timer for an existing break.
            self.tick_element_bonus()
        if self.op.char_id == "char_4145_ulpia":
            if self.op.module_id == "uniequip_003_ulpia":
                self.b.mechanic_diagnostics["ULPIA_Y_AURA_ATTACHMENT_PHASE_CANDIDATE"] = 1
            for unit in self.b.operators:
                self.register_abyssal_ally(unit)
        if self.op.char_id == "char_1035_wisdel" and "2" in self.op.talent_blackboards:
            cfg = self.b.attack_timing.get(self.op.char_id, {}).get("soul_ward", {})
            self.ward_spawn_at = self.b.time + float(cfg.get("spawn_delay", 1))
            self.b.mechanic_diagnostics["WISDEL_WARD_BIRTH_TILE_AND_NOTIFICATION_CANDIDATE"] = 1
        if self.op.char_id == "char_1046_sbell2" and self.talent(1).get("first_snow"):
            self.next_snow = self.b.time
        if self.op.char_id == "char_103_angel" and self.talent(2):
            if not self.angel_blessing_started:
                self.angel_blessing_started = True
                self.add_angel_blessing(self.op)
            pool = [p for p in self.b.operators if self.angel_ally_eligible(p)]
            for _ in range(min(self.angel_blessing_limit()-self.angel_blessing_count, len(pool))):
                chosen = self.b.choose_random_target(pool, mechanism="angel_blessing_target", source=self.op)
                self.bless_angel_ally(chosen)
                pool.remove(chosen)

    def angel_blessing_limit(self):
        return 2 if self.op.module_id == "uniequip_002_angel" and self.op.module_level >= 2 else 1

    def angel_ally_eligible(self, unit):
        # Profession mask 639 includes the eight operator classes, not tokens
        # or traps. TargetFree and HealFree do not disqualify this buff.
        return (unit is not self.op and not unit.dead and unit.char_id.startswith("char_")
                and unit.deployment_index not in self.angel_blessings)

    def add_angel_blessing(self, target):
        atk = float(self.talent(2).get("atk", 0))
        hp = float(self.talent(2).get("max_hp", 0))
        self.angel_blessings[target.deployment_index] = (target, atk, hp)
        target.roster_atk_bonus = getattr(target, "roster_atk_bonus", 0) + atk
        target.roster_hp_bonus = getattr(target, "roster_hp_bonus", 0) + hp
        self.b._squad_field_signature = None
        self.b._sync_squad_effects()

    def bless_angel_ally(self, unit):
        if (self.op.char_id != "char_103_angel" or self.op.dead or not self.angel_blessing_started
                or not self.talent(2) or self.angel_blessing_count >= self.angel_blessing_limit()
                or not self.angel_ally_eligible(unit)):
            return
        self.add_angel_blessing(unit)
        # A departing recipient does not refund the lifetime selection count.
        self.angel_blessing_count += 1
        self.b.mechanic_diagnostics["ANGEL_BLESSING_FIRST_BATCH_PHASE_CANDIDATE"] = 1

    def clear_angel_blessings(self, target=None):
        changed = False
        for ident, (unit, atk, hp) in list(self.angel_blessings.items()):
            if target is not None and unit is not target:
                continue
            unit.roster_atk_bonus -= atk
            unit.roster_hp_bonus -= hp
            self.angel_blessings.pop(ident)
            changed = True
        if changed:
            self.b._squad_field_signature = None
            self.b._sync_squad_effects()

    def tick_soul_wards(self):
        if self.ward_spawn_at is not None and self.b.time + 1e-9 >= self.ward_spawn_at:
            self.ward_spawn_at = None
            self.spawn_soul_wards(1)
        self.clear_ward_camouflage()

    def clear_ward_camouflage(self, *, force=False):
        for key, unit in list(self.ward_camouflage_sources.items()):
            if force or unit.dead or self.op.dead:
                self.op.statuses.revoke(Status.CAMOUFLAGE, ("SOUL_WARD", key))
                self.ward_camouflage_sources.pop(key)

    def spawn_soul_wards(self, count, *, first_sp=0.0):
        """Persistent, host-owned summons; no timed despawn or deployment slot."""
        b, op = self.b, self.op
        if op.dead or "2" not in op.talent_blackboards:
            return []
        char = b.characters.get(SOUL_WARD)
        if char is None:
            b.mechanic_diagnostics["WISDEL_SOUL_WARD_DATA_MISSING"] = 1
            return []
        from . import data as D
        elite = min(op.build_plan.elite, len(char["phases"]) - 1)
        level = min(op.build_plan.level, char["phases"][elite]["maxLevel"])
        attrs = D.char_attributes(char, level, elite)
        cap = int(attrs.get("maxDeployCount", 3))
        living = sum(not unit.dead and unit.char_id == SOUL_WARD for unit in self.tokens)
        choices = [cell for cell in op.range_cells
                   if 0 <= cell[0] < len(b.map_grid) and 0 <= cell[1] < len(b.map_grid[0])
                   and any(b.is_buildable(*cell, position) for position in ("MELEE", "RANGED"))
                   and not b.tile_occupied(*cell)]
        # NO_CHARACTER_AND_NEAREST. Stable map order is still a candidate.
        choices.sort(key=lambda cell: (sum((cell[i]-op.tile[i])**2 for i in (0, 1)),
                                       -cell[0], cell[1]))
        output = []
        cfg = b.attack_timing.get(op.char_id, {}).get("soul_ward", {})
        for index, cell in enumerate(choices[:max(0, min(int(count), cap-living))]):
            plan = replace(op.build_plan, char_id=SOUL_WARD, elite=elite, level=level,
                tile=cell, time=b.time, trust=0, potential_rank=0, module_id=None,
                module_level=0, skill_id="sktok_wisdel_wward", skill_level=1, auto_skill=False)
            unit = b._make_operator(plan, 0)
            unit.summoner = op
            unit.uses_deployment_slot = False
            unit.sp = min(unit.sp_max, float(first_sp) if index == 0 else 0.0)
            unit.damage_type = "ARTS"
            unit.statuses.apply(Status.HEAL_FREE, None, resistible=False)
            b.operators.append(unit)
            self.tokens.append(unit)
            output.append(unit)
            # CreateBuffInRange is an entry action, not a continuous aura.
            cells = b.range_cells_from_id(cfg.get("camouflage_range", "x-5"), cell, 0)
            if op.tile in cells and op.statuses.targetable:
                if op.statuses.grant(Status.CAMOUFLAGE, ("SOUL_WARD", unit.deployment_index)):
                    self.ward_camouflage_sources[unit.deployment_index] = unit
        return output

    def soul_ward_skill_started(self):
        if self.op.char_id == "char_1035_wisdel" and self.sid == "skchr_wisdel_3":
            self.spawn_soul_wards(int(self.value("max_cnt", default=2)), first_sp=self.value("sp"))
            self.b.mechanic_diagnostics["WISDEL_S3_MAIN_ATTACK_INCOMPLETE"] = 1

    def ammo_used(self, user, amount):
        if (amount <= 0 or self.op.char_id != "char_1041_angel2"
                or self.op.dead or not self.talent(1)):
            return
        t = self.talent(1)
        successes = 0
        for _ in range(int(amount)):
            self.b._resolve_operator_heal(self.op, self.op,
                self.op.max_hp * float(t.get("hp_ratio", 0)), secondary=True)
            if self.b._roll_probability(float(t.get("prob", 0)), mechanism="angel2_ammo_bomb", source=self.op, target=user):
                successes += 1
        # The template marks the consuming unit's range once, before casting.
        marked = self.candidates(cells=user.range_cells)
        cfg = self.b.attack_timing.get(self.op.char_id, {}).get("ammo_bomb", {})
        for index in range(successes):
            self.events.append((self.b.time + index * float(cfg.get("interval", .10000000149011612)),
                "AMMO_BOMB_EMIT", marked, float(t.get("aoe_atk_scale", 0))))
        if successes:
            self.b.mechanic_diagnostics["ANGEL2_BOMB_STOP_PHASE_CANDIDATE"] = 1

    def tick_ammo_bombs(self):
        for bomb in list(self.ammo_bombs):
            if self.b.time + 1e-9 < bomb["stop"]:
                continue
            for enemy in self.wisdel_targets(bomb["pos"], bomb["radius"], ground=False):
                self.damage(enemy, bomb["amount"], "PHYSICAL", attack_type=AttackType.NORMAL,
                            apply_way=ApplyWay.RANGED)
            self.ammo_bombs.remove(bomb)

    def tick_snow(self):
        b, op = self.b, self.op
        t = self.talent(1)
        gates = {p.tile for p in self.tokens if not p.dead}
        while b.time + 1e-9 >= self.next_snow:
            self.next_snow += float(t.get("interval", 5.5))
            for cell in set(self.snow) | op.base_range_cells:
                if b._ground_passable_cell(*cell) and cell not in gates:
                    self.snow[cell] = min(int(t.get("max_cast_cnt", 5)), self.snow.get(cell, 0) + 1)
            if self.active:
                cap = int(self.value("talent@max_cast_tile_count", default=20))
                # Direction order is established; tile receiver order remains a candidate.
                for r, col in sorted(cell for cell, layers in self.snow.items() if layers >= 5):
                    for cell in ((r + 1, col), (r, col - 1), (r - 1, col), (r, col + 1)):
                        if cell in self.snow or cell in gates or not b._ground_passable_cell(*cell):
                            continue
                        if cell not in op.base_range_cells and sum(p not in op.base_range_cells for p in self.snow) >= cap:
                            continue
                        self.snow[cell] = 1
                        break
                for cell, layers in list(self.snow.items()):
                    if layers >= 5 and b.tile(*cell).get("tileKey") == "tile_end":
                        self.freeze_gate(cell)
                        self.snow.pop(cell, None)
                        gates.add(cell)
                b.mechanic_diagnostics["SBELL2_SNOW_EXPANSION_ORDER_CANDIDATE"] = 1
        previous = self.snow_occupants
        current = {e.enemy_index: (e, e.tile()) for e in b.enemies if b._enemy_is_attackable(e)
                   and not b._is_airborne(e) and e.tile() in self.snow}
        for key, (enemy, cell) in previous.items():
            if key not in current or current[key][1] != cell:
                self.snow.pop(cell, None)
                if self.active and not enemy.dead:
                    enemy.statuses.apply(Status.COLD, self.value("talent@cold"))
        for key, (enemy, cell) in current.items():
            if key not in previous or previous[key][1] != cell:
                self.damage(enemy, b.operator_attack_power(op) * float(t.get("talent_magic_scale", 0)))
            if self.active:
                self.damage(enemy, b.operator_attack_power(op) * self.value("talent@s2_magic_scale") * b.dt)
        self.snow_occupants = current

    def freeze_gate(self, cell):
        b, op = self.b, self.op
        key = "token_10058_sbell2_icetgt"
        char = b.characters.get(key)
        if char is None:
            b.mechanic_diagnostics["SBELL2_GATE_DATA_MISSING"] = 1
            return
        elite = min(op.build_plan.elite, len(char["phases"]) - 1)
        plan = replace(op.build_plan, char_id=key, elite=elite,
            level=min(op.build_plan.level, char["phases"][elite]["maxLevel"]),
            tile=cell, trust=0, potential_rank=0, module_id=None, module_level=0,
            skill_id=None, skill_level=None, auto_skill=False)
        unit = b._make_operator(plan, 0)
        unit.range_cells = unit.base_range_cells = set()
        unit.summoner = op
        unit.uses_deployment_slot = False
        unit.manual_retreat_forbidden = True
        unit.statuses.apply(Status.HEAL_FREE, float("inf"), resistible=False)
        unit.statuses.apply(Status.DISARMED, float("inf"), resistible=False)
        b.operators.append(unit)
        self.tokens.append(unit)
        b.mechanic_diagnostics["SBELL2_GATE_BLOCK_HANDOFF_CANDIDATE"] = 1

    def launch_notes(self, skill, *, index=None):
        op, b = self.op, self.b
        target = b._current_operator_target(op)
        forward = FACING[op.facing % 4]
        angles = [-13.125 + i * 3.75 for i in range(8)] if skill else [
            b.random_uniform(-float(self.talent(1).get("attack@angle", 20)),
                              float(self.talent(1).get("attack@angle", 20)),
                              mechanism="sakiko_launch_angle", source=op)]
        for i, angle in enumerate(angles):
            if index is not None and i != index:
                continue
            radians = math.radians(angle)
            direction = (forward[0] * math.cos(radians) - forward[1] * math.sin(radians),
                         forward[0] * math.sin(radians) + forward[1] * math.cos(radians))
            scale = self.value("atk_scale" if i == 0 else f"atk_scale_{i + 1}", default=1) if skill else 1
            sign = 1.0
            if target is None and not skill:
                sign = 1 if b._roll_probability(.5, source=op, mechanism="sakiko_sine_direction") else -1
            b.projectile_counter += 1
            ranged = not any(b._enemy_is_attackable(enemy) for enemy in op.blocked)
            lord_scale = float(op.trait_blackboard.get("atk_scale", .8)) if ranged else 1.0
            if skill and op.module_id == "uniequip_002_oblvns" and op.module_level >= 2:
                lord_scale = 1.0
            self.notes.append(Note(b.projectile_counter, op.tile, direction,
                b.operator_attack_power(op) * scale * lord_scale, target, b.time, skill, sign, ranged=ranged))
            note = self.notes[-1]
            key = f"projectile_chr_oblvns_s1_{i+1}" if skill else (
                "projectile_chr_oblvns" if target is not None else "projectile_chr_oblvns_talent")
            # The selected S1 children share the reviewed SPECIAL_3 binding.
            binding = "projectile_chr_oblvns_s1_1" if skill else key
            origin, mount = b._operator_projectile_origin(op, binding, target)
            note.key = key
            note.motion = op.skill_attack_timing.get(self.sid, {}).get("note_motion", {}).get(key, {})
            note.next_update = b.time  # PreciseTimer.Reset(period, waitFirst=False).
            note.free_ready_at = b.time + float(note.motion.get("trace_delay", .6 if skill else .1))
            note.speed = float(note.motion.get("free", {}).get("speed", 1.7 if skill else (2 if target else 1.3)))
            note.origin = origin
            note.position_row, note.position_col, note.position_z = origin
            note.free_origin = origin[:2]
            note.mount_point = mount
            b.mechanic_diagnostics["SAKIKO_NOTE_CHILD_MOUNT_AND_HEIGHT_CANDIDATE"] = 1

    def _note_trace_target(self, note):
        b = self.b
        if note.target is not None and b._enemy_is_attackable(note.target):
            return note.target
        selector = note.motion.get("selector", {})
        candidates = [enemy for enemy in b.enemies if b._enemy_is_attackable(enemy)
            and b._enemy_targetable(enemy) and not enemy.statuses.has(Status.CAMOUFLAGE)
            and enemy.enemy_index not in note.traced_targets
            and b._enemy_intersects_circle(enemy, note.position(), float(selector.get("radius", 1)))]
        # The range selector limits HATRED_DES results before the controller
        # sorts those results by distance to the projectile.
        candidates.sort(key=cmp_to_key(lambda a, z: compare_operator_target_keys(
            operator_target_key(a), operator_target_key(z))))
        limit = max(0, int(self.value(selector.get("max_target_key", "max_target"),
                                     default=selector.get("limit", 2))))
        return min(candidates[:limit], key=lambda enemy: (
            math.dist(enemy.position(), note.position()), enemy.creation_index), default=None)

    def _reset_note_free(self, note):
        note.target = None
        note.state = "FREE"
        note.state_at = self.b.time
        note.free_origin = note.position()
        length = math.hypot(*note.direction)
        note.free_direction = tuple(v/length for v in note.direction) if length > 1e-12 else (0., 0.)
        note.free_elapsed = note.free_distance = 0.0
        if not note.skill and not note.initial_target:
            note.sign = 1 if self.b._roll_probability(.5, source=self.op,
                mechanism="sakiko_sine_direction") else -1
        note.speed = float(note.motion.get("free", {}).get("speed",
            1.7 if note.skill else (2 if note.initial_target else 1.3)))

    def tick_notes(self):
        b, op = self.b, self.op
        active = []
        for note in self.notes:
            if b.time <= note.born + 1e-9:
                active.append(note); continue
            interval = float(note.motion.get("update_interval", .2 if note.skill else .4))
            if b.time + 1e-9 >= note.next_update:
                note.next_update += interval
                if note.state == "FREE":
                    if note.target is not None and not b._enemy_is_attackable(note.target):
                        note.target = None
                    if op.dead and note.target is None:
                        note.dead = True
                        continue
                    target = self._note_trace_target(note)
                    # A searched target before the delay is not a projectile
                    # trace target yet; the next FREE check searches again.
                    if target is not None and b.time + 1e-9 >= note.free_ready_at:
                        note.target = target
                        note.traced_targets.add(target.enemy_index)
                        note.state = "TRACE"; note.state_at = b.time
                        note.speed = float(note.motion.get("trace", {}).get("speed", 2.2 if note.skill else 2))
                elif note.target is None or not b._enemy_is_attackable(note.target):
                    self._reset_note_free(note)
            old_position = note.position()
            if note.state == "TRACE" and note.target is not None:
                if not b._enemy_is_attackable(note.target):
                    active.append(note)
                    continue
                delta = (note.target.position_row - note.position_row, note.target.position_col - note.position_col)
                distance = math.hypot(*delta)
                profile = note.motion.get("trace", {})
                speed = note.speed = movement_speed(note.speed, profile, b.dt)
                if distance <= speed * b.dt:
                    self.damage(note.target, note.amount, "ARTS", attack_type=AttackType.NORMAL,
                        apply_way=ApplyWay.RANGED if note.ranged else ApplyWay.MELEE)
                    note.dead = True
                    continue
                turn_speed = float(profile.get("turn_speed", 5 if note.skill or note.initial_target else 7))
                note.direction = steer_direction(note.direction, delta, turn_speed, b.dt,
                    normalize=not profile.get("add_inertia", True))
            else:
                speed = note.speed = movement_speed(note.speed, note.motion.get("free", {}), b.dt)
            note.position_row += note.direction[0] * speed * b.dt
            note.position_col += note.direction[1] * speed * b.dt
            if note.state == "FREE" and not note.skill and not note.initial_target:
                note.free_elapsed += b.dt
                params = {item["key"]: item["value"] for item in note.motion.get("free_data", [])}
                note.free_distance, lateral = stepped_sine_position(note.free_elapsed, params, note.sign)
                dr, dc = note.free_direction
                note.position_row = note.free_origin[0] + dr * note.free_distance - dc * lateral
                note.position_col = note.free_origin[1] + dc * note.free_distance + dr * lateral
                delta = tuple(a-z for a, z in zip(note.position(), old_position))
                length = math.hypot(*delta)
                note.direction = tuple(v/length for v in delta) if length > 1e-12 else (0., 0.)
            note.travelled += math.dist(old_position, note.position())
            inside = (round(note.position_row), round(note.position_col)) in op.range_cells
            if inside or note.target is not None:
                note.outside_at = None
            elif note.outside_at is None:
                note.outside_at = b.time
            if note.outside_at is not None and b.time - note.outside_at >= float(self.talent(1).get("delay", 1)):
                note.dead = True
                continue
            active.append(note)
        self.notes = active

    def initialize_drone_origin(self, drone, index):
        timing = self.op.skill_attack_timing.get(self.sid, {})
        keys = timing.get("drone_keys", [])
        key = keys[index % len(keys)] if keys else "projectile_chr_whitw2_funnel_s3_1"
        origin, mount = self.b._operator_projectile_origin(self.op, key, self.b._current_operator_target(self.op))
        radius = float(timing.get("drone_initial_radius", .5))
        angle = drone["angle"]
        motion = timing.get("drone_motion", {})
        drone.update(origin=origin, mount=mount, key=key,
            speed=float(motion.get("scatter", {}).get("speed", .1)),
            pos=(origin[0]+radius*math.sin(angle),origin[1]+radius*math.cos(angle)))
        self.b.mechanic_diagnostics["WHITW2_COMPOSITE_MOUNT_AND_INITIAL_ANGLE_CANDIDATE"] = 1

    def funnel_scale(self, hits):
        trait, talent = self.op.trait_blackboard, self.talent(1)
        cap = float(trait.get("max_atk_scale", 1.1))
        if self.b.time - self.op.deployed_at >= float(talent.get("interval", math.inf)):
            cap *= float(talent.get("scale", 1))
        return min(cap, float(trait.get("init_atk_scale", .2))
                   + float(trait.get("delta_atk_scale", .15)) * hits)

    def funnel_stack_limit(self):
        trait, talent = self.op.trait_blackboard, self.talent(1)
        if self.b.time - self.op.deployed_at + 1e-9 < float(talent.get("interval", math.inf)):
            return max(0, int(trait.get("max_stack_cnt", 6)))
        cap = float(trait.get("max_atk_scale", 1.1)) * float(talent.get("scale", 1))
        delta = float(trait.get("delta_atk_scale", .15))
        return max(0, math.ceil((cap - float(trait.get("init_atk_scale", .2))) / delta - 1e-9)) if delta > 0 else 0

    def tick_drones(self):
        if not self.active or not self.drones:
            return
        b, op = self.b, self.op
        timing = op.skill_attack_timing.get(self.sid, {})
        motion = timing.get("drone_motion", {})
        scatter = motion.get("scatter", {"dynamic": True, "acceleration": 1.9, "final_speed": 2})
        trace = motion.get("trace", {"dynamic": True, "acceleration": 1, "final_speed": 4, "turn_speed": 5, "speed": 2})
        base_count = 1 + int(self.value("attack@cnt"))
        if (len(self.drones) == base_count and b.time - op.deployed_at + 1e-9 >=
                float(self.talent(1).get("interval", 16)) * 3):
            forward = FACING[op.facing % 4]
            b.projectile_counter += 1
            self.drones.append({"pos": tuple(map(float, op.tile)), "target": None,
                "arrived": False, "id": b.projectile_counter,
                "angle": math.atan2(forward[0], forward[1]), "speed": .1,
                "phase": "SCATTER", "born": b.time, "next_attack": b.time, "hits": 0})
            self.initialize_drone_origin(self.drones[-1],len(self.drones)-1)
            b.mechanic_diagnostics["WHITW2_EXTRA_DRONE_ENTRY_CANDIDATE"] = 1
        for drone in self.drones:
            if drone["phase"] == "SCATTER":
                if b.time - drone["born"] < self.value("attack@times", default=1.3):
                    drone["speed"] = movement_speed(drone["speed"], scatter, b.dt)
                    angle = drone["angle"]
                    drone["pos"] = (drone["pos"][0] + math.sin(angle) * drone["speed"] * b.dt,
                                    drone["pos"][1] + math.cos(angle) * drone["speed"] * b.dt)
                    continue
                drone["phase"] = "CHASE"
                drone["speed"] = float(trace.get("speed", 2))
            target = drone["target"]
            if target is None or not b._enemy_is_attackable(target) or not b._enemy_targetable(target) or target.statuses.has(Status.CAMOUFLAGE):
                if target is not None:
                    # Native controller offsets the projectile's current position,
                    # not the vanished target's last ground coordinate.
                    lower = motion.get("relocate_min", {"x": -.75, "y": -.75})
                    upper = motion.get("relocate_max", {"x": .75, "y": .75})
                    drone["pos"] = tuple(value + b.random_uniform(
                            float(lower[("y", "x")[axis]]), float(upper[("y", "x")[axis]]),
                            mechanism=f"whitw2_drone_relocate_{axis}", source=op, target=target)
                        for axis, value in enumerate(drone["pos"]))
                    drone.pop("hover", None)
                candidates = [e for e in b.enemies if b._enemy_is_attackable(e) and b._enemy_targetable(e)
                              and not e.statuses.has(Status.CAMOUFLAGE)]
                target = min(candidates, key=lambda e: (math.dist(drone["pos"], e.position()),
                    math.dist(op.tile, e.position()), e.creation_index), default=None)
                drone["target"] = target; drone["arrived"] = False; drone["hits"] = 0
                drone["speed"] = float(trace.get("speed", 2))
                if target is not None:
                    drone["phase"] = "CHASE"
                    drone.pop("hover", None)
            if target is None:
                if "hover" not in drone:
                    hover = motion.get("hover", {})
                    angle = drone["angle"] if hover.get("keep_direction", True) else 0
                    drone["hover"] = HoverMotion(drone["pos"], (math.sin(angle), math.cos(angle)),
                        float(hover.get("radius", timing.get("drone_idle_radius", 1.25))),
                        float(hover.get("speed", timing.get("drone_idle_speed", 1))))
                    b.mechanic_diagnostics["WHITW2_HOVER_ENTRY_ORIENTATION_CANDIDATE"] = 1
                drone["phase"] = "HOVER"
                drone["pos"] = drone["hover"].advance(b.dt)
                drone["angle"] = math.atan2(*drone["hover"].direction)
                continue
            if drone["arrived"]:
                drone["pos"] = target.position()
            distance = math.dist(drone["pos"], target.position())
            drone["speed"] = movement_speed(drone["speed"], trace, b.dt)
            move = drone["speed"] * b.dt
            if distance <= move + .1:
                drone["pos"] = target.position()
                if not drone["arrived"]:
                    target.statuses.apply(Status.FEAR, self.value("attack@fear"))
                    target.fear_origin = op.tile
                    b._release_enemy_block(target)
                    drone["arrived"] = True
                    drone["next_attack"] = b.time
            elif distance:
                angle = drone["angle"]
                direction = steer_direction((math.sin(angle), math.cos(angle)),
                    (target.position_row-drone["pos"][0], target.position_col-drone["pos"][1]),
                    float(trace.get("turn_speed", 5)), b.dt)
                angle = math.atan2(*direction)
                drone["angle"] = angle
                drone["pos"] = (drone["pos"][0] + math.sin(angle) * move,
                                drone["pos"][1] + math.cos(angle) * move)
            if drone["arrived"] and b.time + 1e-9 >= drone["next_attack"]:
                scale = self.funnel_scale(drone["hits"])
                self.damage(target, b.operator_attack_power(op) * scale, attack_type=AttackType.BUFF)
                drone["hits"] += 1
                drone["next_attack"] = b.time + b._effective_attack_interval(op)
                if b.time - op.deployed_at >= float(self.talent(1).get("interval", 16)) * 2:
                    target.statuses.apply(Status.SILENCE, float(self.talent(1).get("attack@silence_duration", 2)))
        if self.b.time + 1e-9 >= self.next_state:
            self.next_state = self.b.time + 1.0
            affected = {e.enemy_index: e for drone in self.drones for e in self.near(drone["pos"], self.value("attack@range_radius"))}
            for enemy in affected.values():
                self.damage(enemy, b.operator_attack_power(op) * self.value("attack@magic_atk_scale"), attack_type=AttackType.BUFF)
                if b.time - op.deployed_at >= float(self.talent(1).get("interval", 16)) * 2:
                    enemy.statuses.apply(Status.SILENCE, float(self.talent(1).get("attack@silence_duration", 2)))
        b.mechanic_diagnostics["WHITW2_DRONE_ATTACK_PHASE_CANDIDATE"] = 1

    def tick_servant_queue(self):
        if self.op.dead:
            self.servant_queue.clear()
            return
        while self.servant_queue and self.b.time + 1e-9 >= self.servant_queue[0][0]:
            _, cell = self.servant_queue.pop(0)
            self.summon_servant(cell)
        if self.servant_queue:
            self.b.mechanic_diagnostics["NECRAS_SPAWN_QUEUE_PHASE_CANDIDATE"] = 1

    def summon_servant(self, cell):
        b, op = self.b, self.op
        if op.dead or not self.talent(1):
            return
        living = [p for p in self.tokens if not p.dead]
        if len(living) >= int(self.talent(1).get("max_token_cnt", 3)):
            normal = [p for p in living if p is not self.special_token
                      and not self.token_levels.get(p.deployment_index, 0)]
            normal.sort(key=lambda p: (p.hp, -p.deployment_index))
            if normal:
                self.upgrade_servant(normal[0], 1)
                self.regen(normal[0], normal[0].max_hp)
            return
        if cell not in op.range_cells or b.tile_occupied(*cell) or not b.is_buildable(*cell, "MELEE"):
            choices = [c for c in sorted(op.range_cells) if b.is_buildable(*c, "MELEE") and not b.tile_occupied(*c)]
            occupied = [c for c in choices if any(b._enemy_is_attackable(e) and e.tile() == c for e in b.enemies)]
            cell = b.choose_random_target(occupied or choices, mechanism="necras_summon_tile", source=op)
            if cell is None:
                return
        from .battle import _Operator
        from . import data as D
        key = "token_10043_necras_skeltn"
        char = b.characters.get(key)
        if not char:
            b.mechanic_diagnostics["NECRAS_TOKEN_DATA_MISSING"] = 1; return
        phase = min(len(char["phases"]) - 1, op.build_plan.elite)
        level = min(int(char["phases"][phase]["maxLevel"]), op.build_plan.level)
        attrs = D.char_attributes(char, level, phase)
        hp = float(attrs.get("maxHp", 1))
        b.creation_counter += 1; b.deployment_counter += 1
        unit = _Operator(key, char.get("name", key), cell, hp, hp,
            float(attrs.get("atk", 0)), float(attrs.get("def", 0)),
            float(attrs.get("magicResistance", 0)), int(attrs.get("blockCnt", 1)), 0,
            float(attrs.get("baseAttackTime", 1.6)), b.range_cells_from_id(char["phases"][phase]["rangeId"], cell, op.facing),
            base_max_hp=hp, facing=op.facing, creation_index=b.creation_counter, deployment_index=b.deployment_counter,
            deployed_at=b.time, first_update_frame=b.frame_index + 1)
        unit.base_range_cells = set(unit.range_cells)
        unit.summoner = op
        unit.damage_type = "ARTS"
        unit.prioritize_blocked = True
        unit.uses_deployment_slot = False
        unit.statuses.apply(Status.HEAL_FREE, None, resistible=False)
        self.tokens.append(unit); b.operators.append(unit)
        b.initialize_operator_target_selector(unit, first_search_frame=b.frame_index + 1)
        self.token_levels[unit.deployment_index] = 0
        if self.special_token is None or self.special_token.dead:
            self.special_token = unit
            t = self.talent(4)
            unit.base_max_hp *= 1 + float(t.get("max_hp", 2))
            unit.atk *= 1 + float(t.get("atk", 2))
            unit.defense *= 1 + float(t.get("def", 1))
            unit.res += float(t.get("magic_resistance", 10))
            unit.block_count += int(t.get("block_cnt", 1))
            unit.attack_interval += float(t.get("base_attack_time", .7))
            b._attach_skill(unit, NECRAS_SERVANT_SKILL)
            if unit.skill is not None:
                unit.auto = True
            else:
                b.mechanic_diagnostics["NECRAS_SERVANT_SKILL_DATA_MISSING"] = 1
            b._squad_field_signature = None; b._sync_squad_effects(); unit.hp = unit.max_hp
        unit.skill_runtime = NecrasServantRuntime(b, unit)
        mode = "special" if unit is self.special_token else "normal"
        cfg = b.attack_timing.get(key, {}).get("ordinary_modes", {}).get(mode, {})
        unit.attack_duration = float(cfg.get("attack_duration", 0))
        unit.attack_hit_time = float(cfg.get("attack_hit_time", 0))
        b.mechanic_diagnostics["NECRAS_TOKEN_SPAWN_ORDER_CANDIDATE"] = 1

    def upgrade_servant(self, unit, times):
        for _ in range(times):
            old = self.token_levels.get(unit.deployment_index, 0)
            cap = int(self.value("attack@max_stack_cnt", default=6)) if unit is self.special_token else 1
            if old >= cap:
                continue
            # Increases add to the original summoning attributes, not the last upgrade.
            if not hasattr(unit, "summon_base"):
                unit.summon_base = (unit.base_max_hp, unit.atk, unit.defense, unit.block_count)
            hp, atk, defense, block = unit.summon_base
            if unit is self.special_token:
                unit.base_max_hp += hp * self.value("attack@max_hp", default=.8) / (1 + float(self.talent(4).get("max_hp", 2)))
                unit.atk += atk * self.value("attack@atk", default=.8) / (1 + float(self.talent(4).get("atk", 2)))
                unit.defense += defense * self.value("attack@def", default=.4) / (1 + float(self.talent(4).get("def", 1)))
            else:
                unit.base_max_hp += hp * float(self.talent(1).get("max_hp", 0))
                unit.atk += atk * float(self.talent(1).get("atk", 0))
                unit.defense += defense * float(self.talent(1).get("def", 0))
                unit.block_count += int(self.talent(1).get("block_cnt", 0))
            self.token_levels[unit.deployment_index] = old + 1
            if unit is self.special_token and old + 1 in {3, 6}:
                unit.block_count += int(self.value("attack@block_cnt", default=1))
                dr, dc = FACING[unit.facing]
                steps = int(self.value("attack@ability_range_forward_extend", default=1))
                for distance in range(1, steps + 1):
                    forward = 1 + (old + 1) // 3 * steps - steps + distance
                    unit.base_range_cells.add((unit.tile[0] + dr * forward, unit.tile[1] + dc * forward))
                if unit.skill_runtime is not None:
                    unit.skill_runtime.refresh_range()
                else:
                    unit.range_cells = set(unit.base_range_cells)
                if unit.target_selector is not None:
                    unit.target_selector.mark_dirty()
            self.b._squad_field_signature = None; self.b._sync_squad_effects()

    def tick_medical_field(self):
        b, op = self.b, self.op
        t = self.talent(2)
        op.elements.max_value = op.max_hp
        if not t:
            return
        cells = b.range_cells_from_id("y-6", op.tile, op.facing)
        inside = {p.deployment_index for p in b.operators if not op.dead and not p.dead and p is not op
                  and p.char_id.startswith("char_") and p.tile in cells}
        for target in b.operators:
            if target.deployment_index not in inside or target.deployment_index in self.entered:
                continue
            expiry = b.time + float(t.get("buff_duration", 30))
            # The shield has infinite lifetime and is created only if absent,
            # including shields left by a previous deployment of this source.
            has_shield = any(p is target and rule.stacks > 0 and any(r is rule for r in target.damage_rules)
                             for runtime in effect_runtimes(b) for p, rule, _ in runtime.medical_shields)
            if not has_shield:
                rule = b.add_damage_rule(target, DamageRule(DamageRuleKind.SHIELD, stacks=1))
                self.medical_shields.append((target, rule, math.inf))
            amount = float(t.get("hp_recovery_per_sec", 50))
            if self.has_group_tag(target, "rhodes"):
                amount *= float(t.get("rhodes_bonus", 2))
            for runtime in effect_runtimes(b):
                runtime.medical_regen.pop(target.deployment_index, None)
            self.medical_regen[target.deployment_index] = (target, expiry, amount)
        self.entered = inside
        for ident, (target, expiry, amount) in list(self.medical_regen.items()):
            if target.dead or b.time >= expiry:
                self.medical_regen.pop(ident)
        for entry in list(self.medical_shields):
            target, rule, expiry = entry
            if target.dead or b.time >= expiry or rule.stacks <= 0:
                target.damage_rules[:] = [r for r in target.damage_rules if r is not rule]
                self.medical_shields.remove(entry)
        b.mechanic_diagnostics["KALTS2_FIELD_ENTRY_PHASE_CANDIDATE"] = 1

    def necras_action(self):
        b, op = self.b, self.op
        for enemy in self.candidates():
            self.damage(enemy, b.operator_attack_power(op) * self.value("attack@atk_scale"), attack_type=AttackType.NORMAL)
        normal = [p for p in self.tokens if not p.dead and p is not self.special_token]
        if normal and self.special_token is not None and not self.special_token.dead:
            unit = normal[0]
            times = 2 if self.token_levels.get(unit.deployment_index, 0) else 1
            unit.dead = unit.retreated = True
            for enemy in list(unit.blocked):
                b._release_enemy_block(enemy)
            for _ in range(times):
                self.regen(self.special_token, self.special_token.max_hp * self.value("attack@hp_ratio", default=.2))
                self.upgrade_servant(self.special_token, 1)
        self.stacks += 1
        if self.stacks >= int(self.value("attack@necras_s_3[attack_cnt].max_stack_cnt", default=2)):
            b._finish_skill(op, "ATTACK_COUNT_COMPLETE")

    def deploy_token(self, plan):
        from .operations import OperationAttempt
        from . import data as D
        b, op = self.b, self.op
        if plan.tile is None:
            return OperationAttempt("skip", "missing_tile")
        key = plan.char_id
        if key == "token_10068_kalts2_mtship":
            if not self.active or not self.anchor_ready:
                return OperationAttempt("wait", "anchor_not_ready")
        elif key == "token_10054_phatm2_encdool":
            if not self.active:
                return OperationAttempt("wait", "token_skill_not_active")
            if self.lure is not None:
                return OperationAttempt("wait", "token_already_deployed")
            if b.time < self.lure_ready_at:
                return OperationAttempt("wait", "token_redeploy_cooldown")
        elif b._active_operator(key) is not None:
            return OperationAttempt("skip", "already_deployed")
        row, col = plan.tile
        if not (0 <= row < len(b.map_grid) and 0 <= col < len(b.map_grid[0])):
            return OperationAttempt("skip", "out_of_bounds")
        if b.tile_occupied(row, col):
            return OperationAttempt("skip", "tile_occupied")
        position = b.characters[key].get("position", "MELEE")
        if not b.is_buildable(row, col, position):
            return OperationAttempt("skip", "not_buildable")
        if key == "token_10068_kalts2_mtship" and plan.tile in op.range_cells:
            return OperationAttempt("skip", "anchor_inside_range")
        attrs = D.char_attributes(b.characters[key], plan.level, plan.elite)
        cost = int(attrs.get("cost", 0))
        if b.dp - cost < b.cost_lower_bound:
            return OperationAttempt("wait", "insufficient_cost", {"cost": cost})
        b.dp -= cost
        unit = b._make_operator(plan, cost)
        unit.block_count = 0
        unit.statuses.apply(Status.FORCE_DISARMED, None, resistible=False)
        unit.statuses.apply(Status.UNTARGETABLE, None, resistible=False)
        unit.summoner = op
        unit.uses_deployment_slot = False
        b.operators.append(unit)
        self.tokens.append(unit)
        if key == "token_10054_phatm2_encdool":
            self.lure_ready_at = b.time + unit.respawn_time
            self.lure = {"unit": unit, "expiry": b.time + 10, "next": b.time,
                         "targets": set(), "atk": b.operator_attack_power(op)}
        elif key == "token_10068_kalts2_mtship":
            self.anchor_ready = False
            others = sorted([p for p in b.operators if p is not op and p is not unit and not p.dead
                             and p.char_id.startswith("char_") and p.tile in op.range_cells
                             and hasattr(p, "build_plan")], key=lambda p: (-p.deployed_at, p.creation_index))
            timing = op.skill_attack_timing.get(self.sid, {}).get("anchor", {})
            duration = math.dist(op.tile, plan.tile) / float(timing.get("speed", 3.5))
            launch_delay = float(timing.get("launch_delay", 0))
            self.anchor_move = {"unit": unit, "end": b.time + launch_delay + duration,
                                "start": b.time, "tile": plan.tile, "facing": plan.facing,
                                "withdraw_at": b.time + float(timing.get("withdraw_delay", .6)),
                                "withdraw_targets": others[:2],
                                "remaining_skill": max(0.0, op.active_until - b.time)}
            op.pending_attack = None
            for enemy in list(op.blocked):
                b._release_enemy_block(enemy)
            op.block_count = 0
            for status in (Status.INVINCIBLE, Status.FORCE_DISARMED, Status.UNTARGETABLE):
                op.statuses.apply(status, None, resistible=False)
            b.mechanic_diagnostics["KALTS2_ANCHOR_TRAVEL_EVENT_CANDIDATE"] = 1
        return OperationAttempt("ok", details={"deployment": unit.deployment_index, "owner": op.deployment_index, "cost": cost})

    def tick_lure(self):
        if self.lure is None:
            return
        from .induced_movement import paths_from
        b, lure = self.b, self.lure
        unit = lure["unit"]
        if unit.dead or b.time + 1e-9 >= lure["expiry"]:
            self.lure_arrived(None)
            return
        if b.time + 1e-9 < lure["next"]:
            return
        lure["next"] += .1
        candidates = self.candidates(cells=unit.range_cells)
        candidates.sort(key=lambda e: (str(e.level_type).upper() not in {"ELITE", "BOSS"}, e.creation_index))
        reachable = paths_from(b, unit.tile, limit=1000)
        for enemy in candidates:
            if len(lure["targets"]) >= int(self.value("max_target", default=4)):
                break
            if enemy.enemy_index in lure["targets"] or enemy.tile() not in reachable:
                continue
            expiry = b.time + 10 * (1 - enemy.statuses.resistance)
            if expiry <= b.time:
                continue
            lure["targets"].add(enemy.enemy_index)
            enemy.lure_target = (unit.tile, expiry, self)
            b._release_enemy_block(enemy)
        b.mechanic_diagnostics["PHATM2_LURE_NAVIGATION_CANDIDATE"] = 1

    def lure_arrived(self, enemy):
        if self.lure is None:
            return
        b, lure = self.b, self.lure
        unit = lure["unit"]
        unit.dead = unit.retreated = True
        b._process_operator_departure(unit, retreated=False)
        self.add_zone(unit.tile, self.value("buff_time", default=6), "LURE_BURST", radius=1.5)
        self.zones[-1]["atk"] = lure["atk"]
        self.zones[-1]["next"] = b.time + self.value("interval_damage", default=.5)
        for target in b.enemies:
            if getattr(target, "lure_target", None) and target.lure_target[2] is self:
                target.lure_target = None; target.induced_path = None
        self.lure = None

    def tick_anchor(self):
        if self.anchor_move is None:
            return
        b, op, state = self.b, self.op, self.anchor_move
        if b.time + 1e-9 >= state["withdraw_at"]:
            for ally in state["withdraw_targets"]:
                if ally.dead or not self.active:
                    continue
                self.relocation_reserve[ally.char_id] = ally.build_plan
                ally.dead = ally.retreated = True
                for enemy in list(ally.blocked):
                    b._release_enemy_block(enemy)
                b._process_operator_departure(ally, retreated=True, refund_withdraw_cost=False)
            state["withdraw_targets"] = []
        if op.active_until > 0:
            op.active_until = b.time + state["remaining_skill"]
        if b.time + 1e-9 < state["end"]:
            return
        unit = state["unit"]
        # The arrival action checks the destination again, after flight.
        occupant = b.tile_occupied(*state["tile"])
        can_land = (not unit.dead and self.active
                    and b.is_buildable(*state["tile"], b.characters[op.char_id]["position"])
                    and (occupant is None or occupant is unit))
        unit.dead = unit.retreated = True
        b._process_operator_departure(unit, retreated=False)
        if not can_land:
            op.dead = True
            for enemy in list(op.blocked):
                b._release_enemy_block(enemy)
            b._process_operator_departure(op, retreated=False)
            return
        # MoveLikeRespawnSelf finishes the old entity, then spawns from its card.
        # The explicit restore templates retain HP ratio and skill progress.
        hp_ratio = op.hp / op.max_hp
        reserve = dict(self.relocation_reserve)
        op.dead = True
        b._process_operator_departure(op, retreated=False, preserve_redeploy_state=True)
        replacement = b._make_operator(replace(op.build_plan, tile=state["tile"],
                                               facing=state["facing"], time=b.time), op.deployed_cost)
        b.operators.append(replacement)
        runtime = replacement.skill_runtime
        runtime.activate()
        runtime.anchor_ready = False
        runtime.relocation_reserve = reserve
        replacement.active_until = b.time + state["remaining_skill"]
        replacement.sp = 0
        replacement.graphic_animation = "Skill_3_Restart_End"
        replacement.graphic_started_at = b.time
        replacement.graphic_rate = 1.0
        b._squad_field_signature = None
        b._operator_deployed(replacement)
        replacement.hp = min(replacement.max_hp, replacement.max_hp * hp_ratio)
        b.mechanic_diagnostics["KALTS2_RESPAWN_LIFECYCLE_CANDIDATE"] = 1

    def redeploy_relocated(self, plan):
        from .operations import OperationAttempt
        b = self.b
        if plan.tile is None:
            return OperationAttempt("skip", "missing_tile")
        if plan.tile not in self.op.range_cells:
            return OperationAttempt("skip", "outside_relocation_range")
        original = self.relocation_reserve[plan.char_id]
        required = b.characters[plan.char_id].get("position")
        if not b.is_buildable(*plan.tile, required):
            return OperationAttempt("skip", "not_buildable")
        if b.tile_occupied(*plan.tile):
            return OperationAttempt("wait", "tile_occupied")
        if b._active_operator(plan.char_id) is not None:
            return OperationAttempt("skip", "already_deployed")
        occupied = sum(not p.dead and getattr(p, "uses_deployment_slot", True) for p in b.operators)
        if occupied >= b.character_limit:
            return OperationAttempt("wait", "deployment_limit", {"occupied_slots": occupied,
                "slot_limit": b.character_limit, "cost": 0})
        unit = b._make_operator(replace(original, tile=plan.tile, facing=plan.facing, time=b.time), 0)
        b.operators.append(unit)
        self.relocation_reserve.pop(plan.char_id)
        b._squad_field_signature = None
        b._operator_deployed(unit)
        return OperationAttempt("ok", details={"deployment": unit.deployment_index, "cost": 0, "relocated": True})

    def snapshot(self):
        return {"armed": self.armed, "coins": self.coins if self.sid == "skchr_swire2_3" else None,
                "endVolley": ({"kind": self.end_volley["kind"], "remaining": self.end_volley["remaining"]}
                              if self.end_volley is not None else None),
                "pendingHitFrames": ([self.op.pending_attack.execute_frame, *self.op.pending_attack.followup_frames]
                    if self.op.pending_attack and self.op.pending_attack.profile is not None
                    and self.op.pending_attack.profile.payload.get("sequenced") else None),
                "notes": len(self.notes) if self.sid == "skchr_oblvns_1" else None,
                "remnants": sorted(self.remnants) if self.op.char_id == "char_1035_wisdel" else None,
                "soulWards": [unit.deployment_index for unit in self.tokens if not unit.dead
                              and unit.char_id == SOUL_WARD] if self.op.char_id == "char_1035_wisdel" else None,
                "fever": self.fever if self.sid == "skchr_oblvns_1" else None,
                "feverActive": self.b.time < self.fever_until,
                "anchorReady": self.anchor_ready if self.sid == "skchr_kalts2_3" else None,
                "summonLevels": dict(self.token_levels) if self.sid == "skchr_necras_3" else None,
                "snow": [[*cell, layers] for cell, layers in sorted(self.snow.items())] if self.sid == "skchr_sbell2_2" else None}

    def visuals(self):
        from types import SimpleNamespace
        return [*self.notes, *[SimpleNamespace(projectile_id=f["id"], key="projectile_chr_angel2_t1", side="OPERATOR",
                position_row=f["pos"][0], position_col=f["pos"][1], position_z=0,
                travelled=0, initial_distance=0, parabolic=False, origin=f["origin"], mount_point=f["mount"])
                for f in self.ammo_bombs], *[SimpleNamespace(projectile_id=f["id"], key="projectile_chr_thorn2_s1", side="OPERATOR",
                position_row=f["pos"][0], position_col=f["pos"][1], position_z=0,
                travelled=math.dist(f["origin"][:2], f["pos"]), initial_distance=math.dist(f["origin"][:2], f["tile"]),
                origin=f["origin"], mount_point=f["mount"], parabolic=False) for f in self.alchemy_flights],
                *[SimpleNamespace(projectile_id=d["id"], key="lappland_drone", side="OPERATOR",
                position_row=d["pos"][0], position_col=d["pos"][1], position_z=0,
                travelled=0, initial_distance=0, parabolic=False,origin=d.get("origin"),
                mount_point=d.get("mount")) for d in self.drones],
                *[SimpleNamespace(projectile_id=z["id"],key="projectile_chr_hsgma2_s2",side="OPERATOR",
                    position_row=z["pos"][0],position_col=z["pos"][1],position_z=0,
                    travelled=0,initial_distance=0,parabolic=False,origin=z["origin"],
                    mount_point=z["mount"]) for z in self.zones if z["kind"]=="SHIELD_ORBIT" and "id" in z]]


class NecrasServantRuntime(OperatorSkillRuntime):
    """Servant attacks use block-scaled target counts; only the dragon has AUTO."""

    def __init__(self, battle, owner):
        super().__init__(battle, owner)
        self.next_hit = math.inf

    def profile(self):
        return AttackProfile(kind="ARTS", target_count=max(1, self.op.block_count))

    def candidates(self, *, cells=None, ground=True):
        values = super().candidates(cells=cells, ground=True)
        if self.active or cells is not None:
            return values
        blocked = [e for e in self.op.blocked if not self.b._is_airborne(e)
                   and self.b._validate_operator_target(self.op, e)]
        return blocked + [e for e in values if all(e is not unit for unit in blocked)]

    def begin_attack(self, first_attack):
        started = super().begin_attack(first_attack)
        if started:
            mode = "special" if self.op is self.op.summoner.skill_runtime.special_token else "normal"
            cfg = self.b.attack_timing.get(self.op.char_id, {}).get("ordinary_modes", {}).get(mode, {})
            self.op.graphic_animation = cfg.get("anim_key", "Attack" if mode == "special" else "Attack_2")
        return started

    def refresh_range(self):
        op = self.op
        if not self.active:
            op.range_cells = set(op.base_range_cells)
            return
        rid = op.skill.get("rangeId")
        cells = self.b.range_cells_from_id(rid, op.tile, op.facing) if rid else set(op.base_range_cells)
        host = op.summoner.skill_runtime
        steps = int(host.value("attack@ability_range_forward_extend", default=1))
        extend = host.token_levels.get(op.deployment_index, 0) // 3 * steps
        dr, dc = FACING[op.facing]
        front = max((r-op.tile[0])*dr + (c-op.tile[1])*dc for r, c in cells)
        for distance in range(1, extend + 1):
            cells.add((op.tile[0] + dr*(front+distance), op.tile[1] + dc*(front+distance)))
        op.range_cells = cells

    def activate(self):
        self.active = self.op.skill_active = True
        self.op.behavior_active = False
        self.op.active_until = self.b.time + float(self.op.skill.get("duration", 0))
        self.op.pending_attack = None
        self.op.attack_cd = 0.0
        self.op.attack_chain_active = False
        self.refresh_range()
        if self.op.target_selector is not None:
            self.op.target_selector.mark_dirty()
        cfg = self.b.attack_timing.get(self.op.char_id, {}).get("continuous_skill", {})
        self.next_hit = self.b.time + float(cfg.get("first_delay", 0))
        return True

    def override_action(self):
        return self.active

    def modify_packet(self, packet, target):
        return self.op.summoner.skill_runtime.modify_packet(packet, target)

    def tick(self):
        op, b = self.op, self.b
        if op.dead or not self.active:
            return
        cfg = b.attack_timing.get(op.char_id, {}).get("continuous_skill", {})
        interval = max(b.dt, float(cfg.get("interval", 1)))
        if b.time + 1e-9 < self.next_hit:
            return
        self.next_hit += (math.floor(max(0.0, b.time-self.next_hit)/interval) + 1)*interval
        if b.time >= op.active_until or not op.statuses.can_attack:
            return
        for enemy in self.candidates(ground=True):
            b._apply_operator_damage(op, enemy, b.operator_attack_power(op)*self.value("attack@atk_scale", default=1),
                                     attack_type=AttackType.NORMAL, apply_way=ApplyWay.MELEE, damage_type="ARTS")
        b.mechanic_diagnostics["NECRAS_SERVANT_CONTINUOUS_MODE_PHASE_CANDIDATE"] = 1

    def finish(self, reason):
        super().finish(reason)
        self.next_hit = math.inf
        self.op.attack_cd = 0.0
        self.op.attack_chain_active = False


class SoulWardRuntime(OperatorSkillRuntime):
    """Use the shared action/projectile path, with no ordinary attack trigger."""

    def __init__(self, battle, owner):
        super().__init__(battle, owner)
        self.no_target_at_emission = False
        battle.mechanic_diagnostics["WISDEL_WARD_ACTION_RANGE_SP_AND_PROJECTILE_CANDIDATE"] = 1

    def candidates(self, *, cells=None, ground=True):
        host = getattr(self.op, "summoner", None)
        if host is None or host.dead:
            return []
        self.op.range_cells = set(host.range_cells)
        self.op.base_range_cells = set(host.range_cells)
        values = super().candidates(cells=host.range_cells, ground=True)
        # Keep the normal hatred order within each marker-priority group.
        values.sort(key=lambda enemy: enemy.wisdel_remnant_source is not None)
        return values

    def can_cast(self):
        return (self.op.pending_attack is None and not self.active
                and self.op.statuses.can_attack and bool(self.candidates()))

    def auto_ready(self):
        return True

    def override_action(self):
        return True  # The prefab has an empty ordinary-attack trigger.

    def activate(self):
        from .battle import _PendingOperatorAttack
        b, op = self.b, self.op
        targets = self.candidates()[:1]
        if not targets:
            b._store_sp_gain(op, b._skill_cost(op))
            return False
        self.armed = self.active = op.skill_active = True
        op.active_until = -1.0
        op.behavior_active = False
        self.no_target_at_emission = False
        timing = op.skill_attack_timing.get(self.sid, {})
        self.action_end = b.time + float(timing.get("attack_duration", .2))
        frame, time = b._operator_attack_deadline(op)
        op.pending_attack = _PendingOperatorAttack(time, frame, "ROSTER", targets[0], targets,
            1.0, op.atk_scale, True, b._attack_hits(op, "ARTS"),
            profile=AttackProfile(kind="ARTS", skill=True,
                                 payload={"projectile_key":"projectile_chr_wisdel_token"}))
        b._register_ability_use(op)
        b._mark_operator_attack_started(op, False)
        op.graphic_rate = float(timing.get("animation_rate", 1))
        b._land_zero_delay_operator_attack(op)
        return True

    def profile(self):
        return AttackProfile(kind="ARTS", skill=True)

    def refresh_attack_targets(self, attack):
        targets = self.candidates()[:1]
        self.no_target_at_emission = not bool(targets)
        return replace(attack, primary=targets[0] if targets else attack.primary, targets=targets)

    def emit_soul_ward_projectile(self, attack):
        b = self.b
        cached = b.operator_attack_power(self.op, attack.source_atk_scale) * attack.attack_scale
        attack = replace(attack, profile=replace(attack.profile, payload={
            **attack.profile.payload, "cached_atk":cached}))
        before = len(b.projectiles)
        b._new_projectile(key="projectile_chr_wisdel_token", side="OPERATOR", source=self.op,
                          target=attack.primary, operator_attack=attack, cached_atk=cached,
                          will_hit=attack.will_hit)
        if len(b.projectiles) > before:
            # LinearMovement does not update its endpoint during flight.
            b.projectiles[-1].fixed_destination = b._projectile_target_position(attack.primary, None)

    def land(self, attack, target=None, *, ranged=False):
        enemy = target or (attack.targets[0] if attack.targets else None)
        if not attack.will_hit or enemy is None or not self.b._enemy_is_attackable(enemy):
            return
        host = getattr(self.op, "summoner", None)
        if host is not None and not host.dead and host.skill_runtime is not None:
            host.skill_runtime.attach_wisdel_remnant(enemy)
        enemy.statuses.apply(Status.SLUGGISH, self.value("sluggish"))
        amount = attack.profile.payload.get("cached_atk",
            self.b.operator_attack_power(self.op, attack.source_atk_scale) * attack.attack_scale)
        self.damage(enemy, amount, "ARTS", attack_type=AttackType.NORMAL,
                    apply_way=ApplyWay.RANGED if ranged else ApplyWay.MELEE)
        if host is not None and host.skill_runtime is not None:
            host.skill_runtime.clear_invalid_remnants()

    def attack_complete(self):
        # Ability finish, not projectile impact, releases SP recovery.
        pass

    def tick(self):
        host = getattr(self.op, "summoner", None)
        if host is not None and not host.dead:
            self.op.range_cells = set(host.range_cells)
            self.op.base_range_cells = set(host.range_cells)
        if self.active and self.b.time + 1e-9 >= self.action_end and self.op.pending_attack is None:
            self.b._finish_skill(self.op, "WARD_ACTION_COMPLETE")

    def finish(self, reason):
        active = self.active
        super().finish(reason)
        if active and reason == "WARD_ACTION_COMPLETE":
            if self.no_target_at_emission:
                self.b._store_sp_gain(self.op, self.b._skill_cost(self.op))
            else:
                amount = math.floor(self.b.random_uniform(self.value("sp_min"), self.value("sp_max"),
                    mechanism="wisdel_ward_finish_sp", source=self.op))
                # The template explicitly uses forceFlag, including SP_BLOCKED.
                self.b._store_sp_gain(self.op, amount)

    def depart(self):
        super().depart()
        host = getattr(self.op, "summoner", None)
        if host is not None and host.skill_runtime is not None:
            host.skill_runtime.clear_ward_camouflage()


@dataclass
class Note:
    projectile_id: int
    origin: tuple
    direction: tuple
    amount: float
    target: Any
    born: float
    skill: bool
    sign: float
    ranged: bool = True
    dead: bool = False
    state: str = "FREE"
    position_row: float = 0
    position_col: float = 0
    position_z: float = 0
    travelled: float = 0
    initial_distance: float = 0
    key: str = "sakiko_note"
    side: str = "OPERATOR"
    parabolic: bool = False
    outside_at: float | None = None
    motion: dict = field(default_factory=dict)
    speed: float = 0.0
    traced_targets: set = field(default_factory=set)
    free_elapsed: float = 0.0

    def __post_init__(self):
        self.position_row, self.position_col = self.origin
        self.initial_target = self.target is not None
        self.speed = 1.7 if self.skill else (2.0 if self.initial_target else 1.3)
        self.state_at = self.born
        self.next_update = self.born
        self.free_ready_at = self.born + (.6 if self.skill else .1)
        self.free_origin = self.origin
        self.free_direction = self.direction
        self.free_distance = 0.0

    def position(self):
        return self.position_row, self.position_col

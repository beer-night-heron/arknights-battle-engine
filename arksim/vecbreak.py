"""act3break_08 and explicitly selected battle supplies.

Table parameters and prefab actions are kept separate from unverified native
dispatch timing. This module never selects a supply or an operator for a plan.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from typing import Any

from . import data as D
from .behavior import BehaviorContext
from .mechanics import DamagePacket, DamageType, AttackType, ApplyWay, ElementType, Status, StatusController
from .runes import blackboard

STAGE = "act3break_08"
TRAP = "trap_339_tchray"
TOUCH = "enemy_8018_etouch_2"
POLLUT = {"enemy_8019_pollut", "enemy_8019_pollut_2", "enemy_8019_pollut_3"}
# Official Touch skeleton: Skill_2_Absorb lasts 3s, OnAttack at frame 26/30.
# Ability playback/notification quantization still needs native runtime evidence.
ABSORB_HIT_TIME = 26 / 30
ABSORB_ANIMATION_TIME = 3.0
MUTEX = [(f"act3break_rune{a:02}", f"act3break_rune{b:02}")
         for a, b in ((4, 5), (7, 8), (12, 13), (15, 16))]
PROFESSIONS = {"WARRIOR": 1, "SNIPER": 2, "TANK": 4, "MEDIC": 8,
               "SUPPORT": 16, "CASTER": 32, "SPECIAL": 64, "TOKEN": 128,
               "TRAP": 256, "PIONEER": 512}
CONTROL_GROUPS = {Status.COLD_FRIENDLY: Status.COLD, Status.COLD_ENEMY: Status.COLD,
                  Status.FROZEN_FRIENDLY: Status.FROZEN, Status.FROZEN_ENEMY: Status.FROZEN}
ADAPTIVE_CONTROLS = {Status.STUNNED, Status.COLD, Status.FROZEN, Status.FEAR, Status.PALSY_TREMOR}


class VecStatuses(StatusController):
    """Observe direct applications too; ordinary stages keep their controller."""
    def refresh_control_immunities(self):
        runtime = getattr(self, "runtime", None)
        enemy = getattr(self, "enemy", None)
        if runtime is None or enemy is None or enemy.key != TOUCH: return
        previous = getattr(self, "adaptive_immunities", set())
        innate = self.immunities - previous
        current = {effect for effect in ADAPTIVE_CONTROLS
                   if runtime.control_immune(enemy, effect.value)}
        current.update(alias for alias, group in CONTROL_GROUPS.items() if group in current)
        self.adaptive_immunities = current - innate
        self.immunities = innate | current

    def active_statuses(self):
        self.refresh_control_immunities()
        # Immunity invalidates the abnormal effect, without treating its Buff
        # notification as absent. Timed records can still expire normally.
        return super().active_statuses() - getattr(self, "adaptive_immunities", set())

    def has(self, status):
        self.refresh_control_immunities()
        effect = Status(str(getattr(status, "value", status)).upper())
        return effect not in getattr(self, "adaptive_immunities", set()) and super().has(effect)

    @property
    def attack_speed_delta(self):
        return -30.0 * sum(self.has(effect) for effect in (Status.COLD_FRIENDLY, Status.COLD_ENEMY))

    def tick(self, dt):
        self.refresh_control_immunities()
        super().tick(dt)

    def grant(self, status, owner):
        self.refresh_control_immunities()
        return super().grant(status, owner)

    def apply(self, status, duration, *, resistible=True, source="FRIENDLY"):
        effect = Status(str(getattr(status, "value", status)).upper())
        group = CONTROL_GROUPS.get(effect, effect)
        tracked = group in ADAPTIVE_CONTROLS
        self.refresh_control_immunities()
        if tracked and self.enemy.key == TOUCH and (duration is None or duration > 0):
            # The native listener observes ON_OTHER_BUFF_START, including a
            # Buff whose abnormal effect is already immune. Keep refreshing.
            self.runtime.add_control_resistance(self.enemy, group.value)
            self.refresh_control_immunities()
        if tracked and resistible and duration is not None:
            resistance = max(0.0, min(1.0, self.resistances.get(effect, self.resistance)))
            if self.enemy.key == TOUCH:
                resistance = max(resistance, self.runtime.control_resistance(self.enemy, group.value))
            scale = max([1.0] + [blackboard(r).get("one_minus_status_resistance", 1)
                        for r in self.runtime.runes if r["key"] == "enemy_attribute_mul"]) if group is not Status.PALSY_TREMOR else 1
            duration *= (1 - resistance) * scale
            resistible = False
        return super().apply(effect, duration, resistible=resistible, source=source)


def mode_for(stage: str, data: dict) -> dict:
    if stage != STAGE:
        return {}
    return next((mode for mode in data.get("activities", {}).values()
                 if stage in mode.get("squadBuffAvailStageList", [])), {})


def supply_catalog(stage: str, data: dict) -> dict:
    mode = mode_for(stage, data)
    return {"limit": int(mode.get("constData", {}).get("defenseEquipBuffLimit", 0)),
            "items": [{"id": key, "name": value["name"], "description": value["desc"],
                       "excludes": [other for pair in MUTEX if key in pair for other in pair if other != key]}
                      for key, value in mode.get("battleBuffDict", {}).items()]}


def select_supplies(level: dict, stage: str, selected: list[str] | None, data: dict) -> dict:
    selected = [] if selected is None else selected
    if not isinstance(selected, list) or any(not isinstance(key, str) for key in selected):
        raise ValueError("战备补给须为字符串数组")
    if len(set(selected)) != len(selected):
        raise ValueError("战备补给不能重复选择")
    mode = mode_for(stage, data)
    catalog = supply_catalog(stage, data)
    valid = {item["id"] for item in catalog["items"]}
    if set(selected) - valid:
        raise ValueError("该关卡不支持所选战备补给：" + ", ".join(sorted(set(selected) - valid)))
    if len(selected) > catalog["limit"]:
        raise ValueError(f"最多选择 {catalog['limit']} 项战备补给")
    if any(set(pair) <= set(selected) for pair in MUTEX):
        raise ValueError("同一战备补给的基础项和升级项不能同时选择")
    result = deepcopy(level)
    result["selectedBattleBuffs"] = list(selected)
    if mode:
        result["vecBreak"] = {"stage": stage, "runes": [deepcopy(rune)
            for key in reversed(selected) for rune in mode["battleBuffDict"][key]["runeData"]["runes"]]}
    return result


def predefined_builds(level: dict) -> dict[str, dict]:
    if level.get("vecBreak", {}).get("stage") != STAGE:
        return {}
    result = {}
    for card in level.get("predefines", {}).get("tokenCards", []):
        inst = card.get("inst", {})
        if inst.get("characterKey") == TRAP:
            result[TRAP] = {"char_id": TRAP, "elite": 0, "level": int(inst.get("level", 1)),
                           "trust": 0, "potential_rank": 0, "skill_id": None, "skill_level": None,
                           "module_id": None, "module_level": 0, "auto_skill": False}
    return result


def normalize_predefined(plan: Any, level: dict) -> Any:
    build = predefined_builds(level).get(plan.char_id)
    if build is None or plan.action != "DEPLOY":
        return plan
    # A stage-owned card's build comes from predefines, never from squad defaults.
    return replace(plan, **{key: value for key, value in build.items() if key != "char_id"})


def rune_matches(rune: dict, cid: str, character: dict) -> bool:
    board = blackboard(rune)
    if board.get("char") and board["char"] != cid:
        return False
    mask = rune.get("selector", {}).get("professionMask", "ALL")
    profession = character.get("profession")
    if isinstance(mask, int):
        return bool(mask & PROFESSIONS.get(profession, 0))
    return mask in (None, "ALL", "NONE") or mask == profession


class VecBreakRuntime:
    def __init__(self, battle):
        self.b = battle
        self.enabled = battle.level.get("vecBreak", {}).get("stage") == STAGE
        self.runes = battle.level.get("vecBreak", {}).get("runes", [])
        self.polluted: dict[tuple[int, int], dict] = {}
        self.exposure: dict[int, dict] = {}
        self.next_respawn: dict[str, float] = {}
        self.events: list[dict] = []
        self.bosses: dict[int, dict] = {}
        self.enemy_state: dict[int, dict] = {}
        self.traps: dict[int, dict] = {}
        self.absorb_dots: list[dict] = []
        self.heal_tick = 1.0
        self.released_waves: set[int] = set()
        if not self.enabled:
            return
        for rune in self.runes:
            board = blackboard(rune)
            if rune["key"] == "global_initial_cost_add":
                battle.initial_cost = min(battle.max_cost, battle.initial_cost + board["value"])
                battle.dp = battle.initial_cost
            elif rune["key"] == "global_placable_char_num_add":
                battle.character_limit += int(board["value"])
        environment = next((blackboard(r) for r in battle.level.get("runes", [])
                            if blackboard(r).get("key") == "act3break_pollute_tile"), {})
        for row, cells in enumerate(battle.map_grid):
            for col, tile in enumerate(cells):
                if any(x["key"] == "etouch_pollute" and x.get("value", 0) for x in battle.tile(row, col).get("blackboard") or []):
                    self.polluted[row, col] = dict(environment)
        battle.mechanic_diagnostics.pop("STAGE_RUNE_UNIMPLEMENTED:env_gbuff_new", None)
        battle.mechanic_diagnostics["VEC08_NATIVE_DISPATCH_TIMING_CANDIDATE"] = 1
        battle.mechanic_diagnostics["VEC08_SUMMON_NEAREST_EXIT_ROUTE_CANDIDATE"] = 1
        battle.mechanic_diagnostics["VEC08_CONTROL_BUFF_NOTIFICATION_CANDIDATE"] = 1
        battle.mechanic_diagnostics["VEC08_WAVE_RELEASE_CANDIDATE"] = 1
        battle.mechanic_diagnostics["VEC08_ABSORB_ACTION_TIMING_CANDIDATE"] = 1
        battle.mechanic_diagnostics["VEC08_TRACE_NAVIGATION_CANDIDATE"] = 1

    def emit(self, kind, **details):
        self.events.append({"t": round(self.b.time, 3), "fixedFrame": self.b.frame_index,
                            "kind": kind, **details})

    def matching(self, key, cid):
        return [blackboard(r) for r in self.runes if r["key"] == key
                and rune_matches(r, cid, self.b.characters[cid])]

    def env(self, key):
        return [blackboard(r) for r in self.runes if blackboard(r).get("key") == key]

    def char_bonus(self, cid, attr):
        return sum(float(board.get(attr, 0)) for board in self.matching("char_attribute_additive_mul", cid))

    def char_addition(self, cid, attr):
        return sum(float(board.get(attr, 0)) for board in self.matching("char_attribute_add", cid))

    def cost(self, cid, cost):
        for board in self.matching("char_cost_add", cid): cost += board.get("value", 0)
        for board in self.matching("char_cost_mul", cid): cost *= board.get("scale", 1)
        return max(0, math.floor(cost))

    def respawn(self, op):
        scale = self.next_respawn.pop(op.char_id, 1.0)
        for board in self.matching("char_respawntime_mul", op.char_id): scale *= board.get("scale", 1)
        return scale

    def enemy_data(self, key):
        return D.enemy_data_for_level(self.b.enemy_index, key, self.b.level)

    def born_enemy(self, enemy, data):
        state = self.enemy_state[enemy.enemy_index] = {"data": data, "entity": enemy, "born": self.b.time,
            "base_block": enemy.block_count, "charge": 0, "charge_clock": 0.0, "charged": True,
            "last_position": enemy.position(), "rush_contacts": set(), "weak_until": 0.0,
            "slow_until": 0.0, "slow": 0.0, "status_stacks": {}}
        for rune in self.runes:
            board = blackboard(rune)
            if board.get("enemy") and enemy.key not in str(board["enemy"]).split("|"): continue
            if rune["key"] == "enemy_attribute_mul":
                enemy.max_hp *= board.get("max_hp", 1)
                enemy.hp = enemy.max_hp
                enemy.atk *= board.get("atk", 1)
            if rune["key"] == "enemy_attribute_add":
                enemy.mass_level = max(0, enemy.mass_level + int(board.get("mass_level", 0)))
        if enemy.key in POLLUT: enemy.block_count = state["base_block"] = 2
        if enemy.key == "enemy_1330_cbrush_2": enemy.block_count = state["base_block"] = 3
        enemy.statuses = VecStatuses(durations=enemy.statuses.durations, immunities=enemy.statuses.immunities,
                                    resistance=enemy.statuses.resistance, grants=enemy.statuses.grants)
        enemy.statuses.runtime, enemy.statuses.enemy = self, enemy
        attributes = data.get("attributes", {})
        enemy.elements.resistance = float(D.mdef(attributes.get("epDamageResistance"), 0))
        for field, status in (("stunImmune", Status.STUNNED), ("sleepImmune", Status.SLEEP),
                              ("frozenImmune", Status.FROZEN), ("silenceImmune", Status.SILENCE),
                              ("levitateImmune", Status.LEVITATE), ("fearedImmune", Status.FEAR)):
            if D.mdef(attributes.get(field), False): enemy.statuses.immunities.add(status)
        if enemy.key == TOUCH:
            enemy.lethal_handler = lambda packet: self.reborn(enemy, packet)
            self.bosses[enemy.enemy_index] = {"phase": 1, "stacks": 6, "reborn_until": 0.0,
                "pollute_at": self.b.time + 7, "aura_at": self.b.time + 25,
                "aura_until": 0.0, "heal_at": 0.0, "block_until": 0.0,
                "summon_at": math.inf, "summon": None, "absorb": None, "trace": None}

    def talent(self, enemy, prefix):
        return {item["key"][len(prefix) + 1:]: item.get("valueStr") or item.get("value", 0)
                for item in self.enemy_state[enemy.enemy_index]["data"].get("talentBlackboard") or []
                if item["key"].startswith(prefix + ".")}

    def skill(self, enemy, name):
        item = next((s for s in self.enemy_state[enemy.enemy_index]["data"].get("skills") or []
                     if s.get("prefabKey") == name), {})
        return {**item, "board": blackboard(item)}

    def alive_enemies(self):
        return [e for e in self.b.enemies if not e.dead and not e.leaked]

    def allies(self):
        return [op for op in self.b.operators if not op.dead and op.char_id.startswith("char_")]

    def pollution_targets(self):
        # P1Pollute/RebornPollute: professionMask 639, target-free filtering,
        # and forceIgnoreCamouflage=false. A tile splash can still cover tokens.
        return [op for op in self.allies() if op.tile not in self.polluted
                and op.statuses.targetable and not op.statuses.has(Status.CAMOUFLAGE)
                and PROFESSIONS.get(self.b.characters[op.char_id].get("profession"), 0) & 639]

    def pollution_tiles(self, center):
        # PolluteSelect's x-5 is the cross, including its center.
        return [(center[0] + dr, center[1] + dc) for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= center[0] + dr < len(self.b.map_grid) and 0 <= center[1] + dc < len(self.b.map_grid[0])
                and (self.b.tile(center[0] + dr, center[1] + dc).get("buildableType") in ("MELEE", "RANGED", "ALL", 1, 2, 3)
                     or self.b._ground_passable_cell(center[0] + dr, center[1] + dc))
                and self.b.tile(center[0] + dr, center[1] + dc).get("tileKey") not in
                {"tile_start", "tile_end", "tile_hole"}]

    def pollute(self, enemy, targets):
        board = self.skill(enemy, "P1Pollute")["board"]
        changed = []
        for target in targets:
            for tile in self.pollution_tiles(target.tile):
                if tile not in self.polluted:
                    self.polluted[tile] = dict(board)
                    changed.append(list(tile))
        if changed: self.emit("vec_pollute", enemy=enemy.enemy_index, tiles=changed)

    def reborn(self, enemy, packet):
        boss = self.bosses[enemy.enemy_index]
        if boss["phase"] != 1 or packet.normalized_damage_type() is DamageType.LIFE_LOSS:
            return False
        boss["phase"] = 0
        boss["reborn_until"] = self.b.time + self.talent(enemy, "Reborn").get("duration", 5)
        boss["aura_until"] = 0.0
        enemy.hp = 1.0
        enemy.pending_attack = None
        enemy.statuses.grant(Status.INVINCIBLE, "vec-reborn")
        enemy.statuses.grant(Status.ROOT, "vec-reborn")
        enemy.statuses.grant(Status.DISARMED, "vec-reborn")
        self.b._release_enemy_block(enemy)
        self.emit("vec_reborn_start", enemy=enemy.enemy_index)
        return True

    def modify_packet(self, packet, target):
        if not self.enabled: return packet
        state = self.enemy_state.get(getattr(target, "enemy_index", -1))
        if state:
            scale = 1.0
            if target.key == TOUCH:
                boss = self.bosses[target.enemy_index]
                if packet.normalized_damage_type() in (DamageType.PHYSICAL, DamageType.ARTS):
                    scale *= max(0.0, 1 - boss["stacks"] * (0.1 if boss["phase"] == 1 else 0.08))
            if state.get("summoned") and packet.normalized_damage_type() is not DamageType.LIFE_LOSS:
                scale *= 1 - state["summon_board"].get("damage_resistance", 0.8)
            if target.level_type == "BOSS" and self.b.time < state["weak_until"]:
                scale *= max([1.0] + [x.get("damage_scale", 1) for x in self.env("act3break_weak[boss]")])
            packet = replace(packet, taken_scale=packet.taken_scale * scale)
        return packet

    def damage_landed(self, packet, target, result):
        if not self.enabled: return
        if getattr(target, "key", None) in POLLUT and result.final_amount > 0:
            source = packet.source
            if source in self.b.operators and not source.dead:
                amount = self.talent(target, "InverseDamage").get("value", 0)
                if amount: self.b.apply_element_damage(source, ElementType.ANGER, amount, source=target)
        if result.killed and target in self.b.enemies:
            self.died_enemy(target)

    def died_enemy(self, enemy):
        state = self.enemy_state.get(enemy.enemy_index, {})
        if state.get("death_notified"): return
        state["death_notified"] = True
        if enemy.key == TOUCH:
            self.bosses[enemy.enemy_index].update(phase=-1, aura_until=0.0, block_until=0.0,
                                                absorb=None, trace=None)
        if enemy.key in POLLUT and not enemy.leaked:
            for boss_id, boss in self.bosses.items():
                if boss["phase"] in (1, 2):
                    boss["stacks"] = max(0, boss["stacks"] - 1)
            for board in self.env("act3break_weak[boss]"):
                for target in self.alive_enemies():
                    if target.level_type == "BOSS":
                        self.enemy_state[target.enemy_index]["weak_until"] = self.b.time + board["duration"]
        self.emit("vec_enemy_finish", enemy=enemy.enemy_index, leaked=enemy.leaked)

    def normal_damage_type(self, enemy):
        return DamageType.ARTS if self.enabled and enemy.key in {TOUCH, "enemy_1328_cbjedi_2"} else DamageType.PHYSICAL

    def normal_attack(self, enemy, op, attack_override):
        if not self.enabled: return attack_override
        state = self.enemy_state.get(enemy.enemy_index, {})
        if enemy.key == TOUCH:
            phase = self.bosses[enemy.enemy_index]["phase"]
            board = self.talent(enemy, "Attack" if phase == 1 else "Reborn")
            self.b.apply_element_damage(op, ElementType.ANGER,
                self.b._effective_enemy_atk(enemy) * board.get("attack@ep_damage_ratio", 0.4), source=enemy)
        if enemy.key in {"enemy_1072_dlancer_2", "enemy_1330_cbrush_2"} and state.get("charged"):
            speed = state.get("attack_speed_snapshot", self.b._effective_enemy_move_speed(enemy)) / self.b.move_multiplier
            board = self.talent(enemy, "firstattack" if "dlancer" in enemy.key else "Combat")
            self.b._apply_damage(DamagePacket(amount=speed * board.get("atk_scale", 0),
                damage_type=DamageType.PHYSICAL, source=enemy, apply_way=ApplyWay.MELEE), op)
            state["charged"] = False
            state["charge"] = 0
            if "cbrush" in enemy.key:
                return self.b._effective_enemy_atk(enemy) * board.get("rush_damage", 0.5)
        return attack_override

    def move_speed(self, enemy, speed):
        if not self.enabled: return speed
        state = self.enemy_state.get(enemy.enemy_index)
        if not state: return speed
        bonus = 0.0
        if "dlancer" in enemy.key: bonus += state["charge"] * self.talent(enemy, "rush").get("dlancer_t[trigger].move_speed", 0.5)
        for boss in self.bosses.values():
            if self.b.time < boss["aura_until"]: bonus += 0.8
        if self.b.time < state["slow_until"]: bonus += state["slow"]
        speed = max(0.0, speed * (1 + bonus))
        if enemy.key == TOUCH and self.bosses[enemy.enemy_index]["phase"] == 1:
            speed = max(speed, enemy.move_speed * self.b.move_multiplier)
        if enemy.key in {"enemy_8019_pollut_2", "enemy_8019_pollut_3"}:
            speed = max(speed, self.talent(enemy, "MoveSpeedImmune").get("move_speed", 0.8) * self.b.move_multiplier)
        return speed

    def element_scale(self, target):
        if target not in self.b.operators: return 1.0
        return 1 - max([0.0] + [x.get("ep_damage_resistance", 0) for x in self.env("element_resistance")])

    def damage(self, target, amount, kind, source=None):
        result = self.b._apply_damage(DamagePacket(amount=amount, damage_type=kind, source=source,
            attack_type=AttackType.BUFF, notify_calculate=kind is not DamageType.LIFE_LOSS), target)
        if result.killed and target in self.b.enemies: self.b.enemies_killed += 1
        return result

    def born_operator(self, op):
        if not self.enabled: return
        if op.char_id != TRAP and op.tile in self.polluted:
            board = self.polluted[op.tile]
            self.exposure[op.deployment_index] = {"active": op.statuses.targetable, "board": board,
                "damage": board.get("base_damage", 100), "tick_at": self.b.time + 1}
        for board in self.env("modify_sp[born]"):
            if op.char_id.startswith("char_"): self.b._store_sp_gain(op, board.get("sp", 0))
        if op.char_id != TRAP: return
        op.uses_deployment_slot = False
        op.statuses.grant(Status.UNTARGETABLE, "vec-trap")
        op.statuses.grant(Status.DISARMED, "vec-trap")
        op.respawn_time = 30.0
        # Predefined token's first talent, not a player-selected operator skill.
        board = op.talent_blackboards.get("1", {})
        cleaned = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                tile = (op.tile[0] + dr, op.tile[1] + dc)
                if tile not in self.polluted: continue
                self.polluted.pop(tile)
                cleaned.append(list(tile))
                for ally in self.allies():
                    if ally.tile == tile:
                        self.b._apply_damage(DamagePacket(amount=board.get("value", 800), damage_type=DamageType.HEAL, source=op), ally)
                        self.b._gain_sp(ally, board.get("sp", 3))
                child = self.spawn(board.get("enemy_key", "enemy_8019_pollut"), tile, source=op)
                child.statuses.apply(Status.ROOT, board.get("unmovable_time", 3), resistible=False)
        for supply in self.env("act3break_modify_sp[character]"):
            for ally in self.allies(): self.b._gain_sp(ally, supply.get("sp", 0))
        for supply in self.env("act3break_move_speed[elite]"):
            for enemy in self.alive_enemies():
                if enemy.level_type == "ELITE":
                    state = self.enemy_state[enemy.enemy_index]
                    state["slow"], state["slow_until"] = supply["move_speed"], self.b.time + supply["duration"]
        channel = bool(self.matching("level_predefines_skill_replace", TRAP))
        self.traps[op.deployment_index] = {"op": op, "until": self.b.time + (10 if channel else 0.1),
                                         "channel_until": self.b.time + 10, "tick_at": self.b.time,
                                         "channel": channel}
        self.emit("vec_cleanse", deployment=op.deployment_index, tiles=cleaned, spawned=len(cleaned))

    def spawn(self, key, tile, source=None, follow=None, forward_index=None):
        from .battle import RoutePoint
        endpoints = [(i, route[-1]) for i, route in enumerate(self.b.routes)
                     if route and route[-1].kind.upper() == "END"]
        reachable = []
        start = self.b._position_cell(*tile)
        for index, point in endpoints:
            cells = self.b._spfa_cells(start, self.b._position_cell(point.row, point.col), force_goal=False)
            if cells: reachable.append((len(cells), index, point))
        if not reachable:
            passable = [(r, c) for r in range(len(self.b.map_grid)) for c in range(len(self.b.map_grid[0]))
                        if abs(r - tile[0]) <= 1 and abs(c - tile[1]) <= 1
                        and self.b.tile(r, c).get("passableMask") in ("ALL", "FLY_AND_WALK")]
            if not passable:
                # Search uses engine passability/path rules rather than a visual tile label.
                passable = [(r, c) for r in range(len(self.b.map_grid)) for c in range(len(self.b.map_grid[0]))
                            if abs(r - tile[0]) <= 1 and abs(c - tile[1]) <= 1
                            and any(self.b._spfa_cells((r, c), self.b._position_cell(p.row, p.col), force_goal=False)
                                    for _, p in endpoints)]
            if not passable: raise ValueError("机关召唤位置附近没有可达终点的地面格")
            tile = min(passable, key=lambda t: (math.dist(t, tile), t))
            start = tile
            for index, point in endpoints:
                cells = self.b._spfa_cells(start, self.b._position_cell(point.row, point.col), force_goal=False)
                if cells: reachable.append((len(cells), index, point))
        _, route_index, end = min(reachable, key=lambda t: (t[0], t[1]))
        enemy = self.b._make_enemy(key, route_index)
        if follow is None:
            enemy.route = self.b._expand_ground_route([RoutePoint(float(tile[0]), float(tile[1]), "START"),
                                                       RoutePoint(end.row, end.col, "END")], allow_diagonal=True)
        else:
            enemy.route = [RoutePoint(float(tile[0]), float(tile[1]), "START"),
                           *follow.route[follow.route_idx if forward_index is None else forward_index:]]
        enemy.path = self.b._routes_to_paths([enemy.route])[0]
        # route_idx is the last reached point; the ordinary mover uses +1.
        enemy.route_idx = 0
        enemy.position_row, enemy.position_col = map(float, tile)
        enemy.remaining_path_distance = self.b._targeting_path_distance(enemy)
        enemy.dont_block_wave = True
        if follow is not None: enemy.wave_index, enemy.fragment_index = follow.wave_index, follow.fragment_index
        self.b.enemies.append(enemy)
        self.b.enemies_spawned += 1
        self.emit("vec_summon", enemy=enemy.enemy_index, key=key, tile=list(tile))
        return enemy

    def depart(self, op):
        exposure = self.exposure.pop(op.deployment_index, None)
        if exposure and exposure["active"]:
            self.next_respawn[op.char_id] = exposure["board"].get("respawn_time", 2)
        self.traps.pop(op.deployment_index, None)

    def tick(self):
        if not self.enabled: return
        now = self.b.time
        for op in self.b.operators:
            if op.dead or op.char_id == TRAP: continue
            board = self.polluted.get(op.tile)
            exposure = self.exposure.get(op.deployment_index)
            active = board is not None and op.statuses.targetable
            if board is not None and not exposure:
                exposure = self.exposure[op.deployment_index] = {"active": active, "board": board,
                    "damage": board.get("base_damage", 100), "tick_at": now + 1}
            if exposure:
                if exposure["active"] and not active:
                    self.next_respawn[op.char_id] = exposure["board"].get("respawn_time", 2)
                if active and not exposure["active"]:
                    exposure.update(damage=board.get("base_damage", 100), tick_at=now + 1, board=board)
                exposure["active"] = active
                if active and now + 1e-9 >= exposure["tick_at"]:
                    exposure["damage"] = min(board.get("max_value", 1800), exposure["damage"] + board.get("add_value", 18))
                    self.damage(op, exposure["damage"], DamageType.ARTS)
                    if not op.dead: self.b.apply_element_damage(op, ElementType.ANGER, board.get("ep_damage", 40))
                    exposure["tick_at"] += 1
            for rune in self.runes:
                bb = blackboard(rune)
                if bb.get("key") in {"modify_sp_recover[support]", "modify_sp_recover[normal]"} and rune_matches(rune, op.char_id, self.b.characters[op.char_id]):
                    if op.sp_type == "INCREASE_WITH_TIME": self.b._gain_sp(op, bb["sp_recovery_per_sec"] * self.b.dt)
        if now + 1e-9 >= self.heal_tick:
            self.heal_tick += 1
            for op in self.b.operators:
                if not op.dead:
                    for board in self.env("hp_recovery_per_sec_by_max_hp_ratio[add]"):
                        self.b._apply_damage(DamagePacket(amount=op.max_hp * board["hp_recovery_per_sec_by_max_hp_ratio"],
                                                         damage_type=DamageType.HEAL, ignore_heal_free=True), op)
        for trap in list(self.traps.values()):
            op = trap["op"]
            if op.dead: continue
            if now + 1e-9 >= trap["until"]:
                op.dead, op.retreated = True, True
                self.b._process_operator_departure(op, retreated=True)
                continue
            if trap["channel"] and now < trap["channel_until"] - 1e-9:
                targets = [e for e in self.alive_enemies() if abs(e.position_row - op.tile[0]) <= 1.5
                           and abs(e.position_col - op.tile[1]) <= 1.5]
                for enemy in targets: enemy.statuses.apply(Status.SLUGGISH, 0.1, resistible=False)
                if now + 1e-9 >= trap["tick_at"]:
                    for enemy in targets: self.damage(enemy, 3500, DamageType.TRUE, op)
                    trap["tick_at"] += 1
        for boss_enemy in self.alive_enemies():
            if boss_enemy.enemy_index in self.bosses:
                self.tick_boss(boss_enemy, self.bosses[boss_enemy.enemy_index])
        for enemy in self.alive_enemies():
            state = self.enemy_state[enemy.enemy_index]
            enemy.block_count = state["base_block"] + (1 if any(now < b["block_until"] for b in self.bosses.values()) else 0)
            if enemy.key in {"enemy_1072_dlancer_2", "enemy_1330_cbrush_2"} and (
                    enemy.statuses.has(Status.STUNNED) or enemy.statuses.has(Status.ROOT) or enemy.statuses.has(Status.BIND)):
                state.update(charge=0, charge_clock=0.0, charged=False)
            elif "dlancer" in enemy.key:
                if enemy.statuses.has(Status.STUNNED) or enemy.statuses.has(Status.ROOT) or enemy.statuses.has(Status.BIND):
                    state.update(charge=0, charge_clock=0.0, charged=False)
                elif not enemy.blocked_by and enemy.statuses.can_move:
                    state["charged"] = True
                    state["charge_clock"] += self.b.dt
                    if state["charge_clock"] + 1e-9 >= 0.5:
                        state["charge_clock"] -= 0.5
                        state["charge"] = min(25, state["charge"] + 1)
            if state.get("summoned") and now + 1e-9 >= state["self_loss_at"]:
                state["self_loss_at"] += 1
                self.damage(enemy, enemy.max_hp * state["summon_board"].get("hp_ratio", 0.1), DamageType.LIFE_LOSS)
        for op in self.b.operators:
            if op.dead: continue
            while sum(e.block_count for e in op.blocked) > op.block_count:
                self.b._release_enemy_block(op.blocked[-1])
        for dot in list(self.absorb_dots):
            if now + 1e-9 >= dot["tick_at"] and now <= dot["until"] + 1e-9:
                for op in dot["targets"]:
                    if not op.dead: self.b.apply_element_damage(op, ElementType.ANGER, dot["board"].get("value", 250), source=dot["source"])
                dot["tick_at"] += 1
            if now > dot["until"]: self.absorb_dots.remove(dot)

    def tick_boss(self, enemy, boss):
        now = self.b.time
        if boss["phase"] == 0:
            if now + 1e-9 < boss["reborn_until"]: return
            boss.update(phase=2, stacks=10, pollute_at=now + 35, summon_at=now + 10)
            enemy.hp = enemy.max_hp
            for status in (Status.INVINCIBLE, Status.ROOT, Status.DISARMED): enemy.statuses.revoke(status, "vec-reborn")
            self.released_waves.add(enemy.wave_index)
            for managed in self.b.enemies:
                if managed.wave_index == enemy.wave_index: managed.dont_block_wave = True
            self.b.wave_scheduler.wave_departures[enemy.wave_index] = now
            self.pollute(enemy, self.pollution_targets())
            self.emit("vec_reborn_finish", enemy=enemy.enemy_index)
        if self.tick_absorb(enemy, boss): return
        if boss["phase"] == 2:
            child = boss["summon"]
            if child is not None and (child.dead or child.leaked):
                boss["summon"] = boss["trace"] = None
            elif child is not None and math.dist(child.position(), enemy.position()) <= self.talent(enemy, "AbsorbListener").get("range_radius", .6):
                self.start_absorb(enemy, boss, child)
                return
        if not enemy.statuses.can_use_skill:
            if boss["aura_until"]:
                boss["aura_until"] = 0
                self.emit("vec_aura_interrupted", enemy=enemy.enemy_index)
            return
        if now + 1e-9 >= boss["pollute_at"]:
            targets = [op for op in self.pollution_targets()
                       if math.dist(op.tile, enemy.position()) <= 4.5]
            if targets:
                target = max(targets, key=lambda op: (op.taunt_level, op.deployment_index))
                self.pollute(enemy, [target])
                boss["pollute_at"] = now + (75 if boss["phase"] == 1 else 55)
        if boss["phase"] == 1:
            if now + 1e-9 >= boss["aura_at"] and len(self.alive_enemies()) > 1:
                boss.update(aura_at=now + 35, aura_until=now + 8, heal_at=now + 4)
                self.emit("vec_aura_start", enemy=enemy.enemy_index)
            if boss["aura_until"] and now + 1e-9 >= boss["aura_until"]:
                boss["aura_until"] = 0
                boss["block_until"] = now + 30
            elif boss["aura_until"] and now + 1e-9 >= boss["heal_at"]:
                boss["heal_at"] += 1
                for ally in self.alive_enemies():
                    self.b._apply_damage(DamagePacket(amount=self.b._effective_enemy_atk(enemy) * 1.8,
                                                     damage_type=DamageType.HEAL, source=enemy), ally)
        else:
            child = boss["summon"]
            if child is not None and (child.dead or child.leaked): boss["summon"] = child = None
            if now + 1e-9 >= boss["summon_at"] and child is None:
                board = self.skill(enemy, "Summon")["board"]
                point, remaining = enemy.position(), board.get("dist", 5)
                forward_index = enemy.route_idx + 1
                for index, next_point in enumerate(enemy.route[forward_index:], forward_index):
                    if next_point.kind.upper() not in {"MOVE", "END", "NAVIGATE", "NAVIGATE_CENTER"}: continue
                    distance = math.dist(point, (next_point.row, next_point.col))
                    if distance >= remaining:
                        forward_index = index
                        ratio = remaining / distance if distance else 0
                        point = (point[0] + (next_point.row - point[0]) * ratio, point[1] + (next_point.col - point[1]) * ratio)
                        break
                    point = next_point.row, next_point.col
                    remaining -= distance
                    forward_index = index + 1
                if forward_index >= len(enemy.route):
                    boss["summon_at"] = now + self.b.dt
                    return
                child = self.spawn(board.get("enemy_key", "enemy_8019_pollut_3"), point, follow=enemy, forward_index=forward_index)
                self.enemy_state[child.enemy_index].update(summoned=True, summon_board=board,
                    self_loss_at=now + 1, summon_host=enemy.enemy_index)
                boss.update(summon=child, summon_at=now + 45, trace=None)

    def absorb_targets(self, enemy):
        # Native CreateBuffInRange: global DEFAULT category, no profession or
        # combat selector. In particular, this is not pollution's mask 639.
        context = BehaviorContext(source=enemy, target=enemy, battle=self.b)
        return self.b.behavior_range_targets(context, {
            "_sourceType": "SOURCE", "_targetType": "SOURCE", "_useGlobalRange": True,
            "_targetOptions": {"targetSide": "ENEMY", "targetMotion": "ALL",
                               "targetCategory": "DEFAULT", "enableAdvancedOptions": True,
                               "purposeMask": "NONE", "professionMask": "NONE",
                               "ignoreTargetFree": False}})

    def apply_absorb_buff(self, enemy, *, duration):
        board = self.talent(enemy, "AbsorbListener")
        affected = []
        for op in self.absorb_targets(enemy):
            # DEFAULT override preserves an existing same-source Buff; an
            # interrupted cast must not repeat its start hit or reset its clock.
            if any(dot["source"] is enemy and op in dot["targets"]
                   and self.b.time <= dot["until"] + 1e-9 for dot in self.absorb_dots):
                continue
            self.b._behavior_hit_damage(enemy, op,
                self.b._effective_enemy_atk(enemy) * board.get("atk_scale", 3.5),
                "ARTS", "NORMAL", "NONE", "")
            affected.append(op.deployment_index)
            if duration > 0 and not op.dead:
                self.absorb_dots.append({"source": enemy, "targets": [op], "board": board,
                    "tick_at": self.b.time + 1, "until": self.b.time + duration})
        return affected

    def consume_absorb_child(self, enemy, boss, child):
        if not child.dead and not child.leaked:
            self.damage(child, child.hp + 1, DamageType.LIFE_LOSS, enemy)
        if boss["summon"] is child: boss["summon"] = None
        boss["trace"] = None

    def start_absorb(self, enemy, boss, child):
        # The native TriggerEnemySkill explicitly clears palsy before trying.
        enemy.statuses.remove(Status.PALSY_TREMOR)
        enemy.attack_target = None
        enemy.attack_hit_frame = None
        enemy.attack_hit_at = -1.0
        enemy.attack_move_resume_frame = None
        enemy.attack_animation_end_frame = None
        enemy.navigation_velocity_row = enemy.navigation_velocity_col = 0.0
        if not enemy.statuses.can_use_skill:
            self.consume_absorb_child(enemy, boss, child)
            # The failure branch's LIMITED Buff has lifeTime=0 and no
            # durationKey; it does not inherit the successful cast's 8s clock.
            targets = self.apply_absorb_buff(enemy, duration=0)
            self.emit("vec_absorb", enemy=enemy.enemy_index, child=child.enemy_index,
                      reason="TRIGGER_FAILED", targets=targets, duration=0)
            return
        boss["absorb"] = {"child": child, "elapsed": 0.0, "updated": self.b.time,
                           "resolved": False}
        self.emit("vec_absorb_start", enemy=enemy.enemy_index, child=child.enemy_index)

    def finish_absorb(self, enemy, boss, reason):
        cast = boss["absorb"]
        if cast is None: return
        if reason in {"INTERRUPTED", "TARGET_DEAD"}:
            self.consume_absorb_child(enemy, boss, cast["child"])
            targets = self.apply_absorb_buff(enemy, duration=self.talent(enemy, "AbsorbListener").get("duration", 8))
            self.emit("vec_absorb", enemy=enemy.enemy_index, child=cast["child"].enemy_index,
                      reason=reason, targets=targets)
        boss["absorb"] = None
        self.emit("vec_absorb_end", enemy=enemy.enemy_index, reason=reason)

    def tick_absorb(self, enemy, boss):
        cast = boss["absorb"]
        if cast is None: return False
        if not enemy.statuses.can_use_skill:
            self.finish_absorb(enemy, boss, "INTERRUPTED")
            return True
        # _interuptIfTargetDead=false: losing the child before OnAttack does
        # not cancel the remaining cast. Its global damage action still runs.
        rate = min(1.0, enemy.attack_interval / self.b._effective_attack_interval(enemy))
        cast["elapsed"] += max(0.0, self.b.time - cast["updated"]) * rate
        cast["updated"] = self.b.time
        if not cast["resolved"] and cast["elapsed"] + 1e-9 >= ABSORB_HIT_TIME:
            cast["resolved"] = True
            self.consume_absorb_child(enemy, boss, cast["child"])
            duration = self.talent(enemy, "AbsorbListener").get("duration", 8)
            targets = self.apply_absorb_buff(enemy, duration=duration)
            self.emit("vec_absorb", enemy=enemy.enemy_index, child=cast["child"].enemy_index,
                      reason="ON_ATTACK", targets=targets, duration=duration)
        if cast["elapsed"] + 1e-9 >= ABSORB_ANIMATION_TIME:
            self.finish_absorb(enemy, boss, "NORMAL_EXIT")
        return True

    def before_enemy_tick(self, enemy):
        if not self.enabled: return False
        boss = self.bosses.get(enemy.enemy_index)
        if boss:
            return boss["phase"] == 0 or boss["absorb"] is not None or self.b.time < boss["aura_until"]
        return False

    def trace_summon(self, enemy):
        state = self.enemy_state.get(enemy.enemy_index, {})
        boss = self.bosses.get(state.get("summon_host"))
        if boss is None or boss["summon"] is not enemy: return False
        host = next((unit for unit in self.b.enemies
                     if unit.enemy_index == state["summon_host"]), None)
        if host is None or host.dead or host.leaked or boss["phase"] != 2:
            boss["trace"] = None
            return False
        # EnemyForceTracePosition._source is the destination, _target the mover:
        # Buff owner Touch is the destination; Buff source pollut is the mover.
        # The native trace Buff refreshes its destination every .5s. Keep a
        # cached passable route and the child's continuous endpoint, rather
        # than rebuilding a cell-center route on every enemy update.
        trace = boss["trace"]
        if trace is None or self.b.time + 1e-9 >= trace["refresh_at"]:
            destination = host.position()
            goal_cell = self.b._position_cell(*destination)
            cells = self.b._spfa_cells(enemy.tile(), goal_cell, force_goal=False)
            if not cells:
                self.b.mechanic_diagnostics["VEC08_TRACE_PATH_NOT_FOUND"] = 1
                return True
            route = [tuple(map(float, cell)) for cell in cells[1:]]
            if route: route[-1] = destination
            else: route.append(destination)
            trace = boss["trace"] = {"route": route, "refresh_at": self.b.time + .5,
                                     "goal_cell": goal_cell}
        route = trace["route"]
        while route and math.dist(enemy.position(), route[0]) <= .1: route.pop(0)
        if route:
            position = self.b._steered_next_position(enemy, target=route[0],
                                                     speed=self.b._effective_enemy_move_speed(enemy))
            self.b._move_with_terrain(enemy, position, forced_goal_cell=trace["goal_cell"],
                                     reflect_collision=True)
            enemy.remaining_path_distance = self.b._targeting_path_distance(enemy)
        return True

    def lure_duration(self, enemy, duration):
        factor = max([1.0] + [blackboard(r).get("one_minus_status_resistance", 1)
                     for r in self.runes if r["key"] == "enemy_attribute_mul"])
        resistance = enemy.statuses.resistance
        if enemy.key == TOUCH:
            if duration > 0: self.add_control_resistance(enemy, "ATTRACTED")
            resistance = max(resistance, self.control_resistance(enemy, "ATTRACTED"))
        return duration * (1 - resistance) * factor

    def control_resistance(self, enemy, key):
        entry = self.enemy_state[enemy.enemy_index]["status_stacks"].get(key)
        board = self.talent(enemy, "StatusResist")
        return min(1.0, board.get("abnormal_resistance", .3) * entry["count"]) if entry and self.b.time < entry["until"] else 0.0

    def control_immune(self, enemy, key):
        entry = self.enemy_state[enemy.enemy_index]["status_stacks"].get(key)
        return bool(entry and self.b.time < entry["until"]
                    and entry["count"] >= self.talent(enemy, "StatusResist").get("max_stack_cnt", 4))

    def add_control_resistance(self, enemy, key):
        stacks = self.enemy_state[enemy.enemy_index]["status_stacks"]
        previous = stacks.get(key, {})
        count = previous.get("count", 0) if self.b.time < previous.get("until", 0) else 0
        board = self.talent(enemy, "StatusResist")
        maximum = int(board.get("max_stack_cnt", 4))
        stacks[key] = {"count": min(maximum, count + 1),
                       "until": self.b.time + board.get("duration", 15)}
        if key == "ATTRACTED" and stacks[key]["count"] >= maximum:
            enemy.lure_target = None

    def after_enemy_move(self, enemy):
        if not self.enabled or enemy.key != "enemy_1330_cbrush_2": return
        state = self.enemy_state[enemy.enemy_index]
        position = enemy.position()
        moved = position != state["last_position"]
        state["last_position"] = position
        if not moved or enemy.unbalanced or not enemy.statuses.can_move: return
        state["charged"] = True
        speed = self.b._effective_enemy_move_speed(enemy) / self.b.move_multiplier
        board = self.talent(enemy, "Rush")
        for op in self.allies():
            touching = math.dist(op.tile, position) <= 0.5 and self.b.tile(*op.tile).get("heightType") == "LOWLAND"
            if not touching:
                state["rush_contacts"].discard(op.deployment_index)
            elif op.deployment_index not in state["rush_contacts"] and enemy.blocked_by is not op:
                state["rush_contacts"].add(op.deployment_index)
                self.damage(op, speed * board.get("atk_scale", 1000), DamageType.PHYSICAL, enemy)
                if not op.dead: self.damage(op, self.b._effective_enemy_atk(enemy) * board.get("rush_damage", 0.5), DamageType.PHYSICAL, enemy)
        self.b.mechanic_diagnostics["VEC08_CBRUSH_CONTACT_PHASE_CANDIDATE"] = 1

    def after_force_motion(self, enemy):
        if not self.enabled or enemy.key != "enemy_1328_cbjedi_2" or enemy.dead: return
        state = self.enemy_state[enemy.enemy_index]
        if enemy.unbalanced:
            state["bleed_clock"] = state.get("bleed_clock", 0) + self.b.dt
            while state["bleed_clock"] + 1e-9 >= 0.066 and not enemy.dead:
                state["bleed_clock"] -= 0.066
                self.damage(enemy, 400, DamageType.LIFE_LOSS)
        else: state["bleed_clock"] = 0

    def snapshot(self):
        return {"stage": STAGE, "selectedBattleBuffs": self.b.level.get("selectedBattleBuffs", []),
                "pollutedTiles": [list(tile) for tile in sorted(self.polluted)],
                "bosses": [{"enemy": key, "phase": value["phase"], "shieldStacks": value["stacks"],
                            "rebornUntil": round(value["reborn_until"], 3),
                            "absorbing": value["absorb"] is not None,
                            "traceMover": value["summon"].enemy_index if value["summon"] is not None else None,
                            "traceTarget": key if value["summon"] is not None else None,
                            "controlImmunities": [name for name in self.enemy_state[key]["status_stacks"]
                                                  if self.control_immune(self.enemy_state[key]["entity"], name)]}
                           for key, value in sorted(self.bosses.items())],
                "exposure": [{"deployment": key, "damage": value["damage"]} for key, value in sorted(self.exposure.items())]}

"""Scoped Mansfield mechanics from the selected stage and native prefab boards.

Native animation/FSM notification phases remain explicitly candidate. This
adapter does not change the frozen database or apply Mansfield rules elsewhere.
"""
from dataclasses import replace
import math

from . import data as D
from .mechanics import ApplyWay, DamageType, Status, enemy_target_key

STAGE = "act15d0_ex08"
BOSS = "enemy_1516_jakill"
DEVICE = "trap_025_prison"


def board(data):
    return {v["key"]: v.get("valueStr") or v.get("value", 0)
            for v in data.get("talentBlackboard") or []}


class MansfieldRuntime:
    def __init__(self, battle):
        self.b = battle
        self.enabled = battle.level.get("mansfieldStage") == STAGE
        self.states = {}
        self.events = []
        self.devices = []
        if self.enabled:
            battle.mechanic_diagnostics["MANSFIELD_FSM_NOTIFICATION_PHASE_CANDIDATE"] = 1
            battle.mechanic_diagnostics["MANSFIELD_SKILL_ANIMATION_TIMING_CANDIDATE"] = 1
            battle.mechanic_diagnostics["MANSFIELD_AURA_OVERRIDE_PHASE_CANDIDATE"] = 1
            battle.mechanic_diagnostics["MANSFIELD_BOSS_SP_TRIGGER_HANDOFF_CANDIDATE"] = 1

    def emit(self, kind, **details):
        self.events.append(dict(t=round(self.b.time, 4), kind=kind,
                                frame=self.b.frame_index, **details))

    def enemy_data(self, key):
        return D.enemy_data_for_level(self.b.enemy_index, key, self.b.level)

    def initialize_devices(self):
        if not self.enabled:
            return
        from .battle import OperatorPlan
        directions = {"RIGHT": 0, "DOWN": 1, "LEFT": 2, "UP": 3}
        for item in self.b.level.get("predefines", {}).get("tokenInsts", []):
            if item["inst"]["characterKey"] != DEVICE:
                continue
            # Level coordinates already use the engine's bottom-origin rows.
            p = OperatorPlan(char_id=DEVICE,
                tile=(item["position"]["row"], item["position"]["col"]), time=0,
                level=item["inst"]["level"], elite=0,
                facing=directions[item["direction"]], skill_id="sktok_prison", skill_level=1)
            op = self.b._make_operator(p, 0)
            op.stage_alias = item["alias"]
            op.uses_deployment_slot = False
            op.auto = False
            op.sp = 0
            op.manual_retreat_forbidden = True
            op.plan_profile = -op.deployment_index
            self.b.operators.append(op)
            self.devices.append(op)
            self.emit("mansfield_device_preset", target=op.deployment_index,
                      alias=op.stage_alias, tile=list(op.tile), facing=op.facing)

    def born(self, enemy, data):
        if not self.enabled:
            return
        bb = board(data)
        prisoner = "confinement.times" in bb
        state = self.states[enemy.enemy_index] = dict(entity=enemy, board=bb,
            prisoner=prisoner, mode=0, attacks=0, released_once=False,
            base=(enemy.atk, enemy.defense, enemy.res, enemy.attack_interval, enemy.move_speed),
            phase=1, reborn_until=0., sp=0, skill=None, targets=[], extra_hits=[])
        for field, status in (("stunImmune", Status.STUNNED), ("sleepImmune", Status.SLEEP),
                ("frozenImmune", Status.FROZEN), ("silenceImmune", Status.SILENCE),
                ("levitateImmune", Status.LEVITATE), ("fearedImmune", Status.FEAR)):
            if D.mdef(data.get("attributes", {}).get(field), False):
                enemy.statuses.immunities.add(status)
        if prisoner:
            self.set_mode(enemy, 0, cause="spawn")
        if enemy.key == BOSS:
            enemy.res += bb.get("enhance.magic_resistance", 0)
            enemy.lethal_handler = lambda packet: self.reborn(enemy, packet)

    def set_mode(self, enemy, mode, *, cause):
        s = self.states[enemy.enemy_index]
        if not s["prisoner"]:
            return
        s["mode"] = mode
        enemy.mode_index = mode
        bb = s["board"]
        atk, defense, res, interval, speed = s["base"]
        liberated = mode == 2
        enemy.atk = atk * (1 + bb.get("liberty.atk", 0) if liberated else 1)
        enemy.defense = defense + (0 if liberated else bb.get("confinement.def", 0))
        enemy.res = res + (bb.get("liberty.magic_resistance", 0) if liberated else 0)
        enemy.can_normal_attack = ("vofwiz" not in enemy.key or liberated)
        if mode == 0:
            s["attacks"] = 0
        enemy.cached_operator_target = None
        self.emit("mansfield_prisoner_mode", enemy=enemy.enemy_index, mode=mode, cause=cause)
        if liberated and "lifbos" in enemy.key and not s["released_once"]:
            s["released_once"] = True
            self.release_all(enemy, cause="recidivist_first_release")

    def release_all(self, source, *, cause):
        # Only prisoners already present receive the instantaneous global buff.
        for enemy in list(self.b.enemies):
            s = self.states.get(enemy.enemy_index)
            if s and s["prisoner"] and s["mode"] != 2 and not enemy.dead and not enemy.leaked:
                self.set_mode(enemy, 2, cause=cause)
                enemy.attack_target = None
                enemy.attack_hit_frame = None
        self.emit("mansfield_global_release", enemy=source.enemy_index, cause=cause)

    def speed_delta(self, enemy):
        if not self.enabled:
            return 0
        s = self.states.get(enemy.enemy_index)
        delta = (s["board"].get("confinement.attack_speed", 0)
                 if s and s["prisoner"] and s["mode"] != 2 else 0)
        # Native aura: one non-independent buff key, maxStackCnt=1.
        aura = max([0] + [v["board"].get("atkspeedup.attack_speed", 0)
            for v in self.states.values() if not v["entity"].dead and not v["entity"].leaked
            and not v["entity"].statuses.has(Status.SILENCE)])
        return delta + aura

    def valid_target(self, enemy, op):
        s = self.states.get(enemy.enemy_index)
        return not (s and enemy.key == BOSS and s["phase"] == 1 and op.char_id == DEVICE)

    def sort_targets(self, enemy, targets):
        s = self.states.get(enemy.enemy_index)
        priority = s and s["mode"] == 2 and any(v in enemy.key for v in ("liiprr", "vofwiz"))
        return sorted((v for v in targets if self.valid_target(enemy, v)),
            key=lambda v: ((0 if v.char_id == DEVICE else 1) if priority else 0, enemy_target_key(v)))

    def targets_in_range(self, enemy):
        return self.sort_targets(enemy, [v for v in self.b.operators if not v.dead
            and v.statuses.targetable and not v.statuses.has(Status.CAMOUFLAGE)
            and math.dist(enemy.position(), v.tile) <= enemy.range_radius + .25 + 1e-9
            and not (v.air_blocking if v.liftoff is None else v.liftoff)])

    def before_attack(self, enemy, target):
        if not self.enabled:
            return
        s = self.states.get(enemy.enemy_index)
        if not s:
            return
        if s["prisoner"] and s["mode"] != 2:
            s["attacks"] += 1
            count = int(s["board"].get("confinement.times", 4))
            if s["attacks"] >= count:
                self.set_mode(enemy, 2, cause="before_fourth_attack")
            elif s["attacks"] == count - 1:
                self.set_mode(enemy, 1, cause="release_warning")
        s["skill"] = None
        s["targets"] = [target]
        if "vofwiz" in enemy.key and s["mode"] == 2:
            s["targets"] = self.targets_in_range(enemy)[:int(s["board"].get("liberty.attack@max_target", 2))]
        if enemy.key == BOSS:
            if s["sp"] >= 3:
                s["sp"] = 0
                s["skill"] = "ironsandstorm" if s["phase"] == 1 else "armorpiercing"
                if s["phase"] == 1:
                    s["targets"] = self.targets_in_range(enemy)[:2]
                self.emit("mansfield_boss_skill", enemy=enemy.enemy_index, skill=s["skill"],
                          targets=[v.deployment_index for v in s["targets"]])
            else:
                s["sp"] = min(3, s["sp"] + 1)
            name = ("C1_Skill" if s["skill"] == "ironsandstorm" else
                    "C2_Skill" if s["skill"] == "armorpiercing" else
                    "C1_Attack" if s["phase"] == 1 else "C2_Attack")
            animation = self.b.attack_timing.get(BOSS, {}).get("mansfieldAnimations", {}).get(name)
            if animation:
                enemy.attack_duration = animation["duration"]
                enemy.attack_hit_time = next(v["time"] for v in animation["events"] if v["name"] == "OnAttack")
        # A flying projectile retains the cast's damage type and attached skill
        # even if its source changes mode or starts another attack before impact.
        projectile = None
        if "liiprr" in enemy.key:
            projectile = "projectile_enemy_liiprr_2" if s["mode"] == 2 else "projectile_enemy_liiprr"
        elif "vofwiz" in enemy.key:
            projectile = "projectile_enemy_vofwiz"
        elif enemy.key == BOSS and s["phase"] == 1:
            projectile = "projectile_enemy_jakill_s1" if s["skill"] else "projectile_enemy_jakill"
        penetration = (.6 if s["skill"] == "armorpiercing" else
                       s["board"].get("liberty.def_penetrate", 0)
                       if s["mode"] == 2 and "lidbox" in enemy.key else 0.)
        s["attack"] = dict(projectile=projectile, skill=s["skill"],
            damage_type=self.damage_type(enemy), penetration=penetration,
            apply_way=ApplyWay.RANGED if projectile else ApplyWay.MELEE)

    def damage_type(self, enemy):
        s = self.states.get(enemy.enemy_index)
        if s and ((enemy.key == BOSS and s["phase"] == 1)
                  or "vofwiz" in enemy.key or ("liiprr" in enemy.key and s["mode"] == 2)):
            return DamageType.ARTS
        return DamageType.PHYSICAL

    def modify_packet(self, packet, target, *, context=None):
        s = self.states.get(getattr(packet.source, "enemy_index", None))
        if not self.enabled or not s:
            return packet
        if context is not None:
            return replace(packet, physical_penetration_ratio=context["penetration"])
        penetration = 0.
        if s["skill"] == "armorpiercing":
            penetration = .6
        elif s["mode"] == 2 and "lidbox" in packet.source.key:
            penetration = s["board"].get("liberty.def_penetrate", 0)
        return replace(packet, physical_penetration_ratio=penetration) if penetration else packet

    def fire_attack(self, enemy, target, *, will_hit):
        if not self.enabled or enemy.enemy_index not in self.states:
            return False
        s = self.states[enemy.enemy_index]
        context = dict(s["attack"])
        for op in s["targets"]:
            if op.dead or not op.statuses.targetable:
                continue
            self.b._fire_single_enemy_attack(enemy, op, will_hit=will_hit, attack_context=context)
        return True

    def after_hit(self, enemy, target, context):
        if not self.enabled or context is None or target.dead:
            return
        if context["skill"] == "ironsandstorm":
            target.statuses.apply(Status.STUNNED, 3, resistible=True)
        elif context["skill"] == "armorpiercing" and not context.get("extra"):
            self.states[enemy.enemy_index]["extra_hits"].append(
                (self.b.time + .2, target, {**context, "extra": True}))

    def reborn(self, enemy, packet):
        s = self.states[enemy.enemy_index]
        if s["phase"] != 1:
            return False
        s["phase"] = 0
        s["reborn_until"] = self.b.time + s["board"].get("reborn.duration", 4)
        s["sp"] = 0
        s["extra_hits"].clear()
        enemy.hp = 1.
        enemy.wisdel_remnant_source = None
        enemy.palsy_stacks = 0
        enemy.attack_target = None
        enemy.attack_hit_frame = None
        for status in (Status.INVINCIBLE, Status.ROOT, Status.DISARMED, Status.UNTARGETABLE):
            enemy.statuses.grant(status, "mansfield-reborn")
        self.b._release_enemy_block(enemy)
        self.emit("mansfield_reborn_start", enemy=enemy.enemy_index)
        return True

    def before_tick(self, enemy):
        s = self.states.get(enemy.enemy_index)
        if not self.enabled or not s:
            return False
        if enemy.key == BOSS and s["phase"] == 0:
            if self.b.time + 1e-9 < s["reborn_until"]:
                return True
            s["phase"] = 2
            for status in (Status.INVINCIBLE, Status.ROOT, Status.DISARMED, Status.UNTARGETABLE):
                enemy.statuses.revoke(status, "mansfield-reborn")
            # Native Reborn retains no mode buffs; clear prior combat ailments.
            enemy.statuses.durations.clear()
            bb = s["board"]
            atk, defense, res, interval, speed = s["base"]
            enemy.hp = enemy.max_hp
            enemy.atk = atk + bb.get("enhance.atk", 0)
            enemy.defense = defense + bb.get("enhance.def", 0)
            enemy.res = res
            enemy.attack_interval = interval + bb.get("enhance.base_attack_time", 0)
            enemy.move_speed = speed + bb.get("enhance.move_speed", 0)
            enemy.apply_way = "MELEE"
            enemy.range_radius = -1
            enemy.attack_cd = 0.
            enemy.attack_rearm_frame = None
            enemy.attack_cd_frames = 0
            enemy.attack_animation_end_frame = None
            enemy.cached_operator_target = None
            self.release_all(enemy, cause="boss_phase_two")
            self.emit("mansfield_reborn_end", enemy=enemy.enemy_index, phase=2)
        if s["mode"] == 2 and "vofsd" in enemy.key:
            enemy.hp = min(enemy.max_hp, enemy.hp + s["board"].get("liberty.hp_recovery_per_sec", 0) * self.b.dt)
        for at, op, context in list(s["extra_hits"]):
            if self.b.time + 1e-9 >= at:
                s["extra_hits"].remove((at, op, context))
                if not op.dead and enemy.statuses.can_attack:
                    self.b._enemy_attack_lands(enemy, op, attack_context=context)
        return False

    def cast_device(self, op):
        if op.dead or not op.statuses.can_use_skill or op.sp < 25 or self.b.dp < 5:
            return False
        self.b.dp -= 5
        op.sp -= 25
        for enemy in self.b.enemies:
            if enemy.dead or enemy.leaked or not enemy.statuses.targetable:
                continue
            if not self.b._enemy_intersects_range(enemy, op.range_cells):
                continue
            enemy.statuses.apply(Status.STUNNED, 2, resistible=True)
            s = self.states.get(enemy.enemy_index)
            if s and s["prisoner"] and s["mode"] == 2:
                self.set_mode(enemy, 0, cause="device_reimprison")
                enemy.attack_target = None
                enemy.attack_hit_frame = None
        self.emit("mansfield_device_cast", target=op.deployment_index, alias=op.stage_alias)
        return True

    def snapshot(self):
        return {str(k): {"mode": s["mode"], "attacks": s["attacks"], "phase": s["phase"],
                         "sp": s["sp"], "skill": s["skill"]}
                for k, s in self.states.items() if not s["entity"].dead and not s["entity"].leaked}

"""Stage-local 9-12 Dublin mechanics; notification/physics phases are candidates."""
from dataclasses import replace
import math

from . import data as D
from .mechanics import ApplyWay, DamagePacket, DamageType, Status
from .runes import blackboard

STAGE = "main_09-10"
GARGOYLE = "enemy_1172_dugago"
FLIER = "enemy_1171_durokt"
PHALANX = "enemy_1169_duphlx"
PILLAR = "trap_043_dupilr"
NORMAL_DAMAGE_TYPES = {
    "enemy_1166_dusbr": DamageType.PHYSICAL,
    "enemy_1168_dumage": DamageType.ARTS,
    PHALANX: DamageType.PHYSICAL,
    "enemy_1170_dushld": DamageType.PHYSICAL,
    FLIER: DamageType.PHYSICAL,
    GARGOYLE: DamageType.PHYSICAL,
}
FALL_CONTROLS = (Status.STUNNED, Status.SLEEP, Status.FROZEN, Status.FROZEN_FRIENDLY,
                 Status.FROZEN_ENEMY, Status.LEVITATE, Status.ROOT, Status.BIND)


class DublinRuntime:
    def __init__(self, battle):
        self.b = battle
        self.enabled = battle.level.get("dublinStage") == STAGE
        self.states = {}
        self.events = []
        self.infected = {}
        self.rubble = set()
        self.initialized = False
        assumptions = battle.level.get("dublinUserAssumptions") or {}
        self.fallen_frames = assumptions.get("fallenAnimationFrames", 0) if self.enabled else 0
        self.rebirth_frames = assumptions.get("rebirthAnimationFrames", 0) if self.enabled else 0
        if any(type(value) is not int or value < 0 for value in (self.fallen_frames, self.rebirth_frames)):
            raise ValueError("Dublin animation assumptions require nonnegative integer frames")
        if self.fallen_frames + self.rebirth_frames:
            if not math.isclose(battle.dt, 1 / 30, abs_tol=1e-12):
                raise ValueError("Dublin user animation assumptions require 30 logic frames/second")
            battle.mechanic_diagnostics["DUBLIN_USER_GARGOYLE_ANIMATION_ASSUMPTION"] = 1
        self.initial_pillars = {
            (inst["position"]["row"], inst["position"]["col"])
            for inst in battle.level.get("predefines", {}).get("tokenInsts", [])
            if inst["inst"]["characterKey"] == PILLAR
        }
        if self.enabled:
            for key in ("DUBLIN_FORM_AND_CONTROL_NOTIFICATION_CANDIDATE",
                        "DUBLIN_AURA_AND_REFRACTION_PHASE_CANDIDATE",
                        "DUBLIN_PILLAR_COLLAPSE_AND_NAVIGATION_CANDIDATE",
                        "DUBLIN_INFECTION_TICK_PHASE_CANDIDATE"):
                battle.mechanic_diagnostics[key] = 1

    def emit(self, kind, **kwargs):
        self.events.append(dict(t=round(self.b.time, 4), fixedFrame=self.b.frame_index,
                                kind=kind, **kwargs))

    def initialize(self):
        if not self.enabled or self.initialized:
            return
        self.initialized = True
        attrs = self.b.characters[PILLAR]["phases"][0]["attributesKeyFrames"][0]["data"]
        data = dict(name={"m_defined": True, "m_value": "破碎支柱"},
                    attributes={k: {"m_defined": True, "m_value": v} for k, v in attrs.items()},
                    motion={"m_defined": True, "m_value": "WALK"})
        self.b.enemy_index = {**self.b.enemy_index, PILLAR: [dict(level=0, enemyData=data)]}
        for inst in self.b.level.get("predefines", {}).get("tokenInsts", []):
            if inst["inst"]["characterKey"] != PILLAR:
                continue
            enemy = self.b._make_enemy(PILLAR, 0)
            pos = inst["position"]
            enemy.position_row, enemy.position_col = float(pos["row"]), float(pos["col"])
            enemy.unblockable = True
            enemy.can_normal_attack = False
            enemy.not_count_in_total = enemy.dont_block_wave = True
            enemy.block_fragment = False
            enemy.life_reduce = 0
            enemy.taunt_level = -1
            enemy.statuses.immunities.update(FALL_CONTROLS + (Status.SILENCE, Status.COLD,))
            enemy.lethal_handler = lambda packet, unit=enemy: self.pillar_lethal(unit, packet)
            self.b.enemies.append(enemy)
            self.emit("dublin_pillar_born", enemy=enemy.enemy_index, tile=list(enemy.position()))

    def born(self, enemy, data):
        if not self.enabled:
            return
        board = blackboard({"blackboard": data.get("talentBlackboard") or []})
        state = self.states[enemy.enemy_index] = dict(entity=enemy, board=board, mode="GROUND",
                    defense=enemy.defense, res=enemy.res, atk=enemy.atk, infected_until=-math.inf,
                    infection_clock=0., collapsed_at=math.inf)
        if enemy.key == GARGOYLE:
            enemy.apply_way = ApplyWay.MELEE.value
            enemy.lethal_handler = lambda packet: self.gargoyle_lethal(enemy, packet)
        elif enemy.key == FLIER:
            state["mode"] = "HOVER"
            enemy.motion = "FLY"
            enemy.unblockable = True
            enemy.apply_way = ApplyWay.RANGED.value
            enemy.projectile_key = "projectile_enemy_durokt"
        self.attributes(enemy)

    def attributes(self, enemy):
        state = self.states[enemy.enemy_index]
        board = state["board"]
        enemy.res = state["res"] + (0 if enemy.statuses.has(Status.SILENCE)
                                   else board.get("refracting.magic_resistance", 0))
        enemy.defense = state["defense"]
        if enemy.key == GARGOYLE and state["mode"] == "STONE":
            enemy.res += board["stone.magic_resistance"]
            enemy.defense += board["stone.def"]
        if enemy.key == PHALANX:
            # Native trigger radius=1; standard target body remains the engine's 0.1 approximation.
            supporters = sum(other is not enemy and other.key == PHALANX and not other.dead
                and not other.leaked and not other.disappeared and
                math.dist(enemy.position(), other.position()) <= 1.1 + 1e-9
                for other in self.b.enemies)
            enemy.defense += supporters * board["auraDefup.def"]
        enemy.atk = state["atk"] * (1.5 if self.b.time < state["infected_until"] else 1.)

    def gargoyle_lethal(self, enemy, packet):
        state = self.states[enemy.enemy_index]
        if state["mode"] != "GROUND" or getattr(packet.source, "key", None) == PILLAR:
            return False
        enemy.hp = enemy.max_hp
        enemy.statuses.durations.clear()
        enemy.unblockable = True
        enemy.can_normal_attack = False
        enemy.statuses.immunities.add(Status.LEVITATE)
        self.b._release_enemy_block(enemy)
        self.clear_attack(enemy)
        frames = self.fallen_frames + self.rebirth_frames
        if frames:
            state.update(mode="FALLEN" if self.fallen_frames else "REBIRTH",
                         fallen_until_frame=self.b.frame_index + self.fallen_frames,
                         animation_until_frame=self.b.frame_index + frames)
            enemy.statuses.apply(Status.INVINCIBLE, frames * self.b.dt, resistible=False)
            enemy.statuses.apply(Status.UNTARGETABLE, frames * self.b.dt, resistible=False)
            self.attributes(enemy)
            self.emit("dublin_gargoyle_animation", enemy=enemy.enemy_index, mode=state["mode"],
                      untilFrame=state["animation_until_frame"])
        else:
            self.enter_stone(enemy)
        return True

    def enter_stone(self, enemy):
        state = self.states[enemy.enemy_index]
        state.update(mode="STONE", stone_until=self.b.time + state["board"]["stone.duration"])
        enemy.statuses.durations.pop(Status.INVINCIBLE, None)
        enemy.statuses.durations.pop(Status.UNTARGETABLE, None)
        self.attributes(enemy)
        self.emit("dublin_stone", enemy=enemy.enemy_index, until=state["stone_until"])

    def pre_tick(self):
        if not self.enabled or not (self.fallen_frames + self.rebirth_frames):
            return
        # User-defined frame boundaries commit before targeting or damage this frame.
        for state in self.states.values():
            enemy = state["entity"]
            if enemy.dead or state["mode"] not in {"FALLEN", "REBIRTH"}:
                continue
            if state["mode"] == "FALLEN" and self.b.frame_index >= state["fallen_until_frame"]:
                state["mode"] = "REBIRTH"
                self.emit("dublin_gargoyle_animation", enemy=enemy.enemy_index, mode="REBIRTH",
                          untilFrame=state["animation_until_frame"])
            if self.b.frame_index >= state["animation_until_frame"]:
                self.enter_stone(enemy)

    @staticmethod
    def clear_attack(enemy):
        enemy.attack_target = None
        enemy.attack_hit_at = -1.
        enemy.attack_hit_frame = None
        enemy.cached_operator_target = None

    def before_tick(self, enemy):
        if not self.enabled:
            return False
        state = self.states[enemy.enemy_index]
        if enemy.key == PILLAR:
            return True
        if enemy.key == GARGOYLE and state["mode"] in {"FALLEN", "REBIRTH"}:
            return True
        if enemy.key == FLIER and state["mode"] == "HOVER" and any(
                enemy.statuses.has(control) for control in FALL_CONTROLS):
            state["mode"] = "GROUND"
            enemy.motion = "WALK"
            enemy.unblockable = False
            enemy.apply_way = ApplyWay.MELEE.value
            enemy.projectile_key = None
            enemy.statuses.apply(Status.STUNNED, .5)
            self.clear_attack(enemy)
            self.emit("dublin_flier_landed", enemy=enemy.enemy_index)
        if enemy.key == GARGOYLE and state["mode"] == "STONE":
            if self.b.time + 1e-9 < state["stone_until"]:
                return True
            state["mode"] = "FLY"
            enemy.mode_index = 2
            enemy.motion = "FLY"
            enemy.can_normal_attack = True
            enemy.apply_way = ApplyWay.RANGED.value
            enemy.damage_type = "ARTS"
            enemy.projectile_key = "projectile_enemy_dugago"
            self.clear_attack(enemy)
            self.emit("dublin_gargoyle_fly", enemy=enemy.enemy_index)
        self.attributes(enemy)
        return False

    def valid_target(self, source, target):
        if not self.enabled:
            return True
        if self.animation_damage_immune(target):
            return False
        if target.key == PILLAR:
            cell = self.b._position_cell(*target.position())
            return source.tile in self.b.range_cells_from_id("x-5", cell, 0)
        if source.char_id == "char_4217_makoto" and self.b._is_airborne(target):
            return source.form_runtime is not None and source.form_runtime.mode == "THANATOS"
        return True

    def blocked_tile(self, row, col):
        return self.enabled and ((not self.initialized and (row, col) in self.initial_pillars)
            or (row, col) in self.rubble or any(
            state["entity"].key == PILLAR and not state["entity"].dead
            and self.b._position_cell(*state["entity"].position()) == (row, col)
            for state in self.states.values()))

    def uses_ground_navigation(self, enemy):
        # Hover targetability does not change the flier's ground MoveController.
        if self.enabled and enemy.key == FLIER:
            return True
        if self.enabled and enemy.key == GARGOYLE:
            return self.states[enemy.enemy_index]["mode"] != "FLY"
        return not self.b._is_airborne(enemy)

    def refresh_ground_routes(self):
        self.b._path_map_cache.clear()
        self.b.routes = self.b._build_routes(self.b.level.get("routes", []))
        self.b.paths = self.b._routes_to_paths(self.b.routes)
        for enemy in self.b.enemies:
            if (enemy.key == PILLAR or enemy.dead or enemy.leaked
                    or not self.uses_ground_navigation(enemy) or not enemy.route):
                continue
            guides = {point.kind for point in enemy.route if point.kind.startswith("NAVIGATE_")}
            navigation_mode = ("NODE_STABLE" if "NAVIGATE_STABLE" in guides else
                               "NODE_CENTER" if "NAVIGATE_CENTER" in guides else "NORMAL")
            pending = [point for point in enemy.route[enemy.route_idx + 1:]
                       if not point.kind.startswith("NAVIGATE_")]
            start = replace(enemy.route[0], row=enemy.position_row, col=enemy.position_col,
                            kind="START", wait_time=0.)
            enemy.route = self.b._expand_ground_route(
                [start, *pending], allow_diagonal=all(point.allow_diagonal for point in pending),
                navigation_mode=navigation_mode)
            enemy.path = self.b._routes_to_paths([enemy.route])[0]
            enemy.route_idx = enemy.tile_idx = 0
            enemy.remaining_path_distance = self.b._targeting_path_distance(enemy)

    def animation_damage_immune(self, target):
        return self.enabled and self.states.get(
            getattr(target, "enemy_index", None), {}).get("mode") in {"FALLEN", "REBIRTH"}

    def modify_packet(self, packet, target):
        if not self.enabled:
            return packet
        if getattr(target, "key", None) == PILLAR:
            source = packet.source
            position = (source.position() if hasattr(source, "enemy_index") else getattr(source, "tile", None))
            cell = self.b._position_cell(*target.position())
            if position is None or self.b._position_cell(*position) not in self.b.range_cells_from_id("x-5", cell, 0):
                return replace(packet, amount=0.)
        return packet

    def pillar_lethal(self, enemy, packet):
        state = self.states[enemy.enemy_index]
        if state["collapsed_at"] < math.inf:
            enemy.hp = 1.
            return True
        source = packet.source
        position = source.position() if hasattr(source, "enemy_index") else source.tile
        dr, dc = enemy.position_row - position[0], enemy.position_col - position[1]
        direction = ((1 if dr >= 0 else -1), 0) if abs(dr) >= abs(dc) else (0, (1 if dc >= 0 else -1))
        state.update(direction=direction, collapsed_at=self.b.time + 2.)
        enemy.hp = 1.
        enemy.statuses.apply(Status.INVINCIBLE, 2., resistible=False)
        enemy.statuses.apply(Status.UNTARGETABLE, 2., resistible=False)
        self.emit("dublin_pillar_collapse_start", enemy=enemy.enemy_index, direction=direction)
        return True

    def tick(self):
        if not self.enabled:
            return
        self.initialize()
        for enemy in list(self.b.enemies):
            if enemy.dead or enemy.leaked:
                continue
            state = self.states[enemy.enemy_index]
            if enemy.key == PILLAR and self.b.time + 1e-9 >= state["collapsed_at"]:
                state["collapsed_at"] = math.inf
                enemy.dead = True
                enemy.hp = 0.
                dr, dc = state["direction"]
                origin = self.b._position_cell(*enemy.position())
                cells = {(origin[0]+i*dr, origin[1]+i*dc) for i in (1, 2)}
                for target in list(self.b.enemies):
                    if target.dead or target.leaked or not self.b._enemy_intersects_range(target, cells):
                        continue
                    result = self.b._apply_damage(DamagePacket(enemy.atk, DamageType.TRUE, source=enemy), target)
                    if result.killed:
                        self.b.enemies_killed += int(not target.not_count_in_total)
                    else:
                        target.statuses.apply(Status.STUNNED, 10.)
                for op in self.b.operators:
                    if not op.dead and op.tile in cells:
                        op.dead = True
                        self.b._process_operator_departure(op, retreated=True, refund_withdraw_cost=False)
                        self.emit("dublin_pillar_forced_retreat", operator=op.deployment_index)
                self.rubble.update(cell for cell in cells if self.b.tile(*cell).get("buildableType") == "MELEE")
                self.refresh_ground_routes()
                self.emit("dublin_pillar_collapsed", enemy=enemy.enemy_index, cells=list(cells))
            elif enemy.key != PILLAR:
                self.infection(enemy, state)

    def infection(self, entity, state):
        if self.b.tile(*self.b._position_cell(*entity.position())).get("tileKey") == "tile_infection":
            if not state.get("touched_infection"):
                state["infected_until"] = self.b.time + 300.
                state["touched_infection"] = True
                self.emit("dublin_infection", enemy=entity.enemy_index)
        else:
            state["touched_infection"] = False
        if self.b.time < state["infected_until"]:
            state["infection_clock"] += self.b.dt
            while state["infection_clock"] + 1e-9 >= 1.:
                state["infection_clock"] -= 1.
                result = self.b._apply_damage(DamagePacket(180., DamageType.LIFE_LOSS), entity)
                if result.killed:
                    self.b.enemies_killed += 1
        self.attributes(entity)

    def attack_speed(self, enemy):
        state = self.states.get(enemy.enemy_index)
        return 50. if self.enabled and state and self.b.time < state["infected_until"] else 0.

    def damage_type(self, enemy):
        if not self.enabled:
            return self.b.vecbreak.normal_damage_type(enemy)
        if enemy.key == GARGOYLE and self.states[enemy.enemy_index]["mode"] == "FLY":
            return DamageType.ARTS
        if enemy.key in NORMAL_DAMAGE_TYPES:
            return NORMAL_DAMAGE_TYPES[enemy.key]
        self.b.mechanic_diagnostics.setdefault(
            f"DUBLIN_NORMAL_ATTACK_TYPE_UNVERIFIED:{enemy.key}", 1)
        return self.b.vecbreak.normal_damage_type(enemy)

    def snapshot(self):
        return dict(enemies={str(key): dict(mode=value["mode"], stoneUntil=value.get("stone_until"),
                            animationUntilFrame=value.get("animation_until_frame"))
                             for key, value in self.states.items() if not value["entity"].dead},
                    rubble=sorted(self.rubble))

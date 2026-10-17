"""CW-EX-3 stage-local gravity, debris and enemy adapters.

Prefab/template parameters are implemented separately from candidate native
notification, collision and navigation phases. No player operation is invented.
"""
from dataclasses import replace
import math

from . import data as D
from .mechanics import DamagePacket, DamageType, AttackType, ApplyWay, Status
from .runes import blackboard

STAGE = "act25side_ex03"
DEBRIS = "enemy_1334_ristar"
MIRROR = "enemy_1329_cbshld_2"
CAPTAIN = "enemy_1326_cbagen_2"
ARC = "enemy_1328_cbjedi_2"
ROAD = {"tile_ristar_road", "tile_ristar_road_forbidden"}
DIRECTIONS = {"UP": (1, 0), "DOWN": (-1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}


class CwRuntime:
    def __init__(self, battle):
        self.b = battle
        self.enabled = battle.level.get("cwStage") == STAGE
        self.states = {}
        self.events = []
        self.button_occupants = set()
        self.block_debuffs = {}
        self.gravity = (-1, 0)
        self.gravity_changed = -math.inf
        if not self.enabled:
            return
        for inst in battle.level.get("predefines", {}).get("tokenInsts", []):
            if inst.get("inst", {}).get("characterKey") == "trap_121_gractrl":
                self.gravity = DIRECTIONS[inst["direction"]]
        self.gravity_board = blackboard(battle.skills["sktok_gractrl"]["levels"][0])
        for key in ("CW_GRAVITY_NOTIFICATION_AND_DIRECTION_CANDIDATE",
                    "CW_RISTAR_MOVEMENT_AND_COLLISION_CANDIDATE",
                    "CW_RISTAR_INCOMING_DIRECTION_CANDIDATE",
                    "CW_ENEMY_ABILITY_NOTIFICATION_CANDIDATE"):
            battle.mechanic_diagnostics[key] = 1

    def emit(self, kind, **details):
        self.events.append(dict(t=round(self.b.time, 4), fixedFrame=self.b.frame_index,
                                kind=kind, **details))

    def born(self, enemy, data):
        if not self.enabled:
            return
        board = {item["key"]: item.get("valueStr") or item.get("value", 0)
                 for item in data.get("talentBlackboard") or []}
        state = self.states[enemy.enemy_index] = dict(entity=enemy, board=board,
            base_def=enemy.defense, base_res=enemy.res, hits=0, bleed_clock=0.)
        for field, status in (("stunImmune", Status.STUNNED), ("sleepImmune", Status.SLEEP),
                              ("frozenImmune", Status.FROZEN), ("silenceImmune", Status.SILENCE),
                              ("levitateImmune", Status.LEVITATE), ("fearedImmune", Status.FEAR)):
            if D.mdef(data.get("attributes", {}).get(field), False):
                enemy.statuses.immunities.add(status)
        if enemy.key != DEBRIS:
            return
        enemy.statuses.immunities.add(Status.SLUGGISH)
        enemy.not_count_in_total = bool(D.mdef(data.get("notCountInTotal"), False))
        enemy.life_reduce = int(D.mdef(data.get("lifePointReduce"), 0))
        enemy.unblockable = True
        enemy.can_normal_attack = False
        waits = [point.wait_time for point in enemy.route if point.kind.upper() == "WAIT_FOR_SECONDS"]
        release = self.b.time + (waits[0] if waits else 0.)
        state.update(destination=None, direction=self.gravity, incoming=self.gravity,
                     release=release, move_at=release + .4,
                     slow_at=release, slow=-.63, invincible_until=-math.inf,
                     contacts={}, last=enemy.position())
        self.emit("cw_debris_born", enemy=enemy.enemy_index, counted=not enemy.not_count_in_total,
                  lifeReduce=enemy.life_reduce, releaseAt=release)

    def counts(self, enemy):
        return not enemy.not_count_in_total

    def valid_target(self, source, enemy):
        if not self.enabled or enemy.key != DEBRIS:
            return True
        # Default's native hitRangeId=x-5 restricts the selector's owner,
        # not the damage source of a later splash or remnant explosion.
        cell = self.b._position_cell(*enemy.position())
        return source.tile in self.b.range_cells_from_id("x-5", cell, 0)

    def blocked_tile(self, row, col):
        if not self.enabled:
            return False
        return any(s["entity"].key == DEBRIS and not s["entity"].dead
                   and s.get("destination") is None
                   and self.b._position_cell(*s["entity"].position()) == (row, col)
                   for s in self.states.values())

    @staticmethod
    def direction(dr, dc):
        if abs(dr) + abs(dc) <= 1e-9:
            return None
        return ((1 if dr > 0 else -1), 0) if abs(dr) >= abs(dc) else (0, (1 if dc > 0 else -1))

    def set_gravity(self, direction, source):
        if direction == self.gravity:
            return
        self.gravity = direction
        self.gravity_changed = self.b.time
        for state in self.states.values():
            if state["entity"].key == DEBRIS:
                state.update(slow_at=self.b.time, slow=-.64)
        self.emit("cw_gravity", direction=list(direction), source=source)

    def buttons(self):
        present = set()
        actors = [(op, "operator", op.deployment_index) for op in self.b.operators if not op.dead]
        actors += [(e, "enemy", e.enemy_index) for e in self.b.enemies if not e.dead and not e.leaked]
        for actor, side, identity in actors:
            tile_pos = actor.tile if side == "operator" else self.b._position_cell(*actor.position())
            tile = self.b.tile(*tile_pos)
            if tile.get("tileKey") != "tile_grvtybtn":
                continue
            board = blackboard(tile)
            eligible = ((actor.block_count >= board.get("activate_block_cnt", 1)
                         and actor.statuses.can_block) if side == "operator" else actor.mass_level >= 3)
            if not eligible:
                continue
            key = (side, identity, tile_pos)
            present.add(key)
            if key not in self.button_occupants:
                # The native direction enum is UP=0, RIGHT=1, DOWN=2, LEFT=3.
                direction = [(1, 0), (0, 1), (-1, 0), (0, -1)][int(board["source_direction"])]
                self.set_gravity(direction, dict(side=side, id=identity, tile=list(tile_pos)))
        self.button_occupants = present

    def tick(self):
        if not self.enabled:
            return
        self.buttons()
        for identity, debuff in list(self.block_debuffs.items()):
            op = debuff["op"]
            if op.dead or self.b.time + 1e-9 >= debuff["until"]:
                op.block_count += debuff["removed"]
                del self.block_debuffs[identity]
        for state in self.states.values():
            enemy = state["entity"]
            if enemy.dead or enemy.leaked or enemy.key != ARC:
                continue
            if enemy.unbalanced:
                state["bleed_clock"] += self.b.dt
                interval = state["board"].get("unbalanced_bleed.interval", .066)
                while state["bleed_clock"] + 1e-9 >= interval and not enemy.dead:
                    state["bleed_clock"] -= interval
                    self.b._apply_damage(DamagePacket(state["board"].get("unbalanced_bleed.damage", 400),
                        DamageType.LIFE_LOSS), enemy)
                    if enemy.dead:
                        self.b.enemies_killed += 1
            else:
                state["bleed_clock"] = 0.

    def before_attack(self, enemy, target):
        if not self.enabled or enemy.key != CAPTAIN or enemy.statuses.has(Status.SILENCE):
            return
        board = self.states[enemy.enemy_index]["board"]
        old = self.block_debuffs.get(target.deployment_index)
        if old is not None:
            old["until"] = self.b.time + board.get("ReduceBlockCnt.duration", 3)
            return
        tile = self.b.tile(*target.tile)
        desired = (int(target.block_count * board.get("ReduceBlockCnt.tile_block_cnt", 0))
                   if tile.get("tileKey") == "tile_grvtybtn" else
                   max(0, target.block_count + int(board.get("ReduceBlockCnt.normal_block_cnt", -1))))
        self.block_debuffs[target.deployment_index] = dict(op=target,
            removed=target.block_count - desired, until=self.b.time + board.get("ReduceBlockCnt.duration", 3))
        target.block_count = desired
        while len(target.blocked) > target.block_count:
            self.b._release_enemy_block(target.blocked[-1])
        self.emit("cw_block_debuff", enemy=enemy.enemy_index, target=target.deployment_index,
                  blockCount=desired)

    def attack_speed(self, op):
        if not self.enabled:
            return 0.
        direction = [(0, 1), (-1, 0), (0, -1), (1, 0)][op.facing]
        if direction == self.gravity:
            return self.gravity_board["gractrl_s_ally[equal].attack_speed"]
        if direction == tuple(-x for x in self.gravity):
            return self.gravity_board["gractrl_s_ally[opposite].attack_speed"]
        return 0.

    def speed(self, enemy, speed, direction=None):
        if not self.enabled:
            return speed
        direction = direction or self.direction(enemy.navigation_velocity_row, enemy.navigation_velocity_col)
        equal = direction == self.gravity
        opposite = direction == tuple(-x for x in self.gravity)
        if not equal and not opposite:
            return speed
        group = "equal" if equal else "opposite"
        if enemy.key == DEBRIS:
            mass = self.gravity_board[f"gractrl_s_enemy[{group}].mass_ristar"]
        else:
            weights = [.9, 1., 1.1, 2., 2.2] if equal else [1.5, 1.1, 1., .6, .5]
            mass = weights[min(4, max(0, enemy.mass_level))]
        return speed * mass * self.gravity_board[f"gractrl_s_enemy[{group}].move_speed"]

    def begin_move(self, enemy, direction, *, attack=False):
        state = self.states[enemy.enemy_index]
        row, col = self.b._position_cell(*enemy.position())
        end = row, col
        while self.b.tile(end[0] + direction[0], end[1] + direction[1]).get("tileKey") in ROAD:
            end = end[0] + direction[0], end[1] + direction[1]
        if end == (row, col):
            return False
        state.update(destination=end, direction=direction, hits=0)
        if attack:
            state["invincible_until"] = self.b.time + .2
        self.emit("cw_debris_move", enemy=enemy.enemy_index, destination=list(end), attackTriggered=attack)
        return True

    def move_debris(self, enemy):
        if not self.enabled or enemy.key != DEBRIS:
            return False
        state = self.states[enemy.enemy_index]
        enemy.wait_remaining = max(0., state["release"] - self.b.time)
        enemy.dont_block_wave = True
        enemy.block_fragment = False
        if self.b.time + 1e-9 < state["move_at"]:
            return True
        if state["destination"] is None and not self.begin_move(enemy, self.gravity):
            return True
        end = state["destination"]
        remaining = max(0., 1 - (self.b.time - state["slow_at"]) / 2.2)
        scale = max(0., 1 + 2 * state["slow"] * remaining)
        if self.b.time < state["invincible_until"]:
            scale += 1.
        speed = self.speed(enemy, enemy.move_speed * self.b.move_multiplier, state["direction"]) * scale
        before = enemy.position()
        distance = math.dist(before, end)
        step = min(distance, speed * self.b.dt)
        if step > 0 and distance > 0:
            enemy.position_row += (end[0] - before[0]) / distance * step
            enemy.position_col += (end[1] - before[1]) / distance * step
            enemy.navigation_velocity_row = (enemy.position_row - before[0]) / self.b.dt
            enemy.navigation_velocity_col = (enemy.position_col - before[1]) / self.b.dt
            self.collide(enemy, state)
        if math.dist(enemy.position(), end) <= 1e-9:
            state["destination"] = None
            enemy.navigation_velocity_row = enemy.navigation_velocity_col = 0.
            self.emit("cw_debris_stop", enemy=enemy.enemy_index, position=list(end))
        return True

    def collide(self, enemy, state):
        for target in list(self.b.enemies) + list(self.b.operators):
            if target is enemy or target.dead or getattr(target, "leaked", False):
                continue
            is_enemy = hasattr(target, "enemy_index")
            if is_enemy and target.key == DEBRIS:
                continue
            if is_enemy and self.b._is_airborne(target):
                continue
            if not is_enemy and self.b.tile(*target.tile).get("heightType") != "LOWLAND":
                continue
            point = target.position() if is_enemy else target.tile
            key = (is_enemy, target.enemy_index if is_enemy else target.deployment_index)
            # Mode collision triggers have native CircleCollider2D radius .6;
            # add the engine's ordinary .1 entity-body approximation.
            if math.dist(point, enemy.position()) > .7 or self.b.time < state["contacts"].get(key, -math.inf):
                continue
            state["contacts"][key] = self.b.time + 2.
            board = state["board"]
            amount = (board["HitEnemy.enemy_damage"] * board["HitEnemy.damage_scale"] if is_enemy
                      else board["HitAlly.ally_damage"] * board["HitAlly.damage_scale"])
            self.b._apply_damage(DamagePacket(amount, DamageType.PHYSICAL, source=enemy,
                attack_type=AttackType.NORMAL, apply_way=ApplyWay.NONE), target)
            if is_enemy:
                if target.dead:
                    self.b.enemies_killed += 1
                else:
                    self.b.apply_push(target, *state["direction"], int(board["HitEnemy.force"]))
                    target.statuses.apply(Status.STUNNED, board["HitEnemy.stun"])
            self.emit("cw_debris_collision", enemy=enemy.enemy_index, target=key[1], targetEnemy=is_enemy)

    def modify_packet(self, packet, target):
        if self.enabled and getattr(target, "key", None) == DEBRIS and packet.normalized_damage_type() is not DamageType.HEAL:
            # Default-mode HpNoLessThanCertainPercentModifier has min_hp_ratio=1.
            return replace(packet, amount=0.)
        return packet

    def taking_damage(self, packet, target):
        if not self.enabled or getattr(target, "key", None) not in {MIRROR, DEBRIS}:
            return
        state = self.states[target.enemy_index]
        if target.key == MIRROR:
            maximum = int(state["board"]["def_reduce.max_stack_cnt"])
            state["hits"] = min(maximum, state["hits"] + 1)
            count = state["hits"] if state["hits"] >= 2 else 0
            target.defense = state["base_def"] + state["board"]["def_reduce.def"] * count / maximum
            target.res = state["base_res"] + state["board"]["def_reduce.magic_resistance"] * count / maximum
            return
        source = packet.source
        if source is not None:
            position = source.position() if hasattr(source, "enemy_index") else getattr(source, "tile", None)
            if position is not None:
                state["incoming"] = self.direction(target.position_row - position[0], target.position_col - position[1])
        if state["destination"] is None:
            state["hits"] += 1
            if state["hits"] >= state["board"]["HitCounter.max_hit_count"] and state["incoming"]:
                self.begin_move(target, state["incoming"], attack=True)

    def snapshot(self):
        return dict(gravity=list(self.gravity), enemies={str(key):
            dict(hits=state["hits"], destination=state.get("destination"))
            for key, state in self.states.items() if not state["entity"].dead})

"""Whitw2 S1 uses independent persistent funnels, not the S3 pursuit emitter."""
from __future__ import annotations

from types import SimpleNamespace

from .mechanics import ApplyWay, AttackType, Status
from .operator_skills import OperatorSkillRuntime


WHITW2_S1 = "skchr_whitw2_1"


class Whitw2S1Runtime(OperatorSkillRuntime):
    def __init__(self, battle, owner):
        super().__init__(battle, owner)
        self.funnels = []
        self.normal_flights = []
        self.normal_target = None
        self.normal_hits = 0
        self.funnel_data = {}
        self.mode_first_index = None
        self.main_funnel_index = None
        battle.mechanic_diagnostics["WHITW2_S1_STATE_AND_PROJECTILE_PHASE_CANDIDATE"] = 1

    def funnel_count(self):
        interval = float(self.talent(1).get("interval", float("inf")))
        extra = self.b.time - self.op.deployed_at + 1e-9 >= interval * 3
        return min(3, 2 + int(extra))  # Base one, S1 passive one, talent one.

    def stationary(self, enemy):
        """Map IsStayStill's state predicate onto the main engine's states.

        Zero speed alone is not a stationary state. Native blink/force-state
        and same-frame state handoffs remain diagnosed candidates.
        """
        b = self.b
        if enemy.disappeared or enemy.spawn_move_lock_frames > 0:
            return False
        if enemy.unbalanced or enemy.pull_remaining > 0 or any(abs(v) > 1e-12 for v in
                (enemy.force_velocity_row, enemy.force_velocity_col)):
            return False
        return (enemy.blocked_by is not None or enemy.wait_remaining > 0
                or enemy.wait_active_at_frame_start or not enemy.statuses.can_move
                or enemy.pause_until > b.time
                or (enemy.attack_move_resume_frame is not None
                    and b.frame_index < enemy.attack_move_resume_frame)
                or not enemy.route)

    def valid_funnel_target(self, enemy, *, stationary=False):
        return (enemy is not None and self.b._enemy_is_attackable(enemy)
                and self.b._enemy_targetable(enemy)
                and not enemy.statuses.has(Status.CAMOUFLAGE)
                and (not stationary or self.stationary(enemy)))

    def select_funnel_target(self, index):
        # StayStillRandom draws before testing IsStayStill, including moving
        # candidates. Highest weight is an engine ranking convention; native
        # tie ordering/FP comparison and the pre-filter order remain candidates.
        ranked = []
        for enemy in self.b.enemies:
            if not self.valid_funnel_target(enemy):
                continue
            weight = self.b.random_uniform(0, 1, source=self.op, target=enemy,
                mechanism=f"whitw2_s1_funnel_{index + 1}_weight")
            if self.stationary(enemy):
                ranked.append((weight, enemy))
        self.b.mechanic_diagnostics["WHITW2_S1_RANDOM_WEIGHT_ORDER_CANDIDATE"] = 1
        return max(ranked, key=lambda item: item[0])[1] if ranked else None

    def activate(self):
        # The shared toggle component explicitly disables SkillBegin playback.
        result = super().activate()
        self.op.skill_mode = 1
        self.op.skill_mode_target = None
        self.op.pending_attack = None
        # The parent explicitly interrupts casting child abilities on detach;
        # their waitForProjectileInvalid / OnCastEnd path clears the flights.
        self.normal_flights.clear()
        self.b._finish_operator_attack_chain(self.op)
        self.op.attack_cd = 0.0
        return result

    def finish(self, reason):
        for funnel in self.funnels:
            if funnel["target"] is not None:
                self.stop_funnel(funnel)
        super().finish(reason)
        self.funnels.clear()
        self.op.pending_attack = None
        self.b._finish_operator_attack_chain(self.op)
        self.op.attack_cd = 0.0

    def stop_funnel(self, funnel):
        # The first ability that actually deals a hit owns mode growth. Its
        # stop callback copies the completed data, not a fixed array slot.
        if funnel["index"] == self.main_funnel_index:
            data = self.funnel_data.get(funnel["index"])
            if data is not None:
                self.normal_target, self.normal_hits = data
            self.main_funnel_index = None

    def next_funnel_hits(self, funnel, target):
        index = funnel["index"]
        if self.mode_first_index is None:
            self.mode_first_index = index
        if self.main_funnel_index is None:
            self.main_funnel_index = self.mode_first_index
            if index == self.mode_first_index:
                self.funnel_data[index] = (self.normal_target, self.normal_hits)
        previous_target, hits = self.funnel_data.get(index, (None, 0))
        hits = min(self.funnel_stack_limit(), hits + 1) if target is previous_target else 0
        self.funnel_data[index] = (target, hits)
        return hits

    def attack_emitted(self, attack):
        super().attack_emitted(attack)
        if self.active or not attack.will_hit:
            return
        target = attack.primary
        if not self.valid_funnel_target(target):
            return
        if target is not self.normal_target:
            self.normal_target, self.normal_hits = target, 0
        else:
            self.normal_hits = min(self.funnel_stack_limit(), self.normal_hits + 1)
        scale = self.funnel_scale(self.normal_hits)
        timing = self.op.skill_attack_timing.get(self.sid, {})
        for index in range(self.funnel_count()):
            key = f"projectile_chr_whitw2_funnel_{index + 1}"
            origin, mount = self.b._operator_projectile_origin(self.op, key, target)
            self.b.projectile_counter += 1
            self.normal_flights.append(dict(id=self.b.projectile_counter, key=key,
                target=target, pos=target.position(), origin=origin, mount=mount,
                at=self.b.time + float(timing.get("normal_funnel_delay", .5)),
                amount=self.b.operator_attack_power(self.op, attack.source_atk_scale) * scale))

    def silence(self, target):
        t = self.talent(1)
        if self.b.time - self.op.deployed_at + 1e-9 >= float(t.get("interval", float("inf"))) * 2:
            target.statuses.apply(Status.SILENCE, float(t.get("attack@silence_duration", 0)))

    def tick(self):
        super().tick()
        if self.op.dead:
            return
        b, op = self.b, self.op
        pending = self.normal_flights
        self.normal_flights = []
        for flight in pending:
            if self.active or not self.valid_funnel_target(flight["target"]):
                continue
            if b.time + 1e-9 < flight["at"]:
                self.normal_flights.append(flight)
                continue
            self.silence(flight["target"])
            self.damage(flight["target"], flight["amount"], attack_type=AttackType.NORMAL,
                        apply_way=ApplyWay.RANGED)
        if not self.active or b._skill_starting(op):
            return
        while len(self.funnels) < self.funnel_count():
            self.funnels.append(dict(index=len(self.funnels), target=None, hits=0,
                                     next=b.time, period=None))
        for funnel in self.funnels:
            target = funnel["target"]
            if not self.valid_funnel_target(target, stationary=True):
                if target is not None:
                    # StopProjectile snapshots the unused timer minus one
                    # logic tick; the ability stays active until restoration.
                    funnel["restore_at"] = max(b.time, funnel["next"] - b.dt)
                    self.stop_funnel(funnel)
                funnel["target"] = None
                if b.time + 1e-9 < funnel.get("restore_at", b.time) or not op.statuses.can_attack:
                    continue
                target = self.select_funnel_target(funnel["index"])
                if target is None:
                    continue
                key = f"projectile_chr_whitw2_funnel_s1_{funnel['index'] + 1}"
                origin, mount = b._operator_projectile_origin(op, key, target)
                b.projectile_counter += 1
                funnel.update(target=target, hits=0, last_hits=None, next=b.time, key=key,
                    id=b.projectile_counter, origin=origin, mount=mount,
                    amount=b.operator_attack_power(op), period=None)
            # Init reads attackTime - deltaPlayTime, waitFirstPeriod=false.
            # Event quantization and OnAttackTimeChanged remain candidates.
            period = max(b.dt, b._effective_attack_interval(op) - b.dt)
            old_period = funnel["period"]
            if old_period is not None and old_period != period:
                funnel["next"] = b.time + max(0, funnel["next"] - b.time) * period / old_period
            funnel["period"] = period
            if b.time + 1e-9 >= funnel["next"]:
                funnel["hits"] = self.next_funnel_hits(funnel, target)
                self.silence(target)
                self.damage(target, funnel["amount"] * self.funnel_scale(funnel["hits"]),
                            attack_type=AttackType.NORMAL, apply_way=ApplyWay.RANGED)
                funnel["last_hits"] = funnel["hits"]
                funnel["hits"] = min(self.funnel_stack_limit(), funnel["hits"] + 1)
                funnel["next"] = b.time + period

    def depart(self):
        self.funnels.clear()
        self.normal_flights.clear()
        self.funnel_data.clear()
        self.mode_first_index = self.main_funnel_index = None
        super().depart()

    def snapshot(self):
        return {**super().snapshot(), "funnelCount": self.funnel_count(),
                "funnelTargets": [f["target"].enemy_index if f["target"] is not None
                                  else None for f in self.funnels]}

    def visuals(self):
        return [*super().visuals(), *[
            SimpleNamespace(projectile_id=f["id"], key=f["key"], side="OPERATOR",
                position_row=f["target"].position_row, position_col=f["target"].position_col,
                position_z=0, travelled=0, initial_distance=0, parabolic=False,
                origin=f["origin"], mount_point=f["mount"])
            for f in [*self.normal_flights, *self.funnels]
            if f.get("target") is not None and "id" in f]]

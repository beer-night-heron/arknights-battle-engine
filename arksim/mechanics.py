"""Generic combat mechanics independent of operators and enemy abilities.

This module models the shared rules described by PRTS: typed damage packets,
ordered damage-judgement effects, abnormal-state capability gates and the
default targeting order.  It deliberately contains no character-specific or
enemy-specific behaviour.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Iterable


class DamageType(str, Enum):
    PHYSICAL = "PHYSICAL"
    ARTS = "ARTS"
    TRUE = "TRUE"
    ELEMENTAL = "ELEMENTAL"
    HEAL = "HEAL"
    LIFE_LOSS = "LIFE_LOSS"


class AttackType(str, Enum):
    NONE = "NONE"
    NORMAL = "NORMAL"
    SPLASH = "SPLASH"
    BUFF = "BUFF"
    ADDITION = "ADDITION"


class ApplyWay(str, Enum):
    NONE = "NONE"
    MELEE = "MELEE"
    RANGED = "RANGED"


class DamageRuleKind(str, Enum):
    SCALE = "SCALE"
    FLAT_REDUCTION = "FLAT_REDUCTION"
    BARRIER = "BARRIER"
    SHIELD = "SHIELD"
    EVADE = "EVADE"
    CANCEL = "CANCEL"


class ElementType(str, Enum):
    SANITY = "SANITY"
    WATER = "WATER"
    FIRE = "FIRE"
    DARK = "DARK"
    ANGER = "ANGER"


_ELEMENT_BREAK_DURATION = {
    ElementType.SANITY: 10.0,
    ElementType.WATER: 10.0,
    ElementType.FIRE: 10.0,
    ElementType.DARK: 15.0,
    ElementType.ANGER: 15.0,
}


@dataclass
class DamagePacket:
    amount: float
    damage_type: DamageType | str
    attack_type: AttackType | str = AttackType.NORMAL
    apply_way: ApplyWay | str = ApplyWay.NONE
    physical_penetration: float = 0.0
    physical_penetration_ratio: float = 0.0
    arts_penetration: float = 0.0
    arts_penetration_ratio: float = 0.0
    output_scale: float = 1.0
    taken_scale: float = 1.0
    minimum_damage_ratio: float = 0.05
    lethal: bool = True
    ignore_invincible: bool = False
    ignore_evasion: bool = False
    bypass_rules: bool = False
    critical_chance: float = 0.0
    critical_multiplier: float = 1.0
    source: Any = None
    target_defense_override: float | None = None
    target_resistance_override: float | None = None
    target_elemental_resistance_override: float | None = None

    def normalized_damage_type(self) -> DamageType:
        value = str(getattr(self.damage_type, "value", self.damage_type)).upper()
        if value == "MAGICAL":
            value = "ARTS"
        if value == "PURE":
            value = "TRUE"
        return DamageType(value)

    def normalized_attack_type(self) -> AttackType:
        value = str(getattr(self.attack_type, "value", self.attack_type)).upper()
        return AttackType(value)


@dataclass
class DamageRule:
    """One ordered damage-judgement effect.

    Higher priority runs first; equal-priority rules retain acquisition order.
    This can represent vulnerability, reduction, numeric barriers and
    count-based shields without hard-coding one global order.
    """

    kind: DamageRuleKind | str
    value: float = 0.0
    stacks: int = 0
    priority: int = 0
    acquired_order: int = 0
    damage_types: frozenset[DamageType] | None = None
    attack_types: frozenset[AttackType] | None = None

    def matches(self, packet: DamagePacket) -> bool:
        damage_type = packet.normalized_damage_type()
        attack_type = packet.normalized_attack_type()
        return (
            (self.damage_types is None or damage_type in self.damage_types)
            and (self.attack_types is None or attack_type in self.attack_types)
        )


@dataclass(frozen=True)
class DamageResult:
    basic_damage: float
    final_amount: float
    hp_delta: float
    absorbed: float = 0.0
    blocked: bool = False
    cancelled: bool = False
    evaded: bool = False
    critical: bool = False
    killed: bool = False
    healed: bool = False


class Status(str, Enum):
    STUNNED = "STUNNED"
    DISABLED = "DISABLED"
    ROOT = "ROOT"
    BIND = "BIND"
    COLD = "COLD"
    COLD_FRIENDLY = "COLD_FRIENDLY"
    COLD_ENEMY = "COLD_ENEMY"
    FROZEN = "FROZEN"
    FROZEN_FRIENDLY = "FROZEN_FRIENDLY"
    FROZEN_ENEMY = "FROZEN_ENEMY"
    SLEEP = "SLEEP"
    LEVITATE = "LEVITATE"
    SILENCE = "SILENCE"
    SKILL_NOT_ACTIVATABLE = "SKILL_NOT_ACTIVATABLE"
    DISARMED = "DISARMED"
    FORCE_DISARMED = "FORCE_DISARMED"
    PALSY_TREMOR = "PALSY_TREMOR"
    SP_BLOCKED = "SP_BLOCKED"
    INVINCIBLE = "INVINCIBLE"
    UNTARGETABLE = "UNTARGETABLE"
    INVISIBLE = "INVISIBLE"
    CAMOUFLAGE = "CAMOUFLAGE"
    HEAL_FREE = "HEAL_FREE"
    ELEMENT_IMMUNE = "ELEMENT_IMMUNE"
    SLUGGISH = "SLUGGISH"


class StatusSource(str, Enum):
    FRIENDLY = "FRIENDLY"
    ENEMY = "ENEMY"


@dataclass
class StatusController:
    """Timed abnormal effects and their common capability checks."""

    durations: dict[Status, float | None] = field(default_factory=dict)
    immunities: set[Status] = field(default_factory=set)
    resistance: float = 0.0
    resistances: dict[Status, float] = field(default_factory=dict)

    def has(self, status: Status | str) -> bool:
        effect = _status(status)
        if effect is Status.COLD:
            return any(
                item in self.durations
                for item in (Status.COLD_FRIENDLY, Status.COLD_ENEMY)
            )
        if effect is Status.FROZEN:
            return any(
                item in self.durations
                for item in (Status.FROZEN_FRIENDLY, Status.FROZEN_ENEMY)
            )
        return effect in self.durations

    def apply(
        self,
        status: Status | str,
        duration: float | None,
        *,
        resistible: bool = True,
        source: StatusSource | str = StatusSource.FRIENDLY,
    ) -> Status | None:
        effect = _status(status)
        if effect in self.immunities:
            return None
        if duration is not None:
            duration = max(float(duration), 0.0)
            if resistible:
                resistance = self.resistances.get(effect, self.resistance)
                duration *= 1.0 - max(0.0, min(resistance, 1.0))
            if duration <= 0:
                return None
        status_source = _status_source(source)
        if effect is Status.COLD:
            cold = (
                Status.COLD_FRIENDLY
                if status_source is StatusSource.FRIENDLY
                else Status.COLD_ENEMY
            )
            frozen = (
                Status.FROZEN_FRIENDLY
                if status_source is StatusSource.FRIENDLY
                else Status.FROZEN_ENEMY
            )
            previous_cold = self.durations.get(cold)
            if cold in self.durations and Status.FROZEN not in self.immunities:
                self.durations.pop(cold, None)
                if previous_cold is None:
                    duration = None
                elif duration is not None:
                    duration = max(previous_cold, duration)
                effect = frozen
            else:
                effect = cold
        elif effect is Status.FROZEN:
            effect = (
                Status.FROZEN_FRIENDLY
                if status_source is StatusSource.FRIENDLY
                else Status.FROZEN_ENEMY
            )
        previous = self.durations.get(effect)
        if previous is None and effect in self.durations:
            return effect
        if duration is None or previous is None:
            self.durations[effect] = duration
        else:
            self.durations[effect] = max(previous, duration)
        if effect in (Status.FROZEN_FRIENDLY, Status.FROZEN_ENEMY):
            return Status.FROZEN
        if effect in (Status.COLD_FRIENDLY, Status.COLD_ENEMY):
            return Status.COLD
        return effect

    def remove(self, status: Status | str) -> None:
        effect = _status(status)
        if effect is Status.COLD:
            self.durations.pop(Status.COLD_FRIENDLY, None)
            self.durations.pop(Status.COLD_ENEMY, None)
            return
        if effect is Status.FROZEN:
            self.durations.pop(Status.FROZEN_FRIENDLY, None)
            self.durations.pop(Status.FROZEN_ENEMY, None)
            return
        self.durations.pop(effect, None)

    def tick(self, dt: float) -> None:
        expired: list[Status] = []
        for status, duration in list(self.durations.items()):
            if duration is None:
                continue
            remaining = duration - dt
            if remaining <= 1e-9:
                expired.append(status)
            else:
                self.durations[status] = remaining
        for status in expired:
            self.durations.pop(status, None)

    @property
    def can_attack(self) -> bool:
        return not self._has_any(
            Status.STUNNED,
            Status.DISABLED,
            Status.FROZEN,
            Status.SLEEP,
            Status.LEVITATE,
            Status.DISARMED,
            Status.FORCE_DISARMED,
            Status.PALSY_TREMOR,
        )

    @property
    def can_move(self) -> bool:
        return not self._has_any(
            Status.STUNNED,
            Status.DISABLED,
            Status.ROOT,
            Status.BIND,
            Status.FROZEN,
            Status.SLEEP,
            Status.LEVITATE,
        )

    @property
    def can_use_skill(self) -> bool:
        return not self._has_any(
            Status.STUNNED,
            Status.DISABLED,
            Status.FROZEN,
            Status.SLEEP,
            Status.LEVITATE,
            Status.PALSY_TREMOR,
            Status.SKILL_NOT_ACTIVATABLE,
        )

    @property
    def can_block(self) -> bool:
        return not self._has_any(
            Status.STUNNED, Status.DISABLED, Status.SLEEP, Status.LEVITATE
        )

    @property
    def targetable(self) -> bool:
        return not self._has_any(
            Status.INVINCIBLE, Status.UNTARGETABLE, Status.INVISIBLE, Status.SLEEP
        )

    @property
    def attack_speed_delta(self) -> float:
        cold_count = sum(
            status in self.durations
            for status in (Status.COLD_FRIENDLY, Status.COLD_ENEMY)
        )
        return -30.0 * cold_count

    def _has_any(self, *statuses: Status) -> bool:
        return any(self.has(status) for status in statuses)


@dataclass(frozen=True)
class ElementResult:
    element: ElementType
    requested: float
    applied: float
    remaining: float
    burst: bool = False
    burst_duration: float = 0.0
    ignored: bool = False


@dataclass
class ElementBurstRuntime:
    """Mutable per-break runtime used by the generic battle loop."""

    element: ElementType
    duration: float
    source: Any = None
    tick_elapsed: float = 0.0
    anger_damage: float = 100.0


@dataclass
class ElementController:
    """Independent EP gauges plus their shared burst cooldown."""

    max_value: float = 1000.0
    recovery_per_sec: float = 0.0
    resistance: float = 0.0
    enemy_unit: bool = False
    values: dict[ElementType, float] = field(default_factory=dict)
    active_break: ElementType | None = None
    cooldown_remaining: float = 0.0

    def value(self, element: ElementType | str) -> float:
        return self.values.get(_element(element), self.max_value)

    def apply(
        self,
        element: ElementType | str,
        amount: float,
        *,
        neutral: bool = False,
        immune: bool = False,
    ) -> ElementResult:
        kind = _element(element)
        requested = max(float(amount), 0.0)
        current = self.value(kind)
        if neutral or immune or self.cooldown_remaining > 0 or requested <= 0:
            return ElementResult(kind, requested, 0.0, current, ignored=True)

        multiplier = 1.0 - max(0.0, min(self.resistance, 100.0)) / 100.0
        applied = requested * multiplier
        remaining = max(current - applied, 0.0)
        self.values[kind] = remaining
        burst = remaining <= 1e-9
        duration = 0.0
        if burst:
            duration = _ELEMENT_BREAK_DURATION[kind]
            if kind is ElementType.WATER and self.enemy_unit:
                duration = 8.0
            self.active_break = kind
            self.cooldown_remaining = duration
        return ElementResult(kind, requested, applied, remaining, burst, duration)

    def recover(self, element: ElementType | str, amount: float) -> float:
        if self.cooldown_remaining > 0:
            return 0.0
        kind = _element(element)
        before = self.value(kind)
        after = min(self.max_value, before + max(float(amount), 0.0))
        if after >= self.max_value - 1e-9:
            self.values.pop(kind, None)
        else:
            self.values[kind] = after
        return after - before

    def tick(self, dt: float) -> ElementType | None:
        if self.cooldown_remaining > 0:
            ended = self.active_break
            self.cooldown_remaining = max(0.0, self.cooldown_remaining - dt)
            if self.cooldown_remaining <= 1e-9:
                self.values.clear()
                self.active_break = None
                return ended
            return None
        if self.recovery_per_sec > 0:
            for element in list(self.values):
                self.recover(element, self.recovery_per_sec * dt)
        return None

    @property
    def current_damaged_element(self) -> ElementType | None:
        damaged = [
            element
            for element in ElementType
            if self.value(element) < self.max_value - 1e-9
        ]
        if not damaged:
            return None
        return min(damaged, key=lambda element: (self.value(element), list(ElementType).index(element)))


def calculate_basic_damage(
    packet: DamagePacket,
    *,
    defense: float = 0.0,
    resistance: float = 0.0,
    elemental_resistance: float = 0.0,
) -> float:
    amount = max(float(packet.amount), 0.0)
    damage_type = packet.normalized_damage_type()
    if damage_type is DamageType.PHYSICAL:
        effective_defense = (
            1.0 - max(0.0, min(packet.physical_penetration_ratio, 1.0))
        ) * max(0.0, defense - packet.physical_penetration)
        return max(amount * packet.minimum_damage_ratio, amount - effective_defense)
    if damage_type is DamageType.ARTS:
        effective_resistance = (
            1.0 - max(0.0, min(packet.arts_penetration_ratio, 1.0))
        ) * max(0.0, resistance - packet.arts_penetration)
        multiplier = max(0.0, 100.0 - effective_resistance) / 100.0
        return max(amount * packet.minimum_damage_ratio, amount * multiplier)
    if damage_type is DamageType.ELEMENTAL:
        multiplier = max(0.0, 100.0 - elemental_resistance) / 100.0
        return amount * multiplier
    return amount


def apply_damage(
    packet: DamagePacket,
    target: Any,
    *,
    roll: Callable[[float], bool] | None = None,
) -> DamageResult:
    """Resolve and apply one packet to an entity with common combat fields."""
    damage_type = packet.normalized_damage_type()
    statuses: StatusController = getattr(target, "statuses", StatusController())

    if damage_type is DamageType.HEAL:
        if statuses.has(Status.HEAL_FREE) or getattr(target, "dead", False):
            return DamageResult(0.0, 0.0, 0.0, cancelled=True, healed=True)
        basic = max(float(packet.amount), 0.0)
        amount = basic * packet.output_scale * packet.taken_scale
        missing = max(float(target.max_hp) - float(target.hp), 0.0)
        healed = min(amount, missing)
        target.hp += healed
        return DamageResult(basic, healed, healed, healed=True)

    if (
        (statuses.has(Status.INVINCIBLE) or statuses.has(Status.SLEEP))
        and not packet.ignore_invincible
        and damage_type is not DamageType.LIFE_LOSS
    ):
        return DamageResult(0.0, 0.0, 0.0, cancelled=True)

    critical = False
    critical_chance = float(packet.critical_chance)
    if critical_chance != critical_chance:
        raise ValueError("critical_chance must not be NaN")
    if critical_chance > 0.0:
        if critical_chance >= 1.0:
            critical = True
        elif roll is None:
            raise ValueError("probabilistic damage requires a roll callback")
        else:
            critical = roll(critical_chance)
    resolved_packet = (
        replace(
            packet,
            amount=float(packet.amount) * max(float(packet.critical_multiplier), 0.0),
            critical_chance=0.0,
        )
        if critical
        else packet
    )

    basic = calculate_basic_damage(
        resolved_packet,
        defense=(
            float(packet.target_defense_override)
            if packet.target_defense_override is not None
            else float(getattr(target, "defense", 0.0))
        ),
        resistance=(
            float(packet.target_resistance_override)
            if packet.target_resistance_override is not None
            else float(getattr(target, "res", 0.0))
        ),
        elemental_resistance=(
            float(packet.target_elemental_resistance_override)
            if packet.target_elemental_resistance_override is not None
            else float(getattr(target, "elemental_resistance", 0.0))
        ),
    )
    amount = basic * packet.output_scale
    absorbed = 0.0
    blocked = False
    cancelled = False
    evaded = False

    if not packet.bypass_rules and damage_type is not DamageType.LIFE_LOSS:
        rules: Iterable[DamageRule] = getattr(target, "damage_rules", [])
        ordered = sorted(rules, key=lambda rule: (-rule.priority, rule.acquired_order))
        for rule in ordered:
            if amount <= 0 or not rule.matches(packet):
                continue
            kind = DamageRuleKind(str(getattr(rule.kind, "value", rule.kind)).upper())
            if kind is DamageRuleKind.SCALE:
                amount *= max(rule.value, 0.0)
            elif kind is DamageRuleKind.FLAT_REDUCTION:
                amount = max(0.0, amount - max(rule.value, 0.0))
            elif kind is DamageRuleKind.BARRIER and rule.value > 0:
                used = min(amount, rule.value)
                rule.value -= used
                amount -= used
                absorbed += used
                blocked = amount <= 1e-9
            elif kind is DamageRuleKind.SHIELD and rule.stacks > 0:
                rule.stacks -= 1
                absorbed += amount
                amount = 0.0
                blocked = True
            elif kind is DamageRuleKind.EVADE:
                if rule.stacks > 0:
                    rule.stacks -= 1
                    success = True
                else:
                    probability = float(rule.value)
                    if probability <= 0.0:
                        success = False
                    elif probability >= 1.0:
                        success = True
                    elif roll is None:
                        raise ValueError(
                            "probabilistic damage requires a roll callback"
                        )
                    else:
                        success = roll(probability)
                if success and not packet.ignore_evasion:
                    amount = 0.0
                    cancelled = True
                    evaded = True
            elif kind is DamageRuleKind.CANCEL:
                amount = 0.0
                cancelled = True

    amount = max(amount * packet.taken_scale, 0.0)
    if not packet.lethal:
        amount = min(amount, max(float(target.hp) - 1.0, 0.0))
    target.hp -= amount
    killed = target.hp <= 0
    if killed:
        target.hp = 0.0
        target.dead = True
    return DamageResult(
        basic_damage=basic,
        final_amount=amount,
        hp_delta=-amount,
        absorbed=absorbed,
        blocked=blocked,
        cancelled=cancelled,
        evaded=evaded,
        critical=critical,
        killed=killed,
    )


_TARGET_FP_SCALE = 1 << 32
# Live 2.7.71 FilterUtil / WeightedTarget / MathUtil values. The non-legacy
# comparator splits large weights before quantizing their fractional parts.
_TARGET_HATRED_GAP_RAW = 429496736
_TARGET_HATRED_BIG_GAP_RAW = 1000000 * _TARGET_FP_SCALE
_TARGET_FP_EQUAL_EPSILON_RAW = 42950


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def operator_target_key(enemy: Any, *, blocked: bool = False) -> tuple[int, int, int]:
    """Build non-legacy client weights; order with compare_operator_target_keys."""
    remaining = float(
        getattr(
            enemy,
            "remaining_path_distance",
            len(enemy.path) - 1 - enemy.tile_idx,
        )
    )
    if math.isnan(remaining):
        remaining = 1e8
    remaining = min(max(remaining, -1e8), 1e8)
    # Enemy.get_hatred converts distance and taunt independently to Q32.32,
    # then adds them as integers. Combining them in float32 loses distance.
    distance_raw = int(_float32(-remaining) * _TARGET_FP_SCALE)
    taunt = _float32(float(int(getattr(enemy, "taunt_level", 0))))
    taunt_raw = int(_float32(taunt * _float32(1e8)) * _TARGET_FP_SCALE)
    hatred_reference = -(distance_raw + taunt_raw)
    return (
        0 if blocked else 1,
        hatred_reference,
        int(getattr(enemy, "creation_index", 0)),
    )


def compare_operator_target_keys(
    left: tuple[int, int, int], right: tuple[int, int, int]
) -> int:
    """Compare default hatred weights through the observed non-legacy branches."""
    if left[0] != right[0]:
        return (left[0] > right[0]) - (left[0] < right[0])
    a, b = left[1], right[1]
    if abs(a) > _TARGET_HATRED_BIG_GAP_RAW and abs(b) > _TARGET_HATRED_BIG_GAP_RAW:
        whole_a = a // _TARGET_FP_SCALE * _TARGET_FP_SCALE
        whole_b = b // _TARGET_FP_SCALE * _TARGET_FP_SCALE
        if whole_a == whole_b:
            a = (a - whole_a) // _TARGET_HATRED_GAP_RAW * _TARGET_FP_SCALE
            b = (b - whole_b) // _TARGET_HATRED_GAP_RAW * _TARGET_FP_SCALE
        else:
            a, b = whole_a, whole_b
    elif abs(a) < _TARGET_HATRED_BIG_GAP_RAW and abs(b) < _TARGET_HATRED_BIG_GAP_RAW:
        # FP division followed by TSMath.Floor, including negative weights.
        a = a // _TARGET_HATRED_GAP_RAW * _TARGET_FP_SCALE
        b = b // _TARGET_HATRED_GAP_RAW * _TARGET_FP_SCALE
    if abs(a - b) >= _TARGET_FP_EQUAL_EPSILON_RAW:
        return (a > b) - (a < b)
    # Preserve the existing creation-order fallback for equal weights.
    return (left[2] > right[2]) - (left[2] < right[2])


def enemy_target_key(operator: Any, *, blocked: bool = False) -> tuple[Any, ...]:
    """Default enemy-side targeting order after candidate filtering."""
    return (
        0 if blocked else 1,
        -int(getattr(operator, "taunt_level", 0)),
        -int(getattr(operator, "deployment_index", 0)),
        int(getattr(operator, "creation_index", 0)),
    )


def _status(value: Status | str) -> Status:
    if isinstance(value, Status):
        return value
    return Status(str(value).upper())


def _element(value: ElementType | str) -> ElementType:
    if isinstance(value, ElementType):
        return value
    return ElementType(str(value).upper())


def _status_source(value: StatusSource | str) -> StatusSource:
    if isinstance(value, StatusSource):
        return value
    return StatusSource(str(value).upper())

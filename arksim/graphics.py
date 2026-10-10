"""Facing and map-space projectile attachment profiles.

Profiles contain converted simulation coordinates, not raw model pixels.
The linear facing transition is a replaceable approximation, not a claim
about a particular renderer's tween or animation update phase.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Any


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("graphic values must be finite")
    return number


def map_offset(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) not in (2, 3):
        raise ValueError("map offsets must contain [row, col] or [row, col, z]")
    return (_finite(value[0]), _finite(value[1]),
            _finite(value[2]) if len(value) == 3 else 0.0)


@dataclass(frozen=True)
class ProjectileMotionProfile:
    distance_dimensions: int = 2
    reach_radius: float = 0.0
    target_mount: str = "center"
    quantized_step: bool = False

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> "ProjectileMotionProfile":
        if not isinstance(spec, dict):
            raise ValueError("projectile motion profiles must be objects")
        dimensions = spec.get("distanceDimensions", 2)
        if isinstance(dimensions, bool) or dimensions not in (2, 3):
            raise ValueError("distanceDimensions must be 2 or 3")
        radius = _finite(spec.get("reachRadius", 0.0))
        if radius < 0.0:
            raise ValueError("reachRadius must be nonnegative")
        mount = spec.get("targetMount", "center")
        if mount not in ("center", "hit"):
            raise ValueError("targetMount must be 'center' or 'hit'")
        quantized = spec.get("quantizedStep", False)
        if not isinstance(quantized, bool):
            raise ValueError("quantizedStep must be boolean")
        return cls(int(dimensions), radius, mount, quantized)


@dataclass(frozen=True)
class EnemyGraphicProfile:
    muzzle_offset: tuple[float, float, float] | None = None
    attack_times: tuple[float, ...] = ()
    attack_offsets: tuple[tuple[float, float, float], ...] = ()
    turn_seconds: float = 0.0
    initial_facing: int = 1
    face_route: bool = True
    hit_offset: tuple[float, float, float] | None = None

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> "EnemyGraphicProfile":
        if not isinstance(spec, dict):
            raise ValueError("enemy graphic profiles must be objects")
        if spec.get("coordinateSpace") != "map":
            raise ValueError("enemy graphics require coordinateSpace='map'")
        offset = (
            map_offset(spec["muzzleOffset"])
            if "muzzleOffset" in spec else None
        )
        times: list[float] = []
        offsets: list[tuple[float, float, float]] = []
        for sample in spec.get("muzzleAttackSamples", []):
            time = _finite(sample["time"])
            if time < 0 or (times and time <= times[-1]):
                raise ValueError("muzzle sample times must increase from zero")
            times.append(time)
            offsets.append(map_offset(sample["offset"]))
        duration = _finite(spec.get("turnSeconds", 0.0))
        if duration < 0:
            raise ValueError("turnSeconds must be nonnegative")
        facing = spec.get("initialFacing", 1)
        if facing not in (-1, 1):
            raise ValueError("initialFacing must be -1 or 1")
        return cls(offset, tuple(times), tuple(offsets), duration,
                   int(facing), bool(spec.get("faceRoute", True)),
                   map_offset(spec["hitOffset"]) if "hitOffset" in spec else None)

    def muzzle_at(self, attack_time: float | None) -> tuple[float, float, float] | None:
        if attack_time is None or not self.attack_times:
            return self.muzzle_offset
        index = bisect_right(self.attack_times, attack_time)
        if index == 0:
            return self.attack_offsets[0]
        if index == len(self.attack_times):
            return self.attack_offsets[-1]
        start, end = self.attack_times[index - 1:index + 1]
        weight = (attack_time - start) / (end - start)
        before, after = self.attack_offsets[index - 1:index + 1]
        return tuple(a + (b - a) * weight for a, b in zip(before, after))


@dataclass
class FacingRuntime:
    target: int = 1
    origin: float = 1.0
    started_at: float = 0.0
    duration: float = 0.0

    def value_at(self, time: float) -> float:
        if self.duration <= 0.0:
            return float(self.target)
        weight = max(0.0, min(1.0, (time - self.started_at) / self.duration))
        return self.origin + (self.target - self.origin) * weight

    def request(self, delta_col: float, time: float, duration: float) -> None:
        # Vertical facing changes preserve the current left/right orientation.
        if abs(delta_col) <= 1e-9:
            return
        target = 1 if delta_col > 0 else -1
        if target == self.target:
            return
        self.origin = self.value_at(time)
        self.target = target
        self.started_at = time
        self.duration = duration


@dataclass(frozen=True)
class OperatorGraphicProfile:
    faces: dict[str, dict[str, Any]]
    projectiles: dict[str, dict[str, Any]]

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> "OperatorGraphicProfile":
        if not isinstance(spec, dict) or spec.get("coordinateSpace") != "map":
            raise ValueError("operator graphics require coordinateSpace='map'")
        faces = {}
        for face, value in spec.get("faces", {}).items():
            if face not in ("front", "back"):
                raise ValueError("operator graphic faces must be front/back")
            animations = {}
            for name, animation in value.get("animations", {}).items():
                times = tuple(_finite(t) for t in animation["times"])
                duration = _finite(animation["duration"])
                if (duration < 0 or not times or times[0] != 0 or times[-1] > duration + 1e-7
                        or any(a >= b for a, b in zip(times, times[1:]))):
                    raise ValueError("operator animation samples must increase from zero within duration")
                offsets = {mount: tuple(map_offset(v) for v in samples)
                           for mount, samples in animation["offsets"].items()}
                if any(len(samples) != len(times) for samples in offsets.values()):
                    raise ValueError("operator mount samples must match animation times")
                animations[name] = (duration, bool(animation.get("loop", False)), times, offsets)
            faces[face] = animations
        projectiles = spec.get("projectiles", {})
        if not isinstance(projectiles, dict):
            raise ValueError("operator graphic projectile rules must be objects")
        for rule in projectiles.values():
            if not isinstance(rule, dict) or rule.get("loopType", "LOOP") not in ("LOOP", "RANDOM"):
                raise ValueError("operator mount group must use LOOP/RANDOM")
            if "group" in rule and (not isinstance(rule["group"], list) or not rule["group"]
                                    or any(not isinstance(v, str) for v in rule["group"])):
                raise ValueError("operator mount group must contain named points")
        return cls(faces, projectiles)

    def offset_at(self, face: str, mount: str, animation: str, time: float):
        if mount == "FOOT":
            return (0.0, 0.0, 0.0)
        animations = self.faces.get(face, {})
        track = animations.get(animation)
        if track is None or (not track[1] and time > track[0] + 1e-7):
            track = animations.get("Idle")
        if track is None:
            return None
        duration, loop, times, offsets = track
        values = offsets.get(mount)
        if not values:
            return None
        time = max(0.0, time)
        if loop and duration > 0:
            time %= duration
        index = bisect_right(times, time)
        if index == 0:
            return values[0]
        if index == len(times):
            return values[-1]
        weight = (time - times[index - 1]) / (times[index] - times[index - 1])
        return tuple(a + (b - a) * weight for a, b in zip(values[index - 1], values[index]))

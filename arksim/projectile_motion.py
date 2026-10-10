"""Small movement primitives shared by the selected composite projectiles.

Profiles come from local prefab components. Their native update interleaving
and world-to-map projection still require separate validation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def movement_speed(speed: float, profile: dict, dt: float) -> float:
    if not profile.get("dynamic", False):
        return speed
    delta = float(profile.get("acceleration", 0)) * dt
    limit = float(profile.get("final_speed", speed))
    return min(limit, speed + delta) if delta >= 0 else max(limit, speed + delta)


def steer_direction(direction: tuple, delta: tuple, turn_speed: float, dt: float,
                    *, normalize: bool = True) -> tuple:
    distance = math.hypot(*delta)
    if distance <= 1e-12:
        return direction
    weight = min(1.0, max(0.0, turn_speed * dt))
    mixed = tuple(direction[i] * (1-weight) + delta[i]/distance * weight for i in (0, 1))
    if not normalize:
        return mixed
    length = math.hypot(*mixed)
    return tuple(value / length for value in mixed) if length > 1e-12 else direction


def stepped_sine_position(elapsed: float, parameters: dict, sign: float) -> tuple:
    """Saved time-based calculator's local curve; FP sine remains approximated.

    Serialized movement speed is not an input to this calculator. The literal
    keys are move_speed / interval / cnt / speed_scale, not count.
    """
    x = elapsed * float(parameters.get("move_speed", 1))
    interval = float(parameters.get("interval", 1))
    step = min(int(parameters.get("cnt", 3)), max(1, math.ceil(x / interval / (2*math.pi))))
    return x, sign * float(parameters.get("speed_scale", 1)) * step * math.sin(x)


@dataclass
class HoverMotion:
    """Half-radius semicircle followed by a circle about the entry position.

    The stage structure and whole-tick angular increment follow the saved
    implementation. The entry tangent's map projection remains a candidate.
    """
    origin: tuple
    direction: tuple
    radius: float
    speed: float
    elapsed: float = 0.0
    stage: str = "SEMICIRCLE"

    def __post_init__(self):
        # The half-circle starts tangent to the incoming direction.
        angle = math.atan2(self.direction[0], self.direction[1]) + math.pi * .5
        self.circle_angle = angle
        self.semicircle_angle = angle + math.pi
        self.semicircle_center = (
            self.origin[0] + self.radius * .5 * math.sin(angle),
            self.origin[1] + self.radius * .5 * math.cos(angle))
        self.position = self.origin

    def advance(self, dt: float) -> tuple:
        if dt <= 0 or self.radius <= 0 or self.speed <= 0:
            return self.position
        self.elapsed += dt
        if self.stage == "SEMICIRCLE" and self.elapsed > math.pi * self.radius / (2*self.speed):
            self.stage = "CIRCLE"
        old = self.position
        if self.stage == "SEMICIRCLE":
            self.semicircle_angle += 2*self.speed/self.radius * dt
            angle, radius, center = self.semicircle_angle, self.radius * .5, self.semicircle_center
        else:
            self.circle_angle += self.speed/self.radius * dt
            angle, radius, center = self.circle_angle, self.radius, self.origin
        self.position = center[0] + radius * math.sin(angle), center[1] + radius * math.cos(angle)
        delta = self.position[0] - old[0], self.position[1] - old[1]
        length = math.hypot(*delta)
        if length > 1e-12:
            self.direction = tuple(value/length for value in delta)
        return self.position

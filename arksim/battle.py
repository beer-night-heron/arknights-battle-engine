"""Tick-based Arknights battle engine (PRTS-flavoured).

Core rules follow PRTS 游戏数据基础 / 战斗机制:
  - physical:  max(ATK*scale*0.05, ATK*scale - DEF)
  - arts:      max(ATK*scale*0.05, ATK*scale*(1 - RES/100))
  - attack interval = baseAttackTime * 100 / attackSpeed
  - enemies move on continuous route segments and use a 0.7071 ground block
    radius; flying motion bypasses ordinary ground blockers
  - melee attack range is the tile(s) in front of the operator's facing
  - SP types, charge capacity, ammo duration and explicit interruption hooks;
    MANUAL skills still auto-cast as a v0 plan compatibility approximation

Skill behaviour trees (`battle/buff_template_data.json`) are dispatched through
an incremental interpreter. Unsupported events/nodes remain observable in the
result instead of being silently treated as exact simulation.
"""

from __future__ import annotations

import math
import struct
from collections import Counter, deque
from dataclasses import dataclass, field
from functools import cmp_to_key, lru_cache
from typing import Any

from . import data as D
from .assumptions import (
    ASSUMPTION_NOTES,
    DEFAULT_ASSUMPTIONS,
    SimulationAssumptions,
)
from .behavior import (
    BehaviorContext,
    BehaviorDiagnostics,
    BehaviorError,
    BehaviorInterpreter,
    resolve_behavior_template,
)
from .buffs import BuffCatalog, BuffContainer, BuffDefinition, BuffInstance
from .mechanics import (
    ApplyWay,
    AttackType,
    DamagePacket,
    DamageRule,
    DamageType,
    ElementController,
    ElementBurstRuntime,
    ElementResult,
    ElementType,
    Status,
    StatusController,
    StatusSource,
    apply_damage,
    compare_operator_target_keys,
    enemy_target_key,
    operator_target_key,
)
from .randomness import DeterministicRng
from .graphics import (
    EnemyGraphicProfile, FacingRuntime, ProjectileMotionProfile, map_offset,
)
from .targeting import FramePeriodicTicker, TargetSelectorRuntime


UNBALANCE_MIN_DURATION = 0.1
UNBALANCE_STOP_SPEED = 0.1
DEFAULT_FRICTION_DECELERATION = 9.81 * 0.5
DEFAULT_BLOCK_RADIUS = math.sqrt(0.5)
OPERATOR_HITBOX_RADIUS = 0.25
BLOCK_SCAN_INTERVAL_FRAMES = 3
BLOCK_OFFSET_DISTANCE = 0.5
BLOCK_OFFSET_DURATION = 0.2
GROUND_STEERING_FACTOR = 8.0
PUSH_SPEED_BY_FORCE_LEVEL = {
    -2: 1.0,
    -1: 2.0,
    0: 4.0,
    1: 4.5,
    2: 5.3,
    3: 5.8,
}
PULL_FORCE_BY_FORCE_LEVEL = {
    -2: 2.0,
    -1: 10.0,
    0: 40.0,
    1: 42.0,
    2: 44.0,
    3: 46.0,
}

CLIENT_LOGIC_RATE = 30
# The H7-2 client capture reports this fixed-frame duration on every sample.
CLIENT_FIXED_TIMER_STEP = 0.03333330154418945
CLIENT_FP_SCALE = 1 << 32
# MathUtil.LessEqual uses the resident Q32.32 equality tolerance.
CLIENT_FP_EPSILON_RAW = 42950
SPAWN_TIMING_FAST = "fast"
SPAWN_TIMING_CLIENT = "client"
SPAWN_TIMING_FRAME_CORE = "frame_core"
ENEMY_ATTACK_TIMING_FLOAT = "float"
ENEMY_ATTACK_TIMING_MORTAR_FRAMES = "mortar_frames"
ENEMY_TARGET_SEARCH_PERIOD_FRAMES = 3
_SPAWN_TIMING_ALIASES = {
    "fast": SPAWN_TIMING_FAST,
    "nominal": SPAWN_TIMING_FAST,
    "client": SPAWN_TIMING_CLIENT,
    "client_coroutine_compat": SPAWN_TIMING_CLIENT,
    "frame_core": SPAWN_TIMING_FRAME_CORE,
    "spawn_core": SPAWN_TIMING_FRAME_CORE,
}
_ENEMY_ATTACK_TIMING_ALIASES = {
    "float": ENEMY_ATTACK_TIMING_FLOAT,
    "legacy": ENEMY_ATTACK_TIMING_FLOAT,
    "mortar_frames": ENEMY_ATTACK_TIMING_MORTAR_FRAMES,
    "logic_frame": ENEMY_ATTACK_TIMING_MORTAR_FRAMES,
}


def _fixed_delay_frames(delay_seconds: float, frame_duration: float) -> int:
    """Convert a delay to the first fixed frame at or after its deadline."""
    frame_count = max(0.0, float(delay_seconds)) / float(frame_duration)
    nearest_frame = round(frame_count)
    # Spine timing data can carry float32 error around whole-frame deadlines.
    if math.isclose(frame_count, nearest_frame, rel_tol=0.0, abs_tol=1e-5):
        return int(nearest_frame)
    return math.ceil(frame_count)


@lru_cache(maxsize=256)
def _animation_event_delay_frames(delay_seconds: float, frame_duration: float) -> int:
    """Candidate zero-start float32 track clock, without integer snapping."""
    step = struct.unpack("<f", struct.pack("<f", frame_duration))[0]
    if not math.isfinite(delay_seconds) or not math.isfinite(step) or step <= 0.0:
        raise ValueError("animation event time and step must be finite; step positive")
    elapsed = 0.0
    frames = 0
    while elapsed < delay_seconds:
        advanced = struct.unpack("<f", struct.pack("<f", elapsed + step))[0]
        if advanced <= elapsed:
            raise ValueError("animation event exceeds float32 clock precision")
        elapsed = advanced
        frames += 1
    return frames


@dataclass
class SpawnEvent:
    time: float
    enemy_key: str
    route_index: int
    wave_start_time: float = 0.0
    nominal_time: float = 0.0
    scheduler_delay_frames: int = 0
    logic_frame: int | None = None


@dataclass(frozen=True)
class _SpawnWorkItem:
    nominal_time: float
    order: int
    kind: str
    enemy_key: str = ""
    route_index: int = 0
    action_count: int = 1


@dataclass(frozen=True)
class RoutePoint:
    row: float
    col: float
    kind: str = "MOVE"
    wait_time: float = 0.0
    reach_distance: float = 0.0
    reachable: bool = True
    # Movement includes reachOffset; the distance map is rooted at the grid.
    path_goal: tuple[int, int] | None = None
    allow_diagonal: bool = True


@dataclass
class OperatorPlan:
    char_id: str
    tile: tuple[int, int] | None
    time: float
    level: int = 1
    elite: int | None = None
    facing: int = 0  # 0 right, 1 down, 2 left, 3 up
    skill_id: str | None = None
    trust: float = 0.0
    potential_rank: int = 0
    module_id: str | None = None
    module_level: int = 0
    action: str = "DEPLOY"

    def __post_init__(self) -> None:
        self.action = self.action.upper()
        if self.tile is not None:
            self.tile = (int(self.tile[0]), int(self.tile[1]))


@dataclass
class Result:
    win: bool
    time: float
    life_points: int
    max_life_points: int
    enemies_spawned: int
    enemies_killed: int
    enemies_leaked: int
    operators_deployed: int
    reason: str
    behavior_warnings: int = 0
    unsupported_behavior_nodes: dict[str, int] = field(default_factory=dict)
    mechanic_warnings: int = 0
    unsupported_mechanics: dict[str, int] = field(default_factory=dict)
    seed: int = 0
    random_draws: int = 0
    assumptions: list[str] = field(default_factory=list)
    operator_metrics: list[dict[str, Any]] = field(default_factory=list)
    spawn_timing: str = SPAWN_TIMING_FAST
    enemy_attack_timing: str = ENEMY_ATTACK_TIMING_FLOAT


@dataclass
class _ScheduledHighlandEffect:
    execute_at: float
    source: "_Operator"
    tile: tuple[int, int]
    amount: float
    sp_recovery_locked: bool = False


@dataclass
class _Enemy:
    key: str
    name: str
    hp: float
    max_hp: float
    atk: float
    defense: float
    res: float
    move_speed: float
    attack_interval: float
    block_count: int
    mass_level: int
    range_radius: float
    life_reduce: int
    level_type: str
    motion: str
    path: list[tuple[int, int]]
    route: list[RoutePoint] = field(default_factory=list)
    apply_way: str = ApplyWay.NONE.value
    can_normal_attack: bool = True
    unblockable: bool = False
    wave_start_time: float = 0.0
    attack_duration: float = 0.0
    attack_hit_time: float = 0.0
    wait_for_attack_event: bool = False
    projectile_key: str | None = None
    graphic_profile: EnemyGraphicProfile | None = None
    facing: FacingRuntime | None = None
    attack_started_at: float | None = None
    tile_idx: int = 0
    move_cd: float = 0.0
    route_idx: int = 0
    position_row: float = 0.0
    position_col: float = 0.0
    spawn_move_lock_frames: int = 0
    spawn_attack_lock_frames: int = 0
    navigation_velocity_row: float = 0.0
    navigation_velocity_col: float = 0.0
    avoidance_row: float = 0.0
    avoidance_col: float = 0.0
    avoidance_tick_remaining: int = 0
    wait_remaining: float = 0.0
    wait_active_at_frame_start: bool = False
    wait_hidden_at_frame_start: bool = False
    wait_phase_consumed: bool = False
    force_velocity_row: float = 0.0
    force_velocity_col: float = 0.0
    force_deceleration: float = DEFAULT_FRICTION_DECELERATION
    unbalanced: bool = False
    unbalance_lock_remaining: float = 0.0
    pull_origin_row: float = 0.0
    pull_origin_col: float = 0.0
    pull_initial_distance: float = 0.0
    pull_force: float = 0.0
    pull_remaining: float = 0.0
    remaining_path_distance: float = 0.0
    attack_cd: float = 0.0
    attack_cd_frames: int = 0
    attack_cd_interval: float = 0.0
    attack_count: int = 0
    blocked_by: "_Operator | None" = None
    block_stable_row: float | None = None
    block_stable_col: float | None = None
    block_shift_target_row: float | None = None
    block_shift_target_col: float | None = None
    block_shift_start_row: float | None = None
    block_shift_start_col: float | None = None
    block_shift_elapsed: float = 0.0
    block_shift_remaining: float = 0.0
    pause_until: float = -1.0
    attack_move_resume_frame: int | None = None
    attack_target: "_Operator | None" = None
    target_search_ticker: FramePeriodicTicker = field(
        default_factory=lambda: FramePeriodicTicker(
            period_frames=ENEMY_TARGET_SEARCH_PERIOD_FRAMES
        )
    )
    cached_operator_target: "_Operator | None" = None
    target_search_after_attack: bool = False
    attack_hit_at: float = -1.0
    attack_hit_frame: int | None = None
    attack_will_hit: bool = True
    dead: bool = False
    leaked: bool = False
    disappeared: bool = False
    portal_targetable: bool = False
    elemental_resistance: float = 0.0
    taunt_level: int = 0
    creation_index: int = 0
    update_priority: int = 0
    first_update_frame: int = 0
    enemy_index: int = 0
    statuses: StatusController = field(default_factory=StatusController)
    damage_rules: list[DamageRule] = field(default_factory=list)
    physical_hit_rate: float = 1.0
    arts_hit_rate: float = 1.0
    critical_chance: float = 0.0
    critical_multiplier: float = 1.0
    elements: ElementController = field(default_factory=ElementController)
    element_burst: ElementBurstRuntime | None = None
    palsy_stacks: int = 0
    death_reason: str | None = None
    buff_blackboards: dict[str, dict[str, Any]] = field(default_factory=dict)
    buffs: BuffContainer = field(default_factory=BuffContainer)
    mode_index: int = 0

    def __post_init__(self) -> None:
        self.buffs.owner = self

    def tile(self) -> tuple[int, int]:
        return (
            math.floor(self.position_row + 0.5),
            math.floor(self.position_col + 0.5),
        )

    def position(self) -> tuple[float, float]:
        return self.position_row, self.position_col


@dataclass
class _PendingOperatorAttack:
    execute_at: float
    execute_frame: int
    kind: str
    primary: _Enemy
    targets: list[_Enemy]
    attack_scale: float
    source_atk_scale: float
    sp_recovery_locked: bool
    will_hit: bool


@dataclass
class _Projectile:
    projectile_id: int
    key: str
    speed: float
    side: str
    source: "_Enemy | _Operator"
    target: "_Enemy | _Operator"
    position_row: float
    position_col: float
    launched_at: float
    operator_attack: _PendingOperatorAttack | None = None
    will_hit: bool = True
    cached_atk: float | None = None
    parabolic: bool = False
    initial_distance: float = 0.0
    travelled: float = 0.0
    position_z: float = 0.0
    motion_profile: ProjectileMotionProfile | None = None

    def position(self) -> tuple[float, float]:
        return self.position_row, self.position_col


@dataclass
class _Operator:
    char_id: str
    name: str
    tile: tuple[int, int]
    hp: float
    max_hp: float
    atk: float
    defense: float
    res: float
    block_count: int
    cost: int
    attack_interval: float
    range_cells: set[tuple[int, int]]
    attack_duration: float = 0.0
    attack_hit_time: float = 0.0
    projectile_key: str | None = None
    skill_attack_timing: dict[str, dict[str, float]] = field(default_factory=dict)
    pending_attack: _PendingOperatorAttack | None = None
    facing: int = 0
    damage_type: str = "PHYSICAL"
    atk_scale: float = 1.0
    def_scale: float = 1.0
    attack_cd: float = 0.0
    dead: bool = False
    elemental_resistance: float = 0.0
    taunt_level: int = 0
    creation_index: int = 0
    update_priority: int = 0
    first_update_frame: int = 0
    deployment_index: int = 0
    statuses: StatusController = field(default_factory=StatusController)
    damage_rules: list[DamageRule] = field(default_factory=list)
    physical_hit_rate: float = 1.0
    arts_hit_rate: float = 1.0
    critical_chance: float = 0.0
    critical_multiplier: float = 1.0
    elements: ElementController = field(default_factory=ElementController)
    element_burst: ElementBurstRuntime | None = None
    palsy_stacks: int = 0
    base_cost: int = 0
    deployed_cost: int = 0
    respawn_time: float = 70.0
    departure_processed: bool = False
    retreated: bool = False
    blocked: list[_Enemy] = field(default_factory=list)
    block_scan_next_frame: int | None = None
    block_radius: float = DEFAULT_BLOCK_RADIUS
    multi_target: bool = False
    subprofession: str = ""
    prioritize_blocked: bool = False
    potential_rank: int = 0
    module_id: str | None = None
    module_level: int = 0
    base_range_cells: set[tuple[int, int]] = field(default_factory=set)
    deployed_at: float = 0.0
    death_time: float | None = None
    damage_dealt: float = 0.0
    kills: int = 0
    # skill state
    skill: dict[str, Any] | None = None
    sp: float = 0.0
    sp_max: float = 0.0
    sp_cost: float = 0.0
    max_charge_time: int = 1
    sp_type: str = ""
    sp_increment: float = 1.0
    sp_recovery_remaining_raw: int = CLIENT_FP_SCALE
    auto: bool = True
    skill_type: str = ""
    skill_duration_type: str = "NONE"
    skill_active: bool = False
    skill_startup_until: float = -1.0
    skill_end_reason: str | None = None
    skill_cast_count: int = 0
    ammo: float | None = None
    ammo_capacity: float | None = None
    ammo_cost_per_attack: float = 1.0
    next_attack_scale: float = 1.0
    timed_atk_scale: float = 1.0
    timed_def_scale: float = 1.0
    active_until: float = -1.0
    skill_blackboard: dict[str, Any] = field(default_factory=dict)
    behavior_blackboard: dict[str, Any] = field(default_factory=dict)
    behavior_template_key: str | None = None
    behavior_active: bool = False
    skill_prefab_id: str = ""
    behavior_trigger_count: int = 0
    behavior_next_trigger_at: float = -1.0
    behavior_finish_dispatched: bool = False
    buff_blackboards: dict[str, dict[str, Any]] = field(default_factory=dict)
    target_selector: TargetSelectorRuntime | None = None
    selector_first_search_frame: int | None = None
    # Character combat is a stateful chain.  The selector cache must survive
    # the hit/target departure until the current AttackState finishes.
    attack_chain_active: bool = False
    attack_first_cast: bool = False
    selector_resume_frame: int | None = None
    # Headb2 S2 can interrupt an in-flight attack.  In that case the native
    # selector does not consume Search/Next ticks during startup and one
    # additional fixed frame after startup.  Keep this separate from
    # selector_resume_frame, which belongs to AttackState target departure.
    selector_pause_until_frame: int | None = None

    @property
    def charge_count(self) -> int:
        if self.sp_cost <= 0:
            return 0
        return min(self.max_charge_time, int(self.sp // self.sp_cost))


_DAMAGE_TYPE_BY_PROFESSION = {
    "CASTER": "ARTS",
    "SUPPORTER": "ARTS",
    "MEDIC": "HEAL",
}

HERALD_KEYS = {"enemy_1080_sotidp", "enemy_1080_sotidp_2"}
SHIELDGUARD_KEYS = {"enemy_1081_sotisd", "enemy_1081_sotisd_2"}
MORTAR_KEYS = {"enemy_1082_soticn", "enemy_1082_soticn_2"}
GUERRILLA_FIGHTER_KEYS = {"enemy_1078_sotisc", "enemy_1078_sotisc_2"}
UNBLOCKABLE_KEYS = {"enemy_1053_norgst", "enemy_1053_norgst_2"}


class Battle:
    def __init__(
        self,
        level: dict[str, Any],
        enemy_index: dict[str, list[dict[str, Any]]],
        characters: dict[str, Any],
        ranges: dict[str, Any],
        plan: list[OperatorPlan],
        skills: dict[str, Any] | None = None,
        attack_timing: dict[str, dict[str, float]] | None = None,
        behavior_templates: dict[str, Any] | None = None,
        battle_equips: dict[str, Any] | None = None,
        projectile_data: dict[str, Any] | None = None,
        buff_abilities: dict[str, Any] | None = None,
        *,
        dt: float = 1.0 / 30.0,
        move_multiplier: float | None = None,
        strict_behaviors: bool = False,
        seed: int = 0,
        assumptions: SimulationAssumptions | None = None,
        spawn_timing: str = SPAWN_TIMING_FAST,
        enemy_attack_timing: str = ENEMY_ATTACK_TIMING_FLOAT,
        enemy_graphics: dict[str, Any] | None = None,
    ) -> None:
        self.level = level
        self.enemy_index = enemy_index
        self.characters = characters
        self.ranges = ranges
        self.skills = skills or {}
        self.attack_timing = attack_timing or {}
        self.behavior_diagnostics = BehaviorDiagnostics()
        self.mechanic_diagnostics: Counter[str] = Counter()
        self.behavior = BehaviorInterpreter(
            behavior_templates,
            strict=strict_behaviors,
            diagnostics=self.behavior_diagnostics,
        )
        self.battle_equips = battle_equips or {}
        self.projectile_data = projectile_data or {}
        if buff_abilities is None and D.DATA_DIR is not None:
            buff_abilities = D.load_enemy_buff_abilities()
        self.buff_catalog = BuffCatalog.from_dict(buff_abilities)
        self.projectile_speeds: dict[str, float] = {
            str(key): float(value)
            for key, value in self.projectile_data.get("speeds", {}).items()
        }
        self.enemy_projectiles = self.projectile_data.get("enemies", {})
        self.operator_projectiles = self.projectile_data.get("operators", {})
        self.plan = sorted(plan, key=lambda p: p.time)
        self.dt = dt
        self.seed = int(seed)
        self.rng = DeterministicRng(self.seed)
        self.assumptions = assumptions or DEFAULT_ASSUMPTIONS
        self.enemy_graphics: dict[str, EnemyGraphicProfile] = {}
        self.operator_hit_offsets: dict[str, tuple[float, float, float]] = {}
        self.projectile_motion_profiles: dict[str, ProjectileMotionProfile] = {}
        if (self.assumptions.enemy_projectile_muzzle
                or self.assumptions.enemy_facing_transition):
            if enemy_graphics is None and D.DATA_DIR is not None:
                enemy_graphics = D.load_enemy_graphics()
            profiles = (enemy_graphics or {}).get("enemies", {})
            if not isinstance(profiles, dict):
                raise ValueError("enemy graphics 'enemies' must be an object")
            for key, spec in profiles.items():
                self.enemy_graphics[key] = EnemyGraphicProfile.from_dict(spec)
            if self.assumptions.enemy_projectile_muzzle:
                hit_profiles = (enemy_graphics or {}).get("operators", {})
                if not isinstance(hit_profiles, dict):
                    raise ValueError("enemy graphics 'operators' must be an object")
                for key, spec in hit_profiles.items():
                    if not isinstance(spec, dict) or spec.get("coordinateSpace") != "map":
                        raise ValueError("operator hit profiles require coordinateSpace='map'")
                    self.operator_hit_offsets[key] = map_offset(spec["hitOffset"])
                motion_profiles = self.projectile_data.get("motion", {})
                if not isinstance(motion_profiles, dict):
                    raise ValueError("projectile motion must be an object")
                for key, spec in motion_profiles.items():
                    self.projectile_motion_profiles[key] = ProjectileMotionProfile.from_dict(spec)
        self.used_assumptions: set[str] = set()
        self.spawn_timing = self._normalize_spawn_timing(spawn_timing)
        self.enemy_attack_timing = self._normalize_enemy_attack_timing(
            enemy_attack_timing
        )
        if self.spawn_timing == SPAWN_TIMING_CLIENT:
            if not math.isclose(self.dt, 1.0 / CLIENT_LOGIC_RATE, abs_tol=1e-12):
                raise ValueError("client spawn timing requires dt=1/30")
            self.used_assumptions.add("CLIENT_COROUTINE_SPAWN_TIMING")
        if self.enemy_attack_timing == ENEMY_ATTACK_TIMING_MORTAR_FRAMES:
            if not math.isclose(self.dt, 1.0 / CLIENT_LOGIC_RATE, abs_tol=1e-12):
                raise ValueError("mortar frame timing requires dt=1/30")
            self.used_assumptions.add("MORTAR_ATTACK_INTERVAL_LOGIC_FRAMES")

        options = level.get("options", {})
        self.max_life_points = options.get("maxLifePoint", 3)
        self.life_points = self.max_life_points
        self.initial_cost = options.get("initialCost", 0)
        self.max_cost = options.get("maxCost", 99)
        self.character_limit = int(options.get("characterLimit", 8))
        self.cost_increase_time = options.get("costIncreaseTime", 1.0)
        self.move_multiplier = (
            options.get("moveMultiplier", 1.0)
            if move_multiplier is None
            else move_multiplier
        )
        self.steering_enabled = bool(options.get("steeringEnabled", True))

        self.map_grid: list[list[int]] = level["mapData"]["map"]
        self.tiles: list[dict[str, Any]] = level["mapData"]["tiles"]
        self._path_map_cache: dict[
            tuple[tuple[int, int], bool],
            dict[tuple[int, int], tuple[int, int]],
        ] = {}
        self.routes = self._build_routes(level.get("routes", []))
        self.paths = self._routes_to_paths(self.routes)
        self.spawn_events = self._build_spawn_events(level.get("waves", []))
        self.spawn_index = 0

        self.dp = self.initial_cost
        self.dp_timer = 0.0
        self.time = 0.0
        self.frame_index = 0
        self.enemies: list[_Enemy] = []
        self.operators: list[_Operator] = []
        self.plan_index = 0
        self.enemies_spawned = 0
        self.enemies_killed = 0
        self.enemies_leaked = 0
        self.end_reason = ""
        self.creation_counter = 0
        self.enemy_counter = 0
        self.deployment_counter = 0
        self.damage_rule_counter = 0
        self.redeploy_penalty: dict[str, int] = {}
        self.redeploy_ready_at: dict[str, float] = {}
        self.scheduled_highland_effects: list[_ScheduledHighlandEffect] = []
        self.projectiles: list[_Projectile] = []
        self.projectile_counter = 0
        self.targeting_diagnostics: list[dict[str, Any]] = []
        self._targeting_diagnostic_index: dict[tuple[int, int], int] = {}
        self._targeting_first_in_range: dict[tuple[int, int], int] = {}

    # ------------------------------------------------------------------ setup
    def _build_routes(
        self, routes: list[dict[str, Any]]
    ) -> list[list[RoutePoint]]:
        built: list[list[RoutePoint]] = []
        for route in routes:
            checkpoints = route.get("checkpoints") or []
            if bool(route.get("visitEveryTileCenter", False)):
                navigation_mode = "TILE_CENTER"
            elif bool(route.get("visitEveryNodeCenter", False)):
                navigation_mode = "NODE_CENTER"
            elif bool(route.get("visitEveryNodeStably", False)) or not checkpoints:
                navigation_mode = "NODE_STABLE"
            else:
                navigation_mode = "NORMAL"
            start = route.get("startPosition", {})
            spawn_offset = route.get("spawnOffset") or {}
            points = [
                RoutePoint(
                    float(start.get("row", 0)) + float(spawn_offset.get("y", 0)),
                    float(start.get("col", 0)) + float(spawn_offset.get("x", 0)),
                    "START",
                )
            ]
            for checkpoint in checkpoints:
                position = checkpoint.get("position") or {}
                offset = checkpoint.get("reachOffset") or {}
                points.append(
                    RoutePoint(
                        float(position.get("row", 0)) + float(offset.get("y", 0)),
                        float(position.get("col", 0)) + float(offset.get("x", 0)),
                        str(checkpoint.get("type", "MOVE")).upper(),
                        max(float(checkpoint.get("time", 0) or 0), 0.0),
                        max(float(checkpoint.get("reachDistance", 0) or 0), 0.0),
                        path_goal=self._position_cell(
                            float(position.get("row", 0)),
                            float(position.get("col", 0)),
                        ),
                        allow_diagonal=bool(route.get("allowDiagonalMove", True)),
                    )
                )
            end = route.get("endPosition", {})
            points.append(
                RoutePoint(
                    float(end.get("row", 0)),
                    float(end.get("col", 0)),
                    "END",
                    allow_diagonal=bool(route.get("allowDiagonalMove", True)),
                )
            )
            motion = str(route.get("motionMode", "WALK")).upper()
            if motion not in {"FLY", "FLYING", "E_NUM"}:
                points = self._expand_ground_route(
                    points,
                    allow_diagonal=bool(route.get("allowDiagonalMove", True)),
                    navigation_mode=navigation_mode,
                )
            built.append(points)
        return built

    @staticmethod
    def _routes_to_paths(
        routes: list[list[RoutePoint]],
    ) -> list[list[tuple[int, int]]]:
        non_spatial = {
            "WAIT_FOR_SECONDS",
            "WAIT_CURRENT_WAVE_TIME",
            "DISAPPEAR",
        }
        return [
            [
                (round(point.row), round(point.col))
                for point in route
                if point.kind.upper() not in non_spatial
            ]
            for route in routes
        ]

    def _expand_ground_route(
        self,
        points: list[RoutePoint],
        *,
        allow_diagonal: bool,
        navigation_mode: str = "NORMAL",
    ) -> list[RoutePoint]:
        if not points:
            return []
        expanded = [points[0]]
        current = points[0]
        spatial_kinds = {"MOVE", "END"}
        teleport_kinds = {"APPEAR_AT_POS", "TELEPORT"}
        for point in points[1:]:
            kind = point.kind.upper()
            if kind in spatial_kinds:
                start_cell = self._position_cell(current.row, current.col)
                goal_cell = self._position_cell(point.row, point.col)
                cells = self._spfa_cells(
                    start_cell,
                    goal_cell,
                    force_goal=kind == "MOVE",
                )
                if cells is None:
                    expanded.append(
                        RoutePoint(
                            point.row,
                            point.col,
                            point.kind,
                            point.wait_time,
                            point.reach_distance,
                            False,
                            point.path_goal,
                            point.allow_diagonal,
                        )
                    )
                    current = point
                    continue
                if len(cells) > 1:
                    self.used_assumptions.add("REVERSE_ENGINEERED_NEXT_NODE")
                if navigation_mode == "TILE_CENTER":
                    guide_kind = "NAVIGATE_CENTER"
                elif allow_diagonal:
                    cells = self._smooth_cell_path(
                        cells,
                        goal=goal_cell,
                        force_goal=kind == "MOVE",
                    )
                    guide_kind = {
                        "NODE_CENTER": "NAVIGATE_CENTER",
                        "NODE_STABLE": "NAVIGATE_STABLE",
                    }.get(navigation_mode, "NAVIGATE_CELL")
                else:
                    cells = self._compress_path_to_turns(cells)
                    guide_kind = {
                        "NODE_CENTER": "NAVIGATE_CENTER",
                        "NODE_STABLE": "NAVIGATE_STABLE",
                    }.get(navigation_mode, "NAVIGATE_CELL")
                if len(cells) > 2:
                    expanded.extend(
                        RoutePoint(
                            float(row),
                            float(col),
                            guide_kind,
                            reach_distance=(
                                0.25
                                if guide_kind == "NAVIGATE_STABLE"
                                else 0.05
                                if guide_kind == "NAVIGATE_CENTER"
                                else 0.0
                            ),
                        )
                        for row, col in cells[1:-1]
                    )
                expanded.append(point)
                current = point
            else:
                expanded.append(point)
                if kind in teleport_kinds:
                    current = point
        return expanded

    def _spfa_cells(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        *,
        force_goal: bool = False,
    ) -> list[tuple[int, int]] | None:
        """Build a four-way Manhattan distance map and follow its nextNodes."""
        if start == goal:
            return [start]
        cache_key = (goal, force_goal)
        next_nodes = self._path_map_cache.get(cache_key)
        if next_nodes is None:
            next_nodes = {}
            costs = {goal: 0.0}
            pending = deque([goal])
            queued = {goal}
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

            while pending:
                current = pending.popleft()
                queued.remove(current)
                for dr, dc in directions:
                    following = (current[0] + dr, current[1] + dc)
                    terrain_cost = self._ground_path_cost(
                        *following,
                        goal=goal,
                        force_goal=force_goal,
                    )
                    if not math.isfinite(terrain_cost):
                        continue
                    new_cost = costs[current] + terrain_cost
                    if new_cost + 1e-9 >= costs.get(following, math.inf):
                        continue
                    costs[following] = new_cost
                    next_nodes[following] = current
                    if following not in queued:
                        pending.append(following)
                        queued.add(following)
            self._path_map_cache[cache_key] = next_nodes

        if start not in next_nodes:
            return None
        path = [start]
        current = start
        visited = {start}
        while current != goal:
            current = next_nodes[current]
            if current in visited:
                self.mechanic_diagnostics.setdefault("PATH_NEXT_NODE_CYCLE", 1)
                return None
            path.append(current)
            visited.add(current)
        return path

    @staticmethod
    def _compress_path_to_turns(
        cells: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if len(cells) <= 2:
            return cells
        result = [cells[0]]
        previous_direction = (
            cells[1][0] - cells[0][0],
            cells[1][1] - cells[0][1],
        )
        for index in range(2, len(cells)):
            direction = (
                cells[index][0] - cells[index - 1][0],
                cells[index][1] - cells[index - 1][1],
            )
            if direction != previous_direction:
                result.append(cells[index - 1])
                previous_direction = direction
        result.append(cells[-1])
        return result

    def _smooth_cell_path(
        self,
        cells: list[tuple[int, int]],
        *,
        goal: tuple[int, int],
        force_goal: bool,
    ) -> list[tuple[int, int]]:
        if len(cells) <= 2:
            return cells
        smoothed = [cells[0]]
        anchor = 0
        while anchor < len(cells) - 1:
            following = anchor + 1
            for candidate in range(len(cells) - 1, anchor, -1):
                if self._smoothed_link_is_clear(
                    cells[anchor],
                    cells[candidate],
                    goal=goal,
                    force_goal=force_goal,
                ):
                    following = candidate
                    break
            smoothed.append(cells[following])
            anchor = following
        return smoothed

    def _smoothed_link_is_clear(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        *,
        goal: tuple[int, int],
        force_goal: bool,
    ) -> bool:
        for cell in self._modified_bresenham_cells(start, end):
            cost = self._ground_path_cost(
                *cell,
                goal=goal,
                force_goal=force_goal,
            )
            if not math.isfinite(cost):
                return False
            if cost > 1.0 and not (force_goal and cell == goal):
                return False
        return True

    @staticmethod
    def _modified_bresenham_cells(
        start: tuple[int, int], end: tuple[int, int]
    ) -> set[tuple[int, int]]:
        row0, col0 = start
        row1, col1 = end
        dr = abs(row1 - row0)
        dc = abs(col1 - col0)
        if dr == 1 or dc == 1:
            return {
                (row, col)
                for row in range(min(row0, row1), max(row0, row1) + 1)
                for col in range(min(col0, col1), max(col0, col1) + 1)
            }

        step_row = 1 if row1 > row0 else -1
        step_col = 1 if col1 > col0 else -1
        error = dr - dc
        row, col = row0, col0
        checked = {(row, col)}
        while (row, col) != (row1, col1):
            previous_row, previous_col = row, col
            doubled = 2 * error
            if doubled > -dc:
                error -= dc
                row += step_row
            if doubled < dr:
                error += dr
                col += step_col
            checked.add((row, col))
            if row != previous_row and col != previous_col:
                checked.add((row, previous_col))
                checked.add((previous_row, col))
        return checked

    def _ground_path_cost(
        self,
        row: int,
        col: int,
        *,
        goal: tuple[int, int] | None = None,
        force_goal: bool = False,
    ) -> float:
        tile = self.tile(row, col)
        if not tile:
            return math.inf
        if self._is_hole_cell(row, col):
            return 1_000_000.0
        if self._ground_passable_cell(row, col):
            return 1.0
        if force_goal and goal == (row, col):
            return 1_000.0
        return math.inf

    @staticmethod
    def _normalize_spawn_timing(value: str) -> str:
        normalized = _SPAWN_TIMING_ALIASES.get(str(value).strip().lower())
        if normalized is None:
            choices = ", ".join(sorted(_SPAWN_TIMING_ALIASES))
            raise ValueError(f"unknown spawn timing {value!r}; expected one of {choices}")
        return normalized

    @staticmethod
    def _normalize_enemy_attack_timing(value: str) -> str:
        normalized = _ENEMY_ATTACK_TIMING_ALIASES.get(str(value).strip().lower())
        if normalized is None:
            choices = ", ".join(sorted(_ENEMY_ATTACK_TIMING_ALIASES))
            raise ValueError(
                f"unknown enemy attack timing {value!r}; expected one of {choices}"
            )
        return normalized

    def _build_spawn_events(self, waves: list[dict[str, Any]]) -> list[SpawnEvent]:
        if self.spawn_timing == SPAWN_TIMING_FRAME_CORE:
            return self._build_frame_core_spawn_events(waves)
        if self.spawn_timing == SPAWN_TIMING_CLIENT:
            return self._build_client_spawn_events(waves)
        return self._build_fast_spawn_events(waves)

    def _build_frame_core_spawn_events(
        self, waves: list[dict[str, Any]]
    ) -> list[SpawnEvent]:
        """Use the published SpawnCore queue arithmetic for static wave data."""
        from .spawn_core import build_spawn_frame_schedule

        rows = build_spawn_frame_schedule(waves)
        events = []
        for row in rows:
            if row["actionType"] != "SPAWN":
                continue
            item = row["item"]
            frame = int(row["actual_frame"])
            events.append(SpawnEvent(
                time=(frame + 1) / CLIENT_LOGIC_RATE,
                enemy_key=item["key"],
                route_index=item["route"],
                wave_start_time=(row["wave_start_frame"] + 1) / CLIENT_LOGIC_RATE,
                nominal_time=row["ideal_frame"] / CLIENT_LOGIC_RATE,
                scheduler_delay_frames=frame - int(row["ideal_frame"]),
                logic_frame=frame + 1,
            ))
        if len(waves) > 1:
            self.mechanic_diagnostics.setdefault(
                "SPAWN_TIMING_FRAME_CORE_STATIC_WAVE_GATE", 1
            )
        self.used_assumptions.add("SPAWN_CORE_FRAME_SCHEDULER")
        return events

    @staticmethod
    def _build_fast_spawn_events(
        waves: list[dict[str, Any]],
    ) -> list[SpawnEvent]:
        events: list[SpawnEvent] = []
        t = 0.0
        for wave in waves:
            t += wave.get("preDelay", 0.0) or 0.0
            wave_start_time = t
            for frag in wave.get("fragments", []):
                t += frag.get("preDelay", 0.0) or 0.0
                frag_end = t
                for action in Battle._client_sorted_actions(
                    frag.get("actions", [])
                ):
                    if action.get("actionType") != "SPAWN":
                        continue
                    count = action.get("count", 1) or 1
                    interval = action.get("interval", 1.0) or 1.0
                    pre_delay = action.get("preDelay", 0.0) or 0.0
                    route_index = action.get("routeIndex", 0) or 0
                    for k in range(count):
                        events.append(
                            SpawnEvent(
                                time=t + pre_delay + k * interval,
                                enemy_key=action.get("key", ""),
                                route_index=route_index,
                                wave_start_time=wave_start_time,
                                nominal_time=t + pre_delay + k * interval,
                            )
                        )
                    frag_end = max(
                        frag_end, t + pre_delay + (count - 1) * interval
                    )
                t = frag_end
            t += wave.get("postDelay", 0.0) or 0.0
        events.sort(key=lambda e: e.time)
        return events

    @staticmethod
    def _client_sorted_actions(
        actions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Reproduce the client's unstable timeOffset quicksort.

        The IL2CPP Scheduler sorts ActionItem only by timeOffset.  Its legacy
        Hoare partition swaps equal elements, which is observable for H7-2's
        five zero-delay shieldguards as 1,2,3,4,5 -> 4,5,3,1,2.
        """
        ordered = list(actions)

        def delay(index: int) -> float:
            return float(ordered[index].get("preDelay", 0.0) or 0.0)

        def quicksort(left: int, right: int) -> None:
            i, j = left, right
            pivot = delay((left + right) // 2)
            while i <= j:
                while delay(i) < pivot:
                    i += 1
                while delay(j) > pivot:
                    j -= 1
                if i <= j:
                    ordered[i], ordered[j] = ordered[j], ordered[i]
                    i += 1
                    j -= 1
            if left < j:
                quicksort(left, j)
            if i < right:
                quicksort(i, right)

        if len(ordered) > 1:
            quicksort(0, len(ordered) - 1)
        return ordered

    @staticmethod
    def _client_logic_frame(time_seconds: float) -> int:
        return max(0, math.ceil(time_seconds * CLIENT_LOGIC_RATE - 1e-9))

    def _build_client_spawn_events(
        self, waves: list[dict[str, Any]]
    ) -> list[SpawnEvent]:
        """Build the observed client-equivalent sequential coroutine timeline.

        H7-2 and 7-16 frame captures show one accumulated logic frame for each
        entered fragment, completed spawn, preview cursor, and calibrated
        display action.  The costs are folded into final spawn frames here so
        the hot battle loop stays flat.

        Auto-preview cursor offsets are relative to the associated spawn time;
        they may therefore precede the nominal start of their Fragment.
        """
        events: list[SpawnEvent] = []
        nominal_cursor = 0.0
        completed_work_frames = 0

        if len(waves) > 1:
            self.mechanic_diagnostics.setdefault(
                "SPAWN_TIMING:STATIC_WAVE_TRANSITION", 1
            )

        for wave in waves:
            nominal_cursor += float(wave.get("preDelay", 0.0) or 0.0)
            wave_start_frame = (
                self._client_logic_frame(nominal_cursor) + completed_work_frames
            )
            wave_start_time = wave_start_frame / CLIENT_LOGIC_RATE

            for fragment in wave.get("fragments", []):
                nominal_cursor += float(fragment.get("preDelay", 0.0) or 0.0)
                fragment_start = nominal_cursor
                fragment_end = fragment_start
                order = 0
                work_items = [
                    _SpawnWorkItem(fragment_start, order, "FRAGMENT_ENTER")
                ]
                order += 1

                actions = self._client_sorted_actions(
                    fragment.get("actions", [])
                )
                deferred_display_times: list[float] = []
                if actions != list(fragment.get("actions", [])):
                    self.used_assumptions.add("CLIENT_UNSTABLE_ACTION_SORT")
                for action in actions:
                    action_type = str(action.get("actionType", "")).upper()
                    count = int(action.get("count", 1) or 1)
                    interval = float(action.get("interval", 1.0) or 1.0)
                    pre_delay = float(action.get("preDelay", 0.0) or 0.0)
                    first_time = fragment_start + pre_delay
                    last_time = first_time + (count - 1) * interval
                    fragment_end = max(fragment_end, last_time)

                    if action_type == "SPAWN" and bool(
                        action.get("autoPreviewRoute", False)
                    ):
                        for cursor_offset in (-3.0, -2.7):
                            # Preview timing is relative to the spawn, not
                            # clamped to the Fragment start.
                            work_items.append(
                                _SpawnWorkItem(
                                    first_time + cursor_offset,
                                    order,
                                    "PREVIEW_CURSOR",
                                    action_count=count,
                                )
                            )
                            order += 1

                    if action_type == "SPAWN":
                        route_index = int(action.get("routeIndex", 0) or 0)
                        for repeat_index in range(count):
                            work_items.append(
                                _SpawnWorkItem(
                                    first_time + repeat_index * interval,
                                    order,
                                    "SPAWN",
                                    str(action.get("key", "")),
                                    route_index,
                                    count,
                                )
                            )
                            order += 1
                    elif action_type == "PREVIEW_CURSOR":
                        for repeat_index in range(count):
                            work_items.append(
                                _SpawnWorkItem(
                                    first_time + repeat_index * interval,
                                    order,
                                    "PREVIEW_CURSOR",
                                    action_count=count,
                                )
                            )
                            order += 1
                    elif action_type == "DISPLAY_ENEMY_INFO":
                        self.used_assumptions.add(
                            "CLIENT_DISPLAY_ENEMY_INFO_WORK_FRAME"
                        )
                        for repeat_index in range(count):
                            deferred_display_times.append(
                                first_time + repeat_index * interval
                            )
                    else:
                        self.mechanic_diagnostics.setdefault(
                            f"SPAWN_TIMING_ACTION:{action_type or 'UNKNOWN'}", 1
                        )

                # DISPLAY_ENEMY_INFO is an executor/coroutine of its own.  The
                # H7-2 same-time fit places its completion after the regular
                # work at that offset, while its autoPreviewRoute flag does
                # not provide evidence for two additional cursor items.
                for display_time in deferred_display_times:
                    work_items.append(
                        _SpawnWorkItem(
                            display_time, order, "DISPLAY_ENEMY_INFO"
                        )
                    )
                    order += 1

                # PreviewRoute is a separate coroutine.  A cursor that lands
                # on the same nominal frame as a SPAWN is observed before
                # that SPAWN completes, but its one-frame work does not shift
                # later action offsets.  Keep the action queue order for the
                # cumulative counter and apply this local completion offset to
                # the affected spawn below.
                work_items.sort(key=lambda item: (item.nominal_time, item.order))
                for item in work_items:
                    if item.kind == "SPAWN":
                        nominal_frame = self._client_logic_frame(item.nominal_time)
                        preview_after = sum(
                            1
                            for preview in work_items
                            if (
                                preview.kind == "PREVIEW_CURSOR"
                                and preview.nominal_time == item.nominal_time
                                and preview.order > item.order
                                and item.action_count == 1
                            )
                        )
                        if preview_after:
                            self.used_assumptions.add(
                                "CLIENT_PREVIEW_COMPLETION_ORDER"
                            )
                        logic_frame = (
                            nominal_frame
                            + completed_work_frames
                            + preview_after
                        )
                        events.append(
                            SpawnEvent(
                                time=logic_frame / CLIENT_LOGIC_RATE,
                                enemy_key=item.enemy_key,
                                route_index=item.route_index,
                                wave_start_time=wave_start_time,
                                nominal_time=item.nominal_time,
                                scheduler_delay_frames=(
                                    completed_work_frames + preview_after
                                ),
                                logic_frame=logic_frame,
                            )
                        )
                    completed_work_frames += 1

                nominal_cursor = fragment_end

            nominal_cursor += float(wave.get("postDelay", 0.0) or 0.0)

        events.sort(
            key=lambda event: (
                event.logic_frame if event.logic_frame is not None else math.inf,
                event.time,
            )
        )
        return events

    # ------------------------------------------------------------------ tiles
    def tile(self, row: int, col: int) -> dict[str, Any]:
        if 0 <= row < len(self.map_grid) and 0 <= col < len(self.map_grid[0]):
            grid_row = len(self.map_grid) - 1 - row
            return self.tiles[self.map_grid[grid_row][col]]
        return {}

    @staticmethod
    def _position_cell(row: float, col: float) -> tuple[int, int]:
        return math.floor(row + 0.5), math.floor(col + 0.5)

    @staticmethod
    def _is_airborne(enemy: _Enemy) -> bool:
        return (
            enemy.statuses.has(Status.LEVITATE)
            or enemy.motion.upper() in {"FLY", "FLYING"}
        )

    def _is_hole_cell(self, row: int, col: int) -> bool:
        return self.tile(row, col).get("tileKey") == "tile_hole"

    def _ground_passable_cell(self, row: int, col: int) -> bool:
        tile = self.tile(row, col)
        if not tile:
            return False
        mask = tile.get("passableMask", "ALL")
        if isinstance(mask, (int, float)):
            return bool(int(mask) & 1)
        normalized = str(mask).upper()
        if normalized in {"ALL", "GROUND_ONLY", "BOTH"}:
            return True
        if normalized in {"FLY_ONLY", "NONE"}:
            return False
        self.mechanic_diagnostics.setdefault(
            f"PASSABLE_MASK:{normalized or '<empty>'}", 1
        )
        return False

    @staticmethod
    def _segment_cells(
        start: tuple[float, float], end: tuple[float, float]
    ) -> list[tuple[tuple[int, int], float]]:
        cuts = {0.0, 1.0}
        for initial, final in zip(start, end):
            delta = final - initial
            if abs(delta) <= 1e-12:
                continue
            low, high = sorted((initial, final))
            boundary_index = math.floor(low - 0.5) + 1
            boundary = boundary_index + 0.5
            while boundary < high - 1e-12:
                cuts.add((boundary - initial) / delta)
                boundary_index += 1
                boundary = boundary_index + 0.5

        raw_ordered = sorted(
            cut for cut in cuts if -1e-12 <= cut <= 1 + 1e-12
        )
        ordered: list[float] = []
        for cut in raw_ordered:
            if not ordered or cut - ordered[-1] > 1e-9:
                ordered.append(cut)
            else:
                ordered[-1] = max(ordered[-1], cut)
        cells: list[tuple[tuple[int, int], float]] = []
        for entry, leave in zip(ordered, ordered[1:]):
            middle = (entry + leave) * 0.5
            row = start[0] + (end[0] - start[0]) * middle
            col = start[1] + (end[1] - start[1]) * middle
            cell = Battle._position_cell(row, col)
            if not cells or cells[-1][0] != cell:
                cells.append((cell, max(entry, 0.0)))
        endpoint = Battle._position_cell(*end)
        if not cells or cells[-1][0] != endpoint:
            cells.append((endpoint, 1.0))
        return cells

    def _move_with_terrain(
        self,
        enemy: _Enemy,
        end: tuple[float, float],
        *,
        forced_goal_cell: tuple[int, int] | None = None,
        reflect_collision: bool = False,
    ) -> str:
        start = enemy.position()
        if not hasattr(self, "map_grid") or self._is_airborne(enemy):
            enemy.position_row, enemy.position_col = end
            return "MOVED"

        for (row, col), entry in self._segment_cells(start, end):
            if self._is_hole_cell(row, col):
                enemy.position_row = start[0] + (end[0] - start[0]) * entry
                enemy.position_col = start[1] + (end[1] - start[1]) * entry
                self.mechanic_diagnostics.setdefault("PIT_HITBOX_CENTER_ONLY", 1)
                self._kill_enemy_by_terrain(enemy, "PIT")
                return "FELL"
            if (
                not self._ground_passable_cell(row, col)
                and (row, col) != forced_goal_cell
            ):
                if reflect_collision:
                    displacement = (end[0] - start[0], end[1] - start[1])
                    obstacle_direction = (row - start[0], col - start[1])
                    obstacle_distance = math.hypot(*obstacle_direction)
                    if obstacle_distance > 1e-12:
                        unit_direction = (
                            obstacle_direction[0] / obstacle_distance,
                            obstacle_direction[1] / obstacle_distance,
                        )
                        projection = (
                            displacement[0] * unit_direction[0]
                            + displacement[1] * unit_direction[1]
                        )
                        corrected = (
                            start[0]
                            + displacement[0]
                            - 2.0 * projection * unit_direction[0],
                            start[1]
                            + displacement[1]
                            - 2.0 * projection * unit_direction[1],
                        )
                        self.mechanic_diagnostics.setdefault(
                            "TERRAIN_COLLISION:REFLECT", 1
                        )
                        return self._move_with_terrain(
                            enemy,
                            corrected,
                            forced_goal_cell=forced_goal_cell,
                        )
                stop = max(0.0, entry - 1e-7)
                enemy.position_row = start[0] + (end[0] - start[0]) * stop
                enemy.position_col = start[1] + (end[1] - start[1]) * stop
                enemy.force_velocity_row = 0.0
                enemy.force_velocity_col = 0.0
                self.mechanic_diagnostics.setdefault(
                    "TERRAIN_COLLISION:CENTER_STOP", 1
                )
                return "COLLISION"

        enemy.position_row, enemy.position_col = end
        return "MOVED"

    def _kill_enemy_by_terrain(self, enemy: _Enemy, reason: str) -> None:
        if enemy.dead:
            return
        enemy.hp = 0.0
        enemy.dead = True
        enemy.death_reason = reason
        enemy.leaked = False
        self.enemies_killed += 1

    def _recover_invalid_terrain(self, enemy: _Enemy) -> bool:
        if (
            not hasattr(self, "map_grid")
            or enemy.disappeared
            or self._is_airborne(enemy)
        ):
            return False
        row, col = self._position_cell(*enemy.position())
        if self._is_hole_cell(row, col):
            self.mechanic_diagnostics.setdefault("PIT_HITBOX_CENTER_ONLY", 1)
            self._kill_enemy_by_terrain(enemy, "PIT")
            return True
        if self._ground_passable_cell(row, col):
            return False

        candidates = [
            (near_row, near_col)
            for near_row in range(row - 1, row + 2)
            for near_col in range(col - 1, col + 2)
            if (near_row, near_col) != (row, col)
            and self._ground_passable_cell(near_row, near_col)
            and not self._is_hole_cell(near_row, near_col)
        ]
        if not candidates:
            self.mechanic_diagnostics.setdefault("TERRAIN_RECOVERY:NO_EXIT", 1)
            return True
        target = min(
            candidates,
            key=lambda cell: (
                math.hypot(cell[0] - enemy.position_row, cell[1] - enemy.position_col),
                cell,
            ),
        )
        dr = target[0] - enemy.position_row
        dc = target[1] - enemy.position_col
        distance = math.hypot(dr, dc)
        travel = min(0.0625 * self.dt, distance)
        if distance > 1e-9:
            enemy.position_row += dr / distance * travel
            enemy.position_col += dc / distance * travel
        self.mechanic_diagnostics.setdefault("TERRAIN_RECOVERY:LOCAL_APPROX", 1)
        return True

    def is_buildable(self, row: int, col: int, profession: str) -> bool:
        tile = self.tile(row, col)
        if tile.get("buildableType") == "NONE":
            return False
        if profession == "MELEE":
            return tile.get("buildableType") == "MELEE"
        return tile.get("buildableType") == "RANGED"

    def tile_occupied(self, row: int, col: int) -> _Operator | None:
        for op in self.operators:
            if op.tile == (row, col) and not op.dead:
                return op
        return None

    # ------------------------------------------------------------- range
    def range_cells(
        self,
        char: dict[str, Any],
        tile: tuple[int, int],
        facing: int,
        elite: int | None = None,
    ) -> set[tuple[int, int]]:
        phases = char.get("phases", [])
        if not phases:
            return {tile}
        phase = phases[0] if elite is None else phases[max(0, min(elite, len(phases) - 1))]
        range_id = phase.get("rangeId")
        if not range_id:
            return {tile}
        return self.range_cells_from_id(str(range_id), tile, facing)

    def range_cells_from_id(
        self, range_id: str, tile: tuple[int, int], facing: int
    ) -> set[tuple[int, int]]:
        range_def = self.ranges.get(range_id, {})
        cells: set[tuple[int, int]] = set()
        for grid in range_def.get("grids", []):
            dr, dc = grid.get("row", 0), grid.get("col", 0)
            if facing % 4 == 0:  # right (identity)
                rr, cc = dr, dc
            elif facing % 4 == 1:  # down
                rr, cc = dc, -dr
            elif facing % 4 == 2:  # left
                rr, cc = -dr, -dc
            else:  # up
                rr, cc = -dc, dr
            cells.add((tile[0] + rr, tile[1] + cc))
        return cells

    # ------------------------------------------------------------- specs
    def _make_enemy(
        self, key: str, route_index: int, *, wave_start_time: float = 0.0
    ) -> _Enemy:
        enemy_data = D.enemy_data_at(self.enemy_index, key, level=0)
        attr = enemy_data.get("attributes", {})
        hp = float(D.mdef(attr.get("maxHp"), 0))
        motion = str(D.mdef(enemy_data.get("motion"), "WALK"))
        life_reduce = int(D.mdef(enemy_data.get("lifePointReduce"), 1)) or 1
        level_type = str(D.mdef(enemy_data.get("levelType"), "NORMAL"))
        name = enemy_data.get("name", {})
        name = name.get("m_value", key) if isinstance(name, dict) else key
        move_speed = float(D.mdef(attr.get("moveSpeed"), 1.0))
        path = list(self.paths[route_index])
        route = list(self.routes[route_index])
        start = route[0] if route else RoutePoint(0.0, 0.0, "START")
        attack_interval = float(D.mdef(attr.get("baseAttackTime"), 1.0))
        apply_way = str(D.mdef(enemy_data.get("applyWay"), "NONE")).upper()
        timing = self.attack_timing.get(key)
        projectile_spec = self.enemy_projectiles.get(key) or {}
        projectile_key = str(projectile_spec.get("key") or "") or None
        if timing:
            attack_duration = float(timing.get("attack_duration") or 0.0)
            attack_hit_time = float(timing.get("attack_hit_time") or 0.0)
        else:
            attack_duration = 0.5 * attack_interval
            attack_hit_time = 0.0
        self.creation_counter += 1
        self.enemy_counter += 1
        allocation_order = self._uses_entity_update_order()
        spawn_move_lock_frames = max(
            0, int(self.assumptions.enemy_spawn_move_lock_frames) - int(allocation_order)
        )
        spawn_attack_lock_frames = 0
        if apply_way != ApplyWay.NONE.value:
            spawn_attack_lock_frames = max(
                0, int(self.assumptions.enemy_spawn_attack_lock_frames) - int(allocation_order)
            )
        if spawn_move_lock_frames:
            self.used_assumptions.add("ENEMY_SPAWN_FRAME_NO_MOVE")
        if spawn_attack_lock_frames:
            self.used_assumptions.add("ENEMY_SPAWN_ACTION_LOCK")
        enemy = _Enemy(
            key=key,
            name=str(name),
            hp=hp,
            max_hp=hp,
            atk=float(D.mdef(attr.get("atk"), 0)),
            defense=float(D.mdef(attr.get("def"), 0)),
            res=float(D.mdef(attr.get("magicResistance"), 0)),
            move_speed=move_speed,
            attack_interval=attack_interval,
            block_count=int(D.mdef(attr.get("blockCnt"), 1)),
            mass_level=int(D.mdef(attr.get("massLevel"), 1)),
            range_radius=float(D.mdef(enemy_data.get("rangeRadius"), -1.0)),
            life_reduce=life_reduce,
            level_type=level_type,
            motion=motion,
            path=path,
            route=route,
            apply_way=apply_way,
            can_normal_attack=apply_way != ApplyWay.NONE.value,
            unblockable=key in UNBLOCKABLE_KEYS,
            wave_start_time=wave_start_time,
            attack_duration=attack_duration,
            attack_hit_time=attack_hit_time,
            wait_for_attack_event=(timing or {}).get("wait_for_attack_event") is True,
            projectile_key=projectile_key,
            move_cd=0.0,
            spawn_move_lock_frames=spawn_move_lock_frames,
            spawn_attack_lock_frames=spawn_attack_lock_frames,
            target_search_ticker=FramePeriodicTicker(
                period_frames=max(
                    1,
                    int(self.assumptions.enemy_target_search_period_frames),
                )
            ),
            position_row=start.row,
            position_col=start.col,
            remaining_path_distance=0.0,
            creation_index=self.creation_counter,
            first_update_frame=(self.frame_index + 1 if self._uses_entity_update_order() else 0),
            enemy_index=self.enemy_counter,
            elements=ElementController(
                max_value=2000.0 if level_type == "BOSS" else 1000.0,
                enemy_unit=True,
            ),
        )
        enemy.target_search_ticker.reset(self.frame_index)
        if (apply_way == ApplyWay.RANGED.value and projectile_key
                and (self.assumptions.enemy_projectile_muzzle
                     or self.assumptions.enemy_facing_transition)):
            enemy.graphic_profile = self.enemy_graphics.get(key)
            if enemy.graphic_profile is None:
                self.mechanic_diagnostics[f"ENEMY_GRAPHIC_PROFILE:{key}"] += 1
            else:
                sign = enemy.graphic_profile.initial_facing
                enemy.facing = FacingRuntime(target=sign, origin=float(sign))
        if (
            apply_way == ApplyWay.RANGED.value
            and self.assumptions.ranged_spawn_search_phase
        ):
            # Anchor the initial idle cycle to birth + period * k.
            # Consume the birth slot so grouped updates cannot tick it twice.
            enemy.target_search_ticker.tick(self.frame_index)
            enemy.target_search_ticker.next(self.frame_index)
            self.used_assumptions.add("RANGED_SPAWN_SEARCH_PHASE")
        self._initialize_enemy_buffs(enemy)
        if key in SHIELDGUARD_KEYS:
            enemy.taunt_level += 1
        enemy.remaining_path_distance = self._targeting_path_distance(enemy)
        return enemy

    def _initialize_enemy_buffs(self, enemy: _Enemy) -> None:
        """Attach prefab-declared listener Buffs at the enemy lifecycle boundary."""
        enemy.buffs.owner = enemy
        enemy.buffs.event_handler = self._dispatch_buff_event
        for definition in self.buff_catalog.listener_definitions(enemy.key):
            enemy.buffs.add(
                definition,
                source=enemy,
                attached_step=self.frame_index,
            )
        enemy.buffs.commit()

    def _make_operator(self, plan: OperatorPlan, deployed_cost: int) -> _Operator:
        char = self.characters[plan.char_id]
        module = (
            self.battle_equips.get(plan.module_id) if plan.module_id else None
        )
        attrs = D.char_attributes(
            char,
            plan.level,
            plan.elite,
            trust=plan.trust,
            potential_rank=plan.potential_rank,
            module=module,
            module_level=plan.module_level,
        )
        name = char.get("name", plan.char_id)
        hp = float(attrs.get("maxHp", 0))
        atk = float(attrs.get("atk", 0))
        attack_speed = float(attrs.get("attackSpeed", 100.0)) or 100.0
        base_attack_time = float(attrs.get("baseAttackTime", 1.6))
        interval = base_attack_time * 100.0 / attack_speed
        timing = self.attack_timing.get(plan.char_id, {})
        projectile_spec = self.operator_projectiles.get(plan.char_id) or {}
        projectile_key = str(projectile_spec.get("key") or "") or None
        profession = str(char.get("profession", ""))
        position = str(char.get("position", "")).upper()
        if plan.tile is None:
            raise ValueError("DEPLOY action requires a tile")
        self.creation_counter += 1
        self.deployment_counter += 1
        base_range_cells = self.range_cells(
            char, plan.tile, plan.facing, plan.elite
        )
        subprofession = str(char.get("subProfessionId", ""))
        op = _Operator(
            char_id=plan.char_id,
            name=str(name),
            tile=plan.tile,
            hp=hp,
            max_hp=hp,
            atk=atk,
            defense=float(attrs.get("def", 0)),
            res=float(attrs.get("magicResistance", 0)),
            block_count=int(attrs.get("blockCnt", 1)),
            block_scan_next_frame=(
                self.frame_index + 1
                if self.assumptions.operator_block_scan_from_deploy_frame
                else None
            ),
            cost=deployed_cost,
            attack_interval=interval,
            range_cells=set(base_range_cells),
            attack_duration=float(timing.get("attack_duration") or 0.0),
            attack_hit_time=float(timing.get("attack_hit_time") or 0.0),
            projectile_key=projectile_key,
            skill_attack_timing=dict(timing.get("skills") or {}),
            facing=plan.facing % 4,
            damage_type=_DAMAGE_TYPE_BY_PROFESSION.get(profession, "PHYSICAL"),
            multi_target=subprofession == "centurion",
            subprofession=subprofession,
            prioritize_blocked=position == "MELEE",
            potential_rank=plan.potential_rank,
            module_id=plan.module_id,
            module_level=plan.module_level,
            base_range_cells=set(base_range_cells),
            deployed_at=self.time,
            creation_index=self.creation_counter,
            first_update_frame=(self.frame_index + 1 if self._uses_entity_update_order() else 0),
            deployment_index=self.deployment_counter,
            base_cost=int(attrs.get("cost", 0)),
            deployed_cost=deployed_cost,
            respawn_time=float(attrs.get("respawnTime", 70.0)),
        )
        if plan.skill_id:
            self._attach_skill(op, plan.skill_id)
        self.initialize_operator_target_selector(
            op, first_search_frame=self.frame_index + 32
        )
        return op

    def _attach_skill(self, op: _Operator, skill_id: str) -> None:
        skill = self.skills.get(skill_id)
        if not skill:
            return
        levels = skill.get("levels", [])
        if not levels:
            return
        level = levels[-1]
        sp_data = level.get("spData", {})
        sp_type = str(sp_data.get("spType", ""))
        sp_cost = float(sp_data.get("spCost", 0) or 0)
        max_charge_time = max(int(sp_data.get("maxChargeTime", 1) or 1), 1)
        op.skill = level
        op.sp_cost = sp_cost
        op.max_charge_time = max_charge_time
        op.sp_max = sp_cost * max_charge_time
        op.sp = min(float(sp_data.get("initSp", 0) or 0), op.sp_max)
        op.sp_increment = float(sp_data.get("increment", 1.0) or 1.0)
        op.sp_recovery_remaining_raw = CLIENT_FP_SCALE
        op.sp_type = sp_type
        op.auto = True  # v0: manual skills also auto-cast when full
        op.skill_type = str(level.get("skillType", ""))
        op.skill_duration_type = str(level.get("durationType", "NONE"))
        op.skill_blackboard = self._blackboard(level)
        prefab_id = str(level.get("prefabId", ""))
        op.skill_prefab_id = prefab_id
        op.behavior_template_key = resolve_behavior_template(
            prefab_id, self.behavior.templates
        )
        if op.behavior_template_key:
            supported_events = {
                "ON_BUFF_START",
                "ON_BUFF_FINISH",
                "ON_SKILL_FINISH",
                "ON_CALCULATE_DAMAGE",
            }
            if prefab_id == "skchr_huang_3":
                supported_events.add("ON_BUFF_TRIGGER")
            template = self.behavior.templates[op.behavior_template_key]
            for event in template.get("eventToActions", {}):
                if event not in supported_events:
                    key = f"EVENT:{event}"
                    self.behavior_diagnostics.unsupported_nodes[key] += 1
                    if self.behavior.strict:
                        raise BehaviorError(
                            f"unsupported behavior event {event!r} in "
                            f"{op.behavior_template_key!r}"
                        )
        if sp_type in ("PASSIVE", "NO_CHARGE", "INCREASE_WHEN_NONE"):
            self._cast_skill(op)

    # ------------------------------------------------------------- loop
    def step(self) -> None:
        self.time += self.dt
        self.frame_index += 1

        for unit in (*self.operators, *self.enemies):
            if not unit.dead:
                unit.statuses.tick(self.dt)
                self._tick_element_burst(unit)
                if unit.elements.tick(self.dt) is not None:
                    unit.element_burst = None

        # Losing block capability immediately releases held enemies.
        for op in self.operators:
            if op.dead or op.statuses.can_block:
                continue
            for enemy in list(op.blocked):
                self._release_enemy_block(enemy)

        self.dp_timer += self.dt
        while (
            self.dp_timer + 1e-9 >= self.cost_increase_time
            and self.dp < self.max_cost
        ):
            self.dp_timer = max(0.0, self.dp_timer - self.cost_increase_time)
            self.dp += 1

        if not self._uses_entity_update_order():
            self._spawn_due_enemies()

        # GlobalAuraAbility registration is a Battle-level phase.  It is
        # deliberately separate from attribute reads so a target keeps its
        # own listener phase when a herald appears later.
        self._sync_herald_auras()

        while self.plan_index < len(self.plan):
            plan = self.plan[self.plan_index]
            if plan.time > self.time + 1e-9:
                break
            status = self._execute_plan_action(plan)
            if status == "wait":
                break
            self.plan_index += 1

        if self._uses_entity_update_order():
            self.used_assumptions.add("ENTITY_UPDATE_ORDER_BY_ALLOCATION")
            self._process_projectiles()
            # Snapshot once; killed receivers skip their pending dispatch.
            units = sorted(
                (*self.enemies, *self.operators),
                key=lambda unit: (-unit.update_priority, unit.creation_index),
            )
            for unit in units:
                if unit.dead or self.frame_index < unit.first_update_frame:
                    continue
                if isinstance(unit, _Operator):
                    self._tick_operator(unit)
                else:
                    self._tick_enemy(unit)
            # Ordinary native spawn resumes after the central Fixed dispatch.
            self._spawn_due_enemies()
        else:
            for op in self.operators:
                if not op.dead:
                    self._update_skill(op)
            self._process_projectiles()
            for op in self.operators:
                self._advance_operator_action(op)
            if self.assumptions.operator_target_selector_before_enemy_movement:
                self.used_assumptions.add("OPERATOR_SELECTOR_PRE_MOVEMENT_SEARCH")
                for op in self.operators:
                    if not op.dead:
                        self._advance_operator_target_selector(op)
            for enemy in list(self.enemies):
                self._tick_enemy(enemy)
            self._update_blocking()
            if not self.assumptions.operator_target_selector_before_enemy_movement:
                for op in self.operators:
                    if not op.dead:
                        self._advance_operator_target_selector(op)
            for op in self.operators:
                self._start_operator_action(op)

        # Hammer callbacks read this frame's completed movement. Their fixed
        # deadline stays unchanged; only the phase within that frame differs.
        self._process_due_hammer_attacks()

        # Highland echo is submitted after ordinary enemy movement.  Its
        # damage and SLUGGISH write therefore affect the next fixed frame.
        self._process_scheduled_highland_effects()

        # Buff timers advance after this frame's movement and attacks.  Any
        # derived state created here is committed for the next fixed frame.
        self._sync_herald_auras()
        self._tick_enemy_buffs()
        self._cleanup()

    def _uses_entity_update_order(self) -> bool:
        # The pre-movement switch explicitly selects the historical grouped
        # comparison instead of changing one receiver in native ordering.
        return (
            self.assumptions.entity_update_order_by_allocation
            and not self.assumptions.operator_target_selector_before_enemy_movement
        )

    def _spawn_due_enemies(self) -> None:
        while (
            self.spawn_index < len(self.spawn_events)
            and self._spawn_event_is_due(self.spawn_events[self.spawn_index])
        ):
            ev = self.spawn_events[self.spawn_index]
            self.spawn_index += 1
            if ev.route_index >= len(self.paths):
                continue
            self.enemies.append(
                self._make_enemy(
                    ev.enemy_key,
                    ev.route_index,
                    wave_start_time=ev.wave_start_time,
                )
            )
            self.enemies_spawned += 1

    def _tick_operator(self, op: _Operator) -> None:
        self._update_skill(op)
        self._advance_operator_action(op)
        if op.dead:
            return
        self._advance_operator_target_selector(op)
        self._start_operator_action(op)
        # Character.OnTick advances Unit before its blockee scan. Older
        # enemies already ticked; newer ones can consume the relation now.
        self._update_operator_blocking(op)

    def _advance_operator_action(self, op: _Operator) -> None:
        if op.dead:
            return
        if self._skill_starting(op):
            return
        if op.skill_prefab_id == "skchr_huang_3" and op.behavior_active:
            return
        if op.pending_attack is not None and not op.statuses.can_attack:
            op.pending_attack = None
        if (
            op.pending_attack is not None
            and op.pending_attack.kind != "HAMMER"
            and self.frame_index >= op.pending_attack.execute_frame
        ):
            self._land_operator_attack(op)
        op.attack_cd -= self.dt

    def _tick_enemy(self, enemy: _Enemy) -> None:
        if enemy.dead or enemy.leaked:
            return
        spawn_move_locked = enemy.spawn_move_lock_frames > 0
        if spawn_move_locked:
            enemy.spawn_move_lock_frames -= 1
        spawn_attack_locked = enemy.spawn_attack_lock_frames > 0
        if spawn_attack_locked:
            enemy.spawn_attack_lock_frames -= 1
        self._tick_enemy_attack_cooldown(enemy)
        self._tick_enemy_wait(enemy)
        if (
            enemy.can_normal_attack
            and enemy.atk > 0
            and enemy.apply_way == ApplyWay.MELEE.value
        ):
            # Melee enemies acquire their blocker directly; there is no
            # Selector/Search cycle between blocking and attack startup.
            target = enemy.blocked_by
            if target is not None and target.dead:
                target = None
            enemy_search_cycle = target is not None
        elif enemy.can_normal_attack and enemy.atk > 0:
            target, enemy_search_cycle = (
                self._advance_enemy_operator_target_search(enemy)
            )
        else:
            enemy_search_cycle = False
            target = None

        # Consume the latest relation when this enemy reaches its update.
        if self._apply_block_shift(enemy):
            if enemy.dead:
                return

        if enemy.disappeared:
            enemy.attack_target = None
            enemy.attack_hit_at = -1.0
            enemy.attack_hit_frame = None
            enemy.attack_will_hit = True
            self._move_enemy(enemy)
            if enemy.disappeared and not enemy.portal_targetable:
                return

        if enemy.attack_target is not None and not enemy.statuses.can_attack:
            enemy.attack_target = None
            enemy.attack_hit_at = -1.0
            enemy.attack_hit_frame = None
            enemy.attack_will_hit = True

        # Land a pending hit once its windup has elapsed.
        if enemy.attack_target is not None and (
            self.frame_index >= enemy.attack_hit_frame
            if enemy.attack_hit_frame is not None
            else self.time >= enemy.attack_hit_at
        ):
            if not enemy.attack_target.dead:
                self._fire_enemy_attack(
                    enemy,
                    enemy.attack_target,
                    will_hit=enemy.attack_will_hit,
                )
            enemy.attack_target = None
            enemy.attack_hit_frame = None
            enemy.attack_will_hit = True

        if not spawn_attack_locked and not enemy.disappeared:
            # A portal-exit hitbox can be targetable while DisappearState
            # still owns the unit. Do not consume an attack interval until
            # the enemy has resumed its visible action state.
            # Start a new attack (windup) when the interval is ready.
            if (
                target is not None
                and enemy_search_cycle
                and enemy.can_normal_attack
                and enemy.atk > 0
                and enemy.statuses.can_attack
                and self._enemy_attack_ready(enemy)
                and enemy.attack_target is None
            ):
                starts_from_block = (
                    enemy.apply_way == ApplyWay.MELEE.value
                    and enemy.blocked_by is target
                )
                if starts_from_block:
                    self.used_assumptions.add(
                        "MELEE_BLOCK_STARTS_ATTACK"
                    )
                    self.used_assumptions.add(
                        "MELEE_BLOCK_ATTACK_HIT_AFTER_WINDUP"
                    )
                self._start_enemy_attack(
                    enemy,
                    target,
                    hit_after_windup=starts_from_block,
                )

        # A live block relation prevents route movement; its offset was
        # applied above, including relations from earlier receivers.
        if enemy.blocked_by and not enemy.blocked_by.dead:
            return
        if enemy.blocked_by is not None:
            self._release_enemy_block(enemy)
        if spawn_move_locked:
            return
        if enemy.attack_move_resume_frame is not None:
            if self.frame_index < enemy.attack_move_resume_frame:
                return
            enemy.attack_move_resume_frame = None
        if enemy.pause_until > self.time:
            return
        if not enemy.statuses.can_move:
            return
        self._request_enemy_route_facing(enemy)
        self._move_enemy(enemy)

    def _start_operator_action(self, op: _Operator) -> None:
        if op.dead:
            return
        if self._skill_starting(op):
            return
        if op.skill_prefab_id == "skchr_huang_3" and op.behavior_active:
            return
        if not op.statuses.can_attack:
            return
        if op.attack_cd > 0 or op.pending_attack is not None:
            return

        if op.attack_chain_active:
            if not self._cached_operator_target_is_valid(op):
                if op.char_id != "char_1051_headb2":
                    self._finish_operator_attack_chain(op)
                    return
                # Headb2's attack boundary consumes Search/Next even when
                # no replacement exists (native f677/f873 reload to 3).
                self._force_refresh_operator_target(op)
                if not self._cached_operator_target_is_valid(op):
                    self._finish_operator_attack_chain(op)
                    return
                op.attack_chain_active = False
                op.attack_first_cast = False
                op.selector_resume_frame = None
                first_attack = True
            else:
                # Continuous attacks use the native force-search path.
                # Search/Next happens after Tick in this fixed frame.
                self._force_refresh_operator_target(op)
                first_attack = False
        else:
            if self._current_operator_target(op) is None:
                return
            first_attack = True

        if self._operator_attack(op, first_attack=first_attack):
            op.attack_cd = self._effective_attack_interval(op)

    def _spawn_event_is_due(self, event: SpawnEvent) -> bool:
        if event.logic_frame is not None:
            return event.logic_frame <= self.frame_index
        return event.time <= self.time

    def run(self, max_time: float = 600.0) -> Result:
        while self.time < max_time:
            self.step()
            if self.life_points <= 0:
                self.end_reason = "life_points_depleted"
                break
            if self.spawn_index >= len(self.spawn_events) and not self.enemies:
                self.end_reason = "all_clear"
                break
        if not self.end_reason:
            self.end_reason = "timeout"
        return self.result()

    def result(self) -> Result:
        return Result(
            win=self.end_reason == "all_clear",
            time=self.time,
            life_points=self.life_points,
            max_life_points=self.max_life_points,
            enemies_spawned=self.enemies_spawned,
            enemies_killed=self.enemies_killed,
            enemies_leaked=self.enemies_leaked,
            operators_deployed=sum(1 for o in self.operators if not o.dead),
            reason=self.end_reason,
            behavior_warnings=self.behavior_diagnostics.warning_count,
            unsupported_behavior_nodes=dict(
                self.behavior_diagnostics.unsupported_nodes
            ),
            mechanic_warnings=sum(self.mechanic_diagnostics.values()),
            unsupported_mechanics=dict(self.mechanic_diagnostics),
            seed=self.seed,
            random_draws=self.rng.draw_count,
            assumptions=[
                ASSUMPTION_NOTES[key] for key in sorted(self.used_assumptions)
            ],
            spawn_timing=self.spawn_timing,
            enemy_attack_timing=self.enemy_attack_timing,
            operator_metrics=[
                {
                    "char_id": op.char_id,
                    "potential_rank": op.potential_rank,
                    "module_id": op.module_id,
                    "module_level": op.module_level,
                    "deployed_at": op.deployed_at,
                    "death_time": op.death_time,
                    "max_hp": op.max_hp,
                    "atk": op.atk,
                    "def": op.defense,
                    "damage_dealt": op.damage_dealt,
                    "kills": op.kills,
                    "skill_casts": op.skill_cast_count,
                    "alive": not op.dead,
                }
                for op in self.operators
            ],
        )

    # ------------------------------------------------------------- skills
    @staticmethod
    def _skill_starting(op: _Operator) -> bool:
        return op.skill_startup_until >= 0.0

    def _update_skill(self, op: _Operator) -> None:
        if op.skill is None:
            return
        recovery_was_stopped = (
            op.skill_active
            or self._skill_starting(op)
            or op.statuses.has(Status.SP_BLOCKED)
        )
        if self._skill_starting(op):
            if self.time + 1e-9 < op.skill_startup_until:
                return
            op.skill_startup_until = -1.0
            self._activate_skill_effect(op)
        if (
            op.skill_prefab_id == "skchr_huang_3"
            and op.behavior_active
            and op.active_until > self.time
        ):
            while (
                op.behavior_trigger_count < 9
                and self.time + 1e-9 >= op.behavior_next_trigger_at
            ):
                op.behavior_trigger_count += 1
                ratio = op.behavior_trigger_count / 9.0
                self._dispatch_skill_event(
                    op, "ON_BUFF_TRIGGER", trigger_ratio=ratio
                )
                if op.behavior_trigger_count <= 8:
                    self._huang_s3_cut(op)
                else:
                    self._dispatch_skill_event(op, "ON_SKILL_FINISH")
                    op.behavior_finish_dispatched = True
                op.behavior_next_trigger_at += 1.0
        if op.skill_active and op.active_until > 0 and self.time >= op.active_until:
            self._finish_skill(op, "DURATION_END")
        if (
            op.sp_type == "INCREASE_WITH_TIME"
            and not op.skill_active
            and not self._skill_starting(op)
            and not op.statuses.has(Status.SP_BLOCKED)
        ):
            self._recover_time_sp(
                op, recovery_was_stopped=recovery_was_stopped
            )
        if (
            self._skill_ready(op)
            and not op.skill_active
            and not self._skill_starting(op)
            and op.auto
            and op.statuses.can_use_skill
        ):
            self._cast_skill(op)

    def _recover_time_sp(
        self, op: _Operator, *, recovery_was_stopped: bool
    ) -> None:
        # The observed S2 profile shares SpController's integer-count path.
        # Other rates/characters keep the explicit historical model until
        # their timer period and initial phase have been checked.
        if not (
            self.assumptions.headb2_periodic_sp_recovery
            and op.skill_prefab_id == "skchr_headb2_2"
            and math.isclose(self.dt, 1 / CLIENT_LOGIC_RATE, abs_tol=1e-12)
            and op.sp_increment == 1.0
        ):
            self._gain_sp(op, op.sp_increment * self.dt)
            return
        self.used_assumptions.add("HEADB2_PERIODIC_SP_RECOVERY")
        # SpController sees the old lock before the skill's expiry transition.
        # Preserve the timer across locks, full SP, and spending the SP cost.
        if recovery_was_stopped or op.sp + 1e-9 >= op.sp_max:
            return
        op.sp_recovery_remaining_raw -= int(
            CLIENT_FIXED_TIMER_STEP * CLIENT_FP_SCALE
        )
        remaining = op.sp_recovery_remaining_raw
        if remaining <= CLIENT_FP_EPSILON_RAW:
            # UpdateMultiple clamps negative overshoot counts to zero;
            # a small positive residual inside LessEqual's tolerance still
            # produces one recovery. NextMultiple retains the overshoot.
            count = max(0, -remaining // CLIENT_FP_SCALE) + 1
            op.sp_recovery_remaining_raw += count * CLIENT_FP_SCALE
            self._gain_sp(op, float(count))

    def _cast_skill(self, op: _Operator) -> bool:
        level = op.skill
        if level is None or op.skill_active or self._skill_starting(op):
            return False
        no_charge = op.sp_type in ("PASSIVE", "NO_CHARGE", "INCREASE_WHEN_NONE")
        if not no_charge and not self._skill_ready(op):
            return False
        if not op.statuses.can_use_skill and not no_charge:
            return False
        if not no_charge:
            op.sp = max(0.0, op.sp - self._skill_cost(op))
        self._register_ability_use(op)
        op.skill_end_reason = None
        op.skill_cast_count += 1
        if op.skill_prefab_id == "skchr_headb2_2":
            op.pending_attack = None
        startup_duration = self._skill_startup_duration(op)
        if startup_duration > 0.0:
            op.skill_startup_until = self.time + startup_duration
            return True
        return self._activate_skill_effect(op)

    @staticmethod
    def _skill_startup_duration(op: _Operator) -> float:
        timing = op.skill_attack_timing.get(op.skill_prefab_id) or {}
        return max(0.0, float(timing.get("startup_duration") or 0.0))

    def _activate_skill_effect(self, op: _Operator) -> bool:
        """Apply a skill after any cast animation has completed."""
        level = op.skill
        if level is None or op.skill_active:
            return False
        if op.skill_prefab_id == "skchr_headb2_2":
            self._headb2_s2_reset_attack(op)
        op.skill_active = True
        blackboard = self._blackboard(level)
        duration = float(level.get("duration", 0.0) or 0.0)
        if op.skill_prefab_id == "skchr_headb2_2":
            blackboard, duration = self._headb2_s2_cast_data(
                op, blackboard, duration
            )
        op.skill_blackboard = dict(blackboard)
        op.behavior_blackboard = dict(blackboard)
        op.behavior_active = op.behavior_template_key is not None
        op.behavior_trigger_count = 0
        op.behavior_next_trigger_at = -1.0
        op.behavior_finish_dispatched = False
        self._dispatch_skill_event(op, "ON_BUFF_START")
        range_id = level.get("rangeId")
        if range_id:
            op.range_cells = self.range_cells_from_id(
                str(range_id), op.tile, op.facing
            )
            if op.target_selector is not None:
                op.target_selector.mark_dirty()
        if duration == -1.0:
            self._apply_stat_buff(op, blackboard, permanent=True)
            op.skill_active = False
            op.skill_end_reason = "PERMANENT"
            op.skill = None
            return True
        if "cost" in blackboard:
            self.dp = min(self.max_cost, self.dp + int(round(float(blackboard["cost"]))))
        if "atk_scale" in blackboard:
            op.next_attack_scale = float(blackboard["atk_scale"])
        if op.skill_duration_type == "AMMO":
            capacity = self._infer_ammo_capacity(blackboard)
            if capacity is None:
                key = f"AMMO_CAPACITY:{op.skill_prefab_id or '<unknown>'}"
                self.mechanic_diagnostics[key] += 1
                self._finish_skill(op, "AMMO_UNCONFIGURED")
                return False
            self.configure_skill_ammo(op, capacity)
            self._apply_stat_buff(op, blackboard, timed=True)
        elif duration > 0:
            if op.skill_prefab_id == "skchr_huang_3":
                op.active_until = self.time + duration
                op.behavior_next_trigger_at = self.time + 1.0
            else:
                self._apply_stat_buff(op, blackboard, timed=True, duration=duration)
        else:
            self._finish_skill(op, "INSTANT")
        return True

    def _headb2_s2_reset_attack(self, op: _Operator) -> None:
        """Restart S2 targeting after its cast animation finishes."""
        op.pending_attack = None
        op.attack_cd = 0.0
        self.initialize_operator_target_selector(
            op, first_search_frame=self.frame_index + 1
        )

    def _headb2_s2_cast_data(
        self,
        op: _Operator,
        blackboard: dict[str, Any],
        duration: float,
    ) -> tuple[dict[str, Any], float]:
        """Resolve S2's first/second cast and the self part of talent 2."""
        out = dict(blackboard)
        talent = 0.18 if op.potential_rank >= 2 else 0.14
        self_bonus = talent * 2.0
        if op.skill_cast_count >= 2:
            out["atk"] = float(out.get("headb2_s_2[second].atk", 0.0))
            out["def"] = float(out.get("headb2_s_2[second].def", 0.0))
            duration = -1.0
        out["atk"] = float(out.get("atk", 0.0)) + self_bonus
        out["def"] = float(out.get("def", 0.0)) + self_bonus
        return out, duration

    def _skill_cost(self, op: _Operator) -> float:
        return op.sp_cost if op.sp_cost > 0 else op.sp_max

    def _skill_ready(self, op: _Operator) -> bool:
        cost = self._skill_cost(op)
        return cost <= 0 or op.sp + 1e-9 >= cost

    def _gain_sp(self, op: _Operator, amount: float) -> float:
        if (
            amount <= 0
            or op.skill_active
            or self._skill_starting(op)
            or op.statuses.has(Status.SP_BLOCKED)
        ):
            return 0.0
        return self._store_sp_gain(op, amount)

    @staticmethod
    def _store_sp_gain(op: _Operator, amount: float) -> float:
        if amount <= 0 or op.sp + 1e-9 >= op.sp_max:
            return 0.0
        before = op.sp
        gained = op.sp + amount
        if gained > op.sp_max:
            # The integer SP bar is capped, but its sub-point remainder survives.
            op.sp = op.sp_max + (gained - math.floor(gained))
        else:
            op.sp = gained
        return op.sp - before

    @staticmethod
    def _infer_ammo_capacity(blackboard: dict[str, Any]) -> float | None:
        candidates = [
            (key, value)
            for key, value in blackboard.items()
            if key == "ammo" or key.endswith("trigger_time")
        ]
        candidates.sort(
            key=lambda item: (
                item[0] != "attack@trigger_time",
                not item[0].startswith("attack@"),
                item[0],
            )
        )
        for _, value in candidates:
            try:
                capacity = float(value)
            except (TypeError, ValueError):
                continue
            if capacity > 0:
                return capacity
        return None

    @staticmethod
    def configure_skill_ammo(
        op: _Operator, capacity: float, *, cost_per_attack: float = 1.0
    ) -> None:
        op.ammo_capacity = max(0.0, float(capacity))
        op.ammo = op.ammo_capacity
        op.ammo_cost_per_attack = max(0.0, float(cost_per_attack))

    def consume_skill_ammo(
        self, op: _Operator, amount: float | None = None
    ) -> float | None:
        if not op.skill_active or op.skill_duration_type != "AMMO":
            return op.ammo
        cost = op.ammo_cost_per_attack if amount is None else max(0.0, amount)
        if op.ammo is not None:
            op.ammo -= cost
        return op.ammo

    def interrupt_skill(self, op: _Operator, reason: str = "INTERRUPTED") -> bool:
        if not op.skill_active:
            return False
        self._finish_skill(op, reason)
        return True

    def _finish_skill(self, op: _Operator, reason: str) -> None:
        if not op.skill_active:
            return
        if not op.behavior_finish_dispatched:
            self._dispatch_skill_event(op, "ON_SKILL_FINISH")
        if op.timed_atk_scale != 1.0:
            op.atk_scale /= op.timed_atk_scale
        if op.timed_def_scale != 1.0:
            op.def_scale /= op.timed_def_scale
        op.timed_atk_scale = 1.0
        op.timed_def_scale = 1.0
        op.active_until = -1.0
        op.range_cells = set(op.base_range_cells)
        if op.target_selector is not None:
            op.target_selector.mark_dirty()
        self._dispatch_skill_event(op, "ON_BUFF_FINISH")
        op.behavior_active = False
        op.skill_active = False
        op.skill_end_reason = reason
        op.ammo = None
        op.ammo_capacity = None
        if op.skill_prefab_id == "skchr_headb2_2":
            # Native S2 expiry replaces the mode/selector (f1049). Do not let
            # the old skill attack chain later consume another Search/Next.
            op.pending_attack = None
            self.initialize_operator_target_selector(
                op, first_search_frame=self.frame_index + 1
            )

    @staticmethod
    def _blackboard(level: dict[str, Any]) -> dict[str, Any]:
        return {
            str(item.get("key")): item.get("value")
            for item in level.get("blackboard", [])
            if item.get("key") is not None
        }

    def _dispatch_skill_event(
        self,
        op: _Operator,
        event: str,
        *,
        target: _Enemy | _Operator | None = None,
        attack_scale: float = 1.0,
        damage_scale: float = 1.0,
        trigger_ratio: float = 0.0,
    ) -> BehaviorContext | None:
        template_key = getattr(op, "behavior_template_key", None)
        if not template_key or not getattr(op, "behavior_active", False):
            return None
        context = BehaviorContext(
            source=op,
            target=target if target is not None else op,
            buff_owner=op,
            buff_source=op,
            blackboard=getattr(op, "behavior_blackboard", {}),
            ability_blackboard=getattr(op, "skill_blackboard", {}),
            battle=self,
            diagnostics=self.behavior_diagnostics,
            attack_scale=attack_scale,
            damage_scale=damage_scale,
            trigger_ratio=trigger_ratio,
        )
        self.behavior.dispatch(template_key, event, context)
        return context

    def _apply_stat_buff(
        self,
        op: _Operator,
        blackboard: dict[str, Any],
        *,
        permanent: bool = False,
        timed: bool = False,
        duration: float | None = None,
    ) -> None:
        if "atk" in blackboard:
            mult = 1.0 + float(blackboard["atk"])
            op.atk_scale *= mult
            if timed:
                op.timed_atk_scale = mult
        if "def" in blackboard:
            mult = 1.0 + float(blackboard["def"])
            op.def_scale *= mult
            if timed:
                op.timed_def_scale = mult
        if timed and duration is not None:
            if op.skill_prefab_id == "skchr_headb2_2" and math.isclose(
                self.dt, 1 / CLIENT_LOGIC_RATE, rel_tol=0.0, abs_tol=1e-12
            ):
                # Native S2's FP timer still has 1/65536 seconds remaining
                # after 480 ticks. Convert its deadline to the simulation
                # clock so expiry and selector replacement occur on tick 481.
                duration *= self.dt / CLIENT_FIXED_TIMER_STEP
            op.active_until = self.time + duration

    # ------------------------------------------------------------- combat
    def _huang_s3_cut(self, op: _Operator) -> None:
        """Apply one of the eight once-per-second S3 cutting hits."""
        amount = op.atk * op.atk_scale
        for enemy in list(self.enemies):
            if (
                not self._enemy_is_attackable(enemy)
                or enemy.tile() not in op.range_cells
            ):
                continue
            if enemy.motion.upper() in {"FLY", "FLYING"}:
                continue
            self.behavior_deal_damage(op, enemy, amount, "PHYSICAL")

    def _operator_attack(
        self, op: _Operator, *, first_attack: bool | None = None
    ) -> bool:
        if first_attack is None:
            first_attack = not op.attack_chain_active
        if op.subprofession == "hammer":
            started = self._hammer_attack(op)
            if started:
                self._mark_operator_attack_started(op, first_attack)
            return started
        if op.multi_target and op.blocked:
            primary = self._current_operator_target(op)
            targets = [
                enemy
                for enemy in op.blocked
                if self._validate_operator_target(op, enemy)
            ]
            if primary is None or not any(enemy is primary for enemy in targets):
                op.attack_cd = 0.0
                return False
            self._finish_depleted_ammo_before_attack(op)
            self._register_ability_use(op)
            execute_frame, execute_at = self._operator_attack_deadline(op)
            op.pending_attack = _PendingOperatorAttack(
                execute_at=execute_at,
                execute_frame=execute_frame,
                kind="MULTI",
                primary=primary,
                targets=targets,
                attack_scale=op.next_attack_scale,
                source_atk_scale=op.atk_scale,
                sp_recovery_locked=op.skill_active,
                will_hit=self._attack_hits(op, op.damage_type),
            )
            op.next_attack_scale = 1.0
            self._record_targeting_diagnostic(
                op,
                pending_created=True,
                pending_execute_at=op.pending_attack.execute_at,
            )
            self._land_zero_delay_operator_attack(op)
            self._mark_operator_attack_started(op, first_attack)
            return True

        target = self._current_operator_target(op)
        if target is None:
            op.attack_cd = 0.0
            return False
        self._finish_depleted_ammo_before_attack(op)
        self._register_ability_use(op)
        execute_frame, execute_at = self._operator_attack_deadline(op)
        op.pending_attack = _PendingOperatorAttack(
            execute_at=execute_at,
            execute_frame=execute_frame,
            kind="SINGLE",
            primary=target,
            targets=[target],
            attack_scale=op.next_attack_scale,
            source_atk_scale=op.atk_scale,
            sp_recovery_locked=op.skill_active,
            will_hit=self._attack_hits(op, op.damage_type),
        )
        op.next_attack_scale = 1.0
        self._record_targeting_diagnostic(
            op,
            pending_created=True,
            pending_execute_at=op.pending_attack.execute_at,
        )
        self._land_zero_delay_operator_attack(op)
        self._mark_operator_attack_started(op, first_attack)
        return True

    def _mark_operator_attack_started(
        self, op: _Operator, first_attack: bool
    ) -> None:
        op.attack_chain_active = True
        op.attack_first_cast = bool(first_attack)
        self._record_targeting_diagnostic(
            op,
            first_attack=bool(first_attack),
            attack_chain_active=True,
        )

    def _hammer_attack(self, op: _Operator) -> bool:
        target = self._current_operator_target(op)
        if target is None:
            op.attack_cd = 0.0
            return False
        self._finish_depleted_ammo_before_attack(op)
        self._register_ability_use(op)
        execute_frame, execute_at = self._operator_attack_deadline(op)
        op.pending_attack = _PendingOperatorAttack(
            execute_at=execute_at,
            execute_frame=execute_frame,
            kind="HAMMER",
            primary=target,
            targets=[target],
            attack_scale=op.next_attack_scale,
            source_atk_scale=op.atk_scale,
            sp_recovery_locked=op.skill_active,
            will_hit=self._attack_hits(op, op.damage_type),
        )
        op.next_attack_scale = 1.0
        self._record_targeting_diagnostic(
            op,
            pending_created=True,
            pending_execute_at=op.pending_attack.execute_at,
        )
        self._land_zero_delay_operator_attack(op)
        return True

    def _operator_attack_hit_delay(self, op: _Operator) -> float:
        timing = None
        if op.skill_active or op.skill_end_reason == "PERMANENT":
            timing = op.skill_attack_timing.get(op.skill_prefab_id)
        duration = float((timing or {}).get("attack_duration") or op.attack_duration)
        hit_time = float((timing or {}).get("attack_hit_time") or op.attack_hit_time)
        if duration <= 0.0 or hit_time <= 0.0:
            return 0.0
        return hit_time * self._effective_attack_interval(op) / duration

    def _operator_attack_deadline(self, op: _Operator) -> tuple[int, float]:
        delay = self._operator_attack_hit_delay(op)
        if delay <= 0.0:
            return self.frame_index, self.time
        # Candidate Spine event phase: resolve on the fixed frame after the
        # positive windup, using frame indices to avoid float-time drift.
        frame_count = delay / self.dt
        nearest_frame = round(frame_count)
        if math.isclose(frame_count, nearest_frame, rel_tol=0.0, abs_tol=1e-5):
            frame_count = float(nearest_frame)
        elapsed_frames = math.floor(frame_count) + 1
        return self.frame_index + elapsed_frames, self.time + elapsed_frames * self.dt

    def _land_zero_delay_operator_attack(self, op: _Operator) -> None:
        if (
            op.pending_attack is not None
            and op.pending_attack.kind != "HAMMER"
            and op.pending_attack.execute_frame <= self.frame_index
        ):
            self._land_operator_attack(op)

    def _process_due_hammer_attacks(self) -> None:
        """Resolve due hammer hits after unit movement, in this fixed frame."""
        for op in sorted(
            self.operators,
            key=lambda unit: (-unit.update_priority, unit.creation_index),
        ):
            attack = op.pending_attack
            if op.dead or attack is None or attack.kind != "HAMMER":
                continue
            if not op.statuses.can_attack:
                op.pending_attack = None
                continue
            if self._skill_starting(op) or attack.execute_frame > self.frame_index:
                continue
            self.used_assumptions.add("HEADB2_HAMMER_POST_MOVEMENT_HIT")
            self._land_operator_attack(op)

    def _land_operator_attack(self, op: _Operator) -> None:
        attack = op.pending_attack
        op.pending_attack = None
        if attack is None:
            return
        self._record_targeting_diagnostic(
            op,
            hit_frame=int(self.frame_index),
        )
        primary_valid = self._enemy_is_attackable(attack.primary)
        if attack.will_hit and primary_valid:
            if attack.kind == "HAMMER":
                self._land_hammer_attack(op, attack)
            elif self._projectile_speed(op.projectile_key) is not None:
                self._emit_operator_projectiles(op, attack)
            elif attack.kind == "MULTI":
                first = True
                for enemy in attack.targets:
                    if not self._enemy_is_attackable(enemy):
                        continue
                    scale = attack.attack_scale if first else 1.0
                    self._deal_damage_to_enemy(
                        op,
                        enemy,
                        scale,
                        source_atk_scale=attack.source_atk_scale,
                    )
                    first = False
            else:
                self._deal_damage_to_enemy(
                    op,
                    attack.primary,
                    attack.attack_scale,
                    source_atk_scale=attack.source_atk_scale,
                )
        elif primary_valid and self._projectile_speed(op.projectile_key) is not None:
            # A miss is still a launched projectile; only damage is suppressed
            # when it reaches the target.
            self._emit_operator_projectiles(op, attack)
        self._sp_on_attack(
            op, sp_recovery_locked=attack.sp_recovery_locked
        )
        self.consume_skill_ammo(op)

    def _projectile_speed(self, key: str | None) -> float | None:
        if not key:
            return None
        speed = self.projectile_speeds.get(key)
        if speed is None or speed <= 0.0:
            self.mechanic_diagnostics[f"PROJECTILE_SPEED:{key}"] += 1
            return None
        return speed

    def _new_projectile(
        self,
        *,
        key: str,
        side: str,
        source: _Enemy | _Operator,
        target: _Enemy | _Operator,
        operator_attack: _PendingOperatorAttack | None = None,
        will_hit: bool = True,
        cached_atk: float | None = None,
    ) -> None:
        speed = self._projectile_speed(key)
        if speed is None:
            return
        if isinstance(source, _Operator):
            source_row, source_col = map(float, source.tile)
        else:
            source_row, source_col = source.position()
        source_z = 0.0
        muzzle_position = None
        if isinstance(source, _Enemy) and self.assumptions.enemy_projectile_muzzle:
            muzzle_position = self.enemy_muzzle_position_v3(source)
            if muzzle_position is not None:
                source_row, source_col, source_z = muzzle_position
                self.used_assumptions.add("ENEMY_PROJECTILE_MUZZLE")
            elif (source.graphic_profile is not None
                  and (self.enemy_projectiles.get(source.key) or {}).get("mountPointType") == 2):
                self.mechanic_diagnostics[f"ENEMY_MUZZLE_OFFSET:{source.key}"] += 1
        profile = self.projectile_motion_profiles.get(key) if side == "ENEMY" else None
        if profile is not None:
            self.used_assumptions.add("PROJECTILE_MOTION_PROFILE")
            if profile.target_mount == "hit":
                self.used_assumptions.add("PROJECTILE_TARGET_HIT")
                if self._unit_hit_position(target) is None:
                    target_key = target.char_id if isinstance(target, _Operator) else target.key
                    self.mechanic_diagnostics[f"PROJECTILE_TARGET_HIT:{target_key}"] += 1
        target_row, target_col, target_z = self._projectile_target_position(target, profile)
        initial_distance = math.hypot(target_row - source_row, target_col - source_col)
        if profile is not None and profile.distance_dimensions == 3:
            initial_distance = math.sqrt(initial_distance ** 2 + (target_z - source_z) ** 2)
        self.projectile_counter += 1
        if muzzle_position is None:
            self.used_assumptions.add("PROJECTILE_SOURCE_CENTER")
        self.projectiles.append(
            _Projectile(
                projectile_id=self.projectile_counter,
                key=key,
                speed=speed,
                side=side,
                source=source,
                target=target,
                position_row=source_row,
                position_col=source_col,
                position_z=source_z,
                motion_profile=profile,
                launched_at=self.time,
                operator_attack=operator_attack,
                will_hit=will_hit,
                cached_atk=cached_atk,
                parabolic="mortar" in key.lower(),
                initial_distance=initial_distance,
            )
        )

    def enemy_muzzle_position(self, enemy: _Enemy) -> tuple[float, float] | None:
        position = self.enemy_muzzle_position_v3(enemy)
        return position[:2] if position is not None else None

    def enemy_muzzle_position_v3(self, enemy: _Enemy) -> tuple[float, float, float] | None:
        """Return the same map-space attachment used by emission and replay.

        Outside the configured attack pose, the profile's fixed offset is
        the current fallback. This query never changes battle state.
        """
        profile = enemy.graphic_profile
        if (not self.assumptions.enemy_projectile_muzzle or profile is None
                or (self.enemy_projectiles.get(enemy.key) or {}).get("mountPointType") != 2):
            return None
        elapsed = (
            max(0.0, self.time - enemy.attack_started_at)
            if enemy.attack_started_at is not None else None
        )
        attack_time = elapsed if (
            elapsed is not None
            and (enemy.attack_target is not None or elapsed <= enemy.attack_duration)
        ) else None
        offset = profile.muzzle_at(attack_time)
        if offset is None:
            return None
        sign = enemy.facing.value_at(self.time) if enemy.facing else 1.0
        return (enemy.position_row + offset[0],
                enemy.position_col + offset[1] * sign, offset[2])

    def _unit_hit_position(self, unit: _Enemy | _Operator) -> tuple[float, float, float] | None:
        if isinstance(unit, _Operator):
            offset = self.operator_hit_offsets.get(unit.char_id)
            row, col = map(float, unit.tile)
            sign = 1.0
        else:
            offset = unit.graphic_profile.hit_offset if unit.graphic_profile else None
            row, col = unit.position()
            sign = unit.facing.value_at(self.time) if unit.facing else 1.0
        if offset is None:
            return None
        return row + offset[0], col + offset[1] * sign, offset[2]

    def _projectile_target_position(
        self, target: _Enemy | _Operator, profile: ProjectileMotionProfile | None,
    ) -> tuple[float, float, float]:
        if profile is not None and profile.target_mount == "hit":
            position = self._unit_hit_position(target)
            if position is not None:
                return position
        row, col = map(float, target.tile) if isinstance(target, _Operator) else target.position()
        return row, col, 0.0

    def _emit_operator_projectiles(
        self, op: _Operator, attack: _PendingOperatorAttack
    ) -> None:
        if not op.projectile_key:
            return
        spec = self.operator_projectiles.get(op.char_id) or {}
        if spec.get("mapping") == "inferred_base_logic_key":
            self.used_assumptions.add("PROJECTILE_OPERATOR_INFERRED")
        for index, enemy in enumerate(attack.targets):
            if not self._enemy_is_attackable(enemy):
                continue
            scale = (
                attack.attack_scale
                if attack.kind != "MULTI" or index == 0
                else 1.0
            )
            payload = _PendingOperatorAttack(
                execute_at=attack.execute_at,
                execute_frame=attack.execute_frame,
                kind="SINGLE",
                primary=enemy,
                targets=[enemy],
                attack_scale=scale,
                source_atk_scale=attack.source_atk_scale,
                sp_recovery_locked=attack.sp_recovery_locked,
                will_hit=attack.will_hit,
            )
            self._new_projectile(
                key=op.projectile_key,
                side="OPERATOR",
                source=op,
                target=enemy,
                operator_attack=payload,
                will_hit=attack.will_hit,
            )

    def _land_hammer_attack(
        self, op: _Operator, attack: _PendingOperatorAttack
    ) -> None:
        target_position = attack.primary.position()
        nearby = [
            enemy
            for enemy in self.enemies
            if self._enemy_is_attackable(enemy)
            and self._enemy_intersects_circle(
                enemy, target_position, radius=1.0
            )
        ]
        module_scale = (
            1.15
            if op.module_id == "uniequip_002_headb2"
            and op.module_level > 0
            and len(nearby) >= 3
            else 1.0
        )
        attack_scale = attack.attack_scale * module_scale
        self._deal_damage_to_enemy(
            op,
            attack.primary,
            attack_scale,
            source_atk_scale=attack.source_atk_scale,
        )
        talent_scale = {2: 1.32, 3: 1.40}.get(op.module_level, 1.24)
        splash_scale = 0.5 * attack_scale
        for enemy in nearby:
            if (
                enemy is attack.primary
                or not self._enemy_is_attackable(enemy)
            ):
                continue
            self._deal_damage_to_enemy(
                op,
                enemy,
                splash_scale,
                attack_type=AttackType.SPLASH,
                apply_way=ApplyWay.MELEE,
                output_scale=talent_scale,
                source_atk_scale=attack.source_atk_scale,
            )
        self._schedule_headb2_highland_effects(
            op,
            target_position,
            module_scale,
            source_atk_scale=attack.source_atk_scale,
            sp_recovery_locked=attack.sp_recovery_locked,
        )

    def _schedule_headb2_highland_effects(
        self,
        op: _Operator,
        target_position: tuple[float, float],
        module_scale: float,
        *,
        source_atk_scale: float,
        sp_recovery_locked: bool,
    ) -> None:
        if op.char_id != "char_1051_headb2":
            return
        self.used_assumptions.add("HEADB2_ECHO_TARGET_AT_EXECUTION")
        self.used_assumptions.add(
            "HIGHLAND_TILE_CENTER"
            if self.assumptions.highland_hit_uses_tile_center
            else "HIGHLAND_TILE_INTERSECTION"
        )
        echo_scale = 0.27 if op.potential_rank >= 4 else 0.24
        # The main attack cast evaluates the X-module condition once.  Every
        # highland splash created by that attack inherits the same decision.
        amount = op.atk * source_atk_scale * echo_scale * module_scale
        for row in range(len(self.map_grid)):
            for col in range(len(self.map_grid[0])):
                tile = self.tile(row, col)
                if tile.get("heightType") != "HIGHLAND":
                    continue
                if not self._headb2_splash_hits_highland(
                    row, col, target_position
                ):
                    continue
                self.scheduled_highland_effects.append(
                    _ScheduledHighlandEffect(
                        execute_at=self.time + 0.1,
                        source=op,
                        tile=(row, col),
                        amount=amount,
                        sp_recovery_locked=sp_recovery_locked,
                    )
                )

    @staticmethod
    def _headb2_highland_affected_tiles(
        row: int, col: int
    ) -> set[tuple[int, int]]:
        return {
            (row, col),
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        }

    def _headb2_splash_hits_highland(
        self,
        row: int,
        col: int,
        target_position: tuple[float, float],
    ) -> bool:
        if self.assumptions.highland_hit_uses_tile_center:
            return math.hypot(
                row - target_position[0], col - target_position[1]
            ) <= 1.0 + 1e-9
        nearest_row = min(max(target_position[0], row - 0.5), row + 0.5)
        nearest_col = min(max(target_position[1], col - 0.5), col + 0.5)
        return math.hypot(
            nearest_row - target_position[0],
            nearest_col - target_position[1],
        ) <= 1.0 + 1e-9

    def _headb2_gain_highland_sp(
        self, op: _Operator, *, sp_recovery_locked: bool = False
    ) -> None:
        if op.skill is None or op.skill_prefab_id != "skchr_headb2_2":
            return
        if sp_recovery_locked:
            return
        if op.skill_active or self._skill_starting(op):
            self.used_assumptions.add("ACTIVE_SKILL_SP_LOCK")
            if not self.assumptions.allow_active_skill_sp_gain:
                return
            self._store_sp_gain(op, 1.0)
            return
        self._gain_sp(op, 1.0)

    def _process_scheduled_highland_effects(self) -> None:
        pending: list[_ScheduledHighlandEffect] = []
        for effect in self.scheduled_highland_effects:
            if effect.execute_at > self.time + 1e-9:
                pending.append(effect)
                continue
            self._headb2_gain_highland_sp(
                effect.source,
                sp_recovery_locked=effect.sp_recovery_locked,
            )
            affected = self._headb2_highland_affected_tiles(*effect.tile)
            # The delayed echo resolves against current positions after this
            # frame's enemy movement, rather than reusing an attack-time list.
            targets = (
                enemy
                for enemy in self.enemies
                if (
                    self._enemy_is_attackable(enemy)
                    and not self._is_airborne(enemy)
                    and enemy.tile() in affected
                )
            )
            for enemy in targets:
                if (
                    not self._enemy_is_attackable(enemy)
                    or self._is_airborne(enemy)
                ):
                    continue
                self._apply_operator_damage(
                    effect.source,
                    enemy,
                    effect.amount,
                    attack_type=AttackType.SPLASH,
                    apply_way=ApplyWay.RANGED,
                )
                if not enemy.dead:
                    sluggish_frames = max(
                        0, int(self.assumptions.sluggish_extra_frames)
                    )
                    self.used_assumptions.add(
                        "SLUGGISH_DURATION_EXTRA_FRAME"
                    )
                    enemy.statuses.apply(
                        Status.SLUGGISH,
                        0.5 + sluggish_frames / CLIENT_LOGIC_RATE,
                        resistible=False,
                    )
        self.scheduled_highland_effects = pending

    def _finish_depleted_ammo_before_attack(self, op: _Operator) -> None:
        if (
            op.skill_active
            and op.skill_duration_type == "AMMO"
            and op.ammo is not None
            and op.ammo <= 0
        ):
            self._finish_skill(op, "AMMO_EXHAUSTED")

    def _sp_on_attack(
        self, op: _Operator, *, sp_recovery_locked: bool = False
    ) -> None:
        if sp_recovery_locked:
            return
        if (
            op.skill is not None
            and op.sp_type == "INCREASE_WHEN_ATTACK"
        ):
            self._gain_sp(op, op.sp_increment)

    def _deal_damage_to_enemy(
        self,
        op: _Operator,
        enemy: _Enemy,
        scale: float = 1.0,
        *,
        attack_type: AttackType = AttackType.NORMAL,
        apply_way: ApplyWay = ApplyWay.MELEE,
        output_scale: float = 1.0,
        source_atk_scale: float | None = None,
    ):
        context = self._dispatch_skill_event(
            op, "ON_CALCULATE_DAMAGE", target=enemy, attack_scale=scale
        )
        if context is not None:
            scale = context.attack_scale
        if source_atk_scale is None:
            source_atk_scale = getattr(op, "atk_scale", 1.0)
        atk = op.atk * source_atk_scale * scale
        return self._apply_operator_damage(
            op,
            enemy,
            atk,
            attack_type=attack_type,
            apply_way=apply_way,
            output_scale=output_scale,
            taken_scale=context.damage_scale if context is not None else 1.0,
        )

    def _apply_operator_damage(
        self,
        op: _Operator,
        enemy: _Enemy,
        amount: float,
        *,
        attack_type: AttackType,
        apply_way: ApplyWay,
        output_scale: float = 1.0,
        taken_scale: float = 1.0,
    ):
        result = apply_damage(
            DamagePacket(
                amount=amount,
                damage_type=getattr(op, "damage_type", "PHYSICAL"),
                attack_type=attack_type,
                apply_way=apply_way,
                output_scale=self._outgoing_damage_scale(op) * output_scale,
                taken_scale=taken_scale,
                source=op,
                critical_chance=float(getattr(op, "critical_chance", 0.0)),
                critical_multiplier=float(
                    getattr(op, "critical_multiplier", 1.0)
                ),
                target_resistance_override=self._effective_resistance(enemy),
                target_defense_override=self._effective_enemy_defense(enemy),
            ),
            enemy,
            roll=self._roll_probability,
        )
        if result.killed:
            self.enemies_killed += 1
            if hasattr(op, "kills"):
                op.kills += 1
        if hasattr(op, "damage_dealt"):
            op.damage_dealt += result.final_amount
        return result

    def _deal_damage_to_operator(
        self,
        enemy: _Enemy,
        op: _Operator,
        *,
        attack_override: float | None = None,
    ) -> None:
        defense = op.defense * getattr(op, "def_scale", 1.0)
        result = apply_damage(
            DamagePacket(
                amount=(
                    self._effective_enemy_atk(enemy)
                    if attack_override is None
                    else attack_override
                ),
                damage_type=DamageType.PHYSICAL,
                attack_type=AttackType.NORMAL,
                apply_way=(
                    ApplyWay.RANGED if enemy.range_radius > 0 else ApplyWay.MELEE
                ),
                output_scale=self._outgoing_damage_scale(enemy),
                source=enemy,
                critical_chance=float(
                    getattr(enemy, "critical_chance", 0.0)
                ),
                critical_multiplier=float(
                    getattr(enemy, "critical_multiplier", 1.0)
                ),
                target_defense_override=defense,
                target_resistance_override=self._effective_resistance(op),
            ),
            op,
            roll=self._roll_probability,
        )
        if (
            result.final_amount > 0
            and op.skill is not None
            and op.sp_type == "INCREASE_WHEN_TAKEN_DAMAGE"
        ):
            self._gain_sp(op, op.sp_increment)
        if result.killed and op.death_time is None:
            op.death_time = self.time

    def _fire_enemy_attack(
        self, enemy: _Enemy, target: _Operator, *, will_hit: bool
    ) -> None:
        spec = self.enemy_projectiles.get(enemy.key) or {}
        destination = int(spec.get("destination", 0) or 0)
        if enemy.projectile_key and destination != 0:
            self.mechanic_diagnostics[
                f"PROJECTILE_DESTINATION:{destination}"
            ] += 1
            if will_hit:
                self._enemy_attack_lands(enemy, target)
            return
        speed = self._projectile_speed(enemy.projectile_key)
        if speed is None:
            if will_hit:
                self._enemy_attack_lands(enemy, target)
            return
        self._new_projectile(
            key=enemy.projectile_key or "",
            side="ENEMY",
            source=enemy,
            target=target,
            will_hit=will_hit,
            cached_atk=(
                self._effective_enemy_atk(enemy)
                if spec.get("useCachedAtkOnly")
                else None
            ),
        )

    @staticmethod
    def _projectile_target_valid(projectile: _Projectile) -> bool:
        target = projectile.target
        if target.dead:
            return False
        if isinstance(target, _Enemy):
            return not target.leaked and (
                not target.disappeared or target.portal_targetable
            )
        return True

    def _process_projectiles(self) -> None:
        active: list[_Projectile] = []
        for projectile in self.projectiles:
            if not self._projectile_target_valid(projectile):
                continue
            profile = projectile.motion_profile
            target_row, target_col, target_z = self._projectile_target_position(
                projectile.target, profile,
            )
            delta_row = target_row - projectile.position_row
            delta_col = target_col - projectile.position_col
            delta_z = target_z - projectile.position_z
            distance = math.hypot(delta_row, delta_col)
            spatial = profile is not None and profile.distance_dimensions == 3
            if spatial:
                distance = math.sqrt(delta_row ** 2 + delta_col ** 2 + delta_z ** 2)
            step_seconds = self.dt
            if (profile is not None and profile.quantized_step
                    and math.isclose(self.dt, 1 / CLIENT_LOGIC_RATE, rel_tol=0.0, abs_tol=1e-12)):
                step_seconds = CLIENT_FIXED_TIMER_STEP
            step_distance = projectile.speed * step_seconds
            epsilon = 1e-9 if profile is None else 0.0
            if distance <= step_distance + epsilon:
                projectile.travelled += distance
                projectile.position_row = target_row
                projectile.position_col = target_col
                if spatial:
                    projectile.position_z = target_z
            else:
                projectile.position_row += delta_row / distance * step_distance
                projectile.position_col += delta_col / distance * step_distance
                if spatial:
                    projectile.position_z += delta_z / distance * step_distance
                projectile.travelled += step_distance
            radius = profile.reach_radius if profile is not None else 0.0
            if distance - step_distance <= radius + epsilon:
                self._projectile_lands(projectile)
            else:
                active.append(projectile)
        self.projectiles = active

    def _projectile_lands(self, projectile: _Projectile) -> None:
        if not projectile.will_hit or not self._projectile_target_valid(projectile):
            return
        if projectile.side == "ENEMY":
            self._enemy_attack_lands(
                projectile.source,
                projectile.target,
                attack_override=projectile.cached_atk,
            )
            return
        attack = projectile.operator_attack
        if attack is None:
            return
        self._deal_damage_to_enemy(
            projectile.source,
            projectile.target,
            attack.attack_scale,
            apply_way=ApplyWay.RANGED,
            source_atk_scale=attack.source_atk_scale,
        )

    def _enemy_attack_lands(
        self,
        enemy: _Enemy,
        target: _Operator,
        *,
        attack_override: float | None = None,
    ) -> None:
        """Resolve one enemy hit, including target-centered splash attacks."""
        self._deal_damage_to_operator(
            enemy, target, attack_override=attack_override
        )
        if enemy.key not in MORTAR_KEYS:
            return
        for op in self.operators:
            if op is target or op.dead:
                continue
            if max(
                abs(op.tile[0] - target.tile[0]),
                abs(op.tile[1] - target.tile[1]),
            ) <= 1:
                self._deal_damage_to_operator(
                    enemy, op, attack_override=attack_override
                )

    def _roll_probability(self, probability: float) -> bool:
        if not hasattr(self, "rng"):
            self.seed = 0
            self.rng = DeterministicRng(0)
        return self.rng.roll(probability)

    def _attack_hits(
        self,
        source: _Enemy | _Operator,
        damage_type: DamageType | str,
    ) -> bool:
        normalized = DamagePacket(0.0, damage_type).normalized_damage_type()
        if normalized is DamageType.PHYSICAL:
            probability = float(getattr(source, "physical_hit_rate", 1.0))
        elif normalized is DamageType.ARTS:
            probability = float(getattr(source, "arts_hit_rate", 1.0))
        else:
            return True
        return self._roll_probability(probability)

    def choose_random_target(self, candidates: list[Any]) -> Any:
        """Choose once from an already-filtered candidate list."""
        if not candidates:
            return None
        if not hasattr(self, "rng"):
            self.seed = 0
            self.rng = DeterministicRng(0)
        return candidates[self.rng.choose_index(len(candidates))]

    @staticmethod
    def _enemy_is_attackable(enemy: _Enemy) -> bool:
        """Portal hold frames keep the hitbox active before route reentry."""
        return (
            not enemy.dead
            and not enemy.leaked
            and (not enemy.disappeared or enemy.portal_targetable)
        )

    def initialize_operator_target_selector(
        self,
        op: _Operator,
        *,
        frame_index: int | None = None,
        first_search_frame: int | None = None,
    ) -> TargetSelectorRuntime:
        """Create/reset an operator selector from data-driven timing config.

        This is also the explicit setup hook for tests that construct an
        ``_Operator`` directly instead of going through deployment.
        """

        timing = self.attack_timing.get(op.char_id, {})
        if not isinstance(timing, dict):
            timing = {}
        selector = timing.get("target_selector") or {}
        if not isinstance(selector, dict):
            selector = {}
        try:
            period_frames = int(selector.get("period_frames", 1))
        except (TypeError, ValueError):
            period_frames = 1
        schedule_mode = str(
            selector.get("schedule_mode", "character_state")
        ).lower()
        search_phase = str(
            selector.get("search_phase", "post_enemy_move")
        ).lower()
        runtime = op.target_selector
        if runtime is None or runtime.ticker.period_frames != max(
            period_frames, 1
        ) or runtime.ticker.wait_first_period != bool(
            selector.get("wait_first_period", False)
        ) or (
            runtime.schedule_mode != schedule_mode
            or runtime.search_phase != search_phase
        ):
            runtime = TargetSelectorRuntime(
                FramePeriodicTicker(
                    period_frames=max(period_frames, 1),
                    wait_first_period=bool(
                        selector.get("wait_first_period", False)
                    ),
                ),
                schedule_mode=schedule_mode,
                search_phase=search_phase,
            )
            op.target_selector = runtime
        runtime.reset(
            self.frame_index if frame_index is None else int(frame_index)
        )
        op.selector_first_search_frame = (
            None if first_search_frame is None else int(first_search_frame)
        )
        op.attack_chain_active = False
        op.attack_first_cast = False
        op.selector_resume_frame = None
        op.selector_pause_until_frame = None
        first_in_range = getattr(self, "_targeting_first_in_range", None)
        if first_in_range is not None:
            for key in [key for key in first_in_range if key[0] == id(op)]:
                del first_in_range[key]
        return runtime

    @staticmethod
    def _targeting_target_id(target: _Enemy | None) -> int | None:
        if target is None:
            return None
        enemy_index = getattr(target, "enemy_index", None)
        if enemy_index is not None:
            return int(enemy_index)
        creation_index = getattr(target, "creation_index", None)
        return int(creation_index) if creation_index is not None else id(target)

    def _targeting_first_in_range_frame(
        self,
        op: _Operator,
        target: _Enemy | None,
        *,
        old_target: _Enemy | None = None,
        invalidated: bool = False,
    ) -> int | None:
        if target is None:
            return None
        first_in_range = getattr(self, "_targeting_first_in_range", None)
        if first_in_range is None:
            return int(self.frame_index)
        key = (id(op), id(target))
        if invalidated or target is not old_target:
            first_in_range.pop(key, None)
        return first_in_range.setdefault(key, int(self.frame_index))

    def _record_targeting_diagnostic(
        self, op: _Operator, **updates: Any
    ) -> None:
        diagnostics = getattr(self, "targeting_diagnostics", None)
        if diagnostics is None:
            return
        index_by_key = getattr(self, "_targeting_diagnostic_index", None)
        if index_by_key is None:
            index_by_key = {}
            self._targeting_diagnostic_index = index_by_key
        key = (id(op), int(self.frame_index))
        index = index_by_key.get(key)
        if index is None:
            runtime = op.target_selector
            cached = (
                runtime.cached_target
                if runtime is not None and runtime.initialized
                else None
            )
            event = {
                "fixed_frame": int(self.frame_index),
                "operator_char_id": op.char_id,
                "operator_creation_index": op.creation_index,
                "ticker_ready": False,
                "ticker_ready_before_tick": False,
                "ticker_ready_after_tick": False,
                "search_executed": False,
                "first_attack": None,
                "attack_chain_active": op.attack_chain_active,
                "old_cached_target": self._targeting_target_id(cached),
                "new_cached_target": self._targeting_target_id(cached),
                "target_first_in_range": None,
                "pending_created": False,
                "pending_execute_at": None,
                "hit_frame": None,
            }
            diagnostics.append(event)
            index = len(diagnostics) - 1
            index_by_key[key] = index
        diagnostics[index].update(updates)

    def _validate_operator_target(
        self, op: _Operator, target: _Enemy | None
    ) -> bool:
        if target is None or not self._enemy_is_attackable(target):
            return False
        if not target.statuses.targetable:
            return False
        blocked = any(enemy is target for enemy in op.blocked)
        if (op.prioritize_blocked or op.subprofession == "hammer") and blocked:
            # Preserve the existing blocked-first rule: once an enemy is
            # blocked, the melee selector does not reapply ranged geometry.
            return True
        if target.statuses.has(Status.CAMOUFLAGE):
            return False
        return self._enemy_intersects_range(target, op.range_cells)

    def _search_operator_target_now(self, op: _Operator) -> _Enemy | None:
        """Perform the one permitted full target scan for a selector refresh."""

        if op.prioritize_blocked or op.subprofession == "hammer":
            for enemy in op.blocked:
                if self._validate_operator_target(op, enemy):
                    return enemy

        candidates: list[_Enemy] = []
        for enemy in self.enemies:
            if not self._validate_operator_target(op, enemy):
                continue
            candidates.append(enemy)
        if not candidates:
            return None
        priority = cmp_to_key(compare_operator_target_keys)
        return min(candidates, key=lambda enemy: priority(operator_target_key(enemy)))

    def _selector_resume_allowed(self, op: _Operator) -> bool:
        resume_frame = op.selector_resume_frame
        return resume_frame is None or self.frame_index >= resume_frame

    def _advance_operator_target_selector(
        self, op: _Operator, force: bool = False
    ) -> bool:
        """Advance one selector frame around the native Search/Next split.

        Idle Search is intentionally before this frame's ticker Tick.  An
        AttackState keeps its cached target untouched until the complete cast
        interval ends; callers then use ``_force_refresh_operator_target`` for
        the post-Tick continuous-attack path.
        """

        runtime = op.target_selector
        if runtime is None or not runtime.initialized:
            return False
        pause_until = op.selector_pause_until_frame
        if pause_until is not None:
            if self.frame_index <= pause_until:
                return False
            op.selector_pause_until_frame = None
        old_target = runtime.cached_target
        ready_before_tick = runtime.ticker.is_ready
        first_search_frame = op.selector_first_search_frame
        startup_complete = (
            first_search_frame is None
            or self.frame_index >= first_search_frame
        )
        invalidated = False
        search_executed = False

        if not op.attack_chain_active:
            invalidated = runtime.invalidate_if_needed(
                lambda target: self._validate_operator_target(op, target)
            )
            if invalidated:
                first_in_range = getattr(self, "_targeting_first_in_range", None)
                if first_in_range is not None and old_target is not None:
                    first_in_range.pop((id(op), id(old_target)), None)
            if (
                startup_complete
                and (force or self._selector_resume_allowed(op))
                and (force or ready_before_tick)
            ):
                search_executed = runtime.refresh(
                    lambda: self._search_operator_target_now(op),
                    lambda target: self._validate_operator_target(op, target),
                    force=force,
                    frame_index=self.frame_index,
                )
                if search_executed and op.selector_resume_frame is not None:
                    op.selector_resume_frame = None

        ticker_ready_after_tick = runtime.tick(self.frame_index)

        # A forced refresh while already in AttackState is deliberately after
        # Tick.  Normal battle stepping reaches this path from Phase C; the
        # ``force`` argument also keeps direct test callers useful.
        if force and op.attack_chain_active:
            search_executed = self._force_refresh_operator_target(op)

        if op.attack_chain_active:
            new_target = runtime.cached_target
        else:
            new_target = runtime.current_target(
                lambda target: self._validate_operator_target(op, target)
            )
        first_in_range = self._targeting_first_in_range_frame(
            op,
            new_target,
            old_target=old_target,
            invalidated=invalidated,
        )
        self._record_targeting_diagnostic(
            op,
            ticker_ready=ready_before_tick or ticker_ready_after_tick,
            ticker_ready_before_tick=ready_before_tick,
            ticker_ready_after_tick=ticker_ready_after_tick,
            search_executed=search_executed,
            old_cached_target=self._targeting_target_id(old_target),
            new_cached_target=self._targeting_target_id(new_target),
            target_first_in_range=first_in_range,
        )
        return search_executed

    def _refresh_operator_target_selector(
        self, op: _Operator, force: bool = False
    ) -> bool:
        """Compatibility wrapper for tests and external diagnostics."""

        return self._advance_operator_target_selector(op, force=force)

    def _force_refresh_operator_target(
        self, op: _Operator, *, candidate: _Enemy | None = None
    ) -> bool:
        """Run AttackState's explicit Search(force) after the frame Tick."""

        runtime = op.target_selector
        if runtime is None or not runtime.initialized:
            return False
        old_target = runtime.cached_target
        searched = runtime.refresh(
            lambda: (
                candidate
                if candidate is not None
                else self._search_operator_target_now(op)
            ),
            lambda target: self._validate_operator_target(op, target),
            force=True,
            frame_index=self.frame_index,
        )
        new_target = runtime.cached_target
        first_in_range = self._targeting_first_in_range_frame(
            op,
            new_target,
            old_target=old_target,
        )
        self._record_targeting_diagnostic(
            op,
            search_executed=searched,
            old_cached_target=self._targeting_target_id(old_target),
            new_cached_target=self._targeting_target_id(new_target),
            target_first_in_range=first_in_range,
        )
        return searched

    def _cached_operator_target_is_valid(self, op: _Operator) -> bool:
        runtime = op.target_selector
        if runtime is None or not runtime.initialized:
            return False
        target = runtime.cached_target
        return target is not None and self._validate_operator_target(op, target)

    def _finish_operator_attack_chain(self, op: _Operator) -> None:
        """Exit AttackState; resume idle processing on the next frame.

        An attack-boundary Search has already rearmed its ticker, including
        empty Headb2 searches. Readiness, rather than an extra continuous-cast
        handoff, determines when the next idle Search can run.
        """

        runtime = op.target_selector
        old_target = runtime.cached_target if runtime is not None else None
        if runtime is not None:
            runtime.clear_cached_target()
        first_in_range = getattr(self, "_targeting_first_in_range", None)
        if first_in_range is not None and old_target is not None:
            first_in_range.pop((id(op), id(old_target)), None)
        op.attack_chain_active = False
        op.attack_first_cast = False
        op.selector_resume_frame = self.frame_index + 1
        self._record_targeting_diagnostic(
            op,
            new_cached_target=None,
            target_first_in_range=None,
        )

    def _current_operator_target(self, op: _Operator) -> _Enemy | None:
        """Read and validate the cache; never perform a fallback scan."""

        runtime = op.target_selector
        if runtime is None or not runtime.initialized:
            return None
        before = runtime.cached_target
        target = runtime.current_target(
            lambda candidate: self._validate_operator_target(op, candidate)
        )
        if target is not before:
            first_in_range = getattr(self, "_targeting_first_in_range", None)
            if first_in_range is not None and before is not None:
                first_in_range.pop((id(op), id(before)), None)
            self._record_targeting_diagnostic(
                op,
                new_cached_target=self._targeting_target_id(target),
            )
        return target

    def _select_target(self, op: _Operator) -> _Enemy | None:
        """Legacy direct-search helper; formal attacks use the selector cache."""

        return self._search_operator_target_now(op)

    def _enemy_intersects_circle(
        self,
        enemy: _Enemy,
        center: tuple[float, float],
        radius: float,
    ) -> bool:
        """Return whether a circular AoE overlaps the enemy body circle.

        Winter's highland echo has an explicit discrete affected-tile rule and
        intentionally does not use this generic splash selector.
        """
        body_radius = max(
            float(self.assumptions.enemy_target_collision_radius), 0.0
        )
        self.used_assumptions.add("ENEMY_TARGET_COLLISION_RADIUS")
        return (
            math.hypot(
                enemy.position_row - center[0],
                enemy.position_col - center[1],
            )
            <= max(float(radius), 0.0) + body_radius + 1e-9
        )

    def _enemy_intersects_range(
        self, enemy: _Enemy, range_cells: set[tuple[int, int]]
    ) -> bool:
        """Return whether an enemy body overlaps any discrete range tile."""
        radius = max(float(self.assumptions.enemy_target_collision_radius), 0.0)
        self.used_assumptions.add("ENEMY_TARGET_COLLISION_RADIUS")
        row, col = enemy.position()
        for cell_row, cell_col in range_cells:
            nearest_row = min(max(row, cell_row - 0.5), cell_row + 0.5)
            nearest_col = min(max(col, cell_col - 0.5), cell_col + 0.5)
            if math.hypot(nearest_row - row, nearest_col - col) <= radius + 1e-9:
                return True
        return False

    def _select_operator_target(self, enemy: _Enemy) -> _Operator | None:
        if enemy.blocked_by is not None and not enemy.blocked_by.dead:
            return enemy.blocked_by
        if enemy.range_radius < 0:
            return None
        candidates: list[_Operator] = []
        r, c = enemy.position()
        for op in self.operators:
            if op.dead:
                continue
            if not op.statuses.targetable or op.statuses.has(Status.CAMOUFLAGE):
                continue
            dist = math.hypot(op.tile[0] - r, op.tile[1] - c)
            if dist <= enemy.range_radius + OPERATOR_HITBOX_RADIUS + 1e-9:
                candidates.append(op)
        if not candidates:
            return None
        return min(candidates, key=enemy_target_key)

    def _advance_enemy_operator_target_search(
        self, enemy: _Enemy
    ) -> tuple[_Operator | None, bool]:
        """Resume Search when the attack interval ends, then use the idle cycle."""
        ticker = enemy.target_search_ticker
        if not ticker.initialized:
            ticker.reset(self.frame_index)
        if enemy.target_search_after_attack:
            if not self._enemy_attack_ready(enemy):
                return None, False
            ticker.reset(self.frame_index)
            enemy.target_search_after_attack = False
        search_cycle = ticker.tick(self.frame_index)
        if search_cycle:
            enemy.cached_operator_target = self._select_operator_target(enemy)
            ticker.next(self.frame_index)
            self.used_assumptions.add("ENEMY_TARGET_SEARCH_CYCLE")

        target = enemy.cached_operator_target
        if target is None:
            return None, search_cycle
        if enemy.blocked_by is target and not target.dead:
            return target, search_cycle
        if target.dead or not target.statuses.targetable:
            enemy.cached_operator_target = None
            return None, search_cycle
        if target.statuses.has(Status.CAMOUFLAGE):
            enemy.cached_operator_target = None
            return None, search_cycle

        row, col = enemy.position()
        distance = math.hypot(target.tile[0] - row, target.tile[1] - col)
        if distance > enemy.range_radius + OPERATOR_HITBOX_RADIUS + 1e-9:
            enemy.cached_operator_target = None
            return None, search_cycle
        return target, search_cycle

    @staticmethod
    def _aura_source_active(enemy: _Enemy) -> bool:
        """A global source Ability remains attached during portal hiding."""
        return (
            not getattr(enemy, "dead", False)
            and not getattr(enemy, "leaked", False)
        )

    def _sync_herald_auras(self) -> None:
        """Register/remove source-owned aura Buffs without touching getters."""
        enemies = list(getattr(self, "enemies", ()))
        aura_sources = [
            (source, self.buff_catalog.aura_definitions(source.key))
            for source in enemies
            if self._aura_source_active(source)
            and self.buff_catalog.aura_definitions(source.key)
        ]
        aura_keys = self.buff_catalog.aura_keys
        for target in enemies:
            if getattr(target, "dead", False):
                continue
            container = getattr(target, "buffs", None)
            if container is None:
                continue
            for source, definitions in aura_sources:
                for definition in definitions:
                    if not container.contains(
                        definition.key, source=source
                    ):
                        container.add(
                            definition,
                            source=source,
                            attached_step=self.frame_index,
                        )
            for instance in tuple(container.active):
                if instance.definition.key not in aura_keys:
                    continue
                if not any(
                    instance.source is source
                    and any(
                        definition.key == instance.definition.key
                        for definition in definitions
                    )
                    for source, definitions in aura_sources
                ):
                    container.finish(instance)
            # A source's base marker is visible to this frame's later phases;
            # the target's periodic listeners still only tick at frame end.
            container.commit()

    def _tick_enemy_buffs(self) -> None:
        for enemy in list(getattr(self, "enemies", ())):
            if enemy.dead:
                continue
            container = getattr(enemy, "buffs", None)
            if container is None:
                continue
            container.tick(self.dt, self.frame_index)
            container.commit()

    def _dispatch_buff_event(self, instance: BuffInstance, event: str) -> bool:
        """Run normalized Buff actions through the existing interpreter."""
        actions = instance.definition.actions_for(event)
        if not actions:
            return True
        owner = instance.owner
        context = BehaviorContext(
            source=instance.source,
            target=owner,
            buff_owner=owner,
            buff_source=instance.source,
            blackboard=instance.blackboard,
            ability_blackboard={},
            battle=self,
            diagnostics=self.behavior_diagnostics,
            template_key=instance.definition.key,
            event=event,
            current_buff_key=instance.definition.key,
            buff_container=getattr(owner, "buffs", None),
            buff_instance=instance,
        )
        return self.behavior.execute_actions(list(actions), context)

    def _living_herald_count(self) -> int:
        return sum(
            enemy.key in HERALD_KEYS
            and self._aura_source_active(enemy)
            for enemy in getattr(self, "enemies", ())
        )

    def _tactical_command_active(self) -> bool:
        return self._living_herald_count() > 0

    @staticmethod
    def _enemy_has_committed_buff(enemy: _Enemy, key: str) -> bool:
        container = getattr(enemy, "buffs", None)
        return bool(
            container is not None
            and hasattr(container, "contains")
            and container.contains(key)
        )

    def _effective_enemy_atk(self, enemy: _Enemy) -> float:
        container = getattr(enemy, "buffs", None)
        if container is None or not hasattr(container, "attribute_value"):
            return enemy.atk
        has_herald_attribute = container.count(
            "enemy_9D0_talent_strength[attribute]"
        ) > 0
        source_limit = None
        if has_herald_attribute:
            self.used_assumptions.add("HERALD_AURAS_STACK")
            if not self.assumptions.herald_auras_stack:
                source_limit = 1
        return container.attribute_value(
            "ATK", enemy.atk, source_limit=source_limit
        )

    def _effective_enemy_defense(self, enemy: _Enemy) -> float:
        container = getattr(enemy, "buffs", None)
        if container is None or not hasattr(container, "attribute_value"):
            return enemy.defense
        has_herald_attribute = container.count(
            "enemy_9D0_talent_strength[attribute]"
        ) > 0
        source_limit = None
        if has_herald_attribute:
            self.used_assumptions.add("HERALD_AURAS_STACK")
            if not self.assumptions.herald_auras_stack:
                source_limit = 1
        return container.attribute_value(
            "DEF", enemy.defense, source_limit=source_limit
        )

    def _enemy_uses_frame_attack_cooldown(self, enemy: _Enemy) -> bool:
        """Return whether the opt-in mortar frame comparison applies."""
        return (
            getattr(self, "enemy_attack_timing", ENEMY_ATTACK_TIMING_FLOAT)
            == ENEMY_ATTACK_TIMING_MORTAR_FRAMES
            and enemy.key in MORTAR_KEYS
        )

    def _tick_enemy_attack_cooldown(self, enemy: _Enemy) -> None:
        frame_based = self._enemy_uses_frame_attack_cooldown(enemy)
        timer_active = (
            enemy.attack_cd_frames > 0 if frame_based else enemy.attack_cd > 0
        )
        if timer_active and enemy.attack_cd_interval > 0.0:
            interval = self._effective_attack_interval(enemy)
            if interval > 0.0 and not math.isclose(
                interval,
                enemy.attack_cd_interval,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                # Native mortar #49 scales an active timer when its period
                # changes; retain this as a replaceable compatibility rule.
                self.used_assumptions.add("ENEMY_ATTACK_COOLDOWN_RETIME")
                if self.assumptions.retime_enemy_attack_cooldown:
                    ratio = interval / enemy.attack_cd_interval
                    if frame_based:
                        enemy.attack_cd_frames = max(
                            0, int(round(enemy.attack_cd_frames * ratio))
                        )
                        enemy.attack_cd = (
                            enemy.attack_cd_frames / CLIENT_LOGIC_RATE
                        )
                    else:
                        enemy.attack_cd *= ratio
                enemy.attack_cd_interval = interval

        if frame_based:
            enemy.attack_cd_frames = max(0, enemy.attack_cd_frames - 1)
            enemy.attack_cd = enemy.attack_cd_frames / CLIENT_LOGIC_RATE
            return
        timer_step = (
            CLIENT_FIXED_TIMER_STEP
            if math.isclose(
                self.dt, 1 / CLIENT_LOGIC_RATE, rel_tol=0.0, abs_tol=1e-12
            )
            else self.dt
        )
        enemy.attack_cd -= timer_step

    def _enemy_attack_ready(self, enemy: _Enemy) -> bool:
        if self._enemy_uses_frame_attack_cooldown(enemy):
            return enemy.attack_cd_frames <= 0
        return enemy.attack_cd <= 0

    def _start_enemy_attack_cooldown(self, enemy: _Enemy) -> None:
        interval = self._effective_attack_interval(enemy)
        first_mortar_attack = (
            enemy.attack_count == 0
            and enemy.key in MORTAR_KEYS
            and self._enemy_has_committed_buff(
                enemy, "enemy_talent_strength[atk_speed]"
            )
        )
        first_extra_frames = 0
        if first_mortar_attack:
            first_extra_frames = max(
                0,
                int(
                    getattr(
                        self.assumptions,
                        "mortar_first_attack_extra_frames",
                        DEFAULT_ASSUMPTIONS.mortar_first_attack_extra_frames,
                    )
                ),
            )
            if first_extra_frames:
                self.used_assumptions.add(
                    "MORTAR_FIRST_ATTACK_EXTRA_FRAMES"
                )
        enemy.attack_cd_interval = interval
        if self._enemy_uses_frame_attack_cooldown(enemy):
            # 3.2 seconds is exactly 96 client logic frames mathematically,
            # but repeated float subtraction can leave a tiny positive tail.
            enemy.attack_cd_frames = max(
                1,
                int(round(interval * CLIENT_LOGIC_RATE)) + first_extra_frames,
            )
            enemy.attack_cd = enemy.attack_cd_frames / CLIENT_LOGIC_RATE
            enemy.attack_count += 1
            return
        enemy.attack_cd_frames = 0
        enemy.attack_cd = interval + first_extra_frames / CLIENT_LOGIC_RATE
        enemy.attack_count += 1

    def _start_enemy_attack(
        self,
        enemy: _Enemy,
        target: _Operator,
        *,
        hit_after_windup: bool = False,
    ) -> None:
        self._start_enemy_attack_cooldown(enemy)
        if enemy.apply_way != ApplyWay.MELEE.value:
            enemy.target_search_after_attack = True
        if enemy.palsy_stacks > 0:
            enemy.palsy_stacks -= 1
            enemy.statuses.apply(
                Status.PALSY_TREMOR, 0.5, resistible=False
            )
            enemy.pause_until = self.time + 0.5
            return
        enemy.attack_target = target
        if enemy.graphic_profile is not None:
            enemy.attack_started_at = self.time
            self._request_enemy_facing(enemy, target.tile[1] - enemy.position_col)
        enemy.attack_will_hit = self._attack_hits(
            enemy, DamageType.PHYSICAL
        )
        hit_delay = enemy.attack_hit_time
        if hit_after_windup:
            # attack_hit_time is the pre-hit windup; resolve on the next
            # fixed frame after that animation segment completes.
            hit_delay = max(0.0, enemy.attack_hit_time) + self.dt
        enemy.attack_hit_at = self.time + hit_delay
        enemy.attack_hit_frame = (
            self.frame_index
            + _fixed_delay_frames(hit_delay, self.dt)
        )
        if (not hit_after_windup and enemy.apply_way == ApplyWay.RANGED.value
                and enemy.wait_for_attack_event and self.assumptions.enemy_animation_event_clock):
            step = (CLIENT_FIXED_TIMER_STEP
                    if math.isclose(self.dt, 1 / CLIENT_LOGIC_RATE, rel_tol=0.0, abs_tol=1e-12)
                    else self.dt)
            frames = _animation_event_delay_frames(hit_delay, step)
            enemy.attack_hit_frame = self.frame_index + frames
            enemy.attack_hit_at = self.time + frames * self.dt
            self.used_assumptions.add("ENEMY_ANIMATION_EVENT_CLOCK")
        enemy.pause_until = self.time + enemy.attack_duration
        enemy.attack_move_resume_frame = None
        if (
            enemy.apply_way == ApplyWay.RANGED.value
            and enemy.attack_duration > 0.0
        ):
            # Model Attack completion -> Move entry -> first displacement.
            # Integer deadlines avoid absolute-time drift around whole frames.
            # The transition offset is a replaceable normal-ranged model;
            # special actions and interrupted attacks need separate validation.
            transition_frames = max(
                0, int(self.assumptions.ranged_attack_move_transition_frames)
            )
            enemy.attack_move_resume_frame = (
                self.frame_index
                + _fixed_delay_frames(enemy.attack_duration, self.dt)
                + transition_frames
            )
            self.used_assumptions.add("RANGED_ATTACK_MOVE_TRANSITION")

    def _request_enemy_facing(self, enemy: _Enemy, delta_col: float) -> None:
        if enemy.facing is None or enemy.graphic_profile is None:
            return
        duration = (
            enemy.graphic_profile.turn_seconds
            if self.assumptions.enemy_facing_transition else 0.0
        )
        enemy.facing.request(delta_col, self.time, duration)
        if self.assumptions.enemy_facing_transition:
            self.used_assumptions.add("ENEMY_FACING_TRANSITION")

    def _request_enemy_route_facing(self, enemy: _Enemy) -> None:
        profile = enemy.graphic_profile
        if (profile is None or not profile.face_route or enemy.disappeared
                or enemy.route_idx >= len(enemy.route)):
            return
        # Follow the route cursor, not avoidance velocity or portal jumps.
        self._request_enemy_facing(
            enemy, enemy.route[enemy.route_idx].col - enemy.position_col
        )

    def _effective_attack_interval(
        self, unit: _Enemy | _Operator | None = None
    ) -> float:
        # Keep the historical class-call form used by formula-only tests.
        if unit is None:
            unit = self
        element_bonus = (
            50.0
            if unit.element_burst is not None
            and unit.element_burst.element is ElementType.ANGER
            and not unit.elements.enemy_unit
            else 0.0
        )
        attack_speed = max(
            1.0,
            100.0
            + unit.statuses.attack_speed_delta
            + element_bonus
        )
        container = getattr(unit, "buffs", None)
        if container is not None and hasattr(container, "attribute_value"):
            attack_speed = max(
                1.0,
                container.attribute_value("ATTACK_SPEED", attack_speed),
            )
        return unit.attack_interval * 100.0 / attack_speed

    def _effective_enemy_move_speed(self, enemy: _Enemy) -> float:
        speed = enemy.move_speed * self.move_multiplier
        container = getattr(enemy, "buffs", None)
        if container is not None and hasattr(container, "attribute_value"):
            speed = container.attribute_value("MOVE_SPEED", speed)
        if enemy.statuses.has(Status.SLUGGISH):
            speed *= 0.2
        return speed

    def _move_enemy(self, enemy: _Enemy) -> None:
        initial_wait_expiry = (
            enemy.route_idx == 1
            and len(enemy.route) >= 2
            and enemy.route[0].kind.upper() == "START"
            and enemy.route[1].kind.upper()
            in {"WAIT_FOR_SECONDS", "WAIT_CURRENT_WAVE_TIME"}
            and enemy.wait_active_at_frame_start
            and enemy.wait_remaining <= 1e-12
            and not enemy.wait_hidden_at_frame_start
        )
        if initial_wait_expiry:
            self.used_assumptions.add("INITIAL_ROUTE_WAIT_SAME_FRAME_MOVE")
            # H7-2 #3/#4/#5 support movement on the visible birth-wait
            # expiry frame; later route waits keep their own phase.
            enemy.wait_active_at_frame_start = False
            enemy.wait_phase_consumed = False
        wait_active_at_frame_start = enemy.wait_active_at_frame_start
        wait_hidden_at_frame_start = enemy.wait_hidden_at_frame_start
        if wait_active_at_frame_start and enemy.wait_phase_consumed:
            return
        if not wait_active_at_frame_start and enemy.wait_remaining > 0.0:
            # Direct movement callers may not run the fixed-update wait ticker;
            # a still-positive wait is nevertheless already active for them.
            wait_active_at_frame_start = True
            wait_hidden_at_frame_start = (
                enemy.disappeared or enemy.portal_targetable
            )
            enemy.wait_active_at_frame_start = True
            enemy.wait_hidden_at_frame_start = wait_hidden_at_frame_start
            enemy.wait_phase_consumed = False
        if enemy.portal_targetable and enemy.wait_remaining <= 1e-12:
            # The portal hitbox is active during the empirical extension;
            # only after those frames does the route-visible state reopen.
            enemy.portal_targetable = False
            enemy.disappeared = False
        if self._recover_invalid_terrain(enemy):
            return
        speed = self._effective_enemy_move_speed(enemy)
        if self._apply_force_motion(enemy):
            if wait_active_at_frame_start:
                enemy.wait_phase_consumed = True
            if enemy.dead:
                return
            enemy.remaining_path_distance = self._targeting_path_distance(enemy)
            return
        if wait_active_at_frame_start:
            enemy.wait_phase_consumed = True
            if not wait_hidden_at_frame_start and not enemy.disappeared:
                self._coast_enemy_navigation(enemy)
            elif wait_hidden_at_frame_start and enemy.wait_remaining <= 0.0:
                self._advance_hidden_route_events(enemy)
            return
        if enemy.wait_remaining > 0:
            enemy.wait_phase_consumed = True
            if not enemy.disappeared:
                self._coast_enemy_navigation(enemy)
            return
        if speed <= 0 or not enemy.route:
            return

        budget = speed * self.dt
        while budget > 1e-9 and enemy.route_idx < len(enemy.route) - 1:
            target = enemy.route[enemy.route_idx + 1]
            if not target.reachable:
                enemy.remaining_path_distance = self._targeting_path_distance(enemy)
                self.mechanic_diagnostics.setdefault("PATH_NOT_FOUND:SPFA", 1)
                return
            kind = target.kind.upper()
            if kind in {
                "WAIT_FOR_SECONDS",
                "WAIT_CURRENT_WAVE_TIME",
                "DISAPPEAR",
            }:
                self._reach_route_point(enemy, target)
                if enemy.wait_remaining > 0:
                    if not enemy.disappeared:
                        self._coast_enemy_navigation(enemy)
                    break
                continue
            if kind in {"TELEPORT", "APPEAR_AT_POS"}:
                enemy.position_row, enemy.position_col = target.row, target.col
                if self._recover_invalid_terrain(enemy):
                    return
                self._reach_route_point(enemy, target)
                if enemy.wait_remaining > 0:
                    break
                continue

            if self._route_point_reached(enemy, target):
                self._reach_route_point(enemy, target)
                if enemy.wait_remaining > 0:
                    if not enemy.disappeared:
                        self._coast_enemy_navigation(enemy)
                    break
                continue

            dr = target.row - enemy.position_row
            dc = target.col - enemy.position_col
            distance = math.hypot(dr, dc)
            next_position = self._steered_next_position(
                enemy,
                target=(target.row, target.col),
                speed=speed,
            )
            travel = math.hypot(
                next_position[0] - enemy.position_row,
                next_position[1] - enemy.position_col,
            )
            if travel <= 1e-12:
                return
            terrain_result = self._move_with_terrain(
                enemy,
                next_position,
                forced_goal_cell=(
                    self._position_cell(target.row, target.col)
                    if kind == "MOVE"
                    else None
                ),
                reflect_collision=True,
            )
            if terrain_result != "MOVED":
                return
            budget = 0.0
            if self._route_point_reached(enemy, target):
                self._reach_route_point(enemy, target)
                if enemy.wait_remaining > 0:
                    if not enemy.disappeared:
                        self._coast_enemy_navigation(enemy)
                break

        enemy.remaining_path_distance = self._targeting_path_distance(enemy)
        if not enemy.dead and enemy.route_idx >= len(enemy.route) - 1:
            enemy.leaked = True

    def _advance_hidden_route_events(self, enemy: _Enemy) -> None:
        """Resolve hidden-route control points without visible movement."""
        while enemy.route_idx < len(enemy.route) - 1:
            target = enemy.route[enemy.route_idx + 1]
            kind = target.kind.upper()
            if kind in {
                "WAIT_FOR_SECONDS",
                "WAIT_CURRENT_WAVE_TIME",
                "DISAPPEAR",
            }:
                self._reach_route_point(enemy, target)
                if enemy.wait_remaining > 0:
                    return
                continue
            if kind in {"TELEPORT", "APPEAR_AT_POS"}:
                enemy.position_row, enemy.position_col = target.row, target.col
                if self._recover_invalid_terrain(enemy):
                    return
                self._reach_route_point(enemy, target)
                if enemy.wait_remaining > 0:
                    return
                continue
            return

    def _coast_enemy_navigation(self, enemy: _Enemy) -> None:
        """Integrate one ground navigation coast step during a visible wait."""
        if self._is_airborne(enemy):
            return
        speed = self._effective_enemy_move_speed(enemy)
        if speed <= 0:
            return
        # A visible WAIT still calculates movement with a zero desired
        # direction.  Its per-controller avoidance ticker keeps advancing;
        # hidden waits bypass this call and preserve the ticker phase.
        avoid_row, avoid_col = self._enemy_avoidance(
            enemy, speed, direction=(0.0, 0.0)
        )
        acceleration = self._clamp_vector(
            (
                -enemy.navigation_velocity_row * GROUND_STEERING_FACTOR
                + avoid_row,
                -enemy.navigation_velocity_col * GROUND_STEERING_FACTOR
                + avoid_col,
            ),
            10.0,
        )
        velocity = self._clamp_vector(
            (
                enemy.navigation_velocity_row + acceleration[0] * self.dt,
                enemy.navigation_velocity_col + acceleration[1] * self.dt,
            ),
            speed,
        )
        enemy.navigation_velocity_row, enemy.navigation_velocity_col = velocity
        if math.hypot(
            enemy.navigation_velocity_row,
            enemy.navigation_velocity_col,
        ) <= 1e-12:
            enemy.navigation_velocity_row = 0.0
            enemy.navigation_velocity_col = 0.0
            return
        next_position = (
            enemy.position_row + enemy.navigation_velocity_row * self.dt,
            enemy.position_col + enemy.navigation_velocity_col * self.dt,
        )
        result = self._move_with_terrain(
            enemy,
            next_position,
            forced_goal_cell=None,
            reflect_collision=True,
        )
        if result == "MOVED":
            enemy.remaining_path_distance = self._targeting_path_distance(enemy)

    def _tick_enemy_wait(self, enemy: _Enemy) -> bool:
        """Advance route wait and record whether it was active at frame start."""
        was_waiting = enemy.wait_remaining > 0
        enemy.wait_active_at_frame_start = was_waiting
        enemy.wait_hidden_at_frame_start = was_waiting and (
            enemy.disappeared or enemy.portal_targetable
        )
        enemy.wait_phase_consumed = False
        if was_waiting:
            remaining = max(0.0, enemy.wait_remaining - self.dt)
            # Keep the ordinary visible-route boundary on the intended
            # floating-point clock, but do not let a tiny subtraction tail
            # postpone the next MOVE frame.  Hidden/portal waits deliberately
            # keep the strict countdown because H7-2 observes 301 frames for a
            # nominal 10-second portal wait.
            if not enemy.disappeared:
                epsilon = max(
                    0.0,
                    float(
                        getattr(
                            getattr(self, "assumptions", DEFAULT_ASSUMPTIONS),
                            "route_wait_expiry_epsilon",
                            DEFAULT_ASSUMPTIONS.route_wait_expiry_epsilon,
                        )
                    ),
                )
                if remaining <= epsilon:
                    remaining = 0.0
                    if epsilon:
                        self.used_assumptions.add(
                            "ROUTE_WAIT_EXPIRY_EPSILON"
                        )
            enemy.wait_remaining = remaining
        return was_waiting

    @staticmethod
    def _route_point_reached(enemy: _Enemy, point: RoutePoint) -> bool:
        kind = point.kind.upper()
        if kind == "NAVIGATE_CELL":
            return enemy.tile() == Battle._position_cell(point.row, point.col)
        if kind == "NAVIGATE_STABLE":
            radius = max(point.reach_distance, 0.25)
        elif kind == "NAVIGATE_CENTER":
            radius = max(point.reach_distance, 0.05)
        else:
            radius = max(point.reach_distance, 0.05)
        return (
            math.hypot(
                point.row - enemy.position_row,
                point.col - enemy.position_col,
            )
            <= radius + 1e-9
        )

    def _steered_next_position(
        self,
        enemy: _Enemy,
        *,
        target: tuple[float, float],
        speed: float,
    ) -> tuple[float, float]:
        dr = target[0] - enemy.position_row
        dc = target[1] - enemy.position_col
        distance = math.hypot(dr, dc)
        if distance <= speed * self.dt + 1e-12:
            return target

        desired_row = dr / distance * speed
        desired_col = dc / distance * speed
        steering_enabled = getattr(self, "steering_enabled", True)
        if steering_enabled:
            steering_factor = (
                20.0
                if self._is_airborne(enemy)
                else GROUND_STEERING_FACTOR
            )
            max_force = 100.0 if self._is_airborne(enemy) else 10.0
        else:
            steering_factor = 100.0
            max_force = 100.0

        direction = (dr / distance, dc / distance)
        avoid_row, avoid_col = self._enemy_avoidance(
            enemy,
            speed,
            direction=direction,
        )
        acceleration = self._clamp_vector(
            (
                (desired_row - enemy.navigation_velocity_row)
                * steering_factor
                + avoid_row,
                (desired_col - enemy.navigation_velocity_col)
                * steering_factor
                + avoid_col,
            ),
            max_force,
        )
        velocity = self._clamp_vector(
            (
                enemy.navigation_velocity_row + acceleration[0] * self.dt,
                enemy.navigation_velocity_col + acceleration[1] * self.dt,
            ),
            speed,
        )
        enemy.navigation_velocity_row, enemy.navigation_velocity_col = velocity
        return (
            enemy.position_row + velocity[0] * self.dt,
            enemy.position_col + velocity[1] * self.dt,
        )

    @staticmethod
    def _clamp_vector(
        vector: tuple[float, float], maximum: float
    ) -> tuple[float, float]:
        length = math.hypot(*vector)
        if length <= maximum or length <= 1e-12:
            return vector
        scale = maximum / length
        return vector[0] * scale, vector[1] * scale

    def _enemy_avoidance(
        self,
        enemy: _Enemy,
        theoretical_speed: float,
        *,
        direction: tuple[float, float],
    ) -> tuple[float, float]:
        if (
            self._is_airborne(enemy)
            or not getattr(self, "steering_enabled", True)
            or not hasattr(self, "map_grid")
        ):
            return 0.0, 0.0
        # MoveController owns a PeriodicTicker(3, false): Tick decrements a
        # positive countdown, and Next reloads it only after a refresh.
        # Advance on movement-force calls, not shared battle-frame buckets.
        if enemy.avoidance_tick_remaining > 0:
            enemy.avoidance_tick_remaining -= 1
        if enemy.avoidance_tick_remaining == 0:
            base_row, base_col = self._base_enemy_avoidance(enemy)
            projection = (
                base_row * direction[0] + base_col * direction[1]
            )
            enemy.avoidance_row = base_row - projection * direction[0]
            enemy.avoidance_col = base_col - projection * direction[1]
            enemy.avoidance_tick_remaining = 3
        inertia = math.hypot(
            enemy.navigation_velocity_row,
            enemy.navigation_velocity_col,
        )
        scale = max(
            inertia / max(theoretical_speed, 1e-9),
            0.5,
        )
        return enemy.avoidance_row * scale, enemy.avoidance_col * scale

    def _base_enemy_avoidance(self, enemy: _Enemy) -> tuple[float, float]:
        center_row, center_col = enemy.tile()
        total_row = 0.0
        total_col = 0.0
        foot_row = (
            enemy.position_row + self.assumptions.steering_foot_offset_row
        )
        foot_col = (
            enemy.position_col + self.assumptions.steering_foot_offset_col
        )
        half_body_width = max(
            float(self.assumptions.steering_half_body_width),
            0.0,
        )
        for relative_row in (-1, 0, 1):
            for relative_col in (-1, 0, 1):
                if relative_row == relative_col == 0:
                    continue
                row = center_row + relative_row
                col = center_col + relative_col
                cost = self._ground_path_cost(row, col)
                if cost <= 1.0 or not self.tile(row, col):
                    continue
                nearest_row = foot_row
                nearest_col = foot_col + (
                    half_body_width
                    if relative_col > 0
                    else -half_body_width
                    if relative_col < 0
                    else 0.0
                )
                effective_row = (
                    max(
                        (nearest_row - center_row) * relative_row,
                        0.0,
                    )
                    - 0.25
                ) * abs(relative_row)
                effective_col = (
                    max(
                        (nearest_col - center_col) * relative_col,
                        0.0,
                    )
                    - 0.25
                ) * abs(relative_col)
                if relative_row == 0 or relative_col == 0:
                    if effective_row > 0 or effective_col > 0:
                        total_row -= effective_row * relative_row
                        total_col -= effective_col * relative_col
                elif effective_row > 0 and effective_col > 0:
                    strength = (effective_row + effective_col) * 0.5
                    total_row -= strength * relative_row
                    total_col -= strength * relative_col

        length = math.hypot(total_row, total_col)
        if length <= 1e-12:
            return 0.0, 0.0
        self.used_assumptions.add("STEERING_FOOT_POSITION")
        return total_row / length, total_col / length

    def _reach_route_point(self, enemy: _Enemy, point: RoutePoint) -> None:
        enemy.route_idx += 1
        enemy.tile_idx = min(enemy.route_idx, max(len(enemy.path) - 1, 0))
        kind = point.kind.upper()
        if kind == "WAIT_CURRENT_WAVE_TIME":
            elapsed = self.time - enemy.wave_start_time
            enemy.wait_remaining = max(0.0, point.wait_time - elapsed)
        elif point.wait_time > 0:
            wait_time = point.wait_time
            enemy.wait_remaining = wait_time
        else:
            enemy.wait_remaining = 0.0
        if kind == "DISAPPEAR":
            enemy.disappeared = True
            enemy.portal_targetable = False
            self._release_enemy_block(enemy)
        elif kind in {"APPEAR_AT_POS", "TELEPORT"}:
            was_disappeared = enemy.disappeared
            if kind == "APPEAR_AT_POS" and was_disappeared:
                extra_frames = max(
                    0,
                    int(
                        getattr(
                            getattr(self, "assumptions", DEFAULT_ASSUMPTIONS),
                            "portal_wait_extra_frames",
                            DEFAULT_ASSUMPTIONS.portal_wait_extra_frames,
                        )
                    ),
                )
                if extra_frames:
                    self.used_assumptions.add("PORTAL_WAIT_EXTRA_FRAMES")
                    # The hitbox is already targetable, while the visible route
                    # state remains in the portal for the extra frames.
                    enemy.disappeared = True
                    enemy.portal_targetable = True
                    enemy.wait_remaining = (
                        extra_frames / CLIENT_LOGIC_RATE
                    )
                    return
            enemy.disappeared = False
            enemy.portal_targetable = False

    def _update_blocking(self) -> None:
        """Historical grouped scan and an explicit helper for isolated checks."""
        for op in sorted(self.operators, key=lambda unit: unit.creation_index):
            self._update_operator_blocking(op)

    def _update_operator_blocking(self, op: _Operator) -> None:
        frame_index = getattr(self, "frame_index", 0)
        deployment_cycle = (
            self.assumptions.operator_block_scan_from_deploy_frame
        )
        if (
            not deployment_cycle
            and (frame_index - 1) % BLOCK_SCAN_INTERVAL_FRAMES != 0
        ):
            return
        if op.dead or not op.statuses.can_block or op.block_count <= 0:
            return
        if deployment_cycle:
            if op.block_scan_next_frame is None:
                # Without a deployment hook, treat this as the first
                # post-movement scan following deployment.
                op.block_scan_next_frame = frame_index
            if frame_index < op.block_scan_next_frame:
                return
            op.block_scan_next_frame = (
                frame_index + BLOCK_SCAN_INTERVAL_FRAMES
            )
            self.used_assumptions.add(
                "OPERATOR_BLOCK_SCAN_DEPLOYMENT_PHASE"
            )
        free_slots = op.block_count - len(op.blocked)
        if free_slots <= 0:
            # The scheduled cycle has already advanced while full.
            return
        candidates = [
            enemy
            for enemy in self.enemies
            if not enemy.dead
            and not enemy.leaked
            and not enemy.disappeared
            and enemy.blocked_by is None
            and not enemy.unblockable
            and not self._is_airborne(enemy)
            and math.hypot(
                op.tile[0] - enemy.position_row,
                op.tile[1] - enemy.position_col,
            )
            <= op.block_radius + 1e-9
        ]
        candidates.sort(
            key=lambda enemy: (
                math.hypot(
                    op.tile[0] - enemy.position_row,
                    op.tile[1] - enemy.position_col,
                ),
                enemy.creation_index,
            )
        )
        for enemy in candidates[:free_slots]:
            self._start_blocking(op, enemy)

    def _start_blocking(self, op: _Operator, enemy: _Enemy) -> None:
        if enemy.unblockable:
            return
        enemy.blocked_by = op
        other_enemies = list(op.blocked)
        op.blocked.append(enemy)
        self.used_assumptions.add("REVERSE_ENGINEERED_BLOCKING")

        ab_row = enemy.position_row - op.tile[0]
        ab_col = enemy.position_col - op.tile[1]
        ab_length = math.hypot(ab_row, ab_col)
        if ab_length <= 0.00001:
            enemy.block_stable_row = enemy.position_row
            enemy.block_stable_col = enemy.position_col
            return

        ac_length = max(ab_length, BLOCK_OFFSET_DISTANCE)
        ac_row = ab_row / ab_length * ac_length
        ac_col = ab_col / ab_length * ac_length
        c_row = op.tile[0] + ac_row
        c_col = op.tile[1] + ac_col
        stable_positions = [
            (
                other.block_stable_row
                if other.block_stable_row is not None
                else other.position_row,
                other.block_stable_col
                if other.block_stable_col is not None
                else other.position_col,
            )
            for other in other_enemies
        ]

        correction_row = 0.0
        correction_col = 0.0
        if any(
            math.hypot(c_row - row, c_col - col) <= 0.1 + 1e-9
            for row, col in stable_positions
        ):
            for d_row, d_col in stable_positions:
                dc_row = c_row - d_row
                dc_col = c_col - d_col
                dc_length = math.hypot(dc_row, dc_col)
                if dc_length < 0.1:
                    ad_row = d_row - op.tile[0]
                    ad_col = d_col - op.tile[1]
                    ad_length = math.hypot(ad_row, ad_col)
                    if ad_length > 1e-12:
                        correction_row += -ad_col / ad_length
                        correction_col += ad_row / ad_length
                elif dc_length < 0.4:
                    scale = (0.4 - dc_length) * 50.0 / dc_length
                    correction_row += dc_row * scale
                    correction_col += dc_col * scale

        correction_length = math.hypot(correction_row, correction_col)
        if correction_length > 1e-12:
            correction_row *= 0.2 / correction_length
            correction_col *= 0.2 / correction_length
            ae_row = ac_row + correction_row
            ae_col = ac_col + correction_col
            ae_length = math.hypot(ae_row, ae_col)
            if ae_length > 1e-12:
                ac_row = ae_row / ae_length * ac_length
                ac_col = ae_col / ae_length * ac_length

        target_row = op.tile[0] + ac_row
        target_col = op.tile[1] + ac_col
        enemy.block_stable_row = target_row
        enemy.block_stable_col = target_col
        enemy.block_shift_start_row = None
        enemy.block_shift_start_col = None
        enemy.block_shift_elapsed = 0.0
        if (
            math.hypot(
                target_row - enemy.position_row,
                target_col - enemy.position_col,
            )
            > 1e-12
        ):
            enemy.block_shift_target_row = target_row
            enemy.block_shift_target_col = target_col
            enemy.block_shift_remaining = BLOCK_OFFSET_DURATION
            enemy.block_shift_start_row = enemy.position_row
            enemy.block_shift_start_col = enemy.position_col

    def _apply_block_shift(self, enemy: _Enemy) -> bool:
        if (
            enemy.blocked_by is None
            or enemy.block_shift_remaining <= 1e-12
            or enemy.block_shift_target_row is None
            or enemy.block_shift_target_col is None
            or enemy.block_shift_start_row is None
            or enemy.block_shift_start_col is None
        ):
            return False
        # The runtime trajectory follows a quadratic ease-out: the shift
        # starts at its highest speed and decelerates to the stable point.
        elapsed = min(
            BLOCK_OFFSET_DURATION,
            max(0.0, enemy.block_shift_elapsed + self.dt),
        )
        normalized = elapsed / BLOCK_OFFSET_DURATION
        progress = 1.0 - (1.0 - normalized) ** 2
        target = (
            enemy.block_shift_start_row
            + (enemy.block_shift_target_row - enemy.block_shift_start_row)
            * progress,
            enemy.block_shift_start_col
            + (enemy.block_shift_target_col - enemy.block_shift_start_col)
            * progress,
        )
        result = self._move_with_terrain(enemy, target)
        enemy.block_shift_elapsed = elapsed
        enemy.block_shift_remaining = max(0.0, BLOCK_OFFSET_DURATION - elapsed)
        if result != "MOVED" or enemy.block_shift_remaining <= 1e-12:
            enemy.block_shift_remaining = 0.0
            enemy.block_shift_target_row = None
            enemy.block_shift_target_col = None
            enemy.block_shift_start_row = None
            enemy.block_shift_start_col = None
            if not enemy.dead:
                enemy.block_stable_row = enemy.position_row
                enemy.block_stable_col = enemy.position_col
        return True

    def apply_displacement(
        self,
        enemy: _Enemy,
        velocity_row: float,
        velocity_col: float,
        *,
        deceleration: float | None = None,
    ) -> bool:
        if enemy.dead or enemy.disappeared or self._is_airborne(enemy):
            return False
        added_row = float(velocity_row)
        added_col = float(velocity_col)
        if math.hypot(added_row, added_col) <= 0:
            return False
        enemy.force_velocity_row += added_row
        enemy.force_velocity_col += added_col
        enemy.force_deceleration = (
            DEFAULT_FRICTION_DECELERATION
            if deceleration is None
            else max(float(deceleration), 0.0)
        )
        enemy.unbalanced = True
        enemy.unbalance_lock_remaining = max(
            enemy.unbalance_lock_remaining, UNBALANCE_MIN_DURATION
        )
        self._release_enemy_block(enemy)
        return True

    @staticmethod
    def _effective_force_level(enemy: _Enemy, force_level: int) -> int:
        return max(-3, min(3, int(force_level) - enemy.mass_level))

    def apply_push(
        self,
        enemy: _Enemy,
        direction_row: float,
        direction_col: float,
        force_level: int,
    ) -> bool:
        direction_length = math.hypot(direction_row, direction_col)
        if direction_length <= 1e-9:
            return False
        effective_level = self._effective_force_level(enemy, force_level)
        if effective_level <= -3:
            return False
        speed = PUSH_SPEED_BY_FORCE_LEVEL[effective_level]
        return self.apply_displacement(
            enemy,
            direction_row / direction_length * speed,
            direction_col / direction_length * speed,
        )

    def apply_pull(
        self,
        enemy: _Enemy,
        origin_row: float,
        origin_col: float,
        force_level: int,
    ) -> bool:
        if enemy.dead or enemy.disappeared or self._is_airborne(enemy):
            return False
        effective_level = self._effective_force_level(enemy, force_level)
        if effective_level <= -3:
            return False
        distance = math.hypot(
            origin_row - enemy.position_row,
            origin_col - enemy.position_col,
        )
        if distance <= 1e-9:
            return False
        if enemy.pull_remaining > 0:
            self.mechanic_diagnostics.setdefault("PULL_STACK:ONE_SOURCE_APPROX", 1)
        enemy.pull_origin_row = float(origin_row)
        enemy.pull_origin_col = float(origin_col)
        enemy.pull_initial_distance = distance
        enemy.pull_force = PULL_FORCE_BY_FORCE_LEVEL[effective_level]
        enemy.pull_remaining = 0.5 if effective_level < -1 else 1.0
        enemy.unbalanced = True
        enemy.unbalance_lock_remaining = max(
            enemy.unbalance_lock_remaining, UNBALANCE_MIN_DURATION
        )
        self._release_enemy_block(enemy)
        return True

    @staticmethod
    def _release_enemy_block(enemy: _Enemy) -> None:
        if enemy.blocked_by is not None:
            if enemy in enemy.blocked_by.blocked:
                enemy.blocked_by.blocked.remove(enemy)
            enemy.blocked_by = None
        enemy.block_stable_row = None
        enemy.block_stable_col = None
        enemy.block_shift_target_row = None
        enemy.block_shift_target_col = None
        enemy.block_shift_start_row = None
        enemy.block_shift_start_col = None
        enemy.block_shift_elapsed = 0.0
        enemy.block_shift_remaining = 0.0

    def _apply_force_motion(self, enemy: _Enemy) -> bool:
        if self._is_airborne(enemy):
            self._end_unbalance(enemy)
            return False
        if enemy.pull_remaining > 0:
            force_dt = min(self.dt, enemy.pull_remaining)
            dr = enemy.pull_origin_row - enemy.position_row
            dc = enemy.pull_origin_col - enemy.position_col
            distance = math.hypot(dr, dc)
            if distance > 1e-9 and enemy.pull_initial_distance > 1e-9:
                force = enemy.pull_force * (
                    distance / enemy.pull_initial_distance
                ) ** 4
                enemy.force_velocity_row += dr / distance * force * force_dt
                enemy.force_velocity_col += dc / distance * force * force_dt
            enemy.pull_remaining = max(0.0, enemy.pull_remaining - self.dt)
        speed = math.hypot(enemy.force_velocity_row, enemy.force_velocity_col)
        if not enemy.unbalanced and speed <= 1e-9:
            return False
        enemy.unbalanced = True
        enemy.unbalance_lock_remaining = max(
            0.0, enemy.unbalance_lock_remaining - self.dt
        )
        if speed > 1e-9:
            end = (
                enemy.position_row + enemy.force_velocity_row * self.dt,
                enemy.position_col + enemy.force_velocity_col * self.dt,
            )
            terrain_result = self._move_with_terrain(enemy, end)
            if terrain_result == "FELL":
                self._end_unbalance(enemy)
                return True
            if terrain_result == "COLLISION":
                speed = 0.0
            elif enemy.force_deceleration > 0:
                new_speed = max(0.0, speed - enemy.force_deceleration * self.dt)
                scale = new_speed / speed
                enemy.force_velocity_row *= scale
                enemy.force_velocity_col *= scale
                speed = new_speed
        if (
            speed <= UNBALANCE_STOP_SPEED
            and enemy.pull_remaining <= 1e-9
            and enemy.unbalance_lock_remaining <= 1e-9
        ):
            self._end_unbalance(enemy)
        return True

    @staticmethod
    def _end_unbalance(enemy: _Enemy) -> None:
        enemy.unbalanced = False
        enemy.unbalance_lock_remaining = 0.0
        enemy.force_velocity_row = 0.0
        enemy.force_velocity_col = 0.0
        enemy.pull_force = 0.0
        enemy.pull_remaining = 0.0
        enemy.pull_initial_distance = 0.0

    def _targeting_path_distance(self, enemy: _Enemy) -> float:
        """Estimate distToExit from checkpoint grids and their next nodes.

        Native MoveCheckpoint/Route distance getters project the locator
        toward the grid center along Node.GetSafeNextDirection, not velocity.
        A checkpoint's terminal node has no nextStep, so its projection is
        zero even while moving toward a reachOffset inside that grid. Ignore
        generated navigation guides; they are not native checkpoints.
        Teleports have no distance cost and unreachable legs keep the
        straight-line fallback. The reachable result uses float32 arithmetic
        before HATRED_DES converts it to FP.
        """
        route = enemy.route
        index = enemy.route_idx
        if not route or index >= len(route) - 1:
            return 0.0

        def f32(value: float) -> float:
            return struct.unpack("<f", struct.pack("<f", value))[0]

        non_spatial = {
            "WAIT_FOR_SECONDS",
            "WAIT_CURRENT_WAVE_TIME",
            "DISAPPEAR",
            "NAVIGATE_CELL",
            "NAVIGATE_STABLE",
            "NAVIGATE_CENTER",
        }
        teleport = {"APPEAR_AT_POS", "TELEPORT"}
        row, col = enemy.position()
        start_cell = self._position_cell(row, col)
        start_position = (row, col)
        total = 0.0
        projection = 0.0
        have_spatial_leg = False
        projection_pending = True

        for point in route[index + 1 :]:
            kind = point.kind.upper()
            if kind in non_spatial:
                continue
            if kind in teleport:
                # A portal hop itself has no path-distance cost.
                start_cell = self._position_cell(point.row, point.col)
                start_position = (point.row, point.col)
                if not have_spatial_leg:
                    projection_pending = False
                continue

            target_cell = point.path_goal or self._position_cell(point.row, point.col)
            source_tile = self.tile(*start_cell)
            target_tile = self.tile(*target_cell)
            reachable = (
                point.reachable
                and bool(source_tile)
                and bool(target_tile)
                and self._ground_passable_cell(*start_cell)
                and (kind == "MOVE" or self._ground_passable_cell(*target_cell))
            )
            cells = (
                self._spfa_cells(start_cell, target_cell, force_goal=kind == "MOVE")
                if reachable
                else None
            )
            if cells is None:
                leg_distance = math.hypot(
                    point.row - start_position[0],
                    point.col - start_position[1],
                )
                if not have_spatial_leg:
                    return leg_distance
                return total + leg_distance + projection

            if projection_pending:
                if len(cells) > 1:
                    nodes = (
                        self._smooth_cell_path(
                            cells, goal=target_cell, force_goal=kind == "MOVE"
                        )
                        if point.allow_diagonal else cells
                    )
                    dr = nodes[1][0] - start_cell[0]
                    dc = nodes[1][1] - start_cell[1]
                    length = f32(math.sqrt(f32(dr * dr + dc * dc)))
                    direction_row = f32(dr / length)
                    direction_col = f32(dc / length)
                    projection = f32(
                        f32(f32(start_cell[0] - f32(row)) * direction_row)
                        + f32(f32(start_cell[1] - f32(col)) * direction_col)
                    )
                projection_pending = False

            total += max(len(cells) - 1, 0)
            have_spatial_leg = True
            start_cell = target_cell
            start_position = (point.row, point.col)

        return f32(total + projection)

    def _cleanup(self) -> None:
        for enemy in list(self.enemies):
            if enemy.leaked and not enemy.dead:
                self.life_points -= enemy.life_reduce
                self.enemies_leaked += 1
                enemy.dead = True
            elif enemy.dead and enemy.blocked_by:
                self._release_enemy_block(enemy)
        for op in self.operators:
            if op.dead:
                self._process_operator_departure(op, retreated=op.retreated)
                for enemy in list(op.blocked):
                    self._release_enemy_block(enemy)
        self.enemies = [e for e in self.enemies if not e.dead]

    # ------------------------------------------------------- generic mechanics
    def apply_status(
        self,
        target: _Enemy | _Operator,
        status: Status | str,
        duration: float | None,
        *,
        resistible: bool = True,
        source: StatusSource | str = StatusSource.FRIENDLY,
    ) -> Status | None:
        effect = Status(str(getattr(status, "value", status)).upper())
        if effect is Status.LEVITATE:
            if isinstance(target, _Operator):
                return None
            if target.motion.upper() in {"FLY", "FLYING"}:
                return None
            if duration is not None and target.mass_level > 3:
                duration *= 0.5
            self._end_unbalance(target)
        return target.statuses.apply(
            effect,
            duration,
            resistible=resistible,
            source=source,
        )

    def add_damage_rule(
        self, target: _Enemy | _Operator, rule: DamageRule
    ) -> DamageRule:
        self.damage_rule_counter += 1
        rule.acquired_order = self.damage_rule_counter
        target.damage_rules.append(rule)
        return rule

    def apply_element_damage(
        self,
        target: _Enemy | _Operator,
        element: ElementType | str,
        amount: float,
        *,
        neutral: bool = False,
        source: _Enemy | _Operator | None = None,
    ) -> ElementResult:
        result = target.elements.apply(
            element,
            amount,
            neutral=neutral,
            immune=target.statuses.has(Status.ELEMENT_IMMUNE),
        )
        if result.burst:
            self._start_element_burst(target, result, source)
        return result

    def _start_element_burst(
        self,
        target: _Enemy | _Operator,
        result: ElementResult,
        source: _Enemy | _Operator | None,
    ) -> None:
        element = result.element
        target.element_burst = ElementBurstRuntime(
            element=element,
            duration=result.burst_duration,
            source=source,
        )
        enemy_target = isinstance(target, _Enemy)

        if element is ElementType.SANITY:
            if enemy_target:
                target.palsy_stacks = 3
                self._apply_element_burst_damage(
                    target, 6000.0, DamageType.ELEMENTAL, source
                )
            else:
                target.statuses.apply(
                    Status.STUNNED, result.burst_duration, resistible=True
                )
                self._apply_element_burst_damage(
                    target, 1000.0, DamageType.TRUE, source
                )
        elif element is ElementType.WATER:
            target.defense -= 120.0 if enemy_target else 100.0
            self._apply_element_burst_damage(
                target,
                5000.0 if enemy_target else 800.0,
                DamageType.ELEMENTAL if enemy_target else DamageType.PHYSICAL,
                source,
            )
        elif element is ElementType.FIRE:
            self._apply_element_burst_damage(
                target,
                7000.0 if enemy_target else 1200.0,
                DamageType.ELEMENTAL if enemy_target else DamageType.ARTS,
                source,
            )
        elif element is ElementType.DARK and not enemy_target:
            target.statuses.apply(
                Status.SP_BLOCKED, result.burst_duration, resistible=False
            )
            target.statuses.apply(
                Status.SKILL_NOT_ACTIVATABLE,
                result.burst_duration,
                resistible=False,
            )

    def _tick_element_burst(self, target: _Enemy | _Operator) -> None:
        runtime = target.element_burst
        if runtime is None or target.elements.cooldown_remaining <= 0:
            return
        runtime.tick_elapsed += self.dt
        while runtime.tick_elapsed + 1e-9 >= 1.0 and not target.dead:
            runtime.tick_elapsed -= 1.0
            if runtime.element is ElementType.DARK:
                if isinstance(target, _Enemy):
                    self._apply_element_burst_damage(
                        target, 800.0, DamageType.ELEMENTAL, runtime.source
                    )
                else:
                    target.sp = max(0.0, target.sp - 1.0)
                    self._apply_element_burst_damage(
                        target, 100.0, DamageType.ARTS, runtime.source
                    )
            elif runtime.element is ElementType.ANGER and isinstance(
                target, _Operator
            ):
                self._apply_element_burst_damage(
                    target, runtime.anger_damage, DamageType.TRUE, runtime.source
                )

    def _apply_element_burst_damage(
        self,
        target: _Enemy | _Operator,
        amount: float,
        damage_type: DamageType,
        source: _Enemy | _Operator | None,
    ) -> None:
        result = apply_damage(
            DamagePacket(
                amount=amount,
                damage_type=damage_type,
                attack_type=AttackType.NORMAL,
                apply_way=ApplyWay.NONE,
                source=source,
                target_resistance_override=self._effective_resistance(target),
            ),
            target,
            roll=self._roll_probability,
        )
        if result.killed and isinstance(target, _Enemy):
            self.enemies_killed += 1

    @staticmethod
    def _effective_resistance(target: _Enemy | _Operator) -> float:
        runtime = getattr(target, "element_burst", None)
        statuses = getattr(target, "statuses", None)
        reduction = (
            15.0
            if statuses is not None and statuses.has(Status.FROZEN_FRIENDLY)
            else 0.0
        )
        reduction += (
            20.0
            if runtime is not None and runtime.element is ElementType.FIRE
            else 0.0
        )
        return float(getattr(target, "res", 0.0)) - reduction

    @staticmethod
    def _outgoing_damage_scale(source: _Enemy | _Operator) -> float:
        runtime = getattr(source, "element_burst", None)
        if (
            runtime is None
            or runtime.element is not ElementType.DARK
            or not getattr(getattr(source, "elements", None), "enemy_unit", False)
        ):
            return 1.0
        remaining_ratio = max(
            0.0,
            min(source.elements.cooldown_remaining / runtime.duration, 1.0),
        )
        return 1.0 - 0.5 * remaining_ratio

    @staticmethod
    def _register_ability_use(unit: _Enemy | _Operator) -> None:
        runtime = unit.element_burst
        if (
            runtime is not None
            and runtime.element is ElementType.ANGER
            and not unit.elements.enemy_unit
        ):
            runtime.anger_damage = min(runtime.anger_damage + 50.0, 600.0)

    # ------------------------------------------------------ behavior adapter
    def behavior_deal_damage(
        self,
        source: _Enemy | _Operator,
        target: _Enemy | _Operator,
        amount: float,
        damage_type: str,
        *,
        undeadable: bool = False,
    ) -> bool:
        """Apply one behavior-node damage instance through the core formula."""
        if (
            target is None
            or target.dead
            or (
                isinstance(target, _Enemy)
                and not self._enemy_is_attackable(target)
            )
        ):
            return False
        result = apply_damage(
            DamagePacket(
                amount=amount,
                damage_type=damage_type,
                attack_type=AttackType.ADDITION,
                apply_way=ApplyWay.NONE,
                lethal=not undeadable,
                output_scale=self._outgoing_damage_scale(source),
                source=source,
                target_resistance_override=self._effective_resistance(target),
            ),
            target,
            roll=self._roll_probability,
        )
        if result.killed and isinstance(target, _Enemy):
            self.enemies_killed += 1
        return True

    def behavior_set_timed_attribute_scale(
        self,
        context: BehaviorContext,
        attribute_type: str,
        progress: float,
    ) -> bool:
        """Update a timed skill multiplier without compounding each trigger."""
        op = context.source
        if not isinstance(op, _Operator):
            return False
        progress = max(0.0, min(float(progress), 1.0))
        attribute = attribute_type.upper()
        if attribute == "ATK":
            new_scale = 1.0 + float(context.blackboard.get("atk", 0.0)) * progress
            op.atk_scale = op.atk_scale / op.timed_atk_scale * new_scale
            op.timed_atk_scale = new_scale
            return True
        if attribute == "DEF":
            new_scale = 1.0 + float(context.blackboard.get("def", 0.0)) * progress
            op.def_scale = op.def_scale / op.timed_def_scale * new_scale
            op.timed_def_scale = new_scale
            return True
        key = f"RemainingRatioToAttributeModifier:{attribute or '<missing>'}"
        self.behavior_diagnostics.unsupported_nodes[key] += 1
        if self.behavior.strict:
            raise BehaviorError(f"unsupported timed attribute {attribute!r}")
        return True

    @staticmethod
    def _front_3x3_cells(op: _Operator) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for side in range(-1, 2):
            for forward in range(1, 4):
                if op.facing == 0:  # right
                    dr, dc = side, forward
                elif op.facing == 1:  # down
                    dr, dc = forward, -side
                elif op.facing == 2:  # left
                    dr, dc = -side, -forward
                else:  # up
                    dr, dc = -forward, side
                cells.add((op.tile[0] + dr, op.tile[1] + dc))
        return cells

    def behavior_aoe_damage(
        self, context: BehaviorContext, node: dict[str, Any]
    ) -> bool:
        source = context.entity(node.get("_sourceType"))
        center = context.entity(node.get("_targetType"))
        if not isinstance(source, _Operator) or center is None:
            return False

        known_huang_range = (
            context.template_key == "huang_s_3"
            and not node.get("_useRadius")
            and not node.get("_useAbilitySelector")
        )
        # Other ability selectors are not exported in this first slice. Keep
        # those approximations visible instead of silently claiming accuracy.
        if (
            not known_huang_range
            and not node.get("_useRadius")
            and not node.get("_useAbilitySelector")
        ):
            key = "AOEDamage:ability_range_approximation"
            self.behavior_diagnostics.unsupported_nodes[key] += 1
            if self.behavior.strict:
                raise BehaviorError(
                    "AOEDamage requires an ability-range selector not yet modeled"
                )

        if node.get("_useRadius"):
            radius = float(node.get("_radius", 0.0))
            center_tile = center.tile if isinstance(center, _Operator) else center.tile()
            targets = [
                enemy
                for enemy in self.enemies
                if not enemy.dead
                and math.hypot(
                    enemy.tile()[0] - center_tile[0],
                    enemy.tile()[1] - center_tile[1],
                )
                <= radius
            ]
        elif known_huang_range:
            cells = self._front_3x3_cells(source)
            targets = [
                enemy
                for enemy in self.enemies
                if not enemy.dead and enemy.tile() in cells
            ]
        else:
            cells = set(source.range_cells) | {source.tile}
            targets = [
                enemy
                for enemy in self.enemies
                if not enemy.dead and enemy.tile() in cells
            ]

        scale_key = str(node.get("_damageScale", "atk_scale"))
        scale = float(context.blackboard.get(scale_key, 1.0))
        amount = source.atk * source.atk_scale * scale
        hit = False
        for enemy in list(targets):
            hit = self.behavior_deal_damage(
                source,
                enemy,
                amount,
                str(node.get("_damageType", "PHYSICAL")),
            ) or hit
        return hit

    def behavior_create_buff(
        self, context: BehaviorContext, node: dict[str, Any]
    ) -> bool:
        owner = context.entity(node.get("_buffOwner"))
        spec = node.get("_buff", {})
        key = str(spec.get("buffKey", ""))
        if owner is None or not key:
            return False

        container = getattr(owner, "buffs", None)
        if context.buff_container is not None and container is not None:
            definition = BuffDefinition.from_create_spec(
                spec, blackboard=context.blackboard
            )
            source = context.source
            if node.get("_useSpecialBuffSource"):
                source = context.entity(node.get("_specialBuffSource"))
            container.add(
                definition,
                source=source,
                parent=context.buff_instance,
                blackboard=context.blackboard,
                attached_step=self.frame_index,
            )
            return True

        modifiers = spec.get("attributes", {}).get("attributeModifiers", [])
        if modifiers:
            diagnostic = "CreateBuff:attributeModifiers"
            self.behavior_diagnostics.unsupported_nodes[diagnostic] += 1
            if self.behavior.strict:
                raise BehaviorError(
                    "CreateBuff attribute modifiers are not yet modeled"
                )
        owner.buff_blackboards[key] = dict(context.blackboard)

        template_key = str(spec.get("templateKey", ""))
        if template_key and template_key != "empty" and template_key in self.behavior.templates:
            nested = BehaviorContext(
                source=context.source,
                target=context.target,
                buff_owner=owner,
                buff_source=context.source,
                blackboard=owner.buff_blackboards[key],
                ability_blackboard=context.ability_blackboard,
                battle=self,
                diagnostics=self.behavior_diagnostics,
                current_buff_key=key,
            )
            self.behavior.dispatch(template_key, "ON_BUFF_START", nested)

        if spec.get("lifeTimeType") == "IMMEDIATELY":
            owner.buff_blackboards.pop(key, None)
        return True

    def behavior_finish_buff(
        self, owner: _Enemy | _Operator | None, key: str
    ) -> bool:
        if owner is None:
            return False
        container = getattr(owner, "buffs", None)
        if container is not None and hasattr(container, "finish_by_key"):
            return bool(container.finish_by_key(key))
        boards = getattr(owner, "buff_blackboards", {})
        if key not in boards:
            return False
        del boards[key]
        return True

    def behavior_switch_mode(
        self, context: BehaviorContext, node: dict[str, Any]
    ) -> bool:
        target = context.entity(node.get("_targetType"))
        if target is None:
            target = context.buff_owner
        if target is None:
            return False
        mode_index: Any = node.get("_modeIndex", 0)
        if node.get("_loadModeFromBlackboard"):
            key = str(node.get("_modeBlackboardKey", "mode_index"))
            mode_index = context.blackboard.get(key, mode_index)
        if node.get("_restoreDefault"):
            mode_index = 0
        try:
            target.mode_index = int(mode_index)
        except (TypeError, ValueError):
            return False
        return True

    # ------------------------------------------------------------- deploy
    def _execute_plan_action(self, plan: OperatorPlan) -> str:
        if plan.action == "RETREAT":
            return self._try_retreat(plan.char_id)
        if plan.action != "DEPLOY":
            return "skip"
        return self._try_deploy(plan)

    def _active_operator(self, char_id: str) -> _Operator | None:
        for op in reversed(self.operators):
            if op.char_id == char_id and not op.dead:
                return op
        return None

    def _try_retreat(self, char_id: str) -> str:
        op = self._active_operator(char_id)
        if op is None:
            return "skip"
        op.retreated = True
        op.dead = True
        self._process_operator_departure(op, retreated=True)
        return "ok"

    def _process_operator_departure(
        self, op: _Operator, *, retreated: bool
    ) -> None:
        if op.departure_processed:
            return
        op.departure_processed = True
        if retreated:
            refund = math.floor(min(op.deployed_cost * 0.5, op.base_cost))
            self.dp = min(self.max_cost, self.dp + refund)
        if op.target_selector is not None:
            op.target_selector.dispose()
        penalty = min(self.redeploy_penalty.get(op.char_id, 0) + 1, 2)
        self.redeploy_penalty[op.char_id] = penalty
        self.redeploy_ready_at[op.char_id] = self.time + op.respawn_time

    def _try_deploy(self, plan: OperatorPlan) -> str:
        if plan.char_id not in self.characters:
            return "skip"
        if plan.tile is None:
            return "skip"
        if self._active_operator(plan.char_id) is not None:
            return "skip"
        if self.time + 1e-9 < self.redeploy_ready_at.get(plan.char_id, 0.0):
            return "wait"
        if sum(1 for op in self.operators if not op.dead) >= self.character_limit:
            return "wait"
        module = (
            self.battle_equips.get(plan.module_id) if plan.module_id else None
        )
        attrs = D.char_attributes(
            self.characters[plan.char_id],
            plan.level,
            plan.elite,
            trust=plan.trust,
            potential_rank=plan.potential_rank,
            module=module,
            module_level=plan.module_level,
        )
        base_cost = int(attrs.get("cost", 0))
        penalty = self.redeploy_penalty.get(plan.char_id, 0)
        cost = min(math.floor(base_cost * (1.0 + 0.5 * penalty)), self.max_cost)
        row, col = plan.tile
        profession = str(self.characters[plan.char_id].get("position", "RANGED"))
        if self.dp < cost:
            return "wait"
        if self.tile_occupied(row, col):
            return "skip"
        if not self.is_buildable(row, col, profession):
            return "skip"
        self.dp -= cost
        self.operators.append(self._make_operator(plan, cost))
        return "ok"

"""Tick-based Arknights battle engine (PRTS-flavoured).

Core rules follow PRTS 游戏数据基础 / 战斗机制:
  - physical:  max(ATK*scale*0.05, ATK*scale - DEF)
  - arts:      max(ATK*scale*0.05, ATK*scale*(1 - RES/100))
  - attack interval = baseAttackTime * 100 / attackSpeed
  - enemies move on continuous route segments and use a 0.7071 ground block
    radius; flying motion bypasses ordinary ground blockers
  - melee attack range is the tile(s) in front of the operator's facing
  - SP types, charge capacity, ammo duration and explicit interruption hooks;
    MANUAL skills wait for a plan operation unless auto_skill is enabled

Skill behaviour trees (`battle/buff_template_data.json`) are dispatched through
an incremental interpreter. Unsupported events/nodes remain observable in the
result instead of being silently treated as exact simulation.
"""

from __future__ import annotations

import copy
import math
import struct
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from functools import cmp_to_key, lru_cache
from typing import Any

from . import data as D
from .skill_controls import SkillControls, skill_controls
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
    rounded_sp_ratio,
    registered_buff_spec_issue,
    read_blackboard_value,
    contains_entity_value_read,
    node_name,
    ABNORMAL_READ_FLAGS,
    resolve_behavior_template,
)
from .buffs import BuffCatalog, BuffContainer, BuffDefinition, BuffInstance, calculate_attribute, element_break_buff_keys
from .health import gain_hp
from .buff_database import BuffDatabase
from .vision import blind_tile
from .behavior_components import ComponentCatalog, ComponentListener, module_resources, numeric_issue, CALCULATE, OUTPUT
from .behavior_preview import BuffAttributePreview
from .mechanics import (
    ApplyWay,
    AttackType,
    DamagePacket,
    DamageResult,
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
from .random_trace import RandomDecisionRng, uniform_bounds, uniform_value
from .graphics import (
    EnemyGraphicProfile, OperatorGraphicProfile, FacingRuntime, ProjectileMotionProfile, map_offset,
)
from .targeting import FramePeriodicTicker, TargetSelectorRuntime
from .loadout import deployment_position, deployment_tile_matches, module_phase, resolve_plan, select_abilities
from .forms import MAKOTO, MakotoS3
from .healing import SHU, SHU_S1, ShuHealing
from .operator_skills import (SUPPORTED as ROSTER_SKILLS, TOKEN_SKILLS, NEXT,
                             AttackProfile, OperatorSkillRuntime, SoulWardRuntime, SOUL_WARD,
                             live_runtimes, effect_runtimes)
from .induced_movement import temporary_move
from .funnels import WHITW2_S1, Whitw2S1Runtime
from .runes import blackboard as rune_blackboard, forbidden_locations, select_runes
from .squad import SquadBonus, SquadEffects, resolve_squad, skill_in_effect
from .vecbreak import VecBreakRuntime, predefined_builds, normalize_predefined, TRAP as VEC_TRAP
from .mansfield import MansfieldRuntime, DEVICE as MANSFIELD_DEVICE
from .cw import CwRuntime
from .wdex import WdExRuntime
from .dublin import DublinRuntime
from .operations import OperationAttempt, OperationLog
from .waves import DynamicWaveScheduler, SpawnFragment
from .spawn_core import PREVIEW_CURSOR_COUNT, PREVIEW_CURSOR_INTERVAL


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
# Quantized fixed-frame duration used by the compatible timer model.
CLIENT_FIXED_TIMER_STEP = 0.03333330154418945
CLIENT_FP_SCALE = 1 << 32
# Equality tolerance for the Q32.32 timer model.
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
    fragment_start_time: float = 0.0
    nominal_time: float = 0.0
    scheduler_delay_frames: int = 0
    logic_frame: int | None = None
    wave_index: int = 0
    fragment_index: int = 0
    dont_block_wave: bool = False
    block_fragment: bool = False
    dispatched: bool = False
    cancelled: bool = False


@dataclass(frozen=True)
class _SpawnWorkItem:
    nominal_time: float
    order: int
    kind: str
    source_action: dict[str, Any] | None = field(default=None, compare=False, kw_only=True)
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
    skill_level: int | None = None
    auto_skill: bool = False
    on_failure: str = "WAIT"
    mode: int | None = None

    def __post_init__(self) -> None:
        self.action = self.action.upper()
        self.on_failure = self.on_failure.upper()
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
    operation_results: list[dict[str, Any]] = field(default_factory=list)
    spawn_timing: str = SPAWN_TIMING_FAST
    enemy_attack_timing: str = ENEMY_ATTACK_TIMING_FLOAT
    behavior_components: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _ScheduledHighlandEffect:
    execute_at: float
    source: "_Operator"
    tile: tuple[int, int]
    amount: float
    sp_recovery_locked: bool = False
    sluggish_duration: float = 0.5


@dataclass
class _PullForce:
    origin_row: float
    origin_col: float
    initial_distance: float
    force: float
    remaining: float
    interrupt_source: _Operator | None = None
    stop_radius: float | None = None
    stop_after_reached: bool = False
    velocity_stopped: bool = False
    stop_within_source_block_range: bool = False


@dataclass
class _Enemy:
    component_listeners: list[ComponentListener] = field(default_factory=list, init=False)
    component_claims: set[str] = field(default_factory=set, init=False)
    component_rejected_buffs: set[str] = field(default_factory=set, init=False)
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
    wave_index: int = 0
    fragment_index: int = 0
    dont_block_wave: bool = False
    block_fragment: bool = False
    attack_duration: float = 0.0
    not_count_in_total: bool = False
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
    pull_interrupt_source: _Operator | None = None
    pull_sources: dict[int | None, _PullForce] = field(default_factory=dict)
    remaining_path_distance: float = 0.0
    attack_cd: float = 0.0
    attack_cd_frames: int = 0
    attack_cd_interval: float = 0.0
    attack_animation_end_frame: int | None = None
    attack_rearm_frame: int | None = None
    attack_waits_for_cooldown: bool | None = None
    attack_count: int = 0
    blocked_by: "_Operator | None" = None
    wisdel_remnant_source: "_Operator | None" = None
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
    shell_active: bool = False
    shell_reflect: float = 0.0
    registered_max_hp_base: float | None = None
    attribute_steals: dict[str, dict[int, tuple[Any, float]]] = field(default_factory=dict)
    position_jump_index: int = 0
    sanity_bind_bonuses: dict[int, tuple[float, float]] = field(default_factory=dict)
    extra_hp: float = 0.0
    max_extra_hp_ratio: float = 0.0
    minus_hp: float = 0.0
    max_minus_hp_ratio: float = 0.0
    negative_hp_heal_free: bool = False
    enemy_tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        self.buffs.owner = self
        self.statuses.base_motion_flying = self.motion.upper() in {"FLY", "FLYING"}

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
    primary: "_Enemy | _Operator"
    targets: list["_Enemy | _Operator"]
    attack_scale: float
    source_atk_scale: float
    sp_recovery_locked: bool
    will_hit: bool
    profile: Any = None
    followup_frames: tuple[int, ...] = ()
    animation_events: tuple[float, ...] = ()
    minimum_execute_at: float = -1.0


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
    impact_at: float | None = None
    origin: tuple[float, float, float] | None = None
    mount_point: str | None = None
    fixed_destination: tuple[float, float, float] | None = None
    lifecycle: dict[str, Any] = field(default_factory=dict)
    last_destination: tuple[float, float, float] | None = None
    born_at: float | None = None
    behavior_payload: dict[str, Any] | None = None
    enemy_attack_context: dict[str, Any] | None = None

    def position(self) -> tuple[float, float]:
        return self.position_row, self.position_col


@dataclass
class _Operator:
    component_listeners: list[ComponentListener] = field(default_factory=list, init=False)
    component_claims: set[str] = field(default_factory=set, init=False)
    component_rejected_buffs: set[str] = field(default_factory=set, init=False)
    component_mode_index: int | None = field(default=None, init=False)
    component_mode_claims: set[str] = field(default_factory=set, init=False)
    component_mode_rejections: set[str] = field(default_factory=set, init=False)
    component_talent_selection: tuple | None = field(default=None, init=False)
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
    graphic_animation: str = "Idle"
    graphic_started_at: float = 0.0
    graphic_rate: float = 1.0
    graphic_timing: dict[str, Any] | None = None
    graphic_face: str = "front"
    graphic_sign: float = 1.0
    last_projectile_mount: str = "MUZZLE"
    mount_group_indices: dict[str, int] = field(default_factory=dict)
    skill_attack_timing: dict[str, dict[str, float]] = field(default_factory=dict)
    pending_attack: _PendingOperatorAttack | None = None
    facing: int = 0
    damage_type: str = "PHYSICAL"
    atk_scale: float = 1.0
    def_scale: float = 1.0
    attack_cd: float = 0.0
    attack_cd_interval: float = 0.0
    dead: bool = False
    elemental_resistance: float = 0.0
    extra_hp: float = 0.0
    max_extra_hp_ratio: float = 0.0
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
    plan_profile: int | None = None
    trait_blackboard: dict[str, Any] = field(default_factory=dict)
    talent_blackboards: dict[str, dict[str, Any]] = field(default_factory=dict)
    talent_blackboards_by_index: dict[int, dict[str, Any]] = field(default_factory=dict)
    hp_recovery_interval: float = 0.0
    hp_recovery_remaining: float = 0.0
    hp_recovery_frame: int = -1
    negative_hp_heal_free: bool = False
    minus_hp: float = 0.0
    max_minus_hp_ratio: float = 0.0
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
    skill_controls: SkillControls = field(default_factory=SkillControls)
    skill_mode: int = 0
    skill_progress_locked: bool = False
    skill_progress_remaining: float | None = None
    skill_mode_target: int | None = None
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
    skill_index: int | None = None
    behavior_trigger_count: int = 0
    behavior_next_trigger_at: float = -1.0
    behavior_finish_dispatched: bool = False
    buff_blackboards: dict[str, dict[str, Any]] = field(default_factory=dict)
    buffs: BuffContainer = field(default_factory=BuffContainer)
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
    base_max_hp: float = 0.0
    max_hp_final_scale: float = 1.0
    registered_max_hp_active: bool = False
    base_attack_speed: float = 100.0
    squad_bonus: SquadBonus = field(default_factory=SquadBonus)
    received_heal_scale: float = 1.0
    squad_sp_ticks: int = 0
    form_runtime: MakotoS3 | None = None
    healing_runtime: ShuHealing | None = None
    skill_runtime: OperatorSkillRuntime | None = None
    air_blocking: bool = False
    # A landing action can suppress motion target-free before its block mode
    # changes. None preserves the existing gate for other aerial blockers.
    liftoff: bool | None = None
    lethal_handler: Any = None

    @property
    def charge_count(self) -> int:
        if self.sp_cost <= 0:
            return 0
        # Match _skill_ready's numerical tolerance at integral charge borders.
        return min(self.max_charge_time, math.floor((self.sp + 1e-9) / self.sp_cost))


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
        operator_graphics: dict[str, Any] | None = None,
        module_index: dict[str, Any] | None = None,
        favor_table: dict[str, Any] | None = None,
        squad: list[OperatorPlan] | None = None,
        operator_forms: dict[str, Any] | None = None,
        random_trace: bool = False,
        random_tape: dict | None = None,
        module_context: dict[str, Any] | None = None,
        behavior_components: dict[str, Any] | None = None,
        element_break_data: dict[str, Any] | None = None,
        buff_database: dict[str, Any] | None = None,
    ) -> None:
        if "selectedDifficulty" not in level:
            level = select_runes(level)
        self.level = level
        self.module_context = dict(module_context or {})
        if set(self.module_context) - {"game_tags", "map_tags", "ingots"}:
            raise ValueError("unknown module_context field")
        for key in ("game_tags", "map_tags"):
            tags = self.module_context.get(key, ())
            if not isinstance(tags, (list, tuple)) or any(not isinstance(tag, str) for tag in tags):
                raise ValueError(f"module_context.{key} must be a list of strings")
        ingots = self.module_context.get("ingots", 0)
        if type(ingots) is not int or ingots < 0:
            raise ValueError("module_context.ingots must be a nonnegative integer")
        self.forbidden_locations = forbidden_locations(level)
        self.operator_forms = (D.load_operator_forms() if operator_forms is None and D.DATA_DIR is not None
                               else operator_forms or {})
        self.enemy_index = enemy_index
        self.characters = characters
        self.ranges = ranges
        self.skills = skills or {}
        self.attack_timing = attack_timing or {}
        self.behavior_diagnostics = BehaviorDiagnostics()
        self.mechanic_diagnostics: Counter[str] = Counter()
        supported_runes = {"enemy_attribute_mul", "enemy_talent_blackb_mul", "enemy_talent_blackb_add",
                           "global_forbid_location", "global_lifepoint", "level_hidden_group_enable"}
        for rune in level.get("runes", []):
            if rune.get("key") not in supported_runes:
                self.mechanic_diagnostics[f"STAGE_RUNE_UNIMPLEMENTED:{rune.get('key')}"] = 1
        self.buff_database = BuffDatabase(D.load_buff_database() if buff_database is None and D.DATA_DIR is not None
                                          else buff_database)
        self.behavior = BehaviorInterpreter(
            behavior_templates,
            strict=strict_behaviors,
            diagnostics=self.behavior_diagnostics,
            buff_database=self.buff_database,
        )
        self.battle_equips = battle_equips or {}
        self.component_catalog = ComponentCatalog(behavior_components)
        self.component_status: list[dict[str, Any]] = []
        self.projectile_data = projectile_data or {}
        element_break_data = (D.load_element_break_buffs() if element_break_data is None and D.DATA_DIR is not None
                              else {} if element_break_data is None else element_break_data)
        self.element_break_buff_keys = element_break_buff_keys(element_break_data)
        self.element_break_durations = {key: (float(entry["elementBreakDuration"]),
            float(entry["enemyElementBreakDuration"])) for key, entry in element_break_data.items()}
        if not element_break_data:
            self.mechanic_diagnostics["EP_BREAK_BUFF_DIRECTORY_MISSING"] = 1
        if buff_abilities is None and D.DATA_DIR is not None:
            buff_abilities = D.load_enemy_buff_abilities()
        self.buff_catalog = BuffCatalog.from_dict(buff_abilities)
        self.buff_catalog.definitions = {key: replace(item, element_break=True)
            if key in self.element_break_buff_keys else item
            for key, item in self.buff_catalog.definitions.items()}
        self.projectile_speeds: dict[str, float] = {
            str(key): float(value)
            for key, value in self.projectile_data.get("speeds", {}).items()
        }
        self.enemy_projectiles = self.projectile_data.get("enemies", {})
        self.operator_projectiles = self.projectile_data.get("operators", {})
        self.module_index = (D.load_module_index() if module_index is None and D.DATA_DIR is not None
                             else module_index or {})
        self.favor_table = (D.load_favor_table() if favor_table is None and D.DATA_DIR is not None
                            else favor_table or {})
        resolved_plan = []
        for index, item in enumerate(plan, 1):
            try:
                resolved_plan.append(resolve_plan(normalize_predefined(item, level), self.characters, self.skills,
                                                   self.battle_equips, self.module_index, self.favor_table))
            except ValueError as error:
                raise ValueError(f"plan item {index}: {error}") from error
        self.plan = sorted(resolved_plan, key=lambda p: p.time)
        self.squad, ambiguous = resolve_squad(self.plan, squad, self.characters, self.skills,
                                              self.battle_equips, self.module_index, self.favor_table, set(predefined_builds(level)))
        for cid in ambiguous:
            self.mechanic_diagnostics[f"SQUAD_INFERRED_BUILD_AMBIGUOUS:{cid}"] = 1
        self.squad_effects = SquadEffects(self.squad, self.characters, self.battle_equips)
        self._squad_field_signature: tuple | None = None
        source_indices = {id(item): index for index, item in enumerate(resolved_plan, 1)}
        self.operation_log = OperationLog(self.plan, [source_indices[id(item)] for item in self.plan])
        self.dt = dt
        self.seed = int(seed)
        self.rng = (RandomDecisionRng(self.seed, tape=random_tape)
                    if random_trace or random_tape is not None else DeterministicRng(self.seed))
        self.assumptions = assumptions or DEFAULT_ASSUMPTIONS
        self.enemy_graphics: dict[str, EnemyGraphicProfile] = {}
        if operator_graphics is None and D.DATA_DIR is not None:
            operator_graphics = D.load_operator_graphics()
        operator_profiles = (operator_graphics or {}).get("operators", {})
        if not isinstance(operator_profiles, dict):
            raise ValueError("operator graphics 'operators' must be an object")
        self.operator_graphics = {key: OperatorGraphicProfile.from_dict(spec)
                                  for key, spec in operator_profiles.items()}
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
        self.cost_lower_bound = self.squad_effects.cost_lower_bound
        self.initial_cost = min(self.max_cost, self.initial_cost + self.squad_effects.initial_cost)
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
        self.dublin = DublinRuntime(self)
        self.routes = self._build_routes(level.get("routes", []))
        self.paths = self._routes_to_paths(self.routes)
        self.spawn_events = self._build_spawn_events(level.get("waves", []))
        self.spawn_index = 0

        self.dp = self.initial_cost
        self.dp_timer = 0.0
        self.time = 0.0
        self.frame_index = 0
        self.enemies: list[_Enemy] = []
        self._lemuen_survivor_sp: dict[int, tuple[_Enemy, list[tuple[_Operator, float]]]] = {}
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
        self.vecbreak = VecBreakRuntime(self)
        self.mansfield = MansfieldRuntime(self)
        self.wdex = WdExRuntime(self)
        self.cw = CwRuntime(self)
        self.mansfield.initialize_devices()
        self.dublin.initialize()
        self.initial_passable_cells = frozenset((row, col)
            for row, cells in enumerate(self.map_grid) for col in range(len(cells))
            if self._ground_passable_cell(row, col))
        for cid in self.squad_effects.boards:
            if self.squad_effects.partial_talent(cid):
                self.mechanic_diagnostics[f"OPERATOR_TALENT_PARTIAL:{cid}"] = 1
        if self.squad_effects.shu_extra_index is not None:
            self.mechanic_diagnostics["SHU_Y_COMBINED_CONDITION_CANDIDATE"] = 1
        if "char_249_mlyss" in self.squad_effects.boards:
            self.mechanic_diagnostics["MLYSS_FIRST_DEPLOY_COST_LIFETIME_CANDIDATE"] = 1
        if "char_1051_headb2" in self.squad_effects.boards:
            self.mechanic_diagnostics["HEADB2_TEAM_AURA_SKILL_EVENT_PHASE_CANDIDATE"] = 1

    def _sync_squad_effects(self) -> None:
        live = [op for op in self.operators if not op.dead]
        signature = tuple((op.deployment_index, skill_in_effect(op), op.tile,
                           op.char_id == "char_1045_svash2" and self.time - op.deployed_at >=
                           float(op.talent_blackboards.get("2", {}).get("interval", 15)),
                           op.statuses.targetable if "char_1051_headb2" in self.squad_effects.boards else False)
                          for op in live)
        if signature == self._squad_field_signature:
            return
        self._squad_field_signature = signature
        for op in live:
            bonus = self.squad_effects.bonus(op.char_id, op.base_cost, live)
            self._set_maximum_hp(op, self._operator_maximum_hp(op, bonus))
            op.squad_bonus = bonus
            op.received_heal_scale = (1.0 + bonus.heal) * (
                float(op.trait_blackboard.get("heal_scale", 1)) if op.char_id == "char_4145_ulpia" else 1)
            immune = any(r.op.char_id == "char_1045_svash2" and r.talent(2) and r.sjerag_ally(op)
                         for r in live_runtimes(self))
            if immune and Status.FROZEN not in op.statuses.immunities:
                op.statuses.immunities.add(Status.FROZEN)
                op.roster_freeze_immunity_added = True
            elif not immune and getattr(op, "roster_freeze_immunity_added", False):
                op.statuses.immunities.discard(Status.FROZEN)
                op.roster_freeze_immunity_added = False

    def operator_attack_power(self, op: _Operator, scale: float | None = None, *, synchronize: bool = True,
                              modifiers=None) -> float:
        if synchronize:
            self._sync_squad_effects()
        runtime = op.skill_runtime
        modifiers = op.buffs.modifiers("ATK", registered_only=True) if modifiers is None else modifiers
        # Stage base-attribute bonuses precede the ordinary skill/team pools.
        base_atk = op.atk * (1.0 + float(getattr(op, "mode_base_atk_bonus", 0)))
        return max(calculate_attribute(base_atk,
            modifiers, multiplier=(
            (op.atk_scale if scale is None else scale) + op.squad_bonus.atk
            + (op.form_runtime.atk_bonus if op.form_runtime else 0)
            + (runtime.attack_bonus() if runtime else 0)
            + float(getattr(op, "roster_atk_bonus", 0))
            + sum(r.aura_atk(op) for r in live_runtimes(self))),
            final_addition=runtime.flat_attack() if runtime else 0), 0.0)

    def _operator_maximum_hp(self, op: _Operator, bonus: Any = None, extra=(), modifiers=None) -> float:
        bonus = op.squad_bonus if bonus is None else bonus
        modifiers = (*(op.buffs.modifiers("MAX_HP", registered_only=True) if modifiers is None else modifiers), *extra)
        return calculate_attribute(op.base_max_hp, modifiers, multiplier=(1.0 + bonus.max_hp
            + (op.form_runtime.hp_bonus if op.form_runtime else 0)
            + (op.skill_runtime.hp_bonus if op.skill_runtime else 0)
            + sum(r.aura_hp(op) for r in live_runtimes(self))
            + float(getattr(op, "roster_hp_bonus", 0))),
            final_addition=op.skill_runtime.stolen_hp if op.skill_runtime else 0,
            final_scale=op.max_hp_final_scale)

    @staticmethod
    def _set_maximum_hp(actor: Any, maximum: float) -> None:
        if not math.isfinite(maximum):
            raise ValueError("non-finite maximum HP")
        maximum = max(1.0, maximum)
        if maximum == actor.max_hp:
            return
        previous = actor.max_hp
        actor.max_hp = maximum
        if actor.dead or actor.hp < 0:
            # The legacy signed debt represents a separate negative HP pool.
            # Changing its limit must not scale the amount already owed.
            return
        ratio = max(0.0, min(1.0, actor.hp / previous)) if previous > 0 else 0.0
        actor.hp = ratio * maximum

    def _registered_attributes_changed(self, container: BuffContainer) -> None:
        actor = container.owner
        modifiers = container.modifiers("MAX_HP", registered_only=True)
        if isinstance(actor, _Operator):
            if not modifiers and not actor.registered_max_hp_active:
                return
            actor.registered_max_hp_active = bool(modifiers)
            self._squad_field_signature = None
            self._sync_squad_effects()
            if actor.dead or not any(unit is actor for unit in self.operators):
                self._set_maximum_hp(actor, self._operator_maximum_hp(actor))
        elif isinstance(actor, _Enemy):
            if not modifiers and actor.registered_max_hp_base is None:
                return
            if actor.registered_max_hp_base is None:
                actor.registered_max_hp_base = actor.max_hp
            self._set_maximum_hp(actor, calculate_attribute(actor.registered_max_hp_base, modifiers,
                final_addition=-self._enemy_attribute_stolen(actor, "MAX_HP")))

    @staticmethod
    def _enemy_attribute_stolen(enemy: _Enemy, attribute: str) -> float:
        # Different character sources reduce each attribute by the highest steal.
        return max((amount for source, amount in enemy.attribute_steals.get(attribute, {}).values()
                    if not source.dead), default=0.0)

    def set_enemy_attribute_steal(self, enemy: _Enemy, source: _Operator,
                                attribute: str, amount: float) -> None:
        entries = enemy.attribute_steals.setdefault(attribute, {})
        if amount > 0:
            entries[id(source)] = (source, amount)
        else:
            entries.pop(id(source), None)
        if attribute == "MAX_HP":
            if enemy.registered_max_hp_base is None:
                enemy.registered_max_hp_base = enemy.max_hp
            self._registered_attributes_changed(enemy.buffs)

    def operator_defense(self, op: _Operator, *, synchronize: bool = True, modifiers=None) -> float:
        if synchronize:
            self._sync_squad_effects()
        modifiers = op.buffs.modifiers("DEF", registered_only=True) if modifiers is None else modifiers
        return max(calculate_attribute(op.defense, modifiers,
            multiplier=(op.def_scale + op.squad_bonus.defense
            + (op.skill_runtime.defense_bonus() if op.skill_runtime else 0)
            + sum(r.aura_def(op) for r in live_runtimes(self))), final_addition=sum(
                runtime.flat_defense_for(op) for runtime in effect_runtimes(self))), 0.0)

    def _module_ability_tags(self, module_id: str | None) -> dict[str, Any]:
        # Keep ISW-A effects paused until integrated strategy is in scope.
        if module_id == "uniequip_004_swire2":
            return {}
        return {key: self.module_context.get(key, ()) for key in ("game_tags", "map_tags")}

    def deployment_cost(self, char_id: str, base_cost: float) -> int:
        penalty = self.redeploy_penalty.get(char_id, 0)
        value = math.floor(base_cost * (1.0 + .5 * penalty) + self.squad_effects.cost_modifier(char_id))
        # UNTIL_NEXT_SPAWN card effects outlive the character that issued them.
        discount = sum(r.deck_discounts.get(char_id, 0) for r in effect_runtimes(self))
        scale = 1.0
        carried = next((p for p in self.squad if p.char_id == char_id), None)
        if carried and carried.module_id:
            phase = module_phase(self.battle_equips[carried.module_id], carried.module_level)
            _, talents, _ = select_abilities(self.characters[char_id], carried.elite, carried.level,
                carried.potential_rank, phase, **self._module_ability_tags(carried.module_id))
            scale = next((float(self._blackboard(c).get("cost_scale", 1)) for c in talents.values()
                          if c.get("prefabKey") == "14_root"), 1.0)
        if getattr(self, "vecbreak", None) and self.vecbreak.enabled:
            if char_id == VEC_TRAP: value = base_cost
            value = self.vecbreak.cost(char_id, value)
        return max(0, min(math.floor((value - discount) * scale), self.max_cost))

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
            "WAIT_CURRENT_FRAGMENT_TIME",
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
        self._spawn_fragments: list[SpawnFragment] = []
        if self.spawn_timing == SPAWN_TIMING_FRAME_CORE:
            events = self._build_frame_core_spawn_events(waves)
        elif self.spawn_timing == SPAWN_TIMING_CLIENT:
            events = self._build_client_spawn_events(waves)
        else:
            events = self._build_fast_spawn_events(waves)
        self.wave_scheduler = DynamicWaveScheduler(self._spawn_fragments)
        if (len(waves) > 1 or any(e.block_fragment for e in events)
                or any(float(w.get("maxTimeWaitingForNextWave", -1) or -1) > 0 for w in waves)):
            self.mechanic_diagnostics.setdefault("DYNAMIC_WAVE_HANDOFF_CANDIDATE", 1)
        if self.level.get("branches"):
            self.mechanic_diagnostics.setdefault("SPAWN_BRANCH_EXECUTOR", 1)
        for wave in waves:
            for fragment in wave.get("fragments", []):
                for action in fragment.get("actions", []):
                    kind = str(action.get("actionType", "")).upper()
                    if kind not in ("SPAWN", "PREVIEW_CURSOR", "DISPLAY_ENEMY_INFO", "EMPTY"):
                        self.mechanic_diagnostics.setdefault(f"SPAWN_ACTION_EXECUTOR:{kind or 'UNKNOWN'}", 1)
                    if action.get("randomSpawnGroupKey") or action.get("hiddenGroup"):
                        self.mechanic_diagnostics.setdefault("SPAWN_CONDITIONAL_GROUP", 1)
                    if (action.get("managedByScheduler") is False or action.get("forceBlockWaveInBranch")
                            or action.get("notCountInTotal") or action.get("isUnharmfulAndAlwaysCountAsKilled")):
                        self.mechanic_diagnostics.setdefault("SPAWN_SPECIAL_ACCOUNTING", 1)
        return events

    @staticmethod
    def _spawn_flags(wave: int, fragment: int, action: dict) -> dict[str, Any]:
        return {"wave_index": wave, "fragment_index": fragment,
                "dont_block_wave": bool(action.get("dontBlockWave", False)),
                "block_fragment": bool(action.get("blockFragment", False))}

    def _record_spawn_fragment(self, wave_index: int, fragment_index: int, end: float,
                               wave_start: float, wave: dict, events: list[SpawnEvent]) -> None:
        self._spawn_fragments.append(SpawnFragment(
            wave_index, fragment_index, end, wave_start,
            post_delay=float(wave.get("postDelay", 0) or 0),
            max_wait=float(wave.get("maxTimeWaitingForNextWave", -1) or -1),
            events=events,
        ))

    def _finish_spawn_wave(self, begin: int, end: float) -> None:
        if begin < len(self._spawn_fragments):
            self._spawn_fragments[-1].last = True
            for group in self._spawn_fragments[begin:]:
                group.wave_end_time = end

    def _build_frame_core_spawn_events(
        self, waves: list[dict[str, Any]]
    ) -> list[SpawnEvent]:
        """Use the published SpawnCore queue arithmetic for static wave data."""
        from .spawn_core import build_spawn_frame_schedule

        timings: list[dict[str, Any]] = []
        hidden_groups = {rune_blackboard(r).get("key") for r in self.level.get("runes", [])
                         if r.get("key") == "level_hidden_group_enable"}
        rows = build_spawn_frame_schedule(waves, fragment_timings=timings,
                                         enabled_hidden_groups=hidden_groups,
                                         truncate_on_timeout=False)
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
                **self._spawn_flags(row["wave"], row["fragment"], row),
            ))
        for timing in timings:
            wi, fi = timing["wave"], timing["fragment"]
            self._record_spawn_fragment(wi, fi, (timing["completion"] + 1) / CLIENT_LOGIC_RATE,
                                        (timing["wave_start"] + 1) / CLIENT_LOGIC_RATE, waves[wi],
                                        [e for e in events if e.wave_index == wi and e.fragment_index == fi])
            for event in self._spawn_fragments[-1].events:
                event.fragment_start_time = (timing["fragment_start"] + 1) / CLIENT_LOGIC_RATE
            self._spawn_fragments[-1].last = timing["last"]
            self._spawn_fragments[-1].wave_end_time = (timing["wave_end"] + 1) / CLIENT_LOGIC_RATE
        self.used_assumptions.add("SPAWN_CORE_FRAME_SCHEDULER")
        return events

    def _build_fast_spawn_events(
        self, waves: list[dict[str, Any]],
    ) -> list[SpawnEvent]:
        events: list[SpawnEvent] = []
        t = 0.0
        for wi, wave in enumerate(waves):
            begin = len(self._spawn_fragments)
            t += wave.get("preDelay", 0.0) or 0.0
            wave_start_time = t
            for fi, frag in enumerate(wave.get("fragments", []) or [{}]):
                event_begin = len(events)
                t += frag.get("preDelay", 0.0) or 0.0
                frag_end = t
                fragment_start_time = t
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
                                fragment_start_time=fragment_start_time,
                                nominal_time=t + pre_delay + k * interval,
                                **self._spawn_flags(wi, fi, action),
                            )
                        )
                    frag_end = max(
                        frag_end, t + pre_delay + (count - 1) * interval
                    )
                t = frag_end
                self._record_spawn_fragment(wi, fi, t, wave_start_time, wave, events[event_begin:])
            t += wave.get("postDelay", 0.0) or 0.0
            self._finish_spawn_wave(begin, t)
        events.sort(key=lambda e: e.time)
        return events

    @staticmethod
    def _client_sorted_actions(
        actions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Reproduce the client's unstable timeOffset quicksort.

        The compatible scheduler sorts actions only by their time offsets.
        Hoare partition swaps equal elements, so equal delays do not preserve
        input order.  This remains part of the candidate scheduling model.
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

        The candidate model adds one accumulated logic frame for each entered
        fragment, completed spawn, preview cursor, and calibrated display
        action.  Costs are folded into final spawn frames here so the hot
        battle loop stays flat.

        Auto-preview cursor offsets are relative to the associated spawn time;
        they may therefore precede the nominal start of their Fragment.
        """
        events: list[SpawnEvent] = []
        nominal_cursor = 0.0
        completed_work_frames = 0

        for wi, wave in enumerate(waves):
            begin = len(self._spawn_fragments)
            nominal_cursor += float(wave.get("preDelay", 0.0) or 0.0)
            wave_start_frame = (
                self._client_logic_frame(nominal_cursor) + completed_work_frames
            )
            wave_start_time = wave_start_frame / CLIENT_LOGIC_RATE

            for fi, fragment in enumerate(wave.get("fragments", []) or [{}]):
                event_begin = len(events)
                nominal_cursor += float(fragment.get("preDelay", 0.0) or 0.0)
                fragment_start = nominal_cursor
                fragment_end = fragment_start
                fragment_start_time = (self._client_logic_frame(fragment_start) + completed_work_frames) / CLIENT_LOGIC_RATE
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
                    if action_type == "PREVIEW_CURSOR":
                        # The preview executor uses its own pulse count and
                        # interval, shared with the frame scheduling model.
                        count = PREVIEW_CURSOR_COUNT
                        interval = PREVIEW_CURSOR_INTERVAL
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
                                    source_action=action,
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
                # candidate schedule places its completion after regular work
                # at that offset.  Its autoPreviewRoute flag does not create
                # two additional cursor items in this model.
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
                                fragment_start_time=fragment_start_time,
                                nominal_time=item.nominal_time,
                                scheduler_delay_frames=(
                                    completed_work_frames + preview_after
                                ),
                                logic_frame=logic_frame,
                                **self._spawn_flags(wi, fi, item.source_action or {}),
                            )
                        )
                    completed_work_frames += 1

                nominal_cursor = fragment_end
                self._record_spawn_fragment(
                    wi, fi, (self._client_logic_frame(nominal_cursor) + completed_work_frames) / CLIENT_LOGIC_RATE,
                    wave_start_time, wave, events[event_begin:],
                )

            nominal_cursor += float(wave.get("postDelay", 0.0) or 0.0)
            self._finish_spawn_wave(begin, (self._client_logic_frame(nominal_cursor) + completed_work_frames) / CLIENT_LOGIC_RATE)

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
        lifted, grounded = enemy.statuses.motion_counts()
        return (enemy.statuses.levitate_active() if lifted or grounded else
                enemy.motion.upper() in {"FLY", "FLYING"})

    def _is_hole_cell(self, row: int, col: int) -> bool:
        return self.tile(row, col).get("tileKey") == "tile_hole"

    def _uses_ground_navigation(self, enemy: _Enemy) -> bool:
        runtime = getattr(self, "dublin", None)
        return (runtime.uses_ground_navigation(enemy) if runtime
                else not self._is_airborne(enemy))

    def _ground_passable_cell(self, row: int, col: int) -> bool:
        if getattr(self, "dublin", None) and self.dublin.blocked_tile(row, col):
            return False
        if getattr(self, "cw", None) and self.cw.blocked_tile(row, col):
            return False
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
        if not hasattr(self, "map_grid") or not self._uses_ground_navigation(enemy):
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
            or not self._uses_ground_navigation(enemy)
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
        if (row, col) in self.forbidden_locations:
            return False
        tile = self.tile(row, col)
        return deployment_tile_matches(profession, tile.get("buildableType"))

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
                rr, cc = -dc, dr
            elif facing % 4 == 2:  # left
                rr, cc = -dr, -dc
            else:  # up
                rr, cc = dc, -dr
            cells.add((tile[0] + rr, tile[1] + cc))
        return cells

    # ------------------------------------------------------------- specs
    def _make_enemy(
        self, key: str, route_index: int, *, wave_start_time: float = 0.0
    ) -> _Enemy:
        enemy_data = (self.mansfield.enemy_data(key) if self.mansfield.enabled else
                      self.vecbreak.enemy_data(key) if self.vecbreak.enabled
                      else D.enemy_data_for_level(self.enemy_index, key, self.level) if self.wdex.enabled
                      else D.enemy_data_at(self.enemy_index, key, level=0))
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
            enemy_tags=frozenset(str(tag) for tag in (D.mdef(enemy_data.get("enemyTags"), []) or [])),
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
        for rune in self.level.get("runes", []):
            board = rune_blackboard(rune)
            if board.get("enemy") and key not in str(board["enemy"]).split("|"):
                continue
            if rune.get("key") == "enemy_attribute_mul":
                enemy.max_hp *= float(board.get("max_hp", 1))
                enemy.hp = enemy.max_hp
                enemy.atk *= float(board.get("atk", 1))
                enemy.defense *= float(board.get("def", 1))
        if key in ("enemy_10126_rkbomb", "enemy_10127_rkmbst"):
            board = {p["key"]: p["value"] for p in enemy_data.get("talentBlackboard", [])}
            prefix = "Shield" if key == "enemy_10126_rkbomb" else "M0Shield"
            for rune in self.level.get("runes", []):
                additions = rune_blackboard(rune)
                if key not in str(additions.get("enemy", "")).split("|"): continue
                for name, value in additions.items():
                    if name in board and isinstance(value, (int, float)):
                        if rune.get("key") == "enemy_talent_blackb_mul": board[name] *= value
                        elif rune.get("key") == "enemy_talent_blackb_add": board[name] += value
            enemy.shell_active = True
            enemy.shell_reflect = float(board.get(prefix + ".value", 0))
            enemy.damage_rules.append(DamageRule(
                "SCALE", 1 - float(board.get(prefix + ".damage_resistance", 0)),
                damage_types=frozenset({DamageType.PHYSICAL, DamageType.ARTS})))
            if key == "enemy_10127_rkmbst":
                enemy.attack_interval *= 100 / (100 + float(board.get(prefix + ".attack_speed", 0)))
            self.mechanic_diagnostics["HETEROGENEOUS_TARGET_PREFERENCE_CANDIDATE"] = 1
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
        if self.vecbreak.enabled:
            self.vecbreak.born_enemy(enemy, enemy_data)
        self.mansfield.born(enemy, enemy_data)
        self.cw.born(enemy, enemy_data)
        self.wdex.born(enemy, enemy_data)
        self.dublin.born(enemy, enemy_data)
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
            base_max_hp=hp,
            base_attack_speed=attack_speed,
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
            respawn_time=float(attrs.get("respawnTime", 70.0)) * max(0, 1 + self.squad_effects.respawn_modifier),
        )
        op.plan_profile = next((index for index, item in enumerate(self.plan) if item is plan), None)
        op.buffs.owner = op
        op.buffs.event_handler = self._dispatch_buff_event
        op.buffs.change_handler = self._registered_attributes_changed
        op.build_plan = plan
        equipped_phase = module_phase(module, plan.module_level) if module is not None else None
        trait, talents, deferred = select_abilities(
            char, plan.elite, plan.level, plan.potential_rank, equipped_phase,
            **self._module_ability_tags(plan.module_id),
        )
        if plan.module_id == "uniequip_004_swire2":
            self.mechanic_diagnostics["OPERATOR_MODULE_EFFECT_PAUSED:uniequip_004_swire2"] = 1
        op.trait_blackboard = self._blackboard(trait) if trait is not None else {}
        shu_module = op.char_id == SHU and plan.module_id == "uniequip_002_shu"
        if op.trait_blackboard and op.subprofession != "hammer" and not shu_module:
            self.mechanic_diagnostics[f"OPERATOR_TRAIT:{op.char_id}"] += 1
        for index, candidate in talents.items():
            prefab = str(candidate.get("prefabKey") or index)
            board = self._blackboard(candidate)
            if op.char_id == SHU:
                for item in candidate.get("blackboard", []):
                    if item.get("key") == "born_range_id" and item.get("valueStr"):
                        board["born_range_id"] = item["valueStr"]
            op.talent_blackboards[prefab] = board
            op.talent_blackboards_by_index[index] = board
            if self.squad_effects.handled_talent(op.char_id, index) or (self.vecbreak.enabled and op.char_id == VEC_TRAP):
                pass
            elif op.char_id == SHU and prefab == "1":
                pass
            elif (board or prefab not in ("#", "")) and not (op.char_id == "char_1051_headb2" and prefab in ("1", "2")):
                self.mechanic_diagnostics[f"OPERATOR_TALENT:{op.char_id}:{index}"] += 1
        for part in deferred:
            if op.char_id == "char_450_necras" and plan.module_id == "uniequip_002_necras" and part.endswith(":token"):
                self.mechanic_diagnostics["NECRAS_TOKEN_DEATH_LISTENER_CANDIDATE"] = 1
            else:
                self.mechanic_diagnostics[f"OPERATOR_MODULE_CONDITION:{plan.module_id}:{part}"] += 1
        if equipped_phase and not (shu_module or (op.char_id == "char_1051_headb2" and plan.module_id == "uniequip_002_headb2")):
            if equipped_phase.get("parts"):
                self.mechanic_diagnostics[f"OPERATOR_MODULE_EFFECT:{plan.module_id}"] += 1
        if self.vecbreak.enabled:
            op.roster_hp_bonus = self.vecbreak.char_bonus(op.char_id, "max_hp")
            op.mode_base_atk_bonus = self.vecbreak.char_bonus(op.char_id, "atk")
        if plan.skill_id:
            self._attach_skill(op, plan.skill_id, plan.skill_level)
            op.sp = min(op.sp_max, op.sp + self.squad_effects.initial_sp_bonus(op.char_id))
            op.auto = op.skill_type == "AUTO" or plan.auto_skill
        if op.char_id == MAKOTO and op.skill_prefab_id == "skchr_makoto_3":
            profile = self.operator_forms.get(MAKOTO, {})
            op.form_runtime = MakotoS3(self, op, profile)
            op.lethal_handler = op.form_runtime.on_lethal
            self.mechanic_diagnostics["MAKOTO_FORM_EVENT_HANDOFF_CANDIDATE"] = 1
            self.mechanic_diagnostics["MAKOTO_MULTI_TARGET_DAMAGE_SNAPSHOT_CANDIDATE"] = 1
            if op.talent_blackboards.get("2", {}).get("multi_attack_total_cnt", 1) != 1:
                self.mechanic_diagnostics["MAKOTO_ALL_OUT_MULTI_ATTACK_UNIMPLEMENTED"] = 1
            if not profile:
                self.mechanic_diagnostics["MAKOTO_FORM_ANIMATION_PROFILE_MISSING"] = 1
        if op.char_id == SHU:
            op.healing_runtime = ShuHealing(self, op)
            op.healing_runtime.born()
            if op.talent_blackboards.get("1"):
                self.mechanic_diagnostics["SHU_FARM_LIFECYCLE_CANDIDATE"] = 1
            if op.skill_prefab_id == SHU_S1:
                self.mechanic_diagnostics["SHU_S1_ACTION_TIMING_CANDIDATE"] = 1
            elif op.skill_prefab_id:
                self.mechanic_diagnostics[f"SHU_SKILL_INCOMPLETE:{op.skill_prefab_id}"] = 1
        if op.char_id == SOUL_WARD:
            op.skill_runtime = SoulWardRuntime(self, op)
        elif op.skill_prefab_id == WHITW2_S1:
            op.skill_runtime = Whitw2S1Runtime(self, op)
        elif op.skill_prefab_id in ROSTER_SKILLS or op.char_id == "char_1035_wisdel":
            op.skill_runtime = OperatorSkillRuntime(self, op)
        self.initialize_operator_target_selector(
            op, first_search_frame=self.frame_index + 32
        )
        self._face_operator_graphic(op)
        op.graphic_started_at = self.time
        module_bindings = list(module_resources(equipped_phase, plan, op.talent_blackboards,
                               **self._module_ability_tags(plan.module_id)))
        self.component_catalog.bind_operator(self, op, module_bindings=module_bindings, selected_talents=talents)
        if self.component_catalog.enabled and "module" in self.component_catalog.resource_kinds:
            for resource, board_key, reason in module_bindings:
                board = op.trait_blackboard if board_key == "trait" else op.talent_blackboards.get(board_key, {})
                self.component_catalog.bind(self, op, resource, board, reason=reason)
        self._sync_component_skill_mode(op, active=op.skill_active)
        return op

    def _attach_skill(self, op: _Operator, skill_id: str, skill_level: int | None = None) -> None:
        skill = self.skills.get(skill_id)
        if not skill:
            return
        levels = skill.get("levels", [])
        if not levels:
            return
        if skill_level is None:
            skill_level = len(levels)
        if type(skill_level) is not int or not 1 <= skill_level <= len(levels):
            raise ValueError(f"skill_level must be 1..{len(levels)}")
        level = levels[skill_level - 1]
        sp_data = level.get("spData", {})
        sp_type = str(sp_data.get("spType", ""))
        sp_cost = float(sp_data.get("spCost", 0) or 0)
        max_charge_time = max(int(sp_data.get("maxChargeTime", 1) or 1), 1)
        op.skill = level
        op.skill_index = next((i for i, item in enumerate(self.characters[op.char_id].get("skills", []))
                              if item.get("skillId") == skill_id), None)
        op.skill_controls = skill_controls(level)
        op.sp_cost = sp_cost
        op.max_charge_time = max_charge_time
        op.sp_max = sp_cost * max_charge_time
        op.sp = min(float(sp_data.get("initSp", 0) or 0), op.sp_max)
        op.sp_increment = float(sp_data.get("increment", 1.0) or 1.0)
        op.sp_recovery_remaining_raw = CLIENT_FP_SCALE
        op.sp_type = sp_type
        op.auto = False
        op.skill_type = str(level.get("skillType", ""))
        op.skill_duration_type = str(level.get("durationType", "NONE"))
        op.skill_blackboard = self._blackboard(level)
        prefab_id = str(level.get("prefabId", ""))
        op.skill_prefab_id = prefab_id
        op.behavior_template_key = resolve_behavior_template(
            prefab_id, self.behavior.templates
        )
        if prefab_id in ROSTER_SKILLS:
            op.behavior_template_key = None
        if prefab_id and self.component_catalog.enabled and "skill" in self.component_catalog.resource_kinds:
            self.component_catalog.bind(self, op, prefab_id, op.skill_blackboard)
        if op.behavior_template_key and not self.behavior_component_claimed(op, op.behavior_template_key):
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
            self._cast_skill(op, automatic=True)

    # ------------------------------------------------------------- loop
    def step(self) -> None:
        self.time += self.dt
        self.frame_index += 1
        self.dublin.pre_tick()

        for unit in (*self.operators, *self.enemies):
            if not unit.dead:
                unit.statuses.tick(self.dt)
                burst_delta = self._tick_element_burst(unit)
                if unit.elements.tick(self.dt, burst_dt=burst_delta) is not None:
                    self._finish_element_burst(unit)
            elif unit.element_burst is not None:
                self._finish_element_templates(unit)

        # Losing block capability immediately releases held enemies.
        for op in self.operators:
            if op.dead or op.statuses.can_block:
                continue
            for enemy in list(op.blocked):
                self._release_enemy_block(enemy)

        for op in self.operators:
            if op.healing_runtime is not None:
                op.healing_runtime.tick_tiles()
            if op.skill_runtime is not None and op.skill_runtime.notes:
                op.skill_runtime.tick_notes()
            if op.dead and op.skill_runtime is not None:
                op.skill_runtime.tick_detached()

        self._sync_squad_effects()
        self.dp_timer += self.dt * (0.5 if self.dp < 0 else 1.0)
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
            if self.end_reason:
                return

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
        self.vecbreak.tick()
        self.cw.tick()
        self.dublin.tick()
        self._cleanup()
        self._sync_squad_effects()

    def _uses_entity_update_order(self) -> bool:
        # The pre-movement switch explicitly selects the historical grouped
        # comparison instead of changing one receiver in native ordering.
        return (
            self.assumptions.entity_update_order_by_allocation
            and not self.assumptions.operator_target_selector_before_enemy_movement
        )

    def _spawn_due_enemies(self) -> None:
        for enemy in self.enemies:
            if enemy.dead or enemy.leaked:
                self.wave_scheduler.departed(enemy, self.time)
        while True:
            self.wave_scheduler.advance(self.time, self.frame_index, self.dt, self.enemies)
            if self.spawn_index >= len(self.spawn_events):
                break
            ev = self.spawn_events[self.spawn_index]
            if ev.cancelled:
                ev.dispatched = True
                self.spawn_index += 1
                continue
            if not self.wave_scheduler.allowed(ev) or not self._spawn_event_is_due(ev):
                break
            self.spawn_index += 1
            ev.dispatched = True
            if not 0 <= ev.route_index < len(self.paths):
                self.mechanic_diagnostics.setdefault("SPAWN_INVALID_ROUTE", 1)
                continue
            enemy = self._make_enemy(
                ev.enemy_key, ev.route_index, wave_start_time=ev.wave_start_time,
            )
            enemy.wave_index = ev.wave_index
            enemy.fragment_index = ev.fragment_index
            enemy.fragment_start_time = ev.fragment_start_time
            enemy.dont_block_wave = ev.dont_block_wave or enemy.not_count_in_total
            if self.vecbreak.enabled and ev.wave_index in self.vecbreak.released_waves:
                enemy.dont_block_wave = True
            enemy.block_fragment = ev.block_fragment and not enemy.not_count_in_total
            self.enemies.append(enemy)
            self.wave_scheduler.last_births[ev.wave_index] = self.time
            if not enemy.not_count_in_total:
                self.enemies_spawned += 1

    def all_spawns_finished(self) -> bool:
        return self.spawn_index >= len(self.spawn_events) and self.wave_scheduler.finished

    def all_counted_enemies_finished(self) -> bool:
        return not any(not enemy.not_count_in_total and not enemy.dead and not enemy.leaked
                       for enemy in self.enemies)

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
        self._sync_operator_attack_clock(op)
        if op.pending_attack is not None and not op.statuses.can_attack:
            if op.skill_runtime is not None:
                op.skill_runtime.cancel_attack_sequence(op.pending_attack)
            op.pending_attack = None
        if (
            op.pending_attack is not None
            and op.pending_attack.kind != "HAMMER"
            and self.frame_index >= op.pending_attack.execute_frame
        ):
            self._land_operator_attack(op)
        op.attack_cd -= self.dt
        if (op.form_runtime is not None and op.form_runtime.mode == "THANATOS"
                and abs(op.attack_cd) < 1e-9):
            # Keep the observed integral-frame form interval from gaining a
            # frame through floating-point subtraction residue.
            op.attack_cd = 0.0

    def _tick_enemy(self, enemy: _Enemy) -> None:
        if enemy.dead or enemy.leaked:
            return
        if self.dublin.before_tick(enemy):
            return
        if self.cw.move_debris(enemy):
            return
        if self.vecbreak.before_enemy_tick(enemy) or self.mansfield.before_tick(enemy):
            return
        self.wdex.before_tick(enemy)
        self._tick_lemuen_survivor_sp(enemy)
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
        if enemy.statuses.has(Status.FEAR) and enemy.blocked_by is not None:
            self._release_enemy_block(enemy)
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
        lure = getattr(enemy, "lure_target", None)
        if lure is not None and self.time < lure[1]:
            arrived = math.dist(enemy.position(), lure[0]) <= .4 or temporary_move(self, enemy, lure[0])
            arrived = arrived or math.dist(enemy.position(), lure[0]) <= .4
            if arrived:
                lure[2].lure_arrived(enemy)
            return
        if enemy.statuses.has(Status.FEAR):
            temporary_move(self, enemy, source=getattr(enemy, "fear_origin", None))
            self.mechanic_diagnostics["FEAR_TEMPORARY_NAVIGATION_CANDIDATE"] = 1
            return
        enemy.induced_path = None
        if self.vecbreak.enabled and self.vecbreak.trace_summon(enemy):
            return
        self._request_enemy_route_facing(enemy)
        self._move_enemy(enemy)
        if self.vecbreak.enabled: self.vecbreak.after_enemy_move(enemy)

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

        # NEXT skills may have been ready before this frame's cooldown update
        # but unable to cast then. Retry at the new attack opportunity, before
        # a normal attack renews that cooldown; do not tick SP/runtime twice.
        if op.skill_runtime is not None and op.skill_runtime.sid in NEXT:
            self._try_auto_skill(op)
            if self._skill_starting(op):
                return

        if op.skill_runtime is not None and op.skill_runtime.override_action():
            return

        if op.healing_runtime is not None and op.skill_prefab_id == SHU_S1 and op.skill_active:
            return

        if op.damage_type == "HEAL":
            if self._operator_heal(op):
                op.attack_cd = self._effective_attack_interval(op)
            return

        if op.attack_chain_active:
            if not self._cached_operator_target_is_valid(op):
                form_chain = (op.form_runtime is not None
                              and op.form_runtime.mode in ("BODY", "THANATOS"))
                if op.char_id != "char_1051_headb2" and not form_chain:
                    self._finish_operator_attack_chain(op)
                    return
                # Headb2's attack boundary consumes Search/Next even when
                # no replacement exists, resetting the three-frame ticker.
                self._force_refresh_operator_target(op)
                if not self._cached_operator_target_is_valid(op):
                    self._finish_operator_attack_chain(op)
                    return
                if form_chain:
                    # Both attacking forms keep their cadence when the old
                    # primary dies and another valid target is available.
                    first_attack = False
                else:
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
        return event.time <= self.time + 1e-9

    def run(self, max_time: float = 600.0) -> Result:
        while self.time < max_time:
            self.step()
            if self.end_reason:
                break
            if self.life_points <= 0:
                self.end_reason = "life_points_depleted"
                break
            if self.all_spawns_finished() and self.all_counted_enemies_finished():
                self.end_reason = "all_clear"
                break
        if not self.end_reason:
            self.end_reason = "timeout"
        if isinstance(getattr(self, "rng", None), RandomDecisionRng):
            self.rng.assert_consumed()
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
            behavior_components=copy.deepcopy(self.component_status),
            seed=self.seed,
            random_draws=self.rng.draw_count,
            assumptions=[
                ASSUMPTION_NOTES[key] for key in sorted(self.used_assumptions)
            ],
            spawn_timing=self.spawn_timing,
            enemy_attack_timing=self.enemy_attack_timing,
            operation_results=self.operation_log.summary(self.end_reason, self.time),
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

    @staticmethod
    def _skill_blocks_sp(op: _Operator) -> bool:
        # This selected skill explicitly allows recovery while affecting.
        return (op.skill_active and not op.skill_controls.can_switch
                and op.skill_prefab_id != "skchr_svash2_1")

    def _update_operator_hp_recovery(self, op: _Operator) -> None:
        if op.dead or op.hp_recovery_frame == self.frame_index:
            return
        op.hp_recovery_frame = self.frame_index
        rate = sum(runtime.passive_hp_recovery_for(op) for runtime in effect_runtimes(self))
        rate += sum(unit.healing_runtime.passive_hp_recovery_for(op) for unit in self.operators
                    if unit.healing_runtime is not None)
        rate += self._registered_attribute_addition(op, "HP_RECOVERY_PER_SEC")
        rate += op.max_hp * self._registered_attribute_addition(op, "HP_RECOVERY_PER_SEC_BY_MAX_HP_RATIO")
        if op.skill_runtime is not None and op.skill_runtime.anchor_move is not None:
            # The transporting mode scales both HP-recovery attributes to zero.
            rate = 0.0
        if rate <= 0:
            op.hp_recovery_interval = op.hp_recovery_remaining = 0.0
            return
        interval = 1.0 / rate
        if interval != op.hp_recovery_interval:
            # ResetButKeepPastTime keeps elapsed seconds, not fractional progress.
            past = op.hp_recovery_interval - op.hp_recovery_remaining
            op.hp_recovery_remaining = max(0.0, interval - past)
            op.hp_recovery_interval = interval
        op.hp_recovery_remaining -= self.dt
        if op.hp_recovery_remaining <= 1e-9:
            count = math.floor((1e-9 - op.hp_recovery_remaining) / interval) + 1
            op.hp_recovery_remaining += count * interval
            gain_hp(op, count)
        self.mechanic_diagnostics["OPERATOR_HP_RECOVERY_TIMER_PHASE_CANDIDATE"] = 1

    def _update_skill(self, op: _Operator) -> None:
        if op.skill_progress_locked:
            self.lock_skill_progress(op)
        if op.skill_runtime is not None:
            op.skill_runtime.tick()
        self._update_operator_hp_recovery(op)
        if op.form_runtime is not None:
            op.form_runtime.tick()
        if op.healing_runtime is not None:
            op.healing_runtime.tick_skill()
        if op.skill is None:
            return
        recovery_was_stopped = (
            self._skill_blocks_sp(op)
            or self._skill_starting(op)
            or op.statuses.has(Status.SP_BLOCKED)
        )
        if self._skill_starting(op):
            if self.time + 1e-9 < op.skill_startup_until:
                self._recover_squad_sp(op)
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
            and not self._skill_blocks_sp(op)
            and not self._skill_starting(op)
            and not op.statuses.has(Status.SP_BLOCKED)
        ):
            self._recover_time_sp(
                op, recovery_was_stopped=recovery_was_stopped
            )
        self._recover_squad_sp(op)
        self._try_auto_skill(op)

    def _try_auto_skill(self, op: _Operator) -> bool:
        """Check automatic permission without advancing any timer again."""
        if (
            not op.dead and self._skill_ready(op)
            and not op.skill_active
            and not self._skill_starting(op)
            and (op.skill_runtime.auto_ready() if op.skill_runtime else op.auto)
            and op.statuses.can_use_skill
            and not (op.skill_runtime is not None and op.skill_runtime.sid in NEXT
                     and op.statuses.has(Status.SP_BLOCKED))
        ):
            return self._cast_skill(op, automatic=True)
        return False

    def _recover_time_sp(
        self, op: _Operator, *, recovery_was_stopped: bool
    ) -> None:
        self._sync_squad_effects()
        attribute_bonus = self._registered_attribute_addition(op, "SP_RECOVERY_PER_SEC")
        # The observed S2 profile shares SpController's integer-count path.
        # Other rates/characters keep the explicit historical model until
        # their timer period and initial phase have been checked.
        if not (
            self.assumptions.headb2_periodic_sp_recovery
            and op.skill_prefab_id == "skchr_headb2_2"
            and math.isclose(self.dt, 1 / CLIENT_LOGIC_RATE, abs_tol=1e-12)
            and op.sp_increment == 1.0
            and op.squad_bonus.sp_recovery == 0.0
            and attribute_bonus == 0.0
        ):
            extra = op.skill_runtime.sp_recovery_bonus() if op.skill_runtime else 0
            self._gain_sp(op, (op.sp_increment + op.squad_bonus.sp_recovery + extra + attribute_bonus) * self.dt)
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

    def _recover_squad_sp(self, op: _Operator) -> None:
        interval = self.squad_effects.sp_interval
        if interval <= 0 or op.dead:
            return
        # Per-target phase: first tick is deployment + interval; locked/full ticks
        # are consumed, rather than queued for the next unlocked update.
        ticks = math.floor((self.time - op.deployed_at + 1e-9) / interval)
        count = max(0, ticks - op.squad_sp_ticks)
        op.squad_sp_ticks = ticks
        if count:
            self._gain_sp(op, count * self.squad_effects.sp_amount)

    def _cast_skill(self, op: _Operator, *, switch_mode: bool = False,
                    automatic: bool = False) -> bool:
        if self.mansfield.enabled and op.char_id == MANSFIELD_DEVICE:
            return self.mansfield.cast_device(op)
        if op.form_runtime is not None and op.form_runtime.mode == "THANATOS":
            return op.form_runtime.manual_switch()
        level = op.skill
        if (level is None or self._skill_starting(op)
                or (op.skill_active and not (switch_mode and op.skill_controls.can_switch))):
            return False
        no_charge = op.sp_type in ("PASSIVE", "NO_CHARGE", "INCREASE_WHEN_NONE")
        if not no_charge and not self._skill_ready(op):
            return False
        if not op.statuses.can_use_skill and not no_charge:
            return False
        if op.skill_runtime is not None and not op.skill_runtime.can_cast():
            return False
        if op.skill_runtime is not None:
            op.skill_runtime.cast_automatic = automatic
        if op.healing_runtime is not None and op.skill_prefab_id == SHU_S1:
            return op.healing_runtime.start_s1()
        # Fever may cast without enough SP, but still spends any stored SP;
        # its forced after-attack recovery is a separate notification.
        if not no_charge:
            op.sp = max(0.0, op.sp - self._skill_cost(op))
        if op.skill_controls.can_switch:
            op.skill_mode_target = 1 - op.skill_mode
        # The ward activates its ability once; Wisdel S1 arms its next attack.
        # Their runtimes register the actual spell, not this early bookkeeping.
        if not (self.vecbreak.enabled and op.skill_prefab_id in {
                "sktok_wisdel_wward", "skchr_wisdel_1"}):
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
        if op.skill_controls.can_switch and op.skill_mode_target == 0:
            self._finish_skill(op, "MODE_SWITCH")
            return True
        if level is None or op.skill_active:
            return False
        self.squad_effects.skill_started_by(op.char_id)
        self._squad_field_signature = None
        if op.skill_runtime is not None and op.skill_runtime.sid in ROSTER_SKILLS:
            activated = op.skill_runtime.activate()
            if activated:
                self._sync_component_skill_mode(op, active=True)
            return activated
        if op.form_runtime is not None:
            op.form_runtime.begin("THANATOS", "SKILL")
            op.skill_end_reason = "FORM_SWITCH"
            return True
        if op.skill_prefab_id == "skchr_headb2_2":
            self._headb2_s2_reset_attack(op)
        op.skill_active = True
        self._sync_squad_effects()
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
        if op.char_id == "char_1035_wisdel" and op.skill_runtime is not None:
            op.skill_runtime.soul_ward_skill_started()
        if op.skill_controls.can_switch:
            op.skill_mode = 1
            op.skill_mode_target = None
        self._sync_component_skill_mode(op, active=True)
        if op.skill_controls.can_end or op.skill_controls.can_switch:
            # Control phases and ability-specific finish effects need their
            # own validation; the description only establishes permission.
            self.mechanic_diagnostics[f"SKILL_CONTROL_TRANSITION_CANDIDATE:{op.skill_prefab_id}"] = 1
            if op.behavior_template_key is None:
                self.mechanic_diagnostics[f"SKILL_CONTROL_EFFECTS_INCOMPLETE:{op.skill_prefab_id}"] = 1
        if (duration == -1.0 and op.skill_duration_type != "AMMO"
                and not op.skill_controls.can_end and not op.skill_controls.can_switch):
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
        elif op.skill_controls.can_switch or (duration == -1.0 and op.skill_controls.can_end):
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
        talent = op.talent_blackboards.get("2", {})
        self_scale = float(talent.get("scale_bonus", 1.0))
        if op.skill_cast_count >= 2:
            out["atk"] = float(out.get("headb2_s_2[second].atk", 0.0))
            out["def"] = float(out.get("headb2_s_2[second].def", 0.0))
            duration = -1.0
        out["atk"] = float(out.get("atk", 0.0)) + float(talent.get("atk", 0.0)) * self_scale
        out["def"] = float(out.get("def", 0.0)) + float(talent.get("def", 0.0)) * self_scale
        return out, duration

    def _skill_cost(self, op: _Operator) -> float:
        return op.sp_cost if op.sp_cost > 0 else op.sp_max

    def _skill_ready(self, op: _Operator) -> bool:
        if op.skill_runtime is not None and op.skill_runtime.free_cast():
            return True
        cost = self._skill_cost(op)
        return cost <= 0 or op.sp + 1e-9 >= cost

    def _gain_sp(self, op: _Operator, amount: float) -> float:
        if (
            amount <= 0
            or self._skill_blocks_sp(op)
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

    def configure_skill_ammo(
        self, op: _Operator, capacity: float, *, cost_per_attack: float = 1.0
    ) -> None:
        capacity += sum(runtime.ammo_capacity_bonus(op) for runtime in live_runtimes(self))
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
            for runtime in live_runtimes(self):
                runtime.ammo_used(op, cost)
        return op.ammo

    def interrupt_skill(self, op: _Operator, reason: str = "INTERRUPTED") -> bool:
        if not op.skill_active:
            return False
        self._finish_skill(op, reason)
        return True

    def _finish_skill(self, op: _Operator, reason: str) -> None:
        if not op.skill_active:
            return
        self.unlock_skill_progress(op)
        if op.skill_runtime is not None:
            op.skill_runtime.finish(reason)
        if not op.behavior_finish_dispatched:
            self._dispatch_skill_event(op, "ON_SKILL_FINISH")
            op.behavior_finish_dispatched = True
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
        op.skill_mode = 0
        self._sync_component_skill_mode(op, active=False)
        self._sync_squad_effects()
        op.skill_mode_target = None
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
            str(item.get("key")): item.get("valueStr") if item.get("valueStr") is not None else item.get("value")
            for item in level.get("blackboard", [])
            if item.get("key") is not None
        }

    def execute_behavior_event(
        self, template_key: str, event: str, *, source: Any,
        target: Any = None, blackboard: dict[str, Any] | None = None,
        buff_owner: Any = None, modifier: DamagePacket | None = None,
        random_mechanism: str | None = None,
        ability_family: str | None = None, projectile_selector: Any = None,
        projectile_key: str | None = None,
        range_buff_sink: Any = None,
        register_buffs: bool = False, refresh_attribute_buffs: bool = False,
        validate_only: bool = False,
        buff_instance: BuffInstance | None = None,
    ) -> BehaviorContext | None:
        """Shared event adapter for explicitly migrated actor effects."""
        if not self.behavior.validate_event(template_key, event):
            return None
        context = BehaviorContext(source=source, target=target,
            buff_owner=buff_owner, buff_source=source,
            modifier_source=modifier.source if modifier is not None else None,
            modifier_target=target if modifier is not None else None,
            blackboard=dict(blackboard or {}), battle=self,
            modifier=modifier, random_mechanism=random_mechanism,
            ability_family=ability_family, projectile_selector=projectile_selector,
            projectile_key=projectile_key, range_buff_sink=range_buff_sink,
            register_buffs=register_buffs, refresh_attribute_buffs=refresh_attribute_buffs)
        if buff_instance is not None:
            context.buff_instance = buff_instance
            context.buff_container = getattr(buff_instance.owner, "buffs", None)
            context.current_buff_key = buff_instance.definition.key
        context.template_key, context.event = template_key, event
        if not self._validate_behavior_bindings(context, self.behavior.templates[template_key]["eventToActions"][event]):
            return None
        if not validate_only:
            self.behavior.dispatch(template_key, event, context)
        return context

    def _validate_behavior_bindings(self, context: BehaviorContext, actions: Any) -> bool:
        """Check event bindings before callbacks, including registered snapshots."""
        template_key, event = context.template_key, context.event
        projectile_key, projectile_selector = context.projectile_key, context.projectile_selector
        register_buffs = context.register_buffs or (context.buff_instance is not None
            and context.buff_instance.definition.registered_behavior)
        # Selector bindings are actor adapters; reject a missing binding before
        # the event's Dice consumes a decision or any action changes state.
        def bound(value):
            if isinstance(value, dict):
                if node_name(value) in {"SetMaxMinusHpRatio", "CheckIfAtMinusHpState", "CheckAbnormalFlag", "CheckAbnormalImmune"}:
                    actor = context.entity(value.get("_targetType"))
                    if actor is not None and not isinstance(actor, (_Enemy, _Operator)):
                        return self.behavior._unsupported(f"{template_key}:{event}:hp_controller_binding")
                    if node_name(value) == "SetMaxMinusHpRatio" and actor is not None:
                        if not math.isfinite(actor.max_hp * value["_value"]):
                            return self.behavior._unsupported(f"{template_key}:{event}:hp_capacity_overflow")
                if node_name(value) in {"LockCharacterSkillProgress", "UnlockCharacterSkillProgress"}:
                    actor = context.entity(value.get("_targetType"))
                    if actor is not None and (not isinstance(actor, _Operator) or actor.skill is None):
                        return self.behavior._unsupported(f"{template_key}:{event}:skill_progress_owner_binding")
                    if isinstance(actor, _Operator) and self.component_catalog.skill_progress_hooker(actor.skill_prefab_id) is None:
                        return self.behavior._unsupported(f"{template_key}:{event}:skill_progress_hooker_binding")
                    if (node_name(value) == "LockCharacterSkillProgress" and actor is not None and actor.skill_active
                            and (actor.skill_duration_type == "AMMO" or actor.active_until <= 0)):
                        return self.behavior._unsupported(f"{template_key}:{event}:skill_progress_kind_binding")
                if node_name(value) == "FinishAllBuffsWithCertainAbnormalFlag":
                    if issue := self._flag_finish_binding_issue(context.entity(value.get("_targetType")), value["_abnormalFlag"]):
                        return self.behavior._unsupported(f"{template_key}:{event}:{issue}")
                if node_name(value) == "ModifySp":
                    owner = context.entity(value.get("_targetType"))
                    if owner is not None and not isinstance(owner, _Operator):
                        return self.behavior._unsupported(f"{template_key}:{event}:sp_controller_binding")
                if node_name(value) == 'NoSourceDamage' and context.target is not None and not isinstance(context.target, (_Enemy, _Operator)):
                    return self.behavior._unsupported(f"{template_key}:{event}:damage_target_binding")
                if node_name(value) == "CheckContainsBuff" and value.get("_checkBuffSource"):
                    source = context.entity(value.get("_buffSourceType"))
                    if not isinstance(source, (_Enemy, _Operator)):
                        return self.behavior._unsupported(f"{template_key}:{event}:buff_source_binding")
                if node_name(value) == "EmitProjectileUseAbilitySelector":
                    key = value.get("_projectileKey") or projectile_key
                    if projectile_selector is None or not key or self._projectile_speed(key) is None:
                        return self.behavior._unsupported(f"{template_key}:{event}:projectile_binding")
                    emitter = context.entity(value.get("_sourceType"))
                    if not isinstance(emitter, (_Enemy, _Operator)):
                        return self.behavior._unsupported(f"{template_key}:{event}:projectile_emitter")
                    if isinstance(emitter, _Enemy) and value.get("_useMountPointGroup"):
                        return self.behavior._unsupported(f"{template_key}:{event}:enemy_mount_group")
                if node_name(value) == "CreateBuffInRange":
                    if not self._validate_range_buff_binding(context, value):
                        return False
                if node_name(value) == "AOEDamage" and not self._validate_aoe_binding(context, value):
                    return False
                if node_name(value) == "CreateBuff":
                    spec = value.get("_buff") or {}
                    attributes = (spec.get("attributes") or {}).get("attributeModifiers")
                    if value.get("_isDerivedBuff") and not self._validate_derived_buff_binding(context, value):
                        return False
                    if attributes and not register_buffs:
                        return self.behavior._unsupported(f"{template_key}:{event}:attribute_registration_binding")
                    if register_buffs:
                        owner = context.entity(value.get("_buffOwner"))
                        emitter = (context.entity(value.get("_specialBuffSource"))
                                   if value.get("_useSpecialBuffSource") else context.source)
                        if not isinstance(owner, (_Enemy, _Operator)) or not isinstance(emitter, (_Enemy, _Operator)):
                            return self.behavior._unsupported(f"{template_key}:{event}:buff_entity_binding")
                        if self._validated_buff_events(spec, blackboard=context.blackboard,
                                owner=owner, source=emitter) is None:
                            return False
                if node_name(value) in {"CheckContainsDerviedBuff", "FinishBuff"}:
                    if not self._validate_derived_buff_binding(context, {"_buffOwner": "BUFF_OWNER"}):
                        return False
                return all(bound(child) for child in value.values())
            if isinstance(value, list):
                return all(bound(child) for child in value)
            return True
        def validate_node(node, board):
            previous = context.blackboard
            context.blackboard = board
            try:
                return None if bound(node) else "event_binding"
            finally:
                context.blackboard = previous
        issue = numeric_issue(actions, context.blackboard, node_validator=validate_node,
            blackboard_reader=lambda node, board: self._read_preflight_blackboard(node, context, board),
            actor_preview=self._behavior_attribute_preview(context))
        if issue == "event_binding":
            return False
        return not issue or self.behavior._unsupported(f"{template_key}:{event}:{issue}")

    def _behavior_attribute_preview(self, context, *, root_definition=None, event=None):
        committing = (context.buff_owner if root_definition is not None and event in {
            "ON_BUFF_START", "ON_BUFF_FINISH", "ON_BUFF_ENABLE", "ON_BUFF_DISABLE"} else None)
        return BuffAttributePreview(self, context, root_definition=root_definition, committing_owner=committing)

    def _read_preflight_blackboard(self, node, context, board):
        preview = getattr(board, "actor_preview", None)
        if preview is None:
            return read_blackboard_value(node, context, board)
        name = node_name(node)
        if (name == "AssignBuffBlackboardFromOthers" and preview.changed
                or preview.dirty and name in {"AssignBuffBlackboardFromOthers", "AssignCharacterSkillBlackboardToBB"}):
            raise ValueError("blackboard_after_actor_mutation_binding")
        return read_blackboard_value(node, context, board,
            entity_value_reader=lambda actor, event, attribute:
                self._preview_entity_value(preview, actor, event, attribute))

    def _preview_entity_value(self, preview, actor, node, attribute):
        if preview.dirty:
            raise ValueError("after_actor_mutation_binding")
        if not isinstance(actor, (_Operator, _Enemy)):
            raise ValueError("attribute_entity_binding")
        if node != "AssignAttributeToBB" or attribute == "MAX_HP":
            return self.behavior_entity_value(actor, node, attribute)
        source_limit = (1 if isinstance(actor, _Enemy) and attribute in {"ATK", "DEF"}
            and preview.contains(actor, "enemy_9D0_talent_strength[attribute]")
            and not self.assumptions.herald_auras_stack else None)
        modifiers = preview.modifiers(actor, attribute,
            registered_only=(isinstance(actor, _Operator) and attribute not in {
                "ONE_MINUS_STATUS_RESISTANCE", "SLOW_DOWN", "EP_BREAK_RECOVER_SPEED"}) or attribute == "MAGIC_RESISTANCE",
            source_limit=source_limit)
        return self.behavior_entity_value(actor, node, attribute, modifiers=modifiers)

    @staticmethod
    def behavior_component_claimed(actor: Any, template: str) -> bool:
        return template in getattr(actor, "component_claims", ())

    def _dispatch_component_calculate(self, packet: DamagePacket, target: Any, phase: str):
        """Shared outgoing event hook; no character or template-key selection."""
        source = packet.source
        if (not packet.notify_calculate or packet.normalized_damage_type() in {DamageType.HEAL, DamageType.LIFE_LOSS}
                or not isinstance(source, (_Enemy, _Operator))
                or not isinstance(target, (_Enemy, _Operator))
                or isinstance(source, _Enemy) == isinstance(target, _Enemy) or target.dead):
            return packet
        for listener in source.component_listeners:
            if listener.instance is not None and (listener.instance.finished
                    or source.buffs._is_pending_finish(listener.instance)):
                continue
            if listener.phase != phase or (listener.ignore_dead and source.dead):
                continue
            if not self.component_catalog.validate(self, listener):
                listener.record.update(state="rejected", reason="preflight")
                if listener.instance is not None:
                    source.buffs.finish(listener.instance)
                continue
            context = self.execute_behavior_event(listener.template, CALCULATE,
                source=source, target=target, buff_owner=source,
                blackboard=listener.board, modifier=packet)
            if context is not None and phase == "scale":
                packet = replace(packet, amount=packet.amount * context.attack_scale * context.damage_scale)
        return packet

    def _dispatch_component_lifecycle(self, actor: Any, event: str, *, listeners=None) -> None:
        """Owner-only passive notifications shared by any attached actor."""
        for listener in tuple(actor.component_listeners if listeners is None else listeners):
            if listener.finished:
                continue
            if listener.runtime_spec is not None:
                if event == "ON_BUFF_START":
                    if not listener.started:
                        definition = self._behavior_buff_definition(listener.runtime_spec, listener.board)
                        if definition is None:
                            listener.record.update(state="rejected", reason="buff_preflight")
                            continue
                        if not self._validated_buff_owner_values(definition, actor):
                            listener.record.update(state="rejected", reason="buff_owner_preflight")
                            listener.finished = True
                            if listener in actor.component_listeners:
                                actor.component_listeners.remove(listener)
                            continue
                        actor.buffs.owner = actor
                        actor.buffs.event_handler = self._dispatch_buff_event
                        actor.buffs.change_handler = self._registered_attributes_changed
                        listener.instance = actor.buffs.add(definition, source=actor,
                            blackboard=listener.board, attached_step=self.frame_index)
                        listener.instance.component_listener = listener
                        listener.record["clock"] = {"lifetime_type": definition.lifetime_type,
                            "lifetime": definition.lifetime, "trigger_interval": definition.trigger_interval,
                            "first_trigger_interval": definition.first_trigger_interval,
                            "trigger_limit": definition.trigger_limit}
                        listener.record["triggers"] = {"completed": 0,
                            "remaining": listener.instance.remaining_triggers}
                        actor.buffs.commit()
                elif listener.instance is not None:
                    if event == "ON_BUFF_FINISH":
                        actor.buffs.finish(listener.instance)
                        actor.buffs.commit()
                    else:
                        self._dispatch_buff_event(listener.instance, event)
                continue
            if listener.phase not in {"lifecycle", "owner-events"}:
                continue
            if event == "ON_BUFF_START":
                if listener.started:
                    continue
                # Field re-entry/relocation is not a new buff instance.
                listener.started = True
            elif not listener.started:
                continue
            if event not in listener.events or (listener.ignore_dead and actor.dead):
                continue
            if not self.component_catalog.validate(self, listener):
                listener.record.update(state="rejected", reason="preflight")
                continue
            context = self.execute_behavior_event(listener.template, event, source=actor,
                buff_owner=actor, blackboard=listener.board)
            if context is None:
                listener.record.update(state="rejected", reason="event_binding")

    def _sync_component_skill_mode(self, op: _Operator, *, active: bool) -> None:
        if op.component_talent_selection is None:
            return  # The selected talent/module blackboards are not ready yet.
        mode = self.component_catalog.explicit_skill_mode(op, active=active)
        if mode is not None:
            self.component_catalog.switch_operator_mode(self, op, mode)

    def _notify_operator_attack_complete(self, op: _Operator, *, legacy_runtime=False) -> None:
        # Reuse the existing final attack callback, never projectile impact or
        # AttackState cleanup (which also runs on interruption/mode detach).
        # Exact native post-delay/OnCastEnd placement remains a candidate.
        self._dispatch_component_lifecycle(op, "ON_AFTER_ATTACK")
        if legacy_runtime and op.skill_runtime is not None:
            op.skill_runtime.attack_complete()

    def _dispatch_component_output(self, packet: DamagePacket, target: Any) -> None:
        source = packet.source
        if (packet.normalized_damage_type() is not DamageType.HEAL
                or not isinstance(source, (_Enemy, _Operator)) or source.dead
                or not isinstance(target, (_Enemy, _Operator)) or target.dead):
            return
        for listener in tuple(source.component_listeners):
            if listener.instance is not None and (listener.instance.finished
                    or source.buffs._is_pending_finish(listener.instance)):
                continue
            if listener.phase != "heal-output":
                continue
            if not self.component_catalog.validate(self, listener):
                listener.record.update(state="rejected", reason="preflight")
                if listener.instance is not None:
                    source.buffs.finish(listener.instance)
                continue
            context = self.execute_behavior_event(listener.template, OUTPUT, source=source,
                target=target, buff_owner=source, blackboard=listener.board, modifier=packet)
            if context is None:
                listener.record.update(state="rejected", reason="event_binding")

    def _validate_derived_buff_binding(self, context: BehaviorContext, node: dict[str, Any]) -> bool:
        parent = context.buff_instance
        owner = context.entity(node.get("_buffOwner"))
        container = getattr(owner, "buffs", None)
        parent_container = parent.container if parent is not None else None
        if (not isinstance(owner, (_Enemy, _Operator)) or owner.dead or container is None
                or parent is None or context.buff_container is not parent_container
                or parent_container is not getattr(parent.owner, "buffs", None)
                or getattr(parent.owner, "dead", False)
                or not parent.definition.registered_behavior or not BuffContainer.live_parent(parent)):
            return self.behavior._unsupported("CreateBuff:derived_parent_binding")
        source = (context.entity(node.get("_specialBuffSource"))
                  if node.get("_useSpecialBuffSource") else context.source)
        if not isinstance(source, (_Enemy, _Operator)):
            return self.behavior._unsupported("CreateBuff:derived_source_binding")
        return True

    def _validate_range_buff_binding(self, context: BehaviorContext, node: dict[str, Any]) -> bool:
        """Resolve the range and every child tree before any event side effects."""
        label = f"{context.template_key or '<range>'}:{context.event or '<direct>'}:CreateBuffInRange"
        if not self._validate_behavior_range(context, node, label):
            return False
        if node.get("_isDerivedBuff"):
            if context.buff_instance is not None:
                if not self._validate_derived_buff_binding(context, {"_buffOwner": "BUFF_OWNER"}):
                    return False
            elif not callable(context.range_buff_sink):
                return self.behavior._unsupported(label + ":derived_lifecycle_binding")
        source = context.entity(node.get("_sourceType"))
        # Validate even an empty selection; then bind SP and child receivers
        # for every actual recipient before the first registration.
        specs = node.get("_buffs") or []
        receiver_kind = (type(source) if (node.get("_targetOptions") or {}).get("targetSide") == "ALLY"
                         else {_Enemy: _Operator, _Operator: _Enemy}.get(type(source)))
        return (all(self._validated_buff_events(spec, blackboard=context.blackboard,
                    owner_kind=receiver_kind, source=source,
                    derived=bool(node.get("_isDerivedBuff") and context.buff_instance is not None)) is not None for spec in specs)
            and all(self._validated_buff_events(spec, blackboard=context.blackboard,
                    owner=owner, source=source) is not None
                for owner in self.behavior_range_targets(context, node) for spec in specs))

    def _validate_behavior_range(self, context: BehaviorContext, node: dict[str, Any], label: str,
                                 *, source_kind: type | None = None, center_kind: type | None = None) -> bool:
        source = context.entity(node.get("_sourceType"))
        center = context.entity(node.get("_targetType"))
        source_kind = type(source) if source is not None else source_kind
        center_kind = type(center) if center is not None else center_kind
        if source_kind not in {_Enemy, _Operator} or center_kind not in {_Enemy, _Operator}:
            return self.behavior._unsupported(label + ":entity_binding")
        if node.get("_useRadius"):
            try:
                radius = float(node.get("_radius", 0))
            except (TypeError, ValueError):
                return self.behavior._unsupported(label + ":radius")
            if not math.isfinite(radius) or radius < 0:
                return self.behavior._unsupported(label + ":radius")
        elif not node.get("_useGlobalRange"):
            range_id = node.get("_rangeId")
            if not range_id or range_id not in self.ranges:
                return self.behavior._unsupported(label + ":range_binding")
            if center_kind is _Enemy:
                # Enemy facing is continuous, not an exported cardinal range
                # orientation. Only rotation-invariant named ranges bind here.
                cells = self.range_cells_from_id(range_id, (0, 0), 0)
                if any(self.range_cells_from_id(range_id, (0, 0), facing) != cells for facing in (1, 2, 3)):
                    return self.behavior._unsupported(label + ":enemy_range_facing")
        return True

    def _validate_aoe_binding(self, context: BehaviorContext, node: dict[str, Any]) -> bool:
        if context.template_key == "huang_s_3" and not node.get("_useRadius") and not node.get("_rangeId"):
            return isinstance(context.entity(node.get("_sourceType")), _Operator)
        return self._validate_behavior_range(context, node,
            f"{context.template_key or '<aoe>'}:{context.event or '<direct>'}:AOEDamage")

    def behavior_range_targets(self, context: BehaviorContext, node: dict[str, Any]) -> list[Any]:
        """Reviewed entry selection: side, availability, motion and midpoint range."""
        source = context.entity(node.get("_sourceType"))
        center = context.entity(node.get("_targetType"))
        options = node.get("_targetOptions") or {"targetSide": "ENEMY"}
        allied = options.get("targetSide") == "ALLY"
        operator_side = isinstance(source, _Operator) == allied
        candidates = self.operators if operator_side else self.enemies
        advanced = options.get("enableAdvancedOptions", False)
        check_free = advanced and options.get("purposeMask", "NONE") != "NONE"
        ignore_free = advanced and (options.get("ignoreTargetFree") or
                                     (allied and options.get("ignoreAllyTargetFree")))
        position = center.tile if isinstance(center, _Operator) else center.position()
        cells = None
        if not node.get("_useRadius") and not node.get("_useGlobalRange"):
            cell = center.tile if isinstance(center, _Operator) else center.tile()
            cells = self.range_cells_from_id(node["_rangeId"], cell,
                center.facing if isinstance(center, _Operator) else 0)
        result = []
        for unit in candidates:
            if unit.dead or (isinstance(unit, _Enemy) and not self._enemy_is_attackable(unit)):
                continue
            if node.get("_excludeTarget") and unit is center:
                continue
            if check_free and not ignore_free:
                available = self._enemy_targetable(unit) if isinstance(unit, _Enemy) else unit.statuses.targetable
                if not available:
                    continue
            # No supported purpose is healing. ignoreHealFree does not make
            # an attribute/status effect a healing selector.
            airborne = self._is_airborne(unit) if isinstance(unit, _Enemy) else unit.statuses.has(Status.LEVITATE)
            motion = options.get("targetMotion", "ALL")
            if (motion == "WALK" and airborne) or (motion == "FLY" and not airborne):
                continue
            point = unit.tile if isinstance(unit, _Operator) else unit.position()
            if node.get("_useRadius"):
                if math.dist(point, position) > float(node.get("_radius", 0)) + 1e-9:
                    continue
            elif cells is not None:
                cell = unit.tile if isinstance(unit, _Operator) else unit.tile()
                if cell not in cells:
                    continue
            result.append(unit)
        return result

    def behavior_create_buff_in_range(self, context: BehaviorContext, node: dict[str, Any]) -> bool:
        if not self._validate_range_buff_binding(context, node):
            return False
        source = context.entity(node.get("_sourceType"))
        if source.dead:
            return False
        targets = self.behavior_range_targets(context, node)
        parent = context.buff_instance if node.get("_isDerivedBuff") else None
        for owner in targets:
            for spec in node.get("_buffs") or []:
                instance = self.register_behavior_buff(spec, owner=owner, source=source,
                    blackboard=context.blackboard, parent=parent, finish_with_parent=parent is not None)
                if (instance is not None and not instance.finished and node.get("_isDerivedBuff")
                        and parent is None and context.range_buff_sink is not None):
                    context.range_buff_sink(owner, instance)
        return bool(targets)

    def behavior_advanced_damage(self, context: BehaviorContext, node: dict[str, Any]) -> bool:
        """One reviewed hit packet; keep actor-specific damage notifications."""
        source = context.entity(node.get("_sourceType"))
        target = context.entity(node.get("_targetType"))
        if source is None or target is None or target.dead:
            return False
        if isinstance(target, _Enemy) and not self._enemy_is_attackable(target):
            return False
        base = context.cached_attack
        if base is None:
            base = self.operator_attack_power(source) if isinstance(source, _Operator) else self._effective_enemy_atk(source)
        scale = float(context.blackboard.get(node.get("_atkScaleVar", "atk_scale"), node.get("_defaultAtkScale", 1)))
        return self._behavior_hit_damage(source, target, base * scale,
            node.get("_damageType", "PHYSICAL"), node.get("_attackType", "NORMAL"),
            node.get("_applyWay", "NONE"), node.get("_modifierKey", ""))

    def behavior_no_source_damage(self, context: BehaviorContext, node: dict, amount: float) -> bool:
        """A target-side modifier with no character output or kill ownership."""
        target = context.target
        if target is None or target.dead:
            return False
        if not isinstance(target, (_Enemy, _Operator)):
            return self.behavior._unsupported('NoSourceDamage:damage_target_binding')
        kind = {'MAGICAL': 'ARTS', 'PURE': 'TRUE', 'ELEMENT': 'ELEMENTAL'}.get(
            node.get('_damageType', 'PHYSICAL'), node.get('_damageType', 'PHYSICAL'))
        packet = DamagePacket(amount, kind, attack_type=node.get('_attackType', 'NORMAL'),
            lethal=not node.get('_isUndeadable', False), modifier_key=node.get('_modifierKey', ''),
            target_defense_override=self._effective_enemy_defense(target) if isinstance(target, _Enemy)
                else self.operator_defense(target), target_resistance_override=self._effective_resistance(target))
        result = self._apply_damage(packet, target)
        if result.killed and isinstance(target, _Enemy):
            self.enemies_killed += 1
        if (not node.get('_ignoreForSp', False) and result.final_amount > 0 and isinstance(target, _Operator)
                and target.skill is not None and target.sp_type == 'INCREASE_WHEN_TAKEN_DAMAGE'):
            self._gain_sp(target, target.sp_increment)
        return result.final_amount > 0

    def _behavior_hit_damage(self, source: Any, target: Any, amount: float,
                             damage_type: str, attack_type: str, apply_way: str,
                             modifier_key: str) -> bool:
        kind = {"MAGICAL": "ARTS", "PURE": "TRUE", "ELEMENT": "ELEMENTAL"}.get(
            damage_type, damage_type)
        attack_type = AttackType(attack_type)
        apply_way = ApplyWay(apply_way)
        if isinstance(source, _Operator) and isinstance(target, _Enemy):
            # This is the same output adapter used before template migration.
            runtime = source.skill_runtime
            if runtime is not None:
                runtime.damage(target, amount, kind, attack_type=attack_type,
                    apply_way=apply_way, modifier_key=modifier_key)
            else:
                self._apply_operator_damage(source, target, amount, damage_type=kind,
                    attack_type=attack_type, apply_way=apply_way, modifier_key=modifier_key)
        else:
            result = self._apply_damage(DamagePacket(amount, kind, source=source,
                attack_type=attack_type, apply_way=apply_way, modifier_key=modifier_key,
                output_scale=self._outgoing_damage_scale(source),
                target_resistance_override=self._effective_resistance(target)), target)
            if result.killed and isinstance(target, _Enemy):
                self.enemies_killed += 1
        # A cancelled or zero packet still permits the next hit action.
        return True

    def behavior_apply_hit_buffs(self, target: Any, specs: list[dict[str, Any]],
                                 context: BehaviorContext) -> None:
        """Apply the caller-validated built-ins after its damage actions."""
        statuses = {"stun": Status.STUNNED, "sluggish": Status.SLUGGISH}
        for spec in specs:
            board = {**context.blackboard, **self._blackboard(spec)}
            duration = float(board.get(spec.get("durationKey") or spec["buffKey"], spec.get("lifeTime", 0)))
            target.statuses.apply(statuses[spec["buffKey"]], duration)

    def behavior_emit_projectile(self, context: BehaviorContext, node: dict[str, Any]) -> bool:
        source = context.entity(node.get("_sourceType"))
        key = node.get("_projectileKey") or context.projectile_key
        if source is None or context.projectile_selector is None or not key:
            return self.behavior._unsupported("EmitProjectileUseAbilitySelector:projectile_binding")
        if isinstance(source, _Enemy) and node.get("_useMountPointGroup"):
            return self.behavior._unsupported("EmitProjectileUseAbilitySelector:enemy_mount_group")
        group = node.get("_mountPointGroup") or {}
        mount_rule = {"group": list(group.get("mountPoints") or []) if node.get("_useMountPointGroup") else [],
            "loopType": group.get("loopType", "SEQUENTIAL"), "mount": node.get("_mountPoint")}
        count = max(0, int(context.blackboard.get("emit_count", node.get("_emitCount", 1))))
        base = (self.operator_attack_power(source) if isinstance(source, _Operator)
                else self._effective_enemy_atk(source)) if node.get("_cacheAtkToActions") else None
        emitted = False
        for _ in range(count):
            target = context.projectile_selector()
            if target is None:
                break
            payload = {"template_key": context.template_key, "event": node["_ev"],
                "actions": copy.deepcopy(node.get("_actions") or []),
                "buffs": copy.deepcopy(node.get("_buffDataList") or []),
                "blackboard": dict(context.blackboard), "cached_attack": base}
            attack = _PendingOperatorAttack(self.time, self.frame_index, "BEHAVIOR", target, [target],
                1, 1, True, True, profile=AttackProfile(payload={"keep_projectile_after_departure": True}))
            self._new_projectile(key=key, side="OPERATOR" if isinstance(source, _Operator) else "ENEMY",
                source=source, target=target, operator_attack=attack, cached_atk=base,
                mount_rule=mount_rule, behavior_payload=payload)
            emitted = True
        return emitted

    def land_behavior_projectile(self, projectile: _Projectile) -> None:
        payload = projectile.behavior_payload
        target = projectile.target
        if payload is None or not self._projectile_target_valid(projectile) or not projectile.will_hit:
            return
        context = BehaviorContext(source=projectile.source, target=target,
            buff_owner=projectile.source, buff_source=projectile.source,
            blackboard=dict(payload["blackboard"]), battle=self,
            template_key=payload["template_key"], event=payload["event"],
            cached_attack=payload["cached_attack"])
        self.behavior.execute_actions(payload["actions"], context)
        if target.dead:
            return
        self.behavior_apply_hit_buffs(target, payload["buffs"], context)

    def behavior_modify_sp(self, context: BehaviorContext, node: dict[str, Any]) -> bool:
        """Use the existing SP controller for reviewed additive/ratio variants."""
        owner = context.entity(node.get("_targetType"))
        if owner is None or getattr(owner, "dead", False):
            return False
        if not isinstance(owner, _Operator):
            return self.behavior._unsupported("ModifySp:missing_sp_controller")
        if owner.skill is None:
            return False
        masks = {"NONE": 0, "INCREASE_WITH_TIME": 1, "INCREASE_WHEN_ATTACK": 2,
                 "INCREASE_WHEN_TAKEN_DAMAGE": 4, "ATTACK_OR_DAMAGE": 6, "ALL": 7}
        if not node.get("_dontCheckSpType"):
            mask = node.get("_spMask", "ALL")
            if mask not in masks and (type(mask) is not int or not 0 <= mask <= 7):
                return self.behavior._unsupported(f"ModifySp:spMask:{mask}")
            mask = masks.get(mask, mask)
            if not (int(mask) & masks.get(owner.sp_type, 0)):
                return False
        key = str(node.get("_spString", ""))
        if node.get("_modifyByRatio"):
            amount = rounded_sp_ratio(int(owner.sp_max),
                context.blackboard.get(key, node.get("_modifyRatio", 0)))
        elif node.get("_modifyByRatioBasedOnCurSP"):
            # The current-SP getter returns FP directly; retain its fraction.
            amount = rounded_sp_ratio(owner.sp, context.blackboard.get(key, 0))
        elif node.get("_modifyBySpCost"):
            amount = self._skill_cost(owner)
        else:
            amount = math.floor(float(context.blackboard.get(key, node.get("_modifyValue", 0))))
        if amount < 0:
            return self.behavior._unsupported("ModifySp:negative_value")
        # Keep caps, fractional remainders and the existing notification phase.
        if node.get("_forceFlag"):
            self._store_sp_gain(owner, amount)
        else:
            self._gain_sp(owner, amount)
        return True

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
        if (not template_key or not getattr(op, "behavior_active", False)
                or self.behavior_component_claimed(op, template_key)):
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
    def _operator_heal(self, op: _Operator) -> bool:
        if self.frame_index < (op.selector_first_search_frame or 0):
            return False
        targets = self._operator_heal_targets(op)
        if not targets:
            return False
        self._begin_operator_heal(op, targets[:1], op.next_attack_scale)
        op.next_attack_scale = 1.0
        return True

    def _operator_heal_targets(
        self, op: _Operator, *, cells: set[tuple[int, int]] | None = None,
        hp_ratio: float = 1.0, include_equal: bool = False,
        include_element_damage: bool = False,
    ) -> list[_Operator]:
        cells = op.range_cells if cells is None else cells
        heal_free_exception = self.attack_timing.get(op.char_id, {}).get("heal_free_target_exception")
        targets = [target for target in self.operators
                   if not target.dead and (target.hp < target.max_hp or
                       (include_element_damage and target.elements.current_damaged_element is not None))
                   and target.tile in cells
                   and (not target.statuses.has(Status.HEAL_FREE) or target.char_id == heal_free_exception)
                   and not target.statuses.has(Status.UNTARGETABLE)
                   and (target.subprofession not in ("musha", "unyield", "reaper")
                        or target.char_id == heal_free_exception)
                   and not target.negative_hp_heal_free
                   and (target.char_id == heal_free_exception or self._operator_can_see(op, target))]
        targets = [target for target in targets
                   if (target.hp / target.max_hp <= hp_ratio if include_equal
                       else target.hp / target.max_hp < hp_ratio)
                   or (include_element_damage and hp_ratio == 1.0
                       and target.elements.current_damaged_element is not None)]
        return sorted(targets, key=lambda unit: (unit.hp / unit.max_hp, unit.creation_index))

    def _begin_operator_heal(self, op: _Operator, targets: list[_Operator], scale: float) -> None:
        execute_frame, execute_at = self._operator_attack_deadline(op)
        timing = self.attack_timing.get(op.char_id, {})
        if op.skill_active:
            timing = op.skill_attack_timing.get(op.skill_prefab_id, timing)
        flags = {"ignore_heal_free": True} if timing.get("ignore_heal_free") else {}
        if op.char_id == "char_1016_agoat2" and op.skill_active and op.skill_prefab_id == "skchr_agoat2_1":
            # S1's own Heal selector runs at emission; ordinary Heal locks at start.
            flags["reselect_heal_count"] = 2
        profile = AttackProfile(payload=flags) if flags else None
        op.pending_attack = _PendingOperatorAttack(
            execute_at, execute_frame, "HEAL", targets[0], targets, scale,
            op.atk_scale, op.skill_active and not op.skill_controls.can_switch, True,
            profile=profile,
        )
        self._mark_operator_attack_started(op, not op.attack_chain_active)
        self._land_zero_delay_operator_attack(op)

    def _resolve_operator_heal(self, op: _Operator, target: _Operator, amount: float,
                               *, secondary: bool = False, ignore_heal_free: bool = False,
                               ability_heal: bool = False) -> Any:
        self._sync_squad_effects()
        if ability_heal and op.skill_runtime is not None:
            op.skill_runtime.attach_heal_buffs(target)
        output_scale = 1.0
        if op.module_id in {"uniequip_002_shu", "uniequip_002_demkni"}:
            if target.hp / target.max_hp < float(op.trait_blackboard.get("hp_ratio", 0)):
                output_scale = float(op.trait_blackboard.get("heal_scale", 1))
        result = apply_damage(DamagePacket(amount=amount, damage_type=DamageType.HEAL,
                                          source=op, output_scale=output_scale,
                                          ignore_heal_free=ignore_heal_free), target,
                              on_output=(op.healing_runtime.output_modifier if op.healing_runtime else
                                         op.skill_runtime.output_modifier if op.skill_runtime else None))
        if ability_heal and op.skill_runtime is not None and not secondary:
            op.skill_runtime.heal_landed(target, amount, result)
        return result

    def _huang_s3_cut(self, op: _Operator) -> None:
        """Apply one of the eight once-per-second S3 cutting hits."""
        amount = self.operator_attack_power(op)
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
        if op.skill_runtime is not None:
            return op.skill_runtime.begin_attack(first_attack)
        if op.form_runtime is not None and op.form_runtime.mode == "THANATOS":
            target = self._current_operator_target(op)
            if target is None:
                return False
            candidates = [e for e in self.enemies if self._validate_operator_target(op, e)]
            priority = cmp_to_key(compare_operator_target_keys)
            candidates.sort(key=lambda e: priority(operator_target_key(e)))
            targets = [target] + [e for e in candidates if e is not target]
            maximum = int(op.skill_blackboard.get("attack@max_target", 1))
            frame, time = self._operator_attack_deadline(op)
            op.pending_attack = _PendingOperatorAttack(
                execute_at=time, execute_frame=frame, kind="FORM_MULTI", primary=target,
                targets=targets[:maximum], attack_scale=float(op.skill_blackboard.get("attack@atk_scale", 1)),
                source_atk_scale=op.atk_scale, sp_recovery_locked=True,
                will_hit=self._attack_hits(op, "ARTS"))
            self._mark_operator_attack_started(op, first_attack)
            return True
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
                sp_recovery_locked=op.skill_active and not op.skill_controls.can_switch,
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
            sp_recovery_locked=op.skill_active and not op.skill_controls.can_switch,
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
        self, op: _Operator, first_attack: bool, *, timing: dict | None = None
    ) -> None:
        op.attack_chain_active = True
        op.attack_first_cast = bool(first_attack)
        if timing is None:
            timing = self.attack_timing.get(op.char_id, {})
            if op.skill_active or op.skill_end_reason == "PERMANENT":
                timing = op.skill_attack_timing.get(op.skill_prefab_id, timing)
        op.graphic_animation = str(timing.get("animation_name") or timing.get("anim_key", "Attack"))
        op.graphic_started_at = self.time
        op.graphic_rate = 1.0 / self._operator_animation_duration_scale(op, timing)
        op.attack_cd_interval = self._effective_attack_interval(op)
        op.graphic_timing = (dict(timing) if "max_anim_duration_scale" in timing
                             and timing.get("timing_scale") != "FIXED" else None)
        attack = op.pending_attack
        if attack is not None and op.graphic_timing is not None and op.form_runtime is None:
            attack.minimum_execute_at = self.time + max(0.0, float(timing.get("min_hit_time", 0)))
            events = tuple(float(v) for v in timing.get("attack_hit_times", ()))
            attack.animation_events = (events if len(events) == len(attack.followup_frames)+1
                                       else (float(timing.get("attack_hit_time") or op.attack_hit_time),))
        target = op.pending_attack.primary if op.pending_attack is not None else None
        self._face_operator_graphic(op, target)
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
            sp_recovery_locked=op.skill_active and not op.skill_controls.can_switch,
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

    def _sync_operator_attack_clock(self, op: _Operator) -> None:
        # Static default OnAttackTimeChanged keeps the cooldown's progress.
        # Observe changes at this unit's update; exact attribute callback phase
        # and Hotfix dispatch remain candidates. Calibrated forms are separate.
        if op.form_runtime is not None or op.attack_cd_interval <= 0:
            return
        timing = self.attack_timing.get(op.char_id, {})
        if op.skill_active or op.skill_end_reason == "PERMANENT":
            timing = op.skill_attack_timing.get(op.skill_prefab_id, timing)
        if "max_anim_duration_scale" not in timing or timing.get("timing_scale") == "FIXED":
            return
        interval = self._effective_attack_interval(op)
        if interval <= 0 or math.isclose(interval, op.attack_cd_interval, rel_tol=1e-12):
            return
        progress_remaining = max(0.0, min(1.0, op.attack_cd / op.attack_cd_interval))
        op.attack_cd = interval * progress_remaining
        op.attack_cd_interval = interval
        self.mechanic_diagnostics["OPERATOR_ATTACK_TIME_CHANGE_PHASE_CANDIDATE"] = 1
        graphic = op.graphic_timing
        if graphic is None:
            return
        elapsed = max(0.0, self.time-op.graphic_started_at) * op.graphic_rate
        if elapsed >= float(graphic.get("attack_duration") or op.attack_duration):
            return
        rate = 1.0 / self._operator_animation_duration_scale(op, graphic)
        if math.isclose(rate, op.graphic_rate, rel_tol=1e-12):
            return
        op.graphic_started_at = self.time - elapsed/rate
        op.graphic_rate = rate
        runtime = op.skill_runtime
        if runtime is not None and getattr(runtime, "glady_emitted", False):
            runtime.glady_anim_end = op.graphic_started_at + float(
                graphic.get("attack_duration", op.attack_duration))/rate
        attack = op.pending_attack
        if attack is None or not attack.animation_events:
            return
        deadlines = [self._operator_attack_deadline(op, delay=max(0.0, (event-elapsed)/rate,
                                                                attack.minimum_execute_at-self.time))
                     for event in attack.animation_events]
        frame_delta = deadlines[0][0] - attack.execute_frame
        attack.execute_frame, attack.execute_at = deadlines[0]
        if len(deadlines) == len(attack.followup_frames)+1:
            attack.followup_frames = tuple(frame for frame, _ in deadlines[1:])
        else:
            # Additional child emissions retain their fixed spacing.
            attack.followup_frames = tuple(frame+frame_delta for frame in attack.followup_frames)

    def _operator_animation_duration_scale(self, op: _Operator, timing: dict | None = None) -> float:
        if timing is None:
            timing = self.attack_timing.get(op.char_id, {})
            if op.skill_active or op.skill_end_reason == "PERMANENT":
                timing = op.skill_attack_timing.get(op.skill_prefab_id, timing)
        if timing.get("timing_scale") == "FIXED":
            return 1.0
        duration = float(timing.get("attack_duration") or op.attack_duration)
        if duration <= 0.0:
            return 1.0
        scale = max(self._effective_attack_interval(op) / duration, 1e-9)
        # Native _maxAnimScale caps duration scaling, not playback speed.
        # Missing metadata retains the existing unbounded timing candidate.
        limit = float(timing.get("max_anim_duration_scale", -1.0))
        return min(scale, max(limit, 1e-9)) if limit >= 0 else scale

    def _operator_action_duration(self, op: _Operator) -> float:
        timing = self.attack_timing.get(op.char_id, {})
        if op.skill_active or op.skill_end_reason == "PERMANENT":
            timing = op.skill_attack_timing.get(op.skill_prefab_id, timing)
        if "max_anim_duration_scale" not in timing or timing.get("timing_scale") == "FIXED":
            return self._effective_attack_interval(op)
        self.mechanic_diagnostics["OPERATOR_ACTION_FINISH_PHASE_CANDIDATE"] = 1
        return float(timing.get("attack_duration") or op.attack_duration) * self._operator_animation_duration_scale(op, timing)

    def _operator_attack_hit_delay(self, op: _Operator, *, timing: dict | None = None) -> float:
        if timing is None:
            timing = {"max_anim_duration_scale": self.attack_timing.get(op.char_id, {}).get(
                "max_anim_duration_scale", -1.0)}
            if op.skill_active or op.skill_end_reason == "PERMANENT":
                timing = op.skill_attack_timing.get(op.skill_prefab_id, timing)
        duration = float((timing or {}).get("attack_duration") or op.attack_duration)
        hit_time = float((timing or {}).get("attack_hit_time") or op.attack_hit_time)
        if (timing or {}).get("timing_scale") == "FIXED":
            # Some abilities have a fixed predelay independent of attack speed.
            return max(0.0, hit_time, float(timing.get("min_hit_time", 0)))
        if duration <= 0.0 or hit_time <= 0.0:
            return 0.0
        return max(float(timing.get("min_hit_time", 0)),
                   hit_time * self._operator_animation_duration_scale(op, timing))

    def _operator_attack_deadline(self, op: _Operator, *, delay: float | None = None) -> tuple[int, float]:
        if delay is None:
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
        if attack.kind == "ALCHEMY":
            op.skill_runtime.launch_alchemy(attack)
            return
        if attack.kind == "ROSTER_STATE":
            op.skill_runtime.land_state_action(attack)
            if op.skill_runtime.continue_attack_sequence(attack):
                return
            self._sp_on_attack(op, sp_recovery_locked=attack.sp_recovery_locked)
            self._notify_operator_attack_complete(op, legacy_runtime=True)
            return
        if attack.kind == "HEAL":
            amount = self.operator_attack_power(op, attack.source_atk_scale) * attack.attack_scale
            healing_flags = attack.profile.payload if attack.profile else {}
            if healing_flags.get("reselect_heal_count"):
                targets = self._operator_heal_targets(op, include_element_damage=True)
                attack = replace(attack, targets=targets[:int(healing_flags["reselect_heal_count"])])
                self.mechanic_diagnostics["AGOAT_HEAL_SELECTOR_EVENT_PHASE_CANDIDATE"] = 1
            timing = (op.skill_attack_timing.get(op.skill_prefab_id, {}) if op.skill_active else
                      self.attack_timing.get(op.char_id, {}))
            key = timing.get("heal_projectile_key")
            for target in attack.targets:
                if not target.dead:
                    if self._projectile_speed(key) is not None:
                        payload = replace(attack, primary=target, targets=[target],
                            profile=AttackProfile(payload={
                                **healing_flags,
                                "keep_projectile_after_departure": bool(timing.get("keep_projectile_after_departure",
                                    self.attack_timing.get(op.char_id, {}).get("keep_projectile_after_departure", False))),
                                "projectile_delay_after_reached": float(timing.get("projectile_delay_after_reached", 0))}))
                        self._new_projectile(key=key, side="OPERATOR", source=op, target=target,
                                             operator_attack=payload, cached_atk=amount)
                        self.mechanic_diagnostics["HEAL_PROJECTILE_CURVE_AND_HIT_PHASE_CANDIDATE"] = 1
                    else:
                        self._resolve_operator_heal(op, target, amount,
                            ignore_heal_free=bool(healing_flags.get("ignore_heal_free")), ability_heal=True)
                elif op.healing_runtime is not None:
                    op.healing_runtime.refund_dead_target()
                elif op.skill_runtime is not None:
                    op.skill_runtime.refund_dead_heal_target(attack)
            self._sp_on_attack(op, sp_recovery_locked=attack.sp_recovery_locked)
            self.consume_skill_ammo(op)
            return
        self._record_targeting_diagnostic(
            op,
            hit_frame=int(self.frame_index),
        )
        if attack.kind == "ROSTER":
            attack = op.skill_runtime.refresh_attack_targets(attack)
            if op.skill_runtime.refund_dead_attack_target(attack):
                return
            primary_valid = any(self._enemy_is_attackable(target) for target in attack.targets)
            if primary_valid:
                op.skill_runtime.attack_emitted(attack)
            key = attack.profile.payload.get("projectile_key", op.projectile_key)
            if primary_valid and (self._projectile_speed(key) is not None
                                  or "hit_on_expiry" in attack.profile.payload):
                self._emit_operator_projectiles(op, attack)
            elif op.skill_runtime is not None:
                op.skill_runtime.land(attack)
            if primary_valid:
                op.skill_runtime.skill_spent = 0.0
            if op.skill_runtime.continue_attack_sequence(attack):
                return
            self._sp_on_attack(op, sp_recovery_locked=attack.sp_recovery_locked)
            self.consume_skill_ammo(op)
            self._notify_operator_attack_complete(op, legacy_runtime=True)
            return
        primary_valid = self._enemy_is_attackable(attack.primary)
        if attack.will_hit and primary_valid:
            if attack.kind == "HAMMER":
                self._land_hammer_attack(op, attack)
            elif self._projectile_speed(op.projectile_key) is not None:
                self._emit_operator_projectiles(op, attack)
            elif attack.kind in ("MULTI", "FORM_MULTI"):
                first = True
                form_amount = (self.operator_attack_power(op, attack.source_atk_scale) * attack.attack_scale
                               if attack.kind == "FORM_MULTI" else None)
                for enemy in attack.targets:
                    if not self._enemy_is_attackable(enemy):
                        continue
                    scale = attack.attack_scale if first or attack.kind == "FORM_MULTI" else 1.0
                    if form_amount is not None:
                        # The action keeps one weakness packet kind while
                        # reflection may trigger a form change during its hits.
                        self._apply_operator_damage(op, enemy, form_amount, attack_type=AttackType.NORMAL,
                                                    apply_way=ApplyWay.MELEE, damage_type="WEAKNESS")
                    else:
                        self._deal_damage_to_enemy(op, enemy, scale, source_atk_scale=attack.source_atk_scale)
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
        self._notify_operator_attack_complete(op)

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
        mount_point: str | None = None,
        origin_override: tuple[float, float, float] | None = None,
        immediate: bool = False,
        hit_on_expiry: float | None = None,
        mount_rule: dict[str, Any] | None = None,
        behavior_payload: dict[str, Any] | None = None,
        enemy_attack_context: dict[str, Any] | None = None,
    ) -> None:
        speed = 0.0 if immediate or hit_on_expiry is not None else self._projectile_speed(key)
        if speed is None:
            return
        if isinstance(source, _Operator):
            source_row, source_col = map(float, source.tile)
        else:
            source_row, source_col = source.position()
        source_z = 0.0
        muzzle_position = None
        if origin_override is not None:
            source_row, source_col, source_z = origin_override
            muzzle_position = origin_override
        elif isinstance(source, _Operator):
            origin, mount_point = self._operator_projectile_origin(source, key, target, mount_point, mount_rule)
            source_row, source_col, source_z = origin
            if mount_point is not None:
                muzzle_position = origin
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
        lifecycle = (dict(self.attack_timing.get(source.char_id, {}).get(
            "projectile_lifecycle", {}).get(key, {})) if isinstance(source, _Operator)
            else self.wdex.projectile_lifecycle(key))
        if hit_on_expiry is not None:
            lifecycle.update(lifetime=hit_on_expiry, no_movement=True,
                             stop_when_source_invalid=False, always_hit_in_end=True)
            self.mechanic_diagnostics["OPERATOR_FIXED_LIFETIME_HIT_PHASE_CANDIDATE"] = 1
        birth_delay = 0.0
        if lifecycle.get("random_birth_delay", 0) > 0:
            birth_delay = self.random_uniform(0, lifecycle["random_birth_delay"],
                mechanism="operator_projectile_birth_delay", source=source, target=target,
                details={"projectile": key})
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
                position_row=target_row if immediate else source_row,
                position_col=target_col if immediate else source_col,
                position_z=target_z if immediate else source_z,
                motion_profile=profile,
                launched_at=self.time,
                operator_attack=operator_attack,
                will_hit=will_hit,
                cached_atk=cached_atk,
                parabolic="mortar" in key.lower(),
                initial_distance=initial_distance,
                origin=(source_row, source_col, source_z),
                mount_point=mount_point if muzzle_position is not None else None,
                lifecycle=lifecycle,
                last_destination=(target_row, target_col, target_z),
                born_at=self.time + birth_delay,
                behavior_payload=behavior_payload,
                enemy_attack_context=enemy_attack_context,
                impact_at=self.time if immediate else None,
                fixed_destination=((target_row, target_col, target_z)
                                   if lifecycle.get("fixed_endpoint") else None),
            )
        )

    def _operator_projectile_origin(
        self, op: _Operator, key: str, target=None, mount_point: str | None = None,
        mount_rule: dict[str, Any] | None = None,
    ) -> tuple[tuple[float, float, float], str | None]:
        """Select an emission point shared by tracking and special flight runtimes."""
        graphic = self.operator_graphics.get(op.char_id)
        rule = graphic.projectiles.get(key, {}) if graphic is not None else {}
        if mount_rule is not None:
            rule = {**rule, **mount_rule}
        if graphic is not None and key not in graphic.projectiles and mount_point is None:
            self.mechanic_diagnostics[f"OPERATOR_PROJECTILE_BINDING_MISSING:{op.char_id}:{key}"] += 1
        if rule.get("faceToTarget"):
            self._face_operator_graphic(op, target)
        if mount_point is None:
            group = rule.get("group", [])
            if group and graphic is not None:
                # Missing geometry must not silently consume an extra random draw.
                if all(self.operator_muzzle_position_v3(op, point) is not None for point in group):
                    if rule.get("loopType") == "RANDOM":
                        mount_point = self.choose_random_target(group, source=op,
                            mechanism="operator_projectile_mount", details={"projectile": key})
                    else:
                        index = op.mount_group_indices.get(key, 0)
                        mount_point = group[index % len(group)]
                        op.mount_group_indices[key] = index + 1
                    self.mechanic_diagnostics["OPERATOR_MOUNT_SELECTION_ORDER_CANDIDATE"] = 1
                else:
                    self.mechanic_diagnostics[f"OPERATOR_MOUNT_GROUP_MISSING:{op.char_id}:{key}"] += 1
            else:
                mount_point = rule.get("mount")
        if mount_point is not None:
            position = self.operator_muzzle_position_v3(op, mount_point)
            if position is not None:
                op.last_projectile_mount = mount_point
                self.used_assumptions.add("OPERATOR_PROJECTILE_MOUNT")
                self.mechanic_diagnostics["OPERATOR_GRAPHIC_POSE_CANDIDATE"] = 1
                return position, mount_point
            self.mechanic_diagnostics[f"OPERATOR_MOUNT_MISSING:{op.char_id}:{mount_point}"] += 1
        self.used_assumptions.add("PROJECTILE_SOURCE_CENTER")
        return (float(op.tile[0]), float(op.tile[1]), 0.0), None

    def _face_operator_graphic(self, op: _Operator, target=None) -> None:
        if isinstance(target, (_Enemy, _Operator)):
            row, col = target.tile if isinstance(target, _Operator) else target.position()
            dr, dc = row - op.tile[0], col - op.tile[1]
        else:
            from .operator_skills import FACING
            dr, dc = FACING[op.facing % 4]
        if abs(dc) > 1e-9:
            op.graphic_sign = 1.0 if dc > 0 else -1.0
        if abs(dr) > 1e-9:
            op.graphic_face = "back" if dr > 0 else "front"

    def operator_muzzle_position_v3(self, op: _Operator, mount_point: str | None = None):
        """Read a pose without choosing a point or advancing the random stream."""
        profile = self.operator_graphics.get(op.char_id)
        if profile is None:
            return None
        mount = mount_point or op.last_projectile_mount
        offset = profile.offset_at(op.graphic_face, mount, op.graphic_animation,
                                   max(0.0, self.time - op.graphic_started_at) * op.graphic_rate)
        if offset is None:
            return None
        return op.tile[0] + offset[0], op.tile[1] + offset[1] * op.graphic_sign, offset[2]

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
        if op.char_id == SOUL_WARD and op.skill_runtime is not None:
            op.skill_runtime.emit_soul_ward_projectile(attack)
            return
        if op.char_id == "char_1035_wisdel" and op.skill_runtime is not None:
            op.skill_runtime.emit_wisdel_projectiles(attack)
            return
        if op.char_id == "char_4039_horn" and op.skill_runtime is not None:
            op.skill_runtime.emit_horn_projectiles(attack)
            return
        if (op.char_id == "char_4213_skybx" and attack.profile is not None
                and attack.profile.skill and op.skill_runtime is not None):
            op.skill_runtime.emit_skybx_skill_projectile(attack)
            return
        if (attack.profile is not None and attack.profile.skill
                and attack.profile.payload.get("sid") == "skchr_ulpia_1"):
            op.skill_runtime.emit_ulpia_projectiles(attack)
            return
        key = (attack.profile.payload.get("projectile_key", op.projectile_key)
               if attack.profile is not None else op.projectile_key)
        if not key:
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
                profile=attack.profile,
            )
            self._new_projectile(
                key=key,
                side="OPERATOR",
                source=op,
                target=enemy,
                operator_attack=payload,
                will_hit=attack.will_hit,
                hit_on_expiry=(attack.profile.payload.get("hit_on_expiry")
                               if attack.profile is not None else None),
            )

    def _land_hammer_attack(
        self, op: _Operator, attack: _PendingOperatorAttack
    ) -> None:
        target_position = attack.primary.position()
        radius = float(op.trait_blackboard.get("attack@ability_range_radius", 1.0))
        nearby = [
            enemy
            for enemy in self.enemies
            if self._enemy_is_attackable(enemy)
            and not (op.char_id == "char_1051_headb2" and self._is_airborne(enemy))
            and self._enemy_intersects_circle(
                enemy, target_position, radius=radius
            )
        ]
        trait = op.trait_blackboard
        module_scale = (float(trait["atk_scale_e"])
                        if op.char_id == "char_1051_headb2" and "atk_scale_e" in trait
                        and len(nearby) >= float(trait.get("cnt", math.inf)) else 1.0)
        attack_scale = attack.attack_scale * module_scale
        self._deal_damage_to_enemy(
            op,
            attack.primary,
            attack_scale,
            source_atk_scale=attack.source_atk_scale,
        )
        talent_scale = float(op.talent_blackboards.get("1", {}).get("damage_scale", 1.0)) if op.char_id == "char_1051_headb2" else 1.0
        splash_scale = float(trait.get("attack@atk_scale_2", 0.5)) * attack_scale
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
        talent = op.talent_blackboards.get("1", {})
        if "attack@splash_atk_scale" not in talent:
            return
        echo_scale = float(talent["attack@splash_atk_scale"])
        # The main attack cast evaluates the X-module condition once.  Every
        # highland splash created by that attack inherits the same decision.
        amount = self.operator_attack_power(op, source_atk_scale) * echo_scale * module_scale
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
                        sluggish_duration=float(talent.get("attack@sluggish", 0.0)),
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
        amount = float(op.skill_blackboard.get("sp_per_highland", 0.0))
        if op.skill_active or self._skill_starting(op):
            self.used_assumptions.add("ACTIVE_SKILL_SP_LOCK")
            if not self.assumptions.allow_active_skill_sp_gain:
                return
            self._store_sp_gain(op, amount)
            return
        self._gain_sp(op, amount)

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
                if not enemy.dead and effect.sluggish_duration > 0:
                    sluggish_frames = max(
                        0, int(self.assumptions.sluggish_extra_frames)
                    )
                    self.used_assumptions.add(
                        "SLUGGISH_DURATION_EXTRA_FRAME"
                    )
                    enemy.statuses.apply(
                        Status.SLUGGISH,
                        effect.sluggish_duration + sluggish_frames / CLIENT_LOGIC_RATE,
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
        atk = self.operator_attack_power(op, source_atk_scale) * scale
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
        damage_type: str | None = None,
        damage_source: Any = ...,
        modifier_key: str = "",
        ability_owner: Any = ...,
        notify_calculate: bool = True,
    ):
        kind = damage_type or getattr(op, "damage_type", "PHYSICAL")
        source = op if damage_source is ... else damage_source
        life_loss = str(getattr(kind, "value", kind)).upper() == DamageType.LIFE_LOSS.value
        if not life_loss:
            self._reflect_shell_damage(enemy, source)
        result = self._apply_damage(
            DamagePacket(
                amount=amount,
                damage_type="ARTS" if kind == "WEAKNESS" else kind,
                weakness=kind == "WEAKNESS",
                attack_type=attack_type,
                apply_way=apply_way,
                output_scale=(1.0 if life_loss else self._outgoing_damage_scale(op)) * output_scale,
                taken_scale=taken_scale,
                source=source,
                modifier_key=modifier_key,
                notify_calculate=notify_calculate,
                ability_owner=(op if attack_type == AttackType.NORMAL else None)
                    if ability_owner is ... else ability_owner,
                critical_chance=0.0 if life_loss else float(getattr(op, "critical_chance", 0.0)),
                critical_multiplier=float(
                    getattr(op, "critical_multiplier", 1.0)
                ),
                target_resistance_override=self._effective_resistance(enemy),
                target_defense_override=self._effective_enemy_defense(enemy),
            ),
            enemy,
        )
        if result.killed:
            self.enemies_killed += 1
            if hasattr(op, "kills"):
                op.kills += 1
            if getattr(op, "skill_runtime", None) is not None:
                op.skill_runtime.killed_enemy(enemy)
        if hasattr(op, "damage_dealt"):
            op.damage_dealt += result.final_amount
        return result

    def _reflect_shell_damage(self, enemy: _Enemy, source: _Operator | None) -> None:
        if (not enemy.shell_active or enemy.shell_reflect <= 0 or source is None
                or source.dead or source.statuses.has(Status.INVINCIBLE)
                or source.statuses.has(Status.UNTARGETABLE)):
            return
        # The source template uses fixed magical damage with no damage source,
        # and does not require the incoming modifier to succeed.
        self._apply_damage(DamagePacket(enemy.shell_reflect, DamageType.ARTS,
                                  attack_type=AttackType.NORMAL, source=None,
                                  target_resistance_override=self._effective_resistance(source)),
                     source)

    def _enemy_normal_damage_type(self, enemy: _Enemy) -> DamageType | str:
        """Use the same scoped type for hit checks and damage resolution."""
        if self.dublin.enabled:
            return self.dublin.damage_type(enemy)
        if self.mansfield.enabled:
            return self.mansfield.damage_type(enemy)
        return self.vecbreak.normal_damage_type(enemy)

    def _deal_damage_to_operator(
        self,
        enemy: _Enemy,
        op: _Operator,
        *,
        attack_override: float | None = None,
        attack_context: dict[str, Any] | None = None,
    ) -> None:
        defense = self.operator_defense(op)
        if self.vecbreak.enabled:
            attack_override = self.vecbreak.normal_attack(enemy, op, attack_override)
            if op.dead: return
        packet = DamagePacket(
                amount=(
                    self._effective_enemy_atk(enemy)
                    if attack_override is None
                    else attack_override
                ),
                damage_type=(attack_context["damage_type"] if attack_context is not None else
                             self._enemy_normal_damage_type(enemy)),
                attack_type=AttackType.NORMAL,
                apply_way=(
                    attack_context["apply_way"] if attack_context is not None else
                    ApplyWay.RANGED if enemy.range_radius > 0 else ApplyWay.MELEE
                ),
                output_scale=self._outgoing_damage_scale(enemy) * self.wdex.attack_scale(enemy, attack_context),
                source=enemy,
                critical_chance=float(
                    getattr(enemy, "critical_chance", 0.0)
                ),
                critical_multiplier=float(
                    getattr(enemy, "critical_multiplier", 1.0)
                ),
                target_defense_override=defense,
                target_resistance_override=self._effective_resistance(op),
            )
        packet = self.mansfield.modify_packet(packet, op, context=attack_context)
        result = self._apply_damage(packet, op)
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
        if self.mansfield.fire_attack(enemy, target, will_hit=will_hit):
            return
        if self.wdex.fire_attack(enemy, target, will_hit=will_hit):
            return
        self._fire_single_enemy_attack(enemy, target, will_hit=will_hit)

    def _fire_single_enemy_attack(self, enemy, target, *, will_hit, attack_context=None):
        """Shared launch path, with a per-cast context for scoped enemy skills."""
        spec = self.enemy_projectiles.get(enemy.key) or {}
        key = enemy.projectile_key if attack_context is None else attack_context["projectile"]
        destination = int(spec.get("destination", 0) or 0)
        if key and destination != 0:
            self.mechanic_diagnostics[
                f"PROJECTILE_DESTINATION:{destination}"
            ] += 1
            if will_hit:
                self._enemy_attack_lands(enemy, target, attack_context=attack_context)
            return
        speed = self._projectile_speed(key)
        if speed is None:
            if will_hit:
                self._enemy_attack_lands(enemy, target, attack_context=attack_context)
            return
        self._new_projectile(
            key=key or "",
            side="ENEMY",
            source=enemy,
            target=target,
            will_hit=will_hit,
            enemy_attack_context=attack_context,
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
            attack = projectile.operator_attack
            wisdel = (attack is not None and attack.profile is not None
                      and "wisdel_component" in attack.profile.payload)
            rules = projectile.lifecycle
            if projectile.born_at is not None and self.time + 1e-9 < projectile.born_at:
                active.append(projectile)
                continue
            valid = self._projectile_target_valid(projectile)
            # Reviewed movements retain their cached endpoint when its mount
            # becomes invalid. Unreviewed attacks keep the historical path.
            if not rules and not (wisdel and projectile.impact_at is not None) and not valid:
                continue
            if valid and projectile.fixed_destination is None:
                projectile.last_destination = self._projectile_target_position(
                    projectile.target, projectile.motion_profile)
            lifetime = rules.get("lifetime")
            start = projectile.born_at if projectile.born_at is not None else projectile.launched_at
            source_invalid = rules.get("stop_when_source_invalid") and projectile.source.dead
            if source_invalid or (lifetime is not None and self.time - start + 1e-9 >= lifetime):
                self._expire_operator_projectile(projectile, wisdel=wisdel)
                continue
            if rules.get("no_movement"):
                active.append(projectile)
                continue
            if projectile.impact_at is not None:
                if self.time + 1e-9 >= projectile.impact_at:
                    self._projectile_lands(projectile)
                else:
                    active.append(projectile)
                continue
            profile = projectile.motion_profile
            target_row, target_col, target_z = (projectile.fixed_destination or
                (projectile.last_destination if rules else None) or
                self._projectile_target_position(projectile.target, profile))
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
                attack = projectile.operator_attack
                if wisdel:
                    projectile.source.skill_runtime.wisdel_projectile_reached(projectile)
                delay = (float(attack.profile.payload.get("projectile_delay_after_reached",
                              rules.get("delay_after_reached", 0)))
                         if attack is not None and attack.profile is not None else 0)
                if delay > 0:
                    projectile.impact_at = self.time + delay
                    active.append(projectile)
                else:
                    self._projectile_lands(projectile)
            else:
                active.append(projectile)
        self.projectiles = active

    def _expire_operator_projectile(self, projectile: _Projectile, *, wisdel: bool) -> None:
        """Run the reviewed stop path, without restarting the arrival delay."""
        rules = projectile.lifecycle
        reached = projectile.impact_at is not None
        if wisdel and not reached and rules.get("always_reach_in_end"):
            # Projectile.OnStop delivers the reached notification before
            # its movement's forced DealReached resets the position.
            projectile.source.skill_runtime.wisdel_projectile_reached(projectile)
        if not reached and rules.get("force_reached_when_timeup") and rules.get("reset_to_target"):
            endpoint = projectile.fixed_destination or projectile.last_destination
            if endpoint is not None:
                projectile.position_row, projectile.position_col, projectile.position_z = endpoint
        attack = projectile.operator_attack
        if (attack is not None and attack.profile is not None
                and "horn_component" in attack.profile.payload):
            projectile.source.skill_runtime.land_horn_projectile(projectile, stopped=True)
            self.mechanic_diagnostics["OPERATOR_PROJECTILE_STOP_ORDER_CANDIDATE"] = 1
            return
        if wisdel or reached or rules.get("force_reached_when_timeup") or rules.get("always_hit_in_end"):
            self._projectile_lands(projectile)
        self.mechanic_diagnostics["OPERATOR_PROJECTILE_STOP_ORDER_CANDIDATE"] = 1

    def _projectile_lands(self, projectile: _Projectile) -> None:
        attack = projectile.operator_attack
        if attack is not None and attack.profile is not None:
            if attack.profile.skill and attack.profile.payload.get("sid") == "skchr_skybx_1":
                if projectile.will_hit:
                    projectile.source.skill_runtime.land_skybx_splash(attack,
                        (projectile.position_row, projectile.position_col), float(projectile.cached_atk or 0))
                return
            if attack.kind == "HALO2_LINK":
                projectile.source.skill_runtime.land_halo2_link(attack, projectile.target,
                    source_key=projectile.projectile_id)
                return
            if attack.profile.skill and attack.profile.payload.get("sid") == "skchr_halo2_2":
                # Reached actions precede the main hit and do not depend on
                # its trace target or damage judgement remaining valid.
                projectile.source.skill_runtime.emit_halo2_links(attack,
                    (projectile.position_row, projectile.position_col, projectile.position_z))
        if (attack is not None and attack.profile is not None
                and "horn_component" in attack.profile.payload):
            projectile.source.skill_runtime.land_horn_projectile(projectile)
            return
        if not projectile.will_hit:
            return
        if (attack is not None and attack.profile is not None
                and "wisdel_component" in attack.profile.payload):
            projectile.source.skill_runtime.wisdel_projectile_stopped(projectile)
            return
        if not self._projectile_target_valid(projectile):
            return
        if projectile.behavior_payload is not None:
            self.land_behavior_projectile(projectile)
            return
        if projectile.side == "ENEMY":
            self._enemy_attack_lands(
                projectile.source,
                projectile.target,
                attack_override=projectile.cached_atk,
                attack_context=projectile.enemy_attack_context,
            )
            return
        attack = projectile.operator_attack
        if attack is None:
            return
        if attack.kind == "HEAL":
            self._resolve_operator_heal(projectile.source, projectile.target,
                projectile.cached_atk if projectile.cached_atk is not None else
                self.operator_attack_power(projectile.source, attack.source_atk_scale) * attack.attack_scale,
                ignore_heal_free=bool(attack.profile and attack.profile.payload.get("ignore_heal_free")),
                ability_heal=True)
            return
        if attack.kind == "LOGOS_TALENT":
            projectile.source.skill_runtime.land_talent_projectile(projectile)
            return
        if attack.kind == "LOGOS_TRANSFER":
            self._apply_operator_damage(projectile.source, projectile.target,
                float(projectile.cached_atk or 0), damage_type="ARTS",
                attack_type=AttackType.NORMAL, apply_way=ApplyWay.NONE)
            return
        if attack.profile is not None and projectile.source.skill_runtime is not None:
            pull_source = ({"projectile_id": projectile.projectile_id}
                           if projectile.source.char_id in {"char_474_glady", "char_4145_ulpia"} else {})
            projectile.source.skill_runtime.land(attack, target=projectile.target, ranged=True,
                **pull_source)
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
        attack_context: dict[str, Any] | None = None,
    ) -> None:
        """Resolve one enemy hit, including target-centered splash attacks."""
        self._deal_damage_to_operator(
            enemy, target, attack_override=attack_override, attack_context=attack_context
        )
        self.mansfield.after_hit(enemy, target, attack_context)
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

    @staticmethod
    def _random_subject(unit: Any) -> Any:
        if unit is None or isinstance(unit, (str, int, bool)):
            return unit
        if isinstance(unit, tuple) and len(unit) == 2 and all(isinstance(v, (int, float)) for v in unit):
            return {"side": "cell", "tile": list(unit)}
        if hasattr(unit, "char_id"):
            return {"side": "operator", "char": unit.char_id,
                    "deployment": getattr(unit, "deployment_index", None)}
        if hasattr(unit, "enemy_index"):
            return {"side": "enemy", "key": getattr(unit, "key", None), "id": unit.enemy_index}
        raise ValueError("random decision subjects require a stable entity ID")

    def _random_context(self, mechanism: str, source: Any = None, target: Any = None,
                        details: dict | None = None) -> dict:
        return {"frame": getattr(self, "frame_index", 0), "mechanism": mechanism,
                "source": self._random_subject(source), "target": self._random_subject(target),
                "details": details or {}}

    def _ensure_rng(self) -> None:
        if not hasattr(self, "rng"):
            self.seed = 0
            self.rng = DeterministicRng(0)

    def _roll_probability(self, probability: float, *, mechanism: str = "probability",
                          source: Any = None, target: Any = None, details: dict | None = None) -> bool:
        self._ensure_rng()
        if isinstance(self.rng, RandomDecisionRng):
            return self.rng.roll(probability, context=self._random_context(mechanism, source, target, details))
        return self.rng.roll(probability)

    def _apply_damage(self, packet: DamagePacket, target: Any):
        dublin = getattr(self, "dublin", None)
        if (dublin and dublin.animation_damage_immune(target)
                and packet.normalized_damage_type() is not DamageType.HEAL):
            return DamageResult(0.0, 0.0, 0.0, cancelled=True)
        if dublin:
            packet = dublin.modify_packet(packet, target)
        if getattr(self, "cw", None):
            packet = self.cw.modify_packet(packet, target)
        if getattr(self, "vecbreak", None):
            packet = self.vecbreak.modify_packet(packet, target)
        labeled = None
        if isinstance(getattr(self, "rng", None), RandomDecisionRng):
            def labeled(probability: float, mechanism: str) -> bool:
                return self._roll_probability(probability, mechanism=mechanism, source=packet.source,
                                              target=target, details={"damage_type": packet.normalized_damage_type().value,
                                              "attack_type": str(getattr(packet.attack_type, "value", packet.attack_type)),
                                              "apply_way": str(getattr(packet.apply_way, "value", packet.apply_way))})
        runtime = getattr(target, "skill_runtime", None)
        source_runtime = getattr(packet.source, "skill_runtime", None)
        source_healing = getattr(packet.source, "healing_runtime", None)
        if source_runtime is None:
            summoner = getattr(packet.source, "summoner", None)
            if summoner is not None and summoner.char_id == "char_450_necras":
                source_runtime = summoner.skill_runtime
        if source_runtime:
            packet = source_runtime.modify_packet(packet, target)
        packet = self._dispatch_component_calculate(packet, target, "scale")
        if runtime:
            packet = runtime.modify_taken_packet(packet)
        kind = packet.normalized_damage_type()
        if (kind is DamageType.ARTS and packet.target_resistance_override is None
                and isinstance(target, (_Enemy, _Operator))
                and any(m.value for m in target.buffs.modifiers("MAGIC_RESISTANCE", registered_only=True)
                        if m.formula == "ADDITION")):
            packet = replace(packet, target_resistance_override=self._effective_resistance(target))
        if isinstance(target, _Enemy):
            if kind is not DamageType.LIFE_LOSS:
                packet = replace(packet, taken_scale=packet.taken_scale * max(
                    [1.0] + [r.wanted_damage_scale(packet.source, target) for r in live_runtimes(self)]))
            if kind is DamageType.ARTS:
                packet = replace(packet, amount=packet.amount + float(getattr(target, "roster_arts_addition", 0)))
            if kind in {DamageType.PHYSICAL, DamageType.ARTS, DamageType.TRUE}:
                packet = replace(packet, taken_scale=packet.taken_scale * max(
                    [1.0] + [r.weakness_scale(target) for r in live_runtimes(self)]))
            if kind is DamageType.ELEMENTAL:
                packet = replace(packet, taken_scale=packet.taken_scale * max(
                    [1.0] + [r.element_vulnerability_scale(target) for r in live_runtimes(self)]))
        elif isinstance(target, _Operator):
            if kind in {DamageType.PHYSICAL, DamageType.ARTS, DamageType.TRUE, DamageType.ELEMENTAL}:
                resistance = max([
                    packet.damage_resistance,
                    runtime.own_damage_resistance(kind) if runtime else 0.0,
                    target.healing_runtime.own_damage_resistance(kind) if target.healing_runtime else 0.0,
                ] + ([r.ally_damage_resistance(target, packet.source) for r in live_runtimes(self)]
                     if kind in {DamageType.PHYSICAL, DamageType.ARTS} and packet.source in self.enemies else []))
                packet = replace(packet, damage_resistance=resistance)
        def calculate(current_packet, current_target):
            self._dispatch_component_calculate(current_packet, current_target, "notify")
            if source_runtime is not None:
                source_runtime.calculate_damage_modifier(current_packet, current_target)
        def output(current_packet, current_target):
            self._dispatch_component_output(current_packet, current_target)
            if source_healing is not None:
                source_healing.output_modifier(current_packet, current_target)
            elif source_runtime is not None:
                source_runtime.output_modifier(current_packet, current_target)
        def taking(current_packet, current_target):
            if getattr(self, "cw", None):
                self.cw.taking_damage(current_packet, current_target)
            if runtime is not None:
                runtime.taking_damage_modifier(current_packet, current_target)
        result = apply_damage(packet, target, roll=self._roll_probability, roll_labeled=labeled,
            on_calculate=calculate if packet.notify_calculate else None,
            on_output=output,
            on_taking_damage=taking if runtime is not None or (getattr(self, "cw", None) and self.cw.enabled) else None,
            before_apply=source_runtime.before_apply_damage
                if source_runtime is not None else None,
            on_block_damage=runtime.blocked_damage if runtime is not None else None)
        if source_runtime is not None and isinstance(target, _Enemy):
            source_runtime.damage_landed(packet, target, result)
        if (source_runtime is not None and source_runtime.op.char_id == "char_4133_logos"
                and source_runtime.op.module_id == "uniequip_002_logos" and kind is DamageType.ARTS
                and isinstance(target, _Enemy) and not target.dead and result.final_amount > 0):
            self.apply_element_damage(target, ElementType.DARK, result.final_amount *
                float(source_runtime.talent(10).get("ep_damage_ratio", 0)), source=packet.source)
        if runtime:
            runtime.taken_damage(packet, result)
        if result.killed and isinstance(target, _Enemy):
            for receiver in live_runtimes(self):
                receiver.enemy_died(target, killer=packet.source)
        if getattr(self, "vecbreak", None):
            self.vecbreak.damage_landed(packet, target, result)
        if result.killed:
            self._finish_element_templates(target)
        return result

    def random_uniform(self, lower: float, upper: float, *, mechanism: str,
                       source: Any = None, target: Any = None, details: dict | None = None) -> float:
        """A named half-open sample for mechanics such as launch directions."""
        self._ensure_rng()
        lower, upper = uniform_bounds(lower, upper)
        if isinstance(self.rng, RandomDecisionRng):
            return self.rng.uniform(lower, upper, context=self._random_context(mechanism, source, target, details))
        return uniform_value(self.rng, lower, upper)

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
        return self._roll_probability(probability, mechanism="hit_rate", source=source,
                                      details={"damage_type": normalized.value})

    def choose_random_target(self, candidates: list[Any], *, mechanism: str = "random_target",
                             source: Any = None, details: dict | None = None) -> Any:
        """Choose once from an already-filtered candidate list."""
        if not candidates:
            return None
        self._ensure_rng()
        if isinstance(self.rng, RandomDecisionRng):
            context = self._random_context(mechanism, source, details={**(details or {}),
                "candidates": [self._random_subject(item) for item in candidates]})
            return candidates[self.rng.choose_index(len(candidates), context=context)]
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

    def _enemy_has_invisibility_immunity(self, target: _Enemy) -> bool:
        return any(not op.dead and op.char_id == "char_172_svrash"
            and "2" in op.talent_blackboards and self._enemy_intersects_range(target, op.range_cells)
            for op in self.operators) or any(r.reveals(target) for r in effect_runtimes(self))

    def _enemy_camouflaged(self, target: _Enemy) -> bool:
        return target.statuses.has(Status.CAMOUFLAGE) and not self._enemy_has_invisibility_immunity(target)

    def _operator_can_see(self, source: _Operator, target: Any) -> bool:
        cell = target.tile() if callable(target.tile) else target.tile
        return not blind_tile(self, source, cell)

    def _enemy_targetable(self, target: _Enemy) -> bool:
        if target.statuses.targetable:
            return True
        if any(target.statuses.has(s) for s in (Status.INVINCIBLE, Status.UNTARGETABLE, Status.SLEEP)):
            return False
        return target.statuses.has(Status.INVISIBLE) and self._enemy_has_invisibility_immunity(target)

    def _validate_operator_target(
        self, op: _Operator, target: _Enemy | None
    ) -> bool:
        if target is None or not self._enemy_is_attackable(target):
            return False
        if not self._enemy_targetable(target):
            return False
        if not self.cw.valid_target(op, target):
            return False
        if not self.dublin.valid_target(op, target):
            return False
        if op.char_id == "char_1051_headb2" and self._is_airborne(target):
            return False
        if op.char_id == "char_4039_horn" and self._is_airborne(target):
            return False
        if op.char_id == "token_10043_necras_skeltn" and self._is_airborne(target):
            return False
        if op.skill_prefab_id == "skchr_skybx_1" and not op.skill_active and not self._is_airborne(target):
            return False
        blocked = any(enemy is target for enemy in op.blocked)
        blocker = target.blocked_by
        if (op.char_id == "char_450_necras" and blocker is not None and not blocker.dead
                and getattr(blocker, "summoner", None) is op):
            return True
        if (op.prioritize_blocked or op.subprofession == "hammer") and blocked:
            # Preserve the existing blocked-first rule: once an enemy is
            # blocked, the melee selector does not reapply ranged geometry.
            return True
        if self._enemy_camouflaged(target):
            return False
        if not self._operator_can_see(op, target):
            return False
        if op.skill_runtime is not None and op.skill_runtime.is_wanted(target):
            return True
        if op.char_id == "char_4039_horn" and self._enemy_intersects_range(target, {op.tile}):
            return True
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
        if op.char_id == "char_4039_horn":
            # The grouped selector tries mortar range before its own-tile
            # melee fallback. Priority within each selector remains unchanged.
            ranged = [e for e in candidates if self._enemy_intersects_range(e, op.range_cells)]
            candidates = ranged or candidates
        if op.skill_prefab_id == "skchr_lemuen_1":
            return min(candidates, key=lambda enemy: (self._effective_enemy_defense(enemy), enemy.creation_index))
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

        if op.damage_type == "HEAL":
            return False
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

    @staticmethod
    def _enemy_within_radius(enemy: _Enemy, center: tuple[float, float], radius: float) -> bool:
        """Midpoint range; distinct from a projectile's collider overlap."""
        return math.dist(enemy.position(), center) <= radius + 1e-9

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
            if (op.air_blocking if op.liftoff is None else op.liftoff) and not self._is_airborne(enemy):
                continue
            dist = math.hypot(op.tile[0] - r, op.tile[1] - c)
            if dist <= enemy.range_radius + OPERATOR_HITBOX_RADIUS + 1e-9:
                candidates.append(op)
        if not candidates:
            return None
        if self.mansfield.enabled:
            candidates = self.mansfield.sort_targets(enemy, candidates)
            return candidates[0] if candidates else None
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
        if target.dead or not target.statuses.targetable or not self.mansfield.valid_target(enemy, target):
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
        # Registered operator Buffs share the existing frame-end timer phase.
        for enemy in (*getattr(self, "enemies", ()), *getattr(self, "operators", ())):
            if enemy.dead:
                continue
            container = getattr(enemy, "buffs", None)
            if container is None:
                continue
            container.tick(self.dt, self.frame_index)
            container.commit()

    def _dispatch_buff_event(self, instance: BuffInstance, event: str) -> bool:
        """Run normalized Buff actions through the existing interpreter."""
        if event in {"ON_BUFF_START", "ON_BUFF_ENABLE"}:
            for flag in instance.definition.status_flags:
                instance.owner.statuses.grant(flag, instance)
        elif event in {"ON_BUFF_FINISH", "ON_BUFF_DISABLE"}:
            for flag in instance.definition.status_flags:
                instance.owner.statuses.revoke(flag, instance)
        if (event in {"ON_BUFF_START", "ON_BUFF_ENABLE", "ON_BUFF_FINISH", "ON_BUFF_DISABLE"}
                and set(instance.definition.status_flags) & {"LEVITATE", "GROUND_BOUND"}):
            owner = instance.owner
            if isinstance(owner, _Enemy):
                if "GROUND_BOUND" in instance.definition.status_flags:
                    self.mechanic_diagnostics["GROUND_BOUND_NAVIGATION_CANDIDATE"] = 1
                if owner.statuses.levitate_active():
                    self._end_unbalance(owner)
                if self._is_airborne(owner) and owner.blocked_by is not None:
                    self._release_enemy_block(owner)
                if event in {"ON_BUFF_FINISH", "ON_BUFF_DISABLE"}:
                    self.mechanic_diagnostics["MOTION_CONTROL_EXIT_PHASE_CANDIDATE"] = 1
        listener = instance.component_listener
        if listener is not None:
            if event != "ON_BUFF_FINISH" and (instance.finished or (
                    event not in {"ON_BUFF_START", "ON_BUFF_ENABLE", "ON_BUFF_DISABLE"}
                    and (not instance.enabled or instance.owner.buffs._is_pending_finish(instance)))):
                return False
            if event == "ON_BUFF_START":
                listener.started = True
            valid = True
            if (event in listener.events and listener.started
                    and not (listener.ignore_dead and instance.owner.dead)):
                valid = self.component_catalog.validate(self, listener, initializing=event == 'ON_BUFF_START')
                if valid:
                    context = self.execute_behavior_event(listener.template, event,
                        source=instance.source, buff_owner=instance.owner,
                        blackboard=instance.blackboard, buff_instance=instance,
                        register_buffs=True)
                    valid = context is not None
                    if valid:
                        # Only the live Buff owns event state. Immutable source
                        # boards and other deployments keep their snapshots.
                        instance.blackboard.clear()
                        instance.blackboard.update(context.blackboard)
                        listener.board = instance.blackboard
                if not valid:
                    listener.record.update(state="rejected", reason="event_binding")
                    instance.owner.buffs.finish(instance)
            if event == "ON_BUFF_FINISH":
                listener.finished = True
                if listener.record["state"] == "attached":
                    listener.record["state"] = ("expired" if instance.remaining_lifetime is not None
                        and instance.remaining_lifetime <= 1e-9 else "detached")
                if listener in instance.owner.component_listeners:
                    instance.owner.component_listeners.remove(listener)
            return valid
        actions = instance.definition.actions_for(event)
        if not actions:
            return True
        if instance.definition.registered_behavior:
            if not self.behavior.validate_actions(list(actions), instance.definition.template_key + ":" + event):
                return False
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
            template_key=(instance.definition.template_key if instance.definition.registered_behavior
                          else instance.definition.key),
            event=event,
            current_buff_key=instance.definition.key,
            buff_container=getattr(owner, "buffs", None),
            buff_instance=instance,
        )
        if instance.definition.registered_behavior and not self._validate_behavior_bindings(context, list(actions)):
            return False
        return self.behavior.execute_actions(list(actions), context)

    @staticmethod
    def _registered_attribute_addition(actor: Any, attribute: str) -> float:
        """Read live reviewed modifiers; legacy adapter effects stay separate."""
        return sum(item.value for item in actor.buffs.modifiers(attribute, registered_only=True)
                   if item.formula == "ADDITION")

    def behavior_skill_progress(self, context: BehaviorContext, node: dict[str, Any]) -> bool:
        actor = context.entity(node.get("_targetType"))
        if actor is None:
            return False
        if not isinstance(actor, _Operator) or actor.skill is None:
            return self.behavior._unsupported(node_name(node) + ":skill_progress_owner_binding")
        if self.component_catalog.skill_progress_hooker(actor.skill_prefab_id) is None:
            return self.behavior._unsupported(node_name(node) + ":skill_progress_hooker_binding")
        if node_name(node) == "UnlockCharacterSkillProgress":
            self.unlock_skill_progress(actor)
            return True
        if not actor.skill_active or actor.dead:
            return False
        if actor.skill_duration_type == "AMMO" or actor.active_until <= 0:
            return self.behavior._unsupported(node_name(node) + ":skill_progress_kind_binding")
        # Only finite active time progress has an established engine binding.
        # Native cached progress is not a freeze of the actor's Buff collection.
        return self.lock_skill_progress(actor)

    def lock_skill_progress(self, op: _Operator) -> bool:
        if not isinstance(op, _Operator) or op.dead or not op.skill_active:
            return False
        if not op.skill_progress_locked:
            op.skill_progress_remaining = max(0.0, op.active_until - self.time) if op.active_until > 0 else None
            op.skill_progress_locked = True
        if op.skill_progress_remaining is not None:
            op.active_until = self.time + op.skill_progress_remaining
        return True

    def unlock_skill_progress(self, op: _Operator) -> None:
        if op.skill_progress_locked and op.skill_progress_remaining is not None and op.skill_active:
            op.active_until = self.time + op.skill_progress_remaining
        op.skill_progress_locked = False
        op.skill_progress_remaining = None

    def behavior_negative_hp_state(self, context: BehaviorContext, node: dict) -> bool:
        actor = context.entity(node.get("_targetType"))
        if actor is None:
            return False
        if not isinstance(actor, (_Operator, _Enemy)):
            return self.behavior._unsupported(node_name(node) + ":hp_controller_binding")
        if node_name(node) == "CheckIfAtMinusHpState":
            return actor.max_minus_hp_ratio > 0 and actor.minus_hp > 0
        ratio = float(node["_value"])
        if not math.isfinite(ratio * actor.max_hp):
            return self.behavior._unsupported("SetMaxMinusHpRatio:hp_capacity_overflow")
        actor.max_minus_hp_ratio = ratio
        actor.negative_hp_heal_free = ratio > 0 and actor.minus_hp > 0
        if actor.minus_hp > 0:
            self.mechanic_diagnostics["NEGATIVE_HP_CAPACITY_CHANGE_NOTIFICATION_CANDIDATE"] = 1
        return True

    def behavior_abnormal_condition(self, context: BehaviorContext, node: dict) -> bool:
        actor = context.entity(node.get("_targetType"))
        if actor is None:
            return False
        if not isinstance(actor, (_Operator, _Enemy)):
            return self.behavior._unsupported(node_name(node) + ":status_controller_binding")
        status = Status(ABNORMAL_READ_FLAGS[node["_abnormalFlag"]])
        active = (status in actor.statuses.immunities if node_name(node) == "CheckAbnormalImmune"
                  else actor.statuses.has(status))
        return not active if node.get("_isUnset", False) else active

    def behavior_entity_value(self, actor: Any, node: str, attribute: str | None, *, modifiers=None) -> float:
        """Read the current model pool without syncing auras during preflight."""
        if not isinstance(actor, (_Operator, _Enemy)):
            raise ValueError("attribute_entity_binding")
        if node == "AssignMaxSpToBlackboard":
            if not isinstance(actor, _Operator) or actor.skill is None:
                raise ValueError("sp_controller_binding")
            return actor.sp_max
        if node == "AssignHpRatioToBB":
            try:
                valid = all(type(value) in (int, float) and math.isfinite(float(value))
                            for value in (actor.hp, actor.max_hp))
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError("hp_ratio_value")
            if actor.max_hp <= 0:
                raise ValueError("hp_ratio_denominator")
            if (isinstance(actor, _Operator) and actor.char_id == "char_1044_hsgma2"
                    and actor.hp <= 0 and actor.negative_hp_heal_free):
                # The old signed storage omits the regular one-HP bar.
                return min(1.0, 1.0 / actor.max_hp)
            # Ratio predicates read the regular bar, excluding negative/extra HP.
            return max(0.0, min(1.0, actor.hp / actor.max_hp))
        if attribute == "MAX_HP": return actor.max_hp
        if attribute in {"ONE_MINUS_STATUS_RESISTANCE", "SLOW_DOWN", "EP_BREAK_RECOVER_SPEED"}:
            return actor.buffs.clock_attribute(attribute, modifiers=modifiers)
        if attribute == "MAGIC_RESISTANCE": return self._effective_resistance(actor, modifiers=modifiers)
        if isinstance(actor, _Operator):
            if attribute == "ATK": return self.operator_attack_power(actor, synchronize=False, modifiers=modifiers)
            if attribute == "DEF": return self.operator_defense(actor, synchronize=False, modifiers=modifiers)
        else:
            if attribute == "ATK": return self._effective_enemy_atk(actor, modifiers=modifiers)
            if attribute == "DEF": return self._effective_enemy_defense(actor, modifiers=modifiers)
            if attribute == "MOVE_SPEED": return self._effective_enemy_move_speed(actor, modifiers=modifiers)
        raise ValueError("attribute_controller_binding")

    def _validated_buff_owner_values(self, definition: BuffDefinition, owner: Any,
                                     owner_kind: type | None = None) -> bool:
        """Validate the live pool again before birth without running actions."""
        owner_kind = type(owner) if owner is not None else owner_kind
        hp_modifiers = tuple(item for item in definition.modifiers if item.attribute == "MAX_HP")
        if (set(definition.status_flags) & {"LEVITATE", "GROUND_BOUND", "FORCE_LEVITATE"}
                and owner_kind is not None and owner_kind is not _Enemy):
            self.behavior._unsupported("RegisterBuff:motion_control_owner_binding")
            return False
        projected = None
        retained_projection = None
        if definition.override_policy is not None and owner_kind is not None and owner_kind not in {_Operator, _Enemy}:
            self.behavior._unsupported("RegisterBuff:override_owner_binding")
            return False
        if owner is not None and (issue := owner.buffs.override_issue(definition)):
            self.behavior._unsupported("RegisterBuff:" + issue)
            return False
        if owner is not None and definition.override_policy is not None and not definition.disable_override:
            if definition.override_policy == "DEFAULT":
                group = [item for item in (*owner.buffs.active, *owner.buffs.pending_add)
                    if not item.finished and not owner.buffs._is_pending_finish(item)
                    and item.definition.override_policy == "DEFAULT" and not item.definition.disable_override
                    and item.definition.override_key == definition.override_key]
                strongest = max(group, key=lambda item: item.definition.override_priority) if group else None
                winner = (strongest.definition if strongest is not None
                    and strongest.definition.override_priority >= definition.override_priority else definition)
                unrelated = tuple(replace(modifier, value=modifier.value * item.stack_count)
                    for item in owner.buffs.active if item.enabled and not item.finished
                    and not owner.buffs._is_pending_finish(item) and item.definition.registered_behavior
                    and not any(old is item for old in group)
                    for modifier in item.definition.modifiers if modifier.attribute == "MAX_HP")
                projected = unrelated + tuple(modifier for modifier in winner.modifiers if modifier.attribute == "MAX_HP")
                if winner is not definition and hp_modifiers:
                    # A retained loser may become enabled after the winner expires.
                    retained_projection = unrelated + hp_modifiers
                hp_modifiers = ()
            existing = next((item for item in (*owner.buffs.active, *owner.buffs.pending_add)
                if item.enabled and not item.finished and not owner.buffs._is_pending_finish(item)
                and not item.definition.disable_override and item.definition.override_key == definition.override_key), None)
            if existing is not None:
                hp_modifiers = (tuple(item for item in existing.definition.modifiers if item.attribute == "MAX_HP")
                    if definition.override_policy == "STACK" and existing.stack_count < existing.definition.max_stack_count else ())
        if hp_modifiers and owner_kind is not None and owner_kind not in {_Operator, _Enemy}:
            self.behavior._unsupported("RegisterBuff:max_hp_owner_binding")
            return False
        if owner is not None and (hp_modifiers or projected is not None):
            for pool in ((projected, retained_projection) if retained_projection is not None else (projected,)):
                if isinstance(owner, _Operator):
                    maximum = self._operator_maximum_hp(owner, extra=hp_modifiers, modifiers=pool)
                else:
                    current = owner.buffs.modifiers("MAX_HP", registered_only=True) if pool is None else pool
                    modifiers = (*current, *hp_modifiers)
                    base = owner.max_hp if owner.registered_max_hp_base is None else owner.registered_max_hp_base
                    maximum = calculate_attribute(base, modifiers,
                        final_addition=-self._enemy_attribute_stolen(owner, "MAX_HP"))
                if not math.isfinite(maximum):
                    self.behavior._unsupported("RegisterBuff:max_hp_overflow")
                    return False
        return True

    def _validated_buff_events(self, spec: dict[str, Any], seen: frozenset[str] = frozenset(),
                               blackboard: dict[str, Any] | None = None,
                               allowed_events=frozenset({"ON_BUFF_START", "ON_BUFF_FINISH", "ON_BUFF_TRIGGER",
                                                         "ON_BUFF_ENABLE", "ON_BUFF_DISABLE"}),
                               owner: Any = None, source: Any = None,
                               owner_kind: type | None = None, source_kind: type | None = None,
                               immediate_children: bool = False, derived: bool = False) -> dict | None:
        """Preflight referenced child trees before registering any state."""
        spec = self._resolve_buff_spec(spec)
        if spec is None:
            return None
        issue = registered_buff_spec_issue(spec)
        if issue:
            self.behavior._unsupported("RegisterBuff:" + issue)
            return None
        definition = self._behavior_buff_definition(spec, blackboard or {})
        if definition is None:
            return None
        if derived and definition.override_policy not in {None, "DEFAULT"} and not definition.disable_override:
            self.behavior._unsupported("RegisterBuff:override_parent_binding")
            return None
        board = self._behavior_buff_blackboard(spec, blackboard or {})
        owner_kind = type(owner) if owner is not None else owner_kind
        source_kind = type(source) if source is not None else source_kind
        if not self._validated_buff_owner_values(definition, owner, owner_kind):
            return None
        if owner_kind is not None and owner_kind is not _Operator and any(
                item.get("attributeType") in {"HP_RECOVERY_PER_SEC", "HP_RECOVERY_PER_SEC_BY_MAX_HP_RATIO",
                                             "SP_RECOVERY_PER_SEC"}
                for item in (spec.get("attributes") or {}).get("attributeModifiers") or []):
            self.behavior._unsupported("RegisterBuff:recovery_controller_binding")
            return None
        if owner_kind is not None and owner_kind is not _Enemy and any(
                item.get("attributeType") == "MOVE_SPEED"
                for item in (spec.get("attributes") or {}).get("attributeModifiers") or []):
            self.behavior._unsupported("RegisterBuff:move_speed_owner_binding")
            return None
        if owner_kind is not None and owner_kind is not _Enemy and any(
                item.get("attributeType") == "EP_BREAK_RECOVER_SPEED"
                for item in (spec.get("attributes") or {}).get("attributeModifiers") or []):
            self.behavior._unsupported("RegisterBuff:operator_ep_effect_clock_binding")
            return None
        roles = {"SOURCE": source, "BUFF_SOURCE": source, "TARGET": owner, "BUFF_OWNER": owner}
        kinds = {"SOURCE": source_kind, "BUFF_SOURCE": source_kind,
                 "TARGET": owner_kind, "BUFF_OWNER": owner_kind}
        key = str(spec.get("templateKey") or "")
        if key in seen:
            self.behavior._unsupported(f"RegisterBuff:{key}:recursive_template")
            return None
        events = {}
        if key and key != "empty":
            template = self.behavior.resolve_template(key)
            if template is None:
                self.behavior.validate_event(key, "ON_BUFF_START")
                return None
            events = template.get("eventToActions", {})
            if ((spec.get("attributes") or {}).get("attributeModifiers")
                    and contains_entity_value_read(events)):
                self.behavior._unsupported(f"RegisterBuff:{key}:own_attribute_snapshot_binding")
                return None
            if any(event not in allowed_events
                   for event in events):
                self.behavior._unsupported(f"RegisterBuff:{key}:event")
                return None
            valid = True
            children = []
            def capture(value, current_board):
                nonlocal valid
                if node_name(value) in {"LockCharacterSkillProgress", "UnlockCharacterSkillProgress"} and owner_kind is not None:
                    if kinds.get(value.get("_targetType")) is not _Operator:
                        self.behavior._unsupported(f"RegisterBuff:{key}:skill_progress_owner_binding")
                        valid = False
                    actor = roles.get(value.get("_targetType"))
                    if isinstance(actor, _Operator) and self.component_catalog.skill_progress_hooker(actor.skill_prefab_id) is None:
                        self.behavior._unsupported(f"RegisterBuff:{key}:skill_progress_hooker_binding")
                        valid = False
                if node_name(value) in {"SetMaxMinusHpRatio", "CheckIfAtMinusHpState", "CheckAbnormalFlag", "CheckAbnormalImmune",
                                      "FinishAllBuffsWithCertainAbnormalFlag"} and owner_kind is not None:
                    role = value.get("_targetType")
                    actor = roles.get(role)
                    if kinds.get(role) not in {_Enemy, _Operator} or (
                            actor is not None and node_name(value) == "SetMaxMinusHpRatio"
                            and not math.isfinite(actor.max_hp * value["_value"])):
                        self.behavior._unsupported(f"RegisterBuff:{key}:hp_controller_binding")
                        valid = False
                if node_name(value) == "ModifySp" and owner_kind is not None:
                    role = value.get("_targetType", "TARGET")
                    if kinds.get(role) is not _Operator:
                        self.behavior._unsupported(f"RegisterBuff:{key}:sp_controller_binding")
                        valid = False
                if node_name(value) == "CheckContainsBuff" and value.get("_checkBuffSource") and owner_kind is not None:
                    if kinds.get(value.get("_buffSourceType", "TARGET")) not in {_Enemy, _Operator}:
                        self.behavior._unsupported(f"RegisterBuff:{key}:buff_source_binding")
                        valid = False
                if node_name(value) in {"EmitProjectileUseAbilitySelector", "AOEDamage"}:
                    self.behavior._unsupported(f"RegisterBuff:{key}:nested_action_binding:{node_name(value)}")
                    valid = False
                if node_name(value) in {"CreateBuff", "CreateBuffInRange"}:
                    if (((value.get("_buff") or {}).get("attributes") or {}).get("attributeModifiers")
                            and not value.get("_isDerivedBuff")):
                        self.behavior._unsupported(f"RegisterBuff:{key}:nested_attribute_binding")
                        valid = False
                    children.append((value, dict(current_board)))
                return None
            started_boards = [board]
            ordered_events = (["ON_BUFF_START"] if "ON_BUFF_START" in events else []) + [
                event for event in events if event != "ON_BUFF_START"]
            for event in ordered_events:
                valid = self.behavior.validate_event(key, event) and valid
                context = BehaviorContext(source=source, target=owner, buff_owner=owner, buff_source=source,
                                          battle=self, register_buffs=immediate_children, event=event)
                completed = []
                for event_board in ([board] if event == "ON_BUFF_START" else started_boards):
                    if issue := numeric_issue(events[event], event_board, node_validator=capture,
                            blackboard_reader=lambda node, state: self._read_preflight_blackboard(node, context, state),
                            actor_preview=self._behavior_attribute_preview(context, root_definition=definition, event=event),
                            final_states=completed):
                        self.behavior._unsupported(f"RegisterBuff:{key}:{issue}")
                        valid = False
                if event == "ON_BUFF_START" and completed:
                    started_boards = completed
            for child, child_board in children:
                if node_name(child) == "CreateBuffInRange":
                    emitter_kind = kinds.get(child.get("_sourceType"))
                    center_kind = kinds.get(child.get("_targetType"))
                    if owner_kind is not None:
                        context = BehaviorContext(source=source, target=owner, buff_owner=owner, buff_source=source)
                        valid = self._validate_behavior_range(context, child, f"RegisterBuff:{key}:range",
                            source_kind=emitter_kind, center_kind=center_kind) and valid
                    receiver_kind = (emitter_kind if (child.get("_targetOptions") or {}).get("targetSide") == "ALLY"
                                     else {_Enemy: _Operator, _Operator: _Enemy}.get(emitter_kind))
                    for child_spec in child.get("_buffs") or []:
                        valid = (self._validated_buff_events(child_spec, seen | {key}, child_board,
                            owner_kind=receiver_kind, source=roles.get(child.get("_sourceType")),
                            source_kind=emitter_kind, derived=bool(child.get("_isDerivedBuff"))) is not None) and valid
                    continue
                child_owner = roles.get(child.get("_buffOwner", "TARGET"))
                child_source = roles.get(child.get("_specialBuffSource")) if child.get("_useSpecialBuffSource") else source
                child_owner_kind = kinds.get(child.get("_buffOwner", "TARGET"))
                child_source_kind = kinds.get(child.get("_specialBuffSource")) if child.get("_useSpecialBuffSource") else source_kind
                if owner_kind is not None and (child_owner_kind not in {_Enemy, _Operator}
                        or child_source_kind not in {_Enemy, _Operator}):
                    self.behavior._unsupported(f"RegisterBuff:{key}:child_entity_binding")
                    valid = False
                valid = (self._validated_buff_events(child.get("_buff") or {}, seen | {key}, child_board,
                    owner=child_owner, source=child_source, owner_kind=child_owner_kind,
                    source_kind=child_source_kind, derived=bool(child.get("_isDerivedBuff"))) is not None) and valid
            if not valid:
                return None
        return events

    def _behavior_buff_blackboard(self, spec: dict[str, Any], blackboard: dict[str, Any]) -> dict[str, Any]:
        if spec.get("stripBlackboardParamsWithBuffKey"):
            prefix = str(spec.get("buffKey", "")) + "."
            board = {key[len(prefix):]: value for key, value in blackboard.items() if key.startswith(prefix)}
        else:
            board = dict(blackboard)
        board.update(self._blackboard(spec))
        return board

    def _resolve_buff_spec(self, spec: dict[str, Any]) -> dict | None:
        try:
            return self.buff_database.resolve_spec(spec)
        except ValueError as exc:
            self.behavior._unsupported("RegisterBuff:" + str(exc))
            return None

    def _behavior_buff_definition(self, spec: dict[str, Any], blackboard: dict[str, Any],
                                  events: dict | None = None) -> BuffDefinition | None:
        spec = self._resolve_buff_spec(spec)
        if spec is None:
            return None
        board = self._behavior_buff_blackboard(spec, blackboard)
        if events is None:
            events = (self.behavior.resolve_template(str(spec.get("templateKey") or "")) or {}).get("eventToActions", {})
        try:
            definition = BuffDefinition.from_create_spec({**spec, "eventToActions": events or {},
                "registered_behavior": True, "unique_by_key": False}, blackboard=board)
            return replace(definition, element_break=True) if definition.key in self.element_break_buff_keys else definition
        except (ValueError, TypeError, OverflowError):
            self.behavior._unsupported("RegisterBuff:invalid_value/lifetime")
            return None

    def register_behavior_buff(
        self, spec: dict[str, Any], *, owner: Any, source: Any,
        blackboard: dict[str, Any] | None = None, parent: BuffInstance | None = None,
        commit: bool = True,
        refresh: bool = False,
        finish_with_parent: bool = False,
    ) -> BuffInstance | None:
        """Register reviewed flags/attributes after whole child-tree preflight."""
        if owner is None or getattr(owner, "dead", False):
            return None
        spec = self._resolve_buff_spec(spec)
        if spec is None:
            return None
        container = owner.buffs
        if finish_with_parent and (not BuffContainer.live_parent(parent)
                or parent.container is not getattr(parent.owner, "buffs", None)
                or getattr(parent.owner, "dead", False)
                or not parent.definition.registered_behavior
                or not isinstance(source, (_Enemy, _Operator))):
            self.behavior._unsupported("RegisterBuff:derived_parent_binding")
            return None
        if ((spec.get("overrideType", "DEFAULT") != "DEFAULT" or spec.get("priority", -1) != -1 or spec.get("priorityBBKeys"))
                and (parent is not None and spec.get("overrideType", "DEFAULT") != "DEFAULT"
                     and not spec.get("disableOverride", False) or refresh)):
            self.behavior._unsupported("RegisterBuff:override_parent/refresh_binding")
            return None
        events = self._validated_buff_events(spec, blackboard=blackboard, owner=owner, source=source,
                                             derived=parent is not None)
        if events is None:
            return None
        board = self._behavior_buff_blackboard(spec, blackboard or {})
        definition = self._behavior_buff_definition(spec, blackboard or {}, events)
        if definition is None:
            return None
        container.owner = owner
        container.event_handler = self._dispatch_buff_event
        container.change_handler = self._registered_attributes_changed
        instance = container.add(definition, source=source, parent=parent,
            blackboard=board, attached_step=self.frame_index, finish_with_parent=finish_with_parent)
        if refresh and definition.modifiers and definition.lifetime_type == "LIMITED":
            # Only explicit adapters request this same-source timer refresh.
            # This is not the general native DEFAULT/stack override algorithm.
            instance.definition = definition
            instance.blackboard = board
            instance.remaining_lifetime = definition.lifetime
            instance.attached_step = self.frame_index
        if commit:
            container.commit()
        return instance

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

    def _effective_enemy_atk(self, enemy: _Enemy, *, modifiers=None) -> float:
        if modifiers is not None:
            return max(calculate_attribute(enemy.atk, modifiers,
                final_addition=-self._enemy_attribute_stolen(enemy, "ATK")), 0.0)
        container = getattr(enemy, "buffs", None)
        if container is None or not hasattr(container, "attribute_value"):
            return max(0.0, enemy.atk - self._enemy_attribute_stolen(enemy, "ATK"))
        has_herald_attribute = container.count(
            "enemy_9D0_talent_strength[attribute]"
        ) > 0
        source_limit = None
        if has_herald_attribute:
            self.used_assumptions.add("HERALD_AURAS_STACK")
            if not self.assumptions.herald_auras_stack:
                source_limit = 1
        return container.attribute_value(
            "ATK", enemy.atk, source_limit=source_limit,
            final_addition=-self._enemy_attribute_stolen(enemy, "ATK")
        )

    def _effective_enemy_defense(self, enemy: _Enemy, *, modifiers=None) -> float:
        if modifiers is not None:
            return max(calculate_attribute(enemy.defense, modifiers), 0.0)
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
                        if enemy.attack_waits_for_cooldown:
                            enemy.attack_rearm_frame = None
                enemy.attack_cd_interval = interval

        if frame_based:
            enemy.attack_cd_frames = max(0, enemy.attack_cd_frames - 1)
            enemy.attack_cd = enemy.attack_cd_frames / CLIENT_LOGIC_RATE
            return
        timer_step = (
            CLIENT_FIXED_TIMER_STEP
            if not self.assumptions.enemy_attack_animation_rearm and math.isclose(
                self.dt, 1 / CLIENT_LOGIC_RATE, rel_tol=0.0, abs_tol=1e-12
            )
            else self.dt
        )
        enemy.attack_cd -= timer_step
        if self.assumptions.enemy_attack_animation_rearm:
            self._update_enemy_attack_rearm(enemy)

    def _update_enemy_attack_rearm(self, enemy: _Enemy) -> None:
        end_frame = enemy.attack_animation_end_frame
        if end_frame is None or self.frame_index < end_frame:
            return
        if enemy.attack_rearm_frame is not None:
            return
        remaining = max(0.0, enemy.attack_cd)
        if enemy.attack_waits_for_cooldown is None:
            # Choose the continuation path once, when the animation completes.
            enemy.attack_waits_for_cooldown = remaining >= self.dt * 0.5
        enemy.attack_rearm_frame = self.frame_index
        if enemy.attack_waits_for_cooldown:
            remaining_frames = remaining / self.dt
            nearest = round(remaining_frames)
            # Remove arithmetic noise only; a fractional cooldown still ceils.
            if math.isclose(remaining_frames, nearest, rel_tol=0.0, abs_tol=1e-12):
                remaining_frames = float(nearest)
            enemy.attack_rearm_frame += math.ceil(remaining_frames) + 1

    def _enemy_attack_ready(self, enemy: _Enemy) -> bool:
        if self._enemy_uses_frame_attack_cooldown(enemy):
            return enemy.attack_cd_frames <= 0
        if (self.assumptions.enemy_attack_animation_rearm
                and enemy.attack_animation_end_frame is not None):
            return (enemy.attack_rearm_frame is not None
                    and self.frame_index >= enemy.attack_rearm_frame)
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
        enemy.attack_animation_end_frame = None
        enemy.attack_rearm_frame = None
        enemy.attack_waits_for_cooldown = None
        self._start_enemy_attack_cooldown(enemy)
        if enemy.apply_way != ApplyWay.MELEE.value:
            enemy.target_search_after_attack = True
        for runtime in live_runtimes(self):
            runtime.before_enemy_normal_attack(enemy)
        if enemy.dead or not enemy.statuses.can_attack:
            return
        if enemy.palsy_stacks > 0:
            enemy.palsy_stacks -= 1
            palsy_applied = enemy.statuses.apply(
                Status.PALSY_TREMOR, 0.5, resistible=False
            )
            if palsy_applied is not None or not (self.vecbreak.enabled and enemy.key == "enemy_8018_etouch_2"):
                enemy.pause_until = self.time + 0.5
                return
        enemy.attack_target = target
        self.mansfield.before_attack(enemy, target)
        self.cw.before_attack(enemy, target)
        if self.vecbreak.enabled:
            self.vecbreak.enemy_state[enemy.enemy_index]["attack_speed_snapshot"] = self._effective_enemy_move_speed(enemy)
        if enemy.graphic_profile is not None:
            enemy.attack_started_at = self.time
            self._request_enemy_facing(enemy, target.tile[1] - enemy.position_col)
        enemy.attack_will_hit = self._attack_hits(enemy, self._enemy_normal_damage_type(enemy))
        animation_duration = enemy.attack_duration
        hit_delay = enemy.attack_hit_time
        if (enemy.apply_way == ApplyWay.MELEE.value
                and self.assumptions.melee_attack_animation_interval_scale
                and animation_duration > enemy.attack_cd_interval > 0.0):
            # A compressed ordinary melee interval speeds up its attack track.
            # Capture the speed at startup; mid-animation changes remain open.
            self.used_assumptions.add("MELEE_ATTACK_ANIMATION_INTERVAL_SCALE")
            scale = enemy.attack_cd_interval / animation_duration
            hit_delay *= scale
            animation_duration *= scale
        if hit_after_windup:
            # attack_hit_time is the pre-hit windup; resolve on the next
            # fixed frame after that animation segment completes.
            hit_delay = max(0.0, hit_delay) + self.dt
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
        enemy.pause_until = self.time + animation_duration
        if (self.assumptions.enemy_attack_animation_rearm
                and not self._enemy_uses_frame_attack_cooldown(enemy)):
            enemy.attack_animation_end_frame = (
                self.frame_index + _fixed_delay_frames(animation_duration, self.dt)
            )
            self.used_assumptions.add("ENEMY_ATTACK_ANIMATION_REARM")
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
        if isinstance(self, Battle):
            self._sync_squad_effects()
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
            + (sum(r.enemy_speed_delta(unit) for r in live_runtimes(self)) if isinstance(self, Battle) and isinstance(unit, _Enemy) else 0)
            + (self.mansfield.speed_delta(unit) if isinstance(self, Battle) and isinstance(unit, _Enemy) else 0)
            + (self.dublin.attack_speed(unit) if isinstance(self, Battle) and isinstance(unit, _Enemy) else 0)
        )
        container = getattr(unit, "buffs", None)
        if container is not None and hasattr(container, "attribute_value"):
            attack_speed = max(
                1.0,
                container.attribute_value("ATTACK_SPEED", attack_speed),
            )
        if isinstance(unit, _Operator):
            own_speed = unit.base_attack_speed
            form = unit.form_runtime
            total = max(1.0, own_speed + unit.squad_bonus.attack_speed
                        + unit.statuses.attack_speed_delta + element_bonus
                        + (self.vecbreak.char_addition(unit.char_id, "attack_speed") if isinstance(self, Battle) and getattr(self, "vecbreak", None) and self.vecbreak.enabled else 0)
                        + (self.cw.attack_speed(unit) if isinstance(self, Battle) and getattr(self, "cw", None) else 0)
                        + (form.attack_speed if form else 0)
                        + (unit.skill_runtime.attack_speed() if unit.skill_runtime else 0)
                        + sum(m.value for m in unit.buffs.modifiers("ATTACK_SPEED", registered_only=True)
                              if m.formula == "ADDITION")
                        + (sum(r.aura_speed(unit) for r in live_runtimes(self)) if isinstance(self, Battle) else 0))
            return (unit.attack_interval * own_speed
                    + (form.attack_time_add * 100 if form else 0)
                    + (unit.skill_runtime.interval_delta * 100 if unit.skill_runtime else 0)) / total
        return unit.attack_interval * 100.0 / attack_speed

    def _effective_enemy_move_speed(self, enemy: _Enemy, *, modifiers=None) -> float:
        speed = enemy.move_speed * self.move_multiplier
        container = getattr(enemy, "buffs", None)
        if modifiers is not None:
            speed = max(calculate_attribute(speed, modifiers), 0.0)
        elif container is not None and hasattr(container, "attribute_value"):
            speed = container.attribute_value("MOVE_SPEED", speed)
        if enemy.statuses.has(Status.SLUGGISH):
            speed *= 0.2
        for runtime in effect_runtimes(self):
            if not runtime.op.dead or runtime.op.char_id == "char_4087_ines":
                speed *= runtime.move_factor(enemy)
        speed = self.cw.speed(enemy, speed) if getattr(self, "cw", None) else speed
        return self.vecbreak.move_speed(enemy, speed) if getattr(self, "vecbreak", None) else speed

    def _move_enemy(self, enemy: _Enemy) -> None:
        initial_wait_expiry = (
            enemy.route_idx == 1
            and len(enemy.route) >= 2
            and enemy.route[0].kind.upper() == "START"
            and enemy.route[1].kind.upper()
            in {"WAIT_FOR_SECONDS", "WAIT_CURRENT_WAVE_TIME", "WAIT_CURRENT_FRAGMENT_TIME"}
            and enemy.wait_active_at_frame_start
            and enemy.wait_remaining <= 1e-12
            and not enemy.wait_hidden_at_frame_start
        )
        if initial_wait_expiry:
            next_point = (enemy.route[enemy.route_idx + 1]
                          if enemy.route_idx + 1 < len(enemy.route) else None)
            if (self.assumptions.initial_wait_zero_move_handoff
                    and next_point is not None and next_point.reachable
                    and next_point.kind.upper() == "MOVE"
                    and math.hypot(next_point.row - enemy.position_row,
                                   next_point.col - enemy.position_col) <= 1e-12):
                self.used_assumptions.add("INITIAL_WAIT_ZERO_MOVE_HANDOFF")
                self._reach_route_point(enemy, next_point)
                enemy.wait_phase_consumed = True
                return
            self.used_assumptions.add("INITIAL_ROUTE_WAIT_SAME_FRAME_MOVE")
            # Initial visible waits may resume movement on the expiry frame;
            # later route waits keep their own phase.
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
                "WAIT_CURRENT_FRAGMENT_TIME",
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
            gate = next((op for op in self.operators if not op.dead
                and op.char_id == "token_10058_sbell2_icetgt"
                and op.statuses.can_block and self._free_block_capacity(op) >= (enemy.block_count if self.vecbreak.enabled else 1)
                and not enemy.unblockable and not self._is_airborne(enemy)
                and not enemy.statuses.has(Status.FEAR)
                and math.dist(enemy.position(), op.tile) <= op.block_radius), None)
            if gate is None:
                enemy.leaked = True
            else:
                self._start_blocking(gate, enemy)

    def _advance_hidden_route_events(self, enemy: _Enemy) -> None:
        """Resolve hidden-route control points without visible movement."""
        while enemy.route_idx < len(enemy.route) - 1:
            target = enemy.route[enemy.route_idx + 1]
            kind = target.kind.upper()
            if kind in {
                "WAIT_FOR_SECONDS",
                "WAIT_CURRENT_WAVE_TIME",
                "WAIT_CURRENT_FRAGMENT_TIME",
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
        if not self._uses_ground_navigation(enemy):
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
            # keep the strict countdown to preserve their handoff boundary.
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
                if not self._uses_ground_navigation(enemy)
                else GROUND_STEERING_FACTOR
            )
            max_force = 10.0 if self._uses_ground_navigation(enemy) else 100.0
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
            not self._uses_ground_navigation(enemy)
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
        if kind in {"DISAPPEAR", "APPEAR_AT_POS", "TELEPORT"}:
            enemy.position_jump_index += 1
        if kind == "WAIT_CURRENT_FRAGMENT_TIME":
            elapsed = self.time - getattr(enemy, "fragment_start_time", enemy.wave_start_time)
            enemy.wait_remaining = max(0.0, point.wait_time - elapsed)
        elif kind == "WAIT_CURRENT_WAVE_TIME":
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
        free_slots = self._free_block_capacity(op)
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
            and not enemy.statuses.has(Status.FEAR)
            and self._is_airborne(enemy) == op.air_blocking
            and not (getattr(enemy, "lure_target", None) and self.time < enemy.lure_target[1])
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
        for enemy in candidates:
            volume = enemy.block_count if self.vecbreak.enabled else 1
            if volume <= free_slots:
                self._start_blocking(op, enemy)
                free_slots -= volume
            if free_slots <= 0: break

    def _free_block_capacity(self, op: _Operator) -> int:
        occupied = sum(e.block_count for e in op.blocked) if getattr(self, "vecbreak", None) and self.vecbreak.enabled else len(op.blocked)
        return max(0, op.block_count - occupied)

    def _start_blocking(self, op: _Operator, enemy: _Enemy) -> None:
        if enemy.unblockable:
            return
        if self.vecbreak.enabled and enemy.block_count > self._free_block_capacity(op):
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
        *, interrupt_source: _Operator | None = None, source_key: int | None = None,
        stop_radius: float | None = None, stop_after_reached: bool = False,
        stop_within_source_block_range: bool = False,
        duration: float = 1.0,
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
        if source_key is None and None in enemy.pull_sources:
            self.mechanic_diagnostics["PULL_STACK:SOURCE_ID_MISSING"] = 1
        enemy.pull_sources[source_key] = _PullForce(float(origin_row), float(origin_col), distance,
            PULL_FORCE_BY_FORCE_LEVEL[effective_level], duration * (0.5 if effective_level < -1 else 1.0),
            interrupt_source, stop_radius, stop_after_reached,
            stop_within_source_block_range=stop_within_source_block_range)
        if stop_radius is not None:
            self.mechanic_diagnostics["HARPOON_STOP_CHECK_PHASE_CANDIDATE"] = 1
        self._sync_pull_summary(enemy)
        enemy.unbalanced = True
        enemy.unbalance_lock_remaining = max(
            enemy.unbalance_lock_remaining, UNBALANCE_MIN_DURATION
        )
        self._release_enemy_block(enemy)
        return True

    @staticmethod
    def _sync_pull_summary(enemy: _Enemy) -> None:
        """Keep the historical scalar fields as a summary, not a force owner."""
        enemy.pull_remaining = max((p.remaining for p in enemy.pull_sources.values()), default=0.0)
        if enemy.pull_sources:
            latest = next(reversed(enemy.pull_sources.values()))
            enemy.pull_origin_row, enemy.pull_origin_col = latest.origin_row, latest.origin_col
            enemy.pull_initial_distance = latest.initial_distance
            enemy.pull_force = latest.force
            enemy.pull_interrupt_source = latest.interrupt_source
        else:
            enemy.pull_force = enemy.pull_initial_distance = 0.0
            enemy.pull_interrupt_source = None

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
        for key, pull in list(enemy.pull_sources.items()):
            source = pull.interrupt_source
            if source is not None and (source.dead or source.statuses.has(Status.SILENCE)
                                       or source.statuses.has(Status.STUNNED)):
                del enemy.pull_sources[key]
                self.mechanic_diagnostics["HARPOON_INTERRUPT_UPDATE_PHASE_CANDIDATE"] = 1
        for key, pull in list(enemy.pull_sources.items()):
            force_dt = min(self.dt, pull.remaining)
            dr = pull.origin_row - enemy.position_row
            dc = pull.origin_col - enemy.position_col
            distance = math.hypot(dr, dc)
            at_destination = pull.stop_radius is not None and distance <= pull.stop_radius
            source = pull.interrupt_source
            if (not at_destination and pull.stop_within_source_block_range
                    and source is not None):
                # CheckInBlockRange subtracts the margin from radius squared,
                # without checking free block slots or ordinary block status.
                at_destination = (sum((enemy.position()[i] - source.tile[i]) ** 2 for i in (0, 1))
                                  <= source.block_radius ** 2 - .05)
            if at_destination:
                # TryEarlyStopPull only clears the shared rigidbody velocity
                # when this is the sole valid pull source. A retained chain
                # attempts this until successful, then never clears it again.
                if not pull.velocity_stopped and len(enemy.pull_sources) == 1:
                    enemy.force_velocity_row = enemy.force_velocity_col = 0.0
                    pull.velocity_stopped = True
                if pull.stop_after_reached:
                    del enemy.pull_sources[key]
                    continue
            elif distance > 1e-9 and pull.initial_distance > 1e-9:
                force = pull.force * (
                    min(1.0, distance / pull.initial_distance)
                ) ** 4
                enemy.force_velocity_row += dr / distance * force * force_dt
                enemy.force_velocity_col += dc / distance * force * force_dt
            pull.remaining = max(0.0, pull.remaining - self.dt)
            if pull.remaining <= 0:
                del enemy.pull_sources[key]
        self._sync_pull_summary(enemy)
        if len(enemy.pull_sources) > 1:
            self.mechanic_diagnostics["PULL_MULTI_SOURCE_PHYSICS_PHASE_CANDIDATE"] = 1
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
        if self.vecbreak.enabled:
            self.vecbreak.after_force_motion(enemy)
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
        enemy.pull_interrupt_source = None
        enemy.pull_sources.clear()

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
            "WAIT_CURRENT_FRAGMENT_TIME",
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

    def _queue_lemuen_survivor_sp(self, source: _Operator, target: _Enemy, amount: float) -> None:
        if not amount or source.dead or target.dead or target.leaked:
            return
        _, markers = self._lemuen_survivor_sp.setdefault(id(target), (target, []))
        markers.append((source, amount))
        self.mechanic_diagnostics["LEMUEN_MODULE_TARGET_LATE_ENABLE_PHASE_CANDIDATE"] = 1

    def _tick_lemuen_survivor_sp(self, target: _Enemy) -> None:
        pending = self._lemuen_survivor_sp.pop(id(target), None)
        if pending is None or target.dead or target.leaked:
            return
        # BuffContainer.Tick calls OnTick for its buffs, then OnLateTick.
        # The marker's finite lifetime is not a delay before granting SP.
        if self.dt >= 0.10000000149011612:
            return
        for source, amount in pending[1]:
            if not source.dead:
                self._gain_sp(source, amount)

    def _cleanup(self) -> None:
        for enemy in list(self.enemies):
            if enemy.dead or enemy.leaked:
                if self.vecbreak.enabled: self.vecbreak.died_enemy(enemy)
                self.wave_scheduler.departed(enemy, self.time)
            if enemy.leaked and not enemy.dead:
                self.life_points -= enemy.life_reduce
                self.enemies_leaked += 1
                enemy.dead = True
            elif enemy.dead and enemy.blocked_by:
                self._release_enemy_block(enemy)
            if enemy.dead and enemy.component_listeners:
                self._dispatch_component_lifecycle(enemy, "ON_OWNER_FINISH")
                self._dispatch_component_lifecycle(enemy, "ON_BUFF_FINISH")
                self.component_catalog.detach(enemy)
            if enemy.dead:
                # Registered recipients and parents leave with this entity;
                # legacy enemy aura definitions keep their existing path.
                registered = [instance for instance in (*enemy.buffs.active, *enemy.buffs.pending_add)
                    if instance.definition.registered_behavior]
                for instance in registered:
                    enemy.buffs.finish(instance)
                if registered:
                    enemy.buffs.commit()
        for op in self.operators:
            if op.dead:
                self._process_operator_departure(op, retreated=op.retreated)
                for enemy in list(op.blocked):
                    self._release_enemy_block(enemy)
        self.enemies = [e for e in self.enemies if not e.dead]
        for key, (target, markers) in list(self._lemuen_survivor_sp.items()):
            markers[:] = [(source, amount) for source, amount in markers if not source.dead]
            if target.dead or target.leaked or not markers:
                self._lemuen_survivor_sp.pop(key)

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
        if effect in {Status.LEVITATE, Status.GROUND_BOUND}:
            if isinstance(target, _Operator):
                return None
            lifetime = None if duration is None else max(0.0, float(duration))
            if lifetime is not None and (not math.isfinite(lifetime) or lifetime <= 0):
                return None
            target.buffs.event_handler = self._dispatch_buff_event
            definition = BuffDefinition("status:" + effect.value,
                lifetime_type="INFINITY" if lifetime is None else "LIMITED", lifetime=lifetime or 0.0,
                status_flags=(effect.value,), unique_by_key=False, disable_override=True,
                status_resistable="YES" if resistible else "NO")
            item = target.buffs.add(definition, attached_step=self.frame_index)
            target.buffs.commit()
            if effect is Status.GROUND_BOUND:
                self.mechanic_diagnostics["GROUND_BOUND_NAVIGATION_CANDIDATE"] = 1
            return effect if not item.finished and (effect, item) in target.statuses.grants else None
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
        normalized = element if isinstance(element, ElementType) else ElementType(str(element).upper())
        if normalized is ElementType.SANITY and isinstance(target, _Enemy):
            target.sanity_bind_bonuses = {key: item for key, item in target.sanity_bind_bonuses.items()
                                        if self.time < item[0]}
            if target.statuses.has(Status.BIND):
                amount *= max((scale for _, scale in target.sanity_bind_bonuses.values()), default=1.0)
        source_runtime = getattr(source, "skill_runtime", None)
        if source_runtime is not None and isinstance(target, _Enemy):
            amount *= source_runtime.element_output_scale(target)
        for runtime in live_runtimes(self):
            amount *= runtime.element_incoming_scale(target, normalized)
        if getattr(self, "vecbreak", None) and self.vecbreak.enabled:
            amount *= self.vecbreak.element_scale(target)
        result = target.elements.apply(
            element,
            amount,
            neutral=neutral,
            immune=target.statuses.has(Status.ELEMENT_IMMUNE),
            burst_duration=self.element_break_durations.get(normalized.value, (None, None))[
                isinstance(target, _Enemy)],
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
        elif element is ElementType.DARK and enemy_target:
            root = self.register_behavior_buff({'buffKey': 'ep_break_dark_enemy', 'loadFromDB': True},
                owner=target, source=source if source is not None else target)
            target.element_burst.template_root = root
            target.element_burst.template_driven = root is not None
            if root is None:
                self.mechanic_diagnostics['DARK_BURST_LEGACY_FALLBACK'] = 1
        elif element is ElementType.DARK:
            target.statuses.apply(
                Status.SP_BLOCKED, result.burst_duration, resistible=False
            )
            target.statuses.apply(
                Status.SKILL_NOT_ACTIVATABLE,
                result.burst_duration,
                resistible=False,
            )
        if enemy_target:
            for runtime in live_runtimes(self):
                runtime.element_break_started(target)

    def _finish_element_templates(self, target: Any) -> None:
        runtime = getattr(target, 'element_burst', None)
        if runtime is not None and runtime.template_root is not None:
            root = runtime.template_root
            runtime.template_root = None
            root.container.finish(root)
            root.container.commit()

    def _finish_element_burst(self, target: _Enemy | _Operator) -> None:
        self._finish_element_templates(target)
        target.element_burst = None
        for runtime in live_runtimes(self):
            runtime.element_break_finished(target)

    def _tick_element_burst(self, target: _Enemy | _Operator) -> float | None:
        runtime = target.element_burst
        if runtime is None or target.elements.cooldown_remaining <= 0:
            return
        speed = target.buffs.clock_attribute("EP_BREAK_RECOVER_SPEED") if isinstance(target, _Enemy) else 1.0
        elapsed = min(self.dt * speed, target.elements.cooldown_remaining)
        active_delta = min(self.dt, target.elements.cooldown_remaining / speed) if speed > 0 else self.dt
        # The periodic damage belongs to an independent permanent child.
        runtime.tick_elapsed += active_delta
        runtime.effect_elapsed += active_delta
        while runtime.tick_elapsed + 1e-9 >= 1.0 and not target.dead:
            runtime.tick_elapsed -= 1.0
            if runtime.element is ElementType.DARK:
                if isinstance(target, _Enemy):
                    if not runtime.template_driven:
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
        return elapsed

    def _apply_element_burst_damage(
        self,
        target: _Enemy | _Operator,
        amount: float,
        damage_type: DamageType,
        source: _Enemy | _Operator | None,
    ) -> None:
        result = self._apply_damage(
            DamagePacket(
                amount=amount,
                damage_type=damage_type,
                attack_type=AttackType.NORMAL,
                apply_way=ApplyWay.NONE,
                source=source,
                target_resistance_override=self._effective_resistance(target),
            ),
            target,
        )
        if result.killed and isinstance(target, _Enemy):
            self.enemies_killed += 1

    @staticmethod
    def _effective_resistance(target: _Enemy | _Operator, *, modifiers=None) -> float:
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
        own = getattr(target, "skill_runtime", None)
        container = getattr(target, "buffs", None)
        if modifiers is None:
            modifiers = (container.modifiers("MAGIC_RESISTANCE", registered_only=True)
                         if container is not None else ())
        addition = sum(m.value for m in modifiers if m.formula == "ADDITION")
        return (float(getattr(target, "res", 0.0)) - reduction
                + addition
                + float(getattr(target, "roster_res_delta", 0))
                + (own.res_bonus() if own else 0))

    @staticmethod
    def _outgoing_damage_scale(source: _Enemy | _Operator) -> float:
        runtime = getattr(source, "element_burst", None)
        if (
            runtime is None
            or runtime.element is not ElementType.DARK
            or runtime.template_driven
            or not getattr(getattr(source, "elements", None), "enemy_unit", False)
        ):
            return 1.0
        remaining_ratio = max(
            0.0,
            min(1.0 - runtime.effect_elapsed / runtime.duration, 1.0),
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
        result = self._apply_damage(
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
                    dr, dc = -forward, side
                elif op.facing == 2:  # left
                    dr, dc = -side, -forward
                else:  # up
                    dr, dc = forward, -side
                cells.add((op.tile[0] + dr, op.tile[1] + dc))
        return cells

    def behavior_aoe_damage(
        self, context: BehaviorContext, node: dict[str, Any]
    ) -> bool:
        source = context.entity(node.get("_sourceType"))
        if not self._validate_aoe_binding(context, node):
            return False
        if context.template_key == "huang_s_3" and not node.get("_useRadius") and not node.get("_rangeId"):
            # Preserve the older reviewed Huang finish adapter; its ability
            # range is not part of the new generic selector coverage.
            cells = self._front_3x3_cells(source)
            amount = source.atk * source.atk_scale * float(context.blackboard.get(node.get("_damageScale"), 1))
            hit = False
            for enemy in tuple(self.enemies):
                if not enemy.dead and enemy.tile() in cells:
                    hit = self.behavior_deal_damage(source, enemy, amount, node.get("_damageType", "PHYSICAL")) or hit
            return hit
        targets = self.behavior_range_targets(context, node)
        base = context.cached_attack
        if base is None:
            base = self.operator_attack_power(source) if isinstance(source, _Operator) else self._effective_enemy_atk(source)
        scale = float(context.blackboard.get(node.get("_damageScale"), 1))
        amount = base * scale
        hit = False
        for target in targets:
            # Selection and base attack are fixed for this action. An earlier
            # hit may nevertheless kill a later receiver through another hook.
            if target.dead or (isinstance(target, _Enemy) and not self._enemy_is_attackable(target)):
                continue
            hit = self._behavior_hit_damage(source, target, amount,
                node.get("_damageType", "PHYSICAL"), node.get("_attackType", "NORMAL"),
                node.get("_sourceApplyWay", "NONE"), node.get("_modifierKey", "")) or hit
            # Keep this adapter's existing post-damage status order, including
            # a lethal packet; exact native disposal/notification is candidate.
            self.behavior_apply_hit_buffs(target, node.get("_buffs") or [], context)
        return hit

    def behavior_create_buff(
        self, context: BehaviorContext, node: dict[str, Any]
    ) -> bool:
        owner = context.entity(node.get("_buffOwner"))
        spec = node.get("_buff", {})
        key = str(spec.get("buffKey", ""))
        if owner is None or not key:
            return False
        if node.get("_isDerivedBuff") and not self._validate_derived_buff_binding(context, node):
            return False

        if context.register_buffs:
            source = (context.entity(node.get("_specialBuffSource"))
                      if node.get("_useSpecialBuffSource") else context.source)
            return self.register_behavior_buff(spec, owner=owner, source=source,
                blackboard=context.blackboard, refresh=context.refresh_attribute_buffs,
                parent=context.buff_instance, finish_with_parent=bool(node.get("_isDerivedBuff"))) is not None

        container = getattr(owner, "buffs", None)
        if context.buff_container is not None and container is not None:
            if context.buff_instance is not None and context.buff_instance.definition.registered_behavior:
                source = (context.entity(node.get("_specialBuffSource"))
                          if node.get("_useSpecialBuffSource") else context.source)
                return self.register_behavior_buff(spec, owner=owner, source=source,
                    blackboard=context.blackboard, parent=context.buff_instance, commit=False,
                    finish_with_parent=bool(node.get("_isDerivedBuff"))) is not None
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

    @staticmethod
    def _flag_finish_binding_issue(owner: Any, flag: str) -> str | None:
        if owner is None:
            return None
        if not isinstance(owner, (_Enemy, _Operator)):
            return "flag_finish_owner_binding"
        if owner.buffs.flagged_instances(ABNORMAL_READ_FLAGS.get(flag, flag), pending=True):
            return "flag_finish_pending_birth_binding"
        return None

    def behavior_finish_flag_buffs(self, context: BehaviorContext, node: dict[str, Any]) -> bool:
        owner = context.entity(node.get("_targetType"))
        flag = ABNORMAL_READ_FLAGS[node["_abnormalFlag"]]
        if issue := self._flag_finish_binding_issue(owner, flag):
            return self.behavior._unsupported(node_name(node) + ":" + issue)
        if owner is not None:
            for item in owner.buffs.flagged_instances(flag):
                owner.buffs.finish(item)
        # Native Execute returns success, including an empty/no-target cleanup.
        # MarkFinish is queued; the existing container owns the finish phase.
        return True

    def behavior_finish_buff(
        self, owner: _Enemy | _Operator | None, key: str
    ) -> bool:
        if owner is None:
            return False
        container = getattr(owner, "buffs", None)
        if container is not None and hasattr(container, "finish_by_key"):
            if container.finish_by_key(key):
                return True
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
            if isinstance(mode_index, bool) or float(mode_index) != int(mode_index) or int(mode_index) < 0:
                return False
            mode_index = int(mode_index)
        except (TypeError, ValueError, OverflowError):
            return False
        if isinstance(target, _Operator):
            return self.component_catalog.switch_operator_mode(self, target, mode_index)
        target.mode_index = mode_index
        return True

    # ------------------------------------------------------------- deploy
    def _execute_plan_action(self, plan: OperatorPlan) -> str:
        if plan.action == "RETREAT":
            attempt = self._try_retreat(plan.char_id)
        elif plan.action == "SKILL":
            attempt = self._try_manual_skill(plan.char_id)
        elif plan.action == "SKILL_END":
            attempt = self._try_skill_end(plan.char_id)
        elif plan.action == "SWITCH_MODE":
            attempt = self._try_switch_mode(plan.char_id, plan.mode)
        elif plan.action == "DEPLOY":
            attempt = self._try_deploy(plan)
        else:
            attempt = OperationAttempt("skip", "unsupported_action")
        if attempt.status == "wait" and plan.on_failure != "WAIT":
            attempt = OperationAttempt("skip", attempt.reason, attempt.details)
        self.operation_log.record(self.plan_index, attempt, self.time, self.frame_index)
        if attempt.status == "skip" and plan.on_failure == "STOP":
            self.end_reason = "operation_failed"
        return attempt.status

    def _active_operator(self, char_id: str) -> _Operator | None:
        for op in reversed(self.operators):
            if (op.char_id == char_id or getattr(op, "stage_alias", None) == char_id) and not op.dead:
                return op
        return None

    def _try_retreat(self, char_id: str) -> OperationAttempt:
        op = self._active_operator(char_id)
        if op is None:
            return OperationAttempt("skip", "not_on_field")
        if getattr(op, "manual_retreat_forbidden", False):
            return OperationAttempt("skip", "retreat_forbidden")
        op.retreated = True
        op.dead = True
        self._process_operator_departure(op, retreated=True)
        self._sync_squad_effects()
        return OperationAttempt("ok", details={"deployment": op.deployment_index})

    def _try_manual_skill(self, char_id: str) -> OperationAttempt:
        op = self._active_operator(char_id)
        if op is None:
            return OperationAttempt("skip", "not_on_field")
        if op.skill is None:
            return OperationAttempt("skip", "skill_unavailable" if op.skill_end_reason == "PERMANENT" else "no_skill")
        if op.skill_type != "MANUAL":
            return OperationAttempt("skip", "not_manual_skill")
        if op.skill_prefab_id == WHITW2_S1:
            return self._try_switch_mode(char_id)
        if op.form_runtime is not None and op.form_runtime.mode == "THANATOS":
            if op.form_runtime.manual_switch():
                return OperationAttempt("ok", details={"deployment":op.deployment_index, "form":"ORPHEUS"})
            return OperationAttempt("wait", "form_switching")
        details = {"deployment": op.deployment_index, "sp": op.sp, "required_sp": self._skill_cost(op),
                   "startup_remaining": max(0.0, op.skill_startup_until - self.time),
                   "active_until": op.active_until,
                   "statuses": sorted(status.value for status in op.statuses.active_statuses())}
        if self._skill_starting(op):
            return OperationAttempt("wait", "skill_starting", details)
        if op.skill_active:
            return OperationAttempt("wait", "skill_active", details)
        if not op.statuses.can_use_skill:
            return OperationAttempt("wait", "skill_disabled", details)
        if not self._skill_ready(op):
            return OperationAttempt("wait", "insufficient_sp", details)
        if not self._cast_skill(op):
            return OperationAttempt("skip", "skill_rejected",
                                    {**details, "sp_after": op.sp, "casts_after": op.skill_cast_count})
        return OperationAttempt("ok", details={**details, "cast": op.skill_cast_count})

    def _try_skill_end(self, char_id: str) -> OperationAttempt:
        op = self._active_operator(char_id)
        if op is None:
            return OperationAttempt("skip", "not_on_field")
        if not op.skill_controls.can_end:
            return OperationAttempt("skip", "skill_not_endable")
        details = {"deployment": op.deployment_index, "sp": op.sp, "ammo": op.ammo}
        if self._skill_starting(op):
            return OperationAttempt("wait", "skill_starting", details)
        if not op.skill_active:
            return OperationAttempt("skip", "skill_not_active", details)
        if not op.statuses.can_use_skill:
            return OperationAttempt("wait", "skill_disabled", details)
        self._finish_skill(op, "MANUAL_END")
        return OperationAttempt("ok", details={**details, "sp_after": op.sp})

    def _try_switch_mode(self, char_id: str, mode: int | None = None) -> OperationAttempt:
        op = self._active_operator(char_id)
        if op is None:
            return OperationAttempt("skip", "not_on_field")
        if not op.skill_controls.can_switch:
            return OperationAttempt("skip", "skill_not_switchable")
        target = 1 - op.skill_mode if mode is None else mode
        details = {"deployment": op.deployment_index, "mode_before": op.skill_mode,
                   "target_mode": target, "sp": op.sp, "required_sp": self._skill_cost(op)}
        if self._skill_starting(op):
            return OperationAttempt("wait", "skill_starting", details)
        if target == op.skill_mode:
            return OperationAttempt("ok", details={**details, "unchanged": True, "sp_after": op.sp})
        if not op.statuses.can_use_skill:
            return OperationAttempt("wait", "skill_disabled", details)
        if not self._skill_ready(op):
            return OperationAttempt("wait", "insufficient_sp", details)
        if not self._cast_skill(op, switch_mode=True):
            return OperationAttempt("skip", "skill_rejected", details)
        return OperationAttempt("ok", details={**details, "sp_after": op.sp,
                                              "cast": op.skill_cast_count})

    def _process_operator_departure(
        self, op: _Operator, *, retreated: bool, refund_withdraw_cost: bool = True,
        preserve_redeploy_state: bool = False,
    ) -> None:
        if op.departure_processed:
            return
        self.unlock_skill_progress(op)
        op.departure_processed = True
        if self.vecbreak.enabled: self.vecbreak.depart(op)
        self._dispatch_component_lifecycle(op, "ON_OWNER_FINISH")
        self._dispatch_component_lifecycle(op, "ON_BUFF_FINISH")
        self.component_catalog.detach(op)
        respawn_scale = min([1.0] + [r.departure_respawn_scale(op) for r in effect_runtimes(self)
                                    if not r.op.dead or r.op is op])
        if self.vecbreak.enabled: respawn_scale *= self.vecbreak.respawn(op)
        if op.form_runtime is not None:
            op.form_runtime.depart()
        if op.healing_runtime is not None:
            op.healing_runtime.depart()
        if op.skill_runtime is not None:
            op.skill_runtime.depart()
        for instance in (*op.buffs.active, *op.buffs.pending_add):
            op.buffs.finish(instance)
        op.buffs.commit()
        for runtime in effect_runtimes(self):
            runtime.clear_angel_blessings(op)
            runtime.remove_element_aura_member(op)
        if retreated and refund_withdraw_cost and op.subprofession != "merchant":
            refund = math.floor(min(op.deployed_cost * 0.5, op.base_cost))
            self.dp = min(self.max_cost, self.dp + refund)
        if op.target_selector is not None:
            op.target_selector.dispose()
        if not preserve_redeploy_state:
            penalty = min(self.redeploy_penalty.get(op.char_id, 0) + 1, 2)
            self.redeploy_penalty[op.char_id] = 0 if self.vecbreak.enabled and op.char_id == VEC_TRAP else penalty
            self.redeploy_ready_at[op.char_id] = self.time + op.respawn_time * respawn_scale

    def _try_deploy(self, plan: OperatorPlan) -> OperationAttempt:
        if plan.char_id.startswith("trap_") and plan.char_id not in predefined_builds(self.level):
            return OperationAttempt("skip", "stage_card_unavailable")
        if plan.char_id == SOUL_WARD:
            return OperationAttempt("skip", "token_automatic_spawn_only")
        if plan.char_id == "token_10057_svash2_eagle1":
            return OperationAttempt("skip", "token_skill_not_active")
        if plan.char_id in TOKEN_SKILLS:
            owner = next((r for r in live_runtimes(self) if r.sid == TOKEN_SKILLS[plan.char_id]), None)
            return owner.deploy_token(plan) if owner else OperationAttempt("wait", "token_owner_not_on_field")
        for receiver in live_runtimes(self):
            if plan.char_id in receiver.relocation_reserve:
                return receiver.redeploy_relocated(plan)
        if plan.char_id not in self.characters:
            return OperationAttempt("skip", "unknown_operator")
        if plan.tile is None:
            return OperationAttempt("skip", "missing_tile")
        if self._active_operator(plan.char_id) is not None:
            return OperationAttempt("skip", "already_deployed")
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
        cost = self.deployment_cost(plan.char_id, base_cost)
        row, col = plan.tile
        profession = deployment_position(self.characters[plan.char_id], plan.char_id)
        details = {"tile": list(plan.tile), "cost": cost, "available_cost": self.dp,
                   "cost_lower_bound": self.cost_lower_bound,
                   "occupied_slots": sum(1 for op in self.operators if not op.dead and getattr(op, "uses_deployment_slot", True)),
                   "slot_limit": self.character_limit,
                   "redeploy_ready_at": self.redeploy_ready_at.get(plan.char_id, 0.0)}
        if not (0 <= row < len(self.map_grid) and 0 <= col < len(self.map_grid[0])):
            return OperationAttempt("skip", "out_of_bounds", details)
        if self.tile_occupied(row, col):
            return OperationAttempt("skip", "tile_occupied", details)
        if not self.is_buildable(row, col, profession):
            return OperationAttempt("skip", "not_buildable", details)
        blockers = []
        if self.time + 1e-9 < details["redeploy_ready_at"]:
            blockers.append("redeploy_cooldown")
        if details["occupied_slots"] >= self.character_limit and plan.char_id != VEC_TRAP:
            blockers.append("deployment_limit")
        if self.dp - cost < self.cost_lower_bound:
            blockers.append("insufficient_cost")
        if blockers:
            details["blockers"] = blockers
            return OperationAttempt("wait", blockers[0], details)
        self.dp -= cost
        self.operators.append(self._make_operator(plan, cost))
        deployed = self.operators[-1]
        if deployed.skill_runtime is not None:
            deployed.skill_runtime.cost_spent(cost)
        self._operator_deployed(deployed)
        return OperationAttempt("ok", details={**details, "deployment": deployed.deployment_index})

    def _operator_deployed(self, deployed: _Operator) -> None:
        """Shared field-entry notifications, including cost-free relocation."""
        self._dispatch_component_lifecycle(deployed, "ON_BUFF_START")
        if deployed.skill_runtime is not None:
            deployed.skill_runtime.born()
        for runtime in live_runtimes(self):
            runtime.deployed(deployed)
        for runtime in effect_runtimes(self):
            shield = runtime.deck_shields.pop(deployed.char_id, None)
            if shield:
                deployed.damage_rules.append(DamageRule("BARRIER", value=shield))
                runtime.deck_discounts.pop(deployed.char_id, None)
                runtime.deck_discount = None
        self.squad_effects.deployed(deployed.char_id)
        self._sync_squad_effects()
        if self.vecbreak.enabled: self.vecbreak.born_operator(deployed)

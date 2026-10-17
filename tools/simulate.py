"""Run a plan and publish replay JSON to the local latest directory."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arksim import __version__
from arksim.battle import Battle, CLIENT_FP_SCALE, CLIENT_LOGIC_RATE
from arksim import data as D
from arksim.mechanics import Status
from arksim.cli import DEFAULT_DATA_DIR, build_config
from arksim.cli import parse_plan, resolve_level_file
from arksim.replay_edit import frame_number, replace_future_plan, verify_prefix, runtime_signature
from arksim.squad import BUILD_FIELDS
from arksim.random_trace import RandomDecisionRng

def compact_enemy(enemy: Any, battle: Battle) -> dict[str, Any]:
    route_index = next(
        (
            index
            for index, route in enumerate(battle.routes)
            if route == enemy.route
        ),
        -1,
    )
    state = {
        "id": enemy.enemy_index,
        "key": enemy.key,
        "name": enemy.name,
        "levelType": enemy.level_type,
        "r": round(enemy.position_row, 4),
        "c": round(enemy.position_col, 4),
        "hp": round(enemy.hp, 2),
        "extraHp": round(enemy.extra_hp, 2),
        "extraHpMax": round(enemy.max_extra_hp_ratio * enemy.max_hp, 2),
        "minusHp": round(enemy.minus_hp, 2),
        "minusHpMax": round(enemy.max_minus_hp_ratio * enemy.max_hp, 2),
        "max": round(enemy.max_hp, 2),
        "blocked": (
            enemy.blocked_by.deployment_index if enemy.blocked_by else None
        ),
        "blockStable": (
            [round(enemy.block_stable_row, 4), round(enemy.block_stable_col, 4)]
            if enemy.block_stable_row is not None
            and enemy.block_stable_col is not None
            else None
        ),
        "blockShift": round(enemy.block_shift_remaining, 4),
        "route": route_index,
        "routePoint": enemy.route_idx,
        # Portal extension frames have an active hitbox and are rendered as
        # present even though route visibility has not reopened yet.
        "hidden": enemy.disappeared and not enemy.portal_targetable,
        "taunt": enemy.taunt_level,
    }
    if enemy.facing is not None:
        state["facing"] = round(enemy.facing.value_at(battle.time), 4)
    if battle.vecbreak.enabled:
        state["statuses"] = sorted(s.value for s in enemy.statuses.active_statuses())
        boss = battle.vecbreak.bosses.get(enemy.enemy_index)
        if boss: state["vecPhase"], state["shieldStacks"] = boss["phase"], boss["stacks"]
    if battle.mansfield.enabled:
        runtime = battle.mansfield.states.get(enemy.enemy_index)
        state["statuses"] = sorted(s.value for s in enemy.statuses.active_statuses())
        if runtime:
            state["mansfieldMode"] = runtime["mode"]
            state["mansfieldPhase"] = runtime["phase"]
    if enemy.wisdel_remnant_source is not None and not enemy.wisdel_remnant_source.dead:
        state["remnantSource"] = enemy.wisdel_remnant_source.deployment_index
    if battle.wdex.enabled and enemy.enemy_index in battle.wdex.states:
        state["wdExMode"] = enemy.mode_index
        state["statuses"] = sorted(s.value for s in enemy.statuses.active_statuses())
    muzzle = battle.enemy_muzzle_position_v3(enemy)
    if muzzle is not None:
        state["muzzle"] = {"r": round(muzzle[0], 4), "c": round(muzzle[1], 4), "z": round(muzzle[2], 4)}
    return state


def compact_operator(operator: Any, battle: Battle) -> dict[str, Any]:
    periodic = (
        battle.assumptions.headb2_periodic_sp_recovery
        and operator.skill_prefab_id == "skchr_headb2_2"
        and math.isclose(battle.dt, 1 / CLIENT_LOGIC_RATE, abs_tol=1e-12)
        and operator.sp_increment == 1.0
        and operator.squad_bonus.sp_recovery == 0.0
    )
    progress = (
        1.0 - operator.sp_recovery_remaining_raw / CLIENT_FP_SCALE
        if periodic else operator.sp % 1.0
        if operator.sp_type == "INCREASE_WITH_TIME" else None
    )
    muzzle = battle.operator_muzzle_position_v3(operator)
    return {
        "id": operator.deployment_index,
        "char": operator.char_id,
        "profile": operator.plan_profile,
        "muzzle": {"r": round(muzzle[0], 4), "c": round(muzzle[1], 4), "z": round(muzzle[2], 4)} if muzzle else None,
        "muzzleMount": operator.last_projectile_mount,
        "name": operator.name,
        "r": operator.tile[0],
        "c": operator.tile[1],
        "hp": round(operator.hp, 2),
        "minusHp": round(operator.minus_hp, 2),
        "minusHpMax": round(operator.max_minus_hp_ratio * operator.max_hp, 2),
        "extraHp": round(operator.extra_hp, 2),
        "extraHpMax": round(operator.max_extra_hp_ratio * operator.max_hp, 2),
        "max": round(operator.max_hp, 2),
        "sp": round(operator.sp, 2),
        "spMax": round(operator.sp_max, 2),
        "spSlot": round(max(0.0, min(1.0, progress)), 6) if progress is not None else None,
        "spSlotMode": "periodic" if periodic else "continuous" if progress is not None else "none",
        "spSlotPaused": (
            operator.dead or (operator.skill_active and not operator.skill_controls.can_switch)
            or operator.skill_startup_until >= 0.0
            or operator.statuses.has(Status.SP_BLOCKED)
            or operator.sp >= operator.sp_max
        ),
        "attributes": {
            "atk": round(battle.operator_attack_power(operator), 2),
            "def": round(battle.operator_defense(operator), 2),
            "res": round(battle._effective_resistance(operator), 2),
            "attackInterval": round(battle._effective_attack_interval(operator), 4),
            "blockCapacity": operator.block_count if operator.statuses.can_block else 0,
        },
        "starting": operator.skill_startup_until >= 0.0,
        "active": operator.skill_active or operator.skill_end_reason == "PERMANENT",
        "casts": operator.skill_cast_count,
        "skillMode": operator.skill_mode if operator.skill_controls.can_switch else None,
        "canEndSkill": operator.skill_controls.can_end,
        "canSwitchMode": operator.skill_controls.can_switch,
        "skillEndReason": operator.skill_end_reason,
        "skillState": operator.skill_runtime.snapshot() if operator.skill_runtime else None,
        "summoner": getattr(getattr(operator, "summoner", None), "deployment_index", None),
        "camouflaged": operator.statuses.has(Status.CAMOUFLAGE),
        "form": operator.form_runtime.mode if operator.form_runtime is not None else None,
        "atkScale": round(operator.atk_scale, 4),
        "defScale": round(operator.def_scale, 4),
        "blocked": len([enemy for enemy in operator.blocked if not enemy.dead]),
        "dead": operator.dead,
        "range": sorted([list(cell) for cell in operator.range_cells]),
    }


def compact_projectile(projectile: Any) -> dict[str, Any]:
    progress = min(
        projectile.travelled / projectile.initial_distance,
        1.0,
    ) if projectile.initial_distance > 1e-9 else 1.0
    return {
        "id": projectile.projectile_id,
        "key": projectile.key,
        "side": projectile.side,
        "origin": list(projectile.origin) if getattr(projectile, "origin", None) is not None else None,
        "mount": getattr(projectile, "mount_point", None),
        "r": round(projectile.position_row, 4),
        "c": round(projectile.position_col, 4),
        "z": round(projectile.position_z, 4),
        "arc": round(4.0 * progress * (1.0 - progress), 4)
        if projectile.parabolic
        else 0.0,
    }


def export_roster(battle: Battle) -> list[dict[str, Any]]:
    """Keep plan profiles separate so redeployments can use different builds."""
    roster = []
    planned = {item.char_id for item in battle.plan if item.action == "DEPLOY"}
    profiles = list(enumerate(battle.plan)) + [
        (len(battle.plan) + index, item) for index, item in enumerate(battle.squad)
        if item.char_id not in planned
    ]
    for index, plan in profiles:
        if plan.action != "DEPLOY" or plan.char_id not in battle.characters:
            continue
        character = battle.characters[plan.char_id]
        attrs = D.char_attributes(
            character, plan.level, plan.elite, trust=plan.trust,
            potential_rank=plan.potential_rank,
            module=battle.battle_equips.get(plan.module_id) if plan.module_id else None,
            module_level=plan.module_level,
        )
        roster.append({
            "profile": index, "char": plan.char_id,
            "planned": plan.char_id in planned,
            "name": character.get("name", plan.char_id),
            "baseCost": int(attrs.get("cost", 0)),
            "loadout": {key: getattr(plan, key) for key in (
                "elite", "level", "trust", "potential_rank", "skill_id", "skill_level", "auto_skill", "module_id", "module_level",
            )},
            "attributes": {
                "maxHp": round(float(attrs.get("maxHp", 0)), 2),
                "atk": round(float(attrs.get("atk", 0)), 2),
                "def": round(float(attrs.get("def", 0)), 2),
                "res": round(float(attrs.get("magicResistance", 0)), 2),
                "attackInterval": round(float(attrs.get("baseAttackTime", 1.6)) * 100 / (float(attrs.get("attackSpeed", 100)) or 100), 4),
                "blockCapacity": int(attrs.get("blockCnt", 1)),
            },
        })
    for operator in battle.operators:
        if not hasattr(operator, "summoner") and not hasattr(operator, "stage_alias"):
            continue
        roster.append({"profile": -operator.deployment_index, "char": operator.char_id,
                       "planned": False, "reserveEligible": False, "name": operator.name,
                       "baseCost": operator.base_cost, "loadout": {},
                       "attributes": {"maxHp": operator.max_hp, "atk": operator.atk,
                                      "def": operator.defense, "res": operator.res,
                                      "attackInterval": operator.attack_interval,
                                      "blockCapacity": operator.block_count}})
    return roster


def reserve_state(battle: Battle, roster: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profiles = {}
    for profile in roster:
        if profile.get("reserveEligible") is False:
            continue
        previous = profiles.get(profile["char"])
        if previous is None or previous["profile"] < battle.plan_index:
            profiles[profile["char"]] = profile
    active = {op.char_id for op in battle.operators if not op.dead}
    cards = []
    for char_id, profile in profiles.items():
        if char_id in active:
            continue
        remaining = max(0.0, battle.redeploy_ready_at.get(char_id, 0.0) - battle.time)
        cost = battle.deployment_cost(char_id, profile["baseCost"])
        status = "cooldown" if remaining > 1e-9 else "cost" if battle.dp - cost < battle.cost_lower_bound else "ready"
        cards.append({"profile": profile["profile"], "char": char_id, "cost": cost, "status": status, "remaining": round(remaining, 3)})
    return cards


def capture_state(battle: Battle, roster: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    dp_progress = min(1.0, max(0.0, battle.dp_timer / battle.cost_increase_time)) if battle.cost_increase_time > 0 else 0.0
    return {
        "vecBreak": battle.vecbreak.snapshot() if battle.vecbreak.enabled else None,
        "mansfield": battle.mansfield.snapshot() if battle.mansfield.enabled else None,
        "wdEx": battle.wdex.snapshot() if battle.wdex.enabled else None,
        "cw": battle.cw.snapshot() if battle.cw.enabled else None,
        "dublin": battle.dublin.snapshot() if battle.dublin.enabled else None,
        "time": round(battle.time, 4),
        "dp": battle.dp,
        "dpDisplay": round(min(battle.max_cost, battle.dp + dp_progress), 4),
        "reserve": reserve_state(battle, roster) if roster is not None else [],
        "life": battle.life_points,
        "spawned": battle.enemies_spawned,
        "killed": battle.enemies_killed,
        "leaked": battle.enemies_leaked,
        "enemies": {
            enemy.enemy_index: compact_enemy(enemy, battle)
            for enemy in battle.enemies
        },
        "enemyRefs": {
            enemy.enemy_index: enemy for enemy in battle.enemies
        },
        "operators": {
            operator.deployment_index: compact_operator(operator, battle)
            for operator in battle.operators
        },
        "operatorRefs": {
            operator.deployment_index: operator for operator in battle.operators
        },
        "projectiles": {
            projectile.projectile_id: compact_projectile(projectile)
            for projectile in [*battle.projectiles, *[visual for operator in battle.operators
                               if operator.skill_runtime for visual in operator.skill_runtime.visuals()]]
            if getattr(projectile, "born_at", None) is None
               or projectile.born_at <= battle.time + 1e-9
        },
        "effects": {
            id(effect): {
                "tile": list(effect.tile),
                "executeAt": round(effect.execute_at, 4),
            }
            for effect in battle.scheduled_highland_effects
        },
    }


def snapshot(state: dict[str, Any]) -> dict[str, Any]:
    return {
        **({"vecBreak": state["vecBreak"]} if state.get("vecBreak") else {}),
        **({"mansfield": state["mansfield"]} if state.get("mansfield") else {}),
        **({"wdEx": state["wdEx"]} if state.get("wdEx") else {}),
        **({"cw": state["cw"]} if state.get("cw") else {}),
        **({"dublin": state["dublin"]} if state.get("dublin") else {}),
        "t": round(state["time"], 3),
        "fixedFrame": round(state["time"] * 30),
        "dp": state["dp"],
        "dpDisplay": state["dpDisplay"],
        "reserve": state["reserve"],
        "life": state["life"],
        "spawned": state["spawned"],
        "killed": state["killed"],
        "leaked": state["leaked"],
        "operators": list(state["operators"].values()),
        "enemies": list(state["enemies"].values()),
        "projectiles": list(state["projectiles"].values()),
    }


def make_event(time: float, kind: str, **details: Any) -> dict[str, Any]:
    return {"t": round(time, 3), "kind": kind, **details}


def derive_events(
    before: dict[str, Any], after: dict[str, Any]
) -> list[dict[str, Any]]:
    time = after["time"]
    events: list[dict[str, Any]] = []
    before_enemies = before["enemies"]
    after_enemies = after["enemies"]
    for enemy_id in after_enemies.keys() - before_enemies.keys():
        enemy = after_enemies[enemy_id]
        events.append(make_event(time, "spawn", target=enemy_id, key=enemy["key"], r=enemy["r"], c=enemy["c"]))
    for enemy_id in before_enemies.keys() & after_enemies.keys():
        old = before_enemies[enemy_id]
        new = after_enemies[enemy_id]
        damage = old["hp"] - new["hp"]
        if damage > 0.005:
            events.append(make_event(time, "enemy_damage", target=enemy_id, amount=round(damage, 2), hp=new["hp"], r=new["r"], c=new["c"]))
        if old["blocked"] != new["blocked"]:
            events.append(make_event(time, "block" if new["blocked"] else "unblock", target=enemy_id, operator=new["blocked"], r=new["r"], c=new["c"]))
        if old["hidden"] != new["hidden"]:
            events.append(make_event(time, "disappear" if new["hidden"] else "appear", target=enemy_id, r=new["r"], c=new["c"]))
    for enemy_id in before_enemies.keys() - after_enemies.keys():
        old = before_enemies[enemy_id]
        ref = before["enemyRefs"][enemy_id]
        kind = "leak" if ref.leaked else "enemy_death"
        events.append(make_event(time, kind, target=enemy_id, key=old["key"], r=round(ref.position_row, 4), c=round(ref.position_col, 4), reason=ref.death_reason))

    before_ops = before["operators"]
    after_ops = after["operators"]
    for operator_id in after_ops.keys() - before_ops.keys():
        operator = after_ops[operator_id]
        events.append(make_event(time, "deploy", target=operator_id, r=operator["r"], c=operator["c"]))
    for operator_id in before_ops.keys() & after_ops.keys():
        old = before_ops[operator_id]
        new = after_ops[operator_id]
        damage = old["hp"] - new["hp"]
        if damage > 0.005:
            events.append(make_event(time, "operator_damage", target=operator_id, amount=round(damage, 2), hp=new["hp"], r=new["r"], c=new["c"]))
        if new["casts"] > old["casts"]:
            events.append(make_event(time, "skill_cast", target=operator_id, cast=new["casts"], atkScale=new["atkScale"], defScale=new["defScale"], r=new["r"], c=new["c"]))
        if old["starting"] and not new["starting"] and new["active"]:
            events.append(make_event(time, "skill_active", target=operator_id, cast=new["casts"], atkScale=new["atkScale"], defScale=new["defScale"], r=new["r"], c=new["c"]))
        if old["active"] and not new["active"]:
            events.append(make_event(time, "skill_end", target=operator_id, cast=new["casts"], reason=new["skillEndReason"], r=new["r"], c=new["c"]))
        if old["skillMode"] != new["skillMode"]:
            events.append(make_event(time, "mode_change", target=operator_id, mode=new["skillMode"], r=new["r"], c=new["c"]))
        if not old["dead"] and new["dead"]:
            ref = after["operatorRefs"][operator_id]
            events.append(make_event(time, "retreat" if ref.retreated else "operator_death",
                                     target=operator_id, r=new["r"], c=new["c"]))

    before_projectiles = before["projectiles"]
    after_projectiles = after["projectiles"]
    for projectile_id in after_projectiles.keys() - before_projectiles.keys():
        projectile = after_projectiles[projectile_id]
        events.append(make_event(time, "projectile_launch", projectile=projectile_id, key=projectile["key"], side=projectile["side"], r=projectile["r"], c=projectile["c"]))
    for projectile_id in before_projectiles.keys() - after_projectiles.keys():
        projectile = before_projectiles[projectile_id]
        events.append(make_event(time, "projectile_end", projectile=projectile_id, key=projectile["key"], side=projectile["side"], r=projectile["r"], c=projectile["c"]))

    for effect_id in after["effects"].keys() - before["effects"].keys():
        effect = after["effects"][effect_id]
        events.append(make_event(time, "highland_trigger", tile=effect["tile"], executeAt=effect["executeAt"], r=effect["tile"][0], c=effect["tile"][1]))
    for effect_id in before["effects"].keys() - after["effects"].keys():
        effect = before["effects"][effect_id]
        events.append(make_event(time, "highland_effect", tile=effect["tile"], r=effect["tile"][0], c=effect["tile"][1]))
    if after["life"] != before["life"]:
        events.append(make_event(time, "life_change", life=after["life"]))
    return events


def export_map(level: dict[str, Any]) -> dict[str, Any]:
    grid = level["mapData"]["map"]
    tiles = level["mapData"]["tiles"]
    height = len(grid)
    width = len(grid[0])
    cells = []
    for row in range(height):
        source_row = height - 1 - row
        cells.append(
            [
                {
                    "key": tiles[grid[source_row][col]].get("tileKey", ""),
                    "height": tiles[grid[source_row][col]].get("heightType", ""),
                    "build": tiles[grid[source_row][col]].get("buildableType", ""),
                    "pass": tiles[grid[source_row][col]].get("passableMask", ""),
                }
                for col in range(width)
            ]
        )
    return {"width": width, "height": height, "cells": cells}


def export_routes(battle: Battle) -> list[list[dict[str, Any]]]:
    return [
        [
            {
                "r": round(point.row, 4),
                "c": round(point.col, 4),
                "kind": point.kind,
                "reachable": point.reachable,
            }
            for point in route
        ]
        for route in battle.routes
    ]


def run_case(
    label: str,
    plan_path: Path,
    shared: dict[str, Any],
    sample_interval: float,
    plan_edits: list[tuple[int, list]] | None = None,
) -> dict[str, Any]:
    battle = Battle(
        shared["level"],
        shared["enemy_index"],
        shared["characters"],
        shared["ranges"],
        shared["plan"],
        shared["skills"],
        shared["attack_timing"],
        shared["behavior_templates"],
        shared["battle_equips"],
        shared["projectile_data"],
        shared.get("buff_abilities"),
        seed=shared["seed"],
        strict_behaviors=shared["strict_behaviors"],
        spawn_timing=shared["spawn_timing"],
        enemy_attack_timing=shared["enemy_attack_timing"],
        enemy_graphics=shared.get("enemy_graphics"),
        assumptions=shared.get("assumptions"),
        module_index=shared.get("module_index"),
        favor_table=shared.get("favor_table"),
        squad=shared.get("squad"),
        operator_forms=shared.get("operator_forms"),
        operator_graphics=shared.get("operator_graphics"),
        random_trace=shared.get("random_trace", False),
        random_tape=shared.get("random_tape"),
        module_context=shared.get("module_context"),
        behavior_components=shared.get("behavior_components"),
    )
    frames: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    roster = export_roster(battle)
    roster_history = {(item["profile"], item["char"]): item for item in roster}
    edits = iter(plan_edits or [])
    edit = next(edits, None)
    before = capture_state(battle, roster)
    frames.append(snapshot(before))
    next_sample = sample_interval
    while battle.time < 600:
        if edit is not None and battle.frame_index == edit[0]:
            replace_future_plan(battle, edit[1], edit[0])
            roster = export_roster(battle)
            roster_history.update({(item["profile"], item["char"]): item for item in roster})
            # Future-only reserve profiles may change; don't report them as
            # battle events or rewrite the already captured boundary frame.
            before = capture_state(battle, roster)
            edit = next(edits, None)
        battle.step()
        if any(hasattr(operator, "summoner") and (-operator.deployment_index, operator.char_id) not in roster_history
               for operator in battle.operators):
            roster = export_roster(battle)
            roster_history.update({(item["profile"], item["char"]): item for item in roster})
        after = capture_state(battle, roster)
        events.extend(derive_events(before, after))
        if battle.time + 1e-9 >= next_sample:
            frames.append(snapshot(after))
            next_sample += sample_interval
        before = after
        if battle.end_reason:
            break
        if battle.life_points <= 0:
            battle.end_reason = "life_points_depleted"
            break
        if battle.all_spawns_finished() and battle.all_counted_enemies_finished():
            battle.end_reason = "all_clear"
            break
    if not battle.end_reason:
        battle.end_reason = "timeout"
    if edit is not None:
        raise ValueError("战斗在切入点之前结束，不能继续编排")
    if isinstance(battle.rng, RandomDecisionRng):
        battle.rng.assert_consumed()
    events.extend(battle.operation_log.events)
    events.extend(battle.wave_scheduler.events)
    events.extend(battle.vecbreak.events)
    events.extend(battle.mansfield.events)
    events.extend(battle.cw.events)
    events.extend(battle.dublin.events)
    events.extend(battle.wdex.events)
    for op in battle.operators:
        if op.form_runtime is not None:
            events.extend(make_event(item["time"], "form_change", target=op.deployment_index,
                                     form=item["mode"], event=item["event"], frame=item["frame"],
                                     hp=item["hp"], sp=item["sp"], expiresAt=item["expires_at"])
                          for item in op.form_runtime.events)
    for operation in battle.operation_log.summary(battle.end_reason, battle.time):
        if operation["status"] in ("unfinished", "not_executed"):
            events.append(make_event(
                battle.time, "operation", index=operation["index"], action=operation["action"],
                char=operation["char_id"], status=operation["status"], reason=operation["reason"],
                message=operation["message"], details=operation["details"],
            ))
    events.sort(key=lambda event: event["t"])
    final_state = snapshot(capture_state(battle, roster))
    if not frames or frames[-1]["t"] != final_state["t"]:
        frames.append(final_state)
    case = {
        "label": label,
        "plan": plan_path.name,
        "sampleInterval": sample_interval,
        "spawnTiming": battle.spawn_timing,
        "enemyMuzzle": battle.assumptions.enemy_projectile_muzzle,
        "enemyTurning": battle.assumptions.enemy_facing_transition,
        "routes": export_routes(battle),
        "roster": list(roster_history.values()),
        "squad": [item.char_id for item in battle.squad],
        "difficulty": battle.level.get("selectedDifficulty", "NORMAL"),
        "optionalRunes": battle.level.get("selectedOptionalRunes", []),
        "battleBuffs": battle.level.get("selectedBattleBuffs", []),
        "frames": frames,
        "events": events,
        "result": asdict(battle.result()),
    }
    if isinstance(battle.rng, RandomDecisionRng):
        case["randomDecisions"] = battle.rng.export()
    return case


def promote(staged: Path, output_root: Path) -> Path:
    """Promote complete output and retain the previous generation."""
    output_root = output_root.resolve()
    latest = output_root / "最新模拟"
    history = output_root / "模拟历史"
    for path in (staged, latest, history):
        if path.is_symlink() or not path.resolve().is_relative_to(output_root):
            raise ValueError("output path leaves the selected output directory")
    if not all((staged / name).is_file() for name in ("replay.json", "result.json", "manifest.json")):
        raise ValueError("simulation output is incomplete")
    archived = None
    if latest.exists():
        if not latest.is_dir() or not (latest / "manifest.json").is_file():
            raise ValueError("latest directory is not a generated simulation; refusing to move it")
        marker = json.loads((latest / "manifest.json").read_text(encoding="utf-8"))
        if marker.get("format") != "arksim-replay":
            raise ValueError("latest directory has an unknown generation marker")
        archived = history / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
        if archived.exists():
            raise ValueError("history directory already exists")
        history.mkdir(parents=True, exist_ok=True)
        latest.rename(archived)
    try:
        staged.rename(latest)
    except OSError:
        if archived is not None and not latest.exists():
            archived.rename(latest)
        raise
    return latest


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--stage", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--squad", help="carried operator builds, including undeployed operators")
    parser.add_argument("--difficulty", choices=("NORMAL", "FOUR_STAR", "SIX_STAR"))
    parser.add_argument("--rune", action="append", default=[], help="optional rune ID or level key; repeat for each selection")
    parser.add_argument("--buff", action="append", default=[], help="explicit battle supply ID; repeat up to the stage limit")
    parser.add_argument("--label", default="模拟")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-interval", type=float, default=1 / 30)
    parser.add_argument("--spawn-timing", choices=("fast", "client", "frame_core"), default="fast")
    parser.add_argument("--enemy-attack-timing", choices=("float", "mortar_frames"), default="float")
    parser.add_argument("--strict-behaviors", action="store_true")
    parser.add_argument("--enemy-muzzle", action="store_true", help="use optional enemy muzzle profiles")
    parser.add_argument("--enemy-turning", action="store_true", help="simulate optional enemy facing transitions")
    parser.add_argument("--output-dir", default=str(ROOT / "local"), help="parent of 最新模拟 and 模拟历史")
    parser.add_argument("--editor-project", help="editor project to retain in the replay")
    parser.add_argument("--continuation", help="validated editor continuation context")
    parser.add_argument("--source-replay", help="source replay whose prefix must match before publishing")
    parser.add_argument("--random-trace", help="new local JSON file for semantic random decisions")
    parser.add_argument("--random-tape", help="local decision JSON to replay outcomes strictly")
    parser.add_argument("--module-context", help="local JSON containing game_tags, map_tags and ingots")
    args = parser.parse_args(argv)
    if bool(args.continuation) != bool(args.source_replay):
        parser.error("--continuation and --source-replay must be used together")
    if args.continuation and args.editor_project:
        parser.error("--editor-project cannot be combined with --continuation")
    if args.module_context and (args.continuation or args.editor_project):
        parser.error("条件模组的额外场景暂不支持编辑器方案与回放续编")
    if args.random_tape and (args.continuation or args.editor_project):
        parser.error("指定随机结果复演暂不支持编辑器方案或回放续编")
    if args.random_tape:
        analysis_root = Path(args.output_dir).resolve()
        local_root = (ROOT / "local").resolve()
        if analysis_root == local_root or not analysis_root.is_relative_to(local_root):
            parser.error("指定随机结果复演请用 --output-dir 选择 local/ 下的独立子目录")
    trace_path = None
    tape = None
    for name in ("random_trace", "random_tape"):
        if not getattr(args, name):
            continue
        path = Path(getattr(args, name)).resolve()
        if not path.is_relative_to((ROOT / "local").resolve()):
            parser.error("随机记录与指定结果文件必须保存在 local/ 内")
        if name == "random_trace":
            if path.exists():
                parser.error("随机记录文件已经存在，请使用新文件名")
            trace_path = path
        else:
            try:
                tape = json.loads(path.read_text(encoding="utf-8"))
                RandomDecisionRng(args.seed, tape=tape)
            except (OSError, ValueError) as error:
                parser.error(f"无法读取指定随机结果：{error}")
    if not math.isfinite(args.sample_interval) or args.sample_interval < 1 / 30:
        parser.error("--sample-interval must be finite and at least 1/30 second")
    try:
        shared = build_config(
            args.data_dir, args.stage, args.plan,
            spawn_timing=args.spawn_timing,
            enemy_attack_timing=args.enemy_attack_timing,
            strict_behaviors=args.strict_behaviors,
            enemy_muzzle=args.enemy_muzzle,
            enemy_turning=args.enemy_turning,
            squad_path=args.squad,
            difficulty=args.difficulty,
            optional_runes=args.rune,
            battle_buffs=args.buff,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"cannot load simulation inputs: {error}")
    shared["seed"] = args.seed
    if args.module_context:
        context_path = Path(args.module_context).resolve()
        if not context_path.is_relative_to((ROOT / "local").resolve()):
            parser.error("模组场景文件必须位于 local/ 内")
        try:
            shared["module_context"] = json.loads(context_path.read_text(encoding="utf-8"))
            if not isinstance(shared["module_context"], dict):
                raise ValueError("模组场景必须是对象")
        except (OSError, ValueError) as error:
            parser.error(f"无法读取模组场景：{error}")
    shared["random_trace"] = trace_path is not None
    shared["random_tape"] = tape
    signature = runtime_signature(Path(args.data_dir), resolve_level_file(D.load_stages(), args.stage))
    context, edits = None, []
    source = None
    if args.source_replay:
        source = json.loads(Path(args.source_replay).read_text(encoding="utf-8"))
        if source["cases"][0].get("randomDecisions", {}).get("outcome_replay"):
            raise ValueError("指定随机结果的分析回放暂不支持续编，请先正常运行原方案")
        if source.get("runtimeSignature") and source["runtimeSignature"] != signature:
            raise ValueError("本地运行数据或假设参数已改变，请先重新运行原方案")
        if source.get("map") != export_map(shared["level"]):
            raise ValueError("原回放地图与当前输入不一致")
    if args.continuation:
        from arksim.editor import EditorData
        from arksim.replay_edit import validate_edit
        context = json.loads(Path(args.continuation).read_text(encoding="utf-8"))
        data = EditorData(Path(args.data_dir))
        root_compiled = data.compile(context["project"])
        original = root_compiled["project"]
        root_plan = parse_plan(root_compiled["plan"])
        if root_plan != shared["plan"] or original["stage"] != args.stage:
            raise ValueError("续编原方案与运行输入不一致")
        squad = [{k: v for k, v in asdict(item).items() if k in BUILD_FIELDS} for item in shared["squad"] or []]
        if squad != original["squad"]:
            raise ValueError("续编原编队与运行输入不一致")
        actual_options = {"seed": args.seed, "spawn_timing": args.spawn_timing,
                          "enemy_attack_timing": args.enemy_attack_timing, "enemy_muzzle": args.enemy_muzzle,
                          "enemy_turning": args.enemy_turning, "optional_runes": args.rune, "battle_buffs": args.buff}
        if actual_options != original["options"] or not context["edits"]:
            raise ValueError("续编选项或记录与运行输入不一致")
        previous_frame = -1
        for change in context["edits"]:
            compiled = data.compile(change["project"])
            frame = change["frame"]
            if type(frame) is not int or frame <= previous_frame:
                raise ValueError("续编切入帧必须递增")
            validate_edit(original, compiled["project"], frame)
            edits.append((frame, parse_plan(compiled["plan"])))
            original, previous_frame = compiled["project"], frame
    elif args.editor_project:
        context = {"project": json.loads(Path(args.editor_project).read_text(encoding="utf-8")), "edits": []}
    case = run_case(args.label, Path(args.plan), shared, args.sample_interval, edits)
    if args.source_replay:
        verify_prefix(source["cases"][0], case, edits[-1][0])
        # Keep the source's sampled prefix byte-for-byte at JSON value level.
        boundary = edits[-1][0]
        case["frames"] = [f for f in source["cases"][0]["frames"] if frame_number(f) <= boundary] + [
            f for f in case["frames"] if frame_number(f) > boundary]
    created = datetime.now(timezone.utc).isoformat()
    payload = {
        "format": "arksim-replay", "schemaVersion": 1,
        "engineVersion": __version__, "createdAt": created,
        "runtimeSignature": signature,
        "stage": args.stage, "spawnTiming": args.spawn_timing,
        "difficulty": shared["level"].get("selectedDifficulty", "NORMAL"),
        "optionalRunes": shared["level"].get("selectedOptionalRunes", []),
        "battleBuffs": shared["level"].get("selectedBattleBuffs", []),
        "enemyAttackTiming": args.enemy_attack_timing,
        "enemyMuzzle": args.enemy_muzzle, "enemyTurning": args.enemy_turning,
        "map": export_map(shared["level"]), "cases": [case],
    }
    if context is not None:
        payload["editorContext"] = context
    if shared.get("module_context") is not None:
        payload["moduleContext"] = shared["module_context"]
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("x", encoding="utf-8") as stream:
            json.dump(case["randomDecisions"], stream, ensure_ascii=False, indent=2, allow_nan=False)
    output_root = Path(args.output_dir).resolve()
    # All moving happens within this resolved root, with no symlink escapes.
    history = output_root / "模拟历史"
    if history.is_symlink() or not history.resolve().is_relative_to(output_root):
        parser.error("history directory leaves the output directory")
    history.mkdir(parents=True, exist_ok=True)
    staged = history / datetime.now(timezone.utc).strftime("_生成中_%Y%m%d_%H%M%S_%fZ")
    staged.mkdir()
    (staged / "replay.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    (staged / "result.json").write_text(json.dumps(case["result"], ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "format": "arksim-replay", "schemaVersion": 1,
        "createdAt": created, "engineVersion": __version__,
        "stage": args.stage, "plan": Path(args.plan).name, "seed": args.seed,
        "difficulty": shared["level"].get("selectedDifficulty", "NORMAL"),
        "optionalRunes": shared["level"].get("selectedOptionalRunes", []),
        "battleBuffs": shared["level"].get("selectedBattleBuffs", []),
        "squad": Path(args.squad).name if args.squad else None,
        "sampleInterval": args.sample_interval,
        "spawnTiming": args.spawn_timing, "enemyAttackTiming": args.enemy_attack_timing,
        "enemyMuzzle": args.enemy_muzzle, "enemyTurning": args.enemy_turning,
        "frames": len(case["frames"]), "result": case["result"],
    }
    if shared.get("module_context") is not None:
        manifest["moduleContext"] = shared["module_context"]
    (staged / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = promote(staged, output_root)
    print(f"Replay: {latest / 'replay.json'}")
    print(f"Viewer: {ROOT / 'viewer/index.html'}")
    print(f"win={case['result']['win']} killed={case['result']['enemies_killed']} leaked={case['result']['enemies_leaked']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

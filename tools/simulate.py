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

def compact_enemy(enemy: Any, battle: Battle) -> dict[str, Any]:
    route_index = next(
        (
            index
            for index, route in enumerate(battle.routes)
            if route == enemy.route
        ),
        -1,
    )
    return {
        "id": enemy.enemy_index,
        "key": enemy.key,
        "name": enemy.name,
        "levelType": enemy.level_type,
        "r": round(enemy.position_row, 4),
        "c": round(enemy.position_col, 4),
        "hp": round(enemy.hp, 2),
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


def compact_operator(operator: Any, battle: Battle) -> dict[str, Any]:
    periodic = (
        battle.assumptions.headb2_periodic_sp_recovery
        and operator.skill_prefab_id == "skchr_headb2_2"
        and math.isclose(battle.dt, 1 / CLIENT_LOGIC_RATE, abs_tol=1e-12)
        and operator.sp_increment == 1.0
    )
    progress = (
        1.0 - operator.sp_recovery_remaining_raw / CLIENT_FP_SCALE
        if periodic else operator.sp % 1.0
        if operator.sp_type == "INCREASE_WITH_TIME" else None
    )
    return {
        "id": operator.deployment_index,
        "char": operator.char_id,
        "name": operator.name,
        "r": operator.tile[0],
        "c": operator.tile[1],
        "hp": round(operator.hp, 2),
        "max": round(operator.max_hp, 2),
        "sp": round(operator.sp, 2),
        "spMax": round(operator.sp_max, 2),
        "spSlot": round(max(0.0, min(1.0, progress)), 6) if progress is not None else None,
        "spSlotMode": "periodic" if periodic else "continuous" if progress is not None else "none",
        "spSlotPaused": (
            operator.dead or operator.skill_active
            or operator.skill_startup_until >= 0.0
            or operator.statuses.has(Status.SP_BLOCKED)
            or operator.sp >= operator.sp_max
        ),
        "attributes": {
            "atk": round(operator.atk * operator.atk_scale, 2),
            "def": round(operator.defense * operator.def_scale, 2),
            "res": round(battle._effective_resistance(operator), 2),
            "attackInterval": round(battle._effective_attack_interval(operator), 4),
            "blockCapacity": operator.block_count if operator.statuses.can_block else 0,
        },
        "starting": operator.skill_startup_until >= 0.0,
        "active": operator.skill_active or operator.skill_end_reason == "PERMANENT",
        "casts": operator.skill_cast_count,
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
        "r": round(projectile.position_row, 4),
        "c": round(projectile.position_col, 4),
        "arc": round(4.0 * progress * (1.0 - progress), 4)
        if projectile.parabolic
        else 0.0,
    }


def export_roster(battle: Battle) -> list[dict[str, Any]]:
    """Keep plan profiles separate so redeployments can use different builds."""
    roster = []
    for index, plan in enumerate(battle.plan):
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
            "name": character.get("name", plan.char_id),
            "baseCost": int(attrs.get("cost", 0)),
            "attributes": {
                "maxHp": round(float(attrs.get("maxHp", 0)), 2),
                "atk": round(float(attrs.get("atk", 0)), 2),
                "def": round(float(attrs.get("def", 0)), 2),
                "res": round(float(attrs.get("magicResistance", 0)), 2),
                "attackInterval": round(float(attrs.get("baseAttackTime", 1.6)) * 100 / (float(attrs.get("attackSpeed", 100)) or 100), 4),
                "blockCapacity": int(attrs.get("blockCnt", 1)),
            },
        })
    return roster


def reserve_state(battle: Battle, roster: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profiles = {}
    for profile in roster:
        previous = profiles.get(profile["char"])
        if previous is None or previous["profile"] < battle.plan_index:
            profiles[profile["char"]] = profile
    active = {op.char_id for op in battle.operators if not op.dead}
    cards = []
    for char_id, profile in profiles.items():
        if char_id in active:
            continue
        remaining = max(0.0, battle.redeploy_ready_at.get(char_id, 0.0) - battle.time)
        penalty = battle.redeploy_penalty.get(char_id, 0)
        cost = min(math.floor(profile["baseCost"] * (1.0 + .5 * penalty)), battle.max_cost)
        status = "cooldown" if remaining > 1e-9 else "cost" if battle.dp < cost else "ready"
        cards.append({"profile": profile["profile"], "char": char_id, "cost": cost, "status": status, "remaining": round(remaining, 3)})
    return cards


def capture_state(battle: Battle, roster: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    dp_progress = min(1.0, max(0.0, battle.dp_timer / battle.cost_increase_time)) if battle.cost_increase_time > 0 else 0.0
    return {
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
            for projectile in battle.projectiles
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
            events.append(make_event(time, "skill_end", target=operator_id, cast=new["casts"], r=new["r"], c=new["c"]))
        if not old["dead"] and new["dead"]:
            events.append(make_event(time, "operator_death", target=operator_id, r=new["r"], c=new["c"]))

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
    )
    frames: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    roster = export_roster(battle)
    before = capture_state(battle, roster)
    frames.append(snapshot(before))
    next_sample = sample_interval
    while battle.time < 600:
        battle.step()
        after = capture_state(battle, roster)
        events.extend(derive_events(before, after))
        if battle.time + 1e-9 >= next_sample:
            frames.append(snapshot(after))
            next_sample += sample_interval
        before = after
        if battle.life_points <= 0:
            battle.end_reason = "life_points_depleted"
            break
        if battle.spawn_index >= len(battle.spawn_events) and not battle.enemies:
            battle.end_reason = "all_clear"
            break
    if not battle.end_reason:
        battle.end_reason = "timeout"
    final_state = snapshot(capture_state(battle, roster))
    if not frames or frames[-1]["t"] != final_state["t"]:
        frames.append(final_state)
    return {
        "label": label,
        "plan": plan_path.name,
        "sampleInterval": sample_interval,
        "spawnTiming": battle.spawn_timing,
        "routes": export_routes(battle),
        "roster": roster,
        "frames": frames,
        "events": events,
        "result": asdict(battle.result()),
    }


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
    parser.add_argument("--label", default="模拟")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-interval", type=float, default=1 / 30)
    parser.add_argument("--spawn-timing", choices=("fast", "client", "frame_core"), default="fast")
    parser.add_argument("--enemy-attack-timing", choices=("float", "mortar_frames"), default="float")
    parser.add_argument("--strict-behaviors", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "local"), help="parent of 最新模拟 and 模拟历史")
    args = parser.parse_args(argv)
    if not math.isfinite(args.sample_interval) or args.sample_interval < 1 / 30:
        parser.error("--sample-interval must be finite and at least 1/30 second")
    try:
        shared = build_config(
            args.data_dir, args.stage, args.plan,
            spawn_timing=args.spawn_timing,
            enemy_attack_timing=args.enemy_attack_timing,
            strict_behaviors=args.strict_behaviors,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"cannot load simulation inputs: {error}")
    shared["seed"] = args.seed
    case = run_case(args.label, Path(args.plan), shared, args.sample_interval)
    created = datetime.now(timezone.utc).isoformat()
    payload = {
        "format": "arksim-replay", "schemaVersion": 1,
        "engineVersion": __version__, "createdAt": created,
        "stage": args.stage, "spawnTiming": args.spawn_timing,
        "enemyAttackTiming": args.enemy_attack_timing,
        "map": export_map(shared["level"]), "cases": [case],
    }
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
        "sampleInterval": args.sample_interval,
        "spawnTiming": args.spawn_timing, "enemyAttackTiming": args.enemy_attack_timing,
        "frames": len(case["frames"]), "result": case["result"],
    }
    (staged / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = promote(staged, output_root)
    print(f"Replay: {latest / 'replay.json'}")
    print(f"Viewer: {ROOT / 'viewer/index.html'}")
    print(f"win={case['result']['win']} killed={case['result']['enemies_killed']} leaked={case['result']['enemies_leaked']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

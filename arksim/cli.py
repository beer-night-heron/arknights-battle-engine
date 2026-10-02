"""Command-line entry point for running a strategy simulation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
import json
import math
from pathlib import Path
import sys
from typing import Any

from . import data as D
from .batch import summarize_results
from .battle import (
    ENEMY_ATTACK_TIMING_FLOAT,
    ENEMY_ATTACK_TIMING_MORTAR_FRAMES,
    SPAWN_TIMING_CLIENT,
    SPAWN_TIMING_FAST,
    SPAWN_TIMING_FRAME_CORE,
    Battle,
    OperatorPlan,
)


_WORKER_CONFIG: dict[str, Any] | None = None
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "local" / "data"


def build_config(
    data_dir: str | Path,
    stage: str,
    plan_path: str | Path,
    *,
    strict_behaviors: bool = False,
    spawn_timing: str = SPAWN_TIMING_FAST,
    enemy_attack_timing: str = ENEMY_ATTACK_TIMING_FLOAT,
) -> dict[str, Any]:
    """Shared input loading for summary and replay commands."""
    D.set_data_dir(data_dir)
    plan = load_plan(str(plan_path))
    characters = D.load_characters()
    skills = D.load_skills()
    equips = D.load_battle_equips()
    for index, item in enumerate(plan, 1):
        if item.char_id not in characters:
            raise ValueError(f"plan item {index}: unknown char_id {item.char_id!r}")
        if item.skill_id and item.skill_id not in skills:
            raise ValueError(f"plan item {index}: unknown skill_id {item.skill_id!r}")
        if item.module_id and item.module_id not in equips:
            raise ValueError(f"plan item {index}: unknown module_id {item.module_id!r}")
    return {
        "level": D.load_level(resolve_level_file(D.load_stages(), stage)),
        "enemy_index": D.build_enemy_index(D.load_enemy_database()),
        "characters": characters,
        "ranges": D.load_ranges(),
        "plan": plan,
        "skills": skills,
        "attack_timing": D.load_attack_timing(),
        "behavior_templates": D.load_behavior_templates(),
        "battle_equips": equips,
        "projectile_data": D.load_projectile_data(),
        "buff_abilities": D.load_enemy_buff_abilities(),
        "strict_behaviors": strict_behaviors,
        "spawn_timing": spawn_timing,
        "enemy_attack_timing": enemy_attack_timing,
    }


def resolve_level_file(stages: dict[str, Any], stage_id: str) -> str:
    stage = stages["stages"].get(stage_id)
    if not stage:
        raise KeyError(f"unknown stage id: {stage_id}")
    level_id = stage.get("levelId") or stage_id
    level_id = level_id.replace("\\", "/")
    if "/" in level_id:
        level_id = level_id.rsplit("/", 1)[-1]
    if level_id.endswith(".json"):
        return level_id
    return f"{level_id}.json"


def _run_once(config: dict[str, Any], seed: int):
    battle = Battle(
        config["level"],
        config["enemy_index"],
        config["characters"],
        config["ranges"],
        config["plan"],
        config["skills"],
        config["attack_timing"],
        config["behavior_templates"],
        config["battle_equips"],
        config.get("projectile_data"),
        config.get("buff_abilities"),
        strict_behaviors=config["strict_behaviors"],
        seed=seed,
        spawn_timing=config.get("spawn_timing", SPAWN_TIMING_FAST),
        enemy_attack_timing=config.get(
            "enemy_attack_timing", ENEMY_ATTACK_TIMING_FLOAT
        ),
    )
    return battle.run()


def _initialize_worker(config: dict[str, Any]) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = config


def _run_worker(seed: int):
    if _WORKER_CONFIG is None:
        raise RuntimeError("batch worker was not initialized")
    return _run_once(_WORKER_CONFIG, seed)


@dataclass(frozen=True)
class BatchExecution:
    results: list[Any]
    executed_runs: int
    deterministic_reuse: bool = False

    @property
    def reused_runs(self) -> int:
        return len(self.results) - self.executed_runs


def run_seed_batch_detailed(
    config: dict[str, Any], seeds: list[int], *, workers: int = 1
) -> BatchExecution:
    """Run a batch and reuse a proven deterministic first result."""
    if not seeds:
        return BatchExecution([], 0)
    first = _run_once(config, seeds[0])
    if first.random_draws == 0:
        results = [first]
        results.extend(replace(first, seed=seed) for seed in seeds[1:])
        return BatchExecution(results, 1, deterministic_reuse=True)

    remaining = seeds[1:]
    if workers <= 1:
        rest = [_run_once(config, seed) for seed in remaining]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(config,),
        ) as executor:
            rest = list(executor.map(_run_worker, remaining, chunksize=1))
    return BatchExecution([first, *rest], len(seeds))


def run_seed_batch(
    config: dict[str, Any], seeds: list[int], *, workers: int = 1
) -> list[Any]:
    """Run an ordered seed batch, optionally in independent processes."""
    return run_seed_batch_detailed(config, seeds, workers=workers).results


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--stage", required=True, help="stage ID from stage_table.json")
    parser.add_argument("--plan", required=True, help="path to a plan JSON file")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workers", type=int, default=1, help="parallel worker processes"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.95, help="Wilson interval level"
    )
    parser.add_argument("--json-out", action="store_true", help="print JSON")
    parser.add_argument("--output", help="save result JSON as UTF-8 (not a replay file)")
    parser.add_argument(
        "--strict-behaviors",
        action="store_true",
        help="fail on the first unsupported behavior-tree node",
    )
    parser.add_argument(
        "--spawn-timing",
        choices=(SPAWN_TIMING_FAST, SPAWN_TIMING_CLIENT, SPAWN_TIMING_FRAME_CORE),
        default=SPAWN_TIMING_FAST,
        help="fast nominal, calibrated client, or SpawnCore static frame schedule",
    )
    parser.add_argument(
        "--enemy-attack-timing",
        choices=(ENEMY_ATTACK_TIMING_FLOAT, ENEMY_ATTACK_TIMING_MORTAR_FRAMES),
        default=ENEMY_ATTACK_TIMING_FLOAT,
        help="float enemy cooldowns or optional integer-frame mortar cooldowns",
    )
    args = parser.parse_args(argv)
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be between zero and one")

    try:
        config = build_config(
            args.data_dir, args.stage, args.plan,
            strict_behaviors=args.strict_behaviors,
            spawn_timing=args.spawn_timing,
            enemy_attack_timing=args.enemy_attack_timing,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"cannot load simulation inputs: {error}")
    seeds = [args.seed + index for index in range(args.runs)]
    execution = run_seed_batch_detailed(config, seeds, workers=args.workers)
    results = execution.results
    summary = summarize_results(results, confidence=args.confidence)

    if args.json_out or args.output:
        serialized = json.dumps(
            {
                "results": [asdict(result) for result in results],
                "summary": summary.to_dict(),
                "execution": {
                    "executed_runs": execution.executed_runs,
                    "reused_runs": execution.reused_runs,
                    "deterministic_reuse": execution.deterministic_reuse,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(serialized + "\n", encoding="utf-8")
        if args.json_out:
            print(serialized)
            return 0

    for r in results if args.runs <= 20 else []:
        print(
            f"seed={r.seed} draws={r.random_draws} "
            f"win={r.win} reason={r.reason} time={r.time:.1f}s "
            f"life={r.life_points}/{r.max_life_points} "
            f"spawned={r.enemies_spawned} killed={r.enemies_killed} "
            f"leaked={r.enemies_leaked} alive={r.operators_deployed} "
            f"behavior_warnings={r.behavior_warnings} "
            f"mechanic_warnings={r.mechanic_warnings} "
            f"spawn_timing={r.spawn_timing} "
            f"enemy_attack_timing={r.enemy_attack_timing}"
        )
    if args.runs > 1:
        if execution.deterministic_reuse:
            print(
                f"runs={summary.runs} deterministic=True "
                f"wins={summary.wins} outcome_win={summary.win_rate:.0%} "
                f"time={summary.mean_time:.1f}s "
                f"behavior_warning_runs={summary.behavior_warning_runs} "
                f"mechanic_warning_runs={summary.mechanic_warning_runs}"
            )
        else:
            level = summary.confidence * 100.0
            print(
                f"runs={summary.runs} wins={summary.wins} "
                f"win_rate={summary.win_rate:.2%} "
                f"wilson_{level:g}%="
                f"[{summary.win_rate_low:.2%},{summary.win_rate_high:.2%}] "
                f"time_mean={summary.mean_time:.1f}s "
                f"time_p50={summary.p50_time:.1f}s "
                f"time_p95={summary.p95_time:.1f}s "
                f"behavior_warning_runs={summary.behavior_warning_runs} "
                f"mechanic_warning_runs={summary.mechanic_warning_runs}"
            )
        print(
            f"executed_runs={execution.executed_runs} "
            f"reused_deterministic_runs={execution.reused_runs}"
        )
    return 0


def load_plan(path: str) -> list[OperatorPlan]:
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError("plan must be a JSON array")
    allowed = {item.name for item in fields(OperatorPlan)}
    plans = []
    for index, item in enumerate(raw, 1):
        prefix = f"plan item {index}"
        if not isinstance(item, dict):
            raise ValueError(f"{prefix}: must be an object")
        unknown = set(item) - allowed
        if unknown:
            raise ValueError(f"{prefix}: unknown fields {sorted(unknown)}")
        for key in ("char_id", "tile", "time"):
            if key not in item:
                raise ValueError(f"{prefix}: missing {key}")
        if not isinstance(item["char_id"], str) or not item["char_id"]:
            raise ValueError(f"{prefix}: char_id must be a nonempty string")
        action = item.get("action", "DEPLOY")
        if not isinstance(action, str) or action.upper() not in ("DEPLOY", "RETREAT"):
            raise ValueError(f"{prefix}: action must be DEPLOY or RETREAT")
        tile = item["tile"]
        if tile is not None and (
            not isinstance(tile, (list, tuple)) or len(tile) != 2
            or any(type(value) is not int or value < 0 for value in tile)
        ):
            raise ValueError(f"{prefix}: tile must be [nonnegative row, column] or null")
        if action.upper() == "DEPLOY" and tile is None:
            raise ValueError(f"{prefix}: DEPLOY requires tile")
        for key in ("time", "trust"):
            value = item.get(key, 0)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{prefix}: {key} must be finite and nonnegative")
        for key, minimum in (("level", 1), ("potential_rank", 0), ("module_level", 0)):
            value = item.get(key, minimum)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{prefix}: {key} must be an integer >= {minimum}")
        if item.get("elite") is not None and (type(item["elite"]) is not int or item["elite"] not in (0, 1, 2)):
            raise ValueError(f"{prefix}: elite must be 0, 1, 2 or null")
        if type(item.get("facing", 0)) is not int or item.get("facing", 0) not in range(4):
            raise ValueError(f"{prefix}: facing must be 0, 1, 2 or 3")
        for key in ("skill_id", "module_id"):
            if item.get(key) is not None and not isinstance(item[key], str):
                raise ValueError(f"{prefix}: {key} must be a string or null")
        plans.append(OperatorPlan(**item))
    return plans


if __name__ == "__main__":
    raise SystemExit(main())

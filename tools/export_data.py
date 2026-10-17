"""Extract the JSON tables/levels we need from a local ArknightsGameData clone.

The ArknightsGameData repo is cloned with `--filter=blob:none --no-checkout`,
so files are not present on disk. We use `git show <ref>:<path>` to materialize
only the blobs the simulator needs, without changing the clone's checkout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


DEFAULT_OUT = Path(__file__).resolve().parents[1] / "local" / "data"

BEHAVIOR_TABLE = "zh_CN/gamedata/battle/buff_template_data.json"
ELEMENT_BREAK_TABLE = "zh_CN/gamedata/battle/ep_breakbuff_table.json"
BUFF_DATABASE_TABLE = "zh_CN/gamedata/buff_table.json"
ACTIVITY_TABLE = "zh_CN/gamedata/excel/activity_table.json"

TABLES = [
    "zh_CN/gamedata/excel/character_table.json",
    "zh_CN/gamedata/excel/skill_table.json",
    "zh_CN/gamedata/excel/range_table.json",
    "zh_CN/gamedata/excel/stage_table.json",
    "zh_CN/gamedata/excel/battle_equip_table.json",
    "zh_CN/gamedata/excel/uniequip_table.json",
    "zh_CN/gamedata/excel/favor_table.json",
    "zh_CN/gamedata/levels/enemydata/enemy_database.json",
    "zh_CN/gamedata/levels/levels_meta.json",
    BEHAVIOR_TABLE,
    ELEMENT_BREAK_TABLE,
    BUFF_DATABASE_TABLE,
]

def git_show(repo: Path, rel_path: str, ref: str = "HEAD") -> bytes:
    proc = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo.resolve().as_posix()}",
            "-C",
            str(repo),
            "show",
            f"{ref}:{rel_path}",
        ],
        check=True,
        capture_output=True,
    )
    return proc.stdout


def battle_supply_data(activity: dict, revision: str) -> dict:
    """Convert the same revision's activity modes to the runtime schema."""
    modes = activity.get("activity", {}).get("VEC_BREAK_V2", {})
    if not modes:
        raise ValueError("selected activity table has no VEC_BREAK_V2 modes")
    return {"sourceRevision": revision, "activities": modes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="path to the local data Git repository")
    parser.add_argument("--ref", default="HEAD", help="local data revision to export; default HEAD")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--level",
        action="append",
        default=[],
        help="tree path of a level json to export (repeatable)",
    )
    parser.add_argument(
        "--only-behaviors",
        action="store_true",
        help="export only buff_template_data.json without replacing other tables",
    )
    parser.add_argument(
        "--only-levels",
        action="store_true",
        help="export only the level files supplied by --level",
    )
    parser.add_argument("--only-battle-supplies", action="store_true",
                        help="export only the activity battle supplies as vecbreak_data.json")
    args = parser.parse_args()
    if sum((args.only_behaviors, args.only_levels, args.only_battle_supplies)) > 1:
        parser.error("the --only-* export modes are mutually exclusive")
    if args.only_battle_supplies and args.level:
        parser.error("--only-battle-supplies does not export levels")
    if args.only_levels and not args.level:
        parser.error("--only-levels requires at least one --level")

    repo = Path(args.repo)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tables = (
        []
        if args.only_levels or args.only_battle_supplies
        else ([BEHAVIOR_TABLE] if args.only_behaviors else TABLES)
    )
    for rel in tables:
        data = git_show(repo, rel, args.ref)
        dest = out / Path(rel).name
        dest.write_bytes(data)
        print(f"exported {dest.name} ({len(data)} bytes)")

    levels = [] if args.only_behaviors or args.only_battle_supplies else args.level
    for rel in levels:
        data = git_show(repo, rel, args.ref)
        dest = out / Path(rel).name
        dest.write_bytes(data)
        print(f"exported {dest.name} ({len(data)} bytes)")

    if args.only_battle_supplies or not (args.only_levels or args.only_behaviors):
        activity = json.loads(git_show(repo, ACTIVITY_TABLE, args.ref))
        dest = out / "vecbreak_data.json"
        dest.write_text(json.dumps(battle_supply_data(activity, args.ref), ensure_ascii=False,
                                   indent=2) + "\n", encoding="utf-8")
        print(f"exported {dest.name}")


if __name__ == "__main__":
    main()

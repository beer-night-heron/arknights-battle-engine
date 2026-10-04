"""Extract the JSON tables/levels we need from a local ArknightsGameData clone.

The ArknightsGameData repo is cloned with `--filter=blob:none --no-checkout`,
so files are not present on disk.  We use `git show HEAD:<path>` to materialize
only the blobs the simulator needs.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_OUT = Path(__file__).resolve().parents[1] / "local" / "data"

BEHAVIOR_TABLE = "zh_CN/gamedata/battle/buff_template_data.json"

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
]

def git_show(repo: Path, rel_path: str) -> bytes:
    proc = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo.resolve().as_posix()}",
            "-C",
            str(repo),
            "show",
            f"HEAD:{rel_path}",
        ],
        check=True,
        capture_output=True,
    )
    return proc.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="path to the local data Git repository")
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
    args = parser.parse_args()
    if args.only_behaviors and args.only_levels:
        parser.error("--only-behaviors and --only-levels are mutually exclusive")
    if args.only_levels and not args.level:
        parser.error("--only-levels requires at least one --level")

    repo = Path(args.repo)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tables = (
        []
        if args.only_levels
        else ([BEHAVIOR_TABLE] if args.only_behaviors else TABLES)
    )
    for rel in tables:
        data = git_show(repo, rel)
        dest = out / Path(rel).name
        dest.write_bytes(data)
        print(f"exported {dest.name} ({len(data)} bytes)")

    levels = [] if args.only_behaviors else args.level
    for rel in levels:
        data = git_show(repo, rel)
        dest = out / Path(rel).name
        dest.write_bytes(data)
        print(f"exported {dest.name} ({len(data)} bytes)")


if __name__ == "__main__":
    main()

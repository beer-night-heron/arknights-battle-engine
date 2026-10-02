"""Look up stage/operator IDs or print a map using local runtime data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arksim import data as D
from arksim.cli import DEFAULT_DATA_DIR, resolve_level_file


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--kind", choices=("stages", "operators", "map"), required=True)
    parser.add_argument("--query", default="", help="filter IDs or names")
    parser.add_argument("--stage", help="required for map")
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    D.set_data_dir(args.data_dir)
    try:
        if args.kind == "map":
            if not args.stage:
                parser.error("--kind map requires --stage")
            from tools.simulate import export_map
            print(json.dumps(export_map(D.load_level(resolve_level_file(D.load_stages(), args.stage))), ensure_ascii=False, indent=2))
            return 0
        if args.kind == "stages":
            stages = D.load_stages()
            rows = [
                {"id": key, "code": value.get("code"), "name": value.get("name"), "levelFile": resolve_level_file(stages, key)}
                for key, value in stages["stages"].items()
                if value.get("levelId") and (Path(args.data_dir) / resolve_level_file(stages, key)).is_file()
            ]
        else:
            rows = [
                {"id": key, "name": value.get("name"), "position": value.get("position"),
                 "skills": [item.get("skillId") for item in value.get("skills", [])]}
                for key, value in D.load_characters().items()
                if key.startswith("char_")
            ]
        rows = [row for row in rows if args.query.casefold() in json.dumps(row, ensure_ascii=False).casefold()]
        print(json.dumps({"matched": len(rows), "shown": min(len(rows), args.limit), "items": rows[:args.limit]}, ensure_ascii=False, indent=2))
    except (OSError, KeyError, ValueError) as error:
        parser.error(f"cannot read local data: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

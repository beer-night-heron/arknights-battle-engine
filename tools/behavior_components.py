"""Compile a local serialized ability export into a runtime directory."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arksim.behavior_components import compile_resources, compile_skill_resources, compile_operator_resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--kind", choices=("module", "skill", "operator"), default="module")
    parser.add_argument("--modules", type=Path, default=Path("local/data/battle_equip_table.json"))
    parser.add_argument("--skills", type=Path, default=Path("local/data/skill_table.json"))
    parser.add_argument("--characters", type=Path, default=Path("local/data/character_table.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.kind == "module":
        modules = json.loads(args.modules.read_text(encoding="utf-8"))
        keys = {p["resKey"] for module in modules.values() for phase in module.get("phases", [])
                for p in phase.get("parts", []) if p.get("resKey")}
        compiler = compile_resources
    elif args.kind == "skill":
        skills = json.loads(args.skills.read_text(encoding="utf-8"))
        keys = {level["prefabId"] for skill in skills.values() for level in skill.get("levels", [])
                if level.get("prefabId")}
        compiler = compile_skill_resources
    else:
        keys = set(json.loads(args.characters.read_text(encoding="utf-8")))
        compiler = compile_operator_resources
    args.output = args.output or Path("local/data/" + (
        {"module": "behavior_components.json", "skill": "skill_behavior_components.json",
         "operator": "operator_behavior_components_v2.json"}[args.kind]))
    directory = compiler(json.loads(args.source.read_text(encoding="utf-8")), keys)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A new path is required; preserve previously compiled evidence/data.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(directory, stream, ensure_ascii=False, separators=(",", ":"))
    print(f"Compiled {len(directory['resources'])} {args.kind} resources to {args.output}")


if __name__ == "__main__":
    main()

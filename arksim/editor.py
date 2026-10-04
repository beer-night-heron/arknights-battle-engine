"""Local editor catalog and compilation into ordinary battle plans."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import data as D
from .cli import parse_plan, resolve_level_file
from .loadout import resolve_plan
from .skill_controls import skill_controls

BUILD_FIELDS = {"char_id", "elite", "level", "trust", "potential_rank", "skill_id",
                "skill_level", "module_id", "module_level", "auto_skill"}
OP_FIELDS = {"char_id", "time", "action", "tile", "facing", "on_failure", "mode"}
DEFAULT_OPTIONS = {"spawn_timing": "fast", "enemy_attack_timing": "float", "seed": 0,
                   "enemy_muzzle": False, "enemy_turning": False}


class EditorData:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir.resolve()
        D.set_data_dir(self.data_dir)
        self.characters = D.load_characters()
        self.skills = D.load_skills()
        self.modules = D.load_battle_equips()
        self.module_index = D.load_module_index()
        self.favor = D.load_favor_table()
        self.stages = D.load_stages()

    def catalog(self) -> dict:
        stages = []
        for key, value in self.stages.get("stages", {}).items():
            if value.get("levelId") and (self.data_dir / resolve_level_file(self.stages, key)).is_file():
                stages.append({"id": key, "code": value.get("code") or key,
                               "name": value.get("name") or key})
        operators = [{"id": k, "name": v.get("name", k), "position": v.get("position"),
                      "profession": v.get("profession"), "rarity": v.get("rarity")}
                     for k, v in self.characters.items() if k.startswith("char_")]
        return {"stages": stages, "operators": operators}

    def stage(self, key: str) -> dict:
        from tools.simulate import export_map
        filename = resolve_level_file(self.stages, key)
        level = D.load_level(filename)
        return {"id": key, "map": export_map(level), "options": level.get("options", {})}

    def operator(self, key: str) -> dict:
        char = self.characters.get(key)
        if char is None or not key.startswith("char_"):
            raise ValueError("找不到该干员")
        skills = []
        for entry in char.get("skills", []):
            sid = entry.get("skillId")
            levels = self.skills.get(sid, {}).get("levels", [])
            if not levels:
                continue
            ranks = []
            normal = char.get("allSkillLvlup", [])
            for rank, lv in enumerate(levels, 1):
                conditions = [c.get("unlockCond", {}) for c in normal[:min(rank - 1, len(normal))]]
                if rank > len(normal) + 1:
                    mastery = entry.get("levelUpCostCond", [])
                    count = rank - len(normal) - 1
                    if count > len(mastery):
                        continue
                    conditions += [c.get("unlockCond", {}) for c in mastery[:count]]
                ranks.append({"rank": rank, "conditions": conditions,
                              "type": lv.get("skillType"), "sp": lv.get("spData", {}).get("spCost", 0),
                              "controls": asdict(skill_controls(lv))})
            skills.append({"id": sid, "name": levels[0].get("name", sid),
                           "unlock": entry.get("unlockCond", {}), "ranks": ranks})
        modules = []
        for mid, meta in self.module_index.get("equipDict", {}).items():
            if meta.get("charId") != key or mid not in self.modules:
                continue
            trust = {}
            for grade, raw in (meta.get("unlockFavors") or {}).items():
                points = [f["data"]["percent"] for f in self.favor.get("favorFrames", [])
                          if f["data"]["favorPoint"] >= raw]
                trust[grade] = min(points) if raw and points else 0 if not raw else None
            modules.append({"id": mid, "name": meta.get("uniEquipName") or mid,
                            "levels": [p.get("equipLevel") for p in self.modules[mid].get("phases", [])],
                            "unlock": {"phase": meta.get("unlockEvolvePhase", 0),
                                       "level": meta.get("unlockLevel", 1)}, "trust": trust})
        return {"id": key, "name": char.get("name", key), "position": char.get("position"),
                "maxLevels": [p.get("maxLevel", 1) for p in char.get("phases", [])],
                "maxPotential": char.get("maxPotentialLevel", 0), "skills": skills, "modules": modules}

    def build(self, raw: dict) -> tuple[dict, dict]:
        if not isinstance(raw, dict) or set(raw) - BUILD_FIELDS:
            raise ValueError("干员配置格式错误或含有未知字段")
        plan = parse_plan([{**raw, "tile": [0, 0], "time": 0}])[0]
        resolved = resolve_plan(plan, self.characters, self.skills, self.modules, self.module_index, self.favor)
        config = {k: v for k, v in asdict(resolved).items() if k in BUILD_FIELDS}
        char = self.characters[resolved.char_id]
        attrs = D.char_attributes(char, resolved.level, resolved.elite, trust=resolved.trust,
                                  potential_rank=resolved.potential_rank,
                                  module=self.modules.get(resolved.module_id), module_level=resolved.module_level)
        return config, {"char_id": resolved.char_id, "name": char.get("name"), "attributes": attrs}

    def compile(self, project: dict) -> dict:
        if (not isinstance(project, dict) or project.get("format") != "arksim-plan-editor"
                or type(project.get("version")) is not int or project["version"] != 1):
            raise ValueError("不是受支持的编辑方案（arksim-plan-editor 版本 1）")
        if set(project) - {"format", "version", "stage", "squad", "operations", "options"}:
            raise ValueError("编辑方案含有未知字段")
        stage_id = project.get("stage")
        if not isinstance(stage_id, str) or not stage_id:
            raise ValueError("请先选择关卡")
        stage = self.stage(stage_id)
        squad, summaries = [], []
        if not isinstance(project.get("squad"), list) or not project["squad"]:
            raise ValueError("请先添加至少一名携带干员")
        if len(project["squad"]) > 100:
            raise ValueError("队伍配置过多")
        builds = {}
        for raw in project["squad"]:
            config, summary = self.build(raw)
            key = config["char_id"]
            if key in builds:
                raise ValueError("队伍不能重复携带同一干员")
            builds[key] = config
            squad.append(config)
            summaries.append(summary)
        options = project.get("options", {})
        if not isinstance(options, dict) or set(options) - DEFAULT_OPTIONS.keys():
            raise ValueError("模拟选项格式错误或含有未知字段")
        options = {**DEFAULT_OPTIONS, **options}
        if options["spawn_timing"] not in ("fast", "client", "frame_core") or options["enemy_attack_timing"] not in ("float", "mortar_frames"):
            raise ValueError("不支持该模拟时序选项")
        if type(options["seed"]) is not int or not 0 <= options["seed"] <= 2**32 - 1:
            raise ValueError("种子须为0至4294967295的整数")
        if any(type(options[k]) is not bool for k in ("enemy_muzzle", "enemy_turning")):
            raise ValueError("挂点和转身开关须为布尔值")
        operations = project.get("operations")
        if not isinstance(operations, list) or len(operations) > 10000:
            raise ValueError("操作须为数组，最多10000条")
        plan = []
        for index, raw in enumerate(operations, 1):
            try:
                if not isinstance(raw, dict) or set(raw) - OP_FIELDS:
                    raise ValueError("操作格式错误或含有未知字段")
                key = raw.get("char_id")
                if key not in builds:
                    raise ValueError("操作引用的干员未携带")
                item = parse_plan([{**builds[key], **raw} if raw.get("action", "DEPLOY").upper() == "DEPLOY" else raw])[0]
                if item.action == "DEPLOY":
                    r, c = item.tile
                    grid = stage["map"]
                    if not (r < grid["height"] and c < grid["width"]):
                        raise ValueError("部署坐标超出地图")
                    required = self.characters[key].get("position")
                    if grid["cells"][r][c]["build"] != required:
                        raise ValueError("此干员不能部署在所选格子")
                elif item.action in ("SKILL", "SKILL_END", "SWITCH_MODE"):
                    build = builds[key]
                    level = self.skills.get(build.get("skill_id"), {}).get("levels", [])
                    if not level:
                        raise ValueError("此干员未携带技能")
                    skill = level[build["skill_level"] - 1]
                    controls = skill_controls(skill)
                    if item.action == "SKILL" and skill.get("skillType") != "MANUAL":
                        raise ValueError("此技能不是手动技能")
                    if item.action == "SKILL_END" and not controls.can_end:
                        raise ValueError("此技能不支持主动结束")
                    if item.action == "SWITCH_MODE" and not controls.can_switch:
                        raise ValueError("此技能不支持模式切换")
                fields = asdict(item)
                fields = {k: v for k, v in fields.items() if k in (BUILD_FIELDS | OP_FIELDS if item.action == "DEPLOY" else OP_FIELDS)}
                plan.append(fields)
            except (ValueError, TypeError, AttributeError) as error:
                raise ValueError(f"第{index}条操作：{error}") from error
        warnings = []
        if any(op["time"] > 600 for op in plan):
            warnings.append("存在600秒以后的操作，当前运行器最多模拟600秒。")
        warnings.append("地图展示计划时刻的部署示意；费用、冷却、技能技力和死亡以实际模拟结果为准。")
        return {"project": {"format": "arksim-plan-editor", "version": 1, "stage": stage_id,
                            "squad": squad, "operations": operations, "options": options},
                "plan": plan, "squad": summaries, "warnings": warnings}

    def import_plan(self, raw: Any, stage: str) -> dict:
        plans = parse_plan(raw)
        builds, operations = {}, []
        for item in plans:
            fields = asdict(item)
            if item.action == "DEPLOY":
                config, _ = self.build({k: v for k, v in fields.items() if k in BUILD_FIELDS})
                previous = builds.get(item.char_id)
                if previous is not None and previous != config:
                    raise ValueError("同一干员多次部署使用了不同练度；编辑器当前按每名干员统一配装，无法无损导入")
                builds[item.char_id] = config
            operations.append({k: v for k, v in fields.items() if k in OP_FIELDS})
        return self.compile({"format": "arksim-plan-editor", "version": 1, "stage": stage,
                             "squad": list(builds.values()), "operations": operations, "options": {}})

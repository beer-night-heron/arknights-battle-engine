# 本地数据准备

运行数据和模拟输出不随 GitHub 仓库发布。默认读取项目内的 `local/data/`；可以把数据放在任意本地目录，运行时用 `--data-dir` 指定。已有工作目录整理后，原来的数据已保留在该默认目录。

## 必需文件

文件直接放在同一个数据目录中，使用 UTF-8 JSON：

| 文件 | 用途 |
| --- | --- |
| `character_table.json` | 干员属性、精英阶段、等级、信赖与潜能 |
| `skill_table.json` | 技能参数与技能等级数据 |
| `range_table.json` | 攻击范围格 |
| `stage_table.json` | 关卡 ID 到关卡文件的映射 |
| `enemy_database.json` | 敌人属性及等级变体 |
| 对应的 `level_*.json` | 地图、路径、出怪和关卡参数 |

引擎按 `stage_table.json` 的 `stages[关卡ID].levelId` 找到文件名，只读取最后一段路径并补上 `.json`。关卡文件必须放在上述数据目录里，不能只导入关卡索引而不导入实际关卡。

## 可选文件及缺失影响

| 文件 | 用途 / 缺失时的限制 |
| --- | --- |
| `buff_template_data.json` | 技能与 Buff 行为模板；缺失会限制数据驱动技能覆盖 |
| `battle_equip_table.json` | 模组属性；使用 `module_id` 时需提供对应条目 |
| `enemy_attack_timing.json`、`operator_attack_timing.json` | 前摇、命中等攻击时序；缺失会采用引擎回退值，不能保证逐帧时序 |
| `projectile_data.json` | 弹体逻辑键、速度及单位映射；缺失无法保持相应弹道配置 |
| `enemy_buff_abilities.json` | 部分敌方全场加成配置；缺失会减少对应能力覆盖 |

`levels_meta.json` 可随基础数据一起保存，当前主运行入口不依赖它。可选文件能省略不等于相应机制已有完整替代实现；需要精细模拟时应提供完整、相互兼容的数据。基础表格中有某角色或某技能，也不代表引擎已覆盖它的所有机制。

## 从本地数据仓库导入

如已有包含游戏 JSON 的本地 Git 仓库，可使用导入工具。此工具额外需要 Git 可执行程序；模拟和 HTML 查看不需要 Git。

导入基础表格：

```powershell
python tools/export_data.py --repo "你的本地数据仓库路径"
```

导入一个或多个关卡，`--level` 填该仓库内部的实际文件路径：

```powershell
python tools/export_data.py --repo "你的本地数据仓库路径" --only-levels --level "zh_CN/gamedata/levels/实际目录/level_实际文件.json"
```

`--level` 可重复使用；也可在导入基础表格的同时传入关卡。没有 `--level` 时只导入表格，不暗中指定示例关卡。`--out` 可更换目标目录，默认 `local/data/`；`--only-behaviors` 只更新行为模板。

工具读取该数据仓库的 `HEAD`，同名目标文件会被更新。请在需要保留原数据版本时先另存一个目录。它不会生成引擎专用的攻击时序、弹体或敌方加成配置；这些文件需自行在本地提供。

## 版本管理

一次模拟应使用同一来源版本的基础表格、关卡与配置。切换数据后需要重新生成回放，已有回放不会自动改变。回放记录引擎版本、关卡、种子和时序模式；当前不会自动记录所有输入数据文件的版本或哈希，若要长期重复某次计算，请自行保留那一版数据与计划。

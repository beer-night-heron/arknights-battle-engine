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
| `enemy_graphics.json` | 可选敌方地图坐标挂点、动画采样与转身配置；只在开启相应开关时加载 |
| `enemy_buff_abilities.json` | 部分敌方全场加成配置；缺失会减少对应能力覆盖 |

`levels_meta.json` 可随基础数据一起保存，当前主运行入口不依赖它。可选文件能省略不等于相应机制已有完整替代实现；需要精细模拟时应提供完整、相互兼容的数据。基础表格中有某角色或某技能，也不代表引擎已覆盖它的所有机制。

## 敌方挂点配置格式

`enemy_graphics.json` 按敌人 ID 提供可选配置，偏移相对实体中心，坐标顺序为 `[row, col, z]`，单位与地图一致。z 为地图坐标的第三分量，不是屏幕上下偏移，也不额外乘相机投影系数。旧 `[row, col]` 配置按 z=0 兼容。填写面朝右时的坐标；左右翻转只作用于 col。下面是格式示例，数值不是任何单位的真实参数：

```json
{
  "enemies": {
    "你的敌人ID": {
      "coordinateSpace": "map",
      "initialFacing": 1,
      "turnSeconds": 0.15,
      "faceRoute": true,
      "muzzleOffset": [-0.1, 0.3, -0.2],
      "muzzleAttackSamples": [
        {"time": 0.0, "offset": [-0.1, 0.3, -0.2]},
        {"time": 0.5, "offset": [-0.2, 0.4, -0.3]}
      ]
    }
  }
}
```

`initialFacing` 为 1（右）或 -1（左）；`turnSeconds=0` 表示即时转向。`faceRoute=false` 时只在攻击起手请求朝向目标。`muzzleAttackSamples` 可省略，此时使用固定 `muzzleOffset`；采样时间为攻击起手后的秒数，必须非负且严格递增，样本间线性插值，首尾之外取最近样本。仅配置转身时可以省略挂点。敌人条目也可提供 `hitOffset`，其 col 随朝向翻转；没有完整的模型播放、动作混合，输入转换的准确性由配置决定。

同一文件可增加 `operators`，以干员 ID 索引，条目例如 `{"coordinateSpace":"map","hitOffset":[0.2,0,-0.25]}`。数值仅示范格式。干员的 HIT 当前采用地图坐标固定偏移，尚无完整动态受击骨骼。

## 弹体距离与到达配置

`projectile_data.json` 可增加 `motion`，以弹体逻辑键索引。只在开启 `--enemy-muzzle` 时用于敌方弹体；干员弹体和未配置弹体保留原流程。

```json
{
  "motion": {
    "你的弹体逻辑键": {
      "distanceDimensions": 3,
      "reachRadius": 0.1,
      "targetMount": "hit",
      "quantizedStep": true
    }
  }
}
```

`distanceDimensions` 为 2 或 3；二维按地图 row/col 前进和到达，三维同时推进 z 并按三维距离到达。每帧先移动，再检查剩余距离是否不超过 `reachRadius`。`targetMount` 为 `center` 或 `hit`，后者每帧跟踪当前目标挂点；缺少 HIT 时回退中心并记录诊断。`quantizedStep=true` 在 30Hz 使用 `0.03333330154418945` 秒步长，其他逻辑频率使用当前 dt。缺省分别为二维、零容差、中心和普通 dt。二维弹体的真实抛物线高度曲线未实现，回放弧线仍是展示近似。

## 动画事件前摇配置

`enemy_attack_timing.json` 的单位条目可增加 `"wait_for_attack_event": true`。只有显式启用、采用普通远程攻击且 `SimulationAssumptions.enemy_animation_event_clock=true` 时，前摇才按 float32 动画时间逐帧累加至 `attack_hit_time`，不贴齐整数帧。在 30Hz 采用上述量化步长；其他频率采用当前 dt。不改近战阻挡攻击、技能、冷却、索敌或攻击结束后的移动恢复。

该计时仍是候选模型：动画从零起算、速度为 1；事件交接子阶段、跳过时间、动作混合和不同动画速度未完整覆盖。可将该假设关闭或去掉单位标记以回退原计时，不能对所有单位直接套用“加一帧”。

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

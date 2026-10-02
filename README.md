# ARKSIM · 主战斗引擎

基于战斗机制构建的 Python 模拟器。用 JSON 编写部署计划，计算战斗结果，再通过 HTML 页面查看地图、单位状态与事件随时间的变化。

当前以 `Battle` 为唯一主战斗引擎。已实现移动、阻挡、索敌、攻击、部分技能与 Buff、伤害及弹体流程；角色和关卡机制尚未全部覆盖。具体范围见 [机制支持与缺口](docs/MECHANICS.md)。

## 开始使用

需要 Python 3.10 或更新版本，以及支持 Canvas 的现代浏览器。运行时只使用 Python 标准库；在项目根目录执行命令，无需安装额外 Python 包。游戏数据由使用者在本地准备，不随仓库发布。

1. 将运行表格和关卡 JSON 放进 `local/data/`，或用 `--data-dir` 指定自己的数据目录。文件清单和导入方法见 [数据准备](docs/DATA.md)。
2. 查询已有数据中的关卡与干员 ID：

   ```powershell
   python tools/list_data.py --kind stages
   python tools/list_data.py --kind operators --query "干员名称"
   ```

3. 参考 [计划 JSON 说明](docs/PLAN.md)，把自己的计划保存为 `local/plans/my-plan.json`。可复制 `examples/plan.template.json`，替换其中的 ID、坐标、时间及配置。
4. 用实际关卡 ID 替换下面的占位文字，再生成模拟：

   ```powershell
   python tools/simulate.py --stage "关卡ID" --plan "local/plans/my-plan.json"
   ```

5. 打开 [`viewer/index.html`](viewer/index.html)，在“打开回放数据”中选择 `local/最新模拟/replay.json`。页面可离线使用；换文件时点击“退出当前文件”。

每次成功生成都会更新 `local/最新模拟/`，上一版完整保存在 `local/模拟历史/`。查看页面是固定文件，只需读取新的 JSON。逐帧播放、单位详情、文件区别和批量运行说明见 [使用指南](docs/USAGE.md)。

## 只计算结果

```powershell
python -m arksim.cli --stage "关卡ID" --plan "local/plans/my-plan.json"
python -m arksim.cli --stage "关卡ID" --plan "local/plans/my-plan.json" --runs 100 --seed 1000 --workers 4 --output "local/results/batch.json"
```

`arksim.cli` 输出结果摘要；`tools/simulate.py` 生成可播放的回放。摘要 JSON 不能当作回放文件读取。

## 项目组成

| 路径 | 用途 |
| --- | --- |
| `arksim/` | 主战斗引擎：状态推进、机制计算、技能与 Buff、索敌和统计 |
| `tools/simulate.py` | 执行计划并生成模拟回放，维护最新与历史结果 |
| `tools/list_data.py` | 查询本地关卡、干员配置和地图坐标 |
| `tools/export_data.py` | 从本地数据 Git 仓库导入基础表格和关卡 |
| `viewer/` | 固定的 HTML 回放读取页 |
| `examples/` | 通用计划模板 |
| `docs/` | 使用、计划字段、数据准备、引擎组成与机制缺口 |
| `local/` | 本地数据、个人计划、模拟输出及归档；不上传 GitHub |
| `CHANGELOG.md` | 引擎与工具更新记录 |

引擎各模块的职责及程序调用方法见 [引擎组成](docs/ENGINE.md)。战斗结果中的机制警告和 `assumptions` 用于说明本局未覆盖的内容与采用的近似规则；没有警告也不代表所有细节都已验证。

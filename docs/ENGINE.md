# 主战斗引擎的组成

主入口是 `arksim.battle.Battle`。单局在默认 `1/30` 秒的逻辑步长内推进，`run()` 一直执行到生命归零、所有已排程敌人处理完毕，或默认 600 秒上限。`step()` 可供程序逐帧读取状态。改变回放采样密度不会改变逻辑步长。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `battle.py` | 战斗状态与阶段协调：部署、费用、出怪、实体更新、移动、阻挡、攻击事件、技能和结束判定 |
| `data.py` | 读取本地 JSON，索引敌人，计算等级、信赖、潜能和模组属性 |
| `mechanics.py` | 伤害包、物理/法术/真实伤害、状态、元素效果及目标排序规则 |
| `targeting.py` | 帧周期计时器、目标筛选与缓存生命周期 |
| `buffs.py` | Buff 定义、实例、叠加、属性修正和生命周期 |
| `behavior.py` | 数据驱动行为模板的节点执行与未支持内容诊断；战斗引擎提供作用对象和副作用接口 |
| `spawn_core.py` | 可选的静态动作队列出怪排程 |
| `assumptions.py` | 尚需完善的规则参数与解释；结果只记录实际使用的假设 |
| `randomness.py` | 固定种子的随机抽取与计数 |
| `batch.py` | 多次运行的胜率、用时和置信区间摘要 |
| `enumeration.py` | 参数组合与种子组合的通用枚举接口；尚未集成到页面 |
| `cli.py` | 计划校验、共享输入加载、单局与多进程批量入口 |

引擎不会把所有干员、敌人、伤害和技能都放进同一种简单顺序。普通实体按更新优先级与共享创建顺序推进；部分延迟效果、弹体、Buff 与特定攻击有自己的处理阶段。改变阶段顺序可能影响同帧可见的目标位置、伤害和技能状态，因此目录整理与查看工具不会改动这些规则。

## 在 Python 中调用

```python
from arksim.battle import Battle
from arksim.cli import build_config

config = build_config(
    "local/data",
    "替换为关卡ID",
    "local/plans/my-plan.json",
    spawn_timing="fast",
)
battle = Battle(**config, seed=0)
result = battle.run()
print(result.win, result.enemies_killed, result.enemies_leaked)
```

如需逐帧使用，创建 `Battle` 后自行调用 `step()` 并检查结束条件。不要在 `run()` 完成后再对同一实例调用 `run()` 期待新的一局；每次计算建立新实例。

`data.py` 的数据目录是进程级配置，切换数据目录需重新加载输入。批量入口的工作进程接收独立配置；不建议在同一进程的多个线程中交替设置不同数据目录。

## 结果字段

| 字段 | 含义 |
| --- | --- |
| `win`、`reason` | 是否全清，以及 `all_clear` / `life_points_depleted` / `timeout` |
| `time` | 结束时的模拟秒数 |
| `life_points`、`max_life_points` | 剩余与初始生命 |
| `enemies_spawned`、`enemies_killed`、`enemies_leaked` | 出场、击杀、漏怪计数 |
| `operators_deployed` | 结束时仍在场的存活干员数，并非部署动作总数 |
| `operator_metrics` | 各次部署的属性、伤害与存活统计 |
| `seed`、`random_draws` | 种子与实际随机抽取次数 |
| `behavior_warnings`、`unsupported_behavior_nodes` | 行为模板未覆盖情况 |
| `mechanic_warnings`、`unsupported_mechanics` | 已能识别的机制缺口 |
| `assumptions` | 本局采用的近似规则说明 |
| `spawn_timing`、`enemy_attack_timing` | 使用的时序模式 |

这是胜负与诊断摘要，不是完整的战斗调用轨迹。回放工具另外保存每帧状态与由状态差生成的事件。

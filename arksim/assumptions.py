"""Centralized switches for mechanics that are not fully public.

The profile keeps guesses explicit and replaceable.  A battle result only
reports assumptions that were actually relevant to that run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationAssumptions:
    herald_auras_stack: bool = True
    allow_active_skill_sp_gain: bool = False
    highland_hit_uses_tile_center: bool = False
    enemy_target_collision_radius: float = 0.1
    steering_foot_offset_row: float = -0.2
    steering_foot_offset_col: float = 0.0
    steering_half_body_width: float = 0.2
    enemy_spawn_move_lock_frames: int = 2
    enemy_spawn_attack_lock_frames: int = 2
    enemy_target_search_period_frames: int = 3
    entity_update_order_by_allocation: bool = True
    headb2_periodic_sp_recovery: bool = True
    operator_target_selector_before_enemy_movement: bool = False
    operator_block_scan_from_deploy_frame: bool = True
    mortar_first_attack_extra_frames: int = 2
    retime_enemy_attack_cooldown: bool = True
    ranged_attack_move_transition_frames: int = 2
    # Highland echo is committed after the current frame's movement.  When
    # the receiving frame has already consumed its movement, two extra
    # logical frames cover the native one-frame extension plus that phase
    # boundary in the duration-based status model.
    sluggish_extra_frames: int = 2
    portal_wait_extra_frames: int = 1
    route_wait_expiry_epsilon: float = 1e-9


DEFAULT_ASSUMPTIONS = SimulationAssumptions()


ASSUMPTION_NOTES = {
    "HEADB2_PERIODIC_SP_RECOVERY": "凛冬S2在30Hz、每秒1SP配置下按Q32.32周期计时器恢复整数SP；技力锁或满SP时保留计时余量。该模型暂不推广到其他角色或恢复速度。",
    "ENTITY_UPDATE_ORDER_BY_ALLOCATION": "普通干员和敌人按更新优先级降序、共享创建序号升序交错更新；特殊对象尚未完整覆盖。",
    "HERALD_AURAS_STACK": "同场存活传令兵的攻击力+10%与防御力+100光环叠加；隐藏不撤销来源。",
    "ACTIVE_SKILL_SP_LOCK": "怒潮凛冬S2生效期间使用技力锁，高台触发不会提前储存下次技力；该规则仍属当前模型。",
    "HIGHLAND_TILE_CENTER": "可选几何模型：溅射圆是否覆盖高台按高台格中心判定。",
    "HIGHLAND_TILE_INTERSECTION": "溅射圆与整块高台格碰撞区域相交时，判为覆盖高台。",
    "REVERSE_ENGINEERED_NEXT_NODE": "路径采用四方向加权搜索、nextNode平整化与路线级转向模式；特殊导航仍有限制。",
    "REVERSE_ENGINEERED_BLOCKING": "阻挡点以二次ease-out轨迹在0.2秒内到达；同帧阻挡争抢按单位创建顺序处理，特殊情形仍待完善。",
    "OPERATOR_BLOCK_SCAN_DEPLOYMENT_PHASE": "干员从部署帧的下一帧开始每3帧阻挡扫描；新关系在对应敌人后续实体更新中消费，该周期仍是可替换模型。",
    "STEERING_FOOT_POSITION": "缺少逐单位避障形状时，FootPoint采用实体下方0.2格、半宽0.2格的近似。",
    "ENEMY_SPAWN_FRAME_NO_MOVE": "出生逻辑帧与紧接的状态过渡帧不执行自主移动；其他出生动作仍需独立适配。",
    "ENEMY_SPAWN_ACTION_LOCK": "普通攻击敌人出生后的两个30Hz逻辑帧不开始新攻击；该时序是当前近似模型。",
    "ENEMY_TARGET_SEARCH_CYCLE": "近战普通攻击沿用阻挡目标；远程敌人在需要搜索时采用3帧周期和目标缓存，特殊敌人不保证适用。",
    "MELEE_BLOCK_STARTS_ATTACK": "阻挡关系建立后，近战敌人在下一次自身更新启动普通攻击，可能位于同一逻辑帧或下一帧；该边界仍待完善。",
    "MELEE_BLOCK_ATTACK_HIT_AFTER_WINDUP": "近战阻挡攻击在前摇结束后的下一逻辑帧造成伤害或发射弹体；缺少前摇数据时回退为0秒，该规则仍是模型假设。",
    "OPERATOR_SELECTOR_PRE_MOVEMENT_SEARCH": "可选分组模式在全部敌人移动前搜索；默认创建序模式中，查询可见位置取决于双方更新顺序。",
    "MORTAR_FIRST_ATTACK_EXTRA_FRAMES": "部分炮兵首轮攻击后的冷却比后续节奏额外跨2个逻辑帧；适用范围有限，不是通用敌人规则。",
    "ENEMY_ATTACK_COOLDOWN_RETIME": "有效攻击间隔变化时，活动冷却按新旧间隔比例缩放；不同敌人或效果的适用性仍需完善。",
    "RANGED_ATTACK_MOVE_TRANSITION": "普通远程敌人的攻击动画时长转换为整数逻辑帧，随后跨2帧状态衔接才恢复自主位移；多单位观察支持该候选模型，特殊动作、打断和不同攻速仍待验证。",
    "HEADB2_ECHO_TARGET_AT_EXECUTION": "高台回响保存触发格，0.1秒后按执行帧的敌人当前位置查询影响格并结算；不预缓存受击目标。",
    "SLUGGISH_DURATION_EXTRA_FRAME": "移动后提交的高台停顿若未在接收帧消费，时长补入2个逻辑帧以表达当前状态阶段模型；不可直接推广到其他控制效果。",
    "PORTAL_WAIT_EXTRA_FRAMES": "传送门隐藏等待使用严格浮点计时；出口追加1帧可受击停顿，归零帧消费后恢复移动；其他传送动作仍待完善。",
    "HEADB2_HAMMER_POST_MOVEMENT_HIT": "凛冬锤击在到期帧的普通实体移动完成后、回响和Buff计时前结算；该阶段不推广到所有伤害事件。",
    "ROUTE_WAIT_EXPIRY_EPSILON": "可见路线等待剩余量不超过1e-9秒时结束以消除浮点尾差；隐藏和传送等待保留严格计时。",
    "INITIAL_ROUTE_WAIT_SAME_FRAME_MOVE": "可见出生路线的首个WAIT可在归零帧恢复移动；途中等待及隐藏等待保留各自阶段，该规则仍为候选模型。",
    "MORTAR_ATTACK_INTERVAL_LOGIC_FRAMES": "可选炮兵整数帧模型按30Hz四舍五入有效攻击间隔；默认采用浮点路径，模式不表示通用精确时序。",
    "ENEMY_TARGET_COLLISION_RADIUS": "普通范围索敌和圆形溅射采用0.1格实体圆近似；凛冬高台回响使用离散影响格规则。",
    "CLIENT_COROUTINE_SPAWN_TIMING": "逐帧出怪模型为Fragment进入、敌人生成和路线预览累积调度帧；未覆盖动作仍采用近似。",
    "SPAWN_CORE_FRAME_SCHEDULER": "静态出怪采用SpawnCore毫帧动作队列、同刻排序和Fragment交接；动态波次门槛仍有限制。",
    "CLIENT_UNSTABLE_ACTION_SORT": "同Fragment相同preDelay的动作使用不稳定快速排序；同刻生成顺序可能与输入数组不同。",
    "CLIENT_DISPLAY_ENEMY_INFO_WORK_FRAME": "DISPLAY_ENEMY_INFO作为独立调度工作项累积一帧，在同刻普通动作后完成；关联预览尚未完整展开。",
    "CLIENT_PREVIEW_COMPLETION_ORDER": "同刻路线预览可改变单次SPAWN完成帧，该局部完成帧不继续延迟后续重复动作；排序边界仍是模型。",
    "PROJECTILE_SOURCE_CENTER": "未提供世界坐标发射挂点时，弹体从攻击者实体中心发射；挂点、缩放和朝向仍待完善。",
    "PROJECTILE_OPERATOR_INFERRED": "干员普通弹道仅在存在对应逻辑键时推断；技能专用和纯视觉弹体不自动套用。"
}

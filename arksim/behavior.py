"""Interpreter for Arknights ``buff_template_data.json`` action nodes.

The client data stores event handlers as small behaviour trees.  This module
owns their control flow, blackboard access and diagnostics; the battle engine
only supplies entity lookup and side-effect adapters such as damage or buffs.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable


class BehaviorError(RuntimeError):
    """Base error for malformed or unsupported behaviour data."""


class UnknownNodeError(BehaviorError):
    """Raised in strict mode when a node has no registered implementation."""


def node_name(node: dict[str, Any]) -> str:
    """Return ``IfElse`` from an assembly-qualified node type."""
    qualified = str(node.get("$type", ""))
    return qualified.split(",", 1)[0].rsplit("+", 1)[-1]


@dataclass
class BehaviorDiagnostics:
    """Observable coverage information for non-strict simulation runs."""

    unsupported_nodes: Counter[str] = field(default_factory=Counter)
    missing_templates: Counter[str] = field(default_factory=Counter)

    @property
    def warning_count(self) -> int:
        return sum(self.unsupported_nodes.values()) + sum(self.missing_templates.values())


@dataclass
class BehaviorContext:
    """Mutable event state shared by nodes in one dispatched event."""

    source: Any = None
    target: Any = None
    buff_owner: Any = None
    buff_source: Any = None
    modifier_source: Any = None
    modifier_target: Any = None
    blackboard: dict[str, Any] = field(default_factory=dict)
    ability_blackboard: dict[str, Any] = field(default_factory=dict)
    battle: Any = None
    diagnostics: BehaviorDiagnostics = field(default_factory=BehaviorDiagnostics)
    template_key: str = ""
    event: str = ""
    attack_scale: float = 1.0
    damage_scale: float = 1.0
    trigger_ratio: float = 0.0
    current_buff_key: str | None = None
    buff_container: Any = None
    buff_instance: Any = None

    def entity(self, target_type: str | None) -> Any:
        roles = {
            "SOURCE": self.source,
            "TARGET": self.target,
            "BUFF_OWNER": self.buff_owner,
            "BUFF_SOURCE": self.buff_source,
            "MODIFIER_SOURCE": self.modifier_source,
            "MODIFIER_TARGET": self.modifier_target,
        }
        return roles.get(str(target_type or "TARGET"), self.target)


NodeHandler = Callable[[dict[str, Any], BehaviorContext], bool]


class BehaviorInterpreter:
    """Execute event action lists with explicit unknown-node handling."""

    FLOW_NODES = {"AlwaysNext", "IfNot", "Not"}

    def __init__(
        self,
        templates: dict[str, Any] | None = None,
        *,
        strict: bool = True,
        diagnostics: BehaviorDiagnostics | None = None,
    ) -> None:
        self.templates = templates or {}
        self.strict = strict
        self.diagnostics = diagnostics or BehaviorDiagnostics()
        self.handlers: dict[str, NodeHandler] = {
            "AOEDamage": self._aoe_damage,
            "AssignBuffBlackboardFromOthers": self._assign_buff_blackboard,
            "AssignCharacterSkillBlackboardToBB": self._assign_skill_blackboard,
            "AssignValueToBB": self._assign_value,
            "AtkScaleUp": self._atk_scale_up,
            "BlackboardAdd": self._blackboard_add,
            "CalculateBlackboardValueViaParams": self._calculate_blackboard,
            "CheckContainsBuff": self._check_contains_buff,
            "CheckEnemyLevelMask": self._check_enemy_level_mask,
            "CreateBuff": self._create_buff,
            "DamageScale": self._damage_scale,
            "DamageViaAttr": self._damage_via_attr,
            "DamageViaMaxHpRatio": self._damage_via_max_hp_ratio,
            "FinishBuff": self._finish_buff,
            "FinishBuffsById": self._finish_buffs_by_id,
            "IfElse": self._if_else,
            "IfTarget": self._if_target,
            "IsBlackboardZero": self._is_blackboard_zero,
            "Not": self._not_condition,
            "RemainingRatioToAttributeModifier": self._remaining_ratio_modifier,
            "SetAtkScaleZero": self._set_atk_scale_zero,
            "SwitchMode": self._switch_mode,
        }

    @property
    def supported_nodes(self) -> frozenset[str]:
        return frozenset(self.handlers) | self.FLOW_NODES

    def dispatch(
        self, template_key: str, event: str, context: BehaviorContext
    ) -> bool:
        template = self.templates.get(template_key)
        if template is None:
            self.diagnostics.missing_templates[template_key] += 1
            if self.strict:
                raise BehaviorError(f"unknown behavior template: {template_key}")
            return True
        context.template_key = template_key
        context.event = event
        context.diagnostics = self.diagnostics
        actions = template.get("eventToActions", {}).get(event, [])
        return self.execute_actions(actions, context)

    def execute_actions(
        self, actions: list[dict[str, Any]], context: BehaviorContext
    ) -> bool:
        """Execute a linear action list.

        A false predicate gates later ordinary nodes. ``IfNot`` can invert that
        state and ``AlwaysNext`` restores it, matching the control markers found
        in real templates.
        """
        state = True
        for node in actions:
            name = node_name(node)
            if name == "AlwaysNext":
                state = True
                continue
            if name in {"IfNot", "Not"}:
                state = not state
                continue
            if not state:
                continue
            state = self.execute_node(node, context)
        return state

    def execute_node(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        name = node_name(node)
        handler = self.handlers.get(name)
        if handler is None:
            self.diagnostics.unsupported_nodes[name or "<missing $type>"] += 1
            if self.strict:
                raise UnknownNodeError(
                    f"unsupported node {name!r} in "
                    f"{context.template_key or '<direct>'}:{context.event or '<direct>'}"
                )
            return True
        return bool(handler(node, context))

    # ------------------------------------------------------------ control
    def _if_else(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        condition = node.get("_conditionNode")
        passed = bool(condition) and self.execute_node(condition, context)
        branch = node.get("_succeedNodes" if passed else "_failNodes", [])
        return self.execute_actions(branch, context)

    def _if_target(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        entity = context.entity(node.get("_targetType"))
        if entity is None:
            return False
        if node.get("_checkTargetAlive") and getattr(entity, "dead", False):
            return False
        if node.get("_checkTargetUnitType"):
            expected = str(node.get("_unitType", ""))
            actual = str(getattr(entity, "unit_type", ""))
            if expected and actual and expected != actual:
                return False
        return True

    def _not_condition(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        condition = node.get("_conditionNode") or node.get("_childNode")
        if condition is None:
            return False
        return not self.execute_node(condition, context)

    # --------------------------------------------------------- blackboard
    def _assign_value(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        key = str(node.get("_blackboardKey", ""))
        if not key:
            return False
        value: Any = node.get("_value", 0.0)
        if node.get("_assignString"):
            value = node.get("_stringValue", node.get("_valueStr", ""))
        context.blackboard[key] = value
        return True

    def _blackboard_add(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        key = str(node.get("_blackboardKey", ""))
        if not key:
            return False
        addition_key = node.get("_additionKey")
        addition = (
            context.blackboard.get(str(addition_key), 0)
            if addition_key
            else node.get("_addition", 0)
        )
        value = float(context.blackboard.get(key, 0)) + float(addition or 0)
        context.blackboard[key] = value if node.get("_isFloat") else int(value)
        return True

    def _calculate_blackboard(
        self, node: dict[str, Any], context: BehaviorContext
    ) -> bool:
        board = context.ability_blackboard if node.get("_useAbilityBlackboard") else context.blackboard
        input_key = str(node.get("_inputKey", ""))
        output_key = str(node.get("_outputKey", input_key))
        if not input_key or input_key not in board:
            return False
        value = float(board[input_key])
        multiply_key = node.get("_multiplyParamKey")
        divide_key = node.get("_dividedParamKey")
        add_key = node.get("_addParamKey")
        if multiply_key:
            value *= float(board.get(str(multiply_key), 0))
        if divide_key:
            divisor = float(board.get(str(divide_key), 0))
            if divisor == 0:
                return False
            value = value % divisor if node.get("_useRemainder") else value / divisor
        if add_key:
            value += float(board.get(str(add_key), 0))
        max_key = node.get("_maxValueKey")
        if max_key and str(max_key) in board:
            value = min(value, float(board[str(max_key)]))
        min_key = node.get("_minValueKey")
        if min_key and str(min_key) in board:
            value = max(value, float(board[str(min_key)]))
        if node.get("_finalAbs"):
            value = abs(value)
        if node.get("_finalCeil"):
            value = math.ceil(value)
        elif node.get("_finalFloor"):
            value = math.floor(value)
        elif node.get("_finalRound"):
            value = round(value)
        board[output_key] = value
        return True

    def _is_blackboard_zero(
        self, node: dict[str, Any], context: BehaviorContext
    ) -> bool:
        key = str(node.get("_var", ""))
        if key not in context.blackboard:
            return False
        return abs(float(context.blackboard[key])) <= 1e-9

    def _assign_skill_blackboard(
        self, node: dict[str, Any], context: BehaviorContext
    ) -> bool:
        entity = context.entity(node.get("_targetType"))
        skill_board = getattr(entity, "skill_blackboard", None)
        source_key = str(node.get("_sourceBlackboardKey", ""))
        target_key = str(node.get("_targetBlackboardKey", source_key))
        if not isinstance(skill_board, dict) or source_key not in skill_board:
            return False
        context.blackboard[target_key] = skill_board[source_key]
        return True

    def _assign_buff_blackboard(
        self, node: dict[str, Any], context: BehaviorContext
    ) -> bool:
        entity = context.entity(node.get("_targetType"))
        boards = getattr(entity, "buff_blackboards", None)
        buff_key = str(node.get("_buffKey", ""))
        source_key = str(node.get("_blackboardKey", ""))
        target_key = str(node.get("_valueKey", source_key))
        if not isinstance(boards, dict):
            return False
        board = boards.get(buff_key)
        if not isinstance(board, dict) or source_key not in board:
            return False
        context.blackboard[target_key] = board[source_key]
        return True

    # --------------------------------------------------------- predicates
    def _check_contains_buff(
        self, node: dict[str, Any], context: BehaviorContext
    ) -> bool:
        entity = context.entity(node.get("_targetType"))
        container = getattr(entity, "buffs", None)
        boards = getattr(entity, "buff_blackboards", {})
        keys = node.get("_buffKeys", [])
        if node.get("_loadFromBlackboard"):
            keys = [context.blackboard.get(str(key), key) for key in keys]
        if container is not None and hasattr(container, "contains"):
            source = None
            if node.get("_checkBuffSource"):
                source = context.entity(node.get("_buffSourceType"))
            results = [
                bool(container.contains(str(key), source=source)) for key in keys
            ]
            return all(results) if node.get("isAND", True) else any(results)
        results = [str(key) in boards for key in keys]
        return all(results) if node.get("isAND", True) else any(results)

    def _check_enemy_level_mask(
        self, node: dict[str, Any], context: BehaviorContext
    ) -> bool:
        entity = context.entity(node.get("_targetType"))
        level_type = str(getattr(entity, "level_type", "NORMAL")).upper()
        mask = str(node.get("_targetLevelMask", "")).upper()
        if mask == "ELITE_AND_BOSS":
            return level_type in {"ELITE", "BOSS"}
        if mask in {"ALL", "NORMAL_AND_ELITE_AND_BOSS"}:
            return True
        return level_type == mask

    # ---------------------------------------------------- damage modifiers
    def _atk_scale_up(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        key = str(node.get("_atkScaleKey", "atk_scale"))
        value = float(context.blackboard.get(key, node.get("_defaultValue", 1.0)))
        if node.get("_overwriteAtkScale"):
            context.attack_scale = value
        else:
            context.attack_scale *= value
        return not (node.get("_cancelIfAtkScaleZero") and context.attack_scale == 0)

    def _set_atk_scale_zero(
        self, node: dict[str, Any], context: BehaviorContext
    ) -> bool:
        context.attack_scale = 0.0
        return True

    def _damage_scale(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        key = str(node.get("_customKey") or "damage_scale")
        value = float(context.blackboard.get(key, 0.0))
        if node.get("_isOneMinus"):
            value = 1.0 - value
        context.damage_scale *= value
        return True

    def _remaining_ratio_modifier(
        self, node: dict[str, Any], context: BehaviorContext
    ) -> bool:
        if context.battle is None or context.source is None:
            return False
        return bool(
            context.battle.behavior_set_timed_attribute_scale(
                context,
                str(node.get("_attributeType", "")),
                context.trigger_ratio,
            )
        )

    def _damage_via_attr(
        self, node: dict[str, Any], context: BehaviorContext
    ) -> bool:
        target = context.entity(node.get("_targetType"))
        source = context.entity(node.get("_sourceType"))
        if target is None or source is None or context.battle is None:
            return False
        attr_name = _attribute_name(str(node.get("_attributeType", "ATK")))
        attr_owner = target if node.get("_getAttrFromTarget") else source
        base = float(getattr(attr_owner, attr_name, 0.0))
        scale = float(context.blackboard.get(str(node.get("_blackboardKey", "atk_scale")), 1.0))
        if node.get("_multiplierByKey"):
            scale *= float(context.blackboard.get(str(node.get("_multiplierKey", "value")), 0.0))
        return bool(
            context.battle.behavior_deal_damage(
                source, target, base * scale, str(node.get("_damageType", "PHYSICAL")),
                undeadable=bool(node.get("_isUndeadable")),
            )
        )

    def _damage_via_max_hp_ratio(
        self, node: dict[str, Any], context: BehaviorContext
    ) -> bool:
        target = context.entity(node.get("_targetType"))
        source = context.source
        if target is None or source is None or context.battle is None:
            return False
        hp_owner = target if node.get("_getMaxHpFromTarget") else source
        ratio_key = str(node.get("_blackboardKey", "hp_ratio"))
        ratio = float(context.blackboard.get(ratio_key, 0.0))
        base = float(getattr(hp_owner, "max_hp", 0.0)) * ratio
        if node.get("_multiplyByKey"):
            base *= float(context.blackboard.get(str(node.get("_multiplierKey", "cnt")), 0.0))
        return bool(
            context.battle.behavior_deal_damage(
                source, target, base, str(node.get("_damageType", "PURE")),
                undeadable=bool(node.get("_isUndeadable")),
            )
        )

    def _aoe_damage(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        if context.battle is None:
            return False
        return bool(context.battle.behavior_aoe_damage(context, node))

    # --------------------------------------------------------------- buffs
    def _create_buff(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        if context.battle is None:
            return False
        return bool(context.battle.behavior_create_buff(context, node))

    def _finish_buff(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        if context.battle is None or not context.current_buff_key:
            return False
        return bool(
            context.battle.behavior_finish_buff(
                context.buff_owner, context.current_buff_key
            )
        )

    def _finish_buffs_by_id(
        self, node: dict[str, Any], context: BehaviorContext
    ) -> bool:
        if context.battle is None:
            return False
        owner = context.entity(node.get("_targetType"))
        key = str(node.get("_buffKey", ""))
        if node.get("_loadFromBlackboard"):
            key = str(context.blackboard.get(key, key))
        return bool(context.battle.behavior_finish_buff(owner, key))

    def _switch_mode(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        if context.battle is None:
            return False
        return bool(context.battle.behavior_switch_mode(context, node))


def _attribute_name(value: str) -> str:
    return {
        "ATK": "atk",
        "DEF": "defense",
        "MAX_HP": "max_hp",
        "MAGIC_RESISTANCE": "res",
    }.get(value.upper(), value.lower())


PREFAB_TEMPLATE_OVERRIDES = {
    "skchr_amiya_2": "amiya2_s_2",
}


def behavior_template_candidates(prefab_id: str) -> list[str]:
    """Return conservative template-key candidates for a skill prefab."""
    candidates = [prefab_id]
    override = PREFAB_TEMPLATE_OVERRIDES.get(prefab_id)
    if override:
        candidates.insert(0, override)
    if prefab_id.startswith("skchr_"):
        body = prefab_id.removeprefix("skchr_")
        name, separator, skill_no = body.rpartition("_")
        if separator and skill_no.isdigit():
            candidates.append(f"{name}_s_{skill_no}")
    return list(dict.fromkeys(candidates))


def resolve_behavior_template(
    prefab_id: str, templates: dict[str, Any]
) -> str | None:
    for candidate in behavior_template_candidates(prefab_id):
        if candidate in templates:
            return candidate
    return None

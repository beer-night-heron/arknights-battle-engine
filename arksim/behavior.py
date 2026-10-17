"""Interpreter for Arknights ``buff_template_data.json`` action nodes.

The client data stores event handlers as small behaviour trees.  This module
owns their control flow, blackboard access and diagnostics; the battle engine
only supplies entity lookup and side-effect adapters such as damage or buffs.
"""

from __future__ import annotations

from .buff_database import BuffDatabase, BUFF_DATA_FIELDS

import math
import struct
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable


class BehaviorError(RuntimeError):
    """Base error for malformed or unsupported behaviour data."""


class UnknownNodeError(BehaviorError):
    """Raised in strict mode when a node has no registered implementation."""


ABNORMAL_READ_FLAGS = {
    "STUNNED": "STUNNED", "FROZEN": "FROZEN", "INVISIBLE": "INVISIBLE",
    "CAMOUFLAGE": "CAMOUFLAGE", "INVINCIBLE": "INVINCIBLE", "HEAL_FREE": "HEAL_FREE",
    "UNDEADABLE": "UNDEADABLE", "SILENCED": "SILENCE", "SP_RECOVER_STOPPED": "SP_BLOCKED",
    "STUNNED_NO_AMPLIFY_DAMAGE": "DISABLED", "DISARMED": "DISARMED",
    "FORCE_DISARMED": "FORCE_DISARMED", "SKILL_NOT_ACTIVATABLE": "SKILL_NOT_ACTIVATABLE",
    "LEVITATE": "LEVITATE", "GROUND_BOUND": "GROUND_BOUND", "FORCE_LEVITATE": "FORCE_LEVITATE",
}


def node_name(node: dict[str, Any]) -> str:
    """Return ``IfElse`` from an assembly-qualified node type."""
    qualified = str(node.get("$type", ""))
    return qualified.split(",", 1)[0].rsplit("+", 1)[-1]


def rounded_sp_ratio(base: float, ratio: float) -> int:
    """Default Q32.32 SP ratio path: float32 ratio, nearest-even integer."""
    scale = 1 << 32
    ratio32 = struct.unpack("<f", struct.pack("<f", float(ratio)))[0]
    raw = (math.trunc(float(base) * scale) * math.trunc(ratio32 * scale)) >> 32
    integer, fraction = divmod(raw, scale)
    return integer + int(fraction > scale // 2 or
                         (fraction == scale // 2 and integer % 2 != 0))


def blackboard_calculation_issue(node: dict[str, Any]) -> str | None:
    """Keep parameter calculations within the reviewed scalar slice."""
    fields = {"$type", "_inputKey", "_outputKey", "_multiplyParamKey", "_dividedParamKey",
        "_addParamKey", "_minusParamKey", "_maxValueKey", "_minValueKey", "_useRemainder",
        "_finalAbs", "_finalCeil", "_finalFloor", "_finalRound", "_useAbilityBlackboard", "_abilityName"}
    if set(node) - fields:
        return "fields"
    for key in ("_useRemainder", "_finalRound", "_useAbilityBlackboard", "_abilityName"):
        if node.get(key):
            return key
    if node.get("_minusParamKey") and any(node.get(key) for key in (
            "_multiplyParamKey", "_dividedParamKey", "_addParamKey", "_maxValueKey", "_minValueKey",
            "_finalAbs", "_finalCeil", "_finalFloor")):
        return "subtraction_combination"
    for key in ("_finalAbs", "_finalCeil", "_finalFloor", "_finalRound", "_useRemainder", "_useAbilityBlackboard"):
        if type(node.get(key, False)) not in (bool, int) or node.get(key, False) not in (False, True):
            return key + ":flag"
    if node.get("_finalCeil") and node.get("_finalFloor"):
        return "rounding_flags"
    for key in ("_inputKey", "_outputKey"):
        value = node.get(key, node.get("_inputKey") if key == "_outputKey" else None)
        if not isinstance(value, str) or not value:
            return key
    for key in ("_multiplyParamKey", "_dividedParamKey", "_addParamKey", "_minusParamKey",
                "_maxValueKey", "_minValueKey", "_abilityName"):
        if node.get(key) is not None and not isinstance(node[key], str):
            return key
    return None


def calculated_blackboard_value(node: dict[str, Any], board: dict[str, Any]) -> tuple[bool, str, float | int | None]:
    """Compute on a snapshot; missing input/zero divisor are false gates."""
    if issue := blackboard_calculation_issue(node):
        raise ValueError(issue)
    key = node["_inputKey"]
    output = node.get("_outputKey", key)
    if key not in board:
        return False, output, None

    def number(value):
        if type(value) not in (int, float):
            raise ValueError("numeric_value")
        try:
            result = float(value)
        except OverflowError as exc:
            raise ValueError("numeric_value") from exc
        if not math.isfinite(result):
            raise ValueError("numeric_value")
        return result

    value = number(board[key])
    minus = node.get("_minusParamKey")
    if minus:
        if minus not in board:
            raise ValueError("subtraction_operand_binding")
        return True, output, number(value - number(board[minus]))
    multiply = node.get("_multiplyParamKey")
    divide = node.get("_dividedParamKey")
    add = node.get("_addParamKey")
    if multiply:
        value = number(value * number(board.get(multiply, 0)))
    if divide:
        divisor = number(board.get(divide, 0))
        if divisor == 0:
            return False, output, None
        value = number(value / divisor)
    if add:
        value = number(value + number(board.get(add, 0)))
    maximum, minimum = node.get("_maxValueKey"), node.get("_minValueKey")
    high = number(board[maximum]) if maximum and maximum in board else None
    low = number(board[minimum]) if minimum and minimum in board else None
    if high is not None and low is not None and low > high:
        raise ValueError("clamp_bounds")
    if high is not None:
        value = min(value, high)
    if low is not None:
        value = max(value, low)
    if node.get("_finalAbs"):
        value = abs(value)
    if node.get("_finalCeil"):
        value = math.ceil(value)
    elif node.get("_finalFloor"):
        value = math.floor(value)
    return True, output, value


ENTITY_VALUE_READ_FIELDS = {
    "AssignAttributeToBB": {"_targetType", "_attributeType", "_scaleVar", "_blackboardKey"},
    "AssignHpRatioToBB": {"_targetType", "_blackboardKey"},
    "AssignMaxSpToBlackboard": {"_targetType", "_blackboardKey"},
}
BLACKBOARD_READ_NODES = frozenset({"AssignCharacterSkillBlackboardToBB", "AssignBuffBlackboardFromOthers"}) | ENTITY_VALUE_READ_FIELDS.keys()

BLACKBOARD_CONDITION_FIELDS = {
    "FilterByExecuteBlackboardValue": {"_blackboardKey", "_valueToCompare", "_anotherKeyToCompare", "_condType"},
    "IsBlackboardEqualWithFloat": {"_var", "_compareValue"},
    "IsBlackboardEqualWithString": {"_var", "_compareValue", "_compareBBKey", "_useBuffBlackboard", "_buffKey"},
}


def blackboard_condition_issue(node: dict[str, Any]) -> str | None:
    name = node_name(node)
    if name not in BLACKBOARD_CONDITION_FIELDS or set(node) - (BLACKBOARD_CONDITION_FIELDS[name] | {"$type"}):
        return "fields"
    key = node.get("_blackboardKey" if name == "FilterByExecuteBlackboardValue" else "_var")
    if not isinstance(key, str) or not key:
        return "key"
    if name == "IsBlackboardEqualWithString" and (node.get("_useBuffBlackboard") or node.get("_buffKey")):
        return "external_buff_binding"
    other = node.get("_anotherKeyToCompare" if name == "FilterByExecuteBlackboardValue" else "_compareBBKey")
    if other is not None and not isinstance(other, str):
        return "comparison_key"
    if name == "FilterByExecuteBlackboardValue" and node.get("_condType") not in {"EQUALS", "GT", "GE", "LT", "LE"}:
        return "condition"
    return None


def blackboard_condition_value(node: dict[str, Any], board: dict) -> bool:
    """Evaluate scalar gates without changing the board or choosing an entity."""
    if issue := blackboard_condition_issue(node):
        raise ValueError(issue)
    name = node_name(node)
    key = node.get("_blackboardKey" if name == "FilterByExecuteBlackboardValue" else "_var")
    other = node.get("_anotherKeyToCompare" if name == "FilterByExecuteBlackboardValue" else "_compareBBKey")
    if key not in board or other and other not in board:
        return False
    left = board[key]
    right = (board[other] if other else node.get("_valueToCompare" if name == "FilterByExecuteBlackboardValue" else "_compareValue", 0))
    if name == "IsBlackboardEqualWithString":
        if not isinstance(left, str) or not isinstance(right, str):
            raise ValueError("string_value")
        return left == right
    try:
        valid = all(type(value) in (int, float) and math.isfinite(float(value)) for value in (left, right))
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("numeric_value")
    condition = node.get("_condType", "EQUALS")
    if condition == "GT": return left > right
    if condition == "GE": return left >= right
    if condition == "LT": return left < right
    if condition == "LE": return left <= right
    return left == right


def default_blackboard_values(node: dict[str, Any], board: dict) -> dict:
    """Prepare the entire default batch before any of its writes are visible."""
    if set(node) - {"$type", "_defaultSettings"} or not isinstance(node.get("_defaultSettings"), list):
        raise ValueError("settings")
    result = dict(board)
    for item in node["_defaultSettings"]:
        if not isinstance(item, dict) or set(item) - {"key", "val", "valStr", "overrideIfExists"}:
            raise ValueError("setting_fields")
        key, flag = item.get("key"), item.get("overrideIfExists", False)
        if not isinstance(key, str) or not key or type(flag) not in (bool, int) or flag not in (False, True):
            raise ValueError("key_or_override")
        value = item.get("valStr")
        if value is None:
            value = item.get("val", 0)
            try:
                valid = type(value) in (int, float) and math.isfinite(float(value))
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError("numeric_value")
        elif not isinstance(value, str):
            raise ValueError("string_value")
        if flag or key not in result:
            result[key] = value
    return result


def contains_entity_value_read(value: Any) -> bool:
    if isinstance(value, dict):
        return node_name(value) in ENTITY_VALUE_READ_FIELDS or any(contains_entity_value_read(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_entity_value_read(item) for item in value)
    return False


def blackboard_assignment_issue(node: dict[str, Any]) -> str | None:
    if set(node) - {"$type", "_blackboardKey", "_value", "_copyFromKey", "_assignString", "_stringValue", "_valueStr"}:
        return "fields"
    if not isinstance(node.get("_blackboardKey"), str) or not node["_blackboardKey"]:
        return "key"
    if node.get("_copyFromKey") is not None and not isinstance(node["_copyFromKey"], str):
        return "copy_key"
    flag = node.get("_assignString", False)
    if type(flag) not in (int, bool) or flag not in (False, True):
        return "string_flag"
    return None


def assigned_blackboard_value(node: dict[str, Any], board: dict) -> tuple[str, Any]:
    """A scalar copy preserves its value and never aliases mutable objects."""
    if issue := blackboard_assignment_issue(node):
        raise ValueError(issue)
    source = node.get("_copyFromKey")
    if source:
        if source not in board:
            raise ValueError("copy_source_binding")
        value = board[source]
    else:
        value = (node.get("_stringValue", node.get("_valueStr", "")) if node.get("_assignString")
                 else node.get("_value", 0))
    if node.get("_assignString"):
        if not isinstance(value, str):
            raise ValueError("string")
    else:
        if type(value) not in (int, float):
            raise ValueError("value")
        try:
            finite = math.isfinite(float(value))
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("value")
    return node["_blackboardKey"], value


def blackboard_read_issue(node: dict[str, Any]) -> str | None:
    name = node_name(node)
    fields = ({"$type"} | ENTITY_VALUE_READ_FIELDS[name] if name in ENTITY_VALUE_READ_FIELDS else
        {"$type", "_targetType", "_sourceBlackboardKey", "_targetBlackboardKey"}
        if name == "AssignCharacterSkillBlackboardToBB" else
        {"$type", "_targetType", "_buffKey", "_blackboardKey", "_valueKey", "_filterBuffSource", "_sourceType"})
    if set(node) - fields:
        return "fields"
    roles = {"SOURCE", "TARGET", "BUFF_OWNER", "BUFF_SOURCE", "MODIFIER_SOURCE", "MODIFIER_TARGET"}
    target = node.get("_targetType", "TARGET")
    if not isinstance(target, str) or target not in roles:
        return "target_role"
    keys = (("_sourceBlackboardKey", None), ("_targetBlackboardKey", node.get("_sourceBlackboardKey")))
    if name in ENTITY_VALUE_READ_FIELDS:
        keys = (("_blackboardKey", None),)
        if name == "AssignAttributeToBB":
            if node.get("_attributeType") not in {"MAX_HP", "ATK", "DEF", "MAGIC_RESISTANCE", "MOVE_SPEED",
                                                "ONE_MINUS_STATUS_RESISTANCE", "SLOW_DOWN", "EP_BREAK_RECOVER_SPEED"}:
                return "attribute"
            if node.get("_scaleVar") is not None and not isinstance(node["_scaleVar"], str):
                return "scale_key"
    if name == "AssignBuffBlackboardFromOthers":
        flag = node.get("_filterBuffSource", False)
        if type(flag) not in (bool, int) or flag not in (False, True):
            return "source_filter"
        if flag and (not isinstance(node.get("_sourceType", "TARGET"), str) or node.get("_sourceType", "TARGET") not in roles):
            return "source_role"
        keys = (("_buffKey", None), ("_blackboardKey", None), ("_valueKey", node.get("_blackboardKey")))
    if any(not isinstance(node.get(key, default), str) or not node.get(key, default) for key, default in keys):
        return "keys"
    return None


FINISH_BUFF_FIELDS = {"_targetType", "_buffKey", "_loadFromBlackboard", "_decCntIfStack",
    "_updateOverrideMap", "_checkBuffSource", "_sourceType", "_alsoClearNullSource",
    "_finishHostBuff", "_excludeThisBuff", "_decCntKey", "_decCnt"}


def read_blackboard_value(node: dict[str, Any], context: "BehaviorContext", current_board: dict,
                          *, entity_value_reader=None) -> tuple[bool, str, Any]:
    """Read a scalar snapshot without guessing among multiple Buff sources."""
    if issue := blackboard_read_issue(node):
        raise ValueError(issue)
    entity = context.entity(node.get("_targetType"))
    if entity is None:
        raise ValueError("entity_binding")
    if node_name(node) in ENTITY_VALUE_READ_FIELDS:
        output = node["_blackboardKey"]
        if context.battle is None:
            raise ValueError("attribute_reader_binding")
        reader = entity_value_reader or context.battle.behavior_entity_value
        value = reader(entity, node_name(node), node.get("_attributeType"))
        scale = node.get("_scaleVar")
        if scale:
            if scale not in current_board:
                raise ValueError("scale_binding")
            amount = current_board[scale]
            try:
                valid = type(amount) in (int, float) and math.isfinite(float(amount))
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError("scale_value")
            value *= amount
        board, key = {output: value}, output
    elif node_name(node) == "AssignCharacterSkillBlackboardToBB":
        board = getattr(entity, "skill_blackboard", None)
        key = node["_sourceBlackboardKey"]
        output = node.get("_targetBlackboardKey", key)
    else:
        key, output = node["_blackboardKey"], node.get("_valueKey", node["_blackboardKey"])
        buff_key = node["_buffKey"]
        legacy = (getattr(entity, "buff_blackboards", None) or {}).get(buff_key)
        source = context.entity(node.get("_sourceType")) if node.get("_filterBuffSource") else None
        if node.get("_filterBuffSource") and source is None:
            raise ValueError("source_binding")
        if legacy is not None and node.get("_filterBuffSource"):
            raise ValueError("legacy_source_binding")
        container = getattr(entity, "buffs", None)
        instances = [item for item in getattr(container, "active", ())
            if item.enabled and not item.finished and not container._is_pending_finish(item)
            and item.definition.key == buff_key and (source is None or item.source is source)]
        if len(instances) > 1 or legacy is not None and instances:
            raise ValueError("multiple_sources")
        board = legacy
        if instances:
            instance = instances[0]
            board = current_board if instance is context.buff_instance else instance.blackboard
    if board is None:
        return False, output, None
    if not isinstance(board, dict):
        raise ValueError("payload")
    if key not in board:
        return False, output, None
    value = board[key]
    if not isinstance(value, str):
        if type(value) not in (int, float):
            raise ValueError("scalar_value")
        try:
            finite = math.isfinite(float(value))
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("numeric_value")
    return True, output, value


def no_source_damage_value(node: dict[str, Any], board: dict) -> float:
    """Reviewed NoSourceDamage parameters; never infer advanced shared flags."""
    allowed = {'$type', '_damageType', '_damageKey', '_ignoreForSp', '_damageWithoutModify',
        '_attackType', '_isEnvDamage', '_isUndeadable', '_ignoreCancelReasonMask',
        '_instantKillLikeDamage', '_isNotChangeableValue', '_multiplierByKey', '_multiplierKey', '_modifierKey'}
    if set(node) - allowed:
        raise ValueError('fields')
    if node.get('_damageType', 'PHYSICAL') not in {'PHYSICAL', 'MAGICAL', 'ARTS', 'PURE', 'TRUE', 'ELEMENT', 'ELEMENTAL'}:
        raise ValueError('damage_type')
    if node.get('_attackType', 'NORMAL') not in {'NONE', 'NORMAL', 'SPLASH', 'BUFF', 'ADDITION'}:
        raise ValueError('attack_type')
    for key in ('_ignoreForSp', '_damageWithoutModify', '_isEnvDamage', '_isUndeadable',
                '_instantKillLikeDamage', '_isNotChangeableValue', '_multiplierByKey'):
        flag = node.get(key, False)
        if type(flag) not in (bool, int) or flag not in (False, True):
            raise ValueError(key)
    for key in ('_damageWithoutModify', '_isEnvDamage', '_instantKillLikeDamage', '_isNotChangeableValue'):
        if node.get(key):
            raise ValueError(key + '_binding')
    if node.get('_ignoreCancelReasonMask', 'NONE') != 'NONE':
        raise ValueError('cancel_mask_binding')
    key = node.get('_damageKey', 'damage')
    if not isinstance(key, str) or not key or not isinstance(node.get('_modifierKey', ''), str):
        raise ValueError('keys')
    values = [board.get(key, 0), board.get('atk_scale', 1)]
    if any(type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0 for value in values):
        raise ValueError('numeric_value')
    if board.get('environment_dmage_flag', False):
        raise ValueError('environment_flag_binding')
    if node.get('_multiplierByKey'):
        key = node.get('_multiplierKey', 'cnt')
        if not isinstance(key, str) or not key:
            raise ValueError('multiplier_key')
        count = board.get(key, 1)
        if type(count) is not int or not 0 <= count <= 2147483647:
            raise ValueError('integer_multiplier_binding')
        values.append(count)
    amount = math.prod(values)
    if not math.isfinite(amount):
        raise ValueError('overflow')
    return amount


def passive_buff_spec_issue(spec: dict[str, Any]) -> str | None:
    """Keep the first registered slice limited to passive, single-stack flags."""
    if not isinstance(spec, dict):
        return "buff_payload"
    attributes = spec.get("attributes") or {}
    if not isinstance(attributes, dict):
        return "attributes_payload"
    if set(attributes) - {"attributeModifiers", "abnormalFlags", "abnormalImmunes", "abnormalAntis",
                          "abnormalCombos", "abnormalComboImmunes"}:
        return "attributes_variant"
    passive = {"CAMOUFLAGE", "INVISIBLE", "UNTARGETABLE", "INVINCIBLE",
               "HEAL_FREE", "SP_BLOCKED", "STUN", "STUNNED", "FROZEN", "UNDEADABLE",
               "LEVITATE", "GROUND_BOUND", "FORCE_LEVITATE"}
    issues = [key for key in ("loadFromDB", "isDurableBuff",
        "isFreezable", "isLevitatable",
        "isGroundBoundable", "disableOverride", "overrideKey", "stripBlackboardParamsWithBuffKey")
        if spec.get(key)]
    issues += [key for key in ("attributeModifiers", "abnormalImmunes", "abnormalAntis",
        "abnormalCombos", "abnormalComboImmunes") if attributes.get(key)]
    issues += [str(flag) for flag in (attributes.get("abnormalFlags") or []) if flag not in passive]
    for key in ("isSilenceable", "isStunnable", "isDamageMissable"):
        if type(spec.get(key, False)) not in (bool, int) or spec.get(key, False) not in (0, 1):
            issues.append(key)
        elif key != "isDamageMissable" and spec.get(key) and spec.get("lifeTimeType", "INFINITY") not in {"INFINITY", "LIMITED"}:
            issues.append("responsive_lifetime")
    if spec.get("maxStackCnt", 1) != 1:
        issues.append("maxStackCnt")
    for key in ("refreshRemainingTimeWhenStackMax", "clearAllStackCntWhenTimeUp",
                "durationKey", "remainingTimeKey"):
        if spec.get(key):
            issues.append(key)
    if spec.get("maxValidStackCnt", -1) not in {-1, 1}:
        issues.append("maxValidStackCnt")
    if spec.get("overrideType", "DEFAULT") != "DEFAULT" or spec.get("independentCharacterSource"):
        issues.append("override/source")
    if spec.get("statusResistable", "AUTOMATIC") not in {"NO", "YES", "AUTOMATIC"}:
        issues.append("statusResistable")
    for key in ("takeSnapshotWhenExtend", "overrideOnEventPriority",
                "priorityBBKeys", "enableInitDirectionFromSource"):
        if spec.get(key):
            issues.append(key)
    for key in ('audioSignal', 'overrideEffectKey'):
        if spec.get(key) is not None and not isinstance(spec[key], str):
            issues.append(key)
    if spec.get("priority", -1) != -1 or spec.get("onEventPriority", "DEFAULT") != "DEFAULT":
        issues.append("priority")
    if spec.get("triggerLifeType", "IMMEDIATELY") != "IMMEDIATELY" or spec.get("triggerCnt", 0):
        issues.append("triggerLifeType/triggerCnt")
    if spec.get("firstTriggerInterval", -1) != -1:
        issues.append("firstTriggerInterval")
    if spec.get("lifeTimeType", "INFINITY") not in {"INFINITY", "LIMITED", "IMMEDIATELY"}:
        issues.append("lifeTimeType")
    return ",".join(issues) if issues else None


def hit_buff_spec_issue(spec: dict[str, Any], allowed: frozenset[str]) -> str | None:
    """Validate only the built-in hit statuses reviewed by the caller."""
    if not spec.get("loadFromDB") or spec.get("buffKey") not in allowed:
        return "unsupported_builtin"
    if spec.get("isDamageMissable"):
        return "projectile_miss_binding"
    if any((spec.get("attributes") or {}).values()):
        return "attributes"
    copied = {**spec, "loadFromDB": False, "durationKey": None}
    issue = passive_buff_spec_issue(copied)
    if issue:
        return issue
    if spec.get("lifeTimeType") != "LIMITED" or spec.get("templateKey", "empty") != "empty":
        return "lifetime/template"
    if spec.get("statusResistable", "AUTOMATIC") != "AUTOMATIC":
        return "statusResistable"
    if spec.get("overrideType", "DEFAULT") != "DEFAULT" or spec.get("independentCharacterSource"):
        return "override/source"
    return None


def registered_buff_spec_issue(spec: dict[str, Any]) -> str | None:
    """Reviewed single-stack attributes; referenced actions need preflight."""
    if not isinstance(spec, dict):
        return "buff_payload"
    fields = BUFF_DATA_FIELDS
    if set(spec) - fields:
        return "buff_fields"
    blackboard = spec.get("blackboard") or []
    if not isinstance(blackboard, list) or any(not isinstance(item, dict)
            or not isinstance(item.get("key"), str) or not item["key"] for item in blackboard):
        return "blackboard_payload"
    attributes = spec.get("attributes") or {}
    if not isinstance(attributes, dict):
        return "attributes_payload"
    policy = spec.get("overrideType", "DEFAULT")
    if not isinstance(policy, str):
        return "override:policy"
    explicit_priority = policy == "DEFAULT" and type(spec.get("priority")) is int and spec["priority"] >= 0
    key_priority = policy == "DEFAULT" and spec.get("priority", -1) == -1 and bool(spec.get("priorityBBKeys"))
    attribute_priority = (policy == "DEFAULT" and type(spec.get("priority", -1)) is int
        and spec.get("priority", -1) < 0 and not spec.get("priorityBBKeys")
        and bool(attributes.get("attributeModifiers")))
    if (spec.get("isSilenceable") or spec.get("isStunnable")) and (
            spec.get("lifeTimeType", "INFINITY") not in {"INFINITY", "LIMITED"} or
            not spec.get("disableOverride", False) and (
                policy != "DEFAULT" or explicit_priority or key_priority or attribute_priority)):
        return "responsive_lifetime/override_binding"
    if (policy in {"STACK", "UNIQUE", "EXTEND", "EXTEND_TIME"} or explicit_priority or key_priority
            or attribute_priority or spec.get("disableOverride", False)):
        if (not spec.get("disableOverride", False) and policy in {"STACK", "EXTEND", "EXTEND_TIME"}
                and (spec.get("triggerInterval", -1) != -1
                     or spec.get("triggerLifeType", "IMMEDIATELY") != "IMMEDIATELY")):
            return "override:trigger_clock"
        if policy in {"EXTEND", "EXTEND_TIME"} and spec.get("lifeTimeType") != "LIMITED":
            return "override:lifetime"
        if explicit_priority and spec["priority"] > 2147483647:
            return "override:priority_lifetime"
        if key_priority:
            keys = spec["priorityBBKeys"]
            if not isinstance(keys, list) or any(not isinstance(key, str) or not key for key in keys):
                return "override:priority_keys"
            spec = {**spec, "priorityBBKeys": []}
        if policy == "STACK":
            cap = spec.get("maxStackCnt", 1)
            if type(cap) is not int or not 1 <= cap <= 2147483647 or spec.get("maxValidStackCnt", -1) != -1:
                return "stack:count"
            flag = spec.get("clearAllStackCntWhenTimeUp", False)
            if type(flag) not in (bool, int) or flag not in (False, True):
                return "stack:clear_flag"
            spec = {**spec, "maxStackCnt": 1, "clearAllStackCntWhenTimeUp": False}
        if type(spec.get("disableOverride", False)) not in (bool, int) or spec.get("disableOverride", False) not in (False, True):
            return "override:disable_flag"
        if spec.get("overrideKey") is not None and not isinstance(spec["overrideKey"], str):
            return "override:key"
        spec = {**spec, "overrideType": "DEFAULT", "overrideKey": None, "disableOverride": False,
                "priority": -1 if explicit_priority or attribute_priority else spec.get("priority", -1)}
    interval = spec.get("triggerInterval", -1)
    if type(interval) not in (int, float) or not math.isfinite(interval) or (interval != -1 and interval <= 0):
        return "triggerInterval"
    first = spec.get("firstTriggerInterval", -1)
    clock_from_board = interval == -1 and spec.get("triggerLifeType") in {"LIMITED", "INFINITY"}
    if (type(first) not in (int, float) or not math.isfinite(first) or first < -1
            or first != -1 and (first < 0 or interval <= 0 and not clock_from_board)):
        return "firstTriggerInterval"
    spec = {**spec, "firstTriggerInterval": -1}
    trigger_life = spec.get("triggerLifeType", "IMMEDIATELY")
    trigger_count = spec.get("triggerCnt", 0)
    if trigger_life == "LIMITED":
        if type(trigger_count) is not int or not 0 <= trigger_count <= 2147483647 or interval <= 0 and not clock_from_board:
            return "trigger_count/clock"
    elif trigger_life not in {"IMMEDIATELY", "INFINITY"} or trigger_count != 0:
        return "triggerLifeType/triggerCnt"
    # Container semantics are validated above; flag-only checks stay narrow.
    spec = {**spec, "triggerLifeType": "IMMEDIATELY", "triggerCnt": 0}
    modifiers = attributes.get("attributeModifiers") or []
    if not isinstance(modifiers, list):
        return "attributeModifiers:payload"
    if not modifiers:
        return passive_buff_spec_issue({**spec, "durationKey": None,
                                        "stripBlackboardParamsWithBuffKey": False})
    if not 1 <= len(modifiers) <= 16 or not all(isinstance(m, dict) for m in modifiers):
        return "attributeModifiers:count"
    four_pools = {"ADDITION", "MULTIPLIER", "FINAL_ADDITION", "FINAL_SCALER"}
    formulas = {"MAX_HP": four_pools,
                "ATK": four_pools, "DEF": four_pools,
                "ATTACK_SPEED": {"ADDITION"}, "MAGIC_RESISTANCE": {"ADDITION"},
                "MOVE_SPEED": four_pools,
                "HP_RECOVERY_PER_SEC": {"ADDITION"},
                "HP_RECOVERY_PER_SEC_BY_MAX_HP_RATIO": {"ADDITION"},
                "SP_RECOVERY_PER_SEC": {"ADDITION"},
                "ONE_MINUS_STATUS_RESISTANCE": four_pools,
                "SLOW_DOWN": four_pools,
                "EP_BREAK_RECOVER_SPEED": four_pools}
    used = set()
    for modifier in modifiers:
        attribute = modifier.get("attributeType")
        formula = modifier.get("formulaItem")
        if attribute not in formulas or formula not in formulas[attribute] or (attribute, formula) in used:
            return "attributeModifiers:attribute/formula"
        used.add((attribute, formula))
        if modifier.get("fetchBaseValueFromSourceEntity"):
            return "attributeModifiers:source_base"
        if set(modifier) - {"attributeType", "formulaItem", "value", "loadFromBlackboard",
                             "fetchBaseValueFromSourceEntity"}:
            return "attributeModifiers:variant"
    if spec.get("lifeTimeType") not in {"LIMITED", "INFINITY"}:
        return "attributeModifiers:lifetime"
    if spec.get("overrideType", "DEFAULT") != "DEFAULT" or spec.get("independentCharacterSource"):
        return "attributeModifiers:override/source"
    if spec.get("statusResistable", "AUTOMATIC") not in {"NO", "YES", "AUTOMATIC"}:
        return "attributeModifiers:status/trigger"
    if spec.get("triggerInterval", -1) != -1 and spec.get("templateKey", "empty") == "empty":
        return "attributeModifiers:status/trigger"
    for key in ("takeSnapshotWhenExtend", "overrideOnEventPriority",
                "priorityBBKeys", "enableInitDirectionFromSource"):
        if spec.get(key):
            return "attributeModifiers:" + key
    if spec.get("priority", -1) != -1 or spec.get("onEventPriority", "DEFAULT") != "DEFAULT":
        return "attributeModifiers:priority"
    copied = {**spec, "attributes": {**(spec.get("attributes") or {}), "attributeModifiers": []},
              "durationKey": None, "stripBlackboardParamsWithBuffKey": False}
    return passive_buff_spec_issue(copied)


def projectile_buff_spec_issue(spec: dict[str, Any]) -> str | None:
    return hit_buff_spec_issue(spec, frozenset({"sluggish"}))


def range_target_options_issue(options: dict[str, Any], purposes: frozenset[str]) -> str | None:
    if options.get("targetSide") not in {"ALLY", "ENEMY"}:
        return "target_side"
    if options.get("targetMotion", "ALL") not in {"ALL", "WALK", "FLY"}:
        return "target_motion"
    if options.get("targetCategory", "DEFAULT") != "DEFAULT":
        return "target_category"
    if options.get("enableAdvancedOptions"):
        for key in ("ignoreMotionMode", "ignoreTargetSide", "excludeSomeAbnormalFlags",
                    "containSomeAbnormalFlags", "checkUnitType", "onlyIgnoreSomeOfTargetFreeCase"):
            if options.get(key):
                return key
        if options.get("purposeMask", "NONE") not in purposes:
            return "purposeMask"
        if options.get("professionMask", "NONE") != "NONE":
            return "professionMask"
    return None


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
    modifier: Any = None
    random_mechanism: str | None = None
    ability_family: str | None = None
    projectile_selector: Callable[[], Any] | None = None
    projectile_key: str | None = None
    cached_attack: float | None = None
    range_buff_sink: Callable[[Any, Any], None] | None = None
    register_buffs: bool = False
    refresh_attribute_buffs: bool = False

    def entity(self, target_type: str | None) -> Any:
        roles = {
            "SOURCE": self.source,
            "TARGET": self.target,
            "BUFF_OWNER": self.buff_owner,
            "BUFF_SOURCE": self.buff_source,
            "MODIFIER_SOURCE": self.modifier_source,
            "MODIFIER_TARGET": self.modifier_target,
        }
        role = "TARGET" if target_type is None or target_type == "" else str(target_type)
        return roles.get(role)


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
        buff_database: BuffDatabase | None = None,
    ) -> None:
        self.templates = dict(templates or {})
        self.buff_database = buff_database or BuffDatabase()
        self.strict = strict
        self.diagnostics = diagnostics or BehaviorDiagnostics()
        self.handlers: dict[str, NodeHandler] = {
            "NoSourceDamage": self._no_source_damage,
            "AdvancedApplyDamage": self._advanced_apply_damage,
            "AOEDamage": self._aoe_damage,
            "AssignBuffBlackboardFromOthers": self._assign_buff_blackboard,
            "AssignCharacterSkillBlackboardToBB": self._assign_skill_blackboard,
            "AssignAttributeToBB": self._assign_skill_blackboard,
            "AssignHpRatioToBB": self._assign_skill_blackboard,
            "AssignMaxSpToBlackboard": self._assign_skill_blackboard,
            "AssignValueToBB": self._assign_value,
            "AtkScaleUp": self._atk_scale_up,
            "BlackboardAdd": self._blackboard_add,
            "CalculateBlackboardValueViaParams": self._calculate_blackboard,
            "CheckContainsBuff": self._check_contains_buff,
            "CheckContainsDerviedBuff": self._check_contains_derived_buff,
            "CheckBlocked": self._check_blocked,
            "CheckEnemyLevelMask": self._check_enemy_level_mask,
            "CheckTargetTokenOrHost": self._check_token_host,
            "CheckIfAtMinusHpState": self._negative_hp_state,
            "CheckAbnormalFlag": self._abnormal_condition,
            "CheckAbnormalImmune": self._abnormal_condition,
            "CreateBuff": self._create_buff,
            "CreateEffect": self._create_effect,
            "CreateBuffInRange": self._create_buff_in_range,
            "DamageScale": self._damage_scale,
            "DamageViaAttr": self._damage_via_attr,
            "DamageViaMaxHpRatio": self._damage_via_max_hp_ratio,
            "Dice": self._dice,
            "EmitProjectileUseAbilitySelector": self._emit_projectile,
            "FilterAbilityFamily": self._filter_ability_family,
            "FinishBuff": self._finish_buff,
            "FinishDerivedBuff": self._finish_derived,
            "FinishBuffsById": self._finish_buffs_by_id,
            "FinishAllBuffsWithCertainAbnormalFlag": self._finish_buffs_with_flag,
            "IfElse": self._if_else,
            "IfTarget": self._if_target,
            "IsBlackboardZero": self._is_blackboard_zero,
            "FilterByExecuteBlackboardValue": self._blackboard_condition,
            "IsBlackboardEqualWithFloat": self._blackboard_condition,
            "IsBlackboardEqualWithString": self._blackboard_condition,
            "EnsureBlackboardDefaultValue": self._ensure_blackboard_defaults,
            "IsHeal": self._is_heal,
            "IsTargetInEPBreakRecovery": self._in_element_recovery,
            "ModifySp": self._modify_sp,
            "LockCharacterSkillProgress": self._lock_skill_progress,
            "UnlockCharacterSkillProgress": self._lock_skill_progress,
            "Not": self._not_condition,
            "RandomSetter": self._random_setter,
            "RemainingRatioToAttributeModifier": self._remaining_ratio_modifier,
            "SetAtkScaleZero": self._set_atk_scale_zero,
            "SetMaxMinusHpRatio": self._negative_hp_state,
            "SwitchMode": self._switch_mode,
        }

    @property
    def supported_nodes(self) -> frozenset[str]:
        return frozenset(self.handlers) | self.FLOW_NODES

    def resolve_template(self, key: str) -> dict | None:
        template = self.templates.get(key)
        if template is not None:
            template = self.buff_database.resolve_tree(template)
            self.templates[key] = template
        return template

    def _unsupported(self, key: str) -> bool:
        self.diagnostics.unsupported_nodes[key] += 1
        if self.strict:
            raise UnknownNodeError(f"unsupported behavior {key}")
        return False

    @staticmethod
    def _unsupported_variant(node: dict[str, Any]) -> str | None:
        name = node_name(node)
        if name in {"LockCharacterSkillProgress", "UnlockCharacterSkillProgress"}:
            if set(node) - {"$type", "_targetType"}:
                return name + ":fields"
            if not isinstance(node.get("_targetType"), str) or node["_targetType"] not in {
                    "SOURCE", "TARGET", "BUFF_OWNER", "BUFF_SOURCE", "MODIFIER_SOURCE", "MODIFIER_TARGET"}:
                return name + ":entity_role"
        if name == "FinishAllBuffsWithCertainAbnormalFlag":
            if set(node) - {"$type", "_targetType", "_abnormalFlag"}:
                return name + ":fields"
            if not isinstance(node.get("_abnormalFlag"), str) or node["_abnormalFlag"] not in ABNORMAL_READ_FLAGS:
                return name + ":flag_binding"
            if not isinstance(node.get("_targetType"), str) or node["_targetType"] not in {
                    "SOURCE", "TARGET", "BUFF_OWNER", "BUFF_SOURCE", "MODIFIER_SOURCE", "MODIFIER_TARGET"}:
                return name + ":entity_role"
        if name in {"CheckAbnormalFlag", "CheckAbnormalImmune"}:
            fields = {"$type", "_abnormalFlag", "_targetType", "_isUnset"}
            if name == "CheckAbnormalFlag":
                fields.add("_abnormalFlagKey")
            if set(node) - fields or node.get("_abnormalFlagKey"):
                return name + ":fields/blackboard_variant"
            if not isinstance(node.get("_abnormalFlag"), str) or node["_abnormalFlag"] not in ABNORMAL_READ_FLAGS:
                return name + ":flag_binding"
            if not isinstance(node.get("_targetType"), str) or node["_targetType"] not in {"SOURCE", "TARGET", "BUFF_OWNER", "BUFF_SOURCE",
                                                "MODIFIER_SOURCE", "MODIFIER_TARGET"}:
                return name + ":entity_role"
            if type(node.get("_isUnset", False)) not in (bool, int) or node.get("_isUnset", False) not in (0, 1):
                return name + ":unset_flag"
        if name in {"SetMaxMinusHpRatio", "CheckIfAtMinusHpState"}:
            fields = {"$type", "_targetType"} | ({"_value"} if name == "SetMaxMinusHpRatio" else set())
            if set(node) - fields:
                return name + ":fields"
            if not isinstance(node.get("_targetType"), str) or node["_targetType"] not in {"SOURCE", "TARGET", "BUFF_OWNER", "BUFF_SOURCE",
                                                "MODIFIER_SOURCE", "MODIFIER_TARGET"}:
                return name + ":entity_role"
            if name == "SetMaxMinusHpRatio":
                value = node.get("_value")
                try:
                    valid = type(value) in (int, float) and math.isfinite(float(value)) and value >= 0
                except OverflowError:
                    valid = False
                if not valid:
                    return name + ":value"
        if name in BLACKBOARD_CONDITION_FIELDS and (issue := blackboard_condition_issue(node)):
            return name + ":" + issue
        if name == "EnsureBlackboardDefaultValue":
            try:
                default_blackboard_values(node, {})
            except ValueError as exc:
                return name + ":" + str(exc)
        if name == "CalculateBlackboardValueViaParams" and (issue := blackboard_calculation_issue(node)):
            return name + ":" + issue
        if name in BLACKBOARD_READ_NODES and (issue := blackboard_read_issue(node)):
            return name + ":" + issue
        roles = {"SOURCE", "TARGET", "BUFF_OWNER", "BUFF_SOURCE", "MODIFIER_SOURCE", "MODIFIER_TARGET"}
        valid_role = lambda value: isinstance(value, str) and value in roles
        if name in {"ModifySp", "CheckContainsBuff"} and not valid_role(node.get("_targetType", "TARGET")):
            return name + ":target_role"
        if (name == "CheckContainsBuff" and node.get("_checkBuffSource")
                and not valid_role(node.get("_buffSourceType", "TARGET"))):
            return name + ":source_role"
        if name == "AssignValueToBB" and (issue := blackboard_assignment_issue(node)):
            return name + ":" + issue
        if name == "BlackboardAdd":
            if node.get("_blackboardKeyFromBb") or node.get("_maxKey") or (
                    node.get("_outputKey") and node["_outputKey"] != node.get("_blackboardKey")):
                return name + ":key_or_clamp_binding"
            if node.get("_outputKey") is not None and not isinstance(node["_outputKey"], str):
                return name + ":output_key"
        if name == "AOEDamage":
            roles = {"SOURCE", "TARGET", "BUFF_OWNER", "BUFF_SOURCE", "MODIFIER_SOURCE", "MODIFIER_TARGET"}
            if not valid_role(node.get("_sourceType")) or not valid_role(node.get("_targetType")):
                return f"{name}:entity_binding"
            for key in ("_useAbilitySelector", "_abilityName", "_createEffect", "_hitEffectUseSourceFaceTo",
                        "_useDamageFromBB", "_isNoSourceDamage", "_checkTargetAlive", "_maxTargetCountKey",
                        "_useGlobalRange"):
                if node.get(key):
                    return f"{name}:{key}"
            if not node.get("_useAbilitySourceSide", True) or node.get("_filterType", "ALL") != "ALL":
                return f"{name}:side_or_filter"
            if node.get("_damageType", "PHYSICAL") not in {"PHYSICAL", "MAGICAL", "ARTS", "PURE", "TRUE", "ELEMENT", "ELEMENTAL"}:
                return f"{name}:damage_type"
            if node.get("_attackType", "NORMAL") not in {"NONE", "NORMAL", "SPLASH", "BUFF", "ADDITION"}:
                return f"{name}:attack_type"
            if node.get("_sourceApplyWay", "NONE") not in {"NONE", "MELEE", "RANGED"}:
                return f"{name}:apply_way"
            options = node.get("_targetOptions") or {"targetSide": "ENEMY"}
            issue = range_target_options_issue(options, frozenset({"NONE", "DAMAGE"}))
            if issue:
                return f"{name}:target_options:{issue}"
            for spec in node.get("_buffs") or []:
                issue = hit_buff_spec_issue(spec, frozenset({"stun", "sluggish"}))
                if issue:
                    return f"{name}:buff:{issue}"
        if name == 'CreateEffect':
            fields = {'$type', '_targetType', '_sourceType', '_effectKey', '_useSourceFaceVactor',
                '_useSourceToTargetDirection', '_useConstDirection', '_constDirection',
                '_createEffectHoldBySource', '_createEffectConstDirectionHoldBySource',
                '_useAttackPlaybackSpeed', '_manualSetPlaybackSpeed', '_blackboardKey', '_playbackSpeed'}
            if set(node) - fields:
                return name + ':fields'
            if not valid_role(node.get('_targetType')) or not valid_role(node.get('_sourceType')):
                return name + ':entity_role'
            if not isinstance(node.get('_effectKey'), str) or not node['_effectKey']:
                return name + ':effect_key'
            for key in fields - {'$type', '_targetType', '_sourceType', '_effectKey',
                    '_constDirection', '_blackboardKey', '_playbackSpeed'}:
                flag = node.get(key, False)
                if type(flag) not in (bool, int) or flag != 0:
                    return name + ':' + key
            if node.get('_blackboardKey') or node.get('_constDirection', 'UP') != 'UP':
                return name + ':visual_variant'
            if type(node.get('_playbackSpeed', 1)) not in (int, float) or node.get('_playbackSpeed', 1) != 1:
                return name + ':playback_speed'
        if name == "NoSourceDamage":
            try:
                no_source_damage_value(node, {})
            except (ValueError, OverflowError) as exc:
                return name + ":" + str(exc)
        if name == "AdvancedApplyDamage":
            if node.get("_sourceType") != "SOURCE" or node.get("_targetType") != "TARGET":
                return f"{name}:entity_binding"
            for key in ("_baseOnHostAtk", "_emitSourceOnCalculateDamage", "_forceUseProjectileCachedAtk",
                        "_ignoreForSp", "_assignRealDamageToBB", "_assignFinalDamageToBB", "_skipModifierEvent",
                        "_noSource", "_onlyUseSourceOnCalculateDamage", "_getCachedAtkFromBlackboard",
                        "_useSourceWhenUseCachedAtk", "_setSharedFlag"):
                if node.get(key):
                    return f"{name}:{key}"
            for key in ("_ignoreMissFlag", "_ignoreCancelReasonMask"):
                if node.get(key, "NONE") != "NONE":
                    return f"{name}:{key}"
            if node.get("_damageType", "PHYSICAL") not in {"PHYSICAL", "MAGICAL", "ARTS", "PURE", "TRUE", "ELEMENT", "ELEMENTAL"}:
                return f"{name}:damage_type"
            if node.get("_attackType", "NORMAL") not in {"NONE", "NORMAL", "SPLASH", "BUFF", "ADDITION"}:
                return f"{name}:attack_type"
            if node.get("_applyWay", "NONE") not in {"NONE", "MELEE", "RANGED"}:
                return f"{name}:apply_way"
        if name == "FilterAbilityFamily" and node.get("_familyGroupMask") not in {"ALL", "ATTACK"}:
            return f"{name}:family_mask"
        if name == "IsTargetInEPBreakRecovery" and node.get("_checkRecoveryType"):
            if node.get("_elementType") not in {"DARK", "FIRE", "WATER", "SANITY", "ANGER"}:
                return f"{name}:element_type"
        if name == "EmitProjectileUseAbilitySelector":
            if node.get("_sourceType") != "BUFF_OWNER" or node.get("_targetType") != "BUFF_OWNER":
                return f"{name}:selector_owner"
            for key in ("_hookStartPoint", "_abilityName", "_useAbilityFromOther", "_useProjectilePos", "_excludeTarget"):
                if node.get(key):
                    return f"{name}:{key}"
            for key in ("_overwriteActions", "_overwriteBuffs", "_extraBlackboard", "_overwriteBlackboard"):
                if not node.get(key):
                    return f"{name}:{key}"
            if node.get("_ev") != "ON_HIT_OBJECT":
                return f"{name}:event"
            for action in node.get("_actions") or []:
                if node_name(action) not in {"AdvancedApplyDamage", "IsTargetInEPBreakRecovery"}:
                    return f"{name}:hit_action:{node_name(action)}"
            if node.get("_useMountPointGroup"):
                group = node.get("_mountPointGroup") or {}
                if group.get("loopType") not in {"RANDOM", "SEQUENTIAL"} or not group.get("mountPoints"):
                    return f"{name}:mount_group"
            for spec in node.get("_buffDataList") or []:
                issue = projectile_buff_spec_issue(spec)
                if issue:
                    return f"{name}:buff:{issue}"
        if name == "CreateBuffInRange":
            if any(not isinstance(spec, dict) or (not spec.get("disableOverride", False) and (
                   spec.get("overrideType", "DEFAULT") != "DEFAULT"))
                   for spec in node.get("_buffs") or []):
                return name + ":override_action_binding"
            roles = {"SOURCE", "TARGET", "BUFF_OWNER", "BUFF_SOURCE", "MODIFIER_SOURCE", "MODIFIER_TARGET"}
            if not valid_role(node.get("_sourceType")) or not valid_role(node.get("_targetType")):
                return f"{name}:entity_binding"
            for key in ("_useCentainSide", "_useHostAsSource", "_excludeCurAtkTarget",
                        "_alwaysIncudeCurAtkTarget", "_alwaysIncludeSourceBlocker", "_randomTarget",
                        "_limitMaxTarget", "_useAttackRange", "_useCurrentModeRange", "_useRangeToShow",
                        "_useTargetRangeInsteadOfSource", "_filterByBuildableType", "_filterByTags",
                        "_checkGiantTrapAllLocateTiles", "_removeExcludeTargetBeforeShrink", "_filterTargets"):
                if node.get(key):
                    return f"{name}:{key}"
            if node.get("_rangeModeIndex", -1) != -1:
                return f"{name}:mode_index"
            if node.get("_filterType", "ALL") != "ALL":
                return f"{name}:filter_type"
            if node.get("_useRadius") and node.get("_useGlobalRange"):
                return f"{name}:range_modes"
            if node.get("_isDerivedBuff") and not node.get("_finishDerivedBuffIfParentFinish"):
                return f"{name}:derived_lifecycle"
            options = node.get("_targetOptions") or {}
            issue = range_target_options_issue(options, frozenset({"NONE"}))
            if issue:
                return f"{name}:target_options:{issue}"
            if not isinstance(node.get("_buffs"), list) or not node["_buffs"]:
                return f"{name}:empty_buffs"
            for spec in node.get("_buffs") or []:
                issue = registered_buff_spec_issue(spec)
                if issue:
                    return f"{name}:buff:{issue}"
        unsupported = {
            "AtkScaleUp": ("_filterApplyWay", "_filterProjectileKey", "_filterNoneApplyWay"),
            "CheckBlocked": ("_checkBlockedCount", "_checkBlockedBySourceToken"),
            "IfTarget": ("_checkApplyWay", "_checkTargetUnitType", "_checkTargetFree", "_checkMotionTargetFree"),
            "CheckContainsBuff": ("_checkSourceHost",),
        }
        for key in unsupported.get(name, ()):
            if node.get(key):
                return f"{name}:{key}"
        if name == "IfTarget" and node.get("_motionMask", "ALL") != "ALL":
            return "IfTarget:_motionMask"
        if name == "CheckContainsDerviedBuff":
            if not isinstance(node.get("_derviedBuffKey"), str) or not node["_derviedBuffKey"]:
                return name + ":key"
            if set(node) - {"$type", "_derviedBuffKey"}:
                return name + ":variant"
        if name == "CheckContainsBuff" and node.get("_checkSourceHost"):
            return name + ":source_host"
        if node_name(node) == "ModifySp":
            # Negative notifications and custom keys need their modifier paths
            # before those variants can be enabled.
            for key in ("_isMinis", "_customModifierKey"):
                if node.get(key):
                    return f"ModifySp:{key}"
        if node_name(node) == "CreateBuff":
            if (not isinstance(node.get("_buff") or {}, dict) or (not (node.get("_buff") or {}).get("disableOverride", False) and (
                    (node.get("_buff") or {}).get("overrideType", "DEFAULT") != "DEFAULT"))):
                return "CreateBuff:override_action_binding"
            if not valid_role(node.get("_buffOwner", "TARGET")) or (node.get("_useSpecialBuffSource")
                    and not valid_role(node.get("_specialBuffSource"))):
                return "CreateBuff:entity_role"
            if bool(node.get("_isDerivedBuff")) != bool(node.get("_finishDerivedBuffIfParentFinish")):
                return "CreateBuff:derived_lifecycle"
            roles = {"SOURCE", "TARGET", "BUFF_OWNER", "BUFF_SOURCE", "MODIFIER_SOURCE", "MODIFIER_TARGET"}
            if node.get("_isDerivedBuff") and (node.get("_buffOwner") not in roles
                    or node.get("_useSpecialBuffSource") and node.get("_specialBuffSource") not in roles):
                return "CreateBuff:derived_owner/source"
            for key in ("_inherateSourceProjectileFromParentBuff", "_hookEffectKeyFromSourceProjectile"):
                if node.get(key):
                    return "CreateBuff:" + key
            issue = registered_buff_spec_issue(node.get("_buff") or {})
            if issue:
                return f"CreateBuff:{issue}"
        if name == "FinishDerivedBuff":
            if set(node) - {"$type", "_updateOverrideMap"}:
                return name + ":fields"
            value = node.get("_updateOverrideMap", True)
            if type(value) not in (bool, int) or value != 1:
                return name + ":override_map_binding"
        if node_name(node) == "FinishBuff" and node.get("_decCntIfStack"):
            return "FinishBuff:decrement_stack"
        if name == "FinishBuffsById":
            if set(node) - (FINISH_BUFF_FIELDS | {"$type"}):
                return name + ":fields"
            if not valid_role(node.get("_targetType", "TARGET")):
                return name + ":entity_role"
            if not isinstance(node.get("_buffKey"), str) or not node["_buffKey"]:
                return name + ":key"
            flag = node.get("_loadFromBlackboard", False)
            if type(flag) not in (bool, int) or flag not in (False, True):
                return name + ":blackboard_flag"
            for key in ("_decCntIfStack", "_checkBuffSource", "_alsoClearNullSource", "_finishHostBuff", "_excludeThisBuff"):
                value = node.get(key, False)
                if type(value) not in (bool, int) or value not in (False, True) or value:
                    return name + ":" + key
            if node.get("_decCntKey") or type(node.get("_decCnt", 1)) is not int or node.get("_decCnt", 1) != 1:
                return name + ":decrement_binding"
            value = node.get("_updateOverrideMap", True)
            if type(value) not in (bool, int) or value not in (True,):
                return name + ":override_map_binding"
            if node.get("_sourceType") is not None and not valid_role(node["_sourceType"]):
                return name + ":source_role"
        return None

    def validate_event(self, template_key: str, event: str) -> bool:
        """Reject incomplete migrated events before any actions take effect."""
        template = self.resolve_template(template_key)
        if template is None:
            self.diagnostics.missing_templates[template_key] += 1
            if self.strict:
                raise BehaviorError(f"unknown behavior template: {template_key}")
            return False
        if event not in template.get("eventToActions", {}):
            return self._unsupported(f"{template_key}:missing_event:{event}")

        return self.validate_actions(template.get("eventToActions", {}).get(event, []), f"{template_key}:{event}")

    def event_action(self, template_key: str, event: str, action_type: str) -> dict[str, Any] | None:
        """Select one explicitly migrated component, without claiming its whole event."""
        template = self.resolve_template(template_key)
        if template is None:
            self.validate_event(template_key, event)
            return None
        events = template.get("eventToActions") or {}
        if event not in events:
            self._unsupported(f"{template_key}:missing_event:{event}")
            return None
        matches = [node for node in events[event] or [] if node_name(node) == action_type]
        if len(matches) != 1:
            self._unsupported(f"{template_key}:{event}:{action_type}:component_count")
            return None
        action = matches[0]
        if not self.validate_actions([action], f"{template_key}:{event}:component:{action_type}"):
            return None
        return action

    def validate_actions(self, actions: Any, label: str) -> bool:
        """Preflight nested projectile callbacks as well as immediate actions."""
        actions = self.buff_database.resolve_tree(actions)
        def visit(value: Any) -> bool:
            valid = True
            if isinstance(value, dict):
                if "$type" in value:
                    name = node_name(value)
                    issue = name if name not in self.supported_nodes else self._unsupported_variant(value)
                    if issue:
                        valid = self._unsupported(f"{label}:{issue}")
                for child in value.values():
                    valid = visit(child) and valid
            elif isinstance(value, list):
                for child in value:
                    valid = visit(child) and valid
            return valid

        return visit(actions)

    def dispatch(
        self, template_key: str, event: str, context: BehaviorContext
    ) -> bool:
        template = self.resolve_template(template_key)
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
        self, actions: list[dict[str, Any]] | None, context: BehaviorContext
    ) -> bool:
        """Execute a linear action list.

        A false predicate gates later ordinary nodes. ``IfNot`` can invert that
        state and ``AlwaysNext`` restores it, matching the control markers found
        in real templates.
        """
        state = True
        for node in actions or ():
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
        node = self.buff_database.resolve_tree(node)
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
        branch = node.get("_succeedNodes" if passed else "_failNodes") or []
        return self.execute_actions(branch, context)

    def _if_target(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        issue = self._unsupported_variant(node)
        if issue:
            return self._unsupported(issue)
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
    def _dice(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        if context.battle is None:
            return False
        return context.battle._roll_probability(float(context.blackboard.get(node.get("_probKey", "prob"), 0)),
            source=context.buff_source or context.source, target=context.target,
            mechanism=context.random_mechanism or f"behavior:{context.template_key}:{context.event}:dice")

    def _filter_ability_family(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        issue = self._unsupported_variant(node)
        if issue:
            return self._unsupported(issue)
        mask = node.get("_familyGroupMask")
        return mask == "ALL" or context.ability_family == mask

    def _emit_projectile(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        issue = self._unsupported_variant(node)
        if issue:
            return self._unsupported(issue)
        return bool(context.battle and context.battle.behavior_emit_projectile(context, node))

    def _in_element_recovery(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        from .mechanics import ElementType
        issue = self._unsupported_variant(node)
        if issue:
            return self._unsupported(issue)
        target = context.entity(node.get("_target", "TARGET"))
        active = getattr(getattr(target, "elements", None), "active_break", None)
        return active is not None and (not node.get("_checkRecoveryType") or active is ElementType(node["_elementType"]))

    def _advanced_apply_damage(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        issue = self._unsupported_variant(node)
        if issue:
            return self._unsupported(issue)
        return bool(context.battle and context.battle.behavior_advanced_damage(context, node))

    def _random_setter(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        key = str(node.get("_targetKey", ""))
        board = context.blackboard
        if (not key or context.battle is None or key + "_min" not in board
                or key + "_max" not in board):
            return False
        value = context.battle.random_uniform(float(board[key + "_min"]),
            float(board[key + "_max"]), source=context.buff_source or context.source,
            mechanism=context.random_mechanism or f"behavior:{context.template_key}:{context.event}:{key}")
        board[key] = math.floor(value) if node.get("_convertToInt") else value
        return True

    def _is_heal(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        from .mechanics import DamageType
        return (context.modifier is not None
                and context.modifier.normalized_damage_type() is DamageType.HEAL)

    def _modify_sp(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        issue = self._unsupported_variant(node)
        if issue:
            return self._unsupported(issue)
        if context.battle is None:
            return False
        return bool(context.battle.behavior_modify_sp(context, node))

    def _negative_hp_state(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        if issue := self._unsupported_variant(node):
            return self._unsupported(issue)
        if context.battle is None:
            return self._unsupported(node_name(node) + ":battle_binding")
        return context.battle.behavior_negative_hp_state(context, node)

    def _lock_skill_progress(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        if issue := self._unsupported_variant(node):
            return self._unsupported(issue)
        if context.battle is None:
            return self._unsupported(node_name(node) + ":battle_binding")
        return context.battle.behavior_skill_progress(context, node)

    def _abnormal_condition(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        if issue := self._unsupported_variant(node):
            return self._unsupported(issue)
        if context.battle is None:
            return self._unsupported(node_name(node) + ":battle_binding")
        return context.battle.behavior_abnormal_condition(context, node)

    def _assign_value(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        if issue := self._unsupported_variant(node):
            return self._unsupported(issue)
        try:
            key, value = assigned_blackboard_value(node, context.blackboard)
        except ValueError as exc:
            return self._unsupported("AssignValueToBB:" + str(exc))
        context.blackboard[key] = value
        return True


    def _blackboard_add(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        if issue := self._unsupported_variant(node):
            return self._unsupported(issue)
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
        try:
            passed, key, value = calculated_blackboard_value(node, context.blackboard)
        except ValueError as exc:
            return self._unsupported("CalculateBlackboardValueViaParams:" + str(exc))
        if passed:
            context.blackboard[key] = value
        return passed


    def _is_blackboard_zero(
        self, node: dict[str, Any], context: BehaviorContext
    ) -> bool:
        key = str(node.get("_var", ""))
        if key not in context.blackboard:
            return False
        return abs(float(context.blackboard[key])) <= 1e-9

    def _blackboard_condition(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        try:
            return blackboard_condition_value(node, context.blackboard)
        except ValueError as exc:
            return self._unsupported(node_name(node) + ":" + str(exc))

    def _ensure_blackboard_defaults(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        try:
            values = default_blackboard_values(node, context.blackboard)
        except ValueError as exc:
            return self._unsupported("EnsureBlackboardDefaultValue:" + str(exc))
        context.blackboard.update(values)
        return True

    def _assign_skill_blackboard(
        self, node: dict[str, Any], context: BehaviorContext
    ) -> bool:
        try:
            passed, key, value = read_blackboard_value(node, context, context.blackboard)
        except ValueError as exc:
            return self._unsupported(node_name(node) + ":" + str(exc))
        if passed:
            context.blackboard[key] = value
        return passed

    def _assign_buff_blackboard(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        return self._assign_skill_blackboard(node, context)


    # --------------------------------------------------------- predicates
    def _check_contains_derived_buff(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        if context.buff_instance is None or context.buff_container is None:
            return self._unsupported("CheckContainsDerviedBuff:parent_binding")
        return context.buff_container.contains_derived(context.buff_instance, node.get("_derviedBuffKey"))

    def _check_blocked(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        issue = self._unsupported_variant(node)
        if issue:
            return self._unsupported(issue)
        target = context.entity(node.get("_targetType"))
        blocker = getattr(target, "blocked_by", None)
        if blocker is None:
            return False
        return not node.get("_checkBlockedBySource") or blocker is context.entity(node.get("_sourceType"))

    def _check_contains_buff(
        self, node: dict[str, Any], context: BehaviorContext
    ) -> bool:
        if issue := self._unsupported_variant(node):
            return self._unsupported(issue)
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
                bool(container.contains(str(key), source=source)) or
                (not node.get("_checkBuffSource") and str(key) in boards) for key in keys
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
        issue = self._unsupported_variant(node)
        if issue:
            return self._unsupported(issue)
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

    def _create_effect(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        if issue := self._unsupported_variant(node):
            return self._unsupported(issue)
        # Native CreateEffect only asks the visual controller to create a
        # transient effect. The simulator records its omission explicitly.
        if context.battle is None or context.entity(node.get('_targetType')) is None:
            return False
        context.battle.mechanic_diagnostics['TEMPLATE_VISUAL_EFFECT_NOT_RENDERED'] = 1
        return True

    def _no_source_damage(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        try:
            amount = no_source_damage_value(node, context.blackboard)
        except (ValueError, OverflowError) as exc:
            return self._unsupported('NoSourceDamage:' + str(exc))
        return context.battle is not None and context.battle.behavior_no_source_damage(context, node, amount)

    def _aoe_damage(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        issue = self._unsupported_variant(node)
        if issue:
            return self._unsupported(issue)
        if context.battle is None:
            return False
        return bool(context.battle.behavior_aoe_damage(context, node))

    # --------------------------------------------------------------- buffs
    def _create_buff(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        if issue := self._unsupported_variant(node):
            return self._unsupported(issue)
        if context.battle is None:
            return False
        return bool(context.battle.behavior_create_buff(context, node))

    def _create_buff_in_range(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        issue = self._unsupported_variant(node)
        if issue:
            return self._unsupported(issue)
        if context.battle is None:
            return False
        return bool(context.battle.behavior_create_buff_in_range(context, node))

    def _finish_buff(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        if context.battle is None or not context.current_buff_key:
            return False
        if context.buff_instance is not None and context.buff_container is not None:
            return context.buff_container.finish(context.buff_instance)
        return bool(
            context.battle.behavior_finish_buff(
                context.buff_owner, context.current_buff_key
            )
        )

    def _finish_derived(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        if issue := self._unsupported_variant(node):
            return self._unsupported(issue)
        if context.buff_instance is None or context.buff_container is None:
            return False
        return context.buff_container.finish_derived(context.buff_instance)

    def _check_token_host(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        host = context.entity(node.get("_hostType"))
        token = context.entity(node.get("_tokenType"))
        return host is not None and token is not None and (
            getattr(token, "summoner", None) is host or
            getattr(host, "summoner", None) is token)

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

    def _finish_buffs_with_flag(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        if issue := self._unsupported_variant(node):
            return self._unsupported(issue)
        if context.battle is None:
            return self._unsupported("FinishAllBuffsWithCertainAbnormalFlag:battle_binding")
        return context.battle.behavior_finish_flag_buffs(context, node)

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

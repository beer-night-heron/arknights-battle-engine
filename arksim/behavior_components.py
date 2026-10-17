"""Resource-driven passive attachment, independent of character identifiers.

Reviewed passive listeners reuse owner-local Buff clocks and attributes.
Other component classes and lifecycle events retain an explicit gap reason.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .behavior import (BehaviorInterpreter, BehaviorContext, node_name, registered_buff_spec_issue,
                       calculated_blackboard_value, assigned_blackboard_value, BLACKBOARD_READ_NODES, read_blackboard_value,
                       BLACKBOARD_CONDITION_FIELDS, blackboard_condition_value, default_blackboard_values,
                       ENTITY_VALUE_READ_FIELDS, FINISH_BUFF_FIELDS, no_source_damage_value)
from .loadout import select_candidate, select_abilities, module_phase


PASSIVE_BUFF_SCRIPT = 3800760148018415462
FORMAT = "arksim-behavior-components"
CALCULATE = "ON_CALCULATE_DAMAGE"
CACHED_CALCULATE = "ON_CALCULATE_CACHED_PROJECTILE_DAMAGE"
LIFECYCLE_EVENTS = {"ON_BUFF_START", "ON_OWNER_FINISH", "ON_BUFF_FINISH", "ON_BUFF_ENABLE", "ON_BUFF_DISABLE"}
PASSIVE_SKILL_SCRIPT = -2816356044270894278
SKILL_PROGRESS_HOOKER_SCRIPT = -8650025640945540263
TOGGLE_SKILL_SCRIPT = -2080331894047689788
TALENT_SCRIPT = -2726233351382997147
OUTPUT = "ON_OUTPUT_MODIFIER"
AFTER_ATTACK = "ON_AFTER_ATTACK"
_TALENT_FLAGS = {"_overwriteParentRangeId", "_overrideDefaultRangeId", "_applyTalentScale",
    "_influenceSkillBlackboard", "_fromRawDataToSkill", "_applyTalentRange", "_applyBlackboardBySkill",
    "_applyStrBlackboardBySkill", "_writeRangeIdToProjectileBlackboard", "_affectWhenNotRootTalent",
    "_scaleCertainKeyFlag", "_scaleCertainKeyList", "_scaleCertainMode", "_scaleModeIndices",
    "_extraAttackBlackboardModeIndices"}
_TALENT_FIELDS = _TALENT_FLAGS | {"m_GameObject", "m_Enabled", "m_Script", "m_Name", "_overwriteTalentKey",
    "_attachInDummy", "_checkValidBySkillIndex", "_defaultModeIndex", "_attackBlackboardModeIndex"}


def compile_operator_resources(exports: dict, character_keys: set[str]) -> dict:
    """Follow Transform ancestry and explicit mode references for talents.

    Exports are separated by character: path IDs need not be globally unique.
    The prefab-key/name correspondence is kept as a static candidate, not an
    assertion about native initialization or mode handoff timing.
    """
    result = {"format": FORMAT, "version": 1, "resource_kinds": ["operator"],
              "resources": {}, "actors": {}}
    for char_id, objects in exports.items():
        if char_id not in character_keys:
            continue
        roots = [(oid, o) for oid, o in objects.items() if o.get("m_Name") == char_id and "m_Component" in o]
        if len(roots) != 1:
            raise ValueError(f"ambiguous operator root: {char_id}")
        root_id, root = roots[0]
        def resolve(ref):
            return objects.get(str((ref or {}).get("m_PathID"))) if (ref or {}).get("m_FileID") == 0 else None
        controllers = [c for r in root["m_Component"] if (c := resolve(r.get("component"))) is not None and "_modes" in c]
        if len(controllers) != 1:
            raise ValueError(f"ambiguous operator controller: {char_id}")
        transforms = {}
        for oid, obj in objects.items():
            if "m_Father" in obj and "m_Children" in obj:
                go_ref = obj.get("m_GameObject") or {}
                if go_ref.get("m_FileID") == 0:
                    go_id = str(go_ref.get("m_PathID"))
                    if go_id in transforms:
                        raise ValueError(f"duplicate transform: {char_id}:{go_id}")
                    transforms[go_id] = (oid, obj)
        modes, mode_graph_issue = {}, None
        for index, reference in enumerate(controllers[0]["_modes"]):
            mode = resolve(reference)
            if mode:
                mode_go = str(mode["m_GameObject"]["m_PathID"])
                if mode_go in modes:
                    mode_graph_issue = "operator_mode_reference"
                modes[mode_go] = index
            else:
                mode_graph_issue = "operator_mode_reference"
        markers = {oid: obj for oid, obj in objects.items()
                   if "_overwriteTalentKey" in obj and "_checkValidBySkillIndex" in obj}
        ordered, visited = [], set()
        def visit(go_id):
            if go_id in visited:
                return
            visited.add(go_id)
            for reference in objects.get(go_id, {}).get("m_Component", []):
                ref = reference.get("component") or {}
                oid = str(ref.get("m_PathID"))
                if ref.get("m_FileID") == 0 and oid in markers and oid not in ordered:
                    ordered.append(oid)
            _, transform = transforms.get(go_id, (None, {}))
            for reference in transform.get("m_Children", []):
                child = resolve(reference)
                if child:
                    visit(str(child.get("m_GameObject", {}).get("m_PathID")))
        visit(root_id)
        # Preserve malformed/orphan components as gaps rather than guessing
        # their parent or silently removing them from the discovery result.
        ordered.extend(oid for oid in markers if oid not in ordered)
        bindings = []
        for oid in ordered:
            marker = markers[oid]
            go = resolve(marker.get("m_GameObject"))
            if go is None:
                continue
            go_id = str(marker["m_GameObject"]["m_PathID"])
            seen, mode_index, problem = set(), None, mode_graph_issue
            if not any(r.get("component", {}).get("m_FileID") == 0 and
                       str(r.get("component", {}).get("m_PathID")) == oid for r in go.get("m_Component", [])):
                problem = "talent_component_reference"
            current = go_id
            while True:
                if current in seen:
                    problem = "talent_hierarchy_cycle"; break
                seen.add(current)
                if not objects.get(current, {}).get("m_IsActive", True):
                    problem = "inactive_talent_ancestor"; break
                if current in modes and mode_index is None:
                    mode_index = modes[current]
                if current == root_id:
                    break
                transform_id, transform = transforms.get(current, (None, {}))
                parent = resolve(transform.get("m_Father"))
                if parent is None:
                    problem = "talent_hierarchy_binding"; break
                if not any(r.get("m_FileID") == 0 and str(r.get("m_PathID")) == transform_id
                           for r in parent.get("m_Children", [])):
                    problem = "talent_hierarchy_reference"; break
                parent_go = parent.get("m_GameObject") or {}
                current = str(parent_go.get("m_PathID"))
            resource = f"{char_id}::talent::{oid}"
            # Repeated names across modes are normal: use this exact GO's refs.
            components = []
            for ref in go.get("m_Component", []):
                data = resolve(ref.get("component"))
                path = str((ref.get("component") or {}).get("m_PathID"))
                if data is marker or (data and "m_Father" in data):
                    continue
                components.append({"id": path, "data": data or {}, "kind": "unresolved" if data is None else
                    "passive-buffs" if (data.get("m_Script") or {}).get("m_PathID") == PASSIVE_BUFF_SCRIPT else "unsupported"})
            result["resources"][resource] = {"active": bool(go.get("m_IsActive", True)), "components": components}
            bindings.append({"resource": resource, "key": marker.get("_overwriteTalentKey") or go.get("m_Name"),
                "mode": mode_index, "marker": marker, "issue": problem,
                "key_evidence": "explicit_override" if marker.get("_overwriteTalentKey") else "prefab_name_candidate"})
        result["actors"][char_id] = {"talents": bindings, "modes": list(modes.values()), "mode_issue": mode_graph_issue}
    return result


def compile_resources(objects: dict, resource_keys: set[str]) -> dict:
    """Follow GameObject component references in their serialized order."""
    resources = {}
    for obj in objects.values():
        key = obj.get("m_Name")
        if key not in resource_keys or "m_Component" not in obj:
            continue
        if key in resources:
            raise ValueError(f"duplicate component resource: {key}")
        components = []
        for reference in obj["m_Component"]:
            ref = reference.get("component") or {}
            path = str(ref.get("m_PathID"))
            component = objects.get(path) if ref.get("m_FileID") == 0 else None
            if component is None:
                components.append({"id": path, "kind": "unresolved", "data": {}})
            elif "m_Script" in component:
                script = (component.get("m_Script") or {}).get("m_PathID")
                components.append({"id": path,
                    "kind": "passive-buffs" if script == PASSIVE_BUFF_SCRIPT else "unsupported",
                    "data": component})
        resources[key] = {"active": bool(obj.get("m_IsActive", True)), "components": components}
    return {"format": FORMAT, "version": 1, "resources": resources}


def compile_skill_resources(objects: dict, resource_keys: set[str]) -> dict:
    """Discover main and extra abilities from the selected skill prefab.

    Main active abilities require a cast adapter. Never attach their passive
    buffs just because they happen to live on the skill's root GameObject.
    """
    directory = compile_resources(objects, resource_keys)
    directory["resource_kinds"] = ["skill"]
    for entry in directory["resources"].values():
        controllers = [c for c in entry["components"]
                       if "_extraAbilities" in c["data"] and "_maxTriggerTime" in c["data"]]
        if len(controllers) != 1:
            for component in entry["components"]:
                component["binding_issue"] = "skill_controller_binding"
            continue
        controller = controllers[0]["data"]
        passive = (controller.get("m_Script") or {}).get("m_PathID") == PASSIVE_SKILL_SCRIPT
        issue = None
        if not controller.get("m_Enabled", 1):
            issue = "disabled_skill"
        elif controller.get("_defaultModeIndex", -1) != -1:
            issue = "skill_mode_binding"
        elif controller.get("_canSilenced") or controller.get("_limitGlobalTriggerTime"):
            issue = "skill_availability_binding"
        for component in entry["components"]:
            component["scope"] = "main"
            component["binding_issue"] = issue or (None if passive and controller.get("_attachInDummy")
                                                    else "active_ability_lifecycle")
        for reference in controller.get("_extraAbilities", []):
            path = str(reference.get("m_PathID"))
            data = objects.get(path) if reference.get("m_FileID") == 0 else None
            if not reference.get("m_PathID") and reference.get("m_FileID") == 0:
                continue
            go_ref = (data or {}).get("m_GameObject") or {}
            go = objects.get(str(go_ref.get("m_PathID"))) if go_ref.get("m_FileID") == 0 else None
            binding_issue = issue or ("ability_gameobject_binding" if go is None else
                                     "inactive_ability" if not go.get("m_IsActive", True) else None)
            entry["components"].append({"id": path, "scope": "extra",
                "binding_issue": binding_issue,
                "kind": "unresolved" if data is None else (
                    "passive-buffs" if (data.get("m_Script") or {}).get("m_PathID") == PASSIVE_BUFF_SCRIPT
                    else "unsupported"), "data": data or {}})
    return directory


def module_resources(phase: dict | None, plan: Any, talents: dict,
                     *, game_tags=(), map_tags=()):
    """Resolve the same phase/potential/condition gates as select_abilities."""
    for part in (phase or {}).get("parts", []):
        key = part.get("resKey")
        if not key:
            continue  # Data-only overrides have no component to attach.
        if part.get("isToken"):
            yield key, None, "token_binding"
            continue
        if ((part.get("validInGameTag") is not None and part["validInGameTag"] not in game_tags)
                or (part.get("validInMapTag") is not None and part["validInMapTag"] not in map_tags)):
            continue
        if part.get("target") == "TRAIT":
            yield key, "trait", None
        elif part.get("target") == "TALENT":
            candidates = (part.get("addOrOverrideTalentDataBundle") or {}).get("candidates") or []
            groups = {}
            for candidate in candidates:
                if candidate.get("validModeIndices") is None and "talentIndex" in candidate:
                    group = (candidate["talentIndex"], candidate.get("prefabKey") if candidate.get("isHideTalent") else None)
                    groups.setdefault(group, []).append(candidate)
            chosen = [c for group in groups.values() if (c := select_candidate(
                group, plan.elite, plan.level, plan.potential_rank)) is not None]
            if not chosen:
                if any(c.get("validModeIndices") is not None for c in candidates):
                    yield key, None, "mode_binding"
                continue
            if any(c.get("validModeIndices") is not None for c in candidates) or len(chosen) != 1:
                yield key, None, "multi_talent_or_mode_binding"
                continue
            prefab = str(chosen[0].get("prefabKey") or chosen[0].get("talentIndex"))
            if prefab not in talents:
                yield key, None, "talent_blackboard"
            else:
                yield key, prefab, None
        else:
            yield key, None, "part_target"


_COMPONENT_FIELDS = {"m_GameObject", "m_Enabled", "m_Script", "m_Name", "_selector", "_metadata",
    "_overrideSignalId", "_interruptAbilityOnDetach", "_attachPassiveBuffsOnDummy",
    "_ignoreIfOwnerDead", "_forceAsDmgOrHealAbility", "_buffs"}
_BUFF_DEFAULTS = {
    "loadFromDB": 0, "isDurableBuff": 0, "isDamageMissable": 0, "isSilenceable": 0,
    "isStunnable": 0, "isFreezable": 0, "isLevitatable": 0, "isGroundBoundable": 0,
    "statusResistable": 2, "disableOverride": 0, "overrideKey": "", "overrideType": 0,
    "maxStackCnt": 1, "refreshRemainingTimeWhenStackMax": 0, "clearAllStackCntWhenTimeUp": 0,
    "maxValidStackCnt": -1, "independentCharacterSource": 0, "overrideEffectKey": "",
    "overrideOnEventPriority": 0, "onEventPriority": 0, "audioSignal": "",
    "lifeTimeType": 2, "takeSnapshotWhenExtend": 0, "remainingTimeKey": "", "lifeTime": 0,
    "triggerLifeType": 0, "triggerCnt": 0, "triggerInterval": -1,
    "waitFirstTriggerInterval": 1, "firstTriggerInterval": -1, "priority": -1,
    "priorityBBKeys": [], "enableInitDirectionFromSource": 0,
}
_NODE_FIELDS = {
    "SetMaxMinusHpRatio": {"_targetType", "_value"},
    "CheckIfAtMinusHpState": {"_targetType"},
    "LockCharacterSkillProgress": {"_targetType"},
    "UnlockCharacterSkillProgress": {"_targetType"},
    "FinishAllBuffsWithCertainAbnormalFlag": {"_targetType", "_abnormalFlag"},
    "CheckAbnormalFlag": {"_abnormalFlag", "_targetType", "_isUnset", "_abnormalFlagKey"},
    "CheckAbnormalImmune": {"_abnormalFlag", "_targetType", "_isUnset"},
    **ENTITY_VALUE_READ_FIELDS,
    **BLACKBOARD_CONDITION_FIELDS,
    "EnsureBlackboardDefaultValue": {"_defaultSettings"},
    "IfElse": {"_conditionNode", "_succeedNodes", "_failNodes"},
    "AlwaysNext": set(), "IfNot": set(), "Not": set(),
    "IfTarget": {"_targetType", "_motionMask", "_checkTargetAlive", "_checkApplyWay",
        "_checkTargetUnitType", "_unitType", "_applyWay", "_checkTargetFree", "_checkMotionTargetFree"},
    "CheckBlocked": {"_targetType", "_checkBlockedBySource", "_checkBlockedBySourceToken",
        "_sourceType", "_checkBlockedCount"},
    "CheckEnemyLevelMask": {"_targetType", "_targetLevelMask"},
    "AtkScaleUp": {"_filterApplyWay", "_filterProjectileKey", "_applyWay", "_defaultValue",
        "_filterNoneApplyWay", "_cancelIfAtkScaleZero", "_atkScaleKey", "_overwriteAtkScale"},
    "ModifySp": {"_targetType", "_modifyByRatio", "_modifyRatio", "_modifyByRatioBasedOnCurSP",
        "_modifyBySpCost", "_dontCheckSpType", "_spMask", "_forceFlag", "_dontShowSpUI",
        "_modifyValue", "_spString", "_isMinis", "_customModifierKey"},
    "CreateBuff": {"_buff", "_buffOwner", "_useSpecialBuffSource", "_specialBuffSource",
        "_isDerivedBuff", "_finishDerivedBuffIfParentFinish", "_inherateSourceProjectileFromParentBuff",
        "_hookEffectKeyFromSourceProjectile"},
    "CreateBuffInRange": {"_sourceType", "_targetType", "_targetOptions", "_useCentainSide",
        "_side", "_useHostAsSource", "_excludeTarget", "_excludeCurAtkTarget", "_alwaysIncudeCurAtkTarget",
        "_alwaysIncludeSourceBlocker", "_randomTarget", "_limitMaxTarget", "_useAttackRange",
        "_useCurrentModeRange", "_useRangeToShow", "_rangeTargetSideType", "_rangeModeIndex",
        "_useTargetRangeInsteadOfSource", "_rangeId", "_useRadius", "_useGlobalRange", "_radius",
        "_filterByBuildableType", "_allowedBuildableType", "_filterByTags", "_allowFilterTags",
        "_checkGiantTrapAllLocateTiles", "_buffs", "_isDerivedBuff", "_finishDerivedBuffIfParentFinish",
        "_removeExcludeTargetBeforeShrink", "_filterTargets", "_filterType", "_maxTargetKey"},
    "FinishBuff": {"_decCntIfStack", "_updateOverrideMap"},
    "FinishDerivedBuff": {"_updateOverrideMap"},
    "FinishBuffsById": FINISH_BUFF_FIELDS,
    "CheckContainsBuff": {"_targetType", "_buffKeys", "isAND", "_loadFromBlackboard",
        "_checkBuffSource", "_buffSourceType", "_checkSourceHost"},
    "CheckContainsDerviedBuff": {"_derviedBuffKey"},
    "AssignValueToBB": {"_blackboardKey", "_value", "_copyFromKey", "_assignString", "_stringValue", "_valueStr"},
    "BlackboardAdd": {"_blackboardKey", "_blackboardKeyFromBb", "_addition", "_additionKey",
                      "_isFloat", "_outputKey", "_maxKey"},
    "IsBlackboardZero": {"_var", "_noVarShowWarning"},
    "CalculateBlackboardValueViaParams": {"_inputKey", "_outputKey", "_multiplyParamKey", "_dividedParamKey",
        "_addParamKey", "_minusParamKey", "_maxValueKey", "_minValueKey", "_useRemainder",
        "_finalAbs", "_finalCeil", "_finalFloor", "_finalRound", "_useAbilityBlackboard", "_abilityName"},
    "AssignCharacterSkillBlackboardToBB": {"_targetType", "_sourceBlackboardKey", "_targetBlackboardKey"},
    "AssignBuffBlackboardFromOthers": {"_targetType", "_blackboardKey", "_buffKey", "_valueKey",
                                     "_filterBuffSource", "_sourceType"},
}
BLACKBOARD_NODES = {"AssignValueToBB", "BlackboardAdd", "IsBlackboardZero", "CalculateBlackboardValueViaParams",
                   "EnsureBlackboardDefaultValue"} | BLACKBOARD_READ_NODES | BLACKBOARD_CONDITION_FIELDS.keys()
OWNER_STATE_NODES = {"SetMaxMinusHpRatio", "CheckIfAtMinusHpState", "CheckAbnormalFlag", "CheckAbnormalImmune",
                     "LockCharacterSkillProgress", "UnlockCharacterSkillProgress", "FinishAllBuffsWithCertainAbnormalFlag",
                     "FinishDerivedBuff"}


_BUFF_ENUMS = {"lifeTimeType": {0: "IMMEDIATELY", 1: "LIMITED", 2: "INFINITY"},
        "triggerLifeType": {0: "IMMEDIATELY", 1: "LIMITED", 2: "INFINITY"},
        "statusResistable": {0: "NO", 1: "YES", 2: "AUTOMATIC"},
        "overrideType": {0: "DEFAULT", 1: "STACK", 2: "UNIQUE", 3: "EXTEND", 4: "EXTEND_TIME"}, "onEventPriority": {0: "DEFAULT"}}


def database_passive_spec(spec: dict) -> dict:
    """Adapt a resolved DB row to the existing serialized-component validator."""
    result = dict(spec)
    for key, default in _BUFF_DEFAULTS.items():
        if result.get(key) is None:
            result[key] = default
    for key, mapping in _BUFF_ENUMS.items():
        inverse = {value: number for number, value in mapping.items()}
        value = result.get(key)
        if isinstance(value, str) and value in inverse:
            result[key] = inverse[value]
    result["blackboard"] = spec.get("blackboard") or []
    return result


def normalized_passive_spec(spec: dict, *, periodic: bool = False) -> dict:
    """Convert reviewed serialized enums, never interpret unknown ordinals."""
    result = dict(spec)
    mappings = _BUFF_ENUMS
    for key, mapping in mappings.items():
        value = spec.get(key, _BUFF_DEFAULTS[key])
        if type(value) is int and value in mapping:
            result[key] = mapping[value]
        elif isinstance(value, str) and value in mapping.values():
            result[key] = value
        else:
            raise ValueError("enum:" + key)
    attributes = dict(spec.get("attributes") or {})
    flag_names = {0: "STUNNED", 5: "INVINCIBLE", 6: "UNDEADABLE", 7: "HEAL_FREE", 9: "INVISIBLE",
                  16: "FROZEN", 17: "CAMOUFLAGE", 25: "LEVITATE", 36: "FORCE_LEVITATE", 45: "GROUND_BOUND"}
    flags = []
    for flag in attributes.get("abnormalFlags") or []:
        if type(flag) is int and flag in flag_names:
            flags.append(flag_names[flag])
        elif isinstance(flag, str):
            flags.append(flag)
        else:
            raise ValueError("enum:abnormalFlag")
    attributes["abnormalFlags"] = flags
    modifiers = []
    for modifier in attributes.get("attributeModifiers") or []:
        item = dict(modifier)
        for key, mapping in (("attributeType", {0: "MAX_HP", 1: "ATK", 2: "DEF", 3: "MAGIC_RESISTANCE", 6: "MOVE_SPEED", 7: "ATTACK_SPEED",
                13: "HP_RECOVERY_PER_SEC", 14: "SP_RECOVERY_PER_SEC", 19: "HP_RECOVERY_PER_SEC_BY_MAX_HP_RATIO",
                26: "ONE_MINUS_STATUS_RESISTANCE", 35: "EP_BREAK_RECOVER_SPEED", 36: "SLOW_DOWN"}),
                             ("formulaItem", {0: "ADDITION", 1: "MULTIPLIER", 2: "FINAL_ADDITION", 3: "FINAL_SCALER"})):
            value = item.get(key)
            if type(value) is int and value in mapping:
                item[key] = mapping[value]
            elif not isinstance(value, str) or value not in mapping.values():
                raise ValueError("enum:" + key)
        modifiers.append(item)
    attributes["attributeModifiers"] = modifiers
    result["attributes"] = attributes
    # BuffContainer owns the interval. An idle serialized clock needs no tick.
    if not periodic:
        result["triggerLifeType"] = "IMMEDIATELY"
    if not periodic:
        result["triggerInterval"] = -1
    if result["lifeTimeType"] == "INFINITY":
        result["durationKey"] = None
    return result


def passive_issue(component: dict, spec: dict, *, idle_timer=False, periodic=False) -> str | None:
    if periodic and not (spec.get("triggerLifeType") in (1, 2) and type(spec.get("triggerInterval")) in (int, float)
                         and math.isfinite(spec["triggerInterval"])
                         and (spec["triggerInterval"] > 0 or spec["triggerInterval"] == -1)):
        return "trigger_clock"
    if set(component) - _COMPONENT_FIELDS:
        return "component_fields"
    if not component.get("m_Enabled", 1):
        return "disabled_component"
    if (component.get("_selector") or {}).get("m_PathID", 0) or (component.get("_selector") or {}).get("m_FileID", 0):
        return "selector"
    if any((component.get("_metadata") or {}).values()) or component.get("_overrideSignalId"):
        return "metadata_signal"
    if component.get("_forceAsDmgOrHealAbility"):
        return "damage_ability"
    for key in ("_interruptAbilityOnDetach", "_ignoreIfOwnerDead", "_attachPassiveBuffsOnDummy"):
        if component.get(key, 0) not in (0, 1):
            return key
    if not component.get("_attachPassiveBuffsOnDummy", 1):
        return "dummy_attachment"
    allowed = set(_BUFF_DEFAULTS) | {"attributes", "buffKey", "templateKey", "durationKey",
        "stripBlackboardParamsWithBuffKey", "blackboard"}
    if set(spec) - allowed:
        return "buff_fields"
    for key, default in _BUFF_DEFAULTS.items():
        if key in {"isSilenceable", "isStunnable", "isDamageMissable"} and spec.get(key, 0) in (0, 1):
            continue
        if key == "statusResistable" and spec.get(key, 2) in (0, 1, 2, "NO", "YES", "AUTOMATIC"):
            continue
        if key == "overrideType" and spec.get(key, 0) in (1, 2, 3, 4):
            continue
        if key == "overrideKey" and (spec.get("overrideType") in (1, 2, 3, 4) or
                type(spec.get("priority")) is int and spec["priority"] >= 0 or
                spec.get("overrideType", 0) == 0 and bool(spec.get("priorityBBKeys"))) and isinstance(spec.get(key, ""), str):
            continue
        if key == "priorityBBKeys" and spec.get("overrideType", 0) == 0 and isinstance(spec.get(key), list):
            continue
        if key == "priority" and spec.get("overrideType", 0) == 0 and type(spec.get(key)) is int and 0 <= spec[key] <= 2147483647:
            continue
        if key == "maxStackCnt" and spec.get("overrideType") == 1 and type(spec.get(key, 1)) is int and 1 <= spec.get(key, 1) <= 2147483647:
            continue
        if key == "clearAllStackCntWhenTimeUp" and spec.get("overrideType") == 1 and spec.get(key, 0) in (0, 1):
            continue
        if periodic and key in {"triggerLifeType", "triggerCnt"} and spec.get("triggerLifeType") == 1:
            if type(spec.get("triggerCnt")) is int and 0 <= spec["triggerCnt"] <= 2147483647:
                continue
            return "trigger_count"
        if key == "disableOverride" and spec.get(key, 0) in (0, 1):
            continue  # Duplicate bindings are still rejected as a whole.
        if key == "waitFirstTriggerInterval" and (periodic or idle_timer) and spec.get(key, 1) in (0, 1):
            continue
        if key == "firstTriggerInterval" and periodic and type(spec.get(key, -1)) in (int, float) and math.isfinite(spec.get(key, -1)) and spec.get(key, -1) >= 0:
            continue
        if key in {"lifeTimeType", "lifeTime"}:
            if spec.get("lifeTimeType", 2) in (1, 2) and type(spec.get("lifeTime", 0)) in (int, float) and math.isfinite(spec.get("lifeTime", 0)) and spec.get("lifeTime", 0) >= 0:
                continue
        if (idle_timer or periodic) and key in {"triggerLifeType", "triggerInterval"}:
            if spec.get("triggerLifeType") in (1, 2) and type(spec.get("triggerInterval")) in (int, float) and math.isfinite(spec["triggerInterval"]) and (spec["triggerInterval"] > 0 or spec["triggerInterval"] == -1):
                continue
        if spec.get(key, default) != default:
            return "buff:" + key
    try:
        normalized = normalized_passive_spec(spec, periodic=periodic)
    except (ValueError, TypeError):
        return "attributes/enums"
    # Duplicate component bindings still have their own conservative gate.
    checked = {**normalized, "durationKey": None,
               "stripBlackboardParamsWithBuffKey": False}
    if issue := registered_buff_spec_issue(checked):
        return issue
    if not spec.get("buffKey") or not spec.get("templateKey"):
        return "buff_template_key"
    # Infinite duration keys are inert; active clocks belong to BuffContainer.
    return None


def event_phase(template: dict) -> tuple[str | None, str | None]:
    events = template.get("eventToActions") or {}
    if not events and template.get("templateKey") == "empty":
        return "buff-events", None
    owner_events = LIFECYCLE_EVENTS | {AFTER_ATTACK, "ON_BUFF_TRIGGER"}
    if events and set(events) <= owner_events and any(
            node_name(node) in {"CreateBuff", "CreateBuffInRange", "FinishBuff", "FinishBuffsById", "CheckContainsBuff", "CheckContainsDerviedBuff"} | BLACKBOARD_NODES | OWNER_STATE_NODES
            for node in _action_nodes(events)):
        allowed = {"IfElse", "AlwaysNext", "IfNot", "Not", "ModifySp", "CreateBuff", "CreateBuffInRange",
                   "FinishBuff", "FinishBuffsById", "CheckContainsBuff", "CheckContainsDerviedBuff"} | BLACKBOARD_NODES | OWNER_STATE_NODES
        for node in _action_nodes(events):
            name = node_name(node)
            if name not in allowed:
                return None, "owned_buff_node:" + name
            if set(node) - (_NODE_FIELDS[name] | {"$type"}):
                return None, "owned_buff_fields:" + name
            if name == "CheckContainsBuff" and node.get("_checkSourceHost"):
                return None, "owned_buff_source_host"
            if name in BLACKBOARD_NODES | OWNER_STATE_NODES and (issue := BehaviorInterpreter._unsupported_variant(node)):
                return None, issue
            if name == "CreateBuff" and node.get("_useSpecialBuffSource") and node.get("_specialBuffSource") not in {"SOURCE", "BUFF_OWNER", "BUFF_SOURCE"}:
                return None, "owned_buff_role"
            for role in ("_targetType", "_buffOwner", "_sourceType", "_buffSourceType"):
                if name == "AssignBuffBlackboardFromOthers" and role == "_sourceType" and not node.get("_filterBuffSource"):
                    continue
                if name == "FinishBuffsById" and role == "_sourceType" and not node.get("_checkBuffSource"):
                    continue
                if role in node and node[role] not in {"SOURCE", "BUFF_OWNER", "BUFF_SOURCE"}:
                    return None, "owned_buff_role"
        if template.get("onEventPriority", "DEFAULT") != "DEFAULT":
            return None, "event_priority"
        return "buff-events", None
    if set(events) == {OUTPUT}:
        actions = events[OUTPUT]
        if (not isinstance(actions, list) or not actions or not isinstance(actions[0], dict)
                or set(actions[0]) != {"$type"} or node_name(actions[0]) != "IsHeal"):
            return None, "heal_output_guard"
        phase, issue = event_phase({**template, "eventToActions": {CALCULATE: actions[1:]}})
        if issue or phase != "notify":
            return None, issue or "heal_output_nodes"
        return "heal-output", None
    if events and set(events) <= owner_events:
        # Lifecycle events have no damage target/modifier. Only an owner's
        # stateless SP action and ordinary sequence control are bound here.
        allowed = {"ModifySp", "AlwaysNext", "IfNot", "Not"}
        for actions in events.values():
            def issue(value):
                if isinstance(value, dict):
                    if "$type" in value:
                        name = node_name(value)
                        if name not in allowed or set(value) - (_NODE_FIELDS[name] | {"$type"}):
                            return "lifecycle_node:" + name
                        if name == "ModifySp" and value.get("_targetType") not in {"SOURCE", "BUFF_OWNER", "BUFF_SOURCE"}:
                            return "lifecycle_entity_role"
                    return next((x for child in value.values() if (x := issue(child))), None)
                if isinstance(value, list):
                    return next((x for child in value if (x := issue(child))), None)
                return None
            if problem := issue(actions):
                return None, problem
            # Reuse the same SP variant validation as outgoing calculation.
            _, problem = event_phase({**template, "eventToActions": {CALCULATE: actions}})
            if problem:
                return None, problem
        if template.get("onEventPriority", "DEFAULT") != "DEFAULT":
            return None, "event_priority"
        return ("owner-events" if AFTER_ATTACK in events else "lifecycle"), None
    if CALCULATE not in events or set(events) - {CALCULATE, CACHED_CALCULATE}:
        return None, "event_lifecycle"
    if CACHED_CALCULATE in events and events[CACHED_CALCULATE] != events[CALCULATE]:
        return None, "cached_event_variant"
    if template.get("onEventPriority", "DEFAULT") != "DEFAULT":
        return None, "event_priority"
    kinds, issues, bindings = set(), [], []
    def visit(value):
        if isinstance(value, dict):
            if "$type" in value:
                name = node_name(value)
                kinds.add(name)
                if name in _NODE_FIELDS and set(value) - (_NODE_FIELDS[name] | {"$type"}):
                    issues.append(name + ":fields")
                if name == "AtkScaleUp" and value.get("_overwriteAtkScale"):
                    issues.append(name + ":overwrite_binding")
                if name in {"IfTarget", "CheckBlocked", "AtkScaleUp"}:
                    issue = BehaviorInterpreter._unsupported_variant(value)
                    if issue:
                        bindings.append(issue)
                roles = {"SOURCE", "TARGET", "BUFF_OWNER", "BUFF_SOURCE", "MODIFIER_SOURCE", "MODIFIER_TARGET"}
                for field in ("_targetType", "_sourceType"):
                    if field in value and value[field] not in roles:
                        issues.append(name + ":entity_role")
                if name == "ModifySp":
                    mask = value.get("_spMask", "ALL")
                    masks = {"NONE", "INCREASE_WITH_TIME", "INCREASE_WHEN_ATTACK", "INCREASE_WHEN_TAKEN_DAMAGE",
                        "ATTACK_OR_DAMAGE", "ALL"}
                    if not value.get("_dontCheckSpType") and not (
                            isinstance(mask, str) and mask in masks or type(mask) is int and 0 <= mask <= 7):
                        issues.append(name + ":sp_mask")
                    if sum(bool(value.get(k)) for k in ("_modifyByRatio", "_modifyByRatioBasedOnCurSP", "_modifyBySpCost")) > 1:
                        issues.append(name + ":ratio_modes")
                if name == "CheckEnemyLevelMask" and value.get("_targetLevelMask") not in {
                        "NORMAL", "ELITE", "BOSS", "ELITE_AND_BOSS", "ALL", "NORMAL_AND_ELITE_AND_BOSS"}:
                    issues.append(name + ":level_mask")
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(events[CALCULATE])
    if issues:
        return None, issues[0]
    if bindings:
        return None, "unbound_variant:" + bindings[0]
    flow = {"IfElse", "AlwaysNext", "IfNot", "Not", "IfTarget", "CheckBlocked", "CheckEnemyLevelMask"}
    if kinds <= flow | {"AtkScaleUp"}:
        return "scale", None
    if kinds <= flow | {"ModifySp"}:
        return "notify", None
    return None, "node_phase:" + ",".join(sorted(kinds - flow))


class _PreviewBoard(dict):
    """Keep abstract actor state outside user blackboard keys."""

    def __init__(self, board, preview):
        super().__init__(board)
        self.actor_preview = preview


def numeric_issue(actions: Any, board: dict, *, node_validator=None, blackboard_reader=None,
                  actor_preview=None, final_states: list[dict] | None = None) -> str | None:
    """Preflight possible ordered states without executing side effects."""
    watched_reads = set(ENTITY_VALUE_READ_FIELDS)
    def read_order(value, dirty=False):
        if isinstance(value, list):
            for item in value:
                issue, dirty = read_order(item, dirty)
                if issue: return issue, dirty
            return None, dirty
        if not isinstance(value, dict): return None, dirty
        name = node_name(value)
        if name in watched_reads and dirty:
            return name + ":after_actor_mutation_binding", dirty
        if name == "IfElse":
            issue, current = read_order(value.get("_conditionNode"), dirty)
            if issue: return issue, current
            issue, yes = read_order(value.get("_succeedNodes") or [], current)
            if issue: return issue, yes
            issue, no = read_order(value.get("_failNodes") or [], current)
            return issue, yes or no
        if name in {"CreateBuff", "CreateBuffInRange", "FinishBuff", "FinishBuffsById", "SwitchMode",
                    "FinishAllBuffsWithCertainAbnormalFlag", "FinishDerivedBuff", "AdvancedApplyDamage", "NoSourceDamage", "AOEDamage", "DamageViaAttr", "DamageViaMaxHpRatio"}:
            dirty = True
        return None, dirty
    if actor_preview is None:
        if issue := read_order(actions)[0]:
            return issue
    else:
        watched_reads |= {"AssignBuffBlackboardFromOthers", "AssignCharacterSkillBlackboardToBB"}
        # Keep existing abstract gates for events that need no actor overlay.
        # Their different final Buff states cannot affect an earlier read.
        override_actions = any(node_name(node) in {"CreateBuff", "CreateBuffInRange"}
            and any(not spec.get("disableOverride", False) and (
                spec.get("priority", -1) != -1 or spec.get("priorityBBKeys") or
                (spec.get("attributes") or {}).get("attributeModifiers"))
                for spec in ([node.get("_buff") or {}] if node_name(node) == "CreateBuff"
                             else node.get("_buffs") or [])) for node in _action_nodes(actions))
        if read_order(actions)[0] is None and not override_actions:
            actor_preview = None
    if final_states is None and actor_preview is None and node_validator is None and not any(node_name(node) in BLACKBOARD_NODES for node in _action_nodes(actions)):
        return _numeric_issue(actions, board)

    def clone(state):
        return (_PreviewBoard(state, state.actor_preview.clone()) if isinstance(state, _PreviewBoard)
                else dict(state))

    def equal(left, right):
        return left == right and (not isinstance(left, _PreviewBoard)
            or isinstance(right, _PreviewBoard) and left.actor_preview.same_state(right.actor_preview))

    def unique(states):
        result = []
        for state, gate in states:
            if not any(equal(state, old) and gate is old_gate for old, old_gate in result):
                result.append((state, gate))
        return result

    def walk(value, state):
        if isinstance(value, list):
            states = [(clone(state), True)]
            for child in value:
                name = node_name(child) if isinstance(child, dict) else ""
                next_states = []
                for current, gate in states:
                    if name == "AlwaysNext":
                        next_states.append((current, True)); continue
                    if name in {"IfNot", "Not"}:
                        next_states.append((current, not gate if gate is not None else None)); continue
                    problem, outcomes = walk(child, clone(current))
                    if problem:
                        return problem, []
                    if gate is not False:
                        next_states.extend(outcomes)
                    if gate is not True:
                        next_states.append((current, False))
                states = unique(next_states)
                if len(states) > 32:
                    return "Blackboard:state_limit", []
            return None, states
        if not isinstance(value, dict):
            return None, [(state, None)]
        name = node_name(value)
        if name in BLACKBOARD_CONDITION_FIELDS:
            try:
                passed = blackboard_condition_value(value, state)
            except ValueError as exc:
                return name + ":" + str(exc), []
            return None, [(state, passed)]
        if name == "EnsureBlackboardDefaultValue":
            try:
                values = default_blackboard_values(value, state)
            except ValueError as exc:
                return name + ":" + str(exc), []
            state.update(values)
            return None, [(state, True)]
        if name == "IfElse":
            condition = value.get("_conditionNode") or {}
            problem, conditions = walk(condition, clone(state))
            if problem:
                return problem, []
            outcomes = []
            for current, passed in conditions:
                problem, yes = walk(value.get("_succeedNodes") or [], clone(current))
                if problem:
                    return problem, []
                problem, no = walk(value.get("_failNodes") or [], clone(current))
                if problem:
                    return problem, []
                if passed is None:
                    yes_boards = unique([(b, True) for b, _ in yes])
                    no_boards = unique([(b, True) for b, _ in no])
                    if any(not any(equal(b, other) for other, _ in no_boards) for b, _ in yes_boards) or any(
                            not any(equal(b, other) for other, _ in yes_boards) for b, _ in no_boards):
                        return "Blackboard:conditional_state_binding", []
                    outcomes.extend(yes + no)
                else:
                    outcomes.extend(yes if passed else no)
            return None, unique(outcomes)
        if name == "AssignValueToBB":
            try:
                key, assigned = assigned_blackboard_value(value, state)
            except ValueError as exc:
                return name + ":" + str(exc), []
            state[key] = assigned
            return None, [(state, True)]
        if name == "BlackboardAdd":
            key = value.get("_blackboardKey")
            if not isinstance(key, str) or not key:
                return name + ":key", []
            base = state.get(key, 0)
            addition = state.get(value["_additionKey"], 0) if value.get("_additionKey") else value.get("_addition", 0)
            if any(type(x) not in (int, float) or not math.isfinite(x) for x in (base, addition)):
                return name + ":value", []
            total = base + addition
            if not math.isfinite(total):
                return name + ":overflow", []
            state[key] = total if value.get("_isFloat") else int(total)
            return None, [(state, True)]
        if name == "IsBlackboardZero":
            key = value.get("_var")
            if not isinstance(key, str) or not key:
                return name + ":key", []
            if key in state and (type(state[key]) not in (int, float) or not math.isfinite(state[key])):
                return name + ":value", []
            return None, [(state, key in state and abs(state[key]) <= 1e-9)]
        if name == "CalculateBlackboardValueViaParams":
            try:
                passed, key, value = calculated_blackboard_value(value, state)
            except ValueError as exc:
                return name + ":" + str(exc), []
            if passed:
                state[key] = value
            return None, [(state, passed)]
        if name in BLACKBOARD_READ_NODES:
            if blackboard_reader is None:
                return name + ":reader_binding", []
            try:
                passed, key, read_value = blackboard_reader(value, state)
            except ValueError as exc:
                return name + ":" + str(exc), []
            if passed:
                state[key] = read_value
            return None, [(state, passed)]
        if problem := _numeric_issue(value, state):
            return problem, []
        if node_validator is not None and (problem := node_validator(value, state)):
            return problem, []
        if isinstance(state, _PreviewBoard):
            try:
                gate = state.actor_preview.apply(value, state)
            except ValueError as exc:
                return name + ":" + str(exc), []
            return None, [(state, gate)]
        return None, [(state, None)]

    problem, outcomes = walk(actions, _PreviewBoard(board, actor_preview) if actor_preview is not None else dict(board))
    if problem is None and final_states is not None:
        for state, _ in outcomes:
            snapshot = dict(state)
            if snapshot not in final_states:
                final_states.append(snapshot)
    return problem


def _numeric_issue(actions: Any, board: dict) -> str | None:
    """Validate supported numeric inputs before any event changes state."""
    if isinstance(actions, list):
        return next((issue for item in actions if (issue := _numeric_issue(item, board))), None)
    if not isinstance(actions, dict):
        return None
    name = node_name(actions)
    if name == 'NoSourceDamage':
        try:
            no_source_damage_value(actions, board)
        except (ValueError, OverflowError) as exc:
            return name + ':' + str(exc)
    fields = {"AtkScaleUp": (("_atkScaleKey", "_defaultValue", 1),),
              "ModifySp": (("_spString", "_modifyValue", 0),)}.get(name, ())
    for key, default_key, default in fields:
        board_key = actions.get(key) or {"AtkScaleUp": "atk_scale"}.get(name)
        value = board.get(board_key, actions.get(default_key, default))
        if name == "ModifySp" and (actions.get("_modifyByRatio") or actions.get("_modifyByRatioBasedOnCurSP")):
            value = board.get(actions.get(key), actions.get("_modifyRatio", 0))
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            return name + ":numeric_value"
    return next((issue for value in actions.values() if (issue := _numeric_issue(value, board))), None)


def contains_sp_action(value: Any) -> bool:
    if isinstance(value, dict):
        return node_name(value) == "ModifySp" or any(contains_sp_action(child) for child in value.values())
    return isinstance(value, list) and any(contains_sp_action(child) for child in value)


def _action_nodes(value):
    if isinstance(value, dict):
        if "$type" in value:
            yield value
        for child in value.values():
            yield from _action_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _action_nodes(child)


def talent_blackboard(marker: dict, board: dict, skill_board: dict, mode: int) -> tuple[dict, str | None]:
    """Snapshot the reviewed talent@ overlay and explicit scale conditions.

    Serialized fields and current skill data support this mapping; exact
    native merge/initialization order remains a blackboard candidate.
    """
    result = dict(board)
    scale_modes = marker.get("_scaleModeIndices") or []
    if not isinstance(scale_modes, list) or any(type(index) is not int or index < 0 for index in scale_modes):
        return result, "talent_scale_modes"
    keys = marker.get("_scaleCertainKeyList") or []
    if not isinstance(keys, list) or any(not isinstance(key, str) or not key for key in keys):
        return result, "talent_scale_keys"
    if marker.get("_applyTalentScale") and (not marker.get("_scaleCertainMode") or mode in scale_modes):
        scale = skill_board.get("talent_scale", 1)
        if type(scale) not in (int, float) or not math.isfinite(scale) or scale < 0:
            return result, "talent_scale_value"
        for key, value in tuple(result.items()):
            if marker.get("_scaleCertainKeyFlag") and key not in keys:
                continue
            if type(value) in (int, float):
                if not math.isfinite(value) or not math.isfinite(value * scale):
                    return result, "talent_scale_numeric"
                result[key] = value * scale
    if marker.get("_applyStrBlackboardBySkill") and not marker.get("_applyBlackboardBySkill"):
        return result, "talent_string_overlay_binding"
    if marker.get("_applyBlackboardBySkill"):
        for key, value in skill_board.items():
            if not isinstance(key, str) or not key.startswith("talent@"):
                continue  # Attack and skill namespaces are separate.
            if isinstance(value, str) and marker.get("_applyStrBlackboardBySkill"):
                result[key[len("talent@"):]] = value
            elif type(value) in (int, float):
                if not math.isfinite(value):
                    return result, "talent_overlay_numeric"
                result[key[len("talent@"):]] = value
            elif not isinstance(value, str):
                return result, "talent_overlay_value"
    return result, None


@dataclass
class ComponentListener:
    template: str
    buff_key: str
    phase: str
    board: dict
    ignore_dead: bool
    record: dict
    events: tuple[str, ...] = (CALCULATE,)
    started: bool = False
    runtime_spec: dict | None = None
    instance: Any = None
    finished: bool = False
    actor: Any = field(default=None, repr=False, compare=False)


class ComponentCatalog:
    def __init__(self, data: dict | None):
        data = data or {}
        if data and (data.get("format") != FORMAT or type(data.get("version")) is not int or data.get("version") != 1):
            raise ValueError("unsupported behavior component directory")
        self.enabled = bool(data)
        self.resource_kinds = set(data.get("resource_kinds", ["module"]))
        if self.resource_kinds - {"module", "skill", "operator"}:
            raise ValueError("unsupported behavior component resource kind")
        self.actors = data.get("actors", {})
        if not isinstance(self.actors, dict):
            raise ValueError("behavior component actors must be a mapping")
        self.resources = data.get("resources", {})
        if not isinstance(self.resources, dict):
            raise ValueError("behavior component resources must be a mapping")

    def skill_progress_hooker(self, resource: str) -> dict | None:
        """Resolve the exact serialized Hooker; do not assume every skill owns one."""
        entry = self.resources.get(resource)
        if not entry or not entry.get("active", True):
            return None
        candidates = [component.get("data") or {} for component in entry.get("components", [])
            if ((component.get("data") or {}).get("m_Script") or {}).get("m_PathID") == SKILL_PROGRESS_HOOKER_SCRIPT
            and (component.get("data") or {}).get("m_Enabled", 1)]
        if len(candidates) != 1:
            return None
        data = candidates[0]
        if not {"_restoreSkillBuff", "_endSkillBuff", "_skillDurationBBKey"} <= data.keys():
            return None
        return data

    def bind(self, battle, actor, resource: str, board: dict, *, reason=None, binding=None):
        """Attach whole supported passive listeners; never execute a partial tree."""
        entry = self.resources.get(resource)
        def record(state, issue=None, **fields):
            item = {"actor": getattr(actor, "char_id", getattr(actor, "key", "")),
                "deployment": getattr(actor, "deployment_index", None), "resource": resource,
                "state": state, "reason": issue, **fields}
            battle.component_status.append(item)
            if binding is not None:
                item["binding"] = dict(binding)
            if issue:
                battle.mechanic_diagnostics[f"COMPONENT_PENDING:{resource}:{issue}"] += 1
            return item
        if reason or entry is None or not entry.get("active", True):
            record("pending", reason or ("missing_resource" if entry is None else "inactive_resource"))
            return
        for component in entry.get("components", []):
            data = component.get("data") or {}
            if component.get("binding_issue"):
                record("pending", component["binding_issue"], component=component.get("id"), scope=component.get("scope"))
                continue
            if component.get("kind") != "passive-buffs":
                record("pending", "component_class", component=component.get("id"))
                continue
            if not data.get("_buffs"):
                record("pending", "empty_passive_buffs", component=component.get("id"))
            for spec in data.get("_buffs") or []:
                if spec.get("loadFromDB"):
                    try:
                        spec = database_passive_spec(battle.buff_database.resolve_spec(spec))
                    except ValueError as exc:
                        record("pending", str(exc), component=component.get("id"), buff=spec.get("buffKey"))
                        continue
                key = spec.get("templateKey", "")
                fields = {"component": component.get("id"), "buff": spec.get("buffKey"), "template": key}
                if (binding or {}).get("mode") is not None and any(
                        old.buff_key == spec.get("buffKey") and (old.record.get("binding") or {}).get("mode") is None
                        for old in actor.component_listeners):
                    record("pending", "mode_buff_collision", **fields)
                    continue  # Do not detach or claim an unrelated root buff.
                template = battle.behavior.resolve_template(key)
                periodic = template is not None and "ON_BUFF_TRIGGER" in (template.get("eventToActions") or {})
                issue = passive_issue(data, spec, idle_timer=template is not None and not periodic,
                                      periodic=periodic)
                if issue:
                    record("pending", issue, **fields)
                    continue
                if template is None:
                    actor.component_claims.add(key)
                    record("rejected", "missing_template", **fields)
                    battle.behavior.validate_event(key, CALCULATE)
                    continue
                events = template.get("eventToActions") or {}
                phase, issue = event_phase(template)
                if issue and issue.startswith("unbound_variant:"):
                    # These selectors belong to an unported ability adapter;
                    # discovery must not disable an already supported effect.
                    record("pending", issue, **fields)
                    continue
                if ((CALCULATE in events and not set(events) - {CALCULATE, CACHED_CALCULATE})
                        or (events and set(events) <= LIFECYCLE_EVENTS | {AFTER_ATTACK} and contains_sp_action(events))
                        or (set(events) == {OUTPUT} and contains_sp_action(events))):
                    actor.component_claims.add(key)
                if issue:
                    if key in actor.component_claims:
                        record("rejected", issue, **fields)
                        battle.behavior._unsupported(key + ":component:" + issue)
                    else:
                        record("pending", issue, **fields)
                    continue
                # Legacy adapters consult these claims to avoid a second effect.
                actor.component_claims.add(key)
                bound_board = dict(board)
                if spec.get("stripBlackboardParamsWithBuffKey"):
                    prefix = spec["buffKey"] + "."
                    bound_board = {k[len(prefix):]: v for k, v in bound_board.items() if k.startswith(prefix)}
                bound_board.update({str(item["key"]): item.get("valueStr") or item.get("value")
                                   for item in spec.get("blackboard", []) if item.get("key") is not None})
                attached_events = tuple(events) if phase in {"lifecycle", "owner-events", "heal-output", "buff-events"} else (CALCULATE,)
                item = record("attached", phase=phase, events=list(attached_events),
                    deferred_events=[event for event in events if event not in attached_events], **fields)
                battle.mechanic_diagnostics[{"lifecycle": "COMPONENT_LIFECYCLE_PHASE_CANDIDATE",
                    "owner-events": "COMPONENT_AFTER_ATTACK_PHASE_CANDIDATE",
                    "heal-output": "COMPONENT_OUTPUT_PHASE_CANDIDATE",
                    "buff-events": "COMPONENT_BUFF_LIFECYCLE_PHASE_CANDIDATE"}.get(phase, "COMPONENT_CALCULATE_PHASE_CANDIDATE")] = 1
                listener = ComponentListener(key, spec["buffKey"], phase, bound_board,
                    bool(data.get("_ignoreIfOwnerDead")), item, attached_events, actor=actor)
                if (spec.get("lifeTimeType") == 1 or periodic or phase == "buff-events"
                        or set(events) & {"ON_BUFF_ENABLE", "ON_BUFF_DISABLE"}
                        or any((spec.get("attributes") or {}).values())):
                    listener.runtime_spec = normalized_passive_spec(spec, periodic=periodic)
                    # The listener's board has already applied prefix/explicit
                    # parameters. Do not strip it for a second time.
                    listener.runtime_spec["stripBlackboardParamsWithBuffKey"] = False
                    listener.runtime_spec["blackboard"] = []
                    battle.mechanic_diagnostics["COMPONENT_BUFF_CLOCK_PHASE_CANDIDATE"] = 1
                    if phase in {"lifecycle", "owner-events", "buff-events"}:
                        if battle._validated_buff_events(listener.runtime_spec, blackboard=bound_board,
                                allowed_events=LIFECYCLE_EVENTS | {AFTER_ATTACK, "ON_BUFF_TRIGGER"},
                                owner=actor, source=actor, immediate_children=True) is None:
                            item.update(state="rejected", reason="buff_preflight")
                            continue
                    elif battle._behavior_buff_definition(listener.runtime_spec, bound_board) is None:
                        item.update(state="rejected", reason="buff_preflight")
                        continue
                duplicates = [old for old in actor.component_listeners if old.buff_key == listener.buff_key]
                if duplicates or listener.buff_key in actor.component_rejected_buffs:
                    actor.component_rejected_buffs.add(listener.buff_key)
                    for old in duplicates:
                        old.record.update(state="rejected", reason="duplicate_buff_binding")
                        if old.instance is not None:
                            actor.buffs.finish(old.instance)
                        actor.component_listeners.remove(old)
                    item.update(state="rejected", reason="duplicate_buff_binding")
                    battle.behavior._unsupported(key + ":component:duplicate_buff_binding")
                    continue
                if listener.runtime_spec is not None and any(
                        old.definition.key == listener.buff_key and not old.finished
                        for old in (*actor.buffs.active, *actor.buffs.pending_add)):
                    item.update(state="rejected", reason="buff_instance_collision")
                    continue
                if (listener.runtime_spec is not None and phase != "buff-events"
                        and contains_sp_action(events) and not hasattr(actor, "sp_type")):
                    item.update(state="rejected", reason="sp_controller_binding")
                    battle.behavior._unsupported(key + ":component:sp_controller_binding")
                    continue
                if not self.validate(battle, listener):
                    item.update(state="rejected", reason="preflight")
                    continue
                if listener.runtime_spec is None and phase in {"lifecycle", "owner-events"} and any(battle.execute_behavior_event(key, event,
                        source=actor, buff_owner=actor, blackboard=bound_board, validate_only=True) is None
                        for event in attached_events):
                    item.update(state="rejected", reason="event_binding")
                    continue
                actor.component_listeners.append(listener)

    @staticmethod
    def validate(battle, listener: ComponentListener, *, initializing: bool = False) -> bool:
        key = listener.template
        template = battle.behavior.resolve_template(key)
        if template is None:
            return battle.behavior.validate_event(key, CALCULATE)
        phase, issue = event_phase(template)
        if issue or phase != listener.phase:
            return battle.behavior._unsupported(key + ":component:" + (issue or "phase_changed"))
        events = tuple(template["eventToActions"]) if phase in {"lifecycle", "owner-events", "heal-output", "buff-events"} else (CALCULATE,)
        if events != listener.events:
            return battle.behavior._unsupported(key + ":component:events_changed")
        started_boards = [listener.board]
        ordered_events = (["ON_BUFF_START"] if "ON_BUFF_START" in events else []) + [
            event for event in events if event != "ON_BUFF_START"]
        for event in ordered_events:
            source = listener.instance.source if listener.instance is not None else listener.actor
            context = BehaviorContext(source=source, target=listener.actor, buff_owner=listener.actor,
                buff_source=source, buff_instance=listener.instance, battle=battle,
                register_buffs=listener.runtime_spec is not None, event=event)
            definition = (listener.instance.definition if listener.instance is not None else
                battle._behavior_buff_definition(listener.runtime_spec, listener.board)
                if listener.runtime_spec is not None else None)
            completed = []
            for event_board in ([listener.board] if event == "ON_BUFF_START" or listener.started and not initializing else started_boards):
                issue = numeric_issue(template["eventToActions"][event], event_board,
                    blackboard_reader=lambda node, board: battle._read_preflight_blackboard(node, context, board),
                    actor_preview=battle._behavior_attribute_preview(context, root_definition=definition, event=event),
                    final_states=completed)
                if issue:
                    return battle.behavior._unsupported(key + ":component:" + issue)
            if event == "ON_BUFF_START" and completed and (not listener.started or initializing):
                started_boards = completed
            if not battle.behavior.validate_event(key, event):
                return False
        return True

    def bind_operator(self, battle, actor, *, module_bindings=(), selected_talents=None, mode=None):
        if "operator" not in self.resource_kinds:
            return
        entry = self.actors.get(actor.char_id)
        if entry is None:
            self.bind(battle, actor, actor.char_id, {}, reason="missing_operator_graph")
            return
        if mode is None:
            actor.component_talent_selection = (tuple(module_bindings), selected_talents)
        overrides = {key for _, key, _ in module_bindings if key and key != "trait"}
        actual_mode = mode if mode is not None else (actor.component_mode_index or 0)
        plan = actor.build_plan
        char = battle.characters[actor.char_id]
        phase = module_phase(battle.battle_equips[plan.module_id], plan.module_level) if plan.module_id else None
        candidates = [c for bundle in char.get("talents", []) if bundle
                      for c in bundle.get("candidates") or []]
        candidates += [c for part in (phase or {}).get("parts", [])
                       for c in (part.get("addOrOverrideTalentDataBundle") or {}).get("candidates") or []]
        for part in (phase or {}).get("parts", []):
            tags = battle._module_ability_tags(plan.module_id)
            if (not part.get("resKey") or part.get("isToken")
                    or part.get("validInGameTag") is not None and part["validInGameTag"] not in tags.get("game_tags", ())
                    or part.get("validInMapTag") is not None and part["validInMapTag"] not in tags.get("map_tags", ())):
                continue
            for candidate in (part.get("addOrOverrideTalentDataBundle") or {}).get("candidates") or []:
                if ((candidate.get("validModeIndices") is None or actual_mode in candidate["validModeIndices"])
                        and select_candidate([candidate], plan.elite, plan.level, plan.potential_rank) is not None):
                    overrides.add(str(candidate.get("prefabKey")))
        mode_keys = {str(c.get("prefabKey")) for c in candidates if c.get("validModeIndices") is not None}
        chosen = selected_talents
        if mode_keys:
            _, chosen, _ = select_abilities(char, plan.elite, plan.level, plan.potential_rank, phase,
                mode=actual_mode, **battle._module_ability_tags(plan.module_id))
        selected = {str(c["prefabKey"]): c for c in (chosen or {}).values() if c.get("prefabKey")
                    and (c.get("validModeIndices") is None or actual_mode in c["validModeIndices"])}
        bindings = []
        for item in entry.get("talents", []):
            key = item.get("key")
            marker = item.get("marker") or {}
            mode_dependent = key in mode_keys or bool(marker.get("_scaleCertainMode"))
            if mode is not None and item.get("mode") != mode and not (item.get("mode") is None and mode_dependent):
                continue
            skill_only = marker.get("_applyBlackboardBySkill") and not any(str(c.get("prefabKey")) == key for c in candidates)
            if not skill_only and (key not in actor.talent_blackboards and key not in selected
                                   or chosen is not None and key not in selected):
                continue  # Locked/unselected talents are not missing support.
            issue = item.get("issue")
            if not marker.get("m_Enabled", 1):
                continue
            if (marker.get("m_Script") or {}).get("m_PathID") != TALENT_SCRIPT:
                issue = issue or "talent_component_class"
            if item.get("mode") is not None and mode is None:
                issue = issue or "talent_mode_handoff"
            skill_indices = marker.get("_checkValidBySkillIndex") or []
            if (not isinstance(skill_indices, list) or any(type(index) is not int or index < 0 for index in skill_indices)
                    or skill_indices and actor.skill_index is None):
                issue = issue or "talent_skill_index_binding"
            elif skill_indices and actor.skill_index not in skill_indices:
                continue
            if set(marker) - _TALENT_FIELDS:
                issue = issue or "talent_fields"
            supported = {"_applyBlackboardBySkill", "_applyStrBlackboardBySkill", "_applyTalentScale",
                         "_scaleCertainKeyFlag", "_scaleCertainKeyList", "_scaleCertainMode", "_scaleModeIndices"}
            if (any(marker.get(k) for k in _TALENT_FLAGS - supported) or marker.get("_defaultModeIndex", -1) != -1
                    or marker.get("_attackBlackboardModeIndex", -1) != -1):
                issue = issue or "talent_blackboard_binding"
            if key in overrides:
                issue = issue or "module_ability_override"
            board = battle._blackboard(selected[key]) if key in mode_keys and key in selected else actor.talent_blackboards.get(key, {})
            board, board_issue = talent_blackboard(marker, board, actor.skill_blackboard, actual_mode)
            issue = issue or board_issue
            bindings.append((item, key, issue, board, mode_dependent))
        for item, key, issue, board, mode_dependent in bindings:
            if item.get("mode") is not None and mode is not None and any(x.get("mode") is None and x.get("key") == key
                                        for x in entry.get("talents", [])):
                issue = issue or "mode_root_talent_collision"
            if issue is None and sum(k == key and problem is None for _, k, problem, _, _ in bindings) > 1:
                issue = "duplicate_root_talent_binding"
            self.bind(battle, actor, item["resource"], board, reason=issue,
                binding={"kind": "talent", "prefab_key": key, "mode": item.get("mode"),
                         "mode_dependent": mode_dependent, "key_evidence": item.get("key_evidence"),
                         "blackboard_evidence": "talent_skill_mode_candidate"})
            if issue is None:
                battle.mechanic_diagnostics["COMPONENT_TALENT_KEY_CANDIDATE"] = 1
                if mode_dependent or any(item["marker"].get(k) for k in supported):
                    battle.mechanic_diagnostics["COMPONENT_TALENT_BLACKBOARD_CANDIDATE"] = 1

    @staticmethod
    def detach(actor):
        for listener in actor.component_listeners:
            listener.record["state"] = "detached"
            if listener.instance is not None:
                actor.buffs.finish(listener.instance)
        actor.component_listeners.clear()
        # Claims remain on the departed instance, preventing legacy fallback
        # for its already emitted packets after owned listeners are detached.

    def switch_operator_mode(self, battle, actor, mode: int) -> bool:
        """Replace only mode talents at an explicit, supported transition.

        This does not infer a native mode from skill_active/skill_mode or from
        the attack-blackboard index. Complex form transitions remain separate.
        """
        if type(mode) is not int or mode < 0 or actor.dead:
            return False
        entry = self.actors.get(actor.char_id)
        if entry is not None:
            # Old directories lack the explicit list. Only indices already
            # proven by their talent ancestry (plus base 0) are eligible.
            modes = entry.get("modes", [0, *[x["mode"] for x in entry.get("talents", []) if x.get("mode") is not None]])
            if entry.get("mode_issue") or mode not in modes:
                battle.mechanic_diagnostics["COMPONENT_MODE_REFERENCE_UNRESOLVED"] = 1
                return False
        if actor.component_mode_index == mode:
            return True
        outgoing = [x for x in actor.component_listeners
                    if (x.record.get("binding") or {}).get("mode") is not None
                    or (x.record.get("binding") or {}).get("mode_dependent")]
        battle._dispatch_component_lifecycle(actor, "ON_BUFF_FINISH", listeners=outgoing)
        for listener in outgoing:
            listener.record["state"] = "detached"
            if listener in actor.component_listeners:
                actor.component_listeners.remove(listener)
        # Claims owned solely by this mode cease to suppress its old adapter.
        # Root/module/skill claims, including rejected listeners, stay intact.
        actor.component_claims.difference_update(actor.component_mode_claims | {x.template for x in outgoing})
        actor.component_claims.update(x.template for x in actor.component_listeners)
        actor.component_rejected_buffs.difference_update(actor.component_mode_rejections)
        actor.component_mode_claims.clear()
        actor.component_mode_rejections.clear()
        actor.component_mode_index = mode
        actor.mode_index = mode
        selection = actor.component_talent_selection
        if selection is not None:
            claims, rejected = set(actor.component_claims), set(actor.component_rejected_buffs)
            old = len(actor.component_listeners)
            self.bind_operator(battle, actor, module_bindings=selection[0], selected_talents=selection[1], mode=mode)
            actor.component_mode_claims = actor.component_claims - claims
            actor.component_mode_rejections = actor.component_rejected_buffs - rejected
            if any(unit is actor for unit in battle.operators):
                battle._dispatch_component_lifecycle(actor, "ON_BUFF_START", listeners=actor.component_listeners[old:])
        battle.mechanic_diagnostics["COMPONENT_MODE_HANDOFF_PHASE_CANDIDATE"] = 1
        return True

    def explicit_skill_mode(self, actor, *, active: bool) -> int | None:
        """Read the reviewed toggle controller, never an attack BB index."""
        entry = self.resources.get(actor.skill_prefab_id, {})
        controllers = [x.get("data", {}) for x in entry.get("components", [])
                       if "_extraAbilities" in x.get("data", {}) and "_maxTriggerTime" in x.get("data", {})]
        if len(controllers) != 1:
            return None
        controller = controllers[0]
        if ((controller.get("m_Script") or {}).get("m_PathID") != TOGGLE_SKILL_SCRIPT
                or not controller.get("m_Enabled", 1) or controller.get("_useAbilityHandleToggle")
                or controller.get("_switchToModeDown") or controller.get("_defaultModeIndex", -1) != -1):
            return None
        mode = controller.get("_switchToMode") if active else 0
        return mode if type(mode) is int and mode >= 0 else None

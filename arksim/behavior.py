"""Interpreter for Arknights ``buff_template_data.json`` action nodes.

The client data stores event handlers as small behaviour trees.  This module
owns their control flow, blackboard access and diagnostics; the battle engine
only supplies entity lookup and side-effect adapters such as damage or buffs.
"""

from __future__ import annotations

import math
import struct
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


def rounded_sp_ratio(base: float, ratio: float) -> int:
    """Default Q32.32 SP ratio path: float32 ratio, nearest-even integer."""
    scale = 1 << 32
    ratio32 = struct.unpack("<f", struct.pack("<f", float(ratio)))[0]
    raw = (math.trunc(float(base) * scale) * math.trunc(ratio32 * scale)) >> 32
    integer, fraction = divmod(raw, scale)
    return integer + int(fraction > scale // 2 or
                         (fraction == scale // 2 and integer % 2 != 0))


def passive_buff_spec_issue(spec: dict[str, Any]) -> str | None:
    """Keep the first registered slice limited to passive, single-stack flags."""
    attributes = spec.get("attributes") or {}
    passive = {"CAMOUFLAGE", "INVISIBLE", "UNTARGETABLE", "INVINCIBLE",
               "HEAL_FREE", "SP_BLOCKED"}
    issues = [key for key in ("loadFromDB", "isDurableBuff", "isDamageMissable",
        "isSilenceable", "isStunnable", "isFreezable", "isLevitatable",
        "isGroundBoundable", "disableOverride", "overrideKey", "stripBlackboardParamsWithBuffKey")
        if spec.get(key)]
    issues += [key for key in ("attributeModifiers", "abnormalImmunes", "abnormalAntis",
        "abnormalCombos", "abnormalComboImmunes") if attributes.get(key)]
    issues += [str(flag) for flag in (attributes.get("abnormalFlags") or []) if flag not in passive]
    if spec.get("maxStackCnt", 1) != 1:
        issues.append("maxStackCnt")
    for key in ("refreshRemainingTimeWhenStackMax", "clearAllStackCntWhenTimeUp",
                "durationKey", "remainingTimeKey"):
        if spec.get(key):
            issues.append(key)
    if spec.get("maxValidStackCnt", -1) not in {-1, 1}:
        issues.append("maxValidStackCnt")
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
    """Reviewed single-stack attributes, with an empty action template."""
    modifiers = (spec.get("attributes") or {}).get("attributeModifiers") or []
    if not modifiers:
        return passive_buff_spec_issue(spec)
    if not 1 <= len(modifiers) <= 3 or not all(isinstance(m, dict) for m in modifiers):
        return "attributeModifiers:count"
    formulas = {"ATK": "MULTIPLIER", "DEF": "MULTIPLIER", "ATTACK_SPEED": "ADDITION"}
    used = set()
    for modifier in modifiers:
        attribute = modifier.get("attributeType")
        if attribute not in formulas or modifier.get("formulaItem") != formulas[attribute] or attribute in used:
            return "attributeModifiers:attribute/formula"
        used.add(attribute)
        if modifier.get("fetchBaseValueFromSourceEntity"):
            return "attributeModifiers:source_base"
        if set(modifier) - {"attributeType", "formulaItem", "value", "loadFromBlackboard",
                             "fetchBaseValueFromSourceEntity"}:
            return "attributeModifiers:variant"
    if spec.get("templateKey", "empty") != "empty":
        return "attributeModifiers:template"
    if spec.get("lifeTimeType") not in {"LIMITED", "INFINITY"}:
        return "attributeModifiers:lifetime"
    if spec.get("overrideType", "DEFAULT") != "DEFAULT" or spec.get("independentCharacterSource"):
        return "attributeModifiers:override/source"
    if spec.get("statusResistable", "AUTOMATIC") != "AUTOMATIC" or spec.get("triggerInterval", -1) != -1:
        return "attributeModifiers:status/trigger"
    for key in ("takeSnapshotWhenExtend", "overrideEffectKey", "overrideOnEventPriority",
                "audioSignal", "priorityBBKeys", "enableInitDirectionFromSource"):
        if spec.get(key):
            return "attributeModifiers:" + key
    if spec.get("priority", -1) != -1 or spec.get("onEventPriority", "DEFAULT") != "DEFAULT":
        return "attributeModifiers:priority"
    copied = {**spec, "attributes": {**(spec.get("attributes") or {}), "attributeModifiers": []},
              "durationKey": None, "stripBlackboardParamsWithBuffKey": False}
    if spec.get("lifeTimeType") == "INFINITY" and spec.get("durationKey"):
        return "attributeModifiers:infinite_duration"
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
            "AdvancedApplyDamage": self._advanced_apply_damage,
            "AOEDamage": self._aoe_damage,
            "AssignBuffBlackboardFromOthers": self._assign_buff_blackboard,
            "AssignCharacterSkillBlackboardToBB": self._assign_skill_blackboard,
            "AssignValueToBB": self._assign_value,
            "AtkScaleUp": self._atk_scale_up,
            "BlackboardAdd": self._blackboard_add,
            "CalculateBlackboardValueViaParams": self._calculate_blackboard,
            "CheckContainsBuff": self._check_contains_buff,
            "CheckContainsDerviedBuff": self._check_contains_derived_buff,
            "CheckBlocked": self._check_blocked,
            "CheckEnemyLevelMask": self._check_enemy_level_mask,
            "CheckTargetTokenOrHost": self._check_token_host,
            "CreateBuff": self._create_buff,
            "CreateBuffInRange": self._create_buff_in_range,
            "DamageScale": self._damage_scale,
            "DamageViaAttr": self._damage_via_attr,
            "DamageViaMaxHpRatio": self._damage_via_max_hp_ratio,
            "Dice": self._dice,
            "EmitProjectileUseAbilitySelector": self._emit_projectile,
            "FilterAbilityFamily": self._filter_ability_family,
            "FinishBuff": self._finish_buff,
            "FinishBuffsById": self._finish_buffs_by_id,
            "IfElse": self._if_else,
            "IfTarget": self._if_target,
            "IsBlackboardZero": self._is_blackboard_zero,
            "IsHeal": self._is_heal,
            "IsTargetInEPBreakRecovery": self._in_element_recovery,
            "ModifySp": self._modify_sp,
            "Not": self._not_condition,
            "RandomSetter": self._random_setter,
            "RemainingRatioToAttributeModifier": self._remaining_ratio_modifier,
            "SetAtkScaleZero": self._set_atk_scale_zero,
            "SwitchMode": self._switch_mode,
        }

    @property
    def supported_nodes(self) -> frozenset[str]:
        return frozenset(self.handlers) | self.FLOW_NODES

    def _unsupported(self, key: str) -> bool:
        self.diagnostics.unsupported_nodes[key] += 1
        if self.strict:
            raise UnknownNodeError(f"unsupported behavior {key}")
        return False

    @staticmethod
    def _unsupported_variant(node: dict[str, Any]) -> str | None:
        name = node_name(node)
        if name == "AOEDamage":
            roles = {"SOURCE", "TARGET", "BUFF_OWNER", "BUFF_SOURCE", "MODIFIER_SOURCE", "MODIFIER_TARGET"}
            if node.get("_sourceType") not in roles or node.get("_targetType") not in roles:
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
            roles = {"SOURCE", "TARGET", "BUFF_OWNER", "BUFF_SOURCE", "MODIFIER_SOURCE", "MODIFIER_TARGET"}
            if node.get("_sourceType") not in roles or node.get("_targetType") not in roles:
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
            if not node.get("_buffs"):
                return f"{name}:empty_buffs"
            for spec in node.get("_buffs") or []:
                issue = passive_buff_spec_issue(spec)
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
        if node_name(node) == "ModifySp":
            # Negative notifications and custom keys need their modifier paths
            # before those variants can be enabled.
            for key in ("_isMinis", "_customModifierKey"):
                if node.get(key):
                    return f"ModifySp:{key}"
        if node_name(node) == "CreateBuff":
            if bool(node.get("_isDerivedBuff")) != bool(node.get("_finishDerivedBuffIfParentFinish")):
                return "CreateBuff:derived_lifecycle"
            if node.get("_isDerivedBuff") and (node.get("_buffOwner") != "BUFF_OWNER" or node.get("_useSpecialBuffSource")):
                return "CreateBuff:derived_owner/source"
            for key in ("_inherateSourceProjectileFromParentBuff", "_hookEffectKeyFromSourceProjectile"):
                if node.get(key):
                    return "CreateBuff:" + key
            issue = registered_buff_spec_issue(node.get("_buff") or {})
            if issue:
                return f"CreateBuff:{issue}"
        if node_name(node) == "FinishBuff" and node.get("_decCntIfStack"):
            return "FinishBuff:decrement_stack"
        return None

    def validate_event(self, template_key: str, event: str) -> bool:
        """Reject incomplete migrated events before any actions take effect."""
        template = self.templates.get(template_key)
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
        template = self.templates.get(template_key)
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
        board = boards.get(buff_key) if isinstance(boards, dict) else None
        if board is None:
            container = getattr(entity, "buffs", None)
            instances = [item for item in getattr(container, "active", ())
                if item.definition.key == buff_key and item.enabled and not item.finished
                and not container._is_pending_finish(item)]
            if len(instances) > 1:
                return self._unsupported("AssignBuffBlackboardFromOthers:multiple_sources")
            if instances:
                board = instances[0].blackboard
        if not isinstance(board, dict) or source_key not in board:
            return False
        context.blackboard[target_key] = board[source_key]
        return True

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

    def _aoe_damage(self, node: dict[str, Any], context: BehaviorContext) -> bool:
        issue = self._unsupported_variant(node)
        if issue:
            return self._unsupported(issue)
        if context.battle is None:
            return False
        return bool(context.battle.behavior_aoe_damage(context, node))

    # --------------------------------------------------------------- buffs
    def _create_buff(self, node: dict[str, Any], context: BehaviorContext) -> bool:
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

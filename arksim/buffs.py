"""Small, deterministic runtime for owner-local enemy and operator Buffs.

The native client has a considerably larger Buff system.  This module keeps
the first simulator slice deliberately narrow: a Buff is an immutable
definition plus a mutable instance, instances live in an owner-local
container, and container changes are committed outside the active-list
traversal.  Event actions are intentionally represented as data and are
executed by the existing behaviour interpreter supplied by Battle.
"""

from __future__ import annotations

import copy
import math
import struct
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable


BUFF_TIMER_EPSILON = 1e-9
LIFETIME_IMMEDIATELY = "IMMEDIATELY"
LIFETIME_LIMITED = "LIMITED"
LIFETIME_INFINITY = "INFINITY"


def _initial_override_priority(values: Iterable[float], *, absolute: bool = False) -> int:
    """Native float32 -> Q32.32 sum, times 1000, then nearest-even integer."""
    scale = 1 << 32
    total = 0
    for value in values:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("invalid automatic priority value")
        value32 = struct.unpack("<f", struct.pack("<f", float(value)))[0]
        if not math.isfinite(value32) or not -(1 << 31) < value32 < (1 << 31):
            raise ValueError("automatic priority value overflow")
        raw = math.trunc(value32 * scale)
        total += abs(raw) if absolute else raw
        if not -(1 << 63) < total < (1 << 63):
            raise ValueError("automatic priority sum overflow")
    raw = total * 1000
    if not -(1 << 63) <= raw < (1 << 63):
        raise ValueError("automatic priority overflow")
    integer, fraction = divmod(raw, scale)
    result = integer + int(fraction > scale // 2 or fraction == scale // 2 and integer % 2 != 0)
    if not -2147483648 <= result <= 2147483647:
        raise ValueError("automatic priority overflow")
    return result


def element_break_buff_keys(table: dict[str, Any]) -> frozenset[str]:
    """Use the native exact-key lists, never infer a child from its prefix."""
    if not isinstance(table, dict):
        raise ValueError("element break table must be a mapping")
    keys = set()
    for entry in table.values():
        if not isinstance(entry, dict) or not isinstance(entry.get("elementBuffs"), list):
            raise ValueError("element break entry requires elementBuffs")
        if any(not isinstance(key, str) or not key for key in entry["elementBuffs"]):
            raise ValueError("element break Buff keys must be nonempty strings")
        for field_name in ("elementBreakDuration", "enemyElementBreakDuration"):
            value = entry.get(field_name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("element break durations must be finite and positive")
        keys.update(entry["elementBuffs"])
    return frozenset(keys)


@dataclass(frozen=True)
class BuffModifier:
    """One additive or multiplicative modifier supplied by a Buff."""

    attribute: str
    formula: str
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "attribute", str(self.attribute).upper())
        object.__setattr__(self, "formula", str(self.formula).upper())
        object.__setattr__(self, "value", float(self.value))


def calculate_attribute(base: float, modifiers: Iterable[BuffModifier], *,
                        multiplier: float = 1.0, final_addition: float = 0.0,
                        final_scale: float = 1.0) -> float:
    """Combine reviewed attribute pools; entity limits belong to the caller."""
    additions = 0.0
    for item in modifiers:
        if item.formula == "ADDITION":
            additions += item.value
        elif item.formula == "MULTIPLIER":
            multiplier += item.value
        elif item.formula == "FINAL_ADDITION":
            final_addition += item.value
        elif item.formula == "FINAL_SCALER":
            final_scale *= item.value + 1.0 if item.value < 0 else item.value
    return ((float(base) + additions) * max(multiplier, 0.0) + final_addition) * final_scale


@dataclass(frozen=True)
class BuffDefinition:
    """Immutable data/configuration for a runtime Buff."""

    key: str
    lifetime_type: str = LIFETIME_INFINITY
    lifetime: float = 0.0
    trigger_interval: float = -1.0
    wait_first_trigger_interval: bool = True
    modifiers: tuple[BuffModifier, ...] = ()
    events: tuple[tuple[str, tuple[dict[str, Any], ...]], ...] = ()
    unique_by_key: bool = True
    status_flags: tuple[str, ...] = ()
    template_key: str = ""
    registered_behavior: bool = False
    first_trigger_interval: float = -1.0
    trigger_limit: int | None = None
    override_policy: str | None = None
    override_key: str = ""
    disable_override: bool = False
    max_stack_count: int = 1
    clear_stack_on_expiry: bool = False
    override_priority: int = -1
    status_resistable: str = "AUTOMATIC"
    is_silenceable: bool = False
    is_stunnable: bool = False
    element_break: bool = False
    is_damage_missable: bool = False
    effect_key: str = ''
    audio_signal: str = ''

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", str(self.key))
        object.__setattr__(
            self,
            "lifetime_type",
            str(self.lifetime_type).upper(),
        )
        object.__setattr__(self, "lifetime", float(self.lifetime))
        object.__setattr__(self, "trigger_interval", float(self.trigger_interval))
        object.__setattr__(self, "first_trigger_interval", float(self.first_trigger_interval))
        if self.status_resistable not in {"NO", "YES", "AUTOMATIC"}:
            raise ValueError("unsupported Buff resistance mode")
        if (self.is_silenceable or self.is_stunnable) and (
                self.lifetime_type not in {LIFETIME_LIMITED, LIFETIME_INFINITY} or
                self.override_policy is not None and not self.disable_override):
            raise ValueError("unbound responsive lifetime/override")
        if self.trigger_limit is not None and (type(self.trigger_limit) is not int or not 0 <= self.trigger_limit <= 2147483647):
            raise ValueError("Buff trigger limit must be a nonnegative integer")
        if self.override_policy not in {None, "DEFAULT", "STACK", "UNIQUE", "EXTEND", "EXTEND_TIME"}:
            raise ValueError("unsupported Buff override policy")
        object.__setattr__(self, "override_key", str(self.override_key or self.key))
        if self.override_policy is not None and (
                (not self.disable_override and self.override_policy in {"STACK", "EXTEND", "EXTEND_TIME"}
                    and (self.trigger_interval != -1 or self.trigger_limit is not None))
                or self.lifetime_type not in {LIFETIME_LIMITED, LIFETIME_INFINITY}
                or self.override_policy in {"EXTEND", "EXTEND_TIME"} and self.lifetime_type != LIFETIME_LIMITED):
            raise ValueError("override requires a reviewed lifetime/trigger clock")
        if self.override_policy == "STACK" and (type(self.max_stack_count) is not int or not 1 <= self.max_stack_count <= 2147483647):
            raise ValueError("invalid Buff maximum stack count")
        if self.override_policy == "DEFAULT" and (
                type(self.override_priority) is not int or not -2147483648 <= self.override_priority <= 2147483647):
            raise ValueError("default override requires integer priority")

    def actions_for(self, event: str) -> tuple[dict[str, Any], ...]:
        wanted = str(event).upper()
        for name, actions in self.events:
            if name.upper() == wanted:
                return actions
        return ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BuffDefinition":
        """Build a normalized definition from either compact or client-like data."""
        key = str(payload.get("key") or payload.get("buffKey") or "")
        if not key:
            raise ValueError("BuffDefinition requires a key")
        raw_modifiers = payload.get("modifiers")
        if raw_modifiers is None:
            raw_modifiers = (
                (payload.get("attributes") or {}).get("attributeModifiers")
                or []
            )
        modifiers = tuple(
            BuffModifier(
                attribute=str(item.get("attribute") or item.get("attributeType", "")),
                formula=str(item.get("formula") or item.get("formulaItem", "ADDITION")),
                value=float(item.get("value", 0.0) or 0.0),
            )
            for item in raw_modifiers
            if isinstance(item, dict)
            and (item.get("attribute") or item.get("attributeType"))
        )
        raw_events = payload.get("events")
        if raw_events is None:
            raw_events = payload.get("eventToActions") or {}
        events: list[tuple[str, tuple[dict[str, Any], ...]]] = []
        if isinstance(raw_events, dict):
            for event, actions in raw_events.items():
                if not isinstance(actions, list):
                    continue
                events.append(
                    (
                        str(event).upper(),
                        tuple(
                            copy.deepcopy(action)
                            for action in actions
                            if isinstance(action, dict)
                        ),
                    )
                )
        return cls(
            key=key,
            lifetime_type=str(
                payload.get("lifetime_type")
                or payload.get("lifeTimeType")
                or LIFETIME_INFINITY
            ),
            lifetime=float(
                payload.get("lifetime", payload.get("lifeTime", 0.0)) or 0.0
            ),
            trigger_interval=float(
                payload.get(
                    "trigger_interval",
                    payload.get("triggerInterval", -1.0),
                )
                or -1.0
            ),
            first_trigger_interval=float(payload.get("firstTriggerInterval", -1)),
            trigger_limit=(payload.get("triggerCnt", 0) if payload.get("triggerLifeType") == "LIMITED"
                           else payload.get("trigger_limit")),
            wait_first_trigger_interval=bool(
                payload.get(
                    "wait_first_trigger_interval",
                    payload.get("waitFirstTriggerInterval", True),
                )
            ),
            modifiers=modifiers,
            events=tuple(events),
            unique_by_key=bool(
                payload.get(
                    "unique_by_key",
                    payload.get("uniqueByKey", True),
                )
            ),
            status_flags=tuple(str(flag).upper() for flag in payload.get("status_flags", ())),
            template_key=str(payload.get("templateKey") or ""),
            registered_behavior=bool(payload.get("registered_behavior", False)),
            override_policy=((payload.get("overrideType", "DEFAULT") if payload.get("overrideType", "DEFAULT") != "DEFAULT"
                or type(payload.get("priority")) is int and payload["priority"] >= 0
                or payload.get("calculated_priority", False) else None)
                if payload.get("registered_behavior") else None),
            override_key=str(payload.get("overrideKey") or key),
            disable_override=bool(payload.get("disableOverride", False)),
            max_stack_count=payload.get("maxStackCnt", 1),
            clear_stack_on_expiry=bool(payload.get("clearAllStackCntWhenTimeUp", False)),
            override_priority=payload.get("priority", -1),
            status_resistable=payload.get("statusResistable", "AUTOMATIC"),
            is_silenceable=bool(payload.get("isSilenceable", False)),
            is_stunnable=bool(payload.get("isStunnable", False)),
            element_break=bool(payload.get("element_break", False)),
            is_damage_missable=bool(payload.get("isDamageMissable", False)),
            effect_key=payload.get('overrideEffectKey') or '',
            audio_signal=payload.get('audioSignal') or '',
        )

    @classmethod
    def from_create_spec(
        cls,
        spec: dict[str, Any],
        *,
        blackboard: dict[str, Any] | None = None,
    ) -> "BuffDefinition":
        """Normalize a behaviour-tree ``CreateBuff`` payload.

        A few exported client modifiers use ``loadFromBlackboard`` instead of
        storing the value in the modifier itself.  The compact lookup here is
        intentionally conservative and is only used by the first supported
        attribute nodes.
        """
        board = blackboard or {}
        normalized = copy.deepcopy(spec)
        if (normalized.get("registered_behavior") and normalized.get("overrideType", "DEFAULT") == "DEFAULT"
                and normalized.get("priority", -1) == -1 and normalized.get("priorityBBKeys")):
            keys = normalized["priorityBBKeys"]
            if not isinstance(keys, list) or any(not isinstance(key, str) or not key or key not in board for key in keys):
                raise ValueError("unbound priority blackboard keys")
            normalized["priority"] = _initial_override_priority((board[key] for key in keys), absolute=True)
        modifiers = []
        attributes = normalized.get("attributes") or {}
        for raw in attributes.get("attributeModifiers") or []:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            if item.get("loadFromBlackboard"):
                attribute = str(
                    item.get("attribute") or item.get("attributeType", "")
                ).lower()
                candidates = [
                    item.get("blackboardKey"),
                    item.get("key"),
                    attribute,
                    str(item.get("attributeType", "")).lower(),
                    "value",
                ]
                for candidate in candidates:
                    if candidate is not None and str(candidate) in board:
                        item["value"] = board[str(candidate)]
                        break
            modifiers.append(item)
        normalized["modifiers"] = modifiers
        if normalized.get("registered_behavior"):
            trigger_life = normalized.get("triggerLifeType")
            if trigger_life == "IMMEDIATELY":
                # An inactive native trigger ignores the blackboard clock keys.
                normalized["triggerInterval"] = -1
                normalized["firstTriggerInterval"] = -1
            elif trigger_life in {"LIMITED", "INFINITY"}:
                normalized["triggerInterval"] = board.get("interval", normalized.get("triggerInterval", -1))
                normalized["firstTriggerInterval"] = board.get("first_trigger_interval",
                    normalized.get("firstTriggerInterval", -1))
                interval = normalized["triggerInterval"]
                if type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0:
                    raise ValueError("unbound Buff trigger interval")
                if "trig_cnt" in board:
                    budget = board["trig_cnt"]
                    if (type(budget) not in (int, float) or not math.isfinite(budget)
                            or not 0 <= budget <= 2147483647 or budget != int(budget)):
                        raise ValueError("invalid blackboard trigger count")
                    normalized["triggerLifeType"] = "LIMITED"
                    normalized["triggerCnt"] = int(budget)
            if (normalized.get("overrideType", "DEFAULT") == "DEFAULT"
                    and normalized.get("priority", -1) < 0
                    and not normalized.get("priorityBBKeys") and modifiers):
                normalized["priority"] = _initial_override_priority(item.get("value", 0) for item in modifiers
                    if item.get("formula", item.get("formulaItem")) in
                       {"ADDITION", "MULTIPLIER", "FINAL_ADDITION"})
                normalized["calculated_priority"] = True
            normalized["status_flags"] = attributes.get("abnormalFlags") or []
            duration_key = normalized.get("durationKey") or "duration"
            if normalized.get("lifeTimeType") == LIFETIME_LIMITED:
                normalized["lifeTime"] = board.get(duration_key, normalized.get("lifeTime", 0))
            if any(isinstance(item.get("value", 0), bool) or not math.isfinite(float(item.get("value", 0))) for item in modifiers):
                raise ValueError("non-finite Buff modifier")
            if isinstance(normalized.get("lifeTime", 0), bool):
                raise ValueError("invalid Buff lifetime")
            lifetime = float(normalized.get("lifeTime", 0))
            if not math.isfinite(lifetime) or lifetime < 0:
                raise ValueError("invalid Buff lifetime")
            interval = normalized.get("triggerInterval", -1)
            if isinstance(interval, bool) or not math.isfinite(float(interval)) or (interval != -1 and interval <= 0):
                raise ValueError("invalid Buff trigger interval")
            first = normalized.get("firstTriggerInterval", -1)
            if isinstance(first, bool) or not math.isfinite(float(first)) or (first != -1 and first < 0):
                raise ValueError("invalid Buff first trigger interval")
        normalized.pop("attributes", None)
        return cls.from_dict(normalized)


@dataclass(eq=False)
class BuffInstance:
    """Mutable owner-local state for one ``BuffDefinition``."""

    definition: BuffDefinition
    owner: Any
    source: Any = None
    parent: "BuffInstance | None" = None
    finish_with_parent: bool = False
    blackboard: dict[str, Any] = field(default_factory=dict)
    attached_step: int | None = None
    remaining_lifetime: float | None = field(init=False)
    trigger_timer: float | None = field(init=False)
    trigger_count: int = 0
    remaining_triggers: int | None = field(init=False)
    stack_count: int = 1
    enabled: bool = True
    response_disabled: bool = False
    frozen: bool = False
    freeze_sources: list[Any] = field(default_factory=list, repr=False)
    start_clock_scale: float = 1.0
    finished: bool = False
    created_order: int = 0
    component_listener: Any = None
    container: "BuffContainer | None" = field(default=None, repr=False)
    derived_children: list["BuffInstance"] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.remaining_triggers = self.definition.trigger_limit
        lifetime_type = self.definition.lifetime_type
        if lifetime_type == LIFETIME_INFINITY:
            self.remaining_lifetime = None
        elif lifetime_type == LIFETIME_LIMITED:
            self.remaining_lifetime = max(self.definition.lifetime, 0.0)
        elif lifetime_type == LIFETIME_IMMEDIATELY:
            self.remaining_lifetime = 0.0
        else:
            raise ValueError(f"unsupported Buff lifetime type: {lifetime_type}")
        if self.definition.trigger_interval > BUFF_TIMER_EPSILON:
            self.trigger_timer = (
                self.definition.first_trigger_interval if self.definition.first_trigger_interval >= 0 else
                self.definition.trigger_interval
                if self.definition.wait_first_trigger_interval
                else 0.0
            )
        else:
            self.trigger_timer = None

    @property
    def clock_frozen(self) -> bool:
        return self.frozen or bool(self.freeze_sources)


BuffEventHandler = Callable[[BuffInstance, str], Any]


@dataclass
class BuffContainer:
    """Committed Buffs owned by one entity.

    ``active`` is never changed while ``tick`` is traversing it.  Event
    handlers therefore enqueue changes into the two pending collections and a
    later ``commit`` makes their visibility boundary explicit.
    """

    owner: Any = None
    event_handler: BuffEventHandler | None = None
    change_handler: Callable[["BuffContainer"], None] | None = field(default=None, repr=False)
    active: list[BuffInstance] = field(default_factory=list)
    pending_add: list[BuffInstance] = field(default_factory=list)
    pending_finish: list[BuffInstance] = field(default_factory=list)
    created_counter: int = 0
    current_step: int | None = None
    _related_commits: list["BuffContainer"] = field(default_factory=list, repr=False)
    _committing: bool = field(default=False, repr=False)

    @staticmethod
    def live_parent(parent: BuffInstance | None) -> bool:
        container = parent.container if parent is not None else None
        return bool(container is not None and parent.enabled and not parent.finished
            and not container._is_pending_finish(parent)
            and any(item is parent for item in (*container.active, *container.pending_add)))

    def _queue_related_commit(self, container: "BuffContainer") -> None:
        if container is not self and not any(item is container for item in self._related_commits):
            self._related_commits.append(container)

    @staticmethod
    def _unlink_parent(instance: BuffInstance) -> None:
        if instance.finish_with_parent and instance.parent is not None:
            children = instance.parent.derived_children
            children[:] = [child for child in children if child is not instance]

    def _find_identity(self, collection: Iterable[BuffInstance], target: BuffInstance):
        return next((item for item in collection if item is target), None)

    @staticmethod
    def _source_equal(left: Any, right: Any) -> bool:
        return left is right

    def _is_pending_finish(self, instance: BuffInstance) -> bool:
        return self._find_identity(self.pending_finish, instance) is not None

    def _find_existing(
        self,
        definition: BuffDefinition,
        source: Any,
        parent: BuffInstance | None = None,
        finish_with_parent: bool = False,
    ) -> BuffInstance | None:
        for instance in (*self.active, *self.pending_add):
            if instance.definition.disable_override:
                continue
            if instance.definition.override_policy is not None and definition.override_policy is None:
                continue
            if instance.finished or not instance.enabled:
                continue
            if self._is_pending_finish(instance):
                continue
            if instance.definition.key != definition.key:
                continue
            if instance.finish_with_parent != finish_with_parent:
                continue
            if finish_with_parent and instance.parent is not parent:
                continue
            if definition.unique_by_key or self._source_equal(instance.source, source):
                return instance
        return None

    def override_issue(self, definition: BuffDefinition) -> str | None:
        if definition.disable_override:
            return None
        for instance in (*self.active, *self.pending_add):
            old = instance.definition
            if (instance.finished or not instance.enabled and definition.override_policy != "DEFAULT" or self._is_pending_finish(instance)
                    or old.disable_override or old.override_key != definition.override_key):
                continue
            if old.override_policy != definition.override_policy:
                return "override_policy_conflict"
            if definition.override_policy in {"EXTEND", "EXTEND_TIME"} and instance.remaining_lifetime is None:
                return "override_lifetime_binding"
            if definition.override_policy == "STACK":
                if ((old.lifetime_type, old.lifetime, old.modifiers, old.status_flags, old.template_key, old.events,
                        old.max_stack_count, old.clear_stack_on_expiry, old.status_resistable, old.element_break) !=
                        (definition.lifetime_type, definition.lifetime, definition.modifiers,
                         definition.status_flags, definition.template_key, definition.events,
                         definition.max_stack_count, definition.clear_stack_on_expiry, definition.status_resistable, definition.element_break)):
                    return "stack_prototype_binding"
                if instance.stack_count < old.max_stack_count and any(not math.isfinite(
                        item.value * (instance.stack_count + 1)) for item in old.modifiers):
                    return "stack_attribute_overflow"
            if (definition.override_policy == "EXTEND_TIME" and not math.isfinite(
                    instance.remaining_lifetime + definition.lifetime)):
                return "override_time_overflow"
        return None

    def add(
        self,
        definition: BuffDefinition,
        *,
        source: Any = None,
        parent: BuffInstance | None = None,
        blackboard: dict[str, Any] | None = None,
        attached_step: int | None = None,
        finish_with_parent: bool = False,
    ) -> BuffInstance:
        """Queue a Buff for the next commit, preserving stable creation order."""
        if finish_with_parent and not self.live_parent(parent):
            raise ValueError("derived Buff requires a live registered parent container")
        if definition.override_policy not in {None, "DEFAULT"} and parent is not None and not definition.disable_override:
            raise ValueError("override parent binding is not reviewed")
        if issue := self.override_issue(definition):
            raise ValueError(issue)
        existing = (None if definition.override_policy is not None or definition.disable_override else
                    self._find_existing(definition, source, parent, finish_with_parent))
        if existing is not None:
            return existing
        self.created_counter += 1
        instance = BuffInstance(
            definition=definition,
            owner=self.owner,
            source=source,
            parent=parent,
            finish_with_parent=finish_with_parent,
            blackboard=dict(blackboard or {}),
            attached_step=(
                self.current_step if attached_step is None else attached_step
            ),
            created_order=self.created_counter,
            container=self,
        )
        self.pending_add.append(instance)
        if finish_with_parent:
            parent.derived_children.append(instance)
            parent.container._queue_related_commit(self)
        return instance

    def contains(self, key: str, *, source: Any = None) -> bool:
        wanted = str(key)
        for instance in self.active:
            if (
                not instance.finished
                and instance.enabled
                and not self._is_pending_finish(instance)
                and instance.definition.key == wanted
                and (source is None or self._source_equal(instance.source, source))
            ):
                return True
        return False

    def count(self, key: str | None = None, *, source: Any = None) -> int:
        return sum(
            1
            for instance in self.active
            if not instance.finished
            and instance.enabled
            and not self._is_pending_finish(instance)
            and (key is None or instance.definition.key == str(key))
            and (source is None or self._source_equal(instance.source, source))
        )

    def finish(self, instance: BuffInstance) -> bool:
        if instance.finished or instance.container is not self:
            return False
        if not self._find_identity(self.pending_finish, instance):
            self.pending_finish.append(instance)
            # Follow exact instances, including recipients in other containers.
            for child in tuple(instance.derived_children):
                if child.container is not None:
                    child.container.finish(child)
                    self._queue_related_commit(child.container)
        return True

    def finish_derived(self, parent: BuffInstance) -> bool:
        if parent.container is not self:
            return False
        children = tuple(parent.derived_children)
        parent.derived_children.clear()
        for child in children:
            if child.container is not None:
                child.container.finish(child)
                self._queue_related_commit(child.container)
        return True

    def contains_derived(self, parent: BuffInstance, key: str | None = None) -> bool:
        if parent.container is not self:
            return False
        return any(item.parent is parent and item.finish_with_parent
            and item.enabled and not item.finished and item.container is not None
            and not item.container._is_pending_finish(item)
            and (key is None or item.definition.key == key)
            for item in parent.derived_children)

    def flagged_instances(self, flag: str, *, pending: bool = False) -> tuple[BuffInstance, ...]:
        names = {flag, "STUN"} if flag == "STUNNED" else {flag}
        items = self.pending_add if pending else self.active
        return tuple(item for item in items if not item.finished and not self._is_pending_finish(item)
                     and names.intersection(item.definition.status_flags))

    def finish_by_key(self, key: str, *, source: Any = None) -> int:
        wanted = str(key)
        targets = [
            instance
            for instance in (*self.active, *self.pending_add)
            if instance.definition.key == wanted
            and not instance.finished
            and (source is None or self._source_equal(instance.source, source))
        ]
        for instance in targets:
            self.finish(instance)
        return len(targets)

    def finish_by_source(self, source: Any, *, key: str | None = None) -> int:
        targets = [
            instance
            for instance in (*self.active, *self.pending_add)
            if not instance.finished
            and self._source_equal(instance.source, source)
            and (key is None or instance.definition.key == str(key))
        ]
        for instance in targets:
            self.finish(instance)
        return len(targets)

    def _dispatch(self, instance: BuffInstance, event: str) -> None:
        if self.event_handler is not None:
            self.event_handler(instance, event)

    def _remove_identity(
        self, collection: list[BuffInstance], target: BuffInstance
    ) -> bool:
        for index, item in enumerate(collection):
            if item is target:
                del collection[index]
                return True
        return False

    def _commit_finishes(self) -> bool:
        if not self.pending_finish:
            return False
        batch = self.pending_finish
        self.pending_finish = []
        changed = False
        for instance in batch:
            if self._remove_identity(self.active, instance):
                if instance.definition.registered_behavior and instance.enabled:
                    self._set_enabled(instance, False)
                instance.finished = True
                instance.enabled = False
                if instance.definition.override_policy == "STACK":
                    instance.stack_count = 0
                self._unlink_parent(instance)
                self._dispatch(instance, "ON_BUFF_FINISH")
                changed = True
            elif self._remove_identity(self.pending_add, instance):
                instance.finished = True
                instance.enabled = False
                self._unlink_parent(instance)
                changed = True
        return changed

    def _set_enabled(self, instance: BuffInstance, enabled: bool) -> None:
        if instance.enabled == enabled:
            return
        instance.enabled = enabled
        self._dispatch(instance, "ON_BUFF_ENABLE" if enabled else "ON_BUFF_DISABLE")

    def _select_default_leaves(self) -> None:
        groups: dict[str, list[BuffInstance]] = {}
        for instance in self.active:
            if (instance.definition.override_policy == "DEFAULT" and not instance.definition.disable_override
                    and not instance.finished and not self._is_pending_finish(instance)):
                groups.setdefault(instance.definition.override_key, []).append(instance)
        for group in groups.values():
            winner = max(group, key=lambda item: (item.definition.override_priority, -item.created_order))
            # Withdraw losers before enabling a replacement, preserving flags.
            for instance in group:
                if instance is not winner:
                    self._set_enabled(instance, False)
            self._set_enabled(winner, True)

    def _commit_additions(self) -> bool:
        if not self.pending_add:
            return False
        batch = self.pending_add
        self.pending_add = []
        changed = False
        for instance in batch:
            if instance.finished or not instance.enabled:
                continue
            if self._is_pending_finish(instance):
                instance.finished = True
                instance.enabled = False
                self._unlink_parent(instance)
                continue
            if instance.finish_with_parent and not self.live_parent(instance.parent):
                instance.finished = True
                instance.enabled = False
                self._unlink_parent(instance)
                continue
            if not self._motion_application_allowed(instance.definition):
                instance.finished = True
                instance.enabled = False
                self._unlink_parent(instance)
                if instance.component_listener is not None:
                    instance.component_listener.record.update(state="cancelled", reason="motion_application_gate")
                continue
            policy = instance.definition.override_policy
            existing = (None if policy is not None or instance.definition.disable_override else self._find_existing(instance.definition, instance.source,
                instance.parent, instance.finish_with_parent))
            if existing is not None:
                instance.finished = True
                instance.enabled = False
                self._unlink_parent(instance)
                continue
            self.active.append(instance)
            motion_flags = set(instance.definition.status_flags) & {"LEVITATE", "GROUND_BOUND"}
            if motion_flags and float(getattr(self.owner, "mass_level", 0)) > 3:
                instance.start_clock_scale = 2.0 ** len(motion_flags)
            self._dispatch(instance, "ON_BUFF_START")
            self._update_response(instance)
            if policy is not None and not instance.definition.disable_override and not self._is_pending_finish(instance):
                old = next((item for item in self.active if item is not instance and item.enabled
                    and not item.finished and not self._is_pending_finish(item)
                    and not item.definition.disable_override
                    and item.definition.override_key == instance.definition.override_key), None)
                if old is not None and policy != "DEFAULT":
                    if policy == "STACK":
                        old.stack_count = min(old.stack_count + 1, old.definition.max_stack_count)
                    elif policy == "EXTEND":
                        old.remaining_lifetime = (max(old.remaining_lifetime, instance.definition.lifetime)
                            if instance.definition.lifetime > 0 else 0.0)
                    elif policy == "EXTEND_TIME":
                        old.remaining_lifetime += instance.definition.lifetime
                    self.finish(instance)
            if policy == "DEFAULT":
                self._select_default_leaves()
            if instance.definition.lifetime_type == LIFETIME_IMMEDIATELY:
                self.finish(instance)
            changed = True
        return changed

    def _motion_application_allowed(self, definition: BuffDefinition) -> bool:
        from .mechanics import Status
        flags = set(definition.status_flags)
        statuses = getattr(self.owner, "statuses", None)
        if statuses is None or not flags & {"LEVITATE", "GROUND_BOUND"}:
            return True
        flying = statuses.base_motion_flying
        lifted = statuses.has(Status.LEVITATE)
        grounded = statuses.has(Status.GROUND_BOUND)
        if "LEVITATE" in flags and not grounded and (
                flying or lifted and "FORCE_LEVITATE" not in flags):
            return False
        if "GROUND_BOUND" in flags and not lifted and (not flying or grounded):
            return False
        return True

    def commit(self) -> None:
        """Apply queued changes and any changes generated by lifecycle events."""
        # A lifecycle callback may enqueue another operation.  Flush those
        # operations in stable waves, never by mutating the list being walked.
        if self._committing:
            return
        self._committing = True
        try:
            while self.pending_finish or self.pending_add or self._related_commits:
                self._commit_finishes()
                self._commit_additions()
                self._select_default_leaves()
                related, self._related_commits = self._related_commits, []
                for container in related:
                    container.commit()
            if self.change_handler is not None:
                self.change_handler(self)
        finally:
            self._committing = False

    def _trigger(self, instance: BuffInstance) -> None:
        if instance.remaining_triggers == 0:
            return
        instance.trigger_count += 1
        self._dispatch(instance, "ON_BUFF_TRIGGER")
        # Exhaustion disables periodic callbacks, not the Buff lifetime or
        # its owned children. Consume attempted triggers even on a false action.
        if instance.remaining_triggers is not None:
            instance.remaining_triggers = max(instance.remaining_triggers - 1, 0)
        if instance.component_listener is not None:
            instance.component_listener.record["triggers"] = {
                "completed": instance.trigger_count, "remaining": instance.remaining_triggers}

    def tick(self, dt: float, frame_index: int | None = None) -> None:
        """Advance timers using the real delta, without committing mutations."""
        delta = max(float(dt), 0.0)
        self.current_step = frame_index
        for instance in tuple(self.active):
            if not instance.finished and not self._is_pending_finish(instance):
                self._update_response(instance)
            if (
                instance.finished
                or self._is_pending_finish(instance)
            ):
                continue
            if frame_index is not None and instance.attached_step == frame_index:
                continue
            if instance.clock_frozen:
                continue

            elapsed, trigger_delta = self._timer_deltas(instance, delta)
            remaining = instance.remaining_lifetime
            active_delta = trigger_delta * (1.0 if remaining is None or elapsed <= 0 else
                min(1.0, max(remaining, 0.0) / elapsed))
            if instance.remaining_lifetime is not None:
                instance.remaining_lifetime -= elapsed

            interval = instance.definition.trigger_interval
            if (instance.enabled and instance.trigger_timer is not None and interval > BUFF_TIMER_EPSILON
                    and (remaining is None or remaining > BUFF_TIMER_EPSILON)
                    and instance.remaining_triggers != 0):
                # A frame crossing expiry cannot emit a trigger beyond the
                # lifetime. An event exactly at the deadline precedes finish.
                instance.trigger_timer -= active_delta
                if instance.definition.registered_behavior:
                    # Native Buff.OnTick advances its PrecisePeriodicTimer once:
                    # carry overshoot to later ticks, never catch up in this tick.
                    if instance.trigger_timer <= BUFF_TIMER_EPSILON:
                        instance.trigger_timer += interval
                        self._trigger(instance)
                else:
                    while (
                        instance.trigger_timer <= BUFF_TIMER_EPSILON
                        and instance.enabled
                        and not instance.clock_frozen
                        and not instance.finished
                        and not self._is_pending_finish(instance)
                        and instance.remaining_triggers != 0
                    ):
                        self._trigger(instance)
                        instance.trigger_timer += interval

            if (
                instance.remaining_lifetime is not None
                and instance.remaining_lifetime <= BUFF_TIMER_EPSILON
            ):
                if instance.definition.override_policy == "STACK" and not instance.definition.clear_stack_on_expiry:
                    period = instance.definition.lifetime
                    expired = (instance.stack_count if period <= BUFF_TIMER_EPSILON else
                        min(instance.stack_count, math.floor((BUFF_TIMER_EPSILON - instance.remaining_lifetime) / period) + 1))
                    instance.stack_count -= expired
                    if instance.stack_count:
                        instance.remaining_lifetime += expired * period
                        continue
                self.finish(instance)

    def _update_response(self, instance: BuffInstance) -> None:
        from .mechanics import Status
        definition = instance.definition
        if not (definition.is_silenceable or definition.is_stunnable):
            return
        statuses = getattr(self.owner, "statuses", None)
        blocked = bool(statuses and (
            definition.is_silenceable and statuses.has(Status.SILENCE) or
            definition.is_stunnable and (statuses.has(Status.STUNNED) or statuses.has(Status.DISABLED))))
        if blocked and instance.enabled:
            instance.response_disabled = True
            self._set_enabled(instance, False)
        elif not blocked and instance.response_disabled:
            instance.response_disabled = False
            self._set_enabled(instance, True)

    def _timer_deltas(self, instance: BuffInstance, delta: float) -> tuple[float, float]:
        """Recompute resistance for both clocks, without scaling stored time."""
        from .mechanics import Status
        definition = instance.definition
        flags = set(definition.status_flags)
        resistible = (definition.status_resistable == "YES" or
            definition.status_resistable == "AUTOMATIC" and bool(flags & {"STUN", "STUNNED", "FROZEN"}))
        statuses = getattr(self.owner, "statuses", None)
        general = 1.0
        if resistible:
            general = self.clock_attribute("ONE_MINUS_STATUS_RESISTANCE")
        antis = getattr(statuses, "abnormal_resistances", {})
        highest = max([0.0] + [float(antis.get(Status("STUNNED" if flag == "STUN" else flag), 0))
                             for flag in flags if flag in {item.value for item in Status} | {"STUN"}])
        factor = max(0.001, general * (1.0 - max(0.0, min(highest, 1.0))))
        if not math.isfinite(factor):
            raise ValueError("non-finite Buff resistance clock")
        speed = (self.clock_attribute("EP_BREAK_RECOVER_SPEED") if definition.element_break else 1.0)
        trigger_delta = delta / factor * instance.start_clock_scale * speed
        slow = self.clock_attribute("SLOW_DOWN") if resistible else 1.0
        return trigger_delta * slow, trigger_delta

    def set_frozen(self, instance: BuffInstance, frozen: bool, *, source: Any = None) -> bool:
        """Pause clocks, keeping attributes, flags and event eligibility alive."""
        if (type(frozen) is not bool or instance.container is not self or instance.finished
                or self._is_pending_finish(instance)
                or not any(item is instance for item in (*self.active, *self.pending_add))):
            return False
        if source is None:
            instance.frozen = frozen
        elif frozen:
            if not any(item is source for item in instance.freeze_sources):
                instance.freeze_sources.append(source)
        else:
            instance.freeze_sources[:] = [item for item in instance.freeze_sources if item is not source]
        return True

    def clock_attribute(self, attribute: str, *, modifiers=None) -> float:
        if attribute == "ONE_MINUS_STATUS_RESISTANCE":
            base = getattr(self.owner, "one_minus_status_resistance", None)
            if base is None:
                base = 1.0 - float(getattr(getattr(self.owner, "statuses", None), "resistance", 0.0))
            lower, upper = .001, 1000.0
        elif attribute == "SLOW_DOWN":
            base = getattr(self.owner, "slow_down", 1.0)
            lower, upper = 0.0, 1.0
        elif attribute == "EP_BREAK_RECOVER_SPEED":
            base = getattr(self.owner, "ep_break_recover_speed", 1.0)
            lower, upper = 0.0, float("inf")
        else:
            raise ValueError("unsupported Buff clock attribute")
        value = (self.attribute_value(attribute, base) if modifiers is None else
                 calculate_attribute(base, modifiers))
        if not math.isfinite(value):
            raise ValueError("non-finite Buff clock attribute")
        return max(lower, min(upper, value))

    def modifiers(self, attribute: str, *, registered_only: bool = False) -> tuple[BuffModifier, ...]:
        wanted = str(attribute).upper()
        return tuple(
            (modifier if instance.stack_count == 1 else replace(modifier, value=modifier.value * instance.stack_count))
            for instance in self.active
            if instance.enabled
            and not instance.finished
            and not self._is_pending_finish(instance)
            and (not registered_only or instance.definition.registered_behavior)
            for modifier in instance.definition.modifiers
            if modifier.attribute == wanted
        )

    def attribute_value(
        self,
        attribute: str,
        base: float,
        *,
        source_limit: int | None = None,
        final_addition: float = 0.0,
    ) -> float:
        """Aggregate the four modifier pools without changing base."""
        modifiers = []
        used_sources: set[int] = set()
        for instance in self.active:
            if (
                not instance.enabled
                or instance.finished
                or self._is_pending_finish(instance)
            ):
                continue
            if source_limit is not None and instance.source is not None:
                source_id = id(instance.source)
                if source_id not in used_sources:
                    if len(used_sources) >= source_limit:
                        continue
                    used_sources.add(source_id)
            for modifier in instance.definition.modifiers:
                if modifier.attribute != str(attribute).upper():
                    continue
                modifiers.append(modifier if instance.stack_count == 1 else
                    replace(modifier, value=modifier.value * instance.stack_count))
        value = calculate_attribute(base, modifiers, final_addition=final_addition)
        return max(value, 0.0) if str(attribute).upper() in {"ATK", "DEF", "MOVE_SPEED"} else value


@dataclass
class BuffCatalog:
    """Normalized definitions and enemy-to-ability registration data."""

    definitions: dict[str, BuffDefinition] = field(default_factory=dict)
    abilities: dict[str, dict[str, Any]] = field(default_factory=dict)
    auras: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "BuffCatalog":
        raw = payload or {}
        definitions: dict[str, BuffDefinition] = {}
        for key, value in (raw.get("definitions") or {}).items():
            if not isinstance(value, dict):
                continue
            item = dict(value)
            item.setdefault("key", key)
            definition = BuffDefinition.from_dict(item)
            definitions[definition.key] = definition
        return cls(
            definitions=definitions,
            abilities={
                str(key): value
                for key, value in (raw.get("abilities") or {}).items()
                if isinstance(value, dict)
            },
            auras={
                str(key): value
                for key, value in (raw.get("auras") or {}).items()
                if isinstance(value, dict)
            },
        )

    def _ability(self, enemy_key: str) -> dict[str, Any]:
        key = str(enemy_key)
        if key in self.abilities:
            return self.abilities[key]
        if key.endswith("_2") and key[:-2] in self.abilities:
            return self.abilities[key[:-2]]
        return {}

    def definition(self, key: str) -> BuffDefinition:
        return self.definitions[str(key)]

    def listener_definitions(self, enemy_key: str) -> tuple[BuffDefinition, ...]:
        result: list[BuffDefinition] = []
        for item in self._ability(enemy_key).get("listeners", []) or []:
            if isinstance(item, str):
                key = item
            elif isinstance(item, dict):
                key = item.get("definition") or item.get("key")
            else:
                continue
            if key and str(key) in self.definitions:
                result.append(self.definitions[str(key)])
        return tuple(result)

    def aura_definitions(self, enemy_key: str) -> tuple[BuffDefinition, ...]:
        entry = self.auras.get(str(enemy_key))
        if entry is None and str(enemy_key).endswith("_2"):
            entry = self.auras.get(str(enemy_key)[:-2])
        if not entry:
            return ()
        result: list[BuffDefinition] = []
        for item in entry.get("buffs", []) or []:
            key = item if isinstance(item, str) else (item or {}).get("definition")
            if key and str(key) in self.definitions:
                result.append(self.definitions[str(key)])
        return tuple(result)

    @property
    def aura_keys(self) -> frozenset[str]:
        keys: set[str] = set()
        for entry in self.auras.values():
            for item in entry.get("buffs", []) or []:
                key = item if isinstance(item, str) else (item or {}).get("definition")
                if key is not None and str(key) in self.definitions:
                    keys.add(str(key))
        return frozenset(keys)

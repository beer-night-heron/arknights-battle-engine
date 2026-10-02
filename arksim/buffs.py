"""Small, deterministic runtime for enemy Buffs.

The native client has a considerably larger Buff system.  This module keeps
the first simulator slice deliberately narrow: a Buff is an immutable
definition plus a mutable instance, instances live in an owner-local
container, and container changes are committed outside the active-list
traversal.  Event actions are intentionally represented as data and are
executed by the existing behaviour interpreter supplied by Battle.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


BUFF_TIMER_EPSILON = 1e-9
LIFETIME_IMMEDIATELY = "IMMEDIATELY"
LIFETIME_LIMITED = "LIMITED"
LIFETIME_INFINITY = "INFINITY"


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", str(self.key))
        object.__setattr__(
            self,
            "lifetime_type",
            str(self.lifetime_type).upper(),
        )
        object.__setattr__(self, "lifetime", float(self.lifetime))
        object.__setattr__(self, "trigger_interval", float(self.trigger_interval))

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
        normalized.pop("attributes", None)
        return cls.from_dict(normalized)


@dataclass(eq=False)
class BuffInstance:
    """Mutable owner-local state for one ``BuffDefinition``."""

    definition: BuffDefinition
    owner: Any
    source: Any = None
    parent: "BuffInstance | None" = None
    blackboard: dict[str, Any] = field(default_factory=dict)
    attached_step: int | None = None
    remaining_lifetime: float | None = field(init=False)
    trigger_timer: float | None = field(init=False)
    trigger_count: int = 0
    enabled: bool = True
    finished: bool = False
    created_order: int = 0

    def __post_init__(self) -> None:
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
                self.definition.trigger_interval
                if self.definition.wait_first_trigger_interval
                else 0.0
            )
        else:
            self.trigger_timer = None


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
    active: list[BuffInstance] = field(default_factory=list)
    pending_add: list[BuffInstance] = field(default_factory=list)
    pending_finish: list[BuffInstance] = field(default_factory=list)
    created_counter: int = 0
    current_step: int | None = None

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
    ) -> BuffInstance | None:
        for instance in (*self.active, *self.pending_add):
            if instance.finished or not instance.enabled:
                continue
            if self._is_pending_finish(instance):
                continue
            if instance.definition.key != definition.key:
                continue
            if definition.unique_by_key or self._source_equal(instance.source, source):
                return instance
        return None

    def add(
        self,
        definition: BuffDefinition,
        *,
        source: Any = None,
        parent: BuffInstance | None = None,
        blackboard: dict[str, Any] | None = None,
        attached_step: int | None = None,
    ) -> BuffInstance:
        """Queue a Buff for the next commit, preserving stable creation order."""
        existing = self._find_existing(definition, source)
        if existing is not None:
            return existing
        self.created_counter += 1
        instance = BuffInstance(
            definition=definition,
            owner=self.owner,
            source=source,
            parent=parent,
            blackboard=dict(blackboard or {}),
            attached_step=(
                self.current_step if attached_step is None else attached_step
            ),
            created_order=self.created_counter,
        )
        self.pending_add.append(instance)
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
        if instance.finished:
            return False
        if not self._find_identity(self.pending_finish, instance):
            self.pending_finish.append(instance)
        return True

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
                instance.finished = True
                instance.enabled = False
                self._dispatch(instance, "ON_BUFF_FINISH")
                changed = True
            elif self._remove_identity(self.pending_add, instance):
                instance.finished = True
                instance.enabled = False
                changed = True
        return changed

    def _commit_additions(self) -> bool:
        if not self.pending_add:
            return False
        batch = self.pending_add
        self.pending_add = []
        changed = False
        for instance in batch:
            if instance.finished or not instance.enabled:
                continue
            existing = self._find_existing(instance.definition, instance.source)
            if existing is not None:
                instance.finished = True
                instance.enabled = False
                continue
            self.active.append(instance)
            self._dispatch(instance, "ON_BUFF_START")
            if instance.definition.lifetime_type == LIFETIME_IMMEDIATELY:
                self.finish(instance)
            changed = True
        return changed

    def commit(self) -> None:
        """Apply queued changes and any changes generated by lifecycle events."""
        # A lifecycle callback may enqueue another operation.  Flush those
        # operations in stable waves, never by mutating the list being walked.
        while self.pending_finish or self.pending_add:
            finished = self._commit_finishes()
            added = self._commit_additions()
            if not finished and not added:
                break

    def _trigger(self, instance: BuffInstance) -> None:
        instance.trigger_count += 1
        self._dispatch(instance, "ON_BUFF_TRIGGER")

    def tick(self, dt: float, frame_index: int | None = None) -> None:
        """Advance timers using the real delta, without committing mutations."""
        delta = max(float(dt), 0.0)
        self.current_step = frame_index
        for instance in tuple(self.active):
            if (
                instance.finished
                or not instance.enabled
                or self._is_pending_finish(instance)
            ):
                continue
            if frame_index is not None and instance.attached_step == frame_index:
                continue

            if instance.remaining_lifetime is not None:
                instance.remaining_lifetime -= delta

            interval = instance.definition.trigger_interval
            if instance.trigger_timer is not None and interval > BUFF_TIMER_EPSILON:
                instance.trigger_timer -= delta
                while (
                    instance.trigger_timer <= BUFF_TIMER_EPSILON
                    and not instance.finished
                    and not self._is_pending_finish(instance)
                ):
                    self._trigger(instance)
                    instance.trigger_timer += interval

            if (
                instance.remaining_lifetime is not None
                and instance.remaining_lifetime <= BUFF_TIMER_EPSILON
            ):
                self.finish(instance)

    def modifiers(self, attribute: str) -> tuple[BuffModifier, ...]:
        wanted = str(attribute).upper()
        return tuple(
            modifier
            for instance in self.active
            if instance.enabled
            and not instance.finished
            and not self._is_pending_finish(instance)
            for modifier in instance.definition.modifiers
            if modifier.attribute == wanted
        )

    def attribute_value(
        self,
        attribute: str,
        base: float,
        *,
        source_limit: int | None = None,
    ) -> float:
        """Aggregate addition and multiplier modifiers without changing base."""
        additions = 0.0
        multiplier = 0.0
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
                if modifier.formula == "ADDITION":
                    additions += modifier.value
                elif modifier.formula == "MULTIPLIER":
                    multiplier += modifier.value
        return (float(base) + additions) * (1.0 + multiplier)


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

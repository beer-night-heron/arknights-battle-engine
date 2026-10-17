"""Read-only attribute projections for reviewed sequential Buff actions.

No lifecycle callback, actor mutation, clock advance or random decision runs
here. More complex actions poison subsequent reads instead of predicting them.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from .behavior import node_name, registered_buff_spec_issue
from .buffs import BuffDefinition


@dataclass(frozen=True)
class _Entry:
    token: int
    definition: BuffDefinition
    source: int | None
    parent: int | None
    derived: bool
    committed: bool
    enabled: bool
    ending: bool
    stack_count: int = 1
    order: int = 0


class BuffAttributePreview:
    """Snapshot identities and modifiers, never copy or edit live entities."""

    def __init__(self, battle, context, *, root_definition=None, committing_owner=None):
        self.battle, self.context = battle, context
        self.root_definition = root_definition
        self.committing_owner = id(committing_owner) if committing_owner is not None else None
        self.pools = {}
        self.actors = {}
        self.next_token = -2
        self.root_token = id(context.buff_instance) if context.buff_instance is not None else -1
        self.parent_finished = False
        self.changed = False
        self.dirty = False

    def clone(self):
        result = copy.copy(self)
        result.pools = {key: list(items) for key, items in self.pools.items()}
        result.actors = dict(self.actors)
        return result

    def same_state(self, other):
        for key, actor in {**self.actors, **other.actors}.items():
            self._pool(actor); other._pool(actor)
        return (self.pools == other.pools and self.next_token == other.next_token
            and self.parent_finished == other.parent_finished and self.dirty == other.dirty
            and self.changed == other.changed)

    def _pool(self, actor):
        key = id(actor)
        if key not in self.pools:
            self.actors[key] = actor
            container = getattr(actor, 'buffs', None)
            if container is None:
                raise ValueError('attribute_container_binding')
            self.pools[key] = [_Entry(id(item), item.definition,
                id(item.source) if item.source is not None else None,
                id(item.parent) if item.parent is not None else None, item.finish_with_parent,
                any(old is item for old in container.active), item.enabled,
                item.finished or container._is_pending_finish(item), item.stack_count, item.created_order)
                for item in (*container.active, *container.pending_add)]
            if (self.context.buff_instance is None and self.root_definition is not None
                    and actor is self.context.buff_owner):
                self.pools[key].append(_Entry(self.root_token, self.root_definition,
                    id(self.context.source) if self.context.source is not None else None,
                    None, False, True, self.context.event not in {'ON_BUFF_FINISH', 'ON_BUFF_DISABLE'},
                    self.context.event == 'ON_BUFF_FINISH', order=container.created_counter + 1))
        return self.pools[key]

    @staticmethod
    def _leaf(definition):
        return (not definition.events and not definition.status_flags
            and definition.trigger_interval == -1 and definition.trigger_limit is None
            and all(item.attribute in {'ATK', 'DEF', 'MAGIC_RESISTANCE', 'MOVE_SPEED',
                                      'ONE_MINUS_STATUS_RESISTANCE', 'SLOW_DOWN', 'EP_BREAK_RECOVER_SPEED'}
                    for item in definition.modifiers)
            and (definition.override_policy in {None, 'DEFAULT'} or definition.disable_override))

    @staticmethod
    def _safe_finish(definition):
        return (not definition.modifiers and not definition.status_flags
            and not any(definition.actions_for(event) for event in
                        ('ON_BUFF_FINISH', 'ON_BUFF_DISABLE', 'ON_BUFF_ENABLE'))
            and (definition.override_policy in {None, 'DEFAULT'} or definition.disable_override))

    def modifiers(self, actor, attribute, *, registered_only=False, source_limit=None):
        if self.dirty:
            raise ValueError('after_actor_mutation_binding')
        result, sources = [], set()
        for item in self._pool(actor):
            if not item.committed or not item.enabled or item.ending:
                continue
            if source_limit is not None and item.source is not None:
                if item.source not in sources and len(sources) >= source_limit:
                    continue
                sources.add(item.source)
            if registered_only and not item.definition.registered_behavior:
                continue
            result.extend(modifier if item.stack_count == 1 else
                replace(modifier, value=modifier.value * item.stack_count)
                for modifier in item.definition.modifiers if modifier.attribute == attribute)
        return tuple(result)

    def contains(self, actor, key):
        return any(item.committed and item.enabled and not item.ending
            and item.definition.key == key for item in self._pool(actor))

    def _flush(self, actor):
        items = self._pool(actor)
        if any((not item.committed and not item.ending and not self._leaf(item.definition))
               or (item.ending and not (self._leaf(item.definition) or self._safe_finish(item.definition)))
               for item in items):
            self.dirty = True
            return
        items = [replace(item, committed=True) for item in items if not item.ending]
        groups = {}
        for item in items:
            if item.definition.override_policy == 'DEFAULT' and not item.definition.disable_override:
                groups.setdefault(item.definition.override_key, []).append(item)
        winners = {}
        for key, group in groups.items():
            winner = max(group, key=lambda item: (item.definition.override_priority, -item.order))
            if any(item.enabled != (item is winner) and not self._leaf(item.definition) for item in group):
                self.dirty = True
                return
            winners[key] = winner.token
        self.pools[id(actor)] = [replace(item, enabled=item.token == winners[item.definition.override_key])
            if item.definition.override_policy == 'DEFAULT' and not item.definition.disable_override else item
            for item in items]

    def _create(self, actor, source, spec, board, derived, *, commit_requested=True):
        if registered_buff_spec_issue(spec):
            self.dirty = True
            return None
        definition = self.battle._behavior_buff_definition(spec, board)
        if definition is not None and definition.override_policy == 'DEFAULT' and not definition.disable_override and not self._leaf(definition):
            raise ValueError('override_action_lifecycle_binding')
        if definition is None or not self._leaf(definition):
            self.dirty = True
            return None
        if derived and self.parent_finished:
            raise ValueError('derived_parent_binding')
        if actor is None:
            self.dirty = True
            return None
        if actor.dead:
            return False
        items = self._pool(actor)
        if not definition.disable_override:
            if self.dirty:
                raise ValueError('override_after_actor_mutation_binding')
            for item in items:
                old = item.definition
                if item.ending or old.disable_override or old.override_key != definition.override_key:
                    continue
                if old.override_policy != definition.override_policy:
                    raise ValueError('override_policy_conflict')
                if definition.override_policy == 'DEFAULT' and not self._leaf(old):
                    raise ValueError('override_action_lifecycle_binding')
        source_token = id(source) if source is not None else None
        if definition.override_policy is None and not definition.disable_override:
            existing = next((item for item in items if item.enabled and not item.ending
                and item.definition.override_policy is None and item.definition.key == definition.key
                and item.derived == derived and (not derived or item.parent == self.root_token)
                and (definition.unique_by_key or item.source == source_token)), None)
        else:
            existing = None
        if existing is None:
            items.append(_Entry(self.next_token, definition, source_token, self.root_token,
                                derived, False, True, False,
                                order=max((item.order for item in items), default=0) + 1))
            self.next_token -= 1
        if commit_requested and not actor.buffs._committing and id(actor) != self.committing_owner:
            self._flush(actor)
        return True

    def _finish_tokens(self, tokens):
        # Resolve original foreign families lazily before following exact ids.
        instance = self.context.buff_instance
        todo = [instance] if instance is not None else []
        todo.extend(item for actor in tuple(self.actors.values())
                    for item in (*actor.buffs.active, *actor.buffs.pending_add) if id(item) in tokens)
        seen = set()
        while todo:
            current = todo.pop()
            if id(current) in seen:
                continue
            seen.add(id(current)); self._pool(current.owner)
            todo.extend(current.derived_children)
        if self.root_token in tokens:
            self.parent_finished = True
            definition = (instance.definition if instance is not None else self.root_definition)
            if definition is not None and not self._safe_finish(definition):
                self.dirty = True
        while True:
            children = {item.token for items in self.pools.values() for item in items
                        if item.derived and item.parent in tokens}
            if children <= tokens:
                break
            tokens |= children
        for key, items in self.pools.items():
            for index, item in enumerate(items):
                if item.token not in tokens or item.ending:
                    continue
                if not (self._leaf(item.definition) or self._safe_finish(item.definition)):
                    self.dirty = True
                items[index] = replace(item, ending=True)

    def apply(self, node, board):
        name, context = node_name(node), self.context
        if name == 'FinishDerivedBuff':
            if context.buff_instance is None and self.root_definition is None:
                return False
            self.changed = True
            self._pool(context.buff_owner)
            self._finish_tokens(set())
            children = {item.token for items in self.pools.values() for item in items
                        if item.derived and item.parent == self.root_token and not item.ending}
            self._finish_tokens(children)
            return True
        if name == 'FinishAllBuffsWithCertainAbnormalFlag':
            # Ending a flagged Buff can revoke attributes and run callbacks.
            self.changed = self.dirty = True
            return True
        if name in {'CreateBuff', 'CreateBuffInRange', 'FinishBuff', 'FinishBuffsById'}:
            self.changed = True
        if name == 'CreateBuff':
            if not (context.register_buffs or context.buff_instance is not None
                    and context.buff_instance.definition.registered_behavior
                    or self.root_definition is not None and self.root_definition.registered_behavior):
                self.dirty = True
                return None
            source = (context.entity(node.get('_specialBuffSource')) if node.get('_useSpecialBuffSource')
                      else context.source)
            return self._create(context.entity(node.get('_buffOwner')), source,
                node.get('_buff') or {}, board, bool(node.get('_isDerivedBuff')),
                commit_requested=context.register_buffs)
        if name == 'CreateBuffInRange':
            source = context.entity(node.get('_sourceType'))
            if source is None:
                self.dirty = True; return None
            if source.dead:
                return False
            targets = self.battle.behavior_range_targets(context, node)
            for actor in targets:
                for spec in node.get('_buffs') or []:
                    self._create(actor, source, spec, board, bool(node.get('_isDerivedBuff')))
            return bool(targets)
        if name == 'FinishBuff':
            if context.event == 'ON_BUFF_FINISH' or (context.buff_instance is not None and context.buff_instance.finished):
                return False
            self._finish_tokens({self.root_token})
            return True
        if name == 'FinishBuffsById':
            actor = context.entity(node.get('_targetType'))
            if actor is None:
                return False
            key = str(node.get('_buffKey', ''))
            if node.get('_loadFromBlackboard'):
                key = str(board.get(key, key))
            tokens = {item.token for item in self._pool(actor) if item.definition.key == key and not item.ending}
            self._finish_tokens(tokens)
            if key in getattr(actor, 'buff_blackboards', {}):
                self.dirty = True
            return bool(tokens) or self.dirty
        if name in {'SwitchMode', 'AdvancedApplyDamage', 'NoSourceDamage', 'AOEDamage', 'DamageViaAttr', 'DamageViaMaxHpRatio'}:
            self.dirty = True
        return None

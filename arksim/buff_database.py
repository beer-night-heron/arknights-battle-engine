"""Resolve native BuffData references before behavior/actor preflight."""
from __future__ import annotations

import copy
from typing import Any


BUFF_DATA_FIELDS = frozenset({"attributes", "buffKey", "loadFromDB", "isDurableBuff", "isDamageMissable",
    "isSilenceable", "isStunnable", "isFreezable", "isLevitatable", "isGroundBoundable",
    "statusResistable", "templateKey", "disableOverride", "overrideKey", "overrideType",
    "maxStackCnt", "refreshRemainingTimeWhenStackMax", "clearAllStackCntWhenTimeUp", "maxValidStackCnt",
    "independentCharacterSource", "overrideEffectKey", "overrideOnEventPriority", "onEventPriority",
    "audioSignal", "lifeTimeType", "takeSnapshotWhenExtend", "durationKey", "remainingTimeKey",
    "lifeTime", "triggerLifeType", "triggerCnt", "triggerInterval", "waitFirstTriggerInterval",
    "firstTriggerInterval", "priority", "priorityBBKeys", "stripBlackboardParamsWithBuffKey",
    "blackboard", "enableInitDirectionFromSource"})


class BuffDatabase:
    """Clone the DB definition and merge reference BB; other fields do not override."""

    def __init__(self, table: dict | None = None):
        if table is None:
            table = {}
        if not isinstance(table, dict):
            raise ValueError("Buff database must be a mapping")
        for key, entry in table.items():
            if (not isinstance(key, str) or not key or not isinstance(entry, dict)
                    or not isinstance(entry.get("buffKey"), str) or not entry["buffKey"]
                    or entry.get("loadFromDB", False)):
                raise ValueError("invalid or recursive Buff database definition")
        self.definitions = copy.deepcopy(table)

    @staticmethod
    def _board(value: Any) -> list[dict]:
        if value is None:
            return []
        if not isinstance(value, list) or any(not isinstance(item, dict)
                or not isinstance(item.get("key"), str) or not item["key"] for item in value):
            raise ValueError("database:blackboard_payload")
        return copy.deepcopy(value)

    def resolve_spec(self, spec: dict) -> dict:
        if not isinstance(spec, dict):
            raise ValueError("database:buff_payload")
        flag = spec.get("loadFromDB", False)
        if type(flag) not in (bool, int) or flag not in (0, 1):
            raise ValueError("database:load_flag")
        if not flag:
            return spec
        if set(spec) - BUFF_DATA_FIELDS:
            raise ValueError("database:reference_fields")
        key = spec.get("buffKey")
        if not isinstance(key, str) or key not in self.definitions:
            raise ValueError("database:missing_definition:" + str(key))
        result = copy.deepcopy(self.definitions[key])
        board = self._board(result.get("blackboard"))
        for item in self._board(spec.get("blackboard")):
            # Native MergeTo replaces the first matching entry, appending new
            # keys. ASCII case-insensitive names cover the exported battle BB.
            name = item["key"]
            if not name.isascii():
                raise ValueError("database:non_ascii_merge_key")
            match = next((i for i, old in enumerate(board) if old["key"].lower() == name.lower()), None)
            if match is None:
                board.append(item)
            else:
                board[match] = item
        result["blackboard"] = board
        return result

    def resolve_tree(self, value: Any) -> Any:
        """Expand reviewed creation nodes, leaving legacy hit-attachment adapters intact."""
        if isinstance(value, dict):
            name = str(value.get("$type", "")).split(",", 1)[0].rsplit("+", 1)[-1]
            def reference(child):
                if isinstance(child, dict) and child.get("loadFromDB"):
                    try:
                        return self.resolve_spec(child)
                    except ValueError:
                        pass
                return child
            children = {}
            for key, child in value.items():
                if name == "CreateBuff" and key == "_buff":
                    children[key] = reference(child)
                elif name == "CreateBuffInRange" and key == "_buffs" and isinstance(child, list):
                    resolved = [reference(item) for item in child]
                    children[key] = resolved if any(new is not old for new, old in zip(resolved, child)) else child
                else:
                    children[key] = self.resolve_tree(child)
            return children if any(children[key] is not child for key, child in value.items()) else value
        if isinstance(value, list):
            children = [self.resolve_tree(child) for child in value]
            return children if any(new is not old for new, old in zip(children, value)) else value
        return value

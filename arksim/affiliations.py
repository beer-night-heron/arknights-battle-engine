"""Combat affiliation queries, including exported hidden powers."""
from __future__ import annotations

from typing import Any


AFFILIATION_FIELDS = frozenset({"nationId", "groupId", "teamId"})


def has_affiliation(character: dict[str, Any], tag: str, field: str | None = None) -> bool:
    """Match one unit once across legacy fields, main power and sub powers."""
    if field is not None and field not in AFFILIATION_FIELDS:
        raise ValueError(f"not an affiliation field: {field}")
    fields = (field,) if field else AFFILIATION_FIELDS
    powers = (character, character.get("mainPower") or {}, *(character.get("subPower") or []))
    return any(power.get(key) == tag for power in powers for key in fields)

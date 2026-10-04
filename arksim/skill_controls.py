"""Recognize explicit player controls; duration alone grants no permission."""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class SkillControls:
    can_end: bool = False
    can_switch: bool = False


def skill_controls(level: dict) -> SkillControls:
    # Table descriptions establish the permission, not the transition timing
    # or every effect of the skill. Missing/localized declarations fail closed.
    text = re.sub(r"\s+", "", re.sub(r"<[^>]*>", "", str(level.get("description", ""))))
    return SkillControls(
        can_end=any(phrase in text for phrase in ("可主动关闭技能", "可随时停止技能")),
        can_switch="可以在下列状态和初始状态间切换" in text,
    )

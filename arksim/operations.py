"""Operation outcomes and compact transition feedback for battle plans."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


MESSAGES = {
    "executed": "操作成功",
    "unknown_operator": "干员不存在",
    "missing_tile": "部署未指定位置",
    "already_deployed": "干员已在场",
    "out_of_bounds": "部署位置超出地图",
    "tile_occupied": "部署位置已被占用",
    "not_buildable": "该位置不允许部署此干员",
    "redeploy_cooldown": "再部署冷却未结束",
    "deployment_limit": "部署席位已满",
    "insufficient_cost": "费用不足",
    "not_on_field": "干员不在场",
    "no_skill": "未配置技能",
    "skill_unavailable": "技能已成为永久效果或不可再次释放",
    "not_manual_skill": "该技能不能手动释放",
    "skill_active": "技能仍在持续",
    "skill_starting": "技能启动中",
    "skill_disabled": "当前状态禁止释放技能",
    "insufficient_sp": "技力不足",
    "skill_rejected": "技能执行流程未完成",
    "skill_not_endable": "该技能不支持主动结束",
    "skill_not_active": "技能当前未在持续",
    "skill_not_switchable": "该技能不支持模式切换",
    "unsupported_action": "不支持的操作类型",
    "scheduled_after_end": "战斗结束时尚未到操作时间",
    "blocked_by_previous": "被前一条等待中的操作阻塞",
    "battle_ended": "战斗已结束",
    "plan_stopped": "前一条操作失败，计划已终止",
}


@dataclass(frozen=True)
class OperationAttempt:
    status: str
    reason: str = "executed"
    details: dict[str, Any] = field(default_factory=dict)


class OperationLog:
    def __init__(self, plans: list, source_indices: list[int]) -> None:
        self.records = [
            {"index": source, "action": plan.action, "char_id": plan.char_id,
             "on_failure": plan.on_failure, "tile": list(plan.tile) if plan.tile is not None else None,
             "mode": plan.mode,
             "scheduled_time": plan.time, "status": "pending", "reason": None,
             "message": "尚未执行", "first_attempt_time": None,
             "last_attempt_time": None, "executed_time": None,
             "first_attempt_frame": None, "last_attempt_frame": None, "executed_frame": None, "details": {}}
            for plan, source in zip(plans, source_indices)
        ]
        self.events: list[dict[str, Any]] = []

    def record(self, index: int, attempt: OperationAttempt, time: float, frame: int) -> None:
        record = self.records[index]
        status = {"ok": "succeeded", "wait": "waiting", "skip": "failed"}[attempt.status]
        changed = (record["status"], record["reason"]) != (status, attempt.reason)
        record.update(status=status, reason=attempt.reason, message=MESSAGES[attempt.reason],
                      last_attempt_time=time, last_attempt_frame=frame, details=attempt.details)
        if record["first_attempt_time"] is None:
            record["first_attempt_time"] = time
            record["first_attempt_frame"] = frame
        if attempt.status == "ok":
            record["executed_time"] = time
            record["executed_frame"] = frame
        if changed:
            self.events.append({"t": round(time, 3), "fixedFrame": frame, "kind": "operation",
                                "index": record["index"], "action": record["action"],
                                "char": record["char_id"], "status": status, "reason": attempt.reason,
                                "message": record["message"], "details": deepcopy(attempt.details)})

    def summary(self, end_reason: str, time: float) -> list[dict[str, Any]]:
        records = deepcopy(self.records)
        blocker = next((r["index"] for r in records if r["status"] == "waiting"), None)
        if end_reason == "operation_failed":
            blocker = next((r["index"] for r in records if r["status"] == "failed"), None)
        if not end_reason:
            return records
        for record in records:
            if record["status"] == "waiting":
                record["status"] = "unfinished"
                record["details"]["end_reason"] = end_reason
            elif record["status"] == "pending":
                reason = ("plan_stopped" if end_reason == "operation_failed"
                          else "scheduled_after_end" if record["scheduled_time"] > time
                          else "blocked_by_previous" if blocker is not None else "battle_ended")
                record.update(status="not_executed", reason=reason, message=MESSAGES[reason])
                record["details"] = {"end_reason": end_reason, "blocking_operation": blocker}
        return records

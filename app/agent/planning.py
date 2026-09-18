"""R-P1-9（s05）：任务规划。

- `todo_write` 工具：Agent 开工前写计划、执行中更新状态；
- 计划写入 `agent_tasks.progress_json`（非业务表），可跨重启读出；
- `Planner`：连续 N 轮未更新计划时，把 reminder 追加到工具结果（照搬 s05 的计数器，不靠模型自觉）。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from app.agent.session import AgentSession, current_session
from app.agent.tools.contract import ToolResult, failure, success
from app.agent.tools.registry import ToolRegistry
from app.db import Database

PLAN_STATUSES: Tuple[str, ...] = ("pending", "in_progress", "done")
DEFAULT_REMINDER_EVERY = 3
_MAX_PLAN_ITEMS = 20
_MAX_TASK_CHARS = 200


def _normalize_items(items: Any) -> Tuple[Optional[List[Dict[str, str]]], Optional[str]]:
    if not isinstance(items, list):
        return None, "items 必须是数组。"
    if not items:
        return None, "items 不能为空；请至少写一项计划。"
    if len(items) > _MAX_PLAN_ITEMS:
        return None, "计划项最多 {} 条。".format(_MAX_PLAN_ITEMS)
    normalized: List[Dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            return None, "每个计划项必须是对象 {task, status}。"
        task = str(item.get("task") or "").strip()
        status = str(item.get("status") or "pending").strip()
        if not task:
            return None, "计划项的 task 不能为空。"
        if status not in PLAN_STATUSES:
            return None, "status 只能是：{}。".format("/".join(PLAN_STATUSES))
        normalized.append({"task": task[:_MAX_TASK_CHARS], "status": status})
    return normalized, None


def _todo_write(db: Database, team_id: int, items: Any = None) -> ToolResult:
    session = current_session()
    if session is None:
        return failure("tool_error", "当前没有活动的 Agent 会话，无法写计划。")
    normalized, error = _normalize_items(items)
    if error:
        return failure("invalid_args", error)
    session.todo = normalized
    session.last_plan_update_step = session.steps + 1
    session.last_reminder_step = 0
    if session.meeting_id:
        db.upsert_agent_task(
            session.meeting_id,
            team_id,
            "AI 分析中",
            meeting_id=session.meeting_id,
            progress_json=json.dumps({"todo": normalized}, ensure_ascii=False),
        )
    done = sum(1 for entry in normalized if entry["status"] == "done")
    return success(
        "计划已更新：{} 项（完成 {}）。".format(len(normalized), done),
        {"items": normalized},
    )


class Planner:
    """s05 的 reminder 计数器。"""

    def __init__(self, reminder_every: int = DEFAULT_REMINDER_EVERY) -> None:
        self.reminder_every = max(1, int(reminder_every))

    def reminder(self, session: AgentSession, step: int) -> Optional[str]:
        if not session.todo:
            return None
        since = step - session.last_plan_update_step
        if since < self.reminder_every:
            return None
        if session.last_reminder_step == step:
            return None
        session.last_reminder_step = step
        return (
            "提醒：已连续 {} 轮没有更新计划。请用 todo_write 更新各项进度后再继续。".format(since)
        )


def register_planning_tools(registry: ToolRegistry, db: Database) -> None:
    def handler(team_id: int, **kwargs: Any) -> ToolResult:
        return _todo_write(db, team_id, **kwargs)

    registry.register_tool(
        name="todo_write",
        description="写下或更新本场会议的分析计划（分几批、先抽哪类信息），并标注每项进度。",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "task": {"type": "string"},
                            "status": {"type": "string", "enum": list(PLAN_STATUSES)},
                        },
                        "required": ["task"],
                    },
                }
            },
            "required": ["items"],
        },
        handler=handler,
    )

"""共享校验与体积护栏（工具层内部使用）。"""

from __future__ import annotations

import json
from typing import Any, Optional

from app.agent.tools.contract import MAX_PAYLOAD_BYTES, ToolResult, failure
from app.db import Database

SNIPPET_CHARS = 200


def check_meeting(db: Database, meeting_id: str, team_id: int) -> Optional[ToolResult]:
    """归属校验：不存在 → not_found；属于别的团队 → team_forbidden（不返回任何数据）。"""
    owner = db.owner_team_id(meeting_id)
    if owner is None:
        return failure("not_found", "没有找到这场会议：{}".format(meeting_id))
    if owner != team_id:
        return failure("team_forbidden", "该会议不属于当前团队，拒绝访问。")
    return None


def check_project(db: Database, project_id: str, team_id: int) -> Optional[ToolResult]:
    owner = db.project_owner_team_id(project_id)
    if owner is None:
        return failure("not_found", "没有找到这个项目：{}".format(project_id))
    if owner != team_id:
        return failure("team_forbidden", "该项目不属于当前团队，拒绝访问。")
    return None


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    value = text or ""
    return value if len(value) <= limit else value[:limit] + "…"


def payload_bytes(payload: Any) -> int:
    return len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))


def guard_size(result: ToolResult, *, next_hint: str) -> ToolResult:
    """兜底：单次返回超过 8KB（PRD §12）时降级为 too_large，避免灌爆模型上下文。"""
    if result.ok and payload_bytes(result.payload) > MAX_PAYLOAD_BYTES:
        return failure("too_large", "结果超过 8KB 上限，请缩小范围或分页。", next_hint=next_hint)
    return result

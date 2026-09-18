"""内置钩子：审计（R-P1-5）与结构化日志（PRD §12）。

两者都是普通钩子函数：主循环不认识它们，注册即可生效，移除也不影响循环。
审计钩子**只写摘要**：不写转写文本、引文原文、口令、密钥。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.agent.hooks import HookDecision, HookManager
from app.db import Database

logger = logging.getLogger(__name__)


def audit_hook(db: Database):
    """PostToolUse 钩子：把每次工具调用（含拒绝）写入 agent_audit。"""

    def handler(event: str, payload: Dict[str, Any]) -> Optional[HookDecision]:
        if event != "PostToolUse":
            return None
        db.record_agent_audit(
            team_id=payload["team_id"],
            meeting_id=payload.get("meeting_id"),
            session_id=payload["session_id"],
            step=payload.get("step"),
            tool_name=payload.get("tool_name"),
            args_digest=payload.get("args_digest"),
            decision=payload.get("decision", "allow"),
            result_code=payload.get("result_code"),
            duration_ms=payload.get("duration_ms"),
        )
        return None

    return handler


def tool_logging_hook(event: str, payload: Dict[str, Any]) -> Optional[HookDecision]:
    """PostToolUse 钩子：结构化记录一次工具调用结果（不含参数原文）。"""
    if event != "PostToolUse":
        return None
    logger.info(
        "tool_call session_id=%s step=%s tool=%s decision=%s result_code=%s duration_ms=%s",
        payload.get("session_id"), payload.get("step"), payload.get("tool_name"),
        payload.get("decision"), payload.get("result_code"), payload.get("duration_ms"),
    )
    return None


def session_stop_hook(event: str, payload: Dict[str, Any]) -> Optional[HookDecision]:
    if event != "Stop":
        return None
    logger.info(
        "agent_session_stopped session_id=%s status=%s steps=%s",
        payload.get("session_id"), payload.get("status"), payload.get("steps"),
    )
    return None


def build_default_hooks(db: Database, *, audit_enabled: bool = True) -> HookManager:
    hooks = HookManager()
    if audit_enabled:
        hooks.register("PostToolUse", audit_hook(db))
    hooks.register("PostToolUse", tool_logging_hook)
    hooks.register("Stop", session_stop_hook)
    return hooks

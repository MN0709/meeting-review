"""R-P1-1：一次 Agent 会话的状态。

session_id 用于把一次会话的全部工具调用串起来（审计回放）。
messages 采用 OpenAI Chat 格式，便于直接传给模型。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.agent.tools.contract import ToolResult


@dataclass
class AgentLimits:
    max_steps: int = 20
    step_timeout_seconds: float = 120.0
    session_timeout_seconds: float = 1800.0


@dataclass
class ToolCallRecord:
    step: int
    tool_name: str
    decision: str
    result_code: str
    duration_ms: int


class AgentSession:
    def __init__(
        self,
        *,
        goal: str,
        system_prompt: str,
        usage_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.session_id = session_id or uuid4().hex
        self.goal = goal
        self.usage_context = dict(usage_context or {})
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": goal},
        ]
        self.steps = 0
        self.records: List[ToolCallRecord] = []
        self.status = "running"
        self.report: Optional[Dict[str, Any]] = None
        self.validation: Optional[Dict[str, Any]] = None
        # 供阶段 6（s05 任务规划 / s17 完成判定）使用
        self.todo: List[Dict[str, Any]] = []
        self.notes: List[str] = []

    @property
    def team_id(self) -> Optional[int]:
        return self.usage_context.get("team_id")

    @property
    def meeting_id(self) -> Optional[str]:
        return self.usage_context.get("meeting_id")

    def append_assistant(self, content: Optional[str], tool_calls: List[Any]) -> None:
        message: Dict[str, Any] = {"role": "assistant", "content": content or ""}
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in tool_calls
        ]
        self.messages.append(message)

    def append_tool_result(self, tool_call_id: str, result: ToolResult) -> None:
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(result.model_dump(), ensure_ascii=False, default=str),
            }
        )

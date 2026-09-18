"""R-P1-1 / R-P1-9 / R-P1-11：一次 Agent 会话的状态。

- session_id 用于把一次会话的全部工具调用串起来（审计回放）；
- messages 采用 OpenAI Chat 格式，便于直接传给模型；
- todo / last_plan_update_step 供 s05 任务规划使用；
- judgments 供 s17 最小完成判定使用。
"""

from __future__ import annotations

import contextlib
import json
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional
from uuid import uuid4

from app.agent.tools.contract import ToolResult


@dataclass
class AgentLimits:
    max_steps: int = 20
    step_timeout_seconds: float = 120.0
    session_timeout_seconds: float = 1800.0
    # s17：完成判定未通过时最多允许「带缺失项继续」的次数
    max_judge_retries: int = 2


@dataclass
class ToolCallRecord:
    step: int
    tool_name: str
    decision: str
    result_code: str
    duration_ms: int


@dataclass
class GoalJudgment:
    done: bool
    reason: str
    missing: List[str]
    evaluator: str = "rule"
    error: bool = False


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
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": goal},
        ]
        self.steps = 0
        self.records: List[ToolCallRecord] = []
        self.status = "running"
        self.report: Optional[Dict[str, Any]] = None
        self.validation: Optional[Dict[str, Any]] = None
        # s05 任务规划
        self.todo: List[Dict[str, Any]] = []
        self.last_plan_update_step = 0
        self.last_reminder_step = 0
        # s17 完成判定
        self.judgments: List[GoalJudgment] = []
        self.judge_retries = 0
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

    def append_tool_result(self, tool_call_id: str, result: ToolResult, extra: str = "") -> None:
        content = json.dumps(result.model_dump(), ensure_ascii=False, default=str)
        if extra:
            content = content + "\n\n" + extra
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

    def append_observation(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def refresh_system_prompt(self, builder: Any) -> None:
        """每轮调用前重建系统提示词（s15 动态装配）。"""
        if builder is None or not self.messages:
            return
        self.system_prompt = builder(self)
        self.messages[0] = {"role": "system", "content": self.system_prompt}


# ---------------------------------------------------------------------------
# 让「工具」能访问当前会话（s05 的 todo_write 需要）
# ---------------------------------------------------------------------------
# 用 ContextVar：asyncio.to_thread 会复制上下文，因此同步工具处理器也能读到同一会话。
_current_session: ContextVar[Optional[AgentSession]] = ContextVar(
    "agent_current_session", default=None
)


def current_session() -> Optional[AgentSession]:
    return _current_session.get()


@contextlib.contextmanager
def bind_session(session: AgentSession) -> Iterator[None]:
    token = _current_session.set(session)
    try:
        yield
    finally:
        _current_session.reset(token)

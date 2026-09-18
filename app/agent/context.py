"""R-P1-7（s08）：上下文治理。

Agent 会话的 messages 会随工具结果增长；4 小时会议尤其明显。本模块：
1. **先整理工具结果**：把较早的工具返回压缩为「code + summary + next_hint」（丢掉大 payload）；
2. **仍超预算再退一步**：把更早的工具结果替换为压缩标记（结构摘要），保留消息顺序与
   `tool_call_id` 配对，避免破坏 OpenAI 的 tool 调用结构；
3. 每次压缩都返回一条可记录、可日志的说明（PRD R-P1-7 ③）。

说明：这里用**确定性结构摘要**而不是再调一次模型生成摘要——压缩本身也花钱，
且结构摘要可复现、可测试。PRD 的「生成历史摘要」在需要时可后续替换实现。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app.agent.session import AgentSession

DEFAULT_KEEP_RECENT = 6
DEFAULT_CONTENT_LIMIT = 400


def message_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """粗略 token 估算：约 2 个字符 1 个 token（中文偏保守），用于触发压缩。"""
    total = sum(len(message_text(message)) for message in messages)
    return max(1, total // 2)


class ContextManager:
    def __init__(
        self,
        budget_tokens: int,
        *,
        keep_recent: int = DEFAULT_KEEP_RECENT,
        content_limit: int = DEFAULT_CONTENT_LIMIT,
    ) -> None:
        self.budget_tokens = max(1, int(budget_tokens))
        self.keep_recent = max(0, int(keep_recent))
        self.content_limit = max(50, int(content_limit))

    def maybe_compact(self, session: AgentSession) -> Optional[str]:
        """超预算时压缩历史；返回本次压缩说明（未压缩返回 None）。"""
        before = estimate_tokens(session.messages)
        if before <= self.budget_tokens:
            return None

        tool_indexes = [i for i, m in enumerate(session.messages) if m.get("role") == "tool"]
        older = tool_indexes[: max(0, len(tool_indexes) - self.keep_recent)]
        for index in older:
            session.messages[index]["content"] = self._shrink(session.messages[index]["content"])
        after_slim = estimate_tokens(session.messages)
        if after_slim <= self.budget_tokens:
            return "compact=tool_results before={} after={}".format(before, after_slim)

        for index in older:
            session.messages[index]["content"] = "[已压缩的工具结果]"
        after_hard = estimate_tokens(session.messages)
        return "compact=structural_summary before={} after={} report_decisions={}".format(
            before, after_hard, len((session.report or {}).get("decisions", []) or [])
        )

    def _shrink(self, content: str) -> str:
        head, separator, tail = content.partition("\n\n")
        try:
            payload = json.loads(head)
        except (json.JSONDecodeError, TypeError):
            payload = None
        if not isinstance(payload, dict):
            return content[: self.content_limit]
        slim = {
            "ok": payload.get("ok"),
            "code": payload.get("code"),
            "summary": payload.get("summary"),
            "truncated": payload.get("truncated"),
            "next_hint": payload.get("next_hint"),
        }
        text = json.dumps(slim, ensure_ascii=False)
        return text + (separator + tail if separator else "")

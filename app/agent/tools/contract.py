"""PRD R-P1-6：统一的工具返回契约（ToolResult）。

设计要点（对齐 PRD 原则 P-7「观察面必须回灌」）：
- 任何工具都返回 ToolResult，**失败返回 ok=false 而不是抛异常**，
  这样模型能读到「哪错了、可以怎么改」，而不是盲试重试。
- summary 是给模型直接读的中文短句（≤200 字），payload 是结构化数据。
- 返回体积超限时标记 truncated=true 并给出 next_hint（如何取下一页 / 缩小范围）。

本模块只定义契约与工厂函数，不含任何具体工具。
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# 与 PRD §9.5 完全一致；新增错误码必须同步 PRD。
ToolResultCode = Literal[
    "ok",
    "not_found",
    "team_forbidden",
    "invalid_args",
    "too_large",
    "denied_host_owned",
    "denied_policy",
    "tool_error",
]

VALID_CODES: tuple[str, ...] = (
    "ok",
    "not_found",
    "team_forbidden",
    "invalid_args",
    "too_large",
    "denied_host_owned",
    "denied_policy",
    "tool_error",
)

# summary 是模型直接读的那句话，必须短。
MAX_SUMMARY_CHARS = 200
# 单次工具返回体积上限（PRD §12：≤8 KB）。
MAX_PAYLOAD_BYTES = 8 * 1024

# 失败时若调用方没写 summary，用这份兜底可读原因（PRD P-7：必须告诉模型哪错了）。
DEFAULT_HINTS: dict[str, str] = {
    "not_found": "没有找到对应数据，请确认 id 是否正确，或先用 list_* 工具列出可用对象。",
    "team_forbidden": "该数据不属于当前团队，越权查询会被拒绝，请不要重试。",
    "invalid_args": "参数不合法，请检查参数名与类型后重试。",
    "too_large": "返回内容过大，请缩小查询范围或分页获取。",
    "denied_host_owned": "该操作只能由人在界面上执行，你无权调用。",
    "denied_policy": "该工具当前未开启或不在允许范围内，请不要重试。",
    "tool_error": "工具执行失败，可换一种方式，或在报告中把该项标记为待确认。",
}


def _clip_summary(summary: str) -> str:
    text = (summary or "").strip()
    if len(text) > MAX_SUMMARY_CHARS:
        return text[: MAX_SUMMARY_CHARS - 1] + "…"
    return text


class ToolResult(BaseModel):
    """所有 Agent 工具的统一返回结构（PRD §9.5）。"""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    code: ToolResultCode
    summary: str = Field(max_length=MAX_SUMMARY_CHARS)
    payload: Any = None
    truncated: bool = False
    artifacts: List[str] = Field(default_factory=list)
    next_hint: Optional[str] = None

    @model_validator(mode="after")
    def _require_hint_when_truncated(self) -> "ToolResult":
        if self.truncated and not self.next_hint:
            raise ValueError("truncated=true 时必须给出 next_hint，告诉模型如何取下一页")
        return self


def success(
    summary: str,
    payload: Any = None,
    *,
    artifacts: Optional[List[str]] = None,
    truncated: bool = False,
    next_hint: Optional[str] = None,
) -> ToolResult:
    """成功返回。summary 必须是 payload 的中文摘要，不是重写。"""
    return ToolResult(
        ok=True,
        code="ok",
        summary=_clip_summary(summary),
        payload=payload,
        truncated=truncated,
        artifacts=list(artifacts or []),
        next_hint=next_hint,
    )


def failure(
    code: str,
    summary: str = "",
    *,
    payload: Any = None,
    next_hint: Optional[str] = None,
) -> ToolResult:
    """失败返回。code 必须是 PRD §9.5 的错误码之一，且不能是 'ok'。"""
    if code == "ok":
        raise ValueError("failure() 的 code 不能是 'ok'，成功请用 success()")
    if code not in VALID_CODES:
        raise ValueError("未知错误码：{}".format(code))
    text = _clip_summary(summary) or DEFAULT_HINTS.get(code, "")
    return ToolResult(
        ok=False,
        code=code,
        summary=text,
        payload=payload,
        truncated=False,
        artifacts=[],
        next_hint=next_hint,
    )

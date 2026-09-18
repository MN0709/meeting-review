"""PRD R-P1-10：事件钩子系统（s04）。

固定钩子点：SessionStart / PreToolUse / PostToolUse / Stop。
- 审计（R-P1-5）、结构化日志都实现为钩子，主循环只负责触发；
- **钩子抛异常不得影响主循环**：只记日志并跳过；
- PreToolUse 返回 deny 时工具不执行（与权限共用同一条路径）。

移除全部钩子后主循环仍可单独跑通（钩子是可插拔的）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

HookEvent = str
HOOK_EVENTS: tuple[str, ...] = ("SessionStart", "PreToolUse", "PostToolUse", "Stop")


@dataclass
class HookDecision:
    deny: bool = False
    reason: str = ""


HookHandler = Callable[[HookEvent, Dict[str, Any]], Optional[HookDecision]]


class HookManager:
    def __init__(self) -> None:
        self._handlers: Dict[str, List[HookHandler]] = {event: [] for event in HOOK_EVENTS}

    def register(self, event: HookEvent, handler: HookHandler) -> None:
        if event not in self._handlers:
            raise ValueError("未知钩子事件：{}（可用：{}）".format(event, "、".join(HOOK_EVENTS)))
        self._handlers[event].append(handler)

    def handlers(self, event: HookEvent) -> List[HookHandler]:
        return list(self._handlers.get(event, []))

    def trigger(self, event: HookEvent, payload: Dict[str, Any]) -> HookDecision:
        """触发某事件的全部钩子；任一钩子异常只记录并跳过，绝不打断主循环。"""
        decision = HookDecision()
        for handler in list(self._handlers.get(event, [])):
            try:
                outcome = handler(event, payload)
            except Exception as exc:
                logger.warning(
                    "hook_failed event=%s handler=%s error_type=%s",
                    event, getattr(handler, "__name__", "unknown"), type(exc).__name__,
                )
                continue
            if outcome is not None and getattr(outcome, "deny", False):
                decision.deny = True
                decision.reason = getattr(outcome, "reason", "") or decision.reason
        return decision

    def __len__(self) -> int:
        return sum(len(items) for items in self._handlers.values())

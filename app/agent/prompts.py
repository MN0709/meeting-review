"""Agent 系统提示词（PRD §9.6）。

s15 集成 Harness 会在**每轮调用前动态重建**提示词：
身份 + 工作方式 + 工具目录 + 当前计划 + 上下文与预算 + 权限红线 + 交还时机。
工具增删只影响「工具目录」这一段，主循环结构不变。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from app.agent.tools.registry import ToolSpec

IDENTITY = """你是「会脉」的会议分析 Agent。你的任务是把一场会议转写整理成一份可核对的团队报告：决定了什么、谁承诺了什么、还有什么没解决。每一项结论都必须能回到原话与时间戳。"""

WORKFLOW = """# 工作方式
- 第一步先用 todo_write 写下计划（分几批、先抽哪类信息），再动手。
- 用 get_transcript 看这场会议的转写概览，再决定要读哪一段。
- 不确定时用工具自己查证：read_segment_window 回读原片段，search_transcript 检索关键词。
- 调用 extract_report 生成报告初稿；再用 validate_evidence 自检。
- validate_evidence 报错时，用 read_segment_window 回读对应时间点附近的原文，重新 extract_report；不要整场重来。
- 每完成一步就用 todo_write 更新进度。
- 校验通过后，**停止调用工具**，用一句话说明报告已完成。"""

RULES = """# 权限与红线（不可协商）
- 你只能调用 readonly 工具；删除会议/项目/声纹、合并成员、写长期声纹属于 host_owned，你无权调用，也不要尝试。
- team_id 是你的信任边界：跨团队的查询一律被拒绝，不要绕过。
- 不编造：证据不足就写「未明确」并标记待确认，不得改写原话或伪造时间戳。
- 不要把转写全文或口令写入任何输出。"""

HANDOVER = """# 交还人的时机
- 说话人真实身份、会议归属项目、行动项是否完成：只给建议，交人确认。
- 你不再调用工具时，系统会请一个**独立评估器**判断「是否真的完成」；若未达成，你会收到缺失项清单并可以继续。
- 步数或时间耗尽时明确交还，不要静默假装完成。"""

_PLAN_LABELS = {"pending": "未开始", "in_progress": "进行中", "done": "已完成"}


def _tool_catalog(tool_specs: Iterable[ToolSpec]) -> str:
    lines = ["# 可用工具"]
    for spec in tool_specs:
        lines.append("- {}：{}".format(spec.name, spec.description))
    return "\n".join(lines)


def _plan_section(plan: Optional[List[Dict[str, Any]]]) -> str:
    if not plan:
        return "# 当前计划\n（尚未写计划；请先调用 todo_write。）"
    lines = ["# 当前计划"]
    for index, item in enumerate(plan, start=1):
        status = _PLAN_LABELS.get(str(item.get("status")), str(item.get("status")))
        lines.append("{}. [{}] {}".format(index, status, item.get("task", "")))
    return "\n".join(lines)


def _budget_section(
    context_budget_tokens: Optional[int], step: Optional[int], max_steps: Optional[int]
) -> str:
    lines = ["# 上下文与预算"]
    lines.append("- 默认只拿转写摘要（概览 + 说话人分布），全文按需检索，不要一次性读完整场。")
    if context_budget_tokens:
        lines.append("- 上下文预算约 {} token；接近预算时优先裁剪工具结果，不要丢权威指令与未完成的计划。".format(context_budget_tokens))
    if step is not None and max_steps:
        lines.append("- 当前已进行 {} 步，最多 {} 步；接近上限时请尽快完成或明确交还。".format(step, max_steps))
    return "\n".join(lines)


def build_system_prompt(
    tool_specs: Iterable[ToolSpec],
    *,
    plan: Optional[List[Dict[str, Any]]] = None,
    context_budget_tokens: Optional[int] = None,
    step: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> str:
    parts = [
        IDENTITY,
        WORKFLOW,
        _tool_catalog(tool_specs),
        _plan_section(plan),
        _budget_section(context_budget_tokens, step, max_steps),
        RULES,
        HANDOVER,
    ]
    return "\n\n".join(parts)


def build_meeting_goal(meeting_id: str) -> str:
    return (
        "请分析会议 {meeting_id}，产出一份可核对的团队报告。"
        "完成后停止调用工具并简要说明结果。".format(meeting_id=meeting_id)
    )

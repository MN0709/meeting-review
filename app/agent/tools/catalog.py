"""工具目录：把 8 个只读工具 + 1 个规划工具 + 3 个能力工具装配进一个注册表。

新增工具的方式：在 readonly.py / planning.py / capability.py 加一个函数，
再在对应 register_* 里加一行 registry.register_tool(...)。
"""

from __future__ import annotations

from typing import Optional

from app.agent.planning import register_planning_tools
from app.agent.tools.capability import register_capability_tools
from app.agent.tools.readonly import register_readonly_tools
from app.agent.tools.registry import ToolRegistry
from app.db import Database
from app.llm import LLMAnalyzer

READONLY_TOOL_NAMES = (
    "get_transcript",
    "search_transcript",
    "read_segment_window",
    "get_report",
    "get_project_memory",
    "list_action_items",
    "list_members",
    "list_meetings",
)
PLANNING_TOOL_NAMES = ("todo_write",)
CAPABILITY_TOOL_NAMES = ("extract_report", "validate_evidence", "suggest_title")


def register_all_tools(
    registry: ToolRegistry, db: Database, analyzer: Optional[LLMAnalyzer] = None
) -> ToolRegistry:
    """注册 8 个只读工具 + 1 个规划工具（+ 3 个能力工具，若给了 analyzer）。"""
    register_readonly_tools(registry, db)
    register_planning_tools(registry, db)
    if analyzer is not None:
        register_capability_tools(registry, db, analyzer)
    return registry

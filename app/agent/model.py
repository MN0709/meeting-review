"""把工具目录转成 OpenAI tools 并调用模型。

模型层只负责「发请求 + 记用量」；权限判定在主循环里，模型无法绕过。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from app.agent.tools.registry import ToolSpec
from app.llm import LLMAnalyzer


class AgentModel:
    def __init__(self, analyzer: LLMAnalyzer) -> None:
        self._analyzer = analyzer

    async def step(
        self,
        messages: List[Dict[str, Any]],
        tool_specs: Iterable[ToolSpec],
        *,
        stage: str = "agent_step",
        usage_context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        tools = [self._to_openai_tool(spec) for spec in tool_specs]
        return await self._analyzer.chat_with_tools(
            messages, tools or None, stage=stage, usage_context=usage_context
        )

    @staticmethod
    def _to_openai_tool(spec: ToolSpec) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema,
            },
        }

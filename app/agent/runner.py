"""组装一次「会议分析」Agent 会话。

注意：Agent 循环自身**不写业务库**；报告落库由确定性代码在循环后执行。
"""

from __future__ import annotations

from typing import Optional

from app.agent.hooks import HookManager
from app.agent.loop import AgentLoop
from app.agent.model import AgentModel
from app.agent.permissions.policy import PermissionPolicy
from app.agent.prompts import build_meeting_goal, build_system_prompt
from app.agent.session import AgentLimits, AgentSession
from app.agent.tools.registry import ToolRegistry
from app.llm import LLMAnalyzer


class MeetingAgentRunner:
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        analyzer: LLMAnalyzer,
        policy: PermissionPolicy,
        hooks: Optional[HookManager] = None,
        limits: Optional[AgentLimits] = None,
    ) -> None:
        self.registry = registry
        self.limits = limits or AgentLimits()
        self.loop = AgentLoop(
            registry=registry,
            model=AgentModel(analyzer),
            policy=policy,
            hooks=hooks or HookManager(),
            limits=self.limits,
        )

    async def run(
        self, *, meeting_id: str, team_id: int, session_id: Optional[str] = None
    ) -> AgentSession:
        session = AgentSession(
            goal=build_meeting_goal(meeting_id),
            system_prompt=build_system_prompt(self.registry.list_tools()),
            usage_context={"team_id": team_id, "meeting_id": meeting_id},
            session_id=session_id,
        )
        return await self.loop.run(session)

"""R-P1-12（s15）：集成 Harness —— 唯一的装配点。

把已选的 P1 机制归一到**一个循环 + 一份动态系统提示词**：
  s01 循环 / s02 工具 / s03 权限 / s04 钩子 / s05 规划 / s17 完成判定。

约束：
- 全仓只有 loop.py 一处主循环；
- 新增或移除一个工具/机制，只改本文件（或工具的注册处），不改循环结构；
- 组件之间不互相 import，全部由本文件注入。
"""

from __future__ import annotations

from typing import Any, List, Optional

from app.agent.context import ContextManager
from app.agent.hooks import HookManager
from app.agent.loop import AgentLoop
from app.agent.model import AgentModel
from app.agent.permissions.policy import PermissionPolicy
from app.agent.prompts import build_meeting_goal, build_system_prompt
from app.agent.session import AgentLimits, AgentSession
from app.agent.tools.registry import ToolRegistry, ToolSpec
from app.llm import LLMAnalyzer


class AgentHarness:
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        analyzer: Optional[LLMAnalyzer],
        policy: PermissionPolicy,
        hooks: Optional[HookManager] = None,
        planner: Optional[Any] = None,
        goal_judge: Optional[Any] = None,
        limits: Optional[AgentLimits] = None,
        context_budget_tokens: Optional[int] = None,
    ) -> None:
        self.registry = registry
        self.analyzer = analyzer
        self.policy = policy
        self.hooks = hooks or HookManager()
        self.planner = planner
        self.goal_judge = goal_judge
        self.limits = limits or AgentLimits()
        self.context_budget_tokens = context_budget_tokens
        self.context_manager = (
            ContextManager(context_budget_tokens) if context_budget_tokens else None
        )
        self.loop = AgentLoop(
            registry=registry,
            model=AgentModel(analyzer) if analyzer is not None else None,
            policy=policy,
            hooks=self.hooks,
            limits=self.limits,
            planner=planner,
            goal_judge=goal_judge,
            prompt_builder=self.build_prompt,
            context_manager=self.context_manager,
        )

    def available_tools(self) -> List[ToolSpec]:
        return [
            spec
            for spec in self.registry.list_tools()
            if self.policy.decide(spec.name, spec.permission_level).allow
        ]

    def build_prompt(self, session: Optional[AgentSession] = None) -> str:
        return build_system_prompt(
            self.available_tools(),
            plan=session.todo if session is not None else None,
            context_budget_tokens=self.context_budget_tokens,
            step=session.steps if session is not None else 0,
            max_steps=self.limits.max_steps,
        )

    async def run_meeting(
        self, *, meeting_id: str, team_id: int, session_id: Optional[str] = None
    ) -> AgentSession:
        session = AgentSession(
            goal=build_meeting_goal(meeting_id),
            system_prompt=self.build_prompt(),
            usage_context={"team_id": team_id, "meeting_id": meeting_id},
            session_id=session_id,
        )
        return await self.loop.run(session)

    # 兼容别名
    async def run(self, *, meeting_id: str, team_id: int, session_id: Optional[str] = None) -> AgentSession:
        return await self.run_meeting(meeting_id=meeting_id, team_id=team_id, session_id=session_id)

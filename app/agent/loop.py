"""R-P1-1：标准 Agent 循环（全仓唯一主循环，s15 在其上装配各机制）。

循环骨架：重建系统提示词 → 调用模型 → 若没有 tool_use 则交给完成判定 →
逐个执行工具并回灌结果 → 继续。

终止条件是「模型不再产生 tool_use」且独立评估器认可，不是固定轮次；
三重保障：max_steps + 单步超时 + 会话超时。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

from app.agent.hooks import HookManager
from app.agent.permissions.policy import PermissionPolicy
from app.agent.session import AgentLimits, AgentSession, GoalJudgment, ToolCallRecord, bind_session
from app.agent.tools.contract import ToolResult, failure
from app.agent.tools.registry import ToolRegistry, ToolSpec

logger = logging.getLogger(__name__)


def _digest(args: Dict[str, Any]) -> str:
    """参数摘要（sha256 前 16 位）——审计只存摘要，不存原文。"""
    payload = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class AgentLoop:
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        model: Any,
        policy: PermissionPolicy,
        hooks: Optional[HookManager] = None,
        limits: Optional[AgentLimits] = None,
        planner: Optional[Any] = None,
        goal_judge: Optional[Any] = None,
        prompt_builder: Optional[Any] = None,
    ) -> None:
        self.registry = registry
        self.model = model
        self.policy = policy
        self.hooks = hooks or HookManager()
        self.limits = limits or AgentLimits()
        self.planner = planner
        self.goal_judge = goal_judge
        self.prompt_builder = prompt_builder

    def available_tools(self) -> List[ToolSpec]:
        """只把策略允许的工具暴露给模型（拒绝判定仍会再走一次，双保险）。"""
        return [
            spec
            for spec in self.registry.list_tools()
            if self.policy.decide(spec.name, spec.permission_level).allow
        ]

    async def run(self, session: AgentSession) -> AgentSession:
        self.hooks.trigger("SessionStart", self._hook_payload(session))
        started = time.monotonic()
        try:
            while True:  # 全仓唯一主循环
                if session.steps >= self.limits.max_steps:
                    session.status = "max_steps"
                    break
                if time.monotonic() - started > self.limits.session_timeout_seconds:
                    session.status = "session_timeout"
                    break
                session.refresh_system_prompt(self.prompt_builder)
                try:
                    response = await asyncio.wait_for(
                        self.model.step(
                            session.messages,
                            self.available_tools(),
                            stage="agent_step",
                            usage_context=session.usage_context,
                        ),
                        timeout=self.limits.step_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    session.status = "step_timeout"
                    break
                except Exception as exc:
                    logger.warning(
                        "agent_model_error session_id=%s error_type=%s",
                        session.session_id, type(exc).__name__,
                    )
                    session.notes.append("model_error:{}".format(type(exc).__name__))
                    session.status = "model_error"
                    break

                message = response.choices[0].message
                tool_calls = list(getattr(message, "tool_calls", None) or [])
                if not tool_calls:
                    if await self._finalize(session):
                        break
                    continue

                session.append_assistant(getattr(message, "content", None), tool_calls)
                for call in tool_calls:
                    await self._execute_tool_call(session, call)
                session.steps += 1
        finally:
            self.hooks.trigger("Stop", self._hook_payload(session, include_status=True))
        return session

    async def _finalize(self, session: AgentSession) -> bool:
        """模型停手：交由独立评估器判定。返回 True 表示可以结束循环。"""
        if self.goal_judge is None:
            session.status = "completed"
            return True
        try:
            judgment = await self.goal_judge.judge(session)
        except Exception as exc:  # 判定器自身异常也必须交还人
            logger.warning("goal_judge_failed error_type=%s", type(exc).__name__)
            judgment = GoalJudgment(
                False, "完成判定失败（{}）".format(type(exc).__name__), [], "unknown", error=True
            )
        session.judgments.append(judgment)
        if judgment.error:
            session.status = "needs_human"
            session.notes.append("goal_judge_error")
            return True
        if judgment.done:
            session.status = "completed"
            return True
        if session.judge_retries >= self.limits.max_judge_retries:
            session.status = "needs_human"
            session.notes.append("goal_not_reached")
            return True
        session.judge_retries += 1
        missing = "；".join(judgment.missing[:5]) or "请补全报告"
        session.append_observation(
            "独立评估认为尚未完成：{}\n缺失项：{}\n请继续修正后再停手。".format(judgment.reason, missing)
        )
        return False

    async def _execute_tool_call(self, session: AgentSession, call: Any) -> None:
        name = getattr(call.function, "name", "") or ""
        raw_arguments = getattr(call.function, "arguments", "") or "{}"
        args, args_ok = self._parse_arguments(raw_arguments)
        spec = self.registry.get(name)
        verdict = self.policy.decide(name, spec.permission_level if spec else None)
        step = session.steps + 1
        started = time.monotonic()
        decision = verdict.decision

        if not verdict.allow:
            result: ToolResult = failure(verdict.code, verdict.reason)
        elif not args_ok:
            decision = "deny_policy"
            result = failure("invalid_args", "工具参数不是合法的 JSON 对象。")
        else:
            pre = self.hooks.trigger(
                "PreToolUse", {**self._hook_payload(session), "step": step, "tool_name": name}
            )
            if pre.deny:
                decision = "deny_policy"
                result = failure("denied_policy", pre.reason or "该工具调用被钩子拦截。")
            else:
                with bind_session(session):
                    result = await self._invoke(spec, session, args)

        duration_ms = int((time.monotonic() - started) * 1000)
        self._collect_artifact(session, name, result)
        reminder = self.planner.reminder(session, step) if self.planner else None
        self.hooks.trigger(
            "PostToolUse",
            {
                **self._hook_payload(session),
                "step": step,
                "tool_name": name,
                "args_digest": _digest(args),
                "decision": decision,
                "result_code": result.code,
                "duration_ms": duration_ms,
            },
        )
        session.records.append(
            ToolCallRecord(
                step=step, tool_name=name, decision=decision,
                result_code=result.code, duration_ms=duration_ms,
            )
        )
        session.append_tool_result(call.id, result, extra=reminder or "")

    async def _invoke(self, spec: Optional[ToolSpec], session: AgentSession, args: Dict[str, Any]) -> ToolResult:
        if spec is None:
            return failure("denied_policy", "未知工具：{}".format(spec))
        team_id = session.team_id
        try:
            if asyncio.iscoroutinefunction(spec.handler):
                return await asyncio.wait_for(
                    spec.handler(team_id, **args), timeout=self.limits.step_timeout_seconds
                )
            return await asyncio.wait_for(
                asyncio.to_thread(spec.handler, team_id, **args),
                timeout=self.limits.step_timeout_seconds,
            )
        except asyncio.TimeoutError:
            return failure("tool_error", "工具执行超时。")
        except TypeError:
            return failure("invalid_args", "工具参数与契约不匹配。")
        except Exception as exc:  # 工具异常降级为可读观察，不穿透主循环
            return failure("tool_error", "工具执行失败（{}）。".format(type(exc).__name__))

    @staticmethod
    def _parse_arguments(raw: Any):
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            return {}, False
        if not isinstance(parsed, dict):
            return {}, False
        return parsed, True

    @staticmethod
    def _collect_artifact(session: AgentSession, name: str, result: ToolResult) -> None:
        if not result.ok or not isinstance(result.payload, dict):
            return
        if name == "extract_report":
            report = result.payload.get("report")
            if isinstance(report, dict):
                session.report = report
        elif name == "validate_evidence":
            session.validation = result.payload

    @staticmethod
    def _hook_payload(session: AgentSession, *, include_status: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "session_id": session.session_id,
            "team_id": session.team_id,
            "meeting_id": session.meeting_id,
        }
        if include_status:
            payload["status"] = session.status
            payload["steps"] = session.steps
        return payload

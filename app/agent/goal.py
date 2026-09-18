"""R-P1-11（s17）：最小完成判定。

主循环不再产生 tool_use 时，由一个**独立评估器**判断「目标是否真的达成」：
- 只接收「目标 + 结果摘要」，**不共享主循环的 messages**；
- 输出 {done, reason, missing}；done=false 时把 missing 回灌，允许继续；
- 评估器失败 / 超限时交还人（needs_human），**绝不当成成功**。

默认用规则评估器（零成本、可复现）；`AGENT_GOAL_JUDGE=model` 时用模型评估器，
按 `stage=goal_judge` 单独计费（R-P1-11 ④）。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.agent.session import AgentSession, GoalJudgment

logger = logging.getLogger(__name__)

GOAL_JUDGE_SYSTEM_PROMPT = (
    "你是独立的成果验收员。只看目标与成果摘要，判断「是否真的完成」，不要复述、不要编造。"
    "只输出 JSON：{\"done\": true/false, \"reason\": \"一句话中文理由\", \"missing\": [\"缺失项\"]}。"
    "成果为空、总览为空、或没有任何要点/决策/行动项/遗留问题时，done 必须为 false。"
)

_CONTENT_KEYS = ("meeting_points", "decisions", "action_items", "unresolved_issues")


class RuleGoalEvaluator:
    """确定性评估器：不需要模型，适合默认与测试。"""

    name = "rule"

    def evaluate(self, session: AgentSession) -> GoalJudgment:
        validation = session.validation
        if isinstance(validation, dict) and validation.get("valid") is False:
            invalid = validation.get("invalid") or []
            missing = [
                "引文需修正：{}（最近原文：{}）".format(
                    item.get("quote", ""), item.get("nearest_segment_preview", "")
                )
                for item in invalid[:5]
            ]
            return GoalJudgment(False, "引文校验未通过（{} 条）".format(len(invalid)), missing or ["请修正引文"], self.name)

        report = session.report
        if not isinstance(report, dict):
            return GoalJudgment(False, "没有产出报告", ["请调用 extract_report 生成报告"], self.name)
        if not str(report.get("overview") or "").strip():
            return GoalJudgment(False, "报告总览为空", ["overview 不能为空"], self.name)
        counts = {key: len(report.get(key) or []) for key in _CONTENT_KEYS}
        if sum(counts.values()) == 0:
            return GoalJudgment(
                False, "报告没有任何要点/决策/行动项/遗留问题",
                ["至少需要一项会议要点或决策或行动项"], self.name,
            )
        return GoalJudgment(
            True, "报告含总览与 {} 项内容".format(sum(counts.values())), [], self.name
        )

    async def __call__(self, session: AgentSession) -> GoalJudgment:
        return self.evaluate(session)


def _extract_json(content: str) -> Dict[str, Any]:
    cleaned = (content or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


class PromptGoalEvaluator:
    """模型评估器：只拿目标与结果摘要，走独立的一次调用。"""

    name = "model"

    def __init__(self, analyzer: Any) -> None:
        self._analyzer = analyzer

    async def __call__(self, session: AgentSession) -> GoalJudgment:
        payload = {
            "goal": session.goal,
            "report": session.report,
            "validation": session.validation,
        }
        messages = [
            {"role": "system", "content": GOAL_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)[:8000]},
        ]
        response = await self._analyzer.chat_with_tools(
            messages, None, stage="goal_judge", usage_context=session.usage_context
        )
        content = getattr(response.choices[0].message, "content", "") or ""
        data = _extract_json(content)
        if not data:
            return GoalJudgment(False, "模型评估器返回无法解析", ["请补全报告后重试"], self.name, error=True)
        missing: List[str] = [str(item) for item in (data.get("missing") or [])][:10]
        return GoalJudgment(
            done=bool(data.get("done")),
            reason=str(data.get("reason") or "")[:200],
            missing=missing,
            evaluator=self.name,
        )


class GoalJudge:
    def __init__(self, evaluator: Any) -> None:
        self.evaluator = evaluator

    @property
    def evaluator_name(self) -> str:
        return getattr(self.evaluator, "name", "unknown")

    async def judge(self, session: AgentSession) -> GoalJudgment:
        try:
            judgment = await self.evaluator(session)
        except Exception as exc:  # 评估器失败必须交还人，不能静默成功
            logger.warning("goal_judge_failed error_type=%s", type(exc).__name__)
            return GoalJudgment(
                False, "完成判定失败（{}）".format(type(exc).__name__), [],
                self.evaluator_name, error=True,
            )
        return judgment


def build_goal_evaluator(mode: str, analyzer: Optional[Any] = None) -> Any:
    """按配置选择评估器；model 模式缺 analyzer 时退回规则评估器。"""
    if mode == "model" and analyzer is not None:
        return PromptGoalEvaluator(analyzer)
    return RuleGoalEvaluator()

import asyncio
import contextlib
import contextvars
import copy
import json
import logging
import re
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Type

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.models import ChunkSummary, SemanticAnalysis, TeamChunkSummary, TeamMeetingReport, Transcript, TranscriptSegment


logger = logging.getLogger(__name__)

# 成本归因上下文（team_id / meeting_id / project_id）。
# 用 ContextVar 而不是实例属性：asyncio.to_thread 会复制上下文，
# 因此工作线程里的 _request_sync 能读到同一个值；若未设置则为 None。
_usage_context_var: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "llm_usage_context", default=None
)


def usage_scope(context: Optional[Dict[str, Any]]) -> contextlib.AbstractContextManager:
    """在 `with` 块内为所有 LLM 调用附加归因上下文；退出时恢复。"""

    @contextlib.contextmanager
    def _scope() -> Iterator[None]:
        token = _usage_context_var.set(context)
        try:
            yield
        finally:
            _usage_context_var.reset(token)

    return _scope()

SYSTEM_PROMPT = """
你是严谨的个人会议表现教练。你的分析对象是录音中的用户。
只能根据输入内容分析，不得补写不存在的信息。
表现分析覆盖表达清晰度、回答是否先给结论、逻辑结构。每条评价必须引用输入中完全一致的原话，不得改写引文。
每个维度和总体表现均按 1-10 分评分：1-4 分代表有明显问题，5-7 分代表合格但有提升空间，8-10 分代表优秀。
维度分必须由该维度 evidence 中的原话支撑，不得凭感觉给分。总分不是三个维度的简单平均：不同维度对当次会议目标的影响不同，应结合关键优点、主要短板和任务完成度综合判断，并在 overall_comment 中用一句话说明综合判断理由（不超过 50 字）。
待办的 owner 或 deadline 不明确时填“未明确”。改进建议限 3-5 条，必须是可操作的具体动作。
只输出符合指定 JSON Schema 的 JSON，不输出 Markdown 或解释。
""".strip()

CHUNK_SYSTEM_PROMPT = """
你负责从会议转写分块中提取后续复盘所需信息。
保留要点、结论、待办、个人表现观察，并保留有代表性的逐字原话及其时间戳。
引文必须是输入原文的完全一致子串，不得改写。
只输出符合指定 JSON Schema 的 JSON。
""".strip()

TEAM_SYSTEM_PROMPT = """
你是严谨的团队正式会议分析助手。只能根据转写内容提炼信息，不得补写。
先根据会议的核心主题生成一个 8-20 个字的中文建议标题，不加书名号，不使用“会议纪要”等空泛名称。
再用不超过 300 字概括会议目标、进展和结果，并输出会议要点、决策清单、行动项与遗留问题。
每条决策、行动项和遗留问题都必须包含输入中完全一致的原话和时间戳，不得改写引文；无法确认决策人或负责人时填“未明确”，但仍必须给出对应原话（evidence）。
行动项的 evidence 用于核对“谁承诺了什么”，不得省略。
遗留问题只记录会上明确提出但尚未解决、尚未决定或需要后续确认的事项，没有则返回空数组。
紧急事项只记录“带时间压力、需要尽快处理”的事项（如“今天必须定”“明天上线前要改完”），必须带原话；普通行动项、已完成安排、泛泛的“要注意”都不算，没有则返回空数组。
如果用户消息里给了「当前团队已有项目」列表，可以额外给出项目归属建议 suggested_project：优先复用已有项目（existing_project_id 必须从给定列表里选，不得编造）；确实都不合适时才给 new_project_name；无法判断时三个字段都留空。绝对不要输出“这几场会属于同一个项目”这类解释性总结。
行动项的截止时间不明确时填“未明确”。speaker_stats_note 输出空字符串，它会由后端本地说话人识别结果覆盖，不要用 LLM 猜测说话人。
只输出符合指定 JSON Schema 的 JSON，不输出 Markdown 或解释。
""".strip()

TEAM_CHUNK_SYSTEM_PROMPT = """
你负责从团队正式会议的一个转写分块中提取会议要点、明确决策、行动项、遗留问题和紧急事项。
每条决策、行动项、遗留问题和紧急事项必须保留完全一致的逐字原话及时间戳，不得改写引文；无法确认决策人、负责人或截止时间时填“未明确”，但仍必须给出对应原话（evidence）。
行动项的 evidence 用于核对“谁承诺了什么”，不得省略。
遗留问题只记录明确提出但尚未解决、尚未决定或需要后续确认的事项，没有则返回空数组。
紧急事项只记录本块里“带时间压力、需要尽快处理”的事项，必须带原话；普通行动项、已完成安排、泛泛的“要注意”都不算，没有则返回空数组。
不要分析个人表现，不要补写分块中不存在的信息。只输出符合指定 JSON Schema 的 JSON。
""".strip()

DIMENSION_ALIASES = {
    "表达清晰度": "表达清晰度",
    "clarity": "表达清晰度",
    "clear": "表达清晰度",
    "清晰度": "表达清晰度",
    "表达清晰": "表达清晰度",
    "结论先行": "结论先行",
    "conclusion_first": "结论先行",
    "conclusion-first": "结论先行",
    "conclusion": "结论先行",
    "是否先给结论": "结论先行",
    "逻辑结构": "逻辑结构",
    "logical_structure": "逻辑结构",
    "logical-structure": "逻辑结构",
    "logic": "逻辑结构",
    "逻辑": "逻辑结构",
}
DIMENSION_ORDER = ("表达清晰度", "结论先行", "逻辑结构")


class AnalysisError(RuntimeError):
    pass


def _json_mode_contract(model_type: Type[BaseModel]) -> str:
    if issubclass(model_type, TeamMeetingReport):
        example = {
            "suggested_title": "内测上线安排确认",
            "overview": "会议围绕内测上线安排展开，明确了发布时间和准备工作。",
            "meeting_points": ["会议要点"],
            "decisions": [{
                "content": "决策内容", "decision_maker": "未明确",
                "evidence": {"quote": "原话", "timestamp": "00:00:00"},
            }],
            "action_items": [{"task": "任务", "owner": "未明确", "deadline": "未明确",
                              "evidence": {"quote": "原话", "timestamp": "00:00:00"}}],
            "unresolved_issues": [{
                "content": "尚未解决的问题",
                "evidence": {"quote": "原话", "timestamp": "00:00:00"},
            }],
            "urgent_items": [{
                "content": "需要尽快处理的事项",
                "evidence": {"quote": "原话", "timestamp": "00:00:00"},
            }],
            "suggested_project": {
                "existing_project_id": "已有项目 id 或留空",
                "new_project_name": "",
                "confidence": 0.0,
                "reason": "建议依据（一句话）",
            },
            "speaker_stats_note": "",
        }
        fields = (
            "suggested_title(必填字符串，8-20 个字)；overview(必填字符串，不超过 300 字)；"
            "meeting_points(必填字符串数组)；"
            "decisions(必填对象数组，每项只有 content/decision_maker/evidence，"
            "evidence 只有 quote/timestamp)；action_items(必填对象数组，每项只有 task/owner/deadline/evidence，"
            "evidence 只有 quote/timestamp)；"
            "unresolved_issues(必填对象数组，每项只有 content/evidence，evidence 只有 quote/timestamp)；"
            "urgent_items(必填对象数组，每项只有 content/evidence，evidence 只有 quote/timestamp；"
            "只收带时间压力、需尽快处理的事项，没有就返回空数组)；"
            "suggested_project(可选对象，只有 existing_project_id/new_project_name/confidence/reason；"
            "优先复用给定项目 id，不得编造；无法判断时给空对象)；"
            "speaker_stats_note(必填空字符串，由后端覆盖)"
        )
    elif issubclass(model_type, TeamChunkSummary):
        example = {
            "meeting_points": ["会议要点"],
            "decisions": [{"content": "决策内容", "decision_maker": "未明确", "evidence": {"quote": "原话", "timestamp": "00:00:00"}}],
            "action_items": [{"task": "任务", "owner": "未明确", "deadline": "未明确", "evidence": {"quote": "原话", "timestamp": "00:00:00"}}],
            "unresolved_issues": [{"content": "尚未解决的问题", "evidence": {"quote": "原话", "timestamp": "00:00:00"}}],
            "urgent_items": [{"content": "需要尽快处理的事项", "evidence": {"quote": "原话", "timestamp": "00:00:00"}}],
        }
        fields = (
            "meeting_points(必填字符串数组)；decisions(必填对象数组，每项只有 content/decision_maker/evidence，"
            "evidence 只有 quote/timestamp)；action_items(必填对象数组，每项只有 task/owner/deadline/evidence，"
            "evidence 只有 quote/timestamp)；"
            "unresolved_issues(必填对象数组，每项只有 content/evidence，evidence 只有 quote/timestamp)；"
            "urgent_items(必填对象数组，每项只有 content/evidence，evidence 只有 quote/timestamp；没有就返回空数组)"
        )
    elif issubclass(model_type, SemanticAnalysis):
        example = {
            "meeting_minutes": {
                "key_points": ["要点"],
                "conclusions": ["结论"],
                "action_items": [{"task": "任务", "owner": "未明确", "deadline": "未明确"}],
            },
            "overall_score": 7,
            "overall_comment": "结论清晰，但行动闭环还需加强。",
            "performance_analysis": [
                {"dimension": "表达清晰度", "score": 7, "assessment": "评价", "evidence": [{"quote": "原话", "timestamp": "00:00:00"}]},
                {"dimension": "结论先行", "score": 7, "assessment": "评价", "evidence": [{"quote": "原话", "timestamp": "00:00:00"}]},
                {"dimension": "逻辑结构", "score": 7, "assessment": "评价", "evidence": [{"quote": "原话", "timestamp": "00:00:00"}]},
            ],
            "improvement_suggestions": [
                {"title": "建议1", "action": "具体做法", "example": "改写示例"},
                {"title": "建议2", "action": "具体做法", "example": ""},
                {"title": "建议3", "action": "具体做法", "example": ""},
            ],
        }
        fields = (
            "meeting_minutes(必填对象): key_points(必填字符串数组)、conclusions(必填字符串数组)、"
            "action_items(必填对象数组，每项只有 task/owner/deadline)\n"
            "overall_score(必填整数，1-10)、overall_comment(必填字符串，最多 50 字)\n"
            "performance_analysis(必填数组，固定 3 项): 每项只有 dimension/score/assessment/evidence；"
            "dimension 依次为表达清晰度、结论先行、逻辑结构；evidence 是 quote/timestamp 对象数组\n"
            "improvement_suggestions(必填数组，3-5 项): 每项只有 title/action/example，example 可省略"
        )
    elif issubclass(model_type, ChunkSummary):
        example = {
            "key_points": ["要点"],
            "conclusions": ["结论"],
            "action_items": ["待办"],
            "personal_observations": ["表现观察"],
            "evidence_quotes": [{"quote": "原话", "timestamp": "00:00:00"}],
        }
        fields = (
            "key_points、conclusions、action_items、personal_observations 都是必填字符串数组；\n"
            "evidence_quotes 是必填对象数组，每项只有 quote 和 timestamp"
        )
    else:
        example = {}
        fields = "请严格遵守所需字段和类型"

    return (
        "\n\n【JSON 字段契约】\n{}\n"
        "最小 JSON 示例（仅示范结构，内容必须替换为本次输入中的真实信息）：\n{}\n"
        "字段名必须与示例完全一致，不要输出多余字段；performance_analysis 不得输出为对象，"
        "improvement_suggestions 不得输出为纯字符串数组。"
    ).format(fields, json.dumps(example, ensure_ascii=False, separators=(",", ":")))


def _canonical_dimension(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace(" ", "_")
    return DIMENSION_ALIASES.get(normalized) or DIMENSION_ALIASES.get(value.strip())


def _normalize_evidence(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for evidence in value:
        if isinstance(evidence, dict) and "quote" in evidence and "timestamp" in evidence:
            normalized.append({"quote": evidence["quote"], "timestamp": evidence["timestamp"]})
    return normalized


def _normalize_score(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"[+-]?\d+", stripped):
            return int(stripped)
    return value


def _performance_parts(value: Any) -> tuple[List[str], List[Dict[str, Any]], List[Any]]:
    entries = value if isinstance(value, list) else [value]
    assessments: List[str] = []
    evidence: List[Dict[str, Any]] = []
    scores: List[Any] = []
    for entry in entries:
        if isinstance(entry, str):
            assessments.append(entry)
        elif isinstance(entry, dict):
            assessment = next(
                (entry[key] for key in ("assessment", "finding", "content", "description") if entry.get(key)),
                None,
            )
            if assessment is not None:
                assessments.append(str(assessment))
            raw_evidence = entry.get("evidence", entry.get("evidence_quotes", entry.get("quotes", [])))
            evidence.extend(_normalize_evidence(raw_evidence))
            if "score" in entry:
                scores.append(_normalize_score(entry["score"]))
    return assessments, evidence, scores


def _normalize_performance(value: Any) -> Any:
    buckets: Dict[str, Dict[str, Any]] = {}
    if isinstance(value, dict):
        candidates = list(value.items())
    elif isinstance(value, list):
        candidates = [
            (item.get("dimension"), item)
            for item in value
            if isinstance(item, dict)
        ]
    else:
        return value

    for raw_dimension, content in candidates:
        dimension = _canonical_dimension(raw_dimension)
        if dimension is None:
            continue
        assessments, evidence, scores = _performance_parts(content)
        bucket = buckets.setdefault(dimension, {"assessment": [], "evidence": [], "scores": []})
        bucket["assessment"].extend(assessments)
        bucket["evidence"].extend(evidence)
        bucket["scores"].extend(scores)

    return [
        {
            "dimension": dimension,
            **({"score": buckets[dimension]["scores"][0]} if buckets[dimension]["scores"] else {}),
            "assessment": "；".join(buckets[dimension]["assessment"]),
            "evidence": buckets[dimension]["evidence"],
        }
        for dimension in DIMENSION_ORDER
        if dimension in buckets
    ]


def normalize_llm_payload(payload: Any, model_type: Type[BaseModel]) -> Any:
    """仅修复常见结构偏差；引文内容和时间戳不在这里改写。"""
    if not isinstance(payload, dict):
        return payload

    if issubclass(model_type, (TeamMeetingReport, TeamChunkSummary)):
        normalized: Dict[str, Any] = {
            key: copy.deepcopy(payload[key])
            for key in ("meeting_points", "decisions", "action_items", "unresolved_issues", "urgent_items")
            if key in payload
        }
        actions = normalized.get("action_items")
        if isinstance(actions, list):
            cleaned = []
            for item in actions:
                if isinstance(item, str):
                    cleaned.append({"task": item, "owner": "未明确", "deadline": "未明确"})
                elif isinstance(item, dict):
                    task = next((item.get(key) for key in ("task", "item", "content", "description") if item.get(key) is not None), "")
                    entry = {
                        "task": str(task),
                        "owner": str(item.get("owner") or "未明确"),
                        "deadline": str(item.get("deadline") or "未明确"),
                    }
                    # 行动项原话证据：结构正确才保留，不在这里改写引文内容
                    evidence = item.get("evidence")
                    if isinstance(evidence, dict) and "quote" in evidence and "timestamp" in evidence:
                        entry["evidence"] = {
                            "quote": evidence.get("quote", ""),
                            "timestamp": evidence.get("timestamp", ""),
                        }
                    cleaned.append(entry)
            normalized["action_items"] = cleaned

        decisions = normalized.get("decisions")
        if isinstance(decisions, list):
            cleaned_decisions = []
            for item in decisions:
                if not isinstance(item, dict):
                    continue
                evidence = item.get("evidence")
                if not isinstance(evidence, dict):
                    continue
                content = next(
                    (item.get(key) for key in ("content", "decision", "item", "description") if item.get(key)),
                    "",
                )
                cleaned_decisions.append({
                    "content": str(content),
                    "decision_maker": str(item.get("decision_maker") or item.get("owner") or "未明确"),
                    "evidence": {
                        "quote": evidence.get("quote", ""),
                        "timestamp": evidence.get("timestamp", ""),
                    },
                })
            normalized["decisions"] = cleaned_decisions

        issues = normalized.get("unresolved_issues")
        if isinstance(issues, list):
            cleaned_issues = []
            for item in issues:
                if not isinstance(item, dict) or not isinstance(item.get("evidence"), dict):
                    continue
                evidence = item["evidence"]
                content = next(
                    (item.get(key) for key in ("content", "question", "item", "description") if item.get(key)),
                    "",
                )
                cleaned_issues.append({
                    "content": str(content),
                    "evidence": {
                        "quote": evidence.get("quote", ""),
                        "timestamp": evidence.get("timestamp", ""),
                    },
                })
            normalized["unresolved_issues"] = cleaned_issues

        # R-P1.5-1：紧急事项与遗留问题结构一致，复用同一套清洗规则（引文不改写）。
        urgent = normalized.get("urgent_items")
        if isinstance(urgent, list):
            cleaned_urgent = []
            for item in urgent:
                if not isinstance(item, dict) or not isinstance(item.get("evidence"), dict):
                    continue
                evidence = item["evidence"]
                content = next(
                    (item.get(key) for key in ("content", "item", "description", "task") if item.get(key)),
                    "",
                )
                cleaned_urgent.append({
                    "content": str(content),
                    "evidence": {
                        "quote": evidence.get("quote", ""),
                        "timestamp": evidence.get("timestamp", ""),
                    },
                })
            normalized["urgent_items"] = cleaned_urgent

        # R-P1.5-8：项目归属建议（只清洗结构，是否合法由上层对照团队项目校验）。
        suggestion = payload.get("suggested_project")
        if isinstance(suggestion, dict):
            existing = str(suggestion.get("existing_project_id") or "").strip()
            new_name = str(suggestion.get("new_project_name") or "").strip()
            reason = str(suggestion.get("reason") or "").strip()
            try:
                confidence = float(suggestion.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            normalized["suggested_project"] = {
                "existing_project_id": existing or None,
                "new_project_name": new_name[:50] or None,
                "confidence": max(0.0, min(1.0, confidence)),
                "reason": reason[:200],
            }
        elif suggestion is not None:
            normalized.pop("suggested_project", None)

        if issubclass(model_type, TeamMeetingReport):
            suggested_title = next(
                (payload.get(key) for key in ("suggested_title", "meeting_title", "title") if payload.get(key)),
                None,
            )
            if suggested_title is not None:
                normalized["suggested_title"] = str(suggested_title)[:100]
            overview = next(
                (payload.get(key) for key in ("overview", "meeting_overview", "summary") if payload.get(key)),
                None,
            )
            if overview is not None:
                normalized["overview"] = str(overview)
            normalized["speaker_stats_note"] = ""
        return normalized

    if not issubclass(model_type, SemanticAnalysis):
        return payload

    normalized = copy.deepcopy(payload)
    if "overall_score" in normalized:
        normalized["overall_score"] = _normalize_score(normalized["overall_score"])
    meeting_minutes = normalized.get("meeting_minutes")
    if isinstance(meeting_minutes, dict) and isinstance(meeting_minutes.get("action_items"), list):
        action_items: List[Dict[str, Any]] = []
        for item in meeting_minutes["action_items"]:
            if isinstance(item, str):
                task = item
                owner = deadline = "未明确"
            elif isinstance(item, dict):
                task = next(
                    (item[key] for key in ("task", "item", "content", "description") if item.get(key) is not None),
                    "",
                )
                owner = item.get("owner") or "未明确"
                deadline = item.get("deadline") or "未明确"
            else:
                continue
            action_items.append({"task": str(task), "owner": str(owner), "deadline": str(deadline)})
        meeting_minutes["action_items"] = action_items

    normalized["performance_analysis"] = _normalize_performance(normalized.get("performance_analysis"))

    suggestions = normalized.get("improvement_suggestions")
    if isinstance(suggestions, list):
        normalized_suggestions: List[Dict[str, Any]] = []
        for suggestion in suggestions:
            if isinstance(suggestion, str):
                normalized_suggestions.append(
                    {"title": suggestion[:10], "action": suggestion, "example": ""}
                )
            elif isinstance(suggestion, dict):
                action = suggestion.get("action", "")
                title = suggestion.get("title") or str(action)[:10]
                normalized_suggestions.append(
                    {"title": str(title), "action": str(action), "example": str(suggestion.get("example") or "")}
                )
        normalized["improvement_suggestions"] = normalized_suggestions

    return normalized


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    return "{:02d}:{:02d}:{:02d}".format(total // 3600, (total % 3600) // 60, total % 60)


def format_segments(segments: Sequence[TranscriptSegment]) -> str:
    return "\n".join(
        "[{}-{}]{} {}".format(
            format_timestamp(item.start), format_timestamp(item.end),
            f"[{item.speaker_label}]" if item.speaker_label else "", item.text,
        )
        for item in segments
    )


def split_segments(segments: Sequence[TranscriptSegment], max_chars: int) -> List[List[TranscriptSegment]]:
    chunks: List[List[TranscriptSegment]] = []
    current: List[TranscriptSegment] = []
    current_size = 0

    expanded: List[TranscriptSegment] = []
    text_limit = max(1, max_chars - 24)
    for segment in segments:
        if len(segment.text) <= text_limit:
            expanded.append(segment)
            continue
        part_count = (len(segment.text) + text_limit - 1) // text_limit
        duration = max(0.0, segment.end - segment.start)
        for part_index in range(part_count):
            start_index = part_index * text_limit
            part_text = segment.text[start_index : start_index + text_limit]
            part_start = segment.start + duration * part_index / part_count
            part_end = segment.start + duration * (part_index + 1) / part_count
            expanded.append(TranscriptSegment(start=part_start, end=part_end, text=part_text))

    for segment in expanded:
        rendered_size = len(segment.text) + 24
        if current and current_size + rendered_size > max_chars:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(segment)
        current_size += rendered_size

    if current:
        chunks.append(current)
    return chunks


def _extract_json(content: str) -> Any:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def _timestamp_to_seconds(timestamp: str) -> int:
    hours, minutes, seconds = (int(part) for part in timestamp.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def _validate_quotes(evidence_quotes: Sequence[Any], segments: Sequence[TranscriptSegment]) -> None:
    invalid: List[str] = []
    for evidence in evidence_quotes:
        timestamp_seconds = _timestamp_to_seconds(evidence.timestamp)
        matching_segment = any(
            evidence.quote in segment.text
            and segment.start - 5 <= timestamp_seconds <= segment.end + 5
            for segment in segments
        )
        if not matching_segment:
            invalid.append("《{}》@{}".format(evidence.quote, evidence.timestamp))
    if invalid:
        raise ValueError("以下引文未出现在对应转写片段，或时间戳超出片段范围（容差 5 秒）: {}".format(invalid[:3]))


def validate_evidence(analysis: SemanticAnalysis, segments: Sequence[TranscriptSegment]) -> None:
    evidence_quotes = [
        evidence
        for finding in analysis.performance_analysis
        for evidence in finding.evidence
    ]
    _validate_quotes(evidence_quotes, segments)


def validate_chunk_evidence(summary: ChunkSummary, segments: Sequence[TranscriptSegment]) -> None:
    _validate_quotes(summary.evidence_quotes, segments)


def validate_team_evidence(report: TeamMeetingReport, segments: Sequence[TranscriptSegment]) -> None:
    _validate_quotes(
        [decision.evidence for decision in report.decisions]
        + [issue.evidence for issue in report.unresolved_issues]
        + [item.evidence for item in getattr(report, "urgent_items", []) or []],
        segments,
    )


def validate_team_action_evidence(report: Any, segments: Sequence[TranscriptSegment]) -> None:
    """行动项原话证据（新增）：要求每一项都有 evidence，且引文/时间戳可核对。

    只在新报告的分析路径上调用；旧报告缺失 evidence 时不会被重新校验（向前兼容）。
    """
    items = list(getattr(report, "action_items", []) or [])
    missing = [item.task for item in items if item.evidence is None]
    if missing:
        raise ValueError(
            "以下行动项缺少原话证据（evidence），无法核对负责人，请为每一项补上输入中转写原文的完全一致子串与时间戳: {}".format(missing[:3])
        )
    _validate_quotes([item.evidence for item in items], segments)


def validate_team_chunk_evidence(report: TeamChunkSummary, segments: Sequence[TranscriptSegment]) -> None:
    _validate_quotes(
        [decision.evidence for decision in report.decisions]
        + [issue.evidence for issue in report.unresolved_issues]
        + [item.evidence for item in getattr(report, "urgent_items", []) or []],
        segments,
    )


class LLMAnalyzer:
    def __init__(
        self,
        settings: Settings,
        client: Optional[OpenAI] = None,
        usage_recorder: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.settings = settings
        self._client = client
        self._client_lock = threading.Lock()
        # 成本落库的旁路回调；不传则只写日志（现有行为）。
        self._usage_recorder = usage_recorder
        # 该模型是否支持 response_format=json_schema。首次实测后缓存：
        # 否则每次调用都要先发一次必然失败的请求再降级，等于双倍请求与双倍计费。
        self._json_schema_supported: Optional[bool] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    if not self.settings.openai_api_key:
                        raise AnalysisError("缺少 OPENAI_API_KEY，请在 .env 中配置")
                    self._client = OpenAI(
                        api_key=self.settings.openai_api_key,
                        base_url=self.settings.openai_base_url,
                        timeout=120.0,
                        max_retries=1,
                    )
        return self._client

    def _record_usage(
        self,
        stage: str,
        model: str,
        usage: Any,
        duration_ms: int,
        usage_context: Optional[Dict[str, Any]],
    ) -> None:
        """把一次调用的用量交给旁路记录器。只交数值与标识，不交 prompt / 回复原文。"""
        if self._usage_recorder is None:
            return
        context = usage_context or {}
        team_id = context.get("team_id")
        if team_id is None:
            return
        try:
            self._usage_recorder(
                {
                    "team_id": team_id,
                    "meeting_id": context.get("meeting_id"),
                    "project_id": context.get("project_id"),
                    "stage": stage,
                    "model": model,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                    "duration_ms": duration_ms,
                }
            )
        except Exception as exc:  # 成本记账是旁路，永远不能影响主链路
            logger.warning(
                "llm_usage_record_failed stage=%s error_type=%s", stage, type(exc).__name__
            )

    def _request_sync(
        self,
        model_type: Type[BaseModel],
        system_prompt: str,
        user_prompt: str,
        stage: str,
        response_mode: str,
        usage_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        if response_mode == "json_schema" and self._json_schema_supported is False:
            # 已知该模型不支持 json_schema：直接走 json_object，省掉一次必然失败的请求。
            response_mode = "json_object"
        if usage_context is None:
            # 未显式传入时，取 usage_scope() 设置的归因上下文。
            usage_context = _usage_context_var.get()
        if response_mode == "json_schema":
            response_format: Dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {
                    "name": model_type.__name__,
                    "strict": True,
                    "schema": model_type.model_json_schema(),
                },
            }
        else:
            response_format = {"type": "json_object"}
            user_prompt += _json_mode_contract(model_type)

        started_at = time.monotonic()
        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                temperature=0.1,
                response_format=response_format,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception:
            if response_mode == "json_schema":
                self._json_schema_supported = False
            raise
        if response_mode == "json_schema":
            self._json_schema_supported = True
        duration_ms = int((time.monotonic() - started_at) * 1000)

        usage = getattr(response, "usage", None)
        logger.info(
            "llm_usage stage=%s model=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s duration_ms=%s",
            stage,
            self.settings.openai_model,
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
            getattr(usage, "total_tokens", None),
            duration_ms,
        )
        self._record_usage(stage, self.settings.openai_model, usage, duration_ms, usage_context)
        content = response.choices[0].message.content
        if not content:
            raise AnalysisError("LLM 返回了空内容")
        return content

    def _chat_with_tools_sync(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        stage: str,
        usage_context: Optional[Dict[str, Any]],
    ) -> Any:
        """R-P1-1：带工具目录的一轮对话（**新增方法**，不改任何现有内部实现）。

        复用同一个 OpenAI 客户端与同一条用量落库路径（R-P0-2），
        使 Agent 的每一次模型调用也能按 stage 归因成本。
        """
        if usage_context is None:
            usage_context = _usage_context_var.get()
        kwargs: Dict[str, Any] = {
            "model": self.settings.openai_model,
            "temperature": 0.1,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        started_at = time.monotonic()
        response = self.client.chat.completions.create(**kwargs)
        duration_ms = int((time.monotonic() - started_at) * 1000)
        usage = getattr(response, "usage", None)
        logger.info(
            "llm_usage stage=%s model=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s duration_ms=%s",
            stage,
            self.settings.openai_model,
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
            getattr(usage, "total_tokens", None),
            duration_ms,
        )
        self._record_usage(stage, self.settings.openai_model, usage, duration_ms, usage_context)
        return response

    async def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stage: str = "agent_step",
        usage_context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """带工具调用的一轮对话，供 Agent 循环使用。"""
        return await asyncio.to_thread(
            self._chat_with_tools_sync, messages, tools, stage, usage_context
        )

    async def _validated_call(
        self,
        model_type: Type[BaseModel],
        system_prompt: str,
        user_prompt: str,
        stage: str,
        transcript_segments: Optional[Sequence[TranscriptSegment]] = None,
        usage_context: Optional[Dict[str, Any]] = None,
    ) -> BaseModel:
        last_error: Optional[Exception] = None
        previous_output = ""
        response_mode = "json_schema"

        for attempt in range(self.settings.llm_max_retries + 1):
            repair = ""
            if last_error is not None:
                repair = (
                    "\n\n上一次输出校验失败，请修复并输出完整 JSON。"
                    "\n错误: {}\n上一次输出: {}".format(str(last_error)[:1000], previous_output[:5000])
                )
            try:
                try:
                    content = await asyncio.to_thread(
                        self._request_sync,
                        model_type,
                        system_prompt,
                        user_prompt + repair,
                        "{}:attempt{}".format(stage, attempt + 1),
                        response_mode,
                        usage_context,
                    )
                except Exception as exc:
                    if response_mode == "json_schema":
                        logger.warning(
                            "json_schema 调用失败，降级为 json_object error_type=%s",
                            type(exc).__name__,
                        )
                        response_mode = "json_object"
                        content = await asyncio.to_thread(
                            self._request_sync,
                            model_type,
                            system_prompt,
                            user_prompt + repair,
                            "{}:attempt{}:fallback".format(stage, attempt + 1),
                            response_mode,
                            usage_context,
                        )
                    else:
                        raise

                previous_output = content
                payload = normalize_llm_payload(_extract_json(content), model_type)
                parsed = model_type.model_validate(payload)
                if transcript_segments is not None:
                    if isinstance(parsed, SemanticAnalysis):
                        validate_evidence(parsed, transcript_segments)
                    elif isinstance(parsed, TeamMeetingReport):
                        validate_team_evidence(parsed, transcript_segments)
                        validate_team_action_evidence(parsed, transcript_segments)
                    elif isinstance(parsed, TeamChunkSummary):
                        validate_team_chunk_evidence(parsed, transcript_segments)
                        validate_team_action_evidence(parsed, transcript_segments)
                    elif isinstance(parsed, ChunkSummary):
                        validate_chunk_evidence(parsed, transcript_segments)
                return parsed
            except (json.JSONDecodeError, ValidationError, ValueError, AnalysisError) as exc:
                last_error = exc
                logger.warning(
                    "LLM 输出校验失败 stage=%s attempt=%s error_type=%s",
                    stage,
                    attempt + 1,
                    type(exc).__name__,
                )
            except Exception as exc:
                raise AnalysisError("LLM 请求失败: {}".format(exc)) from exc

        raise AnalysisError("LLM 输出在重试后仍无法通过结构或引文校验") from last_error

    async def _gather_bounded(self, coroutines: Sequence[Any]) -> List[Any]:
        """R-P1-7 ②：并发受显式上限约束，不无条件把所有分块一次性发出去。"""
        semaphore = asyncio.Semaphore(max(1, self.settings.llm_max_concurrency))

        async def run(coro: Any) -> Any:
            async with semaphore:
                return await coro

        return list(await asyncio.gather(*(run(coro) for coro in coroutines)))

    async def analyze(self, transcript: Transcript) -> SemanticAnalysis:
        if not transcript.segments:
            raise AnalysisError("没有可分析的转写内容")

        if len(transcript.text) <= self.settings.transcript_chunk_chars:
            prompt = "请生成会议纪要、个人表现分析和 3-5 条改进建议。\n\n带时间戳转写：\n" + format_segments(transcript.segments)
            result = await self._validated_call(
                SemanticAnalysis,
                SYSTEM_PROMPT,
                prompt,
                "final_direct",
                transcript.segments,
            )
            return SemanticAnalysis.model_validate(result.model_dump())

        chunks = split_segments(transcript.segments, max(1000, self.settings.transcript_chunk_chars - 1000))
        chunk_calls = [
            self._validated_call(
                ChunkSummary,
                CHUNK_SYSTEM_PROMPT,
                "这是第 {}/{} 块：\n{}".format(index, len(chunks), format_segments(chunk)),
                "chunk_{}".format(index),
                chunk,
            )
            for index, chunk in enumerate(chunks, start=1)
        ]
        chunk_results = await self._gather_bounded(chunk_calls)
        summaries: List[Dict[str, Any]] = [result.model_dump() for result in chunk_results]

        merge_prompt = (
            "以下是按顺序提取的分块信息。请去重并归并为最终报告。"
            "表现分析的引文只能从 evidence_quotes 中原样选取。\n\n"
            + json.dumps(summaries, ensure_ascii=False)
        )
        result = await self._validated_call(
            SemanticAnalysis,
            SYSTEM_PROMPT,
            merge_prompt,
            "final_merge",
            transcript.segments,
        )
        return SemanticAnalysis.model_validate(result.model_dump())

    @staticmethod
    def _project_context(projects: Optional[Sequence[Any]]) -> str:
        """把团队已有项目拼成一行提示（R-P1.5-8）；没有项目时不加任何内容。"""
        rows: List[str] = []
        for item in projects or []:
            identifier = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)
            name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else None)
            if identifier and name:
                rows.append("{}({})".format(name, identifier))
        if not rows:
            return ""
        return "\n当前团队已有项目：{}。项目建议只能从这些里选，不要编造。".format("、".join(rows))

    async def analyze_team(
        self, transcript: Transcript, usage_context: Optional[Dict[str, Any]] = None,
        projects: Optional[Sequence[Any]] = None,
    ) -> TeamMeetingReport:
        if not transcript.segments:
            raise AnalysisError("没有可分析的转写内容")
        project_context = self._project_context(projects)
        if len(transcript.text) <= self.settings.transcript_chunk_chars:
            result = await self._validated_call(
                TeamMeetingReport,
                TEAM_SYSTEM_PROMPT,
                "请生成团队会议报告。\n\n带时间戳转写：\n"
                + format_segments(transcript.segments) + project_context,
                "team_final_direct",
                transcript.segments,
                usage_context,
            )
            return TeamMeetingReport.model_validate(result.model_dump())

        chunks = split_segments(transcript.segments, max(1000, self.settings.transcript_chunk_chars - 1000))
        chunk_results = await self._gather_bounded([
            self._validated_call(
                TeamChunkSummary, TEAM_CHUNK_SYSTEM_PROMPT,
                "这是第 {}/{} 块：\n{}".format(index, len(chunks), format_segments(chunk)),
                "team_chunk_{}".format(index), chunk, usage_context,
            )
            for index, chunk in enumerate(chunks, start=1)
        ])
        summaries = [result.model_dump() for result in chunk_results]
        result = await self._validated_call(
            TeamMeetingReport,
            TEAM_SYSTEM_PROMPT,
            "以下分块按原顺序排列。去重归并；决策、遗留问题和紧急事项的引文只能从各分块 evidence 原样选取；"
            "紧急事项只保留真正带时间压力、需尽快处理的，宁少勿多。\n\n"
            + json.dumps(summaries, ensure_ascii=False) + project_context,
            "team_final_merge",
            transcript.segments,
            usage_context,
        )
        return TeamMeetingReport.model_validate(result.model_dump())

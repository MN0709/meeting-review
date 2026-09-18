"""PRD §9.2：3 个能力工具（readonly —— 内部调用 LLM，但不写业务库）。

设计约束（原则 P-2 能力下沉为工具，而非重写）：
- 复用现有 app/llm.py 的 analyze_team / 引文校验逻辑，**内部实现一行不改**；
- 引文校验把「异常」变成「结构化观察」：返回 {valid, invalid:[...]}，
  让模型知道「哪条错了、错在哪、附近原文是什么」，从而定向回读重抽。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from pydantic import ValidationError

from app.agent.tools.contract import ToolResult, failure, success
from app.agent.tools.registry import ToolRegistry, ToolHandler
from app.agent.tools.support import check_meeting, snippet
from app.db import Database
from app.llm import AnalysisError, LLMAnalyzer, split_segments
from app.models import TeamMeetingReport, Transcript, TranscriptSegment

# 与 app/llm.py _validate_quotes 完全一致的时间戳容差（PRD R-P1-4）。
TOLERANCE_SECONDS = 5
_OBJECT = {"type": "object", "additionalProperties": False}


def _schema(properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {**_OBJECT, "properties": properties, "required": required}


def _timestamp_to_seconds(timestamp: str) -> int:
    hours, minutes, seconds = (int(part) for part in timestamp.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def _quote_matches(quote: str, timestamp: str, segments: List[TranscriptSegment]) -> bool:
    """与 app/llm.py _validate_quotes 的判定完全一致：子串 + ±5 秒。"""
    seconds = _timestamp_to_seconds(timestamp)
    return any(
        quote in segment.text and segment.start - TOLERANCE_SECONDS <= seconds <= segment.end + TOLERANCE_SECONDS
        for segment in segments
    )


def _nearest_preview(timestamp: str, segments: List[TranscriptSegment]) -> str:
    try:
        seconds = _timestamp_to_seconds(timestamp)
    except (ValueError, AttributeError):
        return ""
    if not segments:
        return ""
    nearest = min(segments, key=lambda item: abs((item.start + item.end) / 2 - seconds))
    return snippet(nearest.text, 100)


def _select_segments(
    transcript: Transcript,
    segment_range: Optional[Dict[str, Any]],
    chunk_indexes: Optional[List[int]],
    analyzer: LLMAnalyzer,
) -> Optional[List[TranscriptSegment]]:
    segments = list(transcript.segments)
    if segment_range is not None:
        if not isinstance(segment_range, dict):
            return None
        try:
            start = float(segment_range.get("start", 0.0))
            end = float(segment_range.get("end", segments[-1].end))
        except (TypeError, ValueError):
            return None
        segments = [item for item in segments if item.end > start and item.start < end]
    if chunk_indexes is not None:
        if not isinstance(chunk_indexes, list) or not all(isinstance(i, int) for i in chunk_indexes):
            return None
        chunk_chars = max(1000, analyzer.settings.transcript_chunk_chars - 1000)
        chunks = split_segments(segments, chunk_chars)
        picked: List[TranscriptSegment] = []
        for index in chunk_indexes:
            if 0 <= index < len(chunks):
                picked.extend(chunks[index])
        segments = picked
    return segments or None


# ---------------------------------------------------------------------------
# 9. extract_report
# ---------------------------------------------------------------------------


async def _extract_report(
    db: Database, analyzer: LLMAnalyzer, team_id: int, meeting_id: str,
    segment_range: Optional[Dict[str, Any]] = None, chunk_indexes: Optional[List[int]] = None,
) -> ToolResult:
    err = check_meeting(db, meeting_id, team_id)
    if err:
        return err
    transcript = db.load_transcript(meeting_id, team_id)
    if transcript is None or not transcript.segments:
        return failure("not_found", "这场会议还没有可分析的转写。")
    segments = _select_segments(transcript, segment_range, chunk_indexes, analyzer)
    if not segments:
        return failure("invalid_args", "segment_range / chunk_indexes 没有选中任何转写片段。")
    sliced = Transcript(
        language=transcript.language,
        duration_seconds=max(transcript.duration_seconds, segments[-1].end),
        segments=segments,
    )
    try:
        report = await analyzer.analyze_team(sliced)
    except AnalysisError as exc:
        return failure("tool_error", "抽取报告失败：{}".format(str(exc)[:120]))
    except Exception as exc:  # 任何失败都降级为可读观察，不抛异常穿透
        return failure("tool_error", "抽取报告失败（{}）。".format(type(exc).__name__))
    coverage = {
        "segments_used": len(segments),
        "time_range": {"start": segments[0].start, "end": segments[-1].end},
    }
    return success(
        "已抽取报告：决策 {} 条、行动项 {} 条、遗留问题 {} 条。".format(
            len(report.decisions), len(report.action_items), len(report.unresolved_issues)
        ),
        {"report": report.model_dump(), "coverage": coverage},
    )


# ---------------------------------------------------------------------------
# 10. validate_evidence
# ---------------------------------------------------------------------------


def _validate_evidence(
    db: Database, team_id: int, meeting_id: str, report_payload: Any
) -> ToolResult:
    err = check_meeting(db, meeting_id, team_id)
    if err:
        return err
    try:
        report = TeamMeetingReport.model_validate(report_payload)
    except ValidationError as exc:
        return failure("invalid_args", "report_payload 不符合报告结构（{} 处错误）。".format(exc.error_count()))
    transcript = db.load_transcript(meeting_id, team_id)
    if transcript is None:
        return failure("not_found", "这场会议还没有可核对的转写。")
    segments = list(transcript.segments)
    invalid: List[Dict[str, Any]] = []
    evidence_items = [("decision", item.evidence) for item in report.decisions] + [
        ("unresolved_issue", item.evidence) for item in report.unresolved_issues
    ]
    for kind, evidence in evidence_items:
        if not _quote_matches(evidence.quote, evidence.timestamp, segments):
            invalid.append(
                {
                    "kind": kind,
                    "quote": evidence.quote,
                    "timestamp": evidence.timestamp,
                    "reason": "引文不是任何转写片段的原话子串，或时间戳超出所在片段 ±5 秒",
                    "nearest_segment_preview": _nearest_preview(evidence.timestamp, segments),
                }
            )
    payload = {"valid": not invalid, "invalid": invalid}
    if invalid:
        return success(
            "校验未通过：{} 条引文需要修正（每条已附最近原文）。".format(len(invalid)), payload
        )
    return success("校验通过：{} 条引文与时间戳均可核对。".format(len(evidence_items)), payload)


# ---------------------------------------------------------------------------
# 11. suggest_title
# ---------------------------------------------------------------------------


async def _suggest_title(
    db: Database, analyzer: LLMAnalyzer, team_id: int, meeting_id: str
) -> ToolResult:
    err = check_meeting(db, meeting_id, team_id)
    if err:
        return err
    transcript = db.load_transcript(meeting_id, team_id)
    if transcript is None or not transcript.segments:
        return failure("not_found", "这场会议还没有可分析的转写。")
    try:
        report = await analyzer.analyze_team(transcript)
    except AnalysisError as exc:
        return failure("tool_error", "生成标题失败：{}".format(str(exc)[:120]))
    except Exception as exc:
        return failure("tool_error", "生成标题失败（{}）。".format(type(exc).__name__))
    title = (report.suggested_title or "").strip()
    return success(
        "建议标题：{}".format(title or "（未生成）"),
        {"suggested_title": title, "reused": "app.llm.analyze_team"},
    )


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------


def _bind(db: Database, func: Callable[..., ToolResult]) -> ToolHandler:
    def handler(team_id: int, **kwargs: Any) -> ToolResult:
        return func(db, team_id, **kwargs)

    return handler


def _bind_async(db: Database, analyzer: LLMAnalyzer, func: Callable[..., Any]) -> ToolHandler:
    async def handler(team_id: int, **kwargs: Any) -> ToolResult:
        return await func(db, analyzer, team_id, **kwargs)

    return handler


def register_capability_tools(registry: ToolRegistry, db: Database, analyzer: LLMAnalyzer) -> None:
    registry.register_tool(
        name="extract_report",
        description="从转写（可限定片段范围）抽取团队报告；内部复用现有分析实现。",
        input_schema=_schema(
            {
                "meeting_id": {"type": "string"},
                "segment_range": _schema(
                    {"start": {"type": "number", "minimum": 0}, "end": {"type": "number", "minimum": 0}},
                    [],
                ),
                "chunk_indexes": {"type": "array", "items": {"type": "integer", "minimum": 0}},
            },
            ["meeting_id"],
        ),
        handler=_bind_async(db, analyzer, _extract_report),
    )
    registry.register_tool(
        name="validate_evidence",
        description="校验报告的决策/遗留问题引文是否为原话子串、时间戳是否在 ±5 秒内，并返回每条错误与最近原文。",
        input_schema=_schema(
            {"meeting_id": {"type": "string"}, "report_payload": {"type": "object"}},
            ["meeting_id", "report_payload"],
        ),
        handler=_bind(db, _validate_evidence),
    )
    registry.register_tool(
        name="suggest_title",
        description="为一场会议生成建议标题（复用现有标题生成逻辑）。",
        input_schema=_schema({"meeting_id": {"type": "string"}}, ["meeting_id"]),
        handler=_bind_async(db, analyzer, _suggest_title),
    )

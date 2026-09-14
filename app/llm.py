import asyncio
import copy
import json
import logging
import re
import threading
from typing import Any, Dict, List, Optional, Sequence, Type

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.models import ChunkSummary, SemanticAnalysis, TeamChunkSummary, TeamMeetingReport, Transcript, TranscriptSegment


logger = logging.getLogger(__name__)

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
先用不超过 300 字概括会议目标、进展和结果，再输出会议要点、决策清单、行动项与遗留问题。
每条决策和遗留问题必须包含输入中完全一致的原话和时间戳，不得改写引文；无法确认决策人时填“未明确”。
遗留问题只记录会上明确提出但尚未解决、尚未决定或需要后续确认的事项，没有则返回空数组。
行动项的负责人或截止时间不明确时填“未明确”。speaker_stats_note 必须固定为“说话人识别将于下一版本支持”。
只输出符合指定 JSON Schema 的 JSON，不输出 Markdown 或解释。
""".strip()

TEAM_CHUNK_SYSTEM_PROMPT = """
你负责从团队正式会议的一个转写分块中提取会议要点、明确决策、行动项和遗留问题。
每条决策和遗留问题必须保留完全一致的逐字原话及时间戳，不得改写引文；无法确认决策人、负责人或截止时间时填“未明确”。
遗留问题只记录明确提出但尚未解决、尚未决定或需要后续确认的事项，没有则返回空数组。
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
            "overview": "会议围绕内测上线安排展开，明确了发布时间和准备工作。",
            "meeting_points": ["会议要点"],
            "decisions": [{
                "content": "决策内容", "decision_maker": "未明确",
                "evidence": {"quote": "原话", "timestamp": "00:00:00"},
            }],
            "action_items": [{"task": "任务", "owner": "未明确", "deadline": "未明确"}],
            "unresolved_issues": [{
                "content": "尚未解决的问题",
                "evidence": {"quote": "原话", "timestamp": "00:00:00"},
            }],
            "speaker_stats_note": "说话人识别将于下一版本支持",
        }
        fields = (
            "overview(必填字符串，不超过 300 字)；meeting_points(必填字符串数组)；"
            "decisions(必填对象数组，每项只有 content/decision_maker/evidence，"
            "evidence 只有 quote/timestamp)；action_items(必填对象数组，每项只有 task/owner/deadline)；"
            "unresolved_issues(必填对象数组，每项只有 content/evidence，evidence 只有 quote/timestamp)；"
            "speaker_stats_note(必填固定字符串：说话人识别将于下一版本支持)"
        )
    elif issubclass(model_type, TeamChunkSummary):
        example = {
            "meeting_points": ["会议要点"],
            "decisions": [{"content": "决策内容", "decision_maker": "未明确", "evidence": {"quote": "原话", "timestamp": "00:00:00"}}],
            "action_items": [{"task": "任务", "owner": "未明确", "deadline": "未明确"}],
            "unresolved_issues": [{"content": "尚未解决的问题", "evidence": {"quote": "原话", "timestamp": "00:00:00"}}],
        }
        fields = (
            "meeting_points(必填字符串数组)；decisions(必填对象数组，每项只有 content/decision_maker/evidence，"
            "evidence 只有 quote/timestamp)；action_items(必填对象数组，每项只有 task/owner/deadline)；"
            "unresolved_issues(必填对象数组，每项只有 content/evidence，evidence 只有 quote/timestamp)"
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
            for key in ("meeting_points", "decisions", "action_items", "unresolved_issues")
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
                    cleaned.append({"task": str(task), "owner": str(item.get("owner") or "未明确"), "deadline": str(item.get("deadline") or "未明确")})
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

        if issubclass(model_type, TeamMeetingReport):
            overview = next(
                (payload.get(key) for key in ("overview", "meeting_overview", "summary") if payload.get(key)),
                None,
            )
            if overview is not None:
                normalized["overview"] = str(overview)
            normalized["speaker_stats_note"] = "说话人识别将于下一版本支持"
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
        "[{}-{}] {}".format(format_timestamp(item.start), format_timestamp(item.end), item.text)
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
        + [issue.evidence for issue in report.unresolved_issues],
        segments,
    )


def validate_team_chunk_evidence(report: TeamChunkSummary, segments: Sequence[TranscriptSegment]) -> None:
    _validate_quotes(
        [decision.evidence for decision in report.decisions]
        + [issue.evidence for issue in report.unresolved_issues],
        segments,
    )


class LLMAnalyzer:
    def __init__(self, settings: Settings, client: Optional[OpenAI] = None) -> None:
        self.settings = settings
        self._client = client
        self._client_lock = threading.Lock()

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

    def _request_sync(
        self,
        model_type: Type[BaseModel],
        system_prompt: str,
        user_prompt: str,
        stage: str,
        response_mode: str,
    ) -> str:
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

        response = self.client.chat.completions.create(
            model=self.settings.openai_model,
            temperature=0.1,
            response_format=response_format,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        usage = getattr(response, "usage", None)
        logger.info(
            "llm_usage stage=%s model=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            stage,
            self.settings.openai_model,
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
            getattr(usage, "total_tokens", None),
        )
        content = response.choices[0].message.content
        if not content:
            raise AnalysisError("LLM 返回了空内容")
        return content

    async def _validated_call(
        self,
        model_type: Type[BaseModel],
        system_prompt: str,
        user_prompt: str,
        stage: str,
        transcript_segments: Optional[Sequence[TranscriptSegment]] = None,
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
                    elif isinstance(parsed, TeamChunkSummary):
                        validate_team_chunk_evidence(parsed, transcript_segments)
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
        chunk_results = await asyncio.gather(*chunk_calls)
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

    async def analyze_team(self, transcript: Transcript) -> TeamMeetingReport:
        if not transcript.segments:
            raise AnalysisError("没有可分析的转写内容")
        if len(transcript.text) <= self.settings.transcript_chunk_chars:
            result = await self._validated_call(
                TeamMeetingReport,
                TEAM_SYSTEM_PROMPT,
                "请生成团队会议报告。\n\n带时间戳转写：\n" + format_segments(transcript.segments),
                "team_final_direct",
                transcript.segments,
            )
            return TeamMeetingReport.model_validate(result.model_dump())

        chunks = split_segments(transcript.segments, max(1000, self.settings.transcript_chunk_chars - 1000))
        chunk_results = await asyncio.gather(*[
            self._validated_call(
                TeamChunkSummary, TEAM_CHUNK_SYSTEM_PROMPT,
                "这是第 {}/{} 块：\n{}".format(index, len(chunks), format_segments(chunk)),
                "team_chunk_{}".format(index), chunk,
            )
            for index, chunk in enumerate(chunks, start=1)
        ])
        summaries = [result.model_dump() for result in chunk_results]
        result = await self._validated_call(
            TeamMeetingReport,
            TEAM_SYSTEM_PROMPT,
            "以下分块按原顺序排列。去重归并；决策和遗留问题的引文只能从各分块 evidence 原样选取。\n\n"
            + json.dumps(summaries, ensure_ascii=False),
            "team_final_merge",
            transcript.segments,
        )
        return TeamMeetingReport.model_validate(result.model_dump())

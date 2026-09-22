"""R-P2.1-4：确定性说话人归属（读取时计算，不落库、不调用模型）。

红线 21：不得编造说话人；定位不到一律返回 None（界面显示「未标注说话人」）。
定位策略（与前端「看上下文」一致，但搬到后端、确定性）：
1. 引文是片段子串 **且** 时间戳落在该片段 ±5 秒内；
2. 否则只按引文子串匹配（引文是很强的身份信号）；
3. 否则按时间戳落在片段内（最后手段）；
4. 都不命中 → None。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

TOLERANCE_SECONDS = 5

# 紧急回退开关：关闭后一律不归属（显示「未标注说话人」）。由 app.main 按配置设置。
ATTRIBUTION_ENABLED = True


def set_enabled(value: bool) -> None:
    global ATTRIBUTION_ENABLED
    ATTRIBUTION_ENABLED = bool(value)


def field_of(obj: Any, name: str, default: Any = None) -> Any:
    """兼容 Pydantic 对象与 dict（测试/旧数据）。"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def timestamp_to_seconds(timestamp: Any) -> Optional[int]:
    parts = str(timestamp or "").strip().split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = (int(part) for part in parts)
    except (TypeError, ValueError):
        return None
    return hours * 3600 + minutes * 60 + seconds


def _match_segment(evidence: Any, segments: Sequence[Any]) -> Optional[Any]:
    if evidence is None:
        return None
    quote = str(field_of(evidence, "quote", "") or "")
    seconds = timestamp_to_seconds(field_of(evidence, "timestamp", None))
    if quote:
        # 引文是强身份信号；引文对不上就**不猜**（红线 21），不再回退到“只看时间戳”。
        for segment in segments:
            if quote in str(field_of(segment, "text", "")) and seconds is not None:
                if float(segment.start) - TOLERANCE_SECONDS <= seconds <= float(segment.end) + TOLERANCE_SECONDS:
                    return segment
        for segment in segments:
            if quote in str(field_of(segment, "text", "")):
                return segment
        return None
    if seconds is not None:
        for segment in segments:
            if float(segment.start) <= seconds <= float(segment.end):
                return segment
    return None


def speaker_for_evidence(evidence: Any, segments: Sequence[Any]) -> Optional[str]:
    segment = _match_segment(evidence, segments)
    if segment is None:
        return None
    label = field_of(segment, "speaker_label", None)
    return str(label) if label else None


def display_speaker(label: Optional[str], label_to_name: Dict[str, str]) -> Optional[str]:
    """本地标签 → 本场展示名；没有映射时保留本地标签（如「说话人 1」）。"""
    if not label:
        return None
    return label_to_name.get(label, label)


def _attribute_items(items: Iterable[Any], segments: Sequence[Any], label_to_name: Dict[str, str], out: list) -> None:
    for item in items:
        label = speaker_for_evidence(getattr(item, "evidence", None), segments)
        out.append(item.model_copy(update={"speaker": display_speaker(label, label_to_name)}))


def attribute_report(report: Any, segments: Sequence[Any], label_to_name: Optional[Dict[str, str]] = None) -> Any:
    """给报告四类条目补 `speaker`（读取时计算）。定位不到即 None，不编造。"""
    if not ATTRIBUTION_ENABLED:
        return report
    mapping = label_to_name or {}
    decisions: list = []
    actions: list = []
    issues: list = []
    urgent: list = []
    _attribute_items(report.decisions, segments, mapping, decisions)
    _attribute_items(report.action_items, segments, mapping, actions)
    _attribute_items(report.unresolved_issues, segments, mapping, issues)
    _attribute_items(getattr(report, "urgent_items", []) or [], segments, mapping, urgent)
    return report.model_copy(update={
        "decisions": decisions,
        "action_items": actions,
        "unresolved_issues": issues,
        "urgent_items": urgent,
    })


def coverage(report: Any) -> tuple[int, int]:
    """返回（能定位到说话人的条目数, 三类条目总数），用于内部可观测。"""
    items = list(report.decisions) + list(report.action_items) + list(report.unresolved_issues)
    attributed = sum(1 for item in items if getattr(item, "speaker", None))
    return attributed, len(items)

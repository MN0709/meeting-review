"""R-P2.1-1/2/3：关键结论的确定性后处理（互斥、去重、负责人只认原话）。

纯函数、可单测、幂等。放在抽取归一化之后、写库之前（`finalize_report`），
并可在读取时只做去重（`dedupe_report`）以便旧报告不改库也能看到去重结果。

三类定义与互斥顺序：决策 > 行动项 > 待跟进（命中即停）。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.attribution import field_of, speaker_for_evidence

# 归一化：去掉空白、中英标点，再反复去掉引导词。
_PUNCT = re.compile(r"[\s\u3000,，.。;；:：!！?？、\"'“”‘’()（）\[\]【】<>《》\-—_/\\|~`@#$%^&*+=]+")
_LEADING = (
    "我们", "咱们", "大家", "接下来", "然后", "所以", "因此", "另外", "还有", "关于", "对于",
    "决定", "确定", "明确", "打算", "计划", "要", "需要", "应该", "应当", "必须", "就", "先", "得",
    "可以", "建议", "同意",
)
# 第一人称承诺信号（「我」后面不接「们」）。
_FIRST_PERSON = re.compile(r"我(?!们)")
_FIRST_PERSON_PHRASES = ("我来", "我负责", "我这边", "我会", "我去", "我做", "我准备", "我安排", "我处理", "我牵头", "我出")

_PRIORITY = {"decision": 0, "action": 1, "issue": 2}

# 紧急回退开关：关闭后读取时只做证据级互斥（不做内容级去重）。由 app.main 按配置设置。
CONTENT_DEDUPE_ENABLED = True
CONTENT_DEDUPE_JACCARD = 0.6
CONTENT_DEDUPE_CONTAINMENT = 0.6


def set_content_dedupe(
    enabled: bool, jaccard: float = 0.6, containment: float = 0.6,
) -> None:
    global CONTENT_DEDUPE_ENABLED, CONTENT_DEDUPE_JACCARD, CONTENT_DEDUPE_CONTAINMENT
    CONTENT_DEDUPE_ENABLED = bool(enabled)
    CONTENT_DEDUPE_JACCARD = float(jaccard)
    CONTENT_DEDUPE_CONTAINMENT = float(containment)


def set_content_dedupe_enabled(value: bool) -> None:
    global CONTENT_DEDUPE_ENABLED
    CONTENT_DEDUPE_ENABLED = bool(value)


def normalize_text(value: Any) -> str:
    text = _PUNCT.sub("", str(value or ""))
    changed = True
    while changed and text:
        changed = False
        for word in _LEADING:
            if text.startswith(word) and len(text) > len(word):
                text = text[len(word):]
                changed = True
    return text.lower()


def _bigrams(text: str) -> set:
    if len(text) < 2:
        return set(text)
    return {text[index:index + 2] for index in range(len(text) - 1)}


def jaccard(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    left_set, right_set = _bigrams(left), _bigrams(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def is_duplicate(left: str, right: str, threshold: float, containment_threshold: float = 0.6) -> bool:
    if not left or not right:
        return False
    if left == right or left in right or right in left:
        return True
    left_set, right_set = _bigrams(left), _bigrams(right)
    if left_set and right_set:
        containment = len(left_set & right_set) / min(len(left_set), len(right_set))
        if containment >= containment_threshold:
            return True
    return jaccard(left, right) >= threshold


def evidence_key(evidence: Any) -> Optional[Tuple[str, str]]:
    if evidence is None:
        return None
    quote = re.sub(r"\s+", "", str(field_of(evidence, "quote", "") or ""))
    timestamp = str(field_of(evidence, "timestamp", "") or "").strip()
    if not quote and not timestamp:
        return None
    return (quote, timestamp)


def _content(item: Any, kind: str) -> str:
    return str(getattr(item, "task" if kind == "action" else "content", "") or "")


def resolve_owner(action: Any, segments: Sequence[Any]) -> str:
    """R-P2.1-3：负责人只认原话支撑；模型猜的人名一律丢弃（红线 22）。"""
    owner = str(getattr(action, "owner", "") or "").strip()
    evidence = getattr(action, "evidence", None)
    if evidence is None or not getattr(evidence, "quote", ""):
        return "未明确" if owner in ("", "我") else owner or "未明确"
    quote = str(evidence.quote)
    speaker = speaker_for_evidence(evidence, segments)
    first_person = bool(_FIRST_PERSON.search(quote)) or any(phrase in quote for phrase in _FIRST_PERSON_PHRASES)
    if first_person:
        if owner in ("", "未明确", "我") or owner == speaker:
            return speaker or "未明确"
        if owner in quote:  # 原话里同时点名了别人
            return owner
        return speaker or "未明确"
    if owner and owner not in ("未明确", "我") and owner in quote:
        return owner
    return "未明确"


def _dedupe_by_evidence(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[Tuple[str, str], int] = {}
    for entry in entries:
        key = entry["key"]
        if key is None:
            continue
        existing = seen.get(key)
        if existing is None:
            seen[key] = _PRIORITY[entry["kind"]]
        elif _PRIORITY[entry["kind"]] < existing:
            seen[key] = _PRIORITY[entry["kind"]]
    kept: List[Dict[str, Any]] = []
    within_kind: set = set()
    for entry in entries:
        key = entry["key"]
        if key is None:
            kept.append(entry)
            continue
        if seen[key] != _PRIORITY[entry["kind"]]:
            continue  # 同一 evidence 已被更高优先级的一类占用
        if key in within_kind:
            continue  # 同一类里同一 evidence 只留一次
        within_kind.add(key)
        kept.append(entry)
    return kept


def finalize_report(
    report: Any,
    segments: Optional[Sequence[Any]] = None,
    *,
    dedupe_enabled: bool = True,
    jaccard_threshold: float = 0.6,
    containment_threshold: float = 0.6,
    normalize_owners: bool = True,
) -> Tuple[Any, Dict[str, int]]:
    """互斥 + 去重（+ 可选地按原话校正负责人）。返回 (新报告, 统计)。"""
    decisions = [item.model_copy(deep=True) for item in report.decisions]
    actions = [item.model_copy(deep=True) for item in report.action_items]
    issues = [item.model_copy(deep=True) for item in report.unresolved_issues]
    stats = {
        "decisions_in": len(decisions), "actions_in": len(actions), "issues_in": len(issues),
        "decisions_out": 0, "actions_out": 0, "issues_out": 0,
        "evidence_deduped": 0, "content_deduped": 0, "owners_normalized": 0,
    }

    # R-P2.1-3：负责人口径（只在写入时校正，读取时保留旧数据）
    if normalize_owners and segments:
        for action in actions:
            resolved = resolve_owner(action, segments)
            if resolved != action.owner:
                stats["owners_normalized"] += 1
            action.owner = resolved
    # R-P2.1-4①：新报告不再带「决策人」（写入时清空）；读取旧报告保留原值，界面不展示。
    if normalize_owners:
        for decision in decisions:
            if getattr(decision, "decision_maker", ""):
                decision.decision_maker = ""

    entries: List[Dict[str, Any]] = []
    for kind, items in (("decision", decisions), ("action", actions), ("issue", issues)):
        for item in items:
            entries.append({
                "kind": kind, "item": item,
                "key": evidence_key(getattr(item, "evidence", None)),
                "norm": normalize_text(_content(item, kind)),
            })
    before_evidence = len(entries)
    entries = _dedupe_by_evidence(entries)
    stats["evidence_deduped"] = before_evidence - len(entries)

    # R-P2.1-2：内容级去重（决策 > 行动项 > 待跟进；带明确负责人的行动项不参与——红线 26）
    entries.sort(key=lambda entry: _PRIORITY[entry["kind"]])
    kept: List[Dict[str, Any]] = []
    for entry in entries:
        duplicate = next(
            (item for item in kept if dedupe_enabled and is_duplicate(
                entry["norm"], item["norm"], jaccard_threshold, containment_threshold,
            )),
            None,
        )
        if duplicate is None:
            kept.append(entry)
            continue
        if entry["kind"] == "action":
            owner = str(getattr(entry["item"], "owner", "") or "").strip()
            if owner and owner != "未明确":
                kept.append(entry)  # 不因去重丢掉「我答应的任务」里的条目
                continue
        stats["content_deduped"] += 1

    new_decisions = [entry["item"] for entry in kept if entry["kind"] == "decision"]
    new_actions = [entry["item"] for entry in kept if entry["kind"] == "action"]
    new_issues = [entry["item"] for entry in kept if entry["kind"] == "issue"]
    stats["decisions_out"] = len(new_decisions)
    stats["actions_out"] = len(new_actions)
    stats["issues_out"] = len(new_issues)
    return report.model_copy(update={
        "decisions": new_decisions,
        "action_items": new_actions,
        "unresolved_issues": new_issues,
    }), stats


def dedupe_report(
    report: Any, *, jaccard_threshold: Optional[float] = None,
    containment_threshold: Optional[float] = None,
) -> Tuple[Any, Dict[str, int]]:
    """读取时只做互斥 + 去重（不动负责人，兼容旧报告）。幂等。"""
    return finalize_report(
        report, None, dedupe_enabled=CONTENT_DEDUPE_ENABLED,
        jaccard_threshold=CONTENT_DEDUPE_JACCARD if jaccard_threshold is None else jaccard_threshold,
        containment_threshold=(
            CONTENT_DEDUPE_CONTAINMENT if containment_threshold is None else containment_threshold
        ),
        normalize_owners=False,
    )


def three_class_note() -> str:
    return "决策＝已经定了的；行动项＝有人答应要做的；待跟进＝还没定、悬着的。"

"""R-P2-3：判定「本场哪个说话人是我」的纯函数（本人声纹，比成员匹配更严）。

不认错人（红线 15）——只有**同时**满足下面全部条件才认定为「我」：
1. 与本人声纹相似度 ≥ `threshold`（默认 0.80，高于成员阈值）；
2. 该说话人有效语音 ≥ `min_speech_seconds`（默认 10 秒）；
3. 本场**只有一个**说话人满足 1、2（≥ 2 人同时达标 → 判定不确定，不指定）；
4. 不「更像某个已知项目成员」：若成员匹配分 − 本人分 ≥ `member_margin`，判为成员，不算我。

否则一律返回**未识别**，绝不猜测。抽出来做纯函数，使处理链路与历史回填共用同一套规则。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from app.audio_math import cosine_score


@dataclass
class SpeakerCandidate:
    label: str
    embedding: Optional[list[float]]
    speech_seconds: float
    matched_member_id: Optional[int] = None
    matched_name: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class OwnerDecision:
    label: Optional[str]
    confidence: Optional[float]
    reason: str


def decide_owner(
    candidates: Sequence[SpeakerCandidate],
    owner_embedding: Optional[Sequence[float]],
    owner_name: str,
    *,
    threshold: float,
    min_speech_seconds: float,
    member_margin: float,
) -> OwnerDecision:
    if not owner_embedding:
        return OwnerDecision(None, None, "no_owner_voiceprint")
    passed: list[tuple[SpeakerCandidate, float]] = []
    for candidate in candidates:
        if not candidate.embedding:
            continue
        score = cosine_score(candidate.embedding, owner_embedding)
        if score >= threshold and candidate.speech_seconds >= min_speech_seconds:
            passed.append((candidate, score))
    if not passed:
        return OwnerDecision(None, None, "not_identified")
    if len(passed) > 1:
        # 多人同时达标：无法确定是谁说的，宁可不指定。
        return OwnerDecision(None, max(score for _, score in passed), "ambiguous")
    candidate, score = passed[0]
    if candidate.matched_member_id is not None and (candidate.confidence or 0.0) - score >= member_margin:
        return OwnerDecision(None, score, "member_priority")
    # 展示名优先取该项目成员姓名（同一个人在不同项目叫法不同）；否则用 OWNER_NAME。
    return OwnerDecision(candidate.matched_name or owner_name, score, "identified")

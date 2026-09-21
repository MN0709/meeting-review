"""阶段 14（M3）· R-P2-3「不认错」纯函数与 R-P2-12 回填的单元测试。

红线 15：误认必须为 0——不满足全部条件时一律「未识别」，绝不猜。
"""

from app.owner import SpeakerCandidate, decide_owner

OWNER = [1.0, 0.0]
PARAMS = {"threshold": 0.80, "min_speech_seconds": 10.0, "member_margin": 0.03}


def test_identifies_only_owner() -> None:
    candidate = SpeakerCandidate("说话人 1", [1.0, 0.0], 30.0)
    decision = decide_owner([candidate], OWNER, "我", **PARAMS)
    assert decision.label == "我"
    assert decision.reason == "identified"
    assert decision.confidence is not None and decision.confidence > 0.99


def test_never_guesses_when_below_threshold() -> None:
    candidate = SpeakerCandidate("说话人 1", [0.0, 1.0], 30.0)  # 相似度 0.5
    decision = decide_owner([candidate], OWNER, "我", **PARAMS)
    assert decision.label is None
    assert decision.reason == "not_identified"


def test_never_guesses_when_speech_too_short() -> None:
    candidate = SpeakerCandidate("说话人 1", [1.0, 0.0], 5.0)
    decision = decide_owner([candidate], OWNER, "我", **PARAMS)
    assert decision.label is None
    assert decision.reason == "not_identified"


def test_ambiguous_when_two_speakers_pass() -> None:
    """红线 15 负例：两人同时达标 → 不指定「我」。"""
    candidates = [
        SpeakerCandidate("说话人 1", [1.0, 0.0], 30.0),
        SpeakerCandidate("说话人 2", [0.99, 0.05], 30.0),
    ]
    decision = decide_owner(candidates, OWNER, "我", **PARAMS)
    assert decision.label is None
    assert decision.reason == "ambiguous"


def test_more_like_known_member_is_not_me() -> None:
    candidate = SpeakerCandidate(
        "说话人 1", [0.6, 0.8], 30.0,
        matched_member_id=5, matched_name="马宁", confidence=0.95,
    )
    decision = decide_owner([candidate], OWNER, "我", **PARAMS)
    assert decision.label is None
    assert decision.reason == "member_priority"


def test_uses_member_display_name_when_identified() -> None:
    candidate = SpeakerCandidate(
        "说话人 1", [1.0, 0.0], 30.0,
        matched_member_id=5, matched_name="马宁", confidence=0.85,
    )
    decision = decide_owner([candidate], OWNER, "我", **PARAMS)
    assert decision.label == "马宁"
    assert decision.reason == "identified"


def test_no_owner_voiceprint_never_identifies() -> None:
    candidate = SpeakerCandidate("说话人 1", [1.0, 0.0], 30.0)
    decision = decide_owner([candidate], None, "我", **PARAMS)
    assert decision.label is None
    assert decision.reason == "no_owner_voiceprint"

"""说话人同场合并的回归测试。

背景（2026-09-21 真实录音回归）：
真实录音「个人项目修改方向（胡泊老师指导）」里 WeSpeaker **正确分出了两个说话人**，
但两簇相似度为 **0.789**，高于当时的合并阈值 0.78，于是被合并成一个人，
导致「马宁 0.865」与「胡泊 0.900」两个身份一起丢失，最终只剩一个人名。

修复分两层：
1) 合并阈值 0.78 → 0.85（`SPEAKER_INTRA_MERGE_THRESHOLD`）；
2) **两簇各自命中不同的已知成员时，一律不合并**（身份优先于相似度）。

本文件用与实测一致的数字构造夹具，锁住这两层行为。
"""

import math

import pytest

from app.config import Settings
from app.speaker import KnownVoiceProfile, SpeakerRecognizer, best_profile_match, cosine_score

# 与真实录音实测一致的两个身份得分
MA_NING_SCORE = 0.865
HU_BO_SCORE = 0.900
CROSS_SCORE = 0.789          # 实测：两人的簇相似度
SAME_PERSON_SCORE = 0.92     # 同一个人的过度切分片段（典型值）


def _embedding(angle_degrees: float) -> list[float]:
    """构造确定性的二维向量；两向量的归一化相似度 = (cos 夹角 + 1) / 2。"""
    theta = math.radians(angle_degrees)
    return [math.cos(theta), math.sin(theta), 0.0, 0.0]


def _pair_with_score(target: float) -> tuple[list[float], list[float]]:
    """构造一对向量，使其归一化相似度恰好等于 target（用于对齐实测数字）。"""
    theta = math.degrees(math.acos(max(-1.0, min(1.0, 2.0 * target - 1.0))))
    return _embedding(0.0), _embedding(theta)


def _settings(**overrides) -> Settings:
    base = {"TEAM_TOKENS": "甲团队:mr-Abc123def456GHI789"}
    base.update(overrides)
    return Settings(_env_file=None, **base)


def test_default_merge_threshold_is_raised() -> None:
    """阈值必须高于实测的「不同人」相似度 0.789。"""
    settings = _settings()
    assert settings.speaker_intra_merge_threshold == 0.85
    assert settings.speaker_intra_merge_threshold > 0.789


def test_pair_of_different_people_is_not_merged_by_similarity_alone() -> None:
    """仅凭相似度也不该合并两个不同的已知身份。"""
    recognizer = SpeakerRecognizer(_settings())
    left, right = _pair_with_score(CROSS_SCORE)  # 实测的 0.789
    assert round(cosine_score(left, right), 3) == CROSS_SCORE
    assert cosine_score(left, right) < recognizer.intra_merge_threshold  # 阈值已能挡住这一对
    merged = recognizer._merge_similar_labels(
        {0: left, 1: right}, recognizer.intra_merge_threshold,
        identities={0: 5, 1: 4},
    )
    assert merged[0] != merged[1]


def test_identity_guard_blocks_merge_even_above_threshold() -> None:
    """核心回归：即使相似度高于阈值，只要两簇各自命中不同成员，就不合并。"""
    recognizer = SpeakerRecognizer(_settings())
    left, right = _pair_with_score(SAME_PERSON_SCORE)
    assert cosine_score(left, right) >= 0.9  # 故意构造「会被合并」的相似度
    merged = recognizer._merge_similar_labels(
        {0: left, 1: right}, recognizer.intra_merge_threshold,
        identities={0: 5, 1: 4},  # 马宁 / 胡泊
    )
    assert merged == {0: 0, 1: 1}, "两个不同身份被错误合并"


def test_same_identity_still_merges() -> None:
    """同一个人的过度切分片段（都命中同一个人）仍应合并，保住原有的过度切分修正。"""
    recognizer = SpeakerRecognizer(_settings())
    left, right = _pair_with_score(SAME_PERSON_SCORE)
    merged = recognizer._merge_similar_labels(
        {0: left, 1: right}, recognizer.intra_merge_threshold,
        identities={0: 5, 1: 5},
    )
    assert merged[0] == merged[1]


def test_unknown_fragments_still_merge() -> None:
    """两簇都未命中任何成员时，只要相似度够高就合并（原有的过度切分修正不变）。"""
    recognizer = SpeakerRecognizer(_settings())
    left, right = _pair_with_score(SAME_PERSON_SCORE)
    merged = recognizer._merge_similar_labels(
        {0: left, 1: right}, recognizer.intra_merge_threshold, identities={0: None, 1: None},
    )
    assert merged[0] == merged[1]


def test_missing_identity_map_keeps_old_behaviour() -> None:
    """不传 identities（老调用方式）时行为与修复前一致，避免误伤。"""
    recognizer = SpeakerRecognizer(_settings())
    left, right = _pair_with_score(SAME_PERSON_SCORE)
    merged = recognizer._merge_similar_labels({0: left, 1: right}, recognizer.intra_merge_threshold)
    assert merged[0] == merged[1]


def test_raw_identities_uses_same_threshold_and_margin() -> None:
    """`_raw_identities` 必须复用同一套「阈值 + 与第二候选的分差」规则。"""
    recognizer = SpeakerRecognizer(_settings())
    profiles = [
        KnownVoiceProfile(member_id=5, name="马宁", embedding=_embedding(0.0)),
        KnownVoiceProfile(member_id=4, name="胡泊", embedding=_embedding(90.0)),
    ]
    identities = recognizer._raw_identities(
        {0: list(profiles[0].embedding), 1: list(profiles[1].embedding), 2: None}, profiles,
    )
    assert identities == {0: 5, 1: 4, 2: None}


def test_best_profile_match_requires_margin() -> None:
    """与既有规则一致：第一候选与第二候选差不足时回退「待确认」。"""
    profiles = [
        KnownVoiceProfile(member_id=5, name="马宁", embedding=_embedding(0.0)),
        KnownVoiceProfile(member_id=4, name="胡泊", embedding=_embedding(8.0)),
    ]
    match, top = best_profile_match(_embedding(4.0), profiles, threshold=0.72, margin=0.05)
    assert match is None and top >= 0.72


@pytest.mark.parametrize("threshold", [0.85, 0.9, 0.95])
def test_threshold_is_configurable(threshold) -> None:
    recognizer = SpeakerRecognizer(_settings(SPEAKER_INTRA_MERGE_THRESHOLD=threshold))
    assert recognizer.intra_merge_threshold == threshold

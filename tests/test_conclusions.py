"""阶段 16（M1）· R-P2.1-1/2/3/4 纯逻辑测试：互斥、去重、负责人口径、说话人归属。"""

from app.attribution import attribute_report, coverage, set_enabled, speaker_for_evidence
from app.conclusions import (
    dedupe_report, evidence_key, finalize_report, is_duplicate, jaccard, normalize_text,
    resolve_owner,
)
from app.models import ActionItem, TeamMeetingReport, TranscriptSegment


def _report(decisions=(), actions=(), issues=()):
    return TeamMeetingReport(
        overview="概述", meeting_points=[],
        decisions=list(decisions), action_items=list(actions), unresolved_issues=list(issues),
    )


def _dec(content, quote, ts="00:00:01"):
    return {"content": content, "evidence": {"quote": quote, "timestamp": ts}}


def _act(task, owner="未明确", quote=None, ts="00:00:02"):
    payload = {"task": task, "owner": owner, "deadline": "未明确"}
    if quote:
        payload["evidence"] = {"quote": quote, "timestamp": ts}
    return payload


def _issue(content, quote, ts="00:00:03"):
    return {"content": content, "evidence": {"quote": quote, "timestamp": ts}}


# --- 归一化与相似度 ---------------------------------------------------------


def test_normalize_strips_punctuation_and_leading_words() -> None:
    assert normalize_text("我们决定：围绕会议总结做减法。") == "围绕会议总结做减法"
    assert normalize_text("要先做原型") == "做原型"


def test_is_duplicate_containment_and_negative_cases() -> None:
    assert is_duplicate(normalize_text("去掉通话等不相关模块，避免功能过重"),
                        normalize_text("去掉通话等不相关模块，围绕会议总结做减法"), 0.6, 0.6)
    assert not is_duplicate(normalize_text("准备发布清单"), normalize_text("同步给客户"), 0.6, 0.6)
    assert not is_duplicate(normalize_text("角色管理"), normalize_text("测试计划"), 0.6, 0.6)
    assert jaccard("abc", "abc") == 1.0


# --- R-P2.1-1 证据级互斥 -----------------------------------------------------


def test_same_evidence_kept_only_in_highest_priority_class() -> None:
    quote, ts = "我们本周上线内测", "00:00:02"
    report = _report(
        decisions=[_dec("本周上线内测", quote, ts)],
        actions=[_act("本周上线内测", quote=quote, ts=ts)],
        issues=[_issue("本周上线内测", quote, ts)],
    )
    finalized, stats = finalize_report(report, normalize_owners=False)
    assert [d.content for d in finalized.decisions] == ["本周上线内测"]
    assert finalized.action_items == []
    assert finalized.unresolved_issues == []
    assert stats["evidence_deduped"] == 2


# --- R-P2.1-2 内容级去重 -----------------------------------------------------


def test_content_dedupe_drops_lower_priority_copy() -> None:
    report = _report(
        decisions=[_dec("去掉通话等不相关模块，避免功能过重", "没必要")],
        issues=[_issue("去掉通话等不相关模块，避免功能过重", "没必要做")],
    )
    finalized, stats = finalize_report(report, normalize_owners=False)
    assert len(finalized.decisions) == 1
    assert finalized.unresolved_issues == []
    assert stats["content_deduped"] == 1


def test_owned_action_is_not_deduped_away() -> None:
    """红线 26：带明确负责人的行动项不因去重消失。"""
    report = _report(
        decisions=[_dec("本周上线内测", "我们本周上线内测")],
        actions=[_act("本周上线内测", owner="马宁", quote="我来负责本周上线", ts="00:00:05")],
    )
    finalized, _ = finalize_report(report, normalize_owners=False)
    assert len(finalized.decisions) == 1
    assert len(finalized.action_items) == 1
    assert finalized.action_items[0].owner == "马宁"


def test_unowned_action_is_deduped_away() -> None:
    report = _report(
        decisions=[_dec("本周上线内测", "我们本周上线内测")],
        actions=[_act("本周上线内测", owner="未明确")],
    )
    finalized, _ = finalize_report(report, normalize_owners=False)
    assert finalized.action_items == []


def test_dedupe_is_idempotent() -> None:
    report = _report(
        decisions=[_dec("去掉通话等不相关模块", "没必要")],
        issues=[_issue("去掉通话等不相关模块", "没必要做")],
    )
    once, _ = dedupe_report(report)
    twice, _ = dedupe_report(once)
    assert once.model_dump() == twice.model_dump()


def test_dedupe_disabled_keeps_evidence_level_only() -> None:
    report = _report(
        decisions=[_dec("去掉通话等不相关模块", "没必要")],
        issues=[_issue("去掉通话等不相关模块", "没必要做")],
    )
    finalized, _ = finalize_report(report, normalize_owners=False, dedupe_enabled=False)
    assert len(finalized.decisions) == 1 and len(finalized.unresolved_issues) == 1


# --- R-P2.1-3 负责人口径 -----------------------------------------------------


def test_owner_first_person_uses_evidence_speaker() -> None:
    segments = [TranscriptSegment(start=0, end=5, speaker_label="说话人 1", text="我来准备发布清单")]
    action = ActionItem(task="准备发布清单", owner="我", deadline="未明确",
                        evidence={"quote": "我来准备发布清单", "timestamp": "00:00:02"})
    assert resolve_owner(action, segments) == "说话人 1"


def test_owner_pointed_name_must_appear_in_quote() -> None:
    segments = [TranscriptSegment(start=0, end=5, speaker_label="说话人 1", text="小宁你准备清单")]
    pointed = ActionItem(task="准备清单", owner="小宁", deadline="未明确",
                         evidence={"quote": "小宁你准备清单", "timestamp": "00:00:02"})
    assert resolve_owner(pointed, segments) == "小宁"
    vague = ActionItem(task="准备清单", owner="小宁", deadline="未明确",
                       evidence={"quote": "你准备清单", "timestamp": "00:00:02"})
    assert resolve_owner(vague, segments) == "未明确"  # "你"不猜


def test_model_guessed_name_is_discarded() -> None:
    segments = [TranscriptSegment(start=0, end=5, speaker_label="说话人 2", text="我来跟进")]
    action = ActionItem(task="跟进", owner="张三", deadline="未明确",
                        evidence={"quote": "我来跟进", "timestamp": "00:00:02"})
    assert resolve_owner(action, segments) == "说话人 2"


def test_write_time_clears_decision_maker_read_time_keeps_it() -> None:
    report = _report(decisions=[_dec("本周上线", "我们本周上线")])
    written, _ = finalize_report(report, normalize_owners=True)
    assert written.decisions[0].decision_maker == ""
    read_back, _ = dedupe_report(report)
    assert read_back.decisions[0].decision_maker == ""  # 输入本就为空


# --- R-P2.1-4 说话人归属 -----------------------------------------------------


def test_evidence_key_normalizes_whitespace() -> None:
    assert evidence_key({"quote": " 我们 本周 ", "timestamp": "00:00:02"}) == ("我们本周", "00:00:02")


def test_speaker_for_evidence_matches_quote_and_timestamp() -> None:
    segments = [
        TranscriptSegment(start=0, end=5, speaker_label="说话人 1", text="我们本周上线内测"),
        TranscriptSegment(start=10, end=15, speaker_label="说话人 2", text="同步给客户"),
    ]
    assert speaker_for_evidence({"quote": "我们本周上线内测", "timestamp": "00:00:02"}, segments) == "说话人 1"
    # 时间戳落在片段内，靠引文定位
    assert speaker_for_evidence({"quote": "同步给客户", "timestamp": "00:00:12"}, segments) == "说话人 2"
    # 定位不到 → None（红线 21）
    assert speaker_for_evidence({"quote": "原话里没有这句", "timestamp": "00:09:59"}, segments) is None


def test_attribute_report_maps_local_label_to_member_name() -> None:
    segments = [
        TranscriptSegment(start=0, end=5, speaker_label="说话人 1", text="我们本周上线内测"),
        TranscriptSegment(start=5, end=10, speaker_label="说话人 1", text="我来准备清单"),
    ]
    report = _report(
        decisions=[_dec("本周上线内测", "我们本周上线内测")],
        actions=[_act("准备清单", owner="马宁", quote="我来准备清单", ts="00:00:02")],
        issues=[_issue("阈值未定", "阈值还没定", ts="00:00:03")],
    )
    attributed = attribute_report(report, segments, {"说话人 1": "马宁"})
    assert attributed.decisions[0].speaker == "马宁"
    assert attributed.action_items[0].speaker == "马宁"
    assert attributed.unresolved_issues[0].speaker is None  # 原话不在转写里 → 未标注
    assert coverage(attributed) == (2, 3)


def test_attribution_can_be_disabled() -> None:
    segments = [TranscriptSegment(start=0, end=5, speaker_label="说话人 1", text="我们本周上线内测")]
    report = _report(decisions=[_dec("本周上线内测", "我们本周上线内测")])
    set_enabled(False)
    try:
        assert attribute_report(report, segments, {"说话人 1": "马宁"}).decisions[0].speaker is None
    finally:
        set_enabled(True)

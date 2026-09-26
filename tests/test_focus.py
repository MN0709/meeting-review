"""阶段 18（聚焦重构）· R-P2.2「谁说了什么」按人分组的契约与回归测试。"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

import app.main as main_module
from app.db import Database
from app.models import TeamMeetingReport, Transcript, TranscriptSegment
from app.speaker import SpeakerObservation

AUTH_HEADERS = {"X-Access-Token": "test-access-token"}
OTHER_HEADERS = {"X-Access-Token": "other-team-token"}

ROOT = Path(__file__).parent.parent
HTML = (ROOT / "static" / "index.html").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    database = Database(tmp_path / "focus.db")
    database.initialize(main_module.settings.parsed_team_tokens())
    monkeypatch.setattr(main_module, "database", database)
    monkeypatch.setattr(main_module.speaker_recognizer, "enabled", False)
    return database


def _team(database: Database) -> int:
    team_id = database.authenticate("test-access-token")
    assert team_id is not None
    return team_id


def _seed(
    database: Database, meeting_id: str, *, project_id=None,
    key_name: str = "胡泊", other_name: str = "马宁",
) -> int:
    team_id = _team(database)
    database.create_meeting(meeting_id, team_id, "会议", Path("/tmp/m.wav"), project_id=project_id)
    database.save_transcript(meeting_id, team_id, Transcript(
        language="zh", duration_seconds=20, segments=[
            TranscriptSegment(start=0, end=5, speaker_label="说话人 1", text="我们本周上线内测"),
            TranscriptSegment(start=5, end=10, speaker_label="说话人 1", text="我来准备发布清单"),
            TranscriptSegment(start=10, end=15, speaker_label="说话人 2", text="阈值还没定"),
        ],
    ))
    database.save_meeting_speakers(meeting_id, team_id, [
        SpeakerObservation(local_label="说话人 1", embedding=[1.0, 0.0], speech_seconds=60.0),
        SpeakerObservation(local_label="说话人 2", embedding=[0.0, 1.0], speech_seconds=30.0),
    ])
    database.confirm_meeting_speaker(
        meeting_id, team_id, "说话人 1", key_name, "", True, True, "chinese",
    )
    database.confirm_meeting_speaker(
        meeting_id, team_id, "说话人 2", other_name, "", False, True, "chinese",
    )
    database.save_report(meeting_id, team_id, TeamMeetingReport(
        overview="会议确认本周上线。", meeting_points=["确定本周上线"],
        decisions=[{"content": "本周上线内测", "evidence": {"quote": "我们本周上线内测", "timestamp": "00:00:02"}}],
        action_items=[{"task": "准备发布清单", "owner": "胡泊", "deadline": "周五",
                       "evidence": {"quote": "我来准备发布清单", "timestamp": "00:00:06"}}],
        unresolved_issues=[{"content": "阈值未定", "evidence": {"quote": "阈值还没定", "timestamp": "00:00:12"}}],
    ))
    database.update_status(meeting_id, team_id, "完成")
    return team_id


def test_history_speaker_digest_groups_by_person_and_stars_key(isolated) -> None:
    database = isolated
    _seed(database, "m-focus-1")
    history = database.get_history("m-focus-1", _team(database))
    assert history is not None
    sections = history.speaker_digest.sections
    # ★关键决策人（胡泊）排最前
    assert [s.speaker for s in sections] == ["胡泊", "马宁"]
    assert sections[0].is_key_decision_maker is True
    assert sections[1].is_key_decision_maker is False
    # 行动项不进入该视图（已收起）
    assert {item.kind for item in sections[0].items} == {"decision"}
    assert sections[1].items[0].kind == "issue"


def test_speakers_expose_key_decision_maker_flag(isolated) -> None:
    database = isolated
    _seed(database, "m-focus-2")
    history = database.get_history("m-focus-2", _team(database))
    by_name = {s.display_name: s.is_key_decision_maker for s in history.speakers}
    assert by_name["胡泊"] is True
    assert by_name["马宁"] is False


def test_unlabeled_items_go_last(isolated) -> None:
    database = isolated
    team_id = _team(database)
    database.create_meeting("m-focus-3", team_id, "会议", Path("/tmp/m.wav"))
    database.save_report("m-focus-3", team_id, TeamMeetingReport(
        overview="x", meeting_points=[],
        decisions=[{"content": "没有原话归属的决策", "evidence": {"quote": "原文对不上", "timestamp": "00:09:00"}}],
        action_items=[], unresolved_issues=[],
    ))
    database.update_status("m-focus-3", team_id, "完成")
    history = database.get_history("m-focus-3", team_id)
    sections = history.speaker_digest.sections
    assert len(sections) == 1
    assert sections[0].speaker is None  # 定位不到 → 未标注，不编造
    assert sections[0].is_key_decision_maker is False


def test_project_by_speaker_aggregates_across_meetings(isolated) -> None:
    database = isolated
    team_id = _team(database)
    project = database.create_project("p-focus", team_id, "项目")
    _seed(database, "m-a", project_id=project.id)
    _seed(database, "m-b", project_id=project.id)
    digest = database.get_project_speaker_digest(project.id, team_id)
    stats = {s.speaker: len(s.items) for s in digest.sections}
    # 两场会议里同一个人的话被聚合到一起（行动项不进入）
    assert stats["胡泊"] == 2 and stats["马宁"] == 2
    assert digest.sections[0].speaker == "胡泊" and digest.sections[0].is_key_decision_maker
    # 条目带回来源会议，便于跳转
    assert digest.sections[0].items[0].meeting_id in {"m-a", "m-b"}
    assert digest.sections[0].items[0].meeting_title


def test_project_by_speaker_isolated_between_projects(isolated) -> None:
    database = isolated
    team_id = _team(database)
    p1 = database.create_project("p1", team_id, "项目一")
    p2 = database.create_project("p2", team_id, "项目二")
    _seed(database, "m-p1", project_id=p1.id)
    _seed(database, "m-p2", project_id=p2.id)
    d1 = database.get_project_speaker_digest(p1.id, team_id)
    d2 = database.get_project_speaker_digest(p2.id, team_id)
    def total(digest):
        return sum(len(section.items) for section in digest.sections)

    assert total(d1) == 2 and total(d2) == 2  # 不跨项目合并


def test_by_speaker_api_and_authz(isolated) -> None:
    database = isolated
    team_id = _team(database)
    project = database.create_project("p-api", team_id, "项目")
    _seed(database, "m-api", project_id=project.id)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.get("/api/projects/{}/by-speaker".format(project.id)).json()
        assert [s["speaker"] for s in body["sections"]] == ["胡泊", "马宁"]
        # 会议详情也带 speaker_digest
        meeting = client.get("/api/meetings/m-api").json()
        assert meeting["speaker_digest"]["sections"][0]["speaker"] == "胡泊"
        assert meeting["speakers"][0]["is_key_decision_maker"] is True
        # 越权
        denied = client.get(
            "/api/projects/{}/by-speaker".format(project.id), headers=OTHER_HEADERS,
        )
        assert denied.status_code == 403
        # 不存在
        missing = client.get("/api/projects/none/by-speaker")
        assert missing.status_code == 404


# --- 界面聚焦（阶段 18） -----------------------------------------------------


def test_report_focused_on_who_said_what() -> None:
    """报告页只留「谁说了什么」，杂项从主界面收起（hidden，能力保留）。"""
    assert 'id="digestSection"' in HTML and "谁说了什么" in HTML
    assert 'class="secondary hidden" id="shareMeeting"' in HTML
    assert '<section class="hidden"><h3>① 会议总览</h3>' in HTML
    assert '<section class="hidden"><h3>② 会议要点</h3>' in HTML
    assert '<section id="myTasksSection" class="hidden">' in HTML
    assert '<section class="hidden"><div id="followupBanner" class="hidden"></div><h3>③ 关键结论</h3>' in HTML
    assert '<section id="deliverablesSection" class="hidden">' in HTML
    assert '<section id="imageMinutesSection" class="hidden">' in HTML


def test_project_focused_on_cross_meeting_by_speaker() -> None:
    """项目页新增跨会议「谁说了什么」。"""
    assert 'id="projectDigestPanel"' in HTML
    assert "这个项目里，谁说了什么" in HTML
    assert "/by-speaker" in HTML

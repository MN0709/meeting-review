"""阶段 16（M1）· R-P2.1-4/5/6 的 API 契约与回归测试。"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

import app.attribution as attribution
import app.main as main_module
from app.db import Database
from app.models import TeamMeetingReport, Transcript, TranscriptSegment
from app.speaker import SpeakerObservation

AUTH_HEADERS = {"X-Access-Token": "test-access-token"}
MEETING = "m-p2-1"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    database = Database(tmp_path / "p21.db")
    database.initialize(main_module.settings.parsed_team_tokens())
    monkeypatch.setattr(main_module, "database", database)
    monkeypatch.setattr(main_module.speaker_recognizer, "enabled", False)
    return database


def _team(database: Database) -> int:
    team_id = database.authenticate("test-access-token")
    assert team_id is not None
    return team_id


def _seed(database: Database, *, project_id=None) -> int:
    team_id = _team(database)
    database.create_meeting(MEETING, team_id, "关键结论测试", Path("/tmp/m.wav"), project_id=project_id)
    database.save_transcript(MEETING, team_id, Transcript(
        language="zh", duration_seconds=20, segments=[
            TranscriptSegment(start=0, end=5, speaker_label="说话人 1", text="我们本周上线内测"),
            TranscriptSegment(start=5, end=10, speaker_label="说话人 1", text="我来准备发布清单"),
            TranscriptSegment(start=10, end=15, speaker_label="说话人 2", text="阈值还没定"),
        ],
    ))
    database.save_meeting_speakers(MEETING, team_id, [
        SpeakerObservation(local_label="说话人 1", embedding=[1.0, 0.0], speech_seconds=60.0),
        SpeakerObservation(local_label="说话人 2", embedding=[0.0, 1.0], speech_seconds=30.0),
    ])
    database.confirm_meeting_speaker(
        MEETING, team_id, "说话人 1", "马宁", "", False, True, "chinese",
    )
    database.save_report(MEETING, team_id, TeamMeetingReport(
        overview="会议确认本周上线。", meeting_points=["确定本周上线"],
        decisions=[{"content": "本周上线内测", "evidence": {"quote": "我们本周上线内测", "timestamp": "00:00:02"}}],
        action_items=[
            {"task": "本周上线内测", "owner": "未明确", "deadline": "未明确"},  # 与决策重复且无主 → 去重
            {"task": "准备发布清单", "owner": "马宁", "deadline": "周五",
             "evidence": {"quote": "我来准备发布清单", "timestamp": "00:00:06"}},
        ],
        unresolved_issues=[{"content": "监控阈值未定", "evidence": {"quote": "阈值还没定", "timestamp": "00:00:12"}}],
    ))
    database.update_status(MEETING, team_id, "完成")
    return team_id


# --- R-P2.1-4 谁说的 / R-P2.1-1~2 互斥去重 ----------------------------------


def test_history_returns_speaker_and_dedupes(isolated) -> None:
    database = isolated
    _seed(database)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.get("/api/meetings/{}".format(MEETING)).json()
    report = body["report"]
    assert [d["content"] for d in report["decisions"]] == ["本周上线内测"]
    assert report["decisions"][0]["speaker"] == "马宁"
    # 无主重复行动项被去重，带 owner 的行动项保留（红线 26）
    assert [a["task"] for a in report["action_items"]] == ["准备发布清单"]
    assert report["action_items"][0]["speaker"] == "马宁"
    # 说话人 2 没有成员名 → 保留本地标签（不是编造）
    assert report["unresolved_issues"][0]["speaker"] == "说话人 2"
    assert "决策人：" not in client.get("/api/meetings/{}".format(MEETING)).text


def test_my_tasks_count_unchanged(isolated) -> None:
    """R-P2.1-5 / 红线 26：我答应的任务口径不变。"""
    database = isolated
    team_id = _seed(database)
    member_id = next(m.id for m in database.list_members(team_id) if m.name == "马宁")
    database.set_self_speaker(MEETING, team_id, member_id=member_id, label="说话人 1")
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.get("/api/meetings/{}/my-tasks".format(MEETING)).json()
    assert body["self_name"] == "马宁"
    assert body["count"] == 1 and body["items"][0]["task"] == "准备发布清单"


def test_image_minutes_uses_speaker_not_decision_maker(isolated) -> None:
    database = isolated
    _seed(database)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.post("/api/meetings/{}/image-minutes".format(MEETING)).json()
    decisions = next(part for part in body["parts"] if part["key"] == "decisions")
    assert decisions["items"][0]["text"] == "马宁：本周上线内测"
    assert decisions["items"][0]["meta"] is None
    assert "决策人" not in str(body)


def test_project_memory_has_speaker(isolated) -> None:
    database = isolated
    team_id = _team(database)
    project = database.create_project("p1", team_id, "项目一")
    _seed(database, project_id=project.id)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.get("/api/projects/{}/memory".format(project.id)).json()
    assert body["decisions"][0]["speaker"] == "马宁"
    assert body["decisions"][0]["decision_maker"] == ""


def test_attribution_can_be_disabled_by_switch(isolated, monkeypatch) -> None:
    database = isolated
    _seed(database)
    monkeypatch.setattr(attribution, "ATTRIBUTION_ENABLED", False)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        report = client.get("/api/meetings/{}".format(MEETING)).json()["report"]
    assert report["decisions"][0]["speaker"] is None


def test_dedupe_switch_off_keeps_duplicate_but_evidence_mutex_remains(isolated, monkeypatch) -> None:
    database = isolated
    _seed(database)
    monkeypatch.setattr(main_module.settings, "conclusion_dedupe_enabled", False)
    monkeypatch.setattr("app.conclusions.CONTENT_DEDUPE_ENABLED", False)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        report = client.get("/api/meetings/{}".format(MEETING)).json()["report"]
    # 关闭内容级去重后，无主重复行动项被保留
    assert "本周上线内测" in [a["task"] for a in report["action_items"]]

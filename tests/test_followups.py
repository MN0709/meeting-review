"""阶段 17（M2）· R-P2.1-7/8 待跟进跨会议跟踪的契约与回归测试。"""

import os
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

import app.main as main_module
from app.db import Database
from app.followups import fingerprint
from app.models import TeamMeetingReport

AUTH_HEADERS = {"X-Access-Token": "test-access-token"}
OTHER_HEADERS = {"X-Access-Token": "other-team-token"}

ROOT = Path(__file__).parent.parent
HTML = (ROOT / "static" / "index.html").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    database = Database(tmp_path / "followups.db")
    database.initialize(main_module.settings.parsed_team_tokens())
    monkeypatch.setattr(main_module, "database", database)
    monkeypatch.setattr(main_module.speaker_recognizer, "enabled", False)
    return database


def _team(database: Database) -> int:
    team_id = database.authenticate("test-access-token")
    assert team_id is not None
    return team_id


def _report(*issues) -> TeamMeetingReport:
    return TeamMeetingReport(
        overview="会议总览",
        meeting_points=[],
        decisions=[],
        action_items=[],
        unresolved_issues=[
            {"content": text, "evidence": {"quote": quote, "timestamp": ts}}
            for text, quote, ts in issues
        ],
    )


def _seed_followup(database: Database, project_id, meeting_id, issue=("监控阈值未定", "阈值还没定", "00:00:02")) -> TeamMeetingReport:
    team_id = _team(database)
    database.create_meeting(meeting_id, team_id, "会议", Path("/tmp/m.wav"), project_id=project_id)
    report = _report(issue)
    database.save_report(meeting_id, team_id, report)
    database.update_status(meeting_id, team_id, "完成")
    database.upsert_followups(meeting_id, team_id, report)
    return report


def _followup_count(database: Database) -> int:
    with sqlite3.connect(database.path) as connection:
        return connection.execute("SELECT COUNT(*) FROM followups").fetchone()[0]


# --- 指纹（纯函数） ---------------------------------------------------------


def test_fingerprint_normalizes_whitespace_and_leading_words() -> None:
    assert fingerprint("阈值还没定") == fingerprint("  阈值还没定  ")
    # 引导词「我们/要」被剥离后等价
    assert fingerprint("我们要定排期") == fingerprint("要定排期")
    # 纯标点回退到 sha256 前缀，避免互相误合并
    assert fingerprint("！！！").startswith("sha256:")
    assert fingerprint("") .startswith("sha256:")


# --- upsert 与去重 -----------------------------------------------------------


def test_upsert_merges_same_fingerprint_within_project(isolated) -> None:
    database = isolated
    team_id = _team(database)
    project = database.create_project("p1", team_id, "项目一")
    _seed_followup(database, project.id, "m-follow-1")
    _seed_followup(database, project.id, "m-follow-2")
    followups = database.list_followups(project.id, team_id)
    assert followups.total == 1
    item = followups.items[0]
    assert item.status == "open"
    assert item.meeting_count == 2
    assert item.first_meeting_id == "m-follow-1"
    assert item.last_seen_meeting_id == "m-follow-2"


def test_cross_project_isolated(isolated) -> None:
    """红线：跨项目绝不合并。"""
    database = isolated
    team_id = _team(database)
    p1 = database.create_project("p1", team_id, "项目一")
    p2 = database.create_project("p2", team_id, "项目二")
    _seed_followup(database, p1.id, "m-cross-1")
    _seed_followup(database, p2.id, "m-cross-2")
    assert database.list_followups(p1.id, team_id).total == 1
    assert database.list_followups(p2.id, team_id).total == 1


def test_unclassified_meeting_not_tracked(isolated) -> None:
    database = isolated
    team_id = _team(database)
    database.create_meeting("m-unclassified", team_id, "会议", Path("/tmp/m.wav"), project_id=None)
    report = _report(("阈值未定", "阈值还没定", "00:00:02"))
    database.save_report("m-unclassified", team_id, report)
    database.update_status("m-unclassified", team_id, "完成")
    database.upsert_followups("m-unclassified", team_id, report)
    assert _followup_count(database) == 0


def test_resolved_not_reopened_on_reappearance(isolated) -> None:
    """红线 23：状态只能人改；upsert 不得把 resolved/dropped 悄悄改回 open。"""
    database = isolated
    team_id = _team(database)
    project = database.create_project("p1", team_id, "项目一")
    _seed_followup(database, project.id, "m-r1")
    followups = database.list_followups(project.id, team_id)
    followup_id = followups.items[0].id
    assert database.update_followup_status(followup_id, team_id, "resolved") is True
    # 再次出现：不重复建条目、不改回 open
    _seed_followup(database, project.id, "m-r2")
    followups = database.list_followups(project.id, team_id)
    assert followups.total == 1
    assert followups.items[0].status == "resolved"
    assert followups.items[0].meeting_count == 2


def test_status_filter(isolated) -> None:
    database = isolated
    team_id = _team(database)
    project = database.create_project("p1", team_id, "项目一")
    _seed_followup(database, project.id, "m-f1", issue=("问题甲", "甲还没定", "00:00:01"))
    _seed_followup(database, project.id, "m-f2", issue=("问题乙", "乙还没定", "00:00:02"))
    all_items = database.list_followups(project.id, team_id)
    assert all_items.total == 2 and all_items.open == 2
    database.update_followup_status(all_items.items[0].id, team_id, "dropped")
    assert database.list_followups(project.id, team_id, status="open").total == 1
    assert database.list_followups(project.id, team_id, status="dropped").total == 1
    assert database.list_followups(project.id, team_id, status="resolved").total == 0


# --- 删除级联 ----------------------------------------------------------------


def test_delete_meeting_detaches_but_keeps_followup(isolated) -> None:
    database = isolated
    team_id = _team(database)
    project = database.create_project("p1", team_id, "项目一")
    _seed_followup(database, project.id, "m-del")
    followups = database.list_followups(project.id, team_id)
    assert followups.total == 1
    database.delete_meeting("m-del", team_id)
    followups = database.list_followups(project.id, team_id)
    assert followups.total == 1  # 待跟进本身保留
    item = followups.items[0]
    assert item.first_meeting_id is None
    assert item.meeting_count == 0


def test_delete_project_removes_followups(isolated) -> None:
    database = isolated
    team_id = _team(database)
    project = database.create_project("p1", team_id, "项目一")
    _seed_followup(database, project.id, "m-delp")
    assert _followup_count(database) == 1
    database.delete_project(project.id, team_id, delete_meetings=False)
    assert _followup_count(database) == 0
    assert database.list_followups(project.id, team_id) is None


# --- API 契约 ----------------------------------------------------------------


def test_followups_api_and_authz(isolated) -> None:
    database = isolated
    team_id = _team(database)
    project = database.create_project("p1", team_id, "项目一")
    _seed_followup(database, project.id, "m-api")
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.get("/api/projects/{}/followups".format(project.id)).json()
        assert body["total"] == 1 and body["open"] == 1
        followup_id = body["items"][0]["id"]
        # 人工标记已解决
        resp = client.patch(
            "/api/followups/{}".format(followup_id), json={"status": "resolved"},
        )
        assert resp.status_code == 200 and resp.json()["status"] == "resolved"
        # 默认列表不再出现（过滤 open 为空）
        open_list = client.get(
            "/api/projects/{}/followups?status=open".format(project.id),
        ).json()
        assert open_list["total"] == 0
        # 非法 status
        bad = client.get(
            "/api/projects/{}/followups?status=xxx".format(project.id),
        )
        assert bad.status_code == 422
        # 越权：另一团队
        denied = client.patch(
            "/api/followups/{}".format(followup_id), json={"status": "open"},
            headers=OTHER_HEADERS,
        )
        assert denied.status_code == 403
        # 不存在的待跟进
        missing = client.patch("/api/followups/999999", json={"status": "open"})
        assert missing.status_code == 404


# --- 界面：空则不显示（M2 调整） ---------------------------------------------


def test_empty_followups_hidden_in_ui() -> None:
    """报告页/项目页「待跟进」为空时不显示，且不再显示空态文案。"""
    # 报告页：待跟进栏空则隐藏，去掉「本次会议未识别到明确遗留问题」空话
    assert "本次会议未识别到明确遗留问题" not in HTML
    assert "unresolvedHeading" in HTML
    assert "issuesHeading.classList.toggle('hidden',!unresolved.length)" in HTML
    # 项目页：待跟进板块默认隐藏，无内容时整体不出现
    assert 'class="followup-panel hidden" id="projectFollowups"' in HTML
    assert "panel.classList.add('hidden');return}" in HTML

"""阶段 15（M4）· R-P2-6 高置信自动归类与撤销的测试。

红线 14：不得自动新建项目；红线 16：自动归类必须可撤销且有留痕。
"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

import app.main as main_module
from app.db import Database
from app.models import SuggestedProject, TeamMeetingReport

AUTH_HEADERS = {"X-Access-Token": "test-access-token"}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    database = Database(tmp_path / "assign.db")
    database.initialize(main_module.settings.parsed_team_tokens())
    monkeypatch.setattr(main_module, "database", database)
    monkeypatch.setattr(main_module.speaker_recognizer, "enabled", False)
    return database


def _team(database: Database) -> int:
    team_id = database.authenticate("test-access-token")
    assert team_id is not None
    return team_id


def _report(existing=None, new_name=None, confidence=None) -> TeamMeetingReport:
    suggestion = None
    if existing or new_name:
        suggestion = SuggestedProject(
            existing_project_id=existing, new_project_name=new_name,
            confidence=confidence if confidence is not None else 0.9, reason="测试",
        )
    return TeamMeetingReport(
        overview="概述", meeting_points=[], decisions=[], action_items=[],
        unresolved_issues=[], suggested_project=suggestion,
    )


def _seed(database: Database, meeting_id: str, report: TeamMeetingReport, project_id=None) -> int:
    team_id = _team(database)
    database.create_meeting(meeting_id, team_id, "自动归类测试", Path("/tmp/a.wav"), project_id=project_id)
    database.save_report(meeting_id, team_id, report)
    return team_id


def _meeting(database: Database, team_id: int, meeting_id: str):
    return next(item for item in database.list_meetings(team_id) if item.id == meeting_id)


# --- 数据层：自动归类规则 ---------------------------------------------------


def test_high_confidence_existing_project_is_assigned(isolated) -> None:
    database = isolated
    team_id = _team(database)
    project = database.create_project("p-target", team_id, "目标项目")
    _seed(database, "m-auto", _report(existing=project.id, confidence=0.9))

    main_module._auto_assign_from_suggestion(
        "m-auto", team_id, _report(existing=project.id, confidence=0.9)
    )
    meeting = _meeting(database, team_id, "m-auto")
    assert meeting.project_id == project.id
    assert meeting.assignment_source == "auto"
    assert meeting.assignment_confidence == 0.9
    logs = database.list_assignments("m-auto", team_id)
    assert len(logs) == 1 and logs[0]["source"] == "auto" and logs[0]["undone_at"] is None


def test_low_confidence_stays_unclassified(isolated, monkeypatch) -> None:
    monkeypatch.setattr(main_module.settings, "auto_assign_threshold", 0.75)
    database = isolated
    team_id = _team(database)
    project = database.create_project("p-target", team_id, "目标项目")
    report = _report(existing=project.id, confidence=0.5)
    _seed(database, "m-low", report)

    main_module._auto_assign_from_suggestion("m-low", team_id, report)
    meeting = _meeting(database, team_id, "m-low")
    assert meeting.project_id is None and meeting.assignment_source is None
    assert database.list_assignments("m-low", team_id) == []


def test_new_project_name_is_never_auto_created(isolated) -> None:
    database = isolated
    team_id = _team(database)
    report = _report(new_name="全新项目", confidence=0.99)
    _seed(database, "m-new", report)

    main_module._auto_assign_from_suggestion("m-new", team_id, report)
    assert len(database.list_projects(team_id)) == 0  # 红线 14：不自动新建
    assert _meeting(database, team_id, "m-new").project_id is None


def test_unknown_project_id_is_not_assigned(isolated) -> None:
    database = isolated
    team_id = _team(database)
    report = _report(existing="not-exist", confidence=0.99)
    _seed(database, "m-ghost", report)

    main_module._auto_assign_from_suggestion("m-ghost", team_id, report)
    assert _meeting(database, team_id, "m-ghost").project_id is None


def test_switch_off_returns_to_suggest_only(isolated, monkeypatch) -> None:
    monkeypatch.setattr(main_module.settings, "auto_project_assign_enabled", False)
    database = isolated
    team_id = _team(database)
    project = database.create_project("p-target", team_id, "目标项目")
    report = _report(existing=project.id, confidence=0.99)
    _seed(database, "m-off", report)

    main_module._auto_assign_from_suggestion("m-off", team_id, report)
    assert _meeting(database, team_id, "m-off").project_id is None


def test_undo_restores_unclassified_and_keeps_audit(isolated) -> None:
    database = isolated
    team_id = _team(database)
    project = database.create_project("p-target", team_id, "目标项目")
    report = _report(existing=project.id, confidence=0.9)
    _seed(database, "m-undo", report)
    main_module._auto_assign_from_suggestion("m-undo", team_id, report)
    assert _meeting(database, team_id, "m-undo").project_id == project.id

    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post("/api/meetings/m-undo/undo-assignment")
    assert response.status_code == 200
    assert response.json()["project_id"] is None
    meeting = _meeting(database, team_id, "m-undo")
    assert meeting.project_id is None and meeting.assignment_source is None
    logs = database.list_assignments("m-undo", team_id)
    assert logs[0]["undone_at"] is not None  # 留痕（红线 16）


def test_undo_without_auto_assignment_returns_404(isolated) -> None:
    database = isolated
    _seed(database, "m-none", _report())
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        assert client.post("/api/meetings/m-none/undo-assignment").status_code == 404


# --- API：批量确认归类 -------------------------------------------------------


def test_assign_batch_to_project(isolated) -> None:
    database = isolated
    team_id = _team(database)
    project = database.create_project("p-target", team_id, "目标项目")
    for meeting_id in ("m1", "m2"):
        _seed(database, meeting_id, _report())
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post(
            "/api/meetings/assign-batch",
            json={"meeting_ids": ["m1", "m2"], "project_id": project.id},
        )
    assert response.status_code == 200
    assert response.json()["assigned"] == 2
    assert all(_meeting(database, team_id, mid).project_id == project.id for mid in ("m1", "m2"))
    # 人工归类同样留痕，且不是 auto
    logs = database.list_assignments("m1", team_id)
    assert logs[0]["source"] == "manual"


def test_assign_batch_accept_suggestions_only_existing(isolated) -> None:
    database = isolated
    team_id = _team(database)
    project = database.create_project("p-target", team_id, "目标项目")
    _seed(database, "m-with", _report(existing=project.id, confidence=0.9))
    _seed(database, "m-without", _report(new_name="新项目", confidence=0.9))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.post(
            "/api/meetings/assign-batch",
            json={"meeting_ids": ["m-with", "m-without"], "action": "accept_suggestions"},
        ).json()
    assert body["assigned"] == 1 and body["skipped"] == 1
    assert _meeting(database, team_id, "m-with").project_id == project.id
    assert _meeting(database, team_id, "m-without").project_id is None
    assert len(database.list_projects(team_id)) == 1  # 没有自动新建


def test_assign_batch_requires_target(isolated) -> None:
    database = isolated
    _seed(database, "m1", _report())
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        assert client.post(
            "/api/meetings/assign-batch", json={"meeting_ids": ["m1"]}
        ).status_code == 422

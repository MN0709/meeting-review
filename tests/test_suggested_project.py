"""阶段 10-C（M3）· R-P1.5-8 AI 项目自动聚类命名（建议制）的契约测试。

只新增，不改 tests/test_app.py 中的任何既有断言。
覆盖：建议字段的清洗与丢弃、提示里带上已有项目、**绝不自动移动**（红线）、
采纳 / 改名后加入 / 完全不加入三条路径、入参与隔离校验、旧报告兼容。
"""

import asyncio
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

import app.main as main_module
from app.db import Database
from app.llm import LLMAnalyzer, normalize_llm_payload
from app.models import ProjectListItem, TeamMeetingReport, Transcript, TranscriptSegment

AUTH_HEADERS = {"X-Access-Token": "test-access-token"}
OTHER_HEADERS = {"X-Access-Token": "other-team-token"}
MEETING = "m-suggest"


def _report(existing_id=None, new_name=None, confidence=0.8, reason="讨论的都是该项目的事") -> TeamMeetingReport:
    suggestion = None
    if existing_id or new_name:
        suggestion = {
            "existing_project_id": existing_id, "new_project_name": new_name,
            "confidence": confidence, "reason": reason,
        }
    return TeamMeetingReport(
        overview="会议确认本周上线内测。",
        meeting_points=["确定本周上线内测"],
        decisions=[{
            "content": "本周上线内测", "decision_maker": "胡泊",
            "evidence": {"quote": "我们本周上线内测", "timestamp": "00:00:02"},
        }],
        action_items=[], unresolved_issues=[],
        suggested_project=suggestion,
    )


@pytest.fixture(autouse=True)
def isolated_database(monkeypatch, tmp_path):
    database = Database(tmp_path / "suggest.db")
    database.initialize(main_module.settings.parsed_team_tokens())
    monkeypatch.setattr(main_module, "database", database)
    monkeypatch.setattr(main_module.speaker_recognizer, "enabled", False)
    return database


def _team_id(database: Database, token: str = "test-access-token") -> int:
    team_id = database.authenticate(token)
    assert team_id is not None
    return team_id


def _seed(database: Database, team_id: int, report: TeamMeetingReport, *, project_id=None) -> None:
    database.create_meeting(MEETING, team_id, "周会", Path("/tmp/m.wav"), project_id)
    database.save_report(MEETING, team_id, report)
    database.clear_audio_path(MEETING, team_id)


def _meeting(database: Database, team_id: int):
    return next(item for item in database.list_meetings(team_id) if item.id == MEETING)


# --------------------------------------------------------------------------
# 字段清洗与提示上下文
# --------------------------------------------------------------------------


def test_normalize_cleans_suggested_project() -> None:
    payload = {
        "meeting_points": ["要点"], "decisions": [], "action_items": [], "unresolved_issues": [],
        "suggested_project": {
            "existing_project_id": "  p1  ", "new_project_name": "很长的项目名" * 20,
            "confidence": 3, "reason": "x" * 500,
        },
    }
    normalized = normalize_llm_payload(payload, TeamMeetingReport)
    suggestion = normalized["suggested_project"]
    assert suggestion["existing_project_id"] == "p1"
    assert len(suggestion["new_project_name"]) <= 50
    assert suggestion["confidence"] == 1.0  # 夹到 0-1
    assert len(suggestion["reason"]) <= 200


def test_normalize_drops_malformed_suggestion() -> None:
    payload = {
        "meeting_points": ["要点"], "decisions": [], "action_items": [], "unresolved_issues": [],
        "suggested_project": "建议：放进产品项目",
    }
    normalized = normalize_llm_payload(payload, TeamMeetingReport)
    assert normalized.get("suggested_project") is None


def test_project_context_is_empty_without_projects() -> None:
    assert LLMAnalyzer._project_context(None) == ""
    assert LLMAnalyzer._project_context([]) == ""


def test_project_context_lists_existing_projects() -> None:
    projects = [
        ProjectListItem(id="p1", name="产品发布", parent_id=None, meeting_count=0, created_at="2026-09-21T00:00:00+00:00"),
    ]
    context = LLMAnalyzer._project_context(projects)
    assert "产品发布" in context and "p1" in context
    assert "不要编造" in context


def test_analyze_team_prompt_contains_project_list() -> None:
    """真实调用前先确认：项目列表确实进了提示词。"""
    analyzer = LLMAnalyzer(main_module.settings, client=object())  # client 不会被用到
    captured = {}

    async def fake_validated_call(model_type, system_prompt, user_prompt, stage, *args, **kwargs):
        captured["prompt"] = user_prompt
        return _report(existing_id="p1")

    analyzer._validated_call = fake_validated_call  # type: ignore[assignment]
    transcript = Transcript(
        language="zh", duration_seconds=10.0,
        segments=[TranscriptSegment(start=0.0, end=10.0, text="我们本周上线内测")],
    )
    projects = [
        ProjectListItem(id="p1", name="产品发布", parent_id=None, meeting_count=0, created_at="2026-09-21T00:00:00+00:00"),
    ]
    report = asyncio.run(analyzer.analyze_team(transcript, projects=projects))
    assert "产品发布" in captured["prompt"] and "p1" in captured["prompt"]
    assert report.suggested_project is not None


# --------------------------------------------------------------------------
# 建议必须落在本团队项目上；绝不自动移动（红线）
# --------------------------------------------------------------------------


def test_unknown_project_id_is_dropped(isolated_database) -> None:
    database = isolated_database
    team_id = _team_id(database)
    cleaned = main_module._sanitize_suggested_project(_report(existing_id="nope"), team_id)
    assert cleaned.suggested_project is None


def test_valid_project_id_is_kept(isolated_database) -> None:
    database = isolated_database
    team_id = _team_id(database)
    project = database.create_project("p1", team_id, "产品发布")
    cleaned = main_module._sanitize_suggested_project(_report(existing_id=project.id), team_id)
    assert cleaned.suggested_project is not None
    assert cleaned.suggested_project.existing_project_id == project.id


def test_empty_suggestion_is_dropped(isolated_database) -> None:
    database = isolated_database
    team_id = _team_id(database)
    assert main_module._sanitize_suggested_project(_report(), team_id).suggested_project is None


def test_saving_a_suggestion_never_moves_the_meeting(isolated_database) -> None:
    """红线：AI 只建议，归属永远由人确认。"""
    database = isolated_database
    team_id = _team_id(database)
    project = database.create_project("p1", team_id, "产品发布")
    _seed(database, team_id, _report(existing_id=project.id))
    assert _meeting(database, team_id).project_id is None
    history = database.get_history(MEETING, team_id)
    assert history.report.suggested_project is not None


# --------------------------------------------------------------------------
# 三条路径
# --------------------------------------------------------------------------


def test_accept_moves_meeting_and_clears_suggestion(isolated_database) -> None:
    database = isolated_database
    team_id = _team_id(database)
    project = database.create_project("p1", team_id, "产品发布")
    _seed(database, team_id, _report(existing_id=project.id))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post(
            "/api/meetings/{}/suggested-project".format(MEETING),
            json={"action": "accept", "project_id": project.id},
        )
    assert response.status_code == 200
    assert response.json()["project_id"] == project.id
    assert _meeting(database, team_id).project_id == project.id
    assert database.get_history(MEETING, team_id).report.suggested_project is None


def test_rename_creates_project_then_moves(isolated_database) -> None:
    database = isolated_database
    team_id = _team_id(database)
    _seed(database, team_id, _report(new_name="AI 建议的项目名"))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post(
            "/api/meetings/{}/suggested-project".format(MEETING),
            json={"action": "rename", "name": "我改的项目名"},
        )
    assert response.status_code == 200
    names = [item.name for item in database.list_projects(team_id)]
    assert "我改的项目名" in names
    moved = _meeting(database, team_id)
    assert moved.project_id
    assert database.get_project(moved.project_id, team_id).name == "我改的项目名"
    assert database.get_history(MEETING, team_id).report.suggested_project is None


def test_dismiss_keeps_meeting_unclassified(isolated_database) -> None:
    database = isolated_database
    team_id = _team_id(database)
    project = database.create_project("p1", team_id, "产品发布")
    _seed(database, team_id, _report(existing_id=project.id))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post(
            "/api/meetings/{}/suggested-project".format(MEETING), json={"action": "dismiss"},
        )
    assert response.status_code == 200
    assert response.json()["project_id"] is None  # 仍为未分类
    assert database.get_history(MEETING, team_id).report.suggested_project is None


# --------------------------------------------------------------------------
# 入参与隔离
# --------------------------------------------------------------------------


def test_invalid_action_is_rejected(isolated_database) -> None:
    database = isolated_database
    team_id = _team_id(database)
    _seed(database, team_id, _report(new_name="X"))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post(
            "/api/meetings/{}/suggested-project".format(MEETING), json={"action": "whatever"},
        )
    assert response.status_code == 422


def test_accept_without_project_is_rejected(isolated_database) -> None:
    database = isolated_database
    team_id = _team_id(database)
    _seed(database, team_id, _report(new_name="X"))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post(
            "/api/meetings/{}/suggested-project".format(MEETING), json={"action": "accept"},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_args"


def test_rename_without_name_is_rejected(isolated_database) -> None:
    database = isolated_database
    team_id = _team_id(database)
    _seed(database, team_id, _report(new_name="X"))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post(
            "/api/meetings/{}/suggested-project".format(MEETING), json={"action": "rename"},
        )
    assert response.status_code == 422


def test_accept_unknown_project_returns_not_found(isolated_database) -> None:
    database = isolated_database
    team_id = _team_id(database)
    _seed(database, team_id, _report(new_name="X"))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post(
            "/api/meetings/{}/suggested-project".format(MEETING),
            json={"action": "accept", "project_id": "does-not-exist"},
        )
    assert response.status_code == 404


def test_accept_other_team_project_is_forbidden(isolated_database) -> None:
    database = isolated_database
    team_id = _team_id(database)
    other_team = _team_id(database, "other-team-token")
    foreign = database.create_project("foreign", other_team, "别人的项目")
    _seed(database, team_id, _report(new_name="X"))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post(
            "/api/meetings/{}/suggested-project".format(MEETING),
            json={"action": "accept", "project_id": foreign.id},
        )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "team_forbidden"


def test_other_team_meeting_is_forbidden(isolated_database) -> None:
    database = isolated_database
    other_team = _team_id(database, "other-team-token")
    _seed(database, other_team, _report(new_name="X"))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post(
            "/api/meetings/{}/suggested-project".format(MEETING), json={"action": "dismiss"},
        )
    assert response.status_code == 403


def test_missing_report_returns_not_found(isolated_database) -> None:
    database = isolated_database
    team_id = _team_id(database)
    database.create_meeting(MEETING, team_id, "无报告", Path("/tmp/m.wav"))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post(
            "/api/meetings/{}/suggested-project".format(MEETING), json={"action": "dismiss"},
        )
    assert response.status_code == 404


def test_old_report_without_suggestion_is_compatible(isolated_database) -> None:
    """旧报告没有 suggested_project 字段 → 解析为 None，界面不显示卡片。"""
    database = isolated_database
    team_id = _team_id(database)
    payload = _report().model_dump()
    payload.pop("suggested_project", None)
    _seed(database, team_id, TeamMeetingReport.model_validate(payload))
    history = database.get_history(MEETING, team_id)
    assert history.report.suggested_project is None

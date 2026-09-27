"""阶段 9-C（M2）· R-P1.5-7 交付物状态与「待核对」的契约测试。

只新增，不改 tests/test_app.py 中的任何既有断言。
覆盖：四类交付物独立成/败、失败只标那一项、重试（不重跑已成功的部分）、
转写不可重试、报告重试会调用 AI、跨团队隔离、删除会议级联、老契约不变。
"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

import app.main as main_module
from app.db import Database
from app.deliverables.render import RendererUnavailable
from app.models import TeamMeetingReport, Transcript, TranscriptSegment

AUTH_HEADERS = {"X-Access-Token": "test-access-token"}
OTHER_HEADERS = {"X-Access-Token": "other-team-token"}
MEETING = "m-status"


def _report() -> TeamMeetingReport:
    return TeamMeetingReport(
        overview="会议确认本周上线内测。",
        meeting_points=["确定本周上线内测"],
        decisions=[{
            "content": "本周上线内测", "decision_maker": "胡泊",
            "evidence": {"quote": "我们本周上线内测", "timestamp": "00:00:02"},
        }],
        action_items=[{
            "task": "准备发布清单", "owner": "胡泊", "deadline": "周五",
            "evidence": {"quote": "我们本周上线内测", "timestamp": "00:00:02"},
        }],
        unresolved_issues=[{"content": "谁来验收", "evidence": {"quote": "我们本周上线内测", "timestamp": "00:00:02"}}],
        urgent_items=[],
    )


@pytest.fixture(autouse=True)
def isolated_database(monkeypatch, tmp_path):
    database = Database(tmp_path / "status.db")
    database.initialize(main_module.settings.parsed_team_tokens())
    monkeypatch.setattr(main_module, "database", database)
    monkeypatch.setattr(main_module.speaker_recognizer, "enabled", False)
    return database


def _team_id(database: Database, token: str = "test-access-token") -> int:
    team_id = database.authenticate(token)
    assert team_id is not None
    return team_id


def _seed(database: Database, team_id: int, *, with_report: bool = True) -> None:
    database.create_meeting(MEETING, team_id, "周会", Path("/tmp/m.wav"))
    database.save_transcript(MEETING, team_id, Transcript(
        language="zh", duration_seconds=12.0,
        segments=[TranscriptSegment(start=0.0, end=5.0, speaker_label="说话人 1", text="我们本周上线内测")],
    ))
    if with_report:
        database.save_report(MEETING, team_id, _report())
    database.clear_audio_path(MEETING, team_id)


def _statuses(client) -> dict:
    body = client.get("/api/meetings/{}/deliverables".format(MEETING)).json()
    return {item["kind"]: item for item in body["items"]}


# --------------------------------------------------------------------------
# 状态枚举与默认值
# --------------------------------------------------------------------------


def test_deliverables_default_to_pending(isolated_database) -> None:
    """没有任何数据时四类都是待生成。"""
    team_id = _team_id(isolated_database)
    isolated_database.create_meeting(MEETING, team_id, "空会议", Path("/tmp/m.wav"))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.get("/api/meetings/{}/deliverables".format(MEETING)).json()
    assert [item["kind"] for item in body["items"]] == [
        "transcript", "report", "tasks", "image_minutes",
    ]
    assert all(item["status"] == "pending" for item in body["items"])
    assert body["needs_review"] == 0


def test_legacy_meeting_infers_ok_from_existing_data(isolated_database) -> None:
    """本次升级前的会议没有状态行；已存在的报告/逐字稿应显示为正常而不是待生成。"""
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)  # 有逐字稿也有报告，但不写状态行
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        statuses = _statuses(client)
    assert {kind: statuses[kind]["status"] for kind in ("transcript", "report", "tasks", "image_minutes")} == {
        "transcript": "ok", "report": "ok", "tasks": "ok", "image_minutes": "ok",
    }


def test_inferred_state_is_overridden_by_written_state(isolated_database) -> None:
    """已有状态行时以记录为准（不能被推断值盖过）。"""
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)
    main_module._mark_deliverable(MEETING, team_id, "image_minutes", "needs_review", "renderer_unavailable")
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        statuses = _statuses(client)
    assert statuses["image_minutes"]["status"] == "needs_review"
    assert statuses["report"]["status"] == "ok"


def test_generating_image_minutes_marks_it_ok(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        client.post("/api/meetings/{}/image-minutes".format(MEETING))
        statuses = _statuses(client)
    assert statuses["image_minutes"]["status"] == "ok"
    assert statuses["image_minutes"]["retryable"] is False


def test_pipeline_marks_transcript_report_tasks_ok(isolated_database) -> None:
    """上线链路跑完后，前三类交付物应为 ok（这里直接调状态写入入口模拟）。"""
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)
    main_module._mark_deliverable(MEETING, team_id, "transcript", "ok")
    main_module._mark_deliverable(MEETING, team_id, "report", "ok")
    main_module._mark_deliverable(MEETING, team_id, "tasks", "ok")
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        statuses = _statuses(client)
    assert {kind: statuses[kind]["status"] for kind in ("transcript", "report", "tasks")} == {
        "transcript": "ok", "report": "ok", "tasks": "ok",
    }


# --------------------------------------------------------------------------
# 失败注入：只标那一项，其它照常可看（红线 11）
# --------------------------------------------------------------------------


def test_pdf_failure_marks_needs_review_only(isolated_database, monkeypatch) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)

    async def unavailable(html_text, *, renderer="auto"):
        raise RendererUnavailable("本机没有可用的 Chrome / Chromium，无法导出 PDF")

    monkeypatch.setattr(main_module, "render_pdf", unavailable)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        client.post("/api/meetings/{}/image-minutes".format(MEETING))  # 先生成四板块
        pdf = client.get("/api/meetings/{}/image-minutes.pdf".format(MEETING))
        statuses = _statuses(client)
        report_response = client.get("/api/meetings/{}".format(MEETING))
        tasks = client.get("/api/meetings/{}/my-tasks".format(MEETING))
    assert pdf.status_code == 503
    assert pdf.json()["error"]["code"] == "renderer_unavailable"
    assert statuses["image_minutes"]["status"] == "needs_review"
    assert statuses["image_minutes"]["error_code"] == "renderer_unavailable"
    assert statuses["image_minutes"]["retryable"] is True
    assert statuses["image_minutes"]["message"]
    # 其它交付物不受影响
    assert statuses["report"]["status"] != "needs_review"
    assert report_response.status_code == 200
    assert report_response.json()["report"]["overview"]
    assert tasks.status_code == 200
    assert report_response.json()["report"]["overview"]  # 内容仍在


def test_report_failure_marks_failed_with_code(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)
    main_module._mark_deliverable(MEETING, team_id, "report", "failed", "analysis_failed")
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        statuses = _statuses(client)
    assert statuses["report"]["status"] == "failed"
    assert statuses["report"]["retryable"] is True
    assert "AI" in (statuses["report"]["message"] or "")


# --------------------------------------------------------------------------
# 重试：不重跑已成功的部分
# --------------------------------------------------------------------------


def test_retry_image_minutes_renders_pdf_without_llm(isolated_database, monkeypatch) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)
    calls = {"render": 0}

    async def fake_render(html_text, *, renderer="auto"):
        calls["render"] += 1
        return b"%PDF-1.4 fake"

    monkeypatch.setattr(main_module, "render_pdf", fake_render)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        client.post("/api/meetings/{}/image-minutes".format(MEETING))
        before = isolated_database.usage_summary(team_id)["totals"]["calls"]
        response = client.post("/api/meetings/{}/retry?kind=image_minutes".format(MEETING))
        after = isolated_database.usage_summary(team_id)["totals"]["calls"]
        statuses = _statuses(client)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert before == after == 0, "图片纪要重试不应调用任何 LLM"
    assert calls["render"] == 1, "重试应当真的再渲染一次 PDF"
    assert statuses["image_minutes"]["status"] == "ok"


def test_retry_image_minutes_stays_needs_review_when_pdf_still_fails(isolated_database, monkeypatch) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)
    main_module._mark_deliverable(MEETING, team_id, "image_minutes", "needs_review", "renderer_unavailable")

    async def unavailable(html_text, *, renderer="auto"):
        raise RendererUnavailable("仍然不可用")

    monkeypatch.setattr(main_module, "render_pdf", unavailable)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post("/api/meetings/{}/retry?kind=image_minutes".format(MEETING))
        statuses = _statuses(client)
    assert response.status_code == 503
    assert statuses["image_minutes"]["status"] == "needs_review"


def test_viewing_report_does_not_clear_needs_review(isolated_database) -> None:
    """看一次报告不应该假装 PDF 问题已修好。"""
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)
    main_module._mark_deliverable(MEETING, team_id, "image_minutes", "needs_review", "renderer_unavailable")
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        client.post("/api/meetings/{}/image-minutes".format(MEETING))  # 只是生成四板块
        statuses = _statuses(client)
    assert statuses["image_minutes"]["status"] == "needs_review"


def test_retry_tasks_rebuilds_from_existing_report(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post("/api/meetings/{}/retry?kind=tasks".format(MEETING))
        statuses = _statuses(client)
    assert response.status_code == 200
    assert "未重新调用 AI" in response.json()["message"]
    assert statuses["tasks"]["status"] == "ok"


def test_retry_transcript_is_rejected_with_reason(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post("/api/meetings/{}/retry?kind=transcript".format(MEETING))
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "not_retryable"
    assert "重新上传" in response.json()["error"]["message"]


def test_retry_report_calls_analyzer_and_updates_status(isolated_database, monkeypatch) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)
    calls = {"n": 0}

    async def fake_build(transcript, analyzer):
        calls["n"] += 1
        return _report()

    monkeypatch.setattr(main_module, "build_team_report", fake_build)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post("/api/meetings/{}/retry?kind=report".format(MEETING))
        statuses = _statuses(client)
        meeting = client.get("/api/meetings").json()[0]
    assert response.status_code == 200
    assert calls["n"] == 1
    assert statuses["report"]["status"] == "ok"
    assert statuses["tasks"]["status"] == "ok"
    assert meeting["status"] == "完成"


def test_retry_report_failure_marks_failed(isolated_database, monkeypatch) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)

    async def boom(transcript, analyzer):
        raise RuntimeError("模型炸了")

    monkeypatch.setattr(main_module, "build_team_report", boom)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post("/api/meetings/{}/retry?kind=report".format(MEETING))
        statuses = _statuses(client)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "analysis_failed"
    assert statuses["report"]["status"] == "failed"


def test_retry_unknown_kind_is_rejected(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post("/api/meetings/{}/retry?kind=nope".format(MEETING))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_args"


# --------------------------------------------------------------------------
# 隔离、404、级联、老契约
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method,path", [
    ("get", "/api/meetings/{}/deliverables"),
    ("post", "/api/meetings/{}/retry?kind=tasks"),
])
def test_deliverable_endpoints_are_team_isolated(isolated_database, method, path) -> None:
    other_team = _team_id(isolated_database, "other-team-token")
    _seed(isolated_database, other_team)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = getattr(client, method)(path.format(MEETING))
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "team_forbidden"


def test_missing_meeting_returns_not_found(isolated_database) -> None:
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        assert client.get("/api/meetings/nope/deliverables").status_code == 404
        assert client.post("/api/meetings/nope/retry?kind=tasks").status_code == 404


def test_deleting_meeting_cascades_status(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)
    main_module._mark_deliverable(MEETING, team_id, "report", "ok")
    isolated_database.delete_meeting(MEETING, team_id)
    assert isolated_database.deliverable_statuses(MEETING, team_id) == []


def test_invalid_status_is_rejected(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)
    with pytest.raises(ValueError):
        isolated_database.set_deliverable_status(MEETING, team_id, "report", "搞不懂")


def test_task_status_keeps_old_fields_and_adds_deliverables(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)
    task = main_module.task_manager.get(MEETING, team_id)
    if task is None:  # 没有内存任务时直接构造一次，验证契约字段
        from app.models import TaskStatus
        task = TaskStatus(
            task_id=MEETING, status="完成", queue_position=0, message="完成",
            request_id="req", deliverables=[],
        )
    assert {"task_id", "status", "queue_position", "message", "request_id"}.issubset(
        set(task.model_dump().keys())
    )
    assert "deliverables" in task.model_dump()

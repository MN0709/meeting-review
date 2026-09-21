"""阶段 15（M4）· R-P2-5 批量上传的契约测试。

覆盖：一次多文件 → 多任务 + 多同意记录；单文件失败不影响其他；不选项目 → 未归类；
超上限明确提示；开关关闭；批次进度端点。
"""

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

import app.main as main_module
from app.db import Database
from app.models import TaskAccepted

AUTH_HEADERS = {"X-Access-Token": "test-access-token"}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    database = Database(tmp_path / "batch.db")
    database.initialize(main_module.settings.parsed_team_tokens())
    monkeypatch.setattr(main_module, "database", database)
    monkeypatch.setattr(main_module.speaker_recognizer, "enabled", False)
    main_module.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    async def fake_submit(path, request_id, *, task_id, team_id, long_meeting=False):
        database.upsert_agent_task(task_id, team_id, "排队中", meeting_id=task_id, request_id=request_id)
        created.append(task_id)
        return TaskAccepted(
            task_id=task_id, status="排队中", queue_position=len(created),
            message="", long_meeting=long_meeting,
        )

    monkeypatch.setattr(main_module.task_manager, "submit", fake_submit)
    return database, created


def _team(database: Database) -> int:
    team_id = database.authenticate("test-access-token")
    assert team_id is not None
    return team_id


def _files(*names):
    return [("files", (name, b"RIFF-fake-audio", "audio/wav")) for name in names]


def test_batch_creates_tasks_and_consent_records(isolated) -> None:
    database, created = isolated
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post(
            "/api/reviews", files=_files("a.wav", "b.wav", "c.wav"),
            data={"consent_confirmed": "true"},
        )
        assert response.status_code == 202
        body = response.json()
        progress = client.get("/api/reviews/{}".format(body["batch_id"])).json()
    assert body["total"] == 3 and body["accepted"] == 3
    assert all(item["ok"] for item in body["items"])
    assert len({item["task_id"] for item in body["items"]}) == 3
    assert len(created) == 3
    with database._connect() as connection:  # noqa: SLF001 - 测试内直接对账
        consents = connection.execute("SELECT COUNT(*) FROM consent_records").fetchone()[0]
    assert consents == 3
    assert progress["batch_id"] == body["batch_id"] and len(progress["tasks"]) == 3


def test_batch_one_bad_file_does_not_break_others(isolated) -> None:
    database, _ = isolated
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.post(
            "/api/reviews",
            files=[("files", ("good1.wav", b"RIFF", "audio/wav")),
                   ("files", ("bad.txt", b"x", "text/plain")),
                   ("files", ("good2.wav", b"RIFF", "audio/wav"))],
            data={"consent_confirmed": "true"},
        ).json()
    assert body["accepted"] == 2
    assert [item["ok"] for item in body["items"]] == [True, False, True]
    assert "仅支持" in body["items"][1]["error"]


def test_batch_without_project_goes_unclassified(isolated) -> None:
    database, _ = isolated
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.post(
            "/api/reviews", files=_files("a.wav", "b.wav"), data={"consent_confirmed": "true"},
        ).json()
    unclassified = database.list_meetings(_team(database), unclassified=True)
    assert len(unclassified) == 2
    assert {item.id for item in unclassified} == {item["task_id"] for item in body["items"]}


def test_batch_over_limit_is_rejected(isolated, monkeypatch) -> None:
    monkeypatch.setattr(main_module.settings, "batch_max_files", 2)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post(
            "/api/reviews", files=_files("a.wav", "b.wav", "c.wav"),
            data={"consent_confirmed": "true"},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_args"
    assert "最多上传 2 个" in response.json()["error"]["message"]


def test_batch_disabled(isolated, monkeypatch) -> None:
    monkeypatch.setattr(main_module.settings, "batch_upload_enabled", False)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post(
            "/api/reviews", files=_files("a.wav"), data={"consent_confirmed": "true"},
        )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "batch_disabled"


def test_batch_requires_consent(isolated) -> None:
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post("/api/reviews", files=_files("a.wav"))
    assert response.status_code == 422


def test_unknown_batch_returns_404(isolated) -> None:
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        assert client.get("/api/reviews/nope").status_code == 404

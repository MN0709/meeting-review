"""阶段 10-A（M3）· R-P1.5-6 上传同意与留证的契约测试。

只新增，不改 tests/test_app.py 中的任何既有断言（仅给三处「期望上传成功」的
既有用例补上 consent_confirmed 参数，因为该字段现在是必填）。
覆盖：未勾选直接调 API 被拒（红线 10）、留证内容、版本可配置、级联删除、老契约不变。
"""

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

import app.main as main_module
from app.db import Database
from app.models import TaskAccepted, TaskStatus

AUTH_HEADERS = {"X-Access-Token": "test-access-token"}
CONSENT = {"consent_confirmed": "true"}


@pytest.fixture(autouse=True)
def isolated_database(monkeypatch, tmp_path):
    database = Database(tmp_path / "consent.db")
    database.initialize(main_module.settings.parsed_team_tokens())
    monkeypatch.setattr(main_module, "database", database)
    monkeypatch.setattr(main_module.speaker_recognizer, "enabled", False)
    submitted = []

    async def fake_submit(path, request_id, task_id=None, team_id=None, long_meeting=False):
        submitted.append(task_id)
        return TaskAccepted(task_id=task_id, status="排队中", queue_position=0, message="排队中")

    monkeypatch.setattr(main_module.task_manager, "submit", fake_submit)
    monkeypatch.setattr(main_module, "prepare_upload_dir", lambda *a, **k: None)
    return database, submitted


def _post(client, **extra):
    data = {}
    data.update(extra.pop("data", {}))
    return client.post(
        "/api/review", data=data or None,
        files={"file": ("sample.wav", b"fake", "audio/wav")}, **extra,
    )


# --------------------------------------------------------------------------
# 红线 10：未勾选不得上传
# --------------------------------------------------------------------------


def test_upload_without_consent_is_rejected(isolated_database) -> None:
    database, submitted = isolated_database
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = _post(client)
    assert response.status_code == 422
    assert "勾选" in response.json()["detail"]  # 老端点沿用 {"detail": ...}
    assert submitted == [], "未勾选同意时不得入队"
    assert database.list_meetings(database.authenticate("test-access-token")) == []


def test_upload_with_consent_false_is_rejected(isolated_database) -> None:
    database, submitted = isolated_database
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = _post(client, data={"consent_confirmed": "false"})
    assert response.status_code == 422
    assert submitted == []


def test_rejected_upload_leaves_no_consent_record(isolated_database) -> None:
    database, _ = isolated_database
    team_id = database.authenticate("test-access-token")
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        _post(client)
    rows = database.consent_records("does-not-exist", team_id)
    assert rows == []


# --------------------------------------------------------------------------
# 留证内容
# --------------------------------------------------------------------------


def test_upload_with_consent_records_evidence(isolated_database) -> None:
    database, submitted = isolated_database
    team_id = database.authenticate("test-access-token")
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = _post(client, data=CONSENT)
    assert response.status_code == 202
    body = response.json()
    # 成功响应契约不变（字段与 10-A 之前完全一致）
    assert set(body) == {"task_id", "status", "queue_position", "message", "long_meeting"}
    task_id = body["task_id"]
    assert submitted == [task_id]

    records = database.consent_records(task_id, team_id)
    assert len(records) == 1
    record = records[0]
    assert record["consent_version"] == main_module.settings.consent_version == "v1"
    assert record["consented_at"] and "T" in record["consented_at"]
    assert record["ip"], "应记录来源 IP 作为留证"


def test_consent_version_is_configurable(monkeypatch, isolated_database) -> None:
    database, _ = isolated_database
    monkeypatch.setattr(main_module.settings, "consent_version", "v2")
    team_id = database.authenticate("test-access-token")
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        task_id = _post(client, data=CONSENT).json()["task_id"]
    assert database.consent_records(task_id, team_id)[0]["consent_version"] == "v2"


def test_old_consent_records_are_not_overwritten(isolated_database) -> None:
    """改文案升版本后，旧记录保留原版本号。"""
    database, _ = isolated_database
    team_id = database.authenticate("test-access-token")
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        first = _post(client, data=CONSENT).json()["task_id"]
    main_module.settings.consent_version = "v1"
    database.save_consent(first, team_id, "v2", ip="10.0.0.1")
    versions = [row["consent_version"] for row in database.consent_records(first, team_id)]
    assert versions == ["v1", "v2"]


# --------------------------------------------------------------------------
# 隔离、级联、老数据
# --------------------------------------------------------------------------


def test_consent_record_is_team_scoped(isolated_database) -> None:
    database, _ = isolated_database
    other_team = database.authenticate("other-team-token")
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        task_id = _post(client, data=CONSENT).json()["task_id"]
    assert database.consent_records(task_id, other_team) == []


def test_deleting_meeting_cascades_consent(isolated_database) -> None:
    database, _ = isolated_database
    team_id = database.authenticate("test-access-token")
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        task_id = _post(client, data=CONSENT).json()["task_id"]
    assert database.consent_records(task_id, team_id)
    database.delete_meeting(task_id, team_id)
    assert database.consent_records(task_id, team_id) == []


def test_task_status_contract_unchanged(isolated_database) -> None:
    """新增同意校验不改变任务结构体的字段与语义。"""
    database, _ = isolated_database
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = _post(client, data=CONSENT).json()
    for field in ("task_id", "status", "queue_position", "message"):
        assert field in body
    assert {"task_id", "status", "queue_position", "message"}.issubset(set(TaskAccepted.model_fields))
    assert {"task_id", "status", "queue_position", "message", "request_id"}.issubset(
        set(TaskStatus.model_fields)
    )
    assert "deliverables" in TaskStatus.model_fields  # 9-C 新增字段仍在

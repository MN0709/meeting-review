"""R-P2-1 · 个人模式（AUTH_ENABLED=false）与进阶口令模式的回归测试。

测试进程默认 AUTH_ENABLED=true（见 conftest.py），以保留历史跨工作区隔离测试；
本文件显式覆盖**生产默认的个人模式**，并验证 AUTH_ENABLED=true 能恢复口令校验。
"""

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

import app.main as main_module
from app.config import Settings
from app.db import Database


@pytest.fixture
def personal(monkeypatch, tmp_path):
    monkeypatch.setattr(main_module.settings, "auth_enabled", False)
    database = Database(tmp_path / "personal.db")
    database.initialize({}, owner_name="我", auth_enabled=False)
    monkeypatch.setattr(main_module, "database", database)
    monkeypatch.setattr(main_module.speaker_recognizer, "enabled", False)
    return database


def test_personal_mode_serves_requests_without_any_token(personal) -> None:
    """验收标准 ②：不带任何请求头直调 /api/meetings 返回 200。"""
    with TestClient(main_module.app) as client:
        assert client.get("/api/meetings").status_code == 200
        assert client.get("/api/projects").status_code == 200


def test_auth_check_returns_personal_workspace(personal) -> None:
    with TestClient(main_module.app) as client:
        body = client.get("/api/auth/check").json()
    assert body["status"] == "ok"
    assert body["auth_enabled"] is False
    assert body["workspace"] == "我"
    assert body["debug_panels"] is False
    assert body["team_id"] == personal.owner_workspace_id()


def test_personal_mode_uses_single_workspace_for_writes(personal) -> None:
    with TestClient(main_module.app) as client:
        created = client.post("/api/projects", json={"name": "个人项目"}).json()
        listed = client.get("/api/projects").json()
    assert any(item["id"] == created["id"] for item in listed)


def test_terms_endpoints_are_gone_in_personal_mode(personal) -> None:
    """R-P2-9：删除热词后 /api/terms 返回 404。"""
    with TestClient(main_module.app) as client:
        assert client.get("/api/terms").status_code == 404
        assert client.post("/api/terms", json={"term": "会脉"}).status_code == 404


def test_auth_enabled_restores_token_check(monkeypatch, tmp_path) -> None:
    """验收标准 ③：AUTH_ENABLED=true 时恢复口令校验（回归）。"""
    monkeypatch.setattr(main_module.settings, "auth_enabled", True)
    database = Database(tmp_path / "auth-mode.db")
    database.initialize({"甲团队": "test-access-token"}, auth_enabled=True)
    monkeypatch.setattr(main_module, "database", database)
    with TestClient(main_module.app) as client:
        assert client.get("/api/meetings").status_code == 403
        assert client.get("/api/meetings", headers={"X-Access-Token": "wrong"}).status_code == 403
        assert client.get(
            "/api/meetings", headers={"X-Access-Token": "test-access-token"}
        ).status_code == 200


def test_default_host_is_loopback() -> None:
    """红线 12：默认只能本机访问（不得默认监听 0.0.0.0）。"""
    assert Settings().app_host == "127.0.0.1"


def test_team_tokens_optional_in_personal_mode(monkeypatch) -> None:
    monkeypatch.delenv("TEAM_TOKENS", raising=False)
    settings = Settings(_env_file=None, AUTH_ENABLED=False)
    assert settings.team_tokens is None
    assert settings.parsed_team_tokens() == {}


def test_auth_enabled_requires_team_tokens(monkeypatch) -> None:
    monkeypatch.delenv("TEAM_TOKENS", raising=False)
    settings = Settings(_env_file=None, AUTH_ENABLED=True)
    with pytest.raises(ValueError):
        settings.parsed_team_tokens()


def test_personal_workspace_is_reused_across_restarts(tmp_path) -> None:
    """个人工作区持久化在 app_state，重启后不新建工作区。"""
    path = tmp_path / "reuse.db"
    first = Database(path)
    first.initialize({}, owner_name="我", auth_enabled=False)
    workspace_id = first.owner_workspace_id()
    second = Database(path)
    second.initialize({}, owner_name="我", auth_enabled=False)
    assert second.owner_workspace_id() == workspace_id
    assert second.get_app_state("owner_workspace_id") == str(workspace_id)


def test_app_state_roundtrip(tmp_path) -> None:
    database = Database(tmp_path / "state.db")
    database.initialize({}, owner_name="我", auth_enabled=False)
    assert database.get_app_state("owner_onboarding_done") is None
    database.set_app_state("owner_onboarding_done", "1")
    assert database.get_app_state("owner_onboarding_done") == "1"

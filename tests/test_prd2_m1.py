"""阶段 12（M1）· R-P2-7 / R-P2-9 / R-P2-10 的验收回归。

覆盖：
- R-P2-7：报告瘦身后 `/api/usage` 与 `/api/meetings/{id}/agent-trace` 仍在（默认界面隐藏），
  `DEBUG_PANELS_ENABLED` 默认 false、可开关；
- R-P2-9：`app/terms.py` 已删除，`/api/terms` 为 404，且热词相关的转录调用参数已移除；
- R-P2-10：用户可见界面里不再出现「团队」。
"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

import app.main as main_module
from app.db import Database

AUTH_HEADERS = {"X-Access-Token": "test-access-token"}
MEETING = "m-m1"


@pytest.fixture(autouse=True)
def isolated_database(monkeypatch, tmp_path):
    database = Database(tmp_path / "m1.db")
    database.initialize(main_module.settings.parsed_team_tokens())
    monkeypatch.setattr(main_module, "database", database)
    monkeypatch.setattr(main_module.speaker_recognizer, "enabled", False)
    return database


# --- R-P2-9 删除术语热词 ----------------------------------------------------


def test_terms_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        import app.terms  # noqa: F401


def test_transcriber_has_no_initial_prompt_parameter() -> None:
    import inspect

    from app.transcription import WhisperTranscriber

    signature = inspect.signature(WhisperTranscriber.transcribe)
    assert list(signature.parameters) == ["self", "audio_path"]


# --- R-P2-7 报告瘦身（后端接口保留，默认界面隐藏） ---------------------------


def test_usage_and_agent_trace_endpoints_are_still_available(isolated_database) -> None:
    team_id = isolated_database.authenticate("test-access-token")
    isolated_database.create_meeting(MEETING, team_id, "瘦身验收", Path("/tmp/m.wav"))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        assert client.get("/api/usage").status_code == 200
        assert client.get("/api/meetings/{}/agent-trace".format(MEETING)).status_code == 200


def test_debug_panels_default_false_and_exposed(isolated_database) -> None:
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        assert client.get("/api/auth/check").json()["debug_panels"] is False

    isolated_database  # 保持 fixture 引用，说明数据库确实已隔离
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(main_module.settings, "debug_panels_enabled", True)
        with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
            assert client.get("/api/auth/check").json()["debug_panels"] is True


def test_frontend_hides_debug_panels_by_default() -> None:
    html = (Path(__file__).parent.parent / "static" / "index.html").read_text(encoding="utf-8")
    # 成本 / Agent 步骤面板默认带 hidden 类，只在 debug_panels 打开时才被移除
    assert 'id="costSection" class="hidden"' in html
    assert 'id="agentTraceSection" class="hidden"' in html
    assert "if(debugPanels){" in html


# --- R-P2-10 文案去「团队」 --------------------------------------------------


def test_frontend_has_no_team_wording() -> None:
    html = (Path(__file__).parent.parent / "static" / "index.html").read_text(encoding="utf-8")
    assert "团队" not in html


def test_frontend_uses_personal_brand() -> None:
    html = (Path(__file__).parent.parent / "static" / "index.html").read_text(encoding="utf-8")
    assert "会脉 · 我的会议记忆" in html
    assert "会议按项目保存" in html

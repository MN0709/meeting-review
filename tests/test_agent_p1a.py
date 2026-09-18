"""阶段 3（P1-A）契约与地基的测试。

只新增，不修改 tests/test_app.py 与 tests/test_p0.py 的任何既有断言。
覆盖：R-P1-6（ToolResult 契约）、R-P1-2（工具注册中心）、R-P1-5/8 的存储底座、
以及 PRD §10.3 新增的「删除会议级联清理」。
"""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# 与 tests/test_app.py 保持一致，避免独立运行时误用真实数据库。
os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

from app.agent.tools.contract import (
    MAX_SUMMARY_CHARS,
    ToolResult,
    failure,
    success,
)
from app.agent.tools.registry import ToolRegistry
from app.config import Settings
from app.db import Database
from app.models import Transcript, TranscriptSegment

TEAM = "测试团队"
TOKEN = "test-access-token"
PROJECT_DIR = Path(__file__).resolve().parent.parent
AUTH = {"X-Access-Token": TOKEN}


# ---------------------------------------------------------------------------
# R-P1-6：ToolResult 统一契约
# ---------------------------------------------------------------------------


def test_success_result_shape() -> None:
    result = success("已找到 1 条记录", {"items": [1]})
    assert isinstance(result, ToolResult)
    assert result.ok is True and result.code == "ok"
    assert result.payload == {"items": [1]}
    assert result.truncated is False and result.artifacts == []


@pytest.mark.parametrize(
    "code",
    ["not_found", "team_forbidden", "invalid_args", "too_large", "denied_host_owned", "tool_error"],
)
def test_failure_result_for_each_error_code(code: str) -> None:
    result = failure(code)
    assert result.ok is False and result.code == code
    # 失败必须给模型一句可读的中文原因（PRD P-7 观察面回灌）。
    assert result.summary and len(result.summary) <= MAX_SUMMARY_CHARS


def test_failure_rejects_ok_and_unknown_codes() -> None:
    with pytest.raises(ValueError):
        failure("ok")
    with pytest.raises(ValueError):
        failure("bogus_code")


def test_truncated_must_carry_next_hint() -> None:
    with pytest.raises(ValidationError):
        ToolResult(ok=True, code="ok", summary="x", truncated=True)
    result = success("内容过多", truncated=True, next_hint="用 page=2 取下一页")
    assert result.next_hint


def test_summary_is_clipped_to_the_contract_limit() -> None:
    result = success("甲" * 500)
    assert len(result.summary) <= MAX_SUMMARY_CHARS


# ---------------------------------------------------------------------------
# R-P1-2：工具注册中心
# ---------------------------------------------------------------------------


def _handler(**_kwargs) -> ToolResult:
    return success("ok")


def test_registry_register_get_and_list() -> None:
    registry = ToolRegistry()
    registry.register_tool(
        name="get_transcript",
        description="读取一场会议的分页转写摘要",
        input_schema={"type": "object", "properties": {"meeting_id": {"type": "string"}}},
        handler=_handler,
    )
    assert registry.names() == ["get_transcript"]
    spec = registry.get("get_transcript")
    assert spec is not None and spec.permission_level == "readonly"
    assert registry.get("missing") is None
    assert "get_transcript" in registry and len(registry) == 1


def test_registry_rejects_duplicate_and_invalid_specs() -> None:
    registry = ToolRegistry()
    registry.register_tool(name="t1", description="d", input_schema={}, handler=_handler)
    with pytest.raises(ValueError):
        registry.register(registry.get("t1"))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        registry.register_tool(
            name="t2", description="d", input_schema={}, handler=_handler,
            permission_level="superuser",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        registry.register_tool(name="bad name!", description="d", input_schema={}, handler=_handler)


# ---------------------------------------------------------------------------
# 配置：Agent 开关默认值必须保证零行为变化
# ---------------------------------------------------------------------------


def test_agent_settings_have_safe_defaults() -> None:
    fields = Settings.model_fields
    assert fields["agent_mode"].default == "pipeline"
    assert fields["agent_max_steps"].default == 20
    assert fields["agent_context_budget_tokens"].default == 60000
    assert fields["agent_tools_enabled"].default == "readonly"
    assert fields["agent_write_tools_enabled"].default is False
    assert fields["agent_audit_enabled"].default is True


def test_agent_mode_can_be_overridden_and_is_validated(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_MODE", "shadow")
    assert Settings(TEAM_TOKENS=TOKEN).agent_mode == "shadow"
    monkeypatch.setenv("AGENT_MODE", "bogus")
    with pytest.raises(ValidationError):
        Settings(TEAM_TOKENS=TOKEN)


# ---------------------------------------------------------------------------
# 数据层：新表、FTS5、任务与审计存储、删除级联
# ---------------------------------------------------------------------------


def _fresh(tmp_path, name: str = "agent.db"):
    path = tmp_path / name
    database = Database(path)
    database.initialize({TEAM: TOKEN})
    return database, path


def test_agent_tables_and_fts_are_created(tmp_path) -> None:
    database, path = _fresh(tmp_path)
    with sqlite3.connect(path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"agent_tasks", "agent_task_deps", "agent_audit", "transcript_fts"} <= tables
    assert database.fts5_available() is True


def test_agent_tables_are_backfilled_on_an_old_database(tmp_path) -> None:
    """向前兼容：旧库（没有 Agent 三张表）打开后自动补全。"""
    database, path = _fresh(tmp_path, "older.db")
    with sqlite3.connect(path) as connection:
        for table in ("agent_tasks", "agent_task_deps", "agent_audit"):
            connection.execute("DROP TABLE {}".format(table))
    database.initialize({TEAM: TOKEN})
    with sqlite3.connect(path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"agent_tasks", "agent_task_deps", "agent_audit"} <= tables


def test_fts_index_follows_transcript_changes(tmp_path) -> None:
    database, path = _fresh(tmp_path, "fts.db")
    team_id = database.authenticate(TOKEN)
    assert team_id is not None
    database.create_meeting("m1", team_id, "周会", tmp_path / "a.m4a")
    database.save_transcript(
        "m1", team_id,
        Transcript(
            duration_seconds=4.0,
            segments=[
                TranscriptSegment(start=0.0, end=2.0, text="决定本周上线"),
                TranscriptSegment(start=2.0, end=4.0, text="我来准备发布清单"),
            ],
        ),
    )
    with sqlite3.connect(path) as connection:
        hits = connection.execute(
            "SELECT rowid FROM transcript_fts WHERE transcript_fts MATCH ?", ("本周上线",)
        ).fetchall()
    assert hits, "trigram 索引应支持中文子串匹配（查询词 ≥ 3 字）"

    database.delete_meeting("m1", team_id)
    with sqlite3.connect(path) as connection:
        hits = connection.execute(
            "SELECT rowid FROM transcript_fts WHERE transcript_fts MATCH ?", ("本周上线",)
        ).fetchall()
    assert hits == [], "删除会议后索引项必须同步移除"


def test_agent_task_storage_roundtrip_and_team_scope(tmp_path) -> None:
    database, _ = _fresh(tmp_path, "tasks.db")
    team_id = database.authenticate(TOKEN)
    assert team_id is not None
    database.upsert_agent_task(
        "t1", team_id, "转写中", meeting_id="m1",
        progress_json=json.dumps({"steps": ["read_transcript"]}, ensure_ascii=False),
    )
    row = database.get_agent_task("t1", team_id)
    assert row is not None and row["status"] == "转写中"
    assert json.loads(row["progress_json"])["steps"] == ["read_transcript"]
    assert database.list_active_agent_tasks()[0]["task_id"] == "t1"

    database.upsert_agent_task("t1", team_id, "完成", finished_at="2026-09-18T00:00:00+00:00")
    assert database.get_agent_task("t1", team_id)["status"] == "完成"
    assert database.list_active_agent_tasks() == []
    assert database.get_agent_task("t1", 999) is None


def test_agent_audit_is_team_scoped_and_replayable(tmp_path) -> None:
    database, _ = _fresh(tmp_path, "audit.db")
    team_id = database.authenticate(TOKEN)
    assert team_id is not None
    for step in (1, 2):
        database.record_agent_audit(
            team_id=team_id, session_id="s1", meeting_id="m1", step=step,
            tool_name="get_transcript", args_digest="abc123",
            decision="allow", result_code="ok", duration_ms=12,
        )
    replayed = database.list_agent_audit(team_id, session_id="s1")
    assert [row["step"] for row in replayed] == [1, 2]
    assert database.list_agent_audit(999) == []


def test_deleting_a_meeting_cascades_agent_data(tmp_path) -> None:
    """PRD §10.3：删除会议时，Agent 新增的关联数据一并清理。"""
    database, _ = _fresh(tmp_path, "cascade.db")
    team_id = database.authenticate(TOKEN)
    assert team_id is not None
    database.create_meeting("m1", team_id, "周会", tmp_path / "a.m4a")
    database.record_llm_usage(team_id, "team_final_direct", "deepseek-flash", meeting_id="m1")
    database.record_agent_audit(
        team_id=team_id, session_id="s1", meeting_id="m1", decision="allow", result_code="ok"
    )
    database.upsert_agent_task("t1", team_id, "完成", meeting_id="m1")

    database.delete_meeting("m1", team_id)

    assert database.list_agent_audit(team_id) == []
    assert database.usage_summary(team_id)["totals"]["calls"] == 0
    assert database.get_agent_task("t1", team_id) is None


# ---------------------------------------------------------------------------
# R-P1-2：调试端点默认关闭
# ---------------------------------------------------------------------------


def test_agent_tools_endpoint_is_absent_in_pipeline_mode() -> None:
    from fastapi.testclient import TestClient

    import app.main as main_module

    with TestClient(main_module.app) as client:
        assert client.get("/api/agent/tools", headers=AUTH).status_code == 404


def test_agent_tools_endpoint_exists_in_agent_mode(tmp_path) -> None:
    """子进程真实启动一次；AGENT_MODE=agent 时端点才存在（默认关闭）。"""
    environment = os.environ.copy()
    environment["TEAM_TOKENS"] = "测试团队:test-access-token"
    environment["AGENT_MODE"] = "agent"
    environment["DATABASE_PATH"] = str(tmp_path / "agent-mode.db")
    environment["PYTHONPATH"] = str(PROJECT_DIR)
    code = (
        "from fastapi.testclient import TestClient\n"
        "import app.main as m\n"
        "with TestClient(m.app) as c:\n"
        "    r = c.get('/api/agent/tools', headers={'X-Access-Token': 'test-access-token'})\n"
        "    assert r.status_code == 200, r.text\n"
        "    body = r.json()\n"
        "    assert body['mode'] == 'agent'\n"
        "    assert len(body['tools']) == 11\n"
        "    assert 'get_transcript' in {t['name'] for t in body['tools']}\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(PROJECT_DIR), env=environment,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr

"""P0 可信地基的测试。

只新增，不修改 tests/test_app.py 中的任何既有断言。
覆盖：R-P0-1（可复现构建的配置面）、R-P0-3（口令与密钥加固）、R-P0-2（成本落库，见文件末尾）。
"""

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import MIN_TEAM_TOKEN_LENGTH, Settings
from app.db import Database
from app.llm import LLMAnalyzer, usage_scope
from app.models import Transcript, TranscriptSegment


PROJECT_DIR = Path(__file__).parent.parent
STRONG_TOKEN_A = "mr-Abc123def456GHI789"
STRONG_TOKEN_B = "mr-Zyx987wvu654TSR321"


def _settings(team_tokens: str) -> Settings:
    return Settings(_env_file=None, TEAM_TOKENS=team_tokens)


# --------------------------------------------------------------------------
# R-P0-3：口令强度
# --------------------------------------------------------------------------


def test_strong_team_tokens_are_accepted() -> None:
    teams = _settings(f"甲团队:{STRONG_TOKEN_A},乙团队:{STRONG_TOKEN_B}").parsed_team_tokens()
    assert teams == {"甲团队": STRONG_TOKEN_A, "乙团队": STRONG_TOKEN_B}


def test_short_team_token_is_rejected_with_actionable_message() -> None:
    with pytest.raises(ValueError) as excinfo:
        _settings("产品项目组:1234").parsed_team_tokens()
    message = str(excinfo.value)
    assert f"至少 {MIN_TEAM_TOKEN_LENGTH} 位" in message
    assert "1234" not in message  # 不把口令原文写进错误信息


def test_pure_digit_team_token_is_rejected() -> None:
    with pytest.raises(ValueError, match="不能是纯数字"):
        _settings("甲团队:" + "9" * 20).parsed_team_tokens()


def test_pure_ascii_letter_team_token_is_rejected() -> None:
    with pytest.raises(ValueError, match="不能是纯字母"):
        _settings("甲团队:" + "abcdefghijklmnop").parsed_team_tokens()


def test_long_chinese_passphrase_is_accepted() -> None:
    """中文长口令熵足够高，不应被"纯字母"规则误伤（刻意的细化）。"""
    passphrase = "会脉团队长期口令甲乙丙丁戊己庚辛壬"
    assert len(passphrase) >= MIN_TEAM_TOKEN_LENGTH
    teams = _settings(f"甲团队:{passphrase}").parsed_team_tokens()
    assert teams["甲团队"] == passphrase


@pytest.mark.parametrize("weak", ["产品项目组:1234", "产品项目组:" + "a" * 20, "产品项目组:short"])
def test_weak_team_token_refuses_to_start(weak: str) -> None:
    """弱口令必须让服务拒绝启动，而不是等到运行时才报错。"""
    environment = os.environ.copy()
    environment["TEAM_TOKENS"] = weak
    environment["PYTHONPATH"] = str(PROJECT_DIR)
    result = subprocess.run(
        [sys.executable, "-c", "from app.config import get_settings; get_settings()"],
        cwd=str(PROJECT_DIR),
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "服务拒绝启动" in result.stderr
    assert "口令" in result.stderr


def test_get_settings_reports_redacted_api_key_state() -> None:
    assert Settings(_env_file=None, TEAM_TOKENS=f"甲团队:{STRONG_TOKEN_A}").redacted_api_key_state() == "未配置"
    assert (
        Settings(
            _env_file=None, TEAM_TOKENS=f"甲团队:{STRONG_TOKEN_A}", OPENAI_API_KEY="sk-not-a-real-key"
        ).redacted_api_key_state()
        == "已配置"
    )


# --------------------------------------------------------------------------
# 认证路径健壮性（P0 冒烟发现的现有缺陷）
#
# 现象：客户端发来含非 ASCII 字节的 X-Access-Token 时，hmac.compare_digest 会抛
#       TypeError("comparing strings with non-ASCII characters is not supported")，
#       接口返回 500 而不是 403，控制台还会打出堆栈。
# 修复：在 Database.authenticate 里统一转 UTF-8 字节后比较。
# --------------------------------------------------------------------------


def test_non_ascii_token_is_rejected_by_database(tmp_path) -> None:
    database = Database(tmp_path / "auth.db")
    database.initialize({"甲团队": STRONG_TOKEN_A})
    assert database.authenticate("æ" * 20) is None
    assert database.authenticate("") is None
    assert database.authenticate(STRONG_TOKEN_A) is not None


def test_non_ascii_header_returns_403_not_500(tmp_path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    import app.main as main_module

    teams = main_module.settings.parsed_team_tokens()
    database = Database(tmp_path / "auth-http.db")
    database.initialize(teams)
    monkeypatch.setattr(main_module, "database", database)
    valid_token = next(iter(teams.values()))

    with TestClient(main_module.app) as client:
        # 直接发原始字节，模拟客户端发来畸形口令（Starlette 按 latin-1 解码成非 ASCII）
        malformed = client.get("/api/meetings", headers={"X-Access-Token": b"\xe6" * 20})
        assert malformed.status_code == 403
        assert malformed.json() == {"detail": "团队口令错误"}
        assert client.get("/api/meetings", headers={"X-Access-Token": valid_token}).status_code == 200


def test_startup_log_never_contains_the_api_key(tmp_path) -> None:
    """子进程真实启动一次，确认密钥原文不会进入启动日志。"""
    secret = "sk-startup-log-must-not-contain-me"
    environment = os.environ.copy()
    environment["TEAM_TOKENS"] = f"甲团队:{STRONG_TOKEN_A}"
    environment["OPENAI_API_KEY"] = secret
    environment["DATABASE_PATH"] = str(tmp_path / "log-check.db")
    environment["PYTHONPATH"] = str(PROJECT_DIR)
    code = (
        "from fastapi.testclient import TestClient\n"
        "import app.main as m\n"
        "with TestClient(m.app) as client:\n"
        "    assert client.get('/health').status_code == 200\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_DIR),
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert secret not in combined
    assert "api_key=已配置" in combined


# --------------------------------------------------------------------------
# R-P0-2：LLM 成本落库与归因
# --------------------------------------------------------------------------

TEAM_NAME = "甲团队"

REPORT_PAYLOAD = {
    "overview": "会议决定本周上线",
    "meeting_points": ["决定本周上线"],
    "decisions": [
        {
            "content": "本周上线",
            "decision_maker": "小宁",
            "evidence": {"quote": "决定本周上线", "timestamp": "00:00:00"},
        }
    ],
    "action_items": [{
        "task": "准备清单", "owner": "小宁", "deadline": "周五",
        "evidence": {"quote": "好，我去准备清单", "timestamp": "00:00:03"},
    }],
    "unresolved_issues": [],
}


def _transcript() -> Transcript:
    return Transcript(
        duration_seconds=5.0,
        segments=[
            TranscriptSegment(start=0.0, end=2.0, speaker_label="A", text="决定本周上线"),
            TranscriptSegment(start=2.0, end=5.0, speaker_label="B", text="好，我去准备清单"),
        ],
    )


class _RecordingCompletions:
    """模拟只支持 json_object 的模型（如 deepseek-flash）：json_schema 一律报错。"""

    def __init__(self, payload: dict, reject_json_schema: bool = True) -> None:
        self.payload = payload
        self.reject_json_schema = reject_json_schema
        self.response_formats: list = []

    def create(self, **kwargs):
        mode = kwargs["response_format"]["type"]
        self.response_formats.append(mode)
        if mode == "json_schema" and self.reject_json_schema:
            raise RuntimeError("json_schema not supported by this model")
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.payload, ensure_ascii=False)))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )
        return response


class _FakeClient:
    def __init__(self, completions: _RecordingCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


# ---- 数据库层 --------------------------------------------------------------


def test_llm_usage_table_is_added_to_a_database_that_lacks_it(tmp_path) -> None:
    """向前兼容：旧库（没有 llm_usage）打开后自动补全。"""
    path = tmp_path / "older.db"
    database = Database(path)
    database.initialize({TEAM_NAME: STRONG_TOKEN_A})
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE llm_usage")
    database.initialize({TEAM_NAME: STRONG_TOKEN_A})
    with sqlite3.connect(path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "llm_usage" in tables


def test_initialize_is_idempotent_for_llm_usage(tmp_path) -> None:
    database = Database(tmp_path / "repeat.db")
    for _ in range(3):
        database.initialize({TEAM_NAME: STRONG_TOKEN_A})
    assert database.usage_summary(1)["totals"]["calls"] == 0


def test_record_and_summarize_llm_usage(tmp_path) -> None:
    database = Database(tmp_path / "usage.db")
    database.initialize({TEAM_NAME: STRONG_TOKEN_A})
    team_id = database.authenticate(STRONG_TOKEN_A)
    for stage, prompt, completion, duration in (
        ("team_chunk_1", 100, 20, 1500),
        ("team_chunk_1", 50, 10, 500),
        ("team_final_merge", 10, 5, 100),
    ):
        database.record_llm_usage(
            team_id=team_id, stage=stage, model="deepseek-flash", meeting_id="mtg-1",
            project_id="proj-1", prompt_tokens=prompt, completion_tokens=completion,
            total_tokens=prompt + completion, duration_ms=duration,
        )

    summary = database.usage_summary(team_id, meeting_id="mtg-1")
    assert summary["totals"]["calls"] == 3
    assert summary["totals"]["total_tokens"] == 195
    # 耗时也被记录（手册§九.3 [A] 要求：模型、耗时、Token 用量、错误类型）
    assert summary["totals"]["duration_ms"] == 2100
    by_stage = {row["stage"]: row for row in summary["by_stage"]}
    assert by_stage["team_chunk_1"]["calls"] == 2
    assert by_stage["team_chunk_1"]["total_tokens"] == 180
    assert by_stage["team_final_merge"]["total_tokens"] == 15
    assert [row["meeting_id"] for row in summary["by_meeting"]] == ["mtg-1"]


def test_usage_summary_is_team_scoped(tmp_path) -> None:
    database = Database(tmp_path / "scoped.db")
    database.initialize({TEAM_NAME: STRONG_TOKEN_A, "乙团队": STRONG_TOKEN_B})
    team_a = database.authenticate(STRONG_TOKEN_A)
    team_b = database.authenticate(STRONG_TOKEN_B)
    database.record_llm_usage(
        team_id=team_a, stage="team_final_direct", model="m", meeting_id="mtg-a",
        prompt_tokens=10, completion_tokens=5, total_tokens=15, duration_ms=10,
    )
    assert database.usage_summary(team_a)["totals"]["calls"] == 1
    assert database.usage_summary(team_b)["totals"]["calls"] == 0
    assert database.usage_summary(team_b)["by_stage"] == []


def test_empty_usage_summary_returns_zero_values(tmp_path) -> None:
    database = Database(tmp_path / "empty.db")
    database.initialize({TEAM_NAME: STRONG_TOKEN_A})
    summary = database.usage_summary(database.authenticate(STRONG_TOKEN_A))
    assert summary["totals"]["calls"] == 0
    assert summary["totals"]["total_tokens"] == 0
    assert summary["by_stage"] == []
    assert summary["by_meeting"] == []


# ---- 调用层：耗时记录 + 双倍请求修复 ---------------------------------------


def test_usage_recorder_receives_stage_duration_and_context() -> None:
    recorded: list = []
    completions = _RecordingCompletions(REPORT_PAYLOAD)
    analyzer = LLMAnalyzer(
        Settings(OPENAI_API_KEY="test"),
        client=_FakeClient(completions),
        usage_recorder=recorded.append,
    )
    context = {"team_id": 7, "meeting_id": "mtg-1", "project_id": "proj-1"}
    with usage_scope(context):
        asyncio.run(analyzer.analyze_team(_transcript()))

    assert recorded, "应该至少记录一次用量"
    assert all(item["team_id"] == 7 for item in recorded)
    assert all(item["meeting_id"] == "mtg-1" for item in recorded)
    assert all(item["project_id"] == "proj-1" for item in recorded)
    assert all(isinstance(item["duration_ms"], int) for item in recorded)
    assert recorded[0]["total_tokens"] == 120
    # 只记数值与标识，不记 prompt / 回复原文
    assert set(recorded[0]) == {
        "team_id", "meeting_id", "project_id", "stage", "model",
        "prompt_tokens", "completion_tokens", "total_tokens", "duration_ms",
    }


def test_recorder_failure_never_breaks_the_analysis_path() -> None:
    def broken_recorder(_payload):
        raise RuntimeError("数据库挂了")

    analyzer = LLMAnalyzer(
        Settings(OPENAI_API_KEY="test"),
        client=_FakeClient(_RecordingCompletions(REPORT_PAYLOAD)),
        usage_recorder=broken_recorder,
    )
    with usage_scope({"team_id": 1}):
        report = asyncio.run(analyzer.analyze_team(_transcript()))
    assert report.overview


def test_json_schema_capability_is_cached_to_avoid_double_requests() -> None:
    """修复前：每次调用都先发一次必然失败的 json_schema 请求，等于双倍花费。

    修复后：第一次失败后缓存能力，第二次起直接走 json_object。
    """
    completions = _RecordingCompletions(REPORT_PAYLOAD)
    analyzer = LLMAnalyzer(Settings(OPENAI_API_KEY="test"), client=_FakeClient(completions))
    asyncio.run(analyzer.analyze_team(_transcript()))
    asyncio.run(analyzer.analyze_team(_transcript()))

    assert completions.response_formats == ["json_schema", "json_object", "json_object"]


def test_json_schema_is_kept_when_the_model_supports_it() -> None:
    completions = _RecordingCompletions(REPORT_PAYLOAD, reject_json_schema=False)
    analyzer = LLMAnalyzer(Settings(OPENAI_API_KEY="test"), client=_FakeClient(completions))
    asyncio.run(analyzer.analyze_team(_transcript()))
    asyncio.run(analyzer.analyze_team(_transcript()))
    assert completions.response_formats == ["json_schema", "json_schema"]


# ---- HTTP 层：GET /api/usage ----------------------------------------------


SHADOW_TEAM_ID = 999


@pytest.fixture
def usage_client(tmp_path, monkeypatch):
    """带 HTTP 客户端的夹具。

    注意：必须用**应用自身的**团队口令初始化夹具库，因为 TestClient 进入时会触发
    lifespan 再次 database.initialize(settings.parsed_team_tokens())；
    若夹具自造口令，会被这一步覆盖掉，导致所有请求 403。

    跨团队测试不依赖环境里配置几个团队，而是直接插入一个“影子团队”（id=999）。
    """
    from fastapi.testclient import TestClient

    import app.main as main_module

    teams = main_module.settings.parsed_team_tokens()
    database = Database(tmp_path / "usage-http.db")
    database.initialize(teams)
    monkeypatch.setattr(main_module, "database", database)
    token = teams[sorted(teams)[0]]
    with TestClient(main_module.app) as client:
        yield client, database, token


def _add_shadow_team(database: Database) -> None:
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "INSERT OR IGNORE INTO teams(id,name,token_hash,created_at) VALUES(?,?,?,?)",
            (SHADOW_TEAM_ID, "影子团队", "shadow-hash", "2026-01-01T00:00:00+00:00"),
        )


def test_usage_endpoint_returns_empty_arrays_instead_of_error(usage_client) -> None:
    client, _database, token = usage_client
    response = client.get("/api/usage", headers={"X-Access-Token": token})
    assert response.status_code == 200
    body = response.json()
    assert body["by_stage"] == []
    assert body["by_meeting"] == []
    assert body["totals"] == {
        "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "total_tokens": 0, "duration_ms": 0, "cost": None,
    }
    assert body["filters"]["price_configured"] is False


def test_usage_endpoint_aggregates_by_stage_and_returns_cost_null_without_price(usage_client) -> None:
    client, database, token = usage_client
    team_id = database.authenticate(token)
    assert team_id is not None
    database.create_meeting("mtg-usage", team_id, "用量会议", Path("/tmp/mtg-usage.wav"), None)
    database.record_llm_usage(
        team_id=team_id, stage="team_final_direct", model="deepseek-flash",
        meeting_id="mtg-usage", prompt_tokens=100, completion_tokens=20, total_tokens=120,
        duration_ms=900,
    )
    response = client.get(
        "/api/usage", params={"meeting_id": "mtg-usage"}, headers={"X-Access-Token": token}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["totals"]["total_tokens"] == 120
    assert body["totals"]["duration_ms"] == 900
    assert body["totals"]["cost"] is None  # 未配置单价时不硬编码金额
    assert body["by_stage"][0]["stage"] == "team_final_direct"
    assert body["by_meeting"][0]["meeting_id"] == "mtg-usage"


def test_usage_endpoint_estimates_cost_when_price_configured(usage_client, monkeypatch) -> None:
    client, database, token = usage_client
    import app.main as main_module

    monkeypatch.setattr(main_module.settings, "llm_price_prompt_per_1k", 1.0)
    monkeypatch.setattr(main_module.settings, "llm_price_completion_per_1k", 2.0)
    team_id = database.authenticate(token)
    database.record_llm_usage(
        team_id=team_id, stage="team_final_direct", model="m",
        prompt_tokens=1000, completion_tokens=500, total_tokens=1500, duration_ms=1,
    )
    body = client.get("/api/usage", headers={"X-Access-Token": token}).json()
    assert body["filters"]["price_configured"] is True
    assert body["totals"]["cost"] == pytest.approx(2.0)


def test_usage_endpoint_forbids_cross_team_meeting(usage_client) -> None:
    client, database, token = usage_client
    _add_shadow_team(database)
    database.create_meeting("mtg-b", SHADOW_TEAM_ID, "别团队的会议", Path("/tmp/mtg-b.wav"), None)
    response = client.get(
        "/api/usage", params={"meeting_id": "mtg-b"}, headers={"X-Access-Token": token}
    )
    assert response.status_code == 403
    assert response.json() == {
        "error": {"code": "team_forbidden", "message": "无权访问其他团队的会议"}
    }


def test_usage_endpoint_forbids_cross_team_project(usage_client) -> None:
    client, database, token = usage_client
    _add_shadow_team(database)
    database.create_project("proj-b", SHADOW_TEAM_ID, "别团队的项目")
    response = client.get(
        "/api/usage", params={"project_id": "proj-b"}, headers={"X-Access-Token": token}
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "team_forbidden"


def test_usage_endpoint_rejects_invalid_date_with_unified_error_shape(usage_client) -> None:
    client, _database, token = usage_client
    response = client.get(
        "/api/usage", params={"from": "不是日期"}, headers={"X-Access-Token": token}
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "invalid_args"
    assert "ISO8601" in body["error"]["message"]


def test_usage_endpoint_requires_team_token(usage_client) -> None:
    client, _database, _token = usage_client
    assert client.get("/api/usage").status_code == 403
    assert client.get("/api/usage", headers={"X-Access-Token": "wrong"}).status_code == 403


def test_usage_endpoint_not_found_meeting(usage_client) -> None:
    client, _database, token = usage_client
    response = client.get(
        "/api/usage", params={"meeting_id": "never-existed"}, headers={"X-Access-Token": token}
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_existing_endpoints_keep_their_detail_error_shape(usage_client) -> None:
    """老接口的错误结构保持不变（PRD P-6 契约不动）。"""
    client, _database, token = usage_client
    response = client.get(
        "/api/meetings/never-existed", headers={"X-Access-Token": token}
    )
    assert response.status_code == 404
    assert "detail" in response.json()
    assert "error" not in response.json()

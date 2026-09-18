"""阶段 5（P1-C）Agent 循环 + 权限 + 钩子的测试。

只新增，不修改既有断言。覆盖：
- 三级权限（readonly / write / host_owned）；
- 钩子：PreToolUse 拦截、钩子异常不影响循环、新增钩子不改 loop；
- 主循环：执行工具、回灌结果、无 tool_use 终止、max_steps 安全终止；
- 审计可回放且不含参数原文；
- Agent 循环**不写业务库**。
"""

import asyncio
import hashlib
import json
import os
import sqlite3
from types import SimpleNamespace

import pytest

# 与 tests/test_app.py 保持一致，避免独立运行时误用真实数据库。
os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

from app.agent.builtin_hooks import build_default_hooks
from app.agent.hooks import HookDecision, HookManager
from app.agent.loop import AgentLoop
from app.agent.permissions.policy import HOST_OWNED_TOOLS, PermissionPolicy
from app.agent.session import AgentLimits, AgentSession
from app.agent.tools.contract import success
from app.agent.tools.registry import ToolRegistry
from app.db import Database
from app.models import Transcript, TranscriptSegment

TEAM = "甲团队"
TOKEN = "test-access-token"
LIMITS = AgentLimits(max_steps=5, step_timeout_seconds=5, session_timeout_seconds=30)


# ---------------------------------------------------------------------------
# 假模型与响应
# ---------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, script) -> None:
        self.script = list(script)
        self.calls = 0

    async def step(self, messages, tool_specs, *, stage="agent_step", usage_context=None):
        self.calls += 1
        return self.script.pop(0)


def _resp_tool(call_id: str, name: str, arguments: dict):
    call = SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))])


def _resp_final(text: str = "已完成"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None))])


def _fake_registry(executed):
    registry = ToolRegistry()

    def get_transcript(team_id, **kwargs):
        executed.append(("get_transcript", kwargs))
        return success("概览", {"meeting_id": kwargs.get("meeting_id")})

    def extract_report(team_id, **kwargs):
        executed.append(("extract_report", kwargs))
        return success("已抽取", {"report": {"overview": "x"}, "coverage": {"segments_used": 1}})

    def validate_evidence(team_id, **kwargs):
        executed.append(("validate_evidence", kwargs))
        return success("通过", {"valid": True, "invalid": []})

    registry.register_tool(name="get_transcript", description="d", input_schema={"type": "object"}, handler=get_transcript)
    registry.register_tool(name="extract_report", description="d", input_schema={"type": "object"}, handler=extract_report)
    registry.register_tool(name="validate_evidence", description="d", input_schema={"type": "object"}, handler=validate_evidence)
    return registry


def _run(script, executed, *, limits=None, policy=None, hooks=None, session=None):
    registry = _fake_registry(executed)
    model = _FakeModel(script)
    loop = AgentLoop(
        registry=registry, model=model, policy=policy or PermissionPolicy(),
        hooks=hooks, limits=limits or LIMITS,
    )
    session = session or AgentSession(
        goal="g", system_prompt="s", usage_context={"team_id": 1, "meeting_id": "m1"}
    )
    asyncio.run(loop.run(session))
    return session, model


def _seed_db(tmp_path, name="p1c.db"):
    path = tmp_path / name
    db = Database(path)
    db.initialize({TEAM: TOKEN})
    team_id = db.authenticate(TOKEN)
    db.create_meeting("m1", team_id, "周会", tmp_path / "a.m4a")
    db.save_transcript(
        "m1", team_id,
        Transcript(
            duration_seconds=4.0,
            segments=[
                TranscriptSegment(start=0.0, end=2.0, text="决定本周上线"),
                TranscriptSegment(start=2.0, end=4.0, text="我去准备发布清单"),
            ],
        ),
    )
    return db, path, team_id


# ---------------------------------------------------------------------------
# 三级权限
# ---------------------------------------------------------------------------


def test_policy_allows_readonly_and_denies_write_by_default() -> None:
    policy = PermissionPolicy(tools_enabled="readonly")
    assert policy.decide("get_transcript", "readonly").allow
    write = policy.decide("save_report", "write")
    assert write.allow is False and write.decision == "deny_policy"
    unknown = policy.decide("mystery_tool", None)
    assert unknown.allow is False and unknown.decision == "deny_policy"


def test_policy_write_only_when_enabled_and_allowlisted() -> None:
    policy = PermissionPolicy(tools_enabled="save_report,get_transcript", write_tools_enabled=True)
    assert policy.decide("save_report", "write").allow
    assert policy.decide("get_transcript", "readonly").allow
    # 白名单模式下，未列入的只读工具也被拒
    assert policy.decide("list_members", "readonly").allow is False


@pytest.mark.parametrize("name", sorted(HOST_OWNED_TOOLS))
def test_all_host_owned_tools_are_never_allowed(name: str) -> None:
    policy = PermissionPolicy(tools_enabled="readonly", write_tools_enabled=True)
    decision = policy.decide(name, "readonly")
    assert decision.allow is False and decision.decision == "deny_host_owned"
    assert decision.code == "denied_host_owned"


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------


def test_loop_executes_tools_then_stops_on_no_tool_use() -> None:
    executed = []
    session, model = _run(
        [
            _resp_tool("c1", "get_transcript", {"meeting_id": "m1"}),
            _resp_tool("c2", "extract_report", {"meeting_id": "m1"}),
            _resp_tool("c3", "validate_evidence", {"meeting_id": "m1", "report_payload": {}}),
            _resp_final(),
        ],
        executed,
    )
    assert session.status == "completed"
    assert [name for name, _ in executed] == ["get_transcript", "extract_report", "validate_evidence"]
    assert len(session.records) == 3
    assert session.report == {"overview": "x"}
    assert session.validation == {"valid": True, "invalid": []}
    assert model.calls == 4  # 3 轮带工具 + 1 轮终止


def test_loop_stops_at_max_steps() -> None:
    executed = []
    script = [_resp_tool("c{}".format(i), "get_transcript", {"meeting_id": "m1"}) for i in range(10)]
    session, model = _run(script, executed, limits=AgentLimits(max_steps=2, step_timeout_seconds=5, session_timeout_seconds=30))
    assert session.status == "max_steps"
    assert model.calls == 2 and len(executed) == 2


def test_host_owned_tool_call_is_denied_and_not_executed() -> None:
    executed = []
    session, _ = _run(
        [_resp_tool("c1", "delete_meeting", {"meeting_id": "m1"}), _resp_final()], executed
    )
    assert executed == [], "host-owned 工具绝不能被执行"
    assert session.records[0].result_code == "denied_host_owned"
    tool_messages = [m for m in session.messages if m["role"] == "tool"]
    assert "denied_host_owned" in tool_messages[0]["content"]


def test_invalid_tool_arguments_are_reported_not_crashed() -> None:
    executed = []
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="get_transcript", arguments="{not json"))
    script = [SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))]), _resp_final()]
    session, _ = _run(script, executed)
    assert session.status == "completed"
    assert session.records[0].result_code == "invalid_args"
    assert executed == []


# ---------------------------------------------------------------------------
# 钩子
# ---------------------------------------------------------------------------


def test_pre_tool_use_deny_blocks_execution() -> None:
    executed = []
    hooks = HookManager()
    hooks.register(
        "PreToolUse",
        lambda event, payload: HookDecision(deny=True, reason="测试拦截") if payload["tool_name"] == "get_transcript" else None,
    )
    session, _ = _run(
        [_resp_tool("c1", "get_transcript", {"meeting_id": "m1"}), _resp_final()], executed, hooks=hooks
    )
    assert executed == []
    assert session.records[0].decision == "deny_policy"
    assert session.records[0].result_code == "denied_policy"


def test_hook_exception_does_not_break_the_loop() -> None:
    executed = []

    def boom(event, payload):
        raise RuntimeError("钩子故意报错")

    hooks = HookManager()
    hooks.register("PostToolUse", boom)
    session, _ = _run(
        [_resp_tool("c1", "get_transcript", {"meeting_id": "m1"}), _resp_final()], executed, hooks=hooks
    )
    assert session.status == "completed"
    assert session.records[0].result_code == "ok"
    assert executed


def test_extra_hook_needs_no_loop_change() -> None:
    executed = []
    seen = []
    hooks = HookManager()
    hooks.register("PostToolUse", lambda event, payload: seen.append(payload["tool_name"]))
    hooks.register("Stop", lambda event, payload: seen.append("stop"))
    _run([_resp_tool("c1", "get_transcript", {"meeting_id": "m1"}), _resp_final()], executed, hooks=hooks)
    assert seen == ["get_transcript", "stop"]


def test_loop_runs_without_any_hook() -> None:
    executed = []
    session, _ = _run([_resp_tool("c1", "get_transcript", {"meeting_id": "m1"}), _resp_final()], executed, hooks=None)
    assert session.status == "completed" and executed


# ---------------------------------------------------------------------------
# 审计
# ---------------------------------------------------------------------------


def test_audit_records_every_call_with_digest_only(tmp_path) -> None:
    db, _, team_id = _seed_db(tmp_path)
    executed = []
    hooks = build_default_hooks(db, audit_enabled=True)
    session = AgentSession(goal="g", system_prompt="s", usage_context={"team_id": team_id, "meeting_id": "m1"})
    _run([_resp_tool("c1", "get_transcript", {"meeting_id": "m1", "page": 1}), _resp_final()], executed, hooks=hooks, session=session)

    rows = db.list_agent_audit(team_id, session_id=session.session_id)
    assert len(rows) == 1
    row = rows[0]
    assert row["tool_name"] == "get_transcript" and row["result_code"] == "ok"
    assert row["decision"] == "allow"
    # args_digest 是 16 位十六进制摘要，且不含参数原文
    assert len(row["args_digest"]) == 16 and all(c in "0123456789abcdef" for c in row["args_digest"])
    assert row["args_digest"] == hashlib.sha256(
        json.dumps({"meeting_id": "m1", "page": 1}, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]
    # 审计不写转写原文
    assert "决定本周上线" not in json.dumps(rows, ensure_ascii=False)


def test_audit_also_records_denials(tmp_path) -> None:
    db, _, team_id = _seed_db(tmp_path)
    executed = []
    hooks = build_default_hooks(db, audit_enabled=True)
    session = AgentSession(goal="g", system_prompt="s", usage_context={"team_id": team_id, "meeting_id": "m1"})
    _run([_resp_tool("c1", "delete_meeting", {"meeting_id": "m1"}), _resp_final()], executed, hooks=hooks, session=session)
    rows = db.list_agent_audit(team_id, session_id=session.session_id)
    assert rows and rows[0]["decision"] == "deny_host_owned" and rows[0]["result_code"] == "denied_host_owned"


def test_audit_can_be_disabled(tmp_path) -> None:
    db, _, team_id = _seed_db(tmp_path)
    executed = []
    hooks = build_default_hooks(db, audit_enabled=False)
    session = AgentSession(goal="g", system_prompt="s", usage_context={"team_id": team_id, "meeting_id": "m1"})
    _run([_resp_tool("c1", "get_transcript", {"meeting_id": "m1"}), _resp_final()], executed, hooks=hooks, session=session)
    assert db.list_agent_audit(team_id, session_id=session.session_id) == []


# ---------------------------------------------------------------------------
# Agent 循环不写业务库
# ---------------------------------------------------------------------------


def test_agent_loop_never_writes_business_tables(tmp_path) -> None:
    db, path, team_id = _seed_db(tmp_path)
    executed = []
    hooks = build_default_hooks(db, audit_enabled=True)
    session = AgentSession(goal="g", system_prompt="s", usage_context={"team_id": team_id, "meeting_id": "m1"})
    _run(
        [
            _resp_tool("c1", "get_transcript", {"meeting_id": "m1"}),
            _resp_tool("c2", "extract_report", {"meeting_id": "m1"}),
            _resp_final(),
        ],
        executed, hooks=hooks, session=session,
    )
    # 报告与行动项都没有被 Agent 写入（写库由确定性代码在循环后完成）
    assert db.get_report("m1", team_id) is None
    assert db.list_action_items(team_id) == []
    # 审计有行，证明跑过
    assert db.list_agent_audit(team_id, session_id=session.session_id)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM reports").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# AGENT_MODE=agent 的接线：产出后先过引文校验，本函数本身不落库
# ---------------------------------------------------------------------------

REPORT_PAYLOAD = {
    "suggested_title": "上线准备会",
    "overview": "会议决定本周上线。",
    "meeting_points": ["决定本周上线"],
    "decisions": [
        {"content": "本周上线", "decision_maker": "小王", "evidence": {"quote": "决定本周上线", "timestamp": "00:00:01"}}
    ],
    "action_items": [],
    "unresolved_issues": [],
    "speaker_stats_note": "",
}


def test_agent_trace_endpoint_shape_and_team_scope(tmp_path, monkeypatch) -> None:
    import app.main as main_module
    from fastapi.testclient import TestClient

    path = tmp_path / "trace.db"
    db = Database(path)
    # 用与 settings 相同的团队名初始化，避免 TestClient 启动时 lifespan 重建 token→team 映射后不一致。
    db.initialize(main_module.settings.parsed_team_tokens())
    team_a = db.authenticate(TOKEN)
    db.create_meeting("m1", team_a, "周会", tmp_path / "a.m4a")
    db.record_agent_audit(
        team_id=team_a, session_id="s1", meeting_id="m1", step=1,
        tool_name="get_transcript", decision="allow", result_code="ok", duration_ms=3,
    )
    monkeypatch.setattr(main_module, "database", db)

    with TestClient(main_module.app) as client:
        ok = client.get("/api/meetings/m1/agent-trace", headers={"X-Access-Token": TOKEN})
        assert ok.status_code == 200
        body = ok.json()
        assert body["total_calls"] == 1 and body["tool_names"] == ["get_transcript"]
        assert body["steps"][0]["result_code"] == "ok"
        # 跨团队读同一会议 → 403
        other = client.get("/api/meetings/m1/agent-trace", headers={"X-Access-Token": "other-team-token"})
        assert other.status_code == 403


def test_agent_mode_run_validates_evidence_before_return(tmp_path, monkeypatch) -> None:
    import app.main as main_module
    from app.tasks import TaskProcessingError

    db, _, team_id = _seed_db(tmp_path, "wiring.db")
    monkeypatch.setattr(main_module, "database", db)

    class _FakeRunner:
        def __init__(self, payload):
            self.payload = payload

        async def run_meeting(self, *, meeting_id, team_id, session_id=None):
            return SimpleNamespace(report=self.payload, status="completed")

    monkeypatch.setattr(main_module, "meeting_agent", _FakeRunner(REPORT_PAYLOAD))
    report = asyncio.run(main_module._run_agent_report("m1", team_id))
    assert report.suggested_title == "上线准备会"
    assert db.get_report("m1", team_id) is None, "本步只校验，不落库"

    # 没产出报告 → 明确失败
    monkeypatch.setattr(main_module, "meeting_agent", _FakeRunner(None))
    with pytest.raises(TaskProcessingError):
        asyncio.run(main_module._run_agent_report("m1", team_id))

    # 引文对不上 → 拒绝落库
    bad = dict(REPORT_PAYLOAD)
    bad["decisions"] = [
        {"content": "本周上线", "decision_maker": "小王", "evidence": {"quote": "不存在的原话", "timestamp": "00:00:01"}}
    ]
    monkeypatch.setattr(main_module, "meeting_agent", _FakeRunner(bad))
    with pytest.raises(TaskProcessingError):
        asyncio.run(main_module._run_agent_report("m1", team_id))

"""阶段 7（P1-E）上下文治理（s08）+ 任务持久化（s10）的测试。

覆盖：
- R-P1-8：任务状态落库、重启后 GET 可读、音频还在则续跑、音频已清理则明确失败、跨团队隔离；
- R-P1-7：上下文超预算触发压缩且结构不被破坏、并发上限受约束。
"""

import asyncio
import json
import os
from types import SimpleNamespace


os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

from app.agent.context import ContextManager, estimate_tokens
from app.agent.loop import AgentLoop
from app.agent.permissions.policy import PermissionPolicy
from app.agent.session import AgentLimits, AgentSession
from app.agent.tools.contract import success
from app.agent.tools.registry import ToolRegistry
from app.config import Settings
from app.db import Database
from app.llm import LLMAnalyzer
from app.tasks import TaskManager

TEAM = "甲团队"
OTHER = "乙团队"
TOKEN = "test-access-token"
OTHER_TOKEN = "other-team-token"
LIMITS = AgentLimits(max_steps=6, step_timeout_seconds=5, session_timeout_seconds=30)


def _seed_db(tmp_path, name="p1e.db"):
    path = tmp_path / name
    db = Database(path)
    db.initialize({TEAM: TOKEN, OTHER: OTHER_TOKEN})
    team_id = db.authenticate(TOKEN)
    other_id = db.authenticate(OTHER_TOKEN)
    return db, path, team_id, other_id


async def _noop_processor(path, progress):
    return None


def _manager(db, processor=_noop_processor, **kwargs):
    return TaskManager(
        processor, timeout_seconds=5, retention_seconds=60, database=db, **kwargs
    )


# ---------------------------------------------------------------------------
# R-P1-8 任务持久化
# ---------------------------------------------------------------------------


def test_task_state_is_persisted_and_readable_after_memory_cleared(tmp_path) -> None:
    db, _, team_id, _ = _seed_db(tmp_path)
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x")
    manager = _manager(db)

    async def scenario():
        await manager.start()
        await manager.submit(audio, "req-1", task_id="t1", team_id=team_id)
        await asyncio.sleep(0.05)
        await manager.stop()

    asyncio.run(scenario())
    row = db.get_agent_task("t1", team_id)
    assert row is not None and row["status"] == "完成" and row["request_id"] == "req-1"

    manager.records.clear()  # 模拟内存缓存过期
    status = manager.get("t1", team_id)
    assert status is not None
    assert status.task_id == "t1" and status.request_id == "req-1" and status.status == "完成"


def test_task_read_is_team_scoped_after_restart(tmp_path) -> None:
    db, _, team_id, other_id = _seed_db(tmp_path)
    db.upsert_agent_task("t1", team_id, "完成", meeting_id="t1", message="复盘完成")
    manager = _manager(db)
    assert manager.get("t1", team_id) is not None
    assert manager.get("t1", other_id) is None


def test_hydrate_resumes_task_when_audio_still_exists(tmp_path) -> None:
    db, _, team_id, _ = _seed_db(tmp_path)
    audio = tmp_path / "resume.m4a"
    audio.write_bytes(b"audio")
    db.create_meeting("t-resume", team_id, "周会", audio)
    db.upsert_agent_task(
        "t-resume", team_id, "排队中", meeting_id="t-resume",
        message="排队中", request_id="r1", audio_path=str(audio),
    )
    processed = []

    async def processor(path, progress):
        processed.append(str(path))
        return None

    manager = _manager(db, processor)
    manager.hydrate()
    assert audio in manager.resume_paths(), "待续跑任务的音频必须被保留"

    async def scenario():
        await manager.start()
        await asyncio.sleep(0.1)
        await manager.stop()

    asyncio.run(scenario())
    assert processed == [str(audio)], "重启后应继续处理"
    assert db.get_agent_task("t-resume", team_id)["status"] == "完成"


def test_hydrate_marks_failed_when_audio_already_deleted(tmp_path) -> None:
    db, _, team_id, _ = _seed_db(tmp_path)
    missing = tmp_path / "gone.m4a"
    db.upsert_agent_task(
        "t-gone", team_id, "转写中", meeting_id="t-gone", audio_path=str(missing),
    )
    manager = _manager(db)
    manager.hydrate()
    row = db.get_agent_task("t-gone", team_id)
    assert row["status"] == "失败" and row["error_code"] == 503
    assert "重新提交" in (row["error"] or "")
    assert missing not in manager.resume_paths()
    assert manager.get("t-gone", team_id).error_code == 503


def test_task_status_contract_is_unchanged(tmp_path) -> None:
    db, _, team_id, _ = _seed_db(tmp_path)
    audio = tmp_path / "b.m4a"
    audio.write_bytes(b"x")
    manager = _manager(db)

    async def scenario():
        await manager.start()
        accepted = await manager.submit(audio, "req-2", task_id="t2", team_id=team_id, long_meeting=True)
        await asyncio.sleep(0.05)
        await manager.stop()
        return accepted

    accepted = asyncio.run(scenario())
    assert accepted.task_id == "t2" and accepted.long_meeting is True
    status = manager.get("t2", team_id)
    # TaskStatus 契约字段齐全
    for field in ("task_id", "request_id", "status", "queue_position", "message", "long_meeting"):
        assert hasattr(status, field)


# ---------------------------------------------------------------------------
# R-P1-7 上下文治理与并发上限
# ---------------------------------------------------------------------------


def _big_tool_content(size: int = 2000) -> str:
    return json.dumps(
        {"ok": True, "code": "ok", "summary": "摘要", "payload": {"data": "甲" * size}},
        ensure_ascii=False,
    )


def test_context_manager_compacts_only_when_over_budget() -> None:
    small = AgentSession(goal="g", system_prompt="s")
    assert ContextManager(100000).maybe_compact(small) is None

    session = AgentSession(goal="g", system_prompt="s")
    for index in range(20):
        session.messages.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "c{}".format(index), "type": "function", "function": {"name": "t", "arguments": "{}"}}],
        })
        session.messages.append({"role": "tool", "tool_call_id": "c{}".format(index), "content": _big_tool_content()})
    before = estimate_tokens(session.messages)
    note = ContextManager(2000).maybe_compact(session)
    assert note and note.startswith("compact=")
    assert estimate_tokens(session.messages) < before
    # 结构必须完好：tool 消息仍带 tool_call_id
    assert all("tool_call_id" in m for m in session.messages if m["role"] == "tool")


def test_loop_records_compaction_note() -> None:
    registry = ToolRegistry()

    def heavy(team_id, **kwargs):
        return success("大结果", {"data": "甲" * 3000})

    registry.register_tool(name="get_transcript", description="d", input_schema={}, handler=heavy)

    class _Model:
        def __init__(self):
            self.n = 0

        async def step(self, messages, tool_specs, *, stage="agent_step", usage_context=None):
            self.n += 1
            if self.n > 3:
                return _final()
            return _tool("c{}".format(self.n))

    session = AgentSession(goal="g", system_prompt="s", usage_context={"team_id": 1, "meeting_id": "m1"})
    loop = AgentLoop(
        registry=registry, model=_Model(), policy=PermissionPolicy(),
        limits=LIMITS, context_manager=ContextManager(300),
    )
    asyncio.run(loop.run(session))
    assert loop.compactions >= 1
    assert any("compact=" in note for note in session.notes)


def test_llm_concurrency_is_bounded() -> None:
    settings = Settings(TEAM_TOKENS="test-access-token", LLM_MAX_CONCURRENCY=2)
    analyzer = LLMAnalyzer(settings)
    active = {"now": 0, "peak": 0}

    async def job(index: int) -> int:
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        await asyncio.sleep(0.02)
        active["now"] -= 1
        return index

    results = asyncio.run(analyzer._gather_bounded([job(i) for i in range(6)]))
    assert results == list(range(6))
    assert active["peak"] <= 2, "并发必须受 LLM_MAX_CONCURRENCY 约束"


def test_llm_concurrency_default() -> None:
    assert Settings.model_fields["llm_max_concurrency"].default == 4


def _tool(call_id: str, name: str = "get_transcript", arguments=None):
    call = SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments or {})))
    message = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _final(text: str = "已完成"):
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])

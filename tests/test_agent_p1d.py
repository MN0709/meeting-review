"""阶段 6（P1-D）任务规划 + 最小完成判定 + 集成 Harness 的测试。

覆盖：
- s05：todo_write 工具、计划落库与跨重启可读、Planner reminder；
- s17：规则评估器判定、循环「未通过则继续 / 超限交还人 / 评估器报错交还人」；
- s15：harness 动态提示词含全部工具名与计划状态；组件之间无横向 import。
"""

import ast
import asyncio
import json
import os
import pathlib
from types import SimpleNamespace


os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

import app.agent as agent_pkg
from app.agent.goal import RuleGoalEvaluator
from app.agent.harness import AgentHarness
from app.agent.loop import AgentLoop
from app.agent.permissions.policy import PermissionPolicy
from app.agent.planning import Planner, register_planning_tools
from app.agent.session import AgentLimits, AgentSession, GoalJudgment, bind_session
from app.agent.tools.catalog import register_all_tools
from app.agent.tools.contract import success
from app.agent.tools.registry import ToolRegistry
from app.db import Database
from app.models import Transcript, TranscriptSegment

TEAM = "甲团队"
TOKEN = "test-access-token"
LIMITS = AgentLimits(max_steps=6, step_timeout_seconds=5, session_timeout_seconds=30, max_judge_retries=2)


def _seed_db(tmp_path, name="p1d.db"):
    path = tmp_path / name
    db = Database(path)
    db.initialize({TEAM: TOKEN})
    team_id = db.authenticate(TOKEN)
    db.create_meeting("m1", team_id, "周会", tmp_path / "a.m4a")
    db.save_transcript(
        "m1", team_id,
        Transcript(duration_seconds=4.0, segments=[TranscriptSegment(start=0.0, end=2.0, text="决定本周上线")]),
    )
    return db, path, team_id


# ---------------------------------------------------------------------------
# s05 任务规划
# ---------------------------------------------------------------------------


def test_todo_write_requires_a_session(tmp_path) -> None:
    db, _, team_id = _seed_db(tmp_path)
    registry = ToolRegistry()
    register_planning_tools(registry, db)
    result = registry.get("todo_write").handler(team_id, items=[{"task": "x"}])
    assert result.ok is False and result.code == "tool_error"


def test_todo_write_persists_plan_and_survives_restart(tmp_path) -> None:
    db, path, team_id = _seed_db(tmp_path)
    registry = ToolRegistry()
    register_planning_tools(registry, db)
    session = AgentSession(goal="g", system_prompt="s", usage_context={"team_id": team_id, "meeting_id": "m1"})
    with bind_session(session):
        result = registry.get("todo_write").handler(
            team_id,
            items=[{"task": "读概览", "status": "in_progress"}, {"task": "抽报告", "status": "pending"}],
        )
    assert result.ok and session.todo[0]["task"] == "读概览"
    assert session.last_plan_update_step == session.steps + 1

    task = db.get_agent_task("m1", team_id)
    assert json.loads(task["progress_json"])["todo"][0]["task"] == "读概览"

    # 「重启」：重新打开同一个库，计划仍在
    reopened = Database(path)
    reopened.initialize({TEAM: TOKEN})
    persisted = reopened.get_agent_task("m1", team_id)
    assert json.loads(persisted["progress_json"])["todo"][1]["task"] == "抽报告"


def test_todo_write_validates_items(tmp_path) -> None:
    db, _, team_id = _seed_db(tmp_path)
    registry = ToolRegistry()
    register_planning_tools(registry, db)
    session = AgentSession(goal="g", system_prompt="s", usage_context={"team_id": team_id, "meeting_id": "m1"})
    with bind_session(session):
        bad_status = registry.get("todo_write").handler(team_id, items=[{"task": "x", "status": "doneish"}])
        assert bad_status.code == "invalid_args"
        empty = registry.get("todo_write").handler(team_id, items=[])
        assert empty.code == "invalid_args"
        not_list = registry.get("todo_write").handler(team_id, items={"task": "x"})
        assert not_list.code == "invalid_args"


def test_planner_reminder_counter() -> None:
    session = AgentSession(goal="g", system_prompt="s")
    session.todo = [{"task": "x", "status": "pending"}]
    session.last_plan_update_step = 1
    planner = Planner(reminder_every=3)
    assert planner.reminder(session, 2) is None
    assert planner.reminder(session, 3) is None       # 3-1=2
    reminder = planner.reminder(session, 4)           # 4-1=3 → 触发
    assert reminder and "更新计划" in reminder
    assert planner.reminder(session, 4) is None       # 同一轮不重复


def test_loop_appends_reminder_to_tool_result() -> None:
    executed = []

    def get_transcript(team_id, **kwargs):
        executed.append(kwargs)
        return success("概览", {"meeting_id": "m1"})

    registry = ToolRegistry()
    registry.register_tool(name="get_transcript", description="d", input_schema={}, handler=get_transcript)

    class _Model:
        def __init__(self):
            self.n = 0

        async def step(self, messages, tool_specs, *, stage="agent_step", usage_context=None):
            self.n += 1
            if self.n > 4:
                return _final()
            return _tool("c{}".format(self.n), "get_transcript", {"meeting_id": "m1"})

    session = AgentSession(goal="g", system_prompt="s", usage_context={"team_id": 1, "meeting_id": "m1"})
    session.todo = [{"task": "读概览", "status": "pending"}]
    loop = AgentLoop(registry=registry, model=_Model(), policy=PermissionPolicy(), planner=Planner(), limits=LIMITS)
    asyncio.run(loop.run(session))
    tool_messages = [m for m in session.messages if m["role"] == "tool"]
    assert any("提醒：已连续" in m["content"] for m in tool_messages)


# ---------------------------------------------------------------------------
# s17 最小完成判定
# ---------------------------------------------------------------------------


def test_rule_evaluator_rejects_empty_results() -> None:
    evaluator = RuleGoalEvaluator()
    empty = AgentSession(goal="g", system_prompt="s")
    assert evaluator.evaluate(empty).done is False

    no_overview = AgentSession(goal="g", system_prompt="s")
    no_overview.report = {"overview": "", "meeting_points": ["x"]}
    assert evaluator.evaluate(no_overview).done is False

    no_content = AgentSession(goal="g", system_prompt="s")
    no_content.report = {"overview": "有总览", "meeting_points": [], "decisions": [], "action_items": [], "unresolved_issues": []}
    judgment = evaluator.evaluate(no_content)
    assert judgment.done is False and judgment.missing

    bad_evidence = AgentSession(goal="g", system_prompt="s")
    bad_evidence.report = {"overview": "有总览", "decisions": [{"content": "x"}], "meeting_points": []}
    bad_evidence.validation = {"valid": False, "invalid": [{"quote": "错话", "nearest_segment_preview": "原话"}]}
    assert evaluator.evaluate(bad_evidence).done is False

    good = AgentSession(goal="g", system_prompt="s")
    good.report = {"overview": "有总览", "decisions": [{"content": "x"}], "meeting_points": [], "action_items": [], "unresolved_issues": []}
    assert evaluator.evaluate(good).done is True


def _tool(call_id, name, arguments):
    call = SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))])


def _final(text="已完成"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None))])


class _ScriptModel:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    async def step(self, messages, tool_specs, *, stage="agent_step", usage_context=None):
        self.calls += 1
        return self.script.pop(0)


class _StubJudge:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls = 0

    async def judge(self, session):
        self.calls += 1
        verdict = self.verdicts.pop(0)
        if isinstance(verdict, Exception):
            raise verdict
        return verdict


def _loop_with(script, judge, registry=None):
    registry = registry or ToolRegistry()
    session = AgentSession(goal="g", system_prompt="s", usage_context={"team_id": 1, "meeting_id": "m1"})
    loop = AgentLoop(registry=registry, model=_ScriptModel(script), policy=PermissionPolicy(), goal_judge=judge, limits=LIMITS)
    asyncio.run(loop.run(session))
    return session


def test_loop_continues_when_judge_says_not_done() -> None:
    judge = _StubJudge([
        GoalJudgment(False, "报告为空", ["请调用 extract_report"]),
        GoalJudgment(True, "已达标", []),
    ])
    session = _loop_with([_final("先停一下"), _final("真的完成了")], judge)
    assert session.status == "completed"
    assert len(session.judgments) == 2
    assert any("独立评估认为尚未完成" in m["content"] for m in session.messages if m["role"] == "user")


def test_loop_gives_up_to_human_after_max_retries() -> None:
    judge = _StubJudge([GoalJudgment(False, "还是不行", ["缺内容"]) for _ in range(5)])
    session = _loop_with([_final() for _ in range(6)], judge)
    assert session.status == "needs_human"
    assert session.judge_retries == LIMITS.max_judge_retries


def test_loop_returns_to_human_when_judge_errors() -> None:
    judge = _StubJudge([RuntimeError("评估器炸了")])
    session = _loop_with([_final()], judge)
    assert session.status == "needs_human"
    assert session.judgments[-1].error is True


def test_loop_without_judge_keeps_old_behaviour() -> None:
    session = _loop_with([_final()], judge=None)
    assert session.status == "completed"


# ---------------------------------------------------------------------------
# s15 集成 Harness
# ---------------------------------------------------------------------------


def test_harness_prompt_contains_all_tools_and_plan(tmp_path) -> None:
    db, _, _ = _seed_db(tmp_path)
    registry = register_all_tools(ToolRegistry(), db)  # 无 analyzer → 只读 + 规划
    harness = AgentHarness(registry=registry, analyzer=None, policy=PermissionPolicy(), limits=LIMITS)
    prompt = harness.build_prompt()
    for name in registry.names():
        assert name in prompt, "系统提示词必须含当前全部工具名：{}".format(name)

    session = AgentSession(goal="g", system_prompt="s")
    session.todo = [{"task": "抽决策", "status": "in_progress"}]
    dynamic = harness.build_prompt(session)
    assert "抽决策" in dynamic and "进行中" in dynamic


def test_harness_prompt_rebuilds_with_step_budget(tmp_path) -> None:
    db, _, _ = _seed_db(tmp_path)
    registry = register_all_tools(ToolRegistry(), db)
    harness = AgentHarness(
        registry=registry, analyzer=None, policy=PermissionPolicy(),
        limits=AgentLimits(max_steps=7), context_budget_tokens=12345,
    )
    session = AgentSession(goal="g", system_prompt="s")
    session.steps = 3
    prompt = harness.build_prompt(session)
    assert "12345" in prompt and "最多 7 步" in prompt


def test_agent_components_do_not_import_each_other() -> None:
    base = pathlib.Path(agent_pkg.__file__).parent

    def agent_imports(relative: str):
        tree = ast.parse((base / relative).read_text(encoding="utf-8"))
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("app.agent"):
                found.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("app.agent"):
                        found.add(alias.name)
        return found

    # 叶子组件不得反向依赖循环/装配
    for leaf in ("hooks.py", "permissions/policy.py"):
        imports = agent_imports(leaf)
        assert not (imports & {"app.agent.loop", "app.agent.harness"}), "{} 不得反向 import".format(leaf)
    # 装配方向单向：loop 不 import harness
    assert "app.agent.harness" not in agent_imports("loop.py")
    # 主循环唯一
    assert (base / "loop.py").read_text(encoding="utf-8").count("while True") == 1
    for path in base.rglob("*.py"):
        if path.name != "loop.py":
            assert "while True" not in path.read_text(encoding="utf-8"), "{} 不应有主循环".format(path)

"""阶段 4（P1-B）11 个工具的测试。

只新增，不修改既有断言。覆盖：
- 11 个工具全部为 readonly，且可被枚举；
- 每个只读工具的契约与跨团队 team_forbidden（不返回任何数据）；
- validate_evidence 与 app/llm.py 既有引文校验逐例对拍；
- extract_report / suggest_title 复用现有 analyze_team（用假分析器）；
- 8KB 体积护栏。
"""

import asyncio
import sqlite3
from types import SimpleNamespace

import pytest

from app.agent.tools.catalog import (
    CAPABILITY_TOOL_NAMES,
    READONLY_TOOL_NAMES,
    register_all_tools,
)
from app.agent.tools.registry import ToolRegistry
from app.agent.tools.support import payload_bytes
from app.db import Database
from app.llm import validate_team_evidence
from app.models import (
    ActionItem,
    DecisionItem,
    EvidenceQuote,
    TeamMeetingReport,
    Transcript,
    TranscriptSegment,
)

TEAM_A = "甲团队"
TEAM_B = "乙团队"
TOKEN_A = "test-access-token"
TOKEN_B = "other-team-token"
ALL_TOOL_NAMES = sorted(READONLY_TOOL_NAMES + CAPABILITY_TOOL_NAMES)

REPORT = TeamMeetingReport(
    suggested_title="上线准备会",
    overview="会议决定本周上线，并安排了发布准备。",
    meeting_points=["决定本周上线"],
    decisions=[
        DecisionItem(
            content="本周上线",
            decision_maker="小王",
            evidence=EvidenceQuote(quote="决定本周上线", timestamp="00:00:01"),
        )
    ],
    action_items=[ActionItem(task="准备发布清单", owner="小李", deadline="周五")],
    unresolved_issues=[],
)

TRANSCRIPT = Transcript(
    duration_seconds=4.0,
    segments=[
        TranscriptSegment(start=0.0, end=2.0, speaker_label="A", text="决定本周上线"),
        TranscriptSegment(start=2.0, end=4.0, speaker_label="B", text="我来准备发布清单"),
    ],
)


class _FakeAnalyzer:
    """替代真实 LLM：返回固定报告，并记录调用次数。"""

    def __init__(self, report: TeamMeetingReport = REPORT) -> None:
        self.report = report
        self.settings = SimpleNamespace(transcript_chunk_chars=6000)
        self.calls = 0

    async def analyze_team(self, transcript: Transcript) -> TeamMeetingReport:
        self.calls += 1
        return self.report


def _seed(tmp_path, name: str = "p1b.db"):
    path = tmp_path / name
    db = Database(path)
    db.initialize({TEAM_A: TOKEN_A, TEAM_B: TOKEN_B})
    team_a = db.authenticate(TOKEN_A)
    team_b = db.authenticate(TOKEN_B)
    assert team_a is not None and team_b is not None
    db.create_project("proj1", team_a, "产品项目组")
    db.create_meeting("m1", team_a, "周会", tmp_path / "a.m4a", project_id="proj1")
    db.save_transcript("m1", team_a, TRANSCRIPT)
    db.update_status("m1", team_a, "完成")
    db.save_report("m1", team_a, REPORT)
    # 另一团队的会议与转写，用于验证隔离
    db.create_meeting("m2", team_b, "乙会议", tmp_path / "b.m4a")
    db.save_transcript(
        "m2", team_b,
        Transcript(
            duration_seconds=3.0,
            segments=[TranscriptSegment(start=0.0, end=3.0, speaker_label="A", text="另一个团队的会议内容")],
        ),
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO members(team_id,name,role,is_key_decision_maker,created_at) VALUES(?,?,?,?,?)",
            (team_a, "小王", "组员", 1, "2026-09-18T00:00:00+00:00"),
        )
    analyzer = _FakeAnalyzer()
    registry = register_all_tools(ToolRegistry(), db, analyzer)
    return db, registry, analyzer, team_a, team_b


def _call(registry: ToolRegistry, name: str, team_id: int, **kwargs):
    return registry.get(name).handler(team_id, **kwargs)


def _call_async(registry: ToolRegistry, name: str, team_id: int, **kwargs):
    return asyncio.run(registry.get(name).handler(team_id, **kwargs))


# ---------------------------------------------------------------------------
# 目录与权限
# ---------------------------------------------------------------------------


def test_all_eleven_tools_registered_and_readonly(tmp_path) -> None:
    _, registry, _, _, _ = _seed(tmp_path)
    assert registry.names() == ALL_TOOL_NAMES
    assert len(registry) == 11
    assert {spec.permission_level for spec in registry.list_tools()} == {"readonly"}
    for spec in registry.list_tools():
        assert spec.input_schema.get("type") == "object"
        assert spec.description


# ---------------------------------------------------------------------------
# 只读工具
# ---------------------------------------------------------------------------


def test_get_transcript_pages_and_blocks_cross_team(tmp_path) -> None:
    _, registry, _, team_a, team_b = _seed(tmp_path)
    result = _call(registry, "get_transcript", team_a, meeting_id="m1")
    assert result.ok and result.payload["total_segments"] == 2
    assert result.payload["segments"][0]["text"] == "决定本周上线"
    assert result.truncated is False

    paged = _call(registry, "get_transcript", team_a, meeting_id="m1", page=1, page_size=1)
    assert paged.ok and paged.payload["pages"] == 2 and paged.truncated and paged.next_hint

    forbidden = _call(registry, "get_transcript", team_b, meeting_id="m1")
    assert forbidden.ok is False and forbidden.code == "team_forbidden" and forbidden.payload is None

    missing = _call(registry, "get_transcript", team_a, meeting_id="nope")
    assert missing.code == "not_found"


def test_search_transcript_scoped_and_like_fallback(tmp_path) -> None:
    _, registry, _, team_a, team_b = _seed(tmp_path)
    hit = _call(registry, "search_transcript", team_a, query="本周上线")
    assert hit.ok and hit.payload["hits"], "FTS5(trigram) 应能命中中文子串"
    assert hit.payload["hits"][0]["meeting_title"] == "周会"

    # 短查询（< 3 字）走 LIKE 回退
    short = _call(registry, "search_transcript", team_a, query="清单")
    assert short.ok and short.payload["hits"]

    # 另一团队搜不到甲团队的转写
    isolated = _call(registry, "search_transcript", team_b, query="本周上线")
    assert isolated.ok and isolated.payload["hits"] == []

    # 限定到别人团队的会议 → 直接拒绝
    forbidden = _call(registry, "search_transcript", team_b, query="本周上线", meeting_id="m1")
    assert forbidden.code == "team_forbidden"

    assert _call(registry, "search_transcript", team_a, query="   ").code == "invalid_args"


def test_read_segment_window_by_timestamp_and_index(tmp_path) -> None:
    _, registry, _, team_a, team_b = _seed(tmp_path)
    by_time = _call(registry, "read_segment_window", team_a, meeting_id="m1", timestamp=1.0)
    assert by_time.ok and len(by_time.payload["segments"]) == 2

    by_index = _call(registry, "read_segment_window", team_a, meeting_id="m1", segment_index=1, before=0, after=0)
    assert by_index.ok and by_index.payload["segments"][0]["text"] == "我来准备发布清单"

    assert _call(registry, "read_segment_window", team_a, meeting_id="m1").code == "invalid_args"
    assert _call(registry, "read_segment_window", team_b, meeting_id="m1", timestamp=1.0).code == "team_forbidden"


def test_get_report_and_missing_report(tmp_path) -> None:
    _, registry, _, team_a, team_b = _seed(tmp_path)
    result = _call(registry, "get_report", team_a, meeting_id="m1")
    assert result.ok and result.payload["suggested_title"] == "上线准备会"
    assert _call(registry, "get_report", team_b, meeting_id="m2").code == "not_found"
    assert _call(registry, "get_report", team_b, meeting_id="m1").code == "team_forbidden"


def test_get_project_memory(tmp_path) -> None:
    _, registry, _, team_a, team_b = _seed(tmp_path)
    result = _call(registry, "get_project_memory", team_a, project_id="proj1")
    assert result.ok and result.payload["project_name"] == "产品项目组"
    assert result.payload["recent_meetings"][0]["id"] == "m1"
    assert _call(registry, "get_project_memory", team_a, project_id="nope").code == "not_found"
    assert _call(registry, "get_project_memory", team_b, project_id="proj1").code == "team_forbidden"


def test_list_action_items_with_filter(tmp_path) -> None:
    _, registry, _, team_a, _ = _seed(tmp_path)
    result = _call(registry, "list_action_items", team_a, project_id="proj1")
    assert result.ok and result.payload["items"][0]["task"] == "准备发布清单"
    assert result.payload["items"][0]["source_meeting"]["title"] == "周会"

    pending = _call(registry, "list_action_items", team_a, status="待确认")
    assert pending.ok and pending.payload["items"]
    done = _call(registry, "list_action_items", team_a, status="已完成")
    assert done.ok and done.payload["items"] == []
    assert _call(registry, "list_action_items", team_a, status="乱填").code == "invalid_args"


def test_list_members_is_team_scoped(tmp_path) -> None:
    _, registry, _, team_a, team_b = _seed(tmp_path)
    result = _call(registry, "list_members", team_a)
    assert result.ok and result.payload["members"][0]["name"] == "小王"
    assert _call(registry, "list_members", team_b).payload["members"] == []


def test_list_meetings_and_date_validation(tmp_path) -> None:
    _, registry, _, team_a, _ = _seed(tmp_path)
    result = _call(registry, "list_meetings", team_a, project_id="proj1")
    assert result.ok and result.payload["meetings"][0]["id"] == "m1"
    assert _call(registry, "list_meetings", team_a, project_id="proj1", date_from="不是时间").code == "invalid_args"


# ---------------------------------------------------------------------------
# validate_evidence
# ---------------------------------------------------------------------------


def test_validate_evidence_valid_and_structured_invalid(tmp_path) -> None:
    _, registry, _, team_a, _ = _seed(tmp_path)
    valid = _call(registry, "validate_evidence", team_a, meeting_id="m1", report_payload=REPORT.model_dump())
    assert valid.ok and valid.payload["valid"] is True and valid.payload["invalid"] == []

    broken = REPORT.model_dump()
    broken["decisions"][0]["evidence"] = {"quote": "这句原话不存在", "timestamp": "00:00:01"}
    result = _call(registry, "validate_evidence", team_a, meeting_id="m1", report_payload=broken)
    assert result.ok and result.payload["valid"] is False
    entry = result.payload["invalid"][0]
    assert entry["quote"] == "这句原话不存在"
    assert entry["nearest_segment_preview"], "必须给出最近原文，模型才能定向回读"

    assert _call(registry, "validate_evidence", team_a, meeting_id="m1", report_payload={"bad": 1}).code == "invalid_args"


@pytest.mark.parametrize(
    "quote,timestamp",
    [
        ("决定本周上线", "00:00:01"),   # 正常
        ("决定本周上线", "00:00:00"),   # 容差内
        ("决定本周上线", "00:00:30"),   # 时间戳超范围
        ("这句不存在", "00:00:01"),      # 子串不匹配
        ("我来准备发布清单", "00:00:03"),  # 第二段正常
    ],
)
def test_validate_evidence_matches_existing_validator(tmp_path, quote: str, timestamp: str) -> None:
    """与 app/llm.py 既有引文校验逐例对拍，保证行为一致（R-P1-4）。"""
    _, registry, _, team_a, _ = _seed(tmp_path)
    payload = REPORT.model_dump()
    payload["decisions"][0]["evidence"] = {"quote": quote, "timestamp": timestamp}
    result = _call(registry, "validate_evidence", team_a, meeting_id="m1", report_payload=payload)
    tool_says_valid = result.payload["valid"]

    try:
        validate_team_evidence(TeamMeetingReport.model_validate(payload), TRANSCRIPT.segments)
        existing_says_valid = True
    except ValueError:
        existing_says_valid = False
    assert tool_says_valid == existing_says_valid


# ---------------------------------------------------------------------------
# 能力工具（复用现有 analyze_team，用假分析器）
# ---------------------------------------------------------------------------


def test_extract_report_reuses_analyzer_and_reports_coverage(tmp_path) -> None:
    _, registry, analyzer, team_a, _ = _seed(tmp_path)
    result = _call_async(registry, "extract_report", team_a, meeting_id="m1")
    assert result.ok and analyzer.calls == 1
    assert result.payload["report"]["suggested_title"] == "上线准备会"
    assert result.payload["coverage"]["segments_used"] == 2

    sliced = _call_async(
        registry, "extract_report", team_a, meeting_id="m1",
        segment_range={"start": 2.0, "end": 4.0},
    )
    assert sliced.ok and sliced.payload["coverage"]["segments_used"] == 1

    assert _call_async(registry, "extract_report", team_a, meeting_id="nope").code == "not_found"


def test_suggest_title_reuses_analyzer(tmp_path) -> None:
    _, registry, analyzer, team_a, _ = _seed(tmp_path)
    result = _call_async(registry, "suggest_title", team_a, meeting_id="m1")
    assert result.ok and result.payload["suggested_title"] == "上线准备会"
    assert analyzer.calls == 1


# ---------------------------------------------------------------------------
# 体积护栏
# ---------------------------------------------------------------------------


def test_large_transcript_respects_payload_budget(tmp_path) -> None:
    path = tmp_path / "big.db"
    db = Database(path)
    db.initialize({TEAM_A: TOKEN_A})
    team_a = db.authenticate(TOKEN_A)
    db.create_meeting("big", team_a, "长会", tmp_path / "big.m4a")
    db.save_transcript(
        "big", team_a,
        Transcript(
            duration_seconds=500.0,
            segments=[
                TranscriptSegment(start=i * 10.0, end=i * 10.0 + 9.0, speaker_label="A", text="甲" * 400)
                for i in range(50)
            ],
        ),
    )
    registry = ToolRegistry()
    register_all_tools(registry, db)
    result = _call(registry, "get_transcript", team_a, meeting_id="big", page_size=50)
    if result.ok:
        assert payload_bytes(result.payload) <= 8 * 1024
    else:
        assert result.code == "too_large" and result.next_hint

"""行动项原话证据（A+B 中的 B）：能回到原话核对「谁承诺了什么」。

只新增，不修改既有断言。
"""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

from app.agent.tools.capability import _validate_evidence
from app.db import Database
from app.llm import (
    AnalysisError,
    LLMAnalyzer,
    normalize_llm_payload,
    validate_team_action_evidence,
)
from app.config import Settings
from app.models import ActionItem, TeamMeetingReport, Transcript, TranscriptSegment

TEAM = "甲团队"
TOKEN = "test-access-token"
TRANSCRIPT = Transcript(
    duration_seconds=6.0,
    segments=[
        TranscriptSegment(start=0.0, end=2.0, speaker_label="A", text="决定本周上线"),
        TranscriptSegment(start=2.0, end=4.0, speaker_label="B", text="我来准备发布清单"),
    ],
)


def _report(**overrides) -> TeamMeetingReport:
    payload = {
        "suggested_title": "上线准备会",
        "overview": "会议决定本周上线。",
        "meeting_points": ["决定本周上线"],
        "decisions": [{
            "content": "本周上线", "decision_maker": "小王",
            "evidence": {"quote": "决定本周上线", "timestamp": "00:00:01"},
        }],
        "action_items": [{
            "task": "准备发布清单", "owner": "小李", "deadline": "周五",
            "evidence": {"quote": "我来准备发布清单", "timestamp": "00:00:03"},
        }],
        "unresolved_issues": [],
        "speaker_stats_note": "",
    }
    payload.update(overrides)
    return TeamMeetingReport.model_validate(payload)


# ---------------------------------------------------------------------------
# 校验函数
# ---------------------------------------------------------------------------


def test_action_evidence_passes_when_quote_matches() -> None:
    validate_team_action_evidence(_report(), TRANSCRIPT.segments)


def test_action_evidence_rejects_missing_evidence() -> None:
    report = _report(action_items=[{"task": "准备发布清单", "owner": "小李", "deadline": "周五"}])
    with pytest.raises(ValueError) as excinfo:
        validate_team_action_evidence(report, TRANSCRIPT.segments)
    assert "缺少原话证据" in str(excinfo.value)


def test_action_evidence_rejects_hallucinated_quote() -> None:
    report = _report(action_items=[{
        "task": "准备发布清单", "owner": "小李", "deadline": "周五",
        "evidence": {"quote": "这句会上没说", "timestamp": "00:00:03"},
    }])
    with pytest.raises(ValueError):
        validate_team_action_evidence(report, TRANSCRIPT.segments)


def test_action_evidence_rejects_wrong_timestamp() -> None:
    report = _report(action_items=[{
        "task": "准备发布清单", "owner": "小李", "deadline": "周五",
        "evidence": {"quote": "我来准备发布清单", "timestamp": "00:00:30"},
    }])
    with pytest.raises(ValueError):
        validate_team_action_evidence(report, TRANSCRIPT.segments)


# ---------------------------------------------------------------------------
# 分析路径：新报告必须带行动项原话
# ---------------------------------------------------------------------------


class _Completions:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        message = SimpleNamespace(content=json.dumps(self.payload, ensure_ascii=False))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


def _analyzer(payload: dict) -> LLMAnalyzer:
    settings = Settings(OPENAI_API_KEY="test", TEAM_TOKENS="test-access-token", LLM_MAX_RETRIES="0")
    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions(payload)))
    return LLMAnalyzer(settings, client=client)


def test_analysis_fails_when_action_item_has_no_evidence() -> None:
    payload = _report().model_dump()
    payload["action_items"] = [{"task": "准备发布清单", "owner": "小李", "deadline": "周五"}]
    with pytest.raises(AnalysisError):
        asyncio.run(_analyzer(payload).analyze_team(TRANSCRIPT))


def test_analysis_passes_when_action_item_has_evidence() -> None:
    result = asyncio.run(_analyzer(_report().model_dump()).analyze_team(TRANSCRIPT))
    assert result.action_items[0].evidence is not None
    assert result.action_items[0].evidence.quote == "我来准备发布清单"


def test_normalize_preserves_action_evidence() -> None:
    payload = {
        "overview": "x",
        "meeting_points": [],
        "decisions": [],
        "action_items": [{
            "task": "准备发布清单", "owner": "小李", "deadline": "周五",
            "evidence": {"quote": "我来准备发布清单", "timestamp": "00:00:03"},
        }],
        "unresolved_issues": [],
    }
    normalized = normalize_llm_payload(payload, TeamMeetingReport)
    assert normalized["action_items"][0]["evidence"]["quote"] == "我来准备发布清单"


def test_legacy_action_item_without_evidence_still_parses() -> None:
    item = ActionItem(task="旧任务", owner="未明确", deadline="未明确")
    assert item.evidence is None


# ---------------------------------------------------------------------------
# validate_evidence 工具：把「行动项缺原话」也报出来
# ---------------------------------------------------------------------------


def _seed(tmp_path):
    path = tmp_path / "action-evidence.db"
    db = Database(path)
    db.initialize({TEAM: TOKEN})
    team_id = db.authenticate(TOKEN)
    db.create_meeting("m1", team_id, "周会", tmp_path / "a.m4a")
    db.save_transcript("m1", team_id, TRANSCRIPT)
    return db, team_id


def test_validate_evidence_tool_reports_missing_action_evidence(tmp_path) -> None:
    db, team_id = _seed(tmp_path)
    payload = _report().model_dump()
    payload["action_items"] = [{"task": "准备发布清单", "owner": "小李", "deadline": "周五"}]
    result = _validate_evidence(db, team_id, "m1", payload)
    assert result.ok and result.payload["valid"] is False
    kinds = [item["kind"] for item in result.payload["invalid"]]
    assert "action_item" in kinds


def test_validate_evidence_tool_passes_for_complete_report(tmp_path) -> None:
    db, team_id = _seed(tmp_path)
    result = _validate_evidence(db, team_id, "m1", _report().model_dump())
    assert result.ok and result.payload["valid"] is True and result.payload["invalid"] == []

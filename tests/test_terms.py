"""阶段 10-B（M3）· R-P1.5-9 术语热词注入的契约测试。

只新增，不改 tests/test_app.py 中的任何既有断言。
覆盖：词表合并（手动 + 成员姓名，改名自动生效）、提示组装与长度上限、
**未配置/关闭时转写调用与现状完全一致（单参数）**、配置后把提示传给转写入参、
端点契约与团队隔离、以及热词不影响引文校验规则。
"""

import asyncio
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

import app.main as main_module
from app import terms as terms_module
from app.db import Database
from app.models import TeamMeetingReport, Transcript, TranscriptSegment
from app.transcription import WhisperTranscriber

AUTH_HEADERS = {"X-Access-Token": "test-access-token"}


@pytest.fixture(autouse=True)
def isolated_database(monkeypatch, tmp_path):
    database = Database(tmp_path / "terms.db")
    database.initialize(main_module.settings.parsed_team_tokens())
    monkeypatch.setattr(main_module, "database", database)
    monkeypatch.setattr(main_module.speaker_recognizer, "enabled", False)
    return database


def _team_id(database: Database, token: str = "test-access-token") -> int:
    team_id = database.authenticate(token)
    assert team_id is not None
    return team_id


def _add_member(database: Database, team_id: int, name: str) -> int:
    with database._lock, database._connect() as connection:
        connection.execute(
            "INSERT INTO members(team_id,name,role,is_key_decision_maker,created_at) VALUES(?,?,?,?,?)",
            (team_id, name, "", 0, "2026-09-21T00:00:00+00:00"),
        )
        row = connection.execute(
            "SELECT id FROM members WHERE team_id=? AND name=?", (team_id, name)
        ).fetchone()
    return int(row["id"])


def _rename_member(database: Database, member_id: int, name: str) -> None:
    with database._lock, database._connect() as connection:
        connection.execute("UPDATE members SET name=? WHERE id=?", (name, member_id))


# --------------------------------------------------------------------------
# 提示组装
# --------------------------------------------------------------------------


def test_build_prompt_returns_none_without_terms() -> None:
    assert terms_module.build_prompt([]) is None
    assert terms_module.build_prompt(["", "   "]) is None


def test_build_prompt_contains_terms() -> None:
    prompt = terms_module.build_prompt(["胡泊", "会脉"])
    assert prompt is not None
    assert "胡泊" in prompt and "会脉" in prompt
    assert prompt.startswith(terms_module.PROMPT_PREFIX)


def test_build_prompt_respects_length_cap() -> None:
    budget = len(terms_module.PROMPT_PREFIX) + 2 + len("会脉") + 1
    prompt = terms_module.build_prompt(["会脉", "第二个词"], max_chars=budget)
    assert prompt is not None
    assert len(prompt) <= budget
    assert "会脉" in prompt
    assert "第二个词" not in prompt  # 预算用尽后不再追加


def test_build_prompt_truncates_overlong_term() -> None:
    prompt = terms_module.build_prompt(["超" * 80], max_chars=200)
    assert prompt is not None
    assert "超" * (terms_module.MAX_TERM_CHARS + 1) not in prompt


def test_collect_terms_merges_manual_and_member_names(isolated_database) -> None:
    database = isolated_database
    team_id = _team_id(database)
    database.upsert_term(team_id, "会脉", "产品名")
    _add_member(database, team_id, "胡泊")
    items = terms_module.collect_terms(database, team_id)
    assert [(item["term"], item["source"]) for item in items] == [
        ("会脉", "manual"), ("胡泊", "member"),
    ]
    assert items[0]["id"] and items[1]["id"] is None


def test_member_rename_is_reflected_without_sync(isolated_database) -> None:
    """成员改名后热词自动跟着变（成员姓名不落表，每次组装时读取）。"""
    database = isolated_database
    team_id = _team_id(database)
    member_id = _add_member(database, team_id, "胡泊")
    assert "胡泊" in terms_module.build_team_prompt(
        database, team_id, enabled=True, max_chars=200)
    _rename_member(database, member_id, "胡泊老师")
    prompt = terms_module.build_team_prompt(database, team_id, enabled=True, max_chars=200)
    assert "胡泊老师" in prompt and "胡泊、" not in prompt


def test_team_prompt_is_none_when_disabled_or_no_team(isolated_database) -> None:
    database = isolated_database
    team_id = _team_id(database)
    database.upsert_term(team_id, "会脉")
    assert terms_module.build_team_prompt(database, team_id, enabled=False, max_chars=200) is None
    assert terms_module.build_team_prompt(database, None, enabled=True, max_chars=200) is None


# --------------------------------------------------------------------------
# 转写入参：未配置时行为与现状完全一致
# --------------------------------------------------------------------------


class _FakeWhisperModel:
    def __init__(self) -> None:
        self.calls = []

    def transcribe(self, source, **kwargs):
        self.calls.append(kwargs)
        segment = type("Segment", (), {"start": 0.0, "end": 1.0, "text": "我们本周上线内测"})
        info = type("Info", (), {"duration": 1.0, "language": "zh"})
        return [segment], info


def test_transcribe_forwards_initial_prompt(tmp_path, monkeypatch) -> None:
    transcriber = WhisperTranscriber(main_module.settings)
    model = _FakeWhisperModel()
    monkeypatch.setattr(transcriber, "_get_model", lambda: model)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake")

    transcriber.transcribe(audio, "提示：胡泊")
    assert model.calls[0]["initial_prompt"] == "提示：胡泊"

    transcriber.transcribe(audio)
    assert model.calls[1]["initial_prompt"] is None  # 不传时与旧行为一致


def test_pipeline_uses_single_argument_when_no_terms(monkeypatch, tmp_path) -> None:
    """未配置热词 → 转写仍是单参数调用（与引入热词前逐字节一致）。"""
    database = main_module.database
    team_id = _team_id(database)
    meeting_id = "b" * 32
    audio_path = tmp_path / "{}.wav".format(meeting_id)
    audio_path.write_bytes(b"private audio")
    database.create_meeting(meeting_id, team_id, "周会", audio_path)
    transcript = Transcript(
        duration_seconds=10, segments=[TranscriptSegment(start=0, end=10, text="我们本周上线内测")],
    )
    seen = {}

    def fake_transcribe(*args):
        seen["args"] = args
        return transcript

    monkeypatch.setattr(main_module.transcriber, "transcribe", fake_transcribe)

    async def fake_team_report(received, analyzer):
        return TeamMeetingReport(
            overview="会议确认本周上线内测。", meeting_points=["本周上线"],
            decisions=[], action_items=[], unresolved_issues=[],
        )

    monkeypatch.setattr(main_module, "build_team_report", fake_team_report)
    asyncio.run(main_module._process_audio(audio_path, lambda status, message: True))
    assert len(seen["args"]) == 1, "未配置热词时不应多传参数"


def test_pipeline_passes_prompt_when_terms_configured(monkeypatch, tmp_path) -> None:
    database = main_module.database
    team_id = _team_id(database)
    database.upsert_term(team_id, "会脉")
    meeting_id = "c" * 32
    audio_path = tmp_path / "{}.wav".format(meeting_id)
    audio_path.write_bytes(b"private audio")
    database.create_meeting(meeting_id, team_id, "周会", audio_path)
    transcript = Transcript(
        duration_seconds=10, segments=[TranscriptSegment(start=0, end=10, text="我们本周上线内测")],
    )
    seen = {}

    def fake_transcribe(*args):
        seen["args"] = args
        return transcript

    monkeypatch.setattr(main_module.transcriber, "transcribe", fake_transcribe)

    async def fake_team_report(received, analyzer):
        return TeamMeetingReport(
            overview="会议确认本周上线内测。", meeting_points=["本周上线"],
            decisions=[], action_items=[], unresolved_issues=[],
        )

    monkeypatch.setattr(main_module, "build_team_report", fake_team_report)
    asyncio.run(main_module._process_audio(audio_path, lambda status, message: True))
    assert len(seen["args"]) == 2
    assert "会脉" in seen["args"][1]


def test_pipeline_respects_terms_switch(monkeypatch, tmp_path) -> None:
    database = main_module.database
    team_id = _team_id(database)
    database.upsert_term(team_id, "会脉")
    monkeypatch.setattr(main_module.settings, "team_terms_enabled", False)
    meeting_id = "d" * 32
    audio_path = tmp_path / "{}.wav".format(meeting_id)
    audio_path.write_bytes(b"private audio")
    database.create_meeting(meeting_id, team_id, "周会", audio_path)
    transcript = Transcript(
        duration_seconds=10, segments=[TranscriptSegment(start=0, end=10, text="我们本周上线内测")],
    )
    seen = {}

    def fake_transcribe(*args):
        seen["args"] = args
        return transcript

    monkeypatch.setattr(main_module.transcriber, "transcribe", fake_transcribe)

    async def fake_team_report(received, analyzer):
        return TeamMeetingReport(
            overview="会议确认本周上线内测。", meeting_points=["本周上线"],
            decisions=[], action_items=[], unresolved_issues=[],
        )

    monkeypatch.setattr(main_module, "build_team_report", fake_team_report)
    asyncio.run(main_module._process_audio(audio_path, lambda status, message: True))
    assert len(seen["args"]) == 1, "关闭热词开关后必须回到单参数调用"


# --------------------------------------------------------------------------
# 端点契约
# --------------------------------------------------------------------------


def test_terms_endpoints_roundtrip(isolated_database) -> None:
    database = isolated_database
    team_id = _team_id(database)
    _add_member(database, team_id, "胡泊")
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        created = client.post("/api/terms", json={"term": "  会脉  ", "note": "产品名"})
        assert created.status_code == 200
        assert created.json()["term"] == "会脉"
        assert created.json()["source"] == "manual"

        listed = client.get("/api/terms").json()
        assert [(item["term"], item["source"]) for item in listed["items"]] == [
            ("会脉", "manual"), ("胡泊", "member"),
        ]
        assert "会脉" in listed["prompt"] and "胡泊" in listed["prompt"]

        duplicate = client.post("/api/terms", json={"term": "会脉", "note": "改过的说明"})
        assert duplicate.status_code == 200
        assert len(client.get("/api/terms").json()["items"]) == 2  # 不重复

        deleted = client.delete("/api/terms/{}".format(created.json()["id"]))
        assert deleted.status_code == 200
        after = client.get("/api/terms").json()
        assert [item["term"] for item in after["items"]] == ["胡泊"]  # 成员姓名删不掉


@pytest.mark.parametrize("payload", [{"term": ""}, {"term": "   "}, {"term": "超" * 41}])
def test_term_validation(isolated_database, payload) -> None:
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post("/api/terms", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_args"


def test_terms_are_team_isolated(isolated_database) -> None:
    database = isolated_database
    other_team = _team_id(database, "other-team-token")
    other_id = database.upsert_term(other_team, "别人的词")
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        assert client.get("/api/terms").json()["items"] == []
        assert client.delete("/api/terms/{}".format(other_id)).status_code == 404


def test_delete_unknown_term_returns_not_found(isolated_database) -> None:
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        assert client.delete("/api/terms/99999").status_code == 404


# --------------------------------------------------------------------------
# 红线：热词不改变引文校验规则
# --------------------------------------------------------------------------


def test_terms_do_not_relax_evidence_rule(isolated_database) -> None:
    """热词只影响转写质量；引文仍必须是转写原文的完全一致子串。"""
    database = isolated_database
    team_id = _team_id(database)
    database.upsert_term(team_id, "会脉")
    segments = [TranscriptSegment(start=0.0, end=5.0, text="我们本周上线内测")]
    report = TeamMeetingReport(
        overview="会议确认本周上线内测。", meeting_points=["本周上线"],
        decisions=[{
            "content": "本周上线", "decision_maker": "胡泊",
            "evidence": {"quote": "转写里没有这句话", "timestamp": "00:00:02"},
        }],
        action_items=[], unresolved_issues=[],
    )
    with pytest.raises(ValueError):
        main_module.validate_team_evidence(report, segments)

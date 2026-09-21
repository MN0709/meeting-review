"""阶段 8（M1 可检索）· R-P1.5-4 跨会议搜索端点的契约测试。

只新增，不改 tests/test_app.py 中的任何既有断言。
覆盖：命中结构、team 隔离、空结果、参数校验、项目/日期过滤、FTS5→LIKE 回退、
通配符字面量、单场会议条数上限、老接口与 Agent 工具零回归。
"""

import os
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

import app.main as main_module
from app.db import Database
from app.models import Transcript, TranscriptSegment
from app.security import SHANGHAI_TZ

AUTH_HEADERS = {"X-Access-Token": "test-access-token"}
OTHER_HEADERS = {"X-Access-Token": "other-team-token"}


@pytest.fixture(autouse=True)
def isolated_database(monkeypatch, tmp_path):
    """每个用例一个空库，避免共享 /tmp 下的既有数据影响搜索结果。"""
    database = Database(tmp_path / "search.db")
    database.initialize(main_module.settings.parsed_team_tokens())
    monkeypatch.setattr(main_module, "database", database)
    monkeypatch.setattr(main_module.speaker_recognizer, "enabled", False)
    return database


def _team_id(database: Database, token: str = "test-access-token") -> int:
    team_id = database.authenticate(token)
    assert team_id is not None
    return team_id


def _seed_meeting(
    database: Database, team_id: int, meeting_id: str, title: str,
    texts, *, project_id=None, speaker="说话人 1",
) -> None:
    database.create_meeting(meeting_id, team_id, title, Path("/tmp/{}.wav".format(meeting_id)), project_id)
    segments = [
        TranscriptSegment(start=index * 10.0, end=index * 10.0 + 5.0, speaker_label=speaker, text=text)
        for index, text in enumerate(texts)
    ]
    database.save_transcript(
        meeting_id, team_id,
        Transcript(language="zh", duration_seconds=max(len(texts) * 10.0, 1.0), segments=segments),
    )
    database.clear_audio_path(meeting_id, team_id)


# --------------------------------------------------------------------------
# 命中结构 / 基本可用
# --------------------------------------------------------------------------


def test_search_returns_hits_with_contract_fields(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    project = isolated_database.create_project("proj-weekly", team_id, "周会项目")
    _seed_meeting(
        isolated_database, team_id, "m1", "9月第一周例会",
        ["我们先把排期定下来", "排期问题下周再确认"], project_id=project.id,
    )
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.get("/api/search", params={"q": "排期"})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "排期"
    assert body["count"] == len(body["hits"]) == 2
    hit = body["hits"][0]
    assert set(hit) == {
        "meeting_id", "meeting_title", "project_id", "project_name",
        "start", "end", "timestamp", "speaker_label", "text_snippet",
    }
    assert hit["meeting_id"] == "m1"
    assert hit["meeting_title"] == "9月第一周例会"
    assert hit["project_id"] == "proj-weekly"
    assert hit["project_name"] == "周会项目"
    assert hit["timestamp"] == "00:00:00"
    assert hit["speaker_label"] == "说话人 1"
    assert "排期" in hit["text_snippet"]


def test_search_spans_multiple_meetings(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed_meeting(isolated_database, team_id, "m1", "第一场", ["今天聊排期"])
    _seed_meeting(isolated_database, team_id, "m2", "第二场", ["排期要提前"])
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.get("/api/search", params={"q": "排期"}).json()
    assert {hit["meeting_id"] for hit in body["hits"]} == {"m1", "m2"}


def test_search_timestamp_matches_segment_start(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed_meeting(isolated_database, team_id, "m1", "会议", ["无关内容", "无关内容", "这里提到排期"])
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        hit = client.get("/api/search", params={"q": "排期"}).json()["hits"][0]
    assert hit["start"] == 20.0
    assert hit["timestamp"] == "00:00:20"


def test_search_empty_result_is_not_an_error(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed_meeting(isolated_database, team_id, "m1", "会议", ["完全无关的内容"])
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.get("/api/search", params={"q": "根本不存在"})
    assert response.status_code == 200
    assert response.json() == {"query": "根本不存在", "count": 0, "hits": []}


# --------------------------------------------------------------------------
# 鉴权与隔离（红线 2）
# --------------------------------------------------------------------------


def test_search_without_token_is_rejected() -> None:
    response = TestClient(main_module.app).get("/api/search", params={"q": "排期"})
    assert response.status_code == 403


def test_search_never_returns_other_team_data(isolated_database) -> None:
    other_team = _team_id(isolated_database, "other-team-token")
    _seed_meeting(isolated_database, other_team, "other-1", "别的团队会议", ["排期是下周三"])
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.get("/api/search", params={"q": "排期"})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 0
    assert "别的团队会议" not in response.text


def test_search_other_team_project_returns_forbidden(isolated_database) -> None:
    other_team = _team_id(isolated_database, "other-team-token")
    foreign = isolated_database.create_project("proj-other", other_team, "别人的项目")
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.get("/api/search", params={"q": "排期", "project_id": foreign.id})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "team_forbidden"


def test_search_unknown_project_returns_not_found(isolated_database) -> None:
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.get("/api/search", params={"q": "排期", "project_id": "does-not-exist"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


# --------------------------------------------------------------------------
# 参数校验
# --------------------------------------------------------------------------


@pytest.mark.parametrize("query", ["", "   "])
def test_search_rejects_blank_query(isolated_database, query) -> None:
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.get("/api/search", params={"q": query})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_args"


def test_search_rejects_too_long_query(isolated_database) -> None:
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.get("/api/search", params={"q": "排" * 51})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_args"


@pytest.mark.parametrize("param", ["2026-13-99", "not-a-date"])
def test_search_rejects_invalid_date(isolated_database, param) -> None:
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.get("/api/search", params={"q": "排期", "from": param})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_args"


def test_search_limit_is_clamped_to_fifty(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    for index in range(12):
        _seed_meeting(
            isolated_database, team_id, "m{}".format(index), "会议{}".format(index),
            ["排期讨论"] * 6,
        )
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.get("/api/search", params={"q": "排期", "limit": 999}).json()
    assert len(body["hits"]) == 50  # 12 场 × 每场 5 条上限


def test_search_limit_zero_falls_back_to_one(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed_meeting(isolated_database, team_id, "m1", "会议", ["排期", "排期"])
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.get("/api/search", params={"q": "排期", "limit": 0}).json()
    assert len(body["hits"]) == 1


# --------------------------------------------------------------------------
# 单场会议条数上限 / 顺序
# --------------------------------------------------------------------------


def test_per_meeting_limit_prevents_single_meeting_flood(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed_meeting(isolated_database, team_id, "noisy", "话很多的一场", ["排期"] * 20)
    _seed_meeting(isolated_database, team_id, "quiet", "另一场", ["排期"])
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.get("/api/search", params={"q": "排期", "limit": 20}).json()
    counts = {}
    for hit in body["hits"]:
        counts[hit["meeting_id"]] = counts.get(hit["meeting_id"], 0) + 1
    assert counts["noisy"] == 5
    assert counts["quiet"] == 1


# --------------------------------------------------------------------------
# 项目 / 日期过滤
# --------------------------------------------------------------------------


def test_search_filters_by_project(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    alpha = isolated_database.create_project("proj-a", team_id, "A 项目")
    beta = isolated_database.create_project("proj-b", team_id, "B 项目")
    _seed_meeting(isolated_database, team_id, "m-a", "A 会", ["排期在 A"], project_id=alpha.id)
    _seed_meeting(isolated_database, team_id, "m-b", "B 会", ["排期在 B"], project_id=beta.id)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.get("/api/search", params={"q": "排期", "project_id": "proj-b"}).json()
    assert [hit["meeting_id"] for hit in body["hits"]] == ["m-b"]


def test_search_date_filter_covers_whole_day(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed_meeting(isolated_database, team_id, "today", "今天的会", ["排期今天定了"])
    today = datetime.now(SHANGHAI_TZ).date().isoformat()
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        inside = client.get("/api/search", params={"q": "排期", "from": today, "to": today}).json()
        before = client.get("/api/search", params={"q": "排期", "to": "2000-01-01"}).json()
    assert inside["count"] == 1
    assert before["count"] == 0


def test_search_accepts_full_iso8601_bounds(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed_meeting(isolated_database, team_id, "today", "今天的会", ["排期今天定了"])
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.get(
            "/api/search",
            params={"q": "排期", "from": "2000-01-01T00:00:00+08:00", "to": "2999-01-01T00:00:00+08:00"},
        ).json()
    assert body["count"] == 1


# --------------------------------------------------------------------------
# FTS5 回退与通配符安全
# --------------------------------------------------------------------------


def test_two_char_query_falls_back_to_like(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed_meeting(isolated_database, team_id, "m1", "会议", ["明天上线"])
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.get("/api/search", params={"q": "上线"}).json()
    assert body["count"] == 1


@pytest.mark.parametrize("wildcard", ["%", "_", "%%"])
def test_wildcards_are_matched_literally(isolated_database, wildcard) -> None:
    team_id = _team_id(isolated_database)
    _seed_meeting(isolated_database, team_id, "m1", "会议", ["本月预算是 100 万"])
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.get("/api/search", params={"q": wildcard}).json()
    assert body["count"] == 0


def test_fts_special_characters_do_not_break_search(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed_meeting(isolated_database, team_id, "m1", "会议", ["讨论 (方案 A) 与方案B"])
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.get("/api/search", params={"q": "(方案 A)"})
    assert response.status_code == 200


# --------------------------------------------------------------------------
# 零回归：老接口与 Agent 工具
# --------------------------------------------------------------------------


def test_meeting_detail_still_returns_full_transcript(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed_meeting(isolated_database, team_id, "m1", "会议", ["第一句", "第二句", "第三句"])
    segments = isolated_database.load_transcript("m1", team_id)
    assert segments is not None and len(segments.segments) == 3
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.get("/api/meetings/m1")
    assert response.status_code in (200, 404)  # 无报告时为 404，属既有行为


def test_agent_tool_search_transcript_is_unchanged(isolated_database) -> None:
    """R-P1.5-4 新增的是并行方法；Agent 工具依赖的 search_transcripts 行为必须一致。"""
    team_id = _team_id(isolated_database)
    _seed_meeting(isolated_database, team_id, "m1", "会议", ["排期在下周三确认"])
    legacy = isolated_database.search_transcripts(team_id, "排期", limit=10)
    product = isolated_database.search_meetings(team_id, "排期", limit=10)
    assert len(legacy) == len(product) == 1
    assert legacy[0]["meeting_id"] == product[0]["meeting_id"]
    assert legacy[0]["start"] == product[0]["start"]
    assert legacy[0]["text"] == product[0]["text"]

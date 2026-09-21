"""阶段 9-B（M2）· R-P1.5-1 图片纪要 + 导出 PDF 的契约测试。

只新增，不改 tests/test_app.py 中的任何既有断言。
覆盖：四板块顺序与内容来源、空态文案、「我答应的任务」的红线、HTML 转义、
端点契约与隔离、PDF 渲染器缺失时的降级、以及「紧急事项」抽取字段的清洗与校验。
"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

import app.main as main_module
from app.db import Database
from app.deliverables.image_minutes import build_parts, render_html
from app.deliverables.render import RendererUnavailable
from app.llm import normalize_llm_payload, validate_team_evidence
from app.models import TeamChunkSummary, TeamMeetingReport, TranscriptSegment

AUTH_HEADERS = {"X-Access-Token": "test-access-token"}
OTHER_HEADERS = {"X-Access-Token": "other-team-token"}
MEETING = "m-image"


def report(**overrides) -> TeamMeetingReport:
    payload = {
        "suggested_title": "内测上线安排",
        "overview": "会议确认本周上线内测，并明确发布准备工作的负责人。",
        "meeting_points": ["确定本周上线内测", "发布清单周五前完成"],
        "decisions": [{
            "content": "本周上线内测", "decision_maker": "胡泊",
            "evidence": {"quote": "我们本周上线内测", "timestamp": "00:00:02"},
        }],
        "action_items": [
            {"task": "准备发布清单", "owner": "胡泊", "deadline": "周五",
             "evidence": {"quote": "我来准备发布清单", "timestamp": "00:00:06"}},
            {"task": "同步给客户", "owner": "马宁", "deadline": "周六",
             "evidence": {"quote": "我去同步客户", "timestamp": "00:00:08"}},
        ],
        "unresolved_issues": [],
        "urgent_items": [{
            "content": "今天必须定稿",
            "evidence": {"quote": "今天必须定稿", "timestamp": "00:00:10"},
        }],
    }
    payload.update(overrides)
    return TeamMeetingReport(**payload)


ACTION_ROWS = [
    {"item_index": 0, "task": "准备发布清单", "owner": "胡泊", "deadline": "周五", "status": "待确认"},
    {"item_index": 1, "task": "同步给客户", "owner": "马宁", "deadline": "周六", "status": "已完成"},
]


@pytest.fixture(autouse=True)
def isolated_database(monkeypatch, tmp_path):
    database = Database(tmp_path / "deliverables.db")
    database.initialize(main_module.settings.parsed_team_tokens())
    monkeypatch.setattr(main_module, "database", database)
    monkeypatch.setattr(main_module.speaker_recognizer, "enabled", False)
    return database


def _team_id(database: Database, token: str = "test-access-token") -> int:
    team_id = database.authenticate(token)
    assert team_id is not None
    return team_id


def _seed(database: Database, team_id: int, *, with_urgent: bool = True, owners=("胡泊", "马宁")) -> None:
    payload = report()
    if not with_urgent:
        payload = report(urgent_items=[])
    database.create_meeting(MEETING, team_id, "周会", Path("/tmp/m.wav"))
    database.save_report(MEETING, team_id, payload)
    database.clear_audio_path(MEETING, team_id)


# --------------------------------------------------------------------------
# 四板块结构与内容来源
# --------------------------------------------------------------------------


def test_parts_have_fixed_order_and_titles() -> None:
    parts = build_parts(report=report(), action_rows=ACTION_ROWS, self_name="胡泊")
    assert [part["key"] for part in parts] == ["core", "urgent", "todo", "mine"]
    assert [part["title"] for part in parts] == [
        "① 这次会议的核心是什么", "② 紧急事项是什么",
        "③ 待办是什么", "④ 我答应的任务是什么",
    ]


def test_core_comes_from_overview_points_and_decisions() -> None:
    parts = build_parts(report=report(), action_rows=ACTION_ROWS, self_name="胡泊")
    texts = [item["text"] for item in parts[0]["items"]]
    assert texts == [
        "会议确认本周上线内测，并明确发布准备工作的负责人。",
        "确定本周上线内测",
        "发布清单周五前完成",
        "本周上线内测",  # 关键决策（带时间戳，可回到原话）
    ]
    assert parts[0]["items"][0].get("timestamp") is None
    assert parts[0]["items"][3]["timestamp"] == "00:00:02"


def test_urgent_items_carry_quote_and_timestamp() -> None:
    parts = build_parts(report=report(), action_rows=ACTION_ROWS, self_name="胡泊")
    items = parts[1]["items"]
    assert [item["text"] for item in items] == ["今天必须定稿"]
    assert items[0]["timestamp"] == "00:00:10"


def test_todo_excludes_finished_items() -> None:
    parts = build_parts(report=report(), action_rows=ACTION_ROWS, self_name="胡泊")
    todo = [item["text"] for item in parts[2]["items"]]
    assert todo == ["准备发布清单"]  # 「同步给客户」已完成，不算待办
    assert "状态：待确认" in parts[2]["items"][0]["meta"]


def test_mine_only_contains_my_own_tasks() -> None:
    parts = build_parts(report=report(), action_rows=ACTION_ROWS, self_name="马宁")
    # 马宁自己的那条已完成 → 不在待办；胡泊的不属于马宁
    assert parts[3]["items"] == []
    assert parts[3]["empty_note"] == "本次没有未完成的待办。" or parts[3]["empty_note"] == "未指定你自己。"


def test_mine_is_empty_and_explicit_when_self_unknown() -> None:
    """红线 8：没指定「我」时不得推断。"""
    parts = build_parts(report=report(), action_rows=ACTION_ROWS, self_name=None)
    assert parts[3]["items"] == []
    assert parts[3]["empty_note"] == "未指定你自己。"


def test_empty_parts_have_explicit_notes() -> None:
    parts = build_parts(report=report(urgent_items=[]), action_rows=ACTION_ROWS, self_name="胡泊")
    assert parts[1]["items"] == []
    assert parts[1]["empty_note"] == "本次未识别到紧急事项。"


def test_no_invented_sentences() -> None:
    """四板块里的每一句都必须来自报告字段，不能凭空生成。"""
    source = report()
    parts = build_parts(report=source, action_rows=ACTION_ROWS, self_name="胡泊")
    allowed = set(source.meeting_points) | {source.overview}
    allowed |= {item.task for item in source.action_items}
    allowed |= {item.content for item in source.urgent_items}
    allowed |= {item.content for item in source.decisions}
    for part in parts:
        for item in part["items"]:
            assert item["text"] in allowed


# --------------------------------------------------------------------------
# HTML 渲染
# --------------------------------------------------------------------------


def test_render_html_contains_four_cards_in_order() -> None:
    html_text = render_html(report=report(), action_rows=ACTION_ROWS, self_name="胡泊",
                            title="测试会议", meta="2026-09-21", footer="会脉")
    order = [html_text.index("c-core"), html_text.index("c-urgent"),
             html_text.index("c-todo"), html_text.index("c-mine")]
    assert order == sorted(order)


def test_render_html_escapes_dangerous_text() -> None:
    payload = report(meeting_points=["<script>alert(1)</script>"], urgent_items=[])
    html_text = render_html(report=payload, action_rows=[], self_name=None)
    assert "<script>alert(1)</script>" not in html_text
    assert "&lt;script&gt;" in html_text


def test_render_html_shows_empty_notes() -> None:
    html_text = render_html(report=report(urgent_items=[]), action_rows=[], self_name=None)
    assert "本次未识别到紧急事项。" in html_text
    assert "未指定你自己。" in html_text


# --------------------------------------------------------------------------
# 端点契约
# --------------------------------------------------------------------------


def test_image_minutes_endpoint_returns_four_parts(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post("/api/meetings/{}/image-minutes".format(MEETING))
    assert response.status_code == 200
    body = response.json()
    assert body["meeting_id"] == MEETING
    assert body["title"] == "周会"
    assert [part["key"] for part in body["parts"]] == ["core", "urgent", "todo", "mine"]
    assert body["parts"][1]["items"][0]["timestamp"] == "00:00:10"
    assert "未指定你自己" in body["meta"]


def test_image_minutes_is_idempotent(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        first = client.post("/api/meetings/{}/image-minutes".format(MEETING)).json()
        second = client.post("/api/meetings/{}/image-minutes".format(MEETING)).json()
    assert first == second


def test_image_minutes_missing_report_returns_not_found(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    isolated_database.create_meeting(MEETING, team_id, "无报告", Path("/tmp/m.wav"))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        assert client.post("/api/meetings/{}/image-minutes".format(MEETING)).status_code == 404


def test_image_minutes_other_team_is_forbidden(isolated_database) -> None:
    other_team = _team_id(isolated_database, "other-team-token")
    _seed(isolated_database, other_team)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post("/api/meetings/{}/image-minutes".format(MEETING))
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "team_forbidden"


def test_pdf_endpoint_returns_pdf(isolated_database, monkeypatch) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)

    async def fake_render(html_text, *, renderer="auto"):
        assert "c-core" in html_text and "c-mine" in html_text
        return b"%PDF-1.4 fake"

    monkeypatch.setattr(main_module, "render_pdf", fake_render)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.get("/api/meetings/{}/image-minutes.pdf".format(MEETING))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert "attachment" in response.headers["content-disposition"]
    assert response.content.startswith(b"%PDF-")


def test_pdf_endpoint_degrades_when_renderer_missing(isolated_database, monkeypatch) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)

    async def unavailable(html_text, *, renderer="auto"):
        raise RendererUnavailable("本机没有可用的 Chrome / Chromium，无法导出 PDF")

    monkeypatch.setattr(main_module, "render_pdf", unavailable)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.get("/api/meetings/{}/image-minutes.pdf".format(MEETING))
        # 其它交付物不受影响（红线 11：某类交付物失败不得让整场显示完成/失败）
        report_response = client.get("/api/meetings/{}".format(MEETING))
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "renderer_unavailable"
    assert report_response.status_code == 200
    assert report_response.json()["report"]["overview"]


def test_pdf_endpoint_other_team_is_forbidden(isolated_database, monkeypatch) -> None:
    other_team = _team_id(isolated_database, "other-team-token")
    _seed(isolated_database, other_team)

    async def fake_render(html_text, *, renderer="auto"):
        return b"%PDF-1.4 fake"

    monkeypatch.setattr(main_module, "render_pdf", fake_render)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.get("/api/meetings/{}/image-minutes.pdf".format(MEETING))
    assert response.status_code == 403


def test_old_meeting_report_without_urgent_items_still_works(isolated_database) -> None:
    """旧报告没有 urgent_items 字段时，图片纪要仍可生成，该板块显示空态。"""
    team_id = _team_id(isolated_database)
    payload = report().model_dump()
    payload.pop("urgent_items", None)
    isolated_database.create_meeting(MEETING, team_id, "旧报告会议", Path("/tmp/m.wav"))
    isolated_database.save_report(MEETING, team_id, TeamMeetingReport.model_validate(payload))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.post("/api/meetings/{}/image-minutes".format(MEETING)).json()
    assert body["parts"][1]["items"] == []
    assert body["parts"][1]["empty_note"] == "本次未识别到紧急事项。"


# --------------------------------------------------------------------------
# 紧急事项字段：清洗与强校验
# --------------------------------------------------------------------------


def test_normalize_keeps_urgent_items_and_drops_malformed() -> None:
    payload = {
        "meeting_points": ["要点"],
        "decisions": [], "action_items": [], "unresolved_issues": [],
        "urgent_items": [
            {"content": "今天必须定稿", "evidence": {"quote": "今天必须定稿", "timestamp": "00:00:10"}},
            {"content": "缺少证据", "evidence": None},
            "纯字符串",
        ],
    }
    normalized = normalize_llm_payload(payload, TeamMeetingReport)
    assert [item["content"] for item in normalized["urgent_items"]] == ["今天必须定稿"]


def test_validate_rejects_urgent_item_with_fabricated_quote() -> None:
    segments = [TranscriptSegment(start=0.0, end=5.0, text="我们本周上线内测")]
    good = report(urgent_items=[{
        "content": "本周上线", "evidence": {"quote": "我们本周上线内测", "timestamp": "00:00:02"},
    }])
    validate_team_evidence(good, segments)  # 不抛异常
    bad = report(urgent_items=[{
        "content": "编造", "evidence": {"quote": "这句话原文里没有", "timestamp": "00:00:02"},
    }])
    with pytest.raises(ValueError):
        validate_team_evidence(bad, segments)


def test_chunk_summary_has_urgent_items_default() -> None:
    summary = TeamChunkSummary(meeting_points=["要点"], decisions=[], action_items=[], unresolved_issues=[])
    assert summary.urgent_items == []

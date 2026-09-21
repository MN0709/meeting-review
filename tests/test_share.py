"""阶段 11／M4（R-P1.5-3）· 逐项勾选分享的契约测试。

只新增，不改 tests/test_app.py 中的任何既有断言。
红线 6/7 逐条断言：**只能看到勾选内容**、**不能当团队数据跳板**。
"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

import app.main as main_module
from app import shares
from app.db import Database
from app.models import TeamMeetingReport, Transcript, TranscriptSegment
from app.speaker import SpeakerClipObservation, SpeakerObservation

AUTH_HEADERS = {"X-Access-Token": "test-access-token"}
MEETING = "m-share"


def _report() -> TeamMeetingReport:
    return TeamMeetingReport(
        overview="会议确认本周上线内测，并明确发布准备工作的负责人。",
        meeting_points=["确定本周上线内测"],
        decisions=[{
            "content": "本周上线内测", "decision_maker": "胡泊",
            "evidence": {"quote": "我们本周上线内测", "timestamp": "00:00:02"},
        }],
        action_items=[{
            "task": "准备发布清单", "owner": "胡泊", "deadline": "周五",
            "evidence": {"quote": "我来准备发布清单", "timestamp": "00:00:06"},
        }],
        unresolved_issues=[],
        urgent_items=[{
            "content": "今天必须定稿",
            "evidence": {"quote": "今天必须定稿", "timestamp": "00:00:10"},
        }],
    )


@pytest.fixture(autouse=True)
def isolated_database(monkeypatch, tmp_path):
    database = Database(tmp_path / "share.db")
    database.initialize(main_module.settings.parsed_team_tokens())
    monkeypatch.setattr(main_module, "database", database)
    monkeypatch.setattr(main_module.speaker_recognizer, "enabled", False)
    return database


def _team_id(database: Database, token: str = "test-access-token") -> int:
    team_id = database.authenticate(token)
    assert team_id is not None
    return team_id


def _seed(database: Database, team_id: int) -> None:
    database.create_meeting(MEETING, team_id, "周会", Path("/tmp/m.wav"))
    database.save_transcript(MEETING, team_id, Transcript(
        language="zh", duration_seconds=20.0,
        segments=[
            TranscriptSegment(start=0.0, end=5.0, speaker_label="说话人 1", text="我们本周上线内测"),
            TranscriptSegment(start=10.0, end=15.0, speaker_label="说话人 1", text="今天必须定稿"),
        ],
    ))
    database.save_report(MEETING, team_id, _report())
    database.save_meeting_speakers(MEETING, team_id, [SpeakerObservation(
        local_label="说话人 1", embedding=None, speech_seconds=10.0,
        excerpts=["我们本周上线内测"], clips=[SpeakerClipObservation(
            start=0.0, end=2.0, text="我们本周上线内测", audio=b"RIFF-share",
        )],
    )])
    database.clear_audio_path(MEETING, team_id)


def _create(client, scopes):
    response = client.post(
        "/api/meetings/{}/share".format(MEETING), json={"scopes": scopes}, headers=AUTH_HEADERS,
    )
    return response


# --------------------------------------------------------------------------
# 创建
# --------------------------------------------------------------------------


def test_create_share_returns_one_time_token(isolated_database) -> None:
    database = isolated_database
    team_id = _team_id(database)
    _seed(database, team_id)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = _create(client, ["image_minutes", "report"])
    assert response.status_code == 200
    body = response.json()
    assert body["token"] and body["url"].endswith("/s/" + body["token"])
    assert body["scopes"] == ["image_minutes", "report"]
    assert body["expires_at"]
    # 表里只存哈希，查不到令牌原文
    assert database.share_link(shares.token_hash(body["token"])) is not None
    assert database.share_link(body["token"]) is None


def test_create_share_requires_auth_and_report(isolated_database) -> None:
    database = isolated_database
    team_id = _team_id(database)
    database.create_meeting(MEETING, team_id, "无报告", Path("/tmp/m.wav"))
    with TestClient(main_module.app) as client:
        response = client.post(
            "/api/meetings/{}/share".format(MEETING), json={"scopes": ["report"]},
        )
        assert response.status_code == 403
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        assert _create(client, ["report"]).status_code == 404


@pytest.mark.parametrize("scopes", [[], ["unknown_thing"], ["report", "nope"]])
def test_create_share_rejects_bad_scopes(isolated_database, scopes) -> None:
    database = isolated_database
    team_id = _team_id(database)
    _seed(database, team_id)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = _create(client, scopes)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_args"


def test_create_share_for_other_team_is_forbidden(isolated_database) -> None:
    database = isolated_database
    other_team = _team_id(database, "other-team-token")
    _seed(database, other_team)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        assert _create(client, ["report"]).status_code == 403


# --------------------------------------------------------------------------
# 读取：只能看到勾选内容（红线 6）
# --------------------------------------------------------------------------


def test_share_read_does_not_require_team_token(isolated_database) -> None:
    database = isolated_database
    _seed(database, _team_id(database))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        token = _create(client, ["report"]).json()["token"]
    with TestClient(main_module.app) as client:  # 不带任何口令
        response = client.get("/api/shares/{}".format(token))
    assert response.status_code == 200
    assert response.json()["title"] == "周会"


def test_share_only_exposes_selected_scopes(isolated_database) -> None:
    database = isolated_database
    _seed(database, _team_id(database))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        token = _create(client, ["image_minutes"]).json()["token"]
    with TestClient(main_module.app) as client:
        body = client.get("/api/shares/{}".format(token)).json()
    assert set(body) == {
        "title", "meeting_date", "duration_seconds", "scopes", "expires_at",
        "privacy_note", "image_minutes",
    }
    for leaked in ("transcript", "report", "tasks", "voice"):
        assert leaked not in body


def test_share_transcript_and_tasks_only_when_selected(isolated_database) -> None:
    database = isolated_database
    _seed(database, _team_id(database))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        transcript_token = _create(client, ["transcript"]).json()["token"]
        tasks_token = _create(client, ["tasks"]).json()["token"]
    with TestClient(main_module.app) as client:
        transcript_body = client.get("/api/shares/{}".format(transcript_token)).json()
        tasks_body = client.get("/api/shares/{}".format(tasks_token)).json()
    assert len(transcript_body["transcript"]) == 2
    assert "tasks" not in transcript_body and "report" not in transcript_body
    assert tasks_body["tasks"]["action_items"][0]["task"] == "准备发布清单"
    assert tasks_body["tasks"]["urgent_items"][0]["content"] == "今天必须定稿"
    assert "transcript" not in tasks_body


def test_share_image_minutes_parts_are_merged_and_ordered(isolated_database) -> None:
    database = isolated_database
    _seed(database, _team_id(database))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        token = _create(client, ["image_minutes"]).json()["token"]
    with TestClient(main_module.app) as client:
        body = client.get("/api/shares/{}".format(token)).json()
    # R-P2-8：紧急并入待办；未识别到「我」时「我答应的任务」整块隐藏。
    assert [part["key"] for part in body["image_minutes"]] == ["core", "todo", "decisions"]


def test_share_voice_lists_clips_and_serves_audio(isolated_database) -> None:
    database = isolated_database
    _seed(database, _team_id(database))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        token = _create(client, ["voice"]).json()["token"]
    with TestClient(main_module.app) as client:
        body = client.get("/api/shares/{}".format(token)).json()
        assert len(body["voice"]) == 1
        clip_id = body["voice"][0]["id"]
        audio = client.get("/api/shares/{}/clips/{}".format(token, clip_id))
    assert audio.status_code == 200
    assert audio.content == b"RIFF-share"


def test_share_clip_requires_voice_scope(isolated_database) -> None:
    database = isolated_database
    team_id = _team_id(database)
    _seed(database, team_id)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        token = _create(client, ["report"]).json()["token"]
        clip_id = database.get_history(MEETING, team_id).speakers[0].clips[0].id
        response = client.get("/api/shares/{}/clips/{}".format(token, clip_id))
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "team_forbidden"


def test_share_does_not_expose_team_library_or_costs(isolated_database) -> None:
    database = isolated_database
    _seed(database, _team_id(database))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        token = _create(client, list(shares.ALL_SCOPES)).json()["token"]
    with TestClient(main_module.app) as client:
        text = client.get("/api/shares/{}".format(token)).text
    for leaked in ("members", "voiceprint", "llm_usage", "cost", "token_hash", "agent_audit"):
        assert leaked not in text


# --------------------------------------------------------------------------
# 过期、撤销、篡改（红线 7）
# --------------------------------------------------------------------------


def test_unknown_token_is_not_found(isolated_database) -> None:
    with TestClient(main_module.app) as client:
        assert client.get("/api/shares/whatever").status_code == 404


def test_tampered_token_is_not_found(isolated_database) -> None:
    database = isolated_database
    _seed(database, _team_id(database))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        token = _create(client, ["report"]).json()["token"]
    tampered = token[:-2] + ("aa" if not token.endswith("aa") else "bb")
    with TestClient(main_module.app) as client:
        assert client.get("/api/shares/{}".format(tampered)).status_code == 404


def test_expired_share_returns_410(isolated_database, monkeypatch) -> None:
    database = isolated_database
    team_id = _team_id(database)
    _seed(database, team_id)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        token = _create(client, ["report"]).json()["token"]
    # 直接把过期时间改到过去
    with database._lock, database._connect() as connection:
        connection.execute(
            "UPDATE share_links SET expires_at=? WHERE token_hash=?",
            ("2000-01-01T00:00:00+00:00", shares.token_hash(token)),
        )
    with TestClient(main_module.app) as client:
        response = client.get("/api/shares/{}".format(token))
    assert response.status_code == 410
    assert response.json()["error"]["code"] == "share_expired"
    assert database.share_audit_rows(MEETING, team_id)[-1]["result_code"] == "expired"


def test_revoked_share_returns_410_immediately(isolated_database) -> None:
    database = isolated_database
    team_id = _team_id(database)
    _seed(database, team_id)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        token = _create(client, ["report"]).json()["token"]
        assert client.get("/api/shares/{}".format(token)).status_code == 200
        revoked = client.delete(
            "/api/meetings/{}/shares/{}".format(MEETING, token), headers=AUTH_HEADERS,
        )
        assert revoked.status_code == 200
        again = client.delete(
            "/api/meetings/{}/shares/{}".format(MEETING, token), headers=AUTH_HEADERS,
        )
    assert again.status_code == 404
    with TestClient(main_module.app) as client:
        response = client.get("/api/shares/{}".format(token))
    assert response.status_code == 410
    assert response.json()["error"]["code"] == "share_revoked"
    assert database.share_audit_rows(MEETING, team_id)[-1]["result_code"] == "revoked"


def test_revoke_requires_team_token(isolated_database) -> None:
    database = isolated_database
    _seed(database, _team_id(database))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        token = _create(client, ["report"]).json()["token"]
    with TestClient(main_module.app) as client:
        response = client.delete("/api/meetings/{}/shares/{}".format(MEETING, token))
    assert response.status_code == 403


# --------------------------------------------------------------------------
# 审计、计数、列表、级联、鉴权边界
# --------------------------------------------------------------------------


def test_share_access_is_audited_and_counted(isolated_database) -> None:
    database = isolated_database
    team_id = _team_id(database)
    _seed(database, team_id)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        token = _create(client, ["report"]).json()["token"]
    with TestClient(main_module.app) as client:
        for _ in range(2):
            client.get("/api/shares/{}".format(token))
    rows = database.share_audit_rows(MEETING, team_id)
    assert len(rows) == 2
    assert all(row["result_code"] == "ok" for row in rows)
    assert all(row["ip"] for row in rows)
    record = database.share_link(shares.token_hash(token))
    assert record["view_count"] == 2


def test_list_shares_never_returns_raw_token(isolated_database) -> None:
    database = isolated_database
    _seed(database, _team_id(database))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        token = _create(client, ["report"]).json()["token"]
        listed = client.get("/api/meetings/{}/shares".format(MEETING)).json()
    assert len(listed["items"]) == 1
    item = listed["items"][0]
    assert item["active"] is True
    assert token not in str(listed)
    assert item["token_prefix"] == shares.token_hash(token)[:8]
    assert item["share_id"] > 0


def test_revoke_share_by_id_from_list(isolated_database) -> None:
    """界面拿不到令牌原文（库里只有哈希），所以用列表里的 share_id 撤销。"""
    database = isolated_database
    _seed(database, _team_id(database))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        token = _create(client, ["report"]).json()["token"]
        share_id = client.get("/api/meetings/{}/shares".format(MEETING)).json()["items"][0]["share_id"]
        revoked = client.delete("/api/meetings/{}/shares/{}".format(MEETING, share_id))
    assert revoked.status_code == 200
    with TestClient(main_module.app) as client:
        assert client.get("/api/shares/{}".format(token)).status_code == 410


def test_deleting_meeting_cascades_share_rows(isolated_database) -> None:
    database = isolated_database
    team_id = _team_id(database)
    _seed(database, team_id)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        token = _create(client, ["report"]).json()["token"]
        client.get("/api/shares/{}".format(token))
    database.delete_meeting(MEETING, team_id)
    assert database.share_link(shares.token_hash(token)) is None
    assert database.share_audit_rows(MEETING, team_id) == []


def test_other_api_still_requires_token(isolated_database) -> None:
    """分享只是受限出口：其它 /api/* 仍然必须带团队口令。"""
    with TestClient(main_module.app) as client:
        assert client.get("/api/meetings").status_code == 403
        assert client.get("/api/projects").status_code == 403
        assert client.get("/api/meetings/{}/deliverables".format(MEETING)).status_code == 403


def test_share_page_is_served_without_auth(isolated_database) -> None:
    with TestClient(main_module.app) as client:
        page = client.get("/s/any-token")
    assert page.status_code == 200
    assert "共享的会议纪要" in page.text
    # 页面自身不含团队数据，只通过 /api/shares/{token} 取数
    assert "members" not in page.text

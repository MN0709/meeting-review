"""阶段 14（M3）· R-P2-2 本人声纹注册与 R-P2-3 门控的 API 测试。

覆盖：首次状态、跳过（幂等）、有效语音不足 422、录入成功自动回填、删除回到未解锁、
my-tasks 门控（skipped / not_identified / 手动指认优先），以及回填不调用 LLM。
"""

import os
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

import app.main as main_module
from app.db import Database
from app.models import TeamMeetingReport
from app.speaker import SpeakerObservation

AUTH_HEADERS = {"X-Access-Token": "test-access-token"}
MEETING = "m-owner"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    path = tmp_path / "owner.db"
    database = Database(path)
    database.initialize(main_module.settings.parsed_team_tokens())
    monkeypatch.setattr(main_module, "database", database)
    monkeypatch.setattr(main_module.speaker_recognizer, "enabled", False)
    main_module.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return database, path


def _team(database: Database) -> int:
    team_id = database.authenticate("test-access-token")
    assert team_id is not None
    return team_id


def _meeting_with_owner(database: Database, meeting_id=MEETING, embedding=(1.0, 0.0),
                        speech=30.0) -> int:
    team_id = _team(database)
    database.create_meeting(meeting_id, team_id, "本人识别测试", Path("/tmp/o.wav"))
    database.save_meeting_speakers(meeting_id, team_id, [SpeakerObservation(
        local_label="说话人 1", embedding=list(embedding), speech_seconds=speech,
        excerpts=["我来说两句"],
    )])
    database.update_status(meeting_id, team_id, "完成")
    return team_id


def _enroll(client, monkeypatch, embedding=(1.0, 0.0), speech=25.0):
    monkeypatch.setattr(
        main_module.speaker_recognizer, "extract_owner_embedding",
        lambda path: (list(embedding), speech),
    )
    return client.post(
        "/api/owner/voiceprint",
        files={"file": ("me.wav", b"RIFF-owner-voice", "audio/wav")},
    )


def _llm_rows(path: Path) -> int:
    with sqlite3.connect(path) as connection:
        return connection.execute("SELECT COUNT(*) FROM llm_usage").fetchone()[0]


def test_owner_status_initial(isolated) -> None:
    database, _ = isolated
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.get("/api/owner").json()
    assert body == {
        "enrolled": False, "skipped": False, "name": "我",
        "sample_seconds": None, "enrolled_at": None, "onboarding_done": False,
    }


def test_skip_is_idempotent_and_marks_state(isolated) -> None:
    database, _ = isolated
    _meeting_with_owner(database)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        first = client.post("/api/owner/skip").json()
        second = client.post("/api/owner/skip").json()
        tasks = client.get("/api/meetings/{}/my-tasks".format(MEETING)).json()
    assert first["skipped"] is True and first["onboarding_done"] is True
    assert second["skipped"] is True
    assert tasks["owner_voiceprint_state"] == "skipped"
    assert tasks["self_speaker_set"] is False and tasks["count"] == 0


def test_enroll_rejects_short_speech(isolated, monkeypatch) -> None:
    database, _ = isolated
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = _enroll(client, monkeypatch, speech=5.0)
        status = client.get("/api/owner").json()
    assert response.status_code == 422
    assert "有效语音不足" in response.json()["error"]["message"]
    assert database.owner_voiceprint(_team(database)) is None  # 不写脏数据
    assert status["enrolled"] is False


def test_enroll_success_and_auto_backfill(isolated, monkeypatch) -> None:
    database, path = isolated
    _meeting_with_owner(database, "m-older", embedding=(0.0, 1.0))
    _meeting_with_owner(database, "m-recent-1")
    _meeting_with_owner(database, "m-recent-2")
    before = _llm_rows(path)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = _enroll(client, monkeypatch)
        status = client.get("/api/owner").json()
    assert response.status_code == 200
    body = response.json()
    assert body["enrolled"] is True and body["sample_seconds"] == 25.0
    assert len(body["backfilled"]) == 2  # 只回填最近 2 场
    assert status["enrolled"] is True and status["skipped"] is False
    assert _llm_rows(path) == before  # 回填不产生任何 LLM 调用
    first = database.self_speaker("m-recent-1", _team(database))
    assert first is not None and first["source"] == "voiceprint"
    assert first["self_name"] == "我"


def test_enroll_then_delete_returns_to_locked(isolated, monkeypatch) -> None:
    database, _ = isolated
    _meeting_with_owner(database)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        _enroll(client, monkeypatch)
        deleted = client.delete("/api/owner/voiceprint").json()
        tasks = client.get("/api/meetings/{}/my-tasks".format(MEETING)).json()
    assert deleted["enrolled"] is False
    assert tasks["owner_voiceprint_state"] == "not_enrolled"
    assert database.owner_voiceprint(_team(database)) is None


def test_my_tasks_not_identified_when_enrolled_but_no_match(isolated, monkeypatch) -> None:
    database, _ = isolated
    _meeting_with_owner(database, embedding=(0.0, 1.0))  # 与本人声纹不同
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        _enroll(client, monkeypatch)
        body = client.get("/api/meetings/{}/my-tasks".format(MEETING)).json()
    assert body["owner_voiceprint_state"] == "enrolled"
    assert body["self_speaker_set"] is False
    assert body["reason"] == "not_identified"
    assert body["count"] == 0 and body["items"] == []


def test_manual_designation_wins_over_voiceprint(isolated, monkeypatch) -> None:
    database, _ = isolated
    database.create_meeting(MEETING, _team(database), "本人识别测试", Path("/tmp/o.wav"))
    database.save_meeting_speakers(MEETING, _team(database), [SpeakerObservation(
        local_label="说话人 1", embedding=[1.0, 0.0], speech_seconds=30.0,
        excerpts=["我来说两句"],
    )])
    database.update_status(MEETING, _team(database), "完成")
    # 先手动指认（db 层，等价于整理弹窗），再录入声纹触发回填
    database.set_self_speaker(MEETING, _team(database), label="说话人 1")
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        _enroll(client, monkeypatch)
        body = client.get("/api/meetings/{}/my-tasks".format(MEETING)).json()
    current = database.self_speaker(MEETING, _team(database))
    assert current is not None and current["source"] == "manual"
    assert body["self_speaker_set"] is True


def test_backfill_endpoint_requires_enrollment(isolated) -> None:
    database, _ = isolated
    _meeting_with_owner(database)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post("/api/owner/backfill")
    assert response.status_code == 422


def test_backfill_endpoint_skips_manual_and_limits(isolated) -> None:
    database, _ = isolated
    team_id = _meeting_with_owner(database, "m-manual")
    database.set_self_speaker("m-manual", team_id, label="说话人 1")
    _meeting_with_owner(database, "m-auto")
    database.save_owner_voiceprint(team_id, "我", [1.0, 0.0], "chinese", 25.0)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.post("/api/owner/backfill?limit=5").json()
    assert body["scanned"] == 2
    assert body["skipped_manual"] == 1
    assert body["identified"] == 1
    assert database.self_speaker("m-manual", team_id)["source"] == "manual"


# --- 跨设备补救：用本场会议的声音认领「我」 -----------------------------------


def _meeting_with_member(database: Database, meeting_id: str, embedding=(1.0, 0.0),
                         speech=30.0, name="马宁") -> int:
    team_id = _team(database)
    database.create_meeting(meeting_id, team_id, "认领测试", Path("/tmp/c.wav"))
    database.save_meeting_speakers(meeting_id, team_id, [SpeakerObservation(
        local_label="说话人 1", embedding=list(embedding), speech_seconds=speech,
        excerpts=["我来说两句"],
    )])
    database.confirm_meeting_speaker(
        meeting_id, team_id, "说话人 1", name, "", False, False, "chinese",
    )
    return team_id


def test_claim_from_meeting_saves_voiceprint_and_identifies(isolated, monkeypatch) -> None:
    database, _ = isolated
    team_id = _meeting_with_member(database, "m-claim")
    _meeting_with_member(database, "m-other")
    database.update_status("m-other", team_id, "完成")
    database.save_report("m-other", team_id, TeamMeetingReport(
        overview="x", meeting_points=[],
        decisions=[],
        action_items=[{"task": "写方案", "owner": "马宁", "deadline": "周五"}],
        unresolved_issues=[],
    ))
    # 先用「不像」的麦克风声纹（与会议录音不同设备）→ 认不出来
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        _enroll(client, monkeypatch, embedding=(0.0, 1.0))
        before = client.get("/api/meetings/m-other/my-tasks").json()
        response = client.post(
            "/api/meetings/m-claim/claim-owner-voice",
            json={"local_label": "说话人 1", "consent_confirmed": True},
        )
        assert response.status_code == 200
        after = client.get("/api/meetings/m-other/my-tasks").json()
    assert before["self_speaker_set"] is False
    assert before["reason"] == "not_identified"
    # 认领本场说话人后：声纹被换成同通道的声音 → 另一场也能认出
    record = database.owner_voiceprint(team_id)
    assert record is not None and record["embedding"] == [1.0, 0.0]
    assert after["self_speaker_set"] is True
    assert after["self_name"] == "马宁"
    assert after["count"] == 1 and after["items"][0]["task"] == "写方案"
    # 本场被人工指认，手动优先
    assert database.self_speaker("m-claim", team_id)["source"] == "manual"


def test_claim_requires_consent(isolated) -> None:
    database, _ = isolated
    _meeting_with_member(database, "m-claim")
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post(
            "/api/meetings/m-claim/claim-owner-voice",
            json={"local_label": "说话人 1", "consent_confirmed": False},
        )
    assert response.status_code == 422
    assert database.owner_voiceprint(_team(database)) is None


def test_claim_unknown_speaker_and_short_speech(isolated) -> None:
    database, _ = isolated
    _meeting_with_member(database, "m-claim", speech=5.0)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        unknown = client.post(
            "/api/meetings/m-claim/claim-owner-voice",
            json={"local_label": "说话人 9", "consent_confirmed": True},
        )
        short = client.post(
            "/api/meetings/m-claim/claim-owner-voice",
            json={"local_label": "说话人 1", "consent_confirmed": True},
        )
    assert unknown.status_code == 422
    assert short.status_code == 422 and "太短" in short.json()["error"]["message"]

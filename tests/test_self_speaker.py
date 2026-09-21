"""阶段 9-A（M2）· R-P1.5-5「我」的指认与「我答应的任务」契约测试。

只新增，不改 tests/test_app.py 中的任何既有断言。
覆盖：未指认时的空口径、按成员/按标签指认、清除、非法入参、跨团队隔离、
完全一致匹配（不做模糊匹配）、老接口回归。
"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

import app.main as main_module
from app.db import Database
from app.models import TeamMeetingReport, Transcript, TranscriptSegment
from app.speaker import SpeakerObservation

AUTH_HEADERS = {"X-Access-Token": "test-access-token"}
OTHER_HEADERS = {"X-Access-Token": "other-team-token"}
MEETING = "m-self"


@pytest.fixture(autouse=True)
def isolated_database(monkeypatch, tmp_path):
    database = Database(tmp_path / "self.db")
    database.initialize(main_module.settings.parsed_team_tokens())
    monkeypatch.setattr(main_module, "database", database)
    monkeypatch.setattr(main_module.speaker_recognizer, "enabled", False)
    return database


def _team_id(database: Database, token: str = "test-access-token") -> int:
    team_id = database.authenticate(token)
    assert team_id is not None
    return team_id


def _report(owners) -> TeamMeetingReport:
    return TeamMeetingReport(
        overview="本周确认上线内测，并明确各自的交付责任。",
        meeting_points=["确定本周上线内测"],
        decisions=[{
            "content": "本周上线内测",
            "decision_maker": "胡泊",
            "evidence": {"quote": "我们本周上线内测", "timestamp": "00:00:02"},
        }],
        action_items=[
            {"task": "准备发布清单 {}".format(index), "owner": owner, "deadline": "周五"}
            for index, owner in enumerate(owners)
        ],
        unresolved_issues=[],
    )


def _seed(
    database: Database, team_id: int, *, meeting_id: str = MEETING,
    owners=("胡泊", "马宁", "未明确"),
) -> None:
    database.create_meeting(meeting_id, team_id, "周会", Path("/tmp/{}.wav".format(meeting_id)))
    database.save_transcript(
        meeting_id, team_id,
        Transcript(language="zh", duration_seconds=20.0, segments=[
            TranscriptSegment(start=0.0, end=5.0, speaker_label="说话人 1", text="我们本周上线内测"),
            TranscriptSegment(start=5.0, end=10.0, speaker_label="说话人 2", text="我来准备发布清单"),
        ]),
    )
    database.save_report(meeting_id, team_id, _report(owners))
    database.save_meeting_speakers(meeting_id, team_id, [
        SpeakerObservation(local_label="说话人 1", embedding=None, speech_seconds=5.0, excerpts=["我们本周上线内测"]),
        SpeakerObservation(local_label="说话人 2", embedding=None, speech_seconds=5.0, excerpts=["我来准备发布清单"]),
    ])
    database.clear_audio_path(meeting_id, team_id)


def _add_member(database: Database, team_id: int, name: str) -> int:
    with database._lock, database._connect() as connection:  # 测试直接造一条成员
        connection.execute(
            "INSERT INTO members(team_id,name,role,is_key_decision_maker,created_at) VALUES(?,?,?,?,?)",
            (team_id, name, "", 0, "2026-09-21T00:00:00+00:00"),
        )
        row = connection.execute(
            "SELECT id FROM members WHERE team_id=? AND name=?", (team_id, name)
        ).fetchone()
    return int(row["id"])


def _bind_speaker_to_member(database: Database, team_id: int, label: str, member_id: int) -> None:
    with database._lock, database._connect() as connection:
        connection.execute(
            "UPDATE meeting_speakers SET member_id=?,status='已确认' WHERE meeting_id=? AND team_id=? AND local_label=?",
            (member_id, MEETING, team_id, label),
        )


# --------------------------------------------------------------------------
# 未指认：一律为空，绝不推断（红线 8）
# --------------------------------------------------------------------------


def test_my_tasks_without_self_speaker_is_empty(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.get("/api/meetings/{}/my-tasks".format(MEETING)).json()
    assert body["self_speaker_set"] is False
    assert body["count"] == 0
    assert body["items"] == []
    assert body["self_name"] is None
    assert body["owner_unknown"] == 1  # 「未明确」那条


def test_history_without_self_speaker_has_null_field(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        body = client.get("/api/meetings/{}".format(MEETING)).json()
    assert body["self_speaker"] is None
    # 老字段语义不变（回归）
    assert body["report"]["overview"]
    assert len(body["transcript"]) == 2


def test_unnamed_speaker_never_becomes_self_by_guess(isolated_database) -> None:
    """负例：本场有说话人但从未指认 → 任何地方都不得出现猜测的「我」。"""
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id, owners=("说话人 1",))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        tasks = client.get("/api/meetings/{}/my-tasks".format(MEETING)).json()
        history = client.get("/api/meetings/{}".format(MEETING)).json()
    assert tasks["count"] == 0 and tasks["self_speaker_set"] is False
    assert history["self_speaker"] is None


# --------------------------------------------------------------------------
# 指认：按成员 / 按说话人标签
# --------------------------------------------------------------------------


def test_self_speaker_by_local_label_matches_owner_exactly(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id, owners=("说话人 1", "马宁", "未明确"))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        saved = client.post(
            "/api/meetings/{}/self-speaker".format(MEETING), json={"local_label": "说话人 1"},
        )
        assert saved.status_code == 200
        assert saved.json() == {
            "self_speaker_set": True, "local_label": "说话人 1", "member_id": None,
            "member_name": None, "self_name": "说话人 1",
        }
        tasks = client.get("/api/meetings/{}/my-tasks".format(MEETING)).json()
    assert tasks["self_speaker_set"] is True
    assert tasks["self_name"] == "说话人 1"
    assert tasks["count"] == 1
    assert tasks["items"][0]["task"] == "准备发布清单 0"
    assert tasks["owner_unknown"] == 1


def test_self_speaker_by_member_id_uses_member_name(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id, owners=("胡泊", "马宁"))
    member_id = _add_member(isolated_database, team_id, "胡泊")
    _bind_speaker_to_member(isolated_database, team_id, "说话人 1", member_id)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        saved = client.post(
            "/api/meetings/{}/self-speaker".format(MEETING), json={"member_id": member_id},
        ).json()
        tasks = client.get("/api/meetings/{}/my-tasks".format(MEETING)).json()
    assert saved["self_name"] == "胡泊" and saved["local_label"] == "说话人 1"
    assert [item["task"] for item in tasks["items"]] == ["准备发布清单 0"]


def test_owner_match_is_exact_not_fuzzy(isolated_database) -> None:
    """「胡泊老师」不得匹配「胡泊」——不许模糊匹配、不许猜。"""
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id, owners=("胡说", "胡泊老师"))
    member_id = _add_member(isolated_database, team_id, "胡泊")
    _bind_speaker_to_member(isolated_database, team_id, "说话人 1", member_id)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        client.post("/api/meetings/{}/self-speaker".format(MEETING), json={"member_id": member_id})
        tasks = client.get("/api/meetings/{}/my-tasks".format(MEETING)).json()
    assert tasks["count"] == 0


def test_self_speaker_can_be_changed(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id, owners=("说话人 1", "说话人 2"))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        client.post("/api/meetings/{}/self-speaker".format(MEETING), json={"local_label": "说话人 1"})
        client.post("/api/meetings/{}/self-speaker".format(MEETING), json={"local_label": "说话人 2"})
        tasks = client.get("/api/meetings/{}/my-tasks".format(MEETING)).json()
    assert tasks["self_name"] == "说话人 2"
    assert [item["task"] for item in tasks["items"]] == ["准备发布清单 1"]


def test_self_speaker_can_be_cleared(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id, owners=("说话人 1",))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        client.post("/api/meetings/{}/self-speaker".format(MEETING), json={"local_label": "说话人 1"})
        cleared = client.post("/api/meetings/{}/self-speaker".format(MEETING), json={})
        tasks = client.get("/api/meetings/{}/my-tasks".format(MEETING)).json()
        history = client.get("/api/meetings/{}".format(MEETING)).json()
    assert cleared.json() == {
        "self_speaker_set": False, "local_label": None, "member_id": None,
        "member_name": None, "self_name": None,
    }
    assert tasks["count"] == 0 and tasks["self_speaker_set"] is False
    assert history["self_speaker"] is None


# --------------------------------------------------------------------------
# 入参校验与隔离
# --------------------------------------------------------------------------


def test_unknown_local_label_is_rejected(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post(
            "/api/meetings/{}/self-speaker".format(MEETING), json={"local_label": "说话人 9"},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_args"


def test_unknown_member_id_is_rejected(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post(
            "/api/meetings/{}/self-speaker".format(MEETING), json={"member_id": 99999},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_args"


def test_member_not_in_this_meeting_is_rejected(isolated_database) -> None:
    team_id = _team_id(isolated_database)
    _seed(isolated_database, team_id)
    member_id = _add_member(isolated_database, team_id, "未参会的人")
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post(
            "/api/meetings/{}/self-speaker".format(MEETING), json={"member_id": member_id},
        )
    assert response.status_code == 422


def test_missing_meeting_returns_not_found(isolated_database) -> None:
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        assert client.get("/api/meetings/nope/my-tasks").status_code == 404
        assert client.post("/api/meetings/nope/self-speaker", json={}).status_code == 404


@pytest.mark.parametrize("method,path", [
    ("get", "/api/meetings/{}/my-tasks"),
    ("post", "/api/meetings/{}/self-speaker"),
])
def test_other_team_is_forbidden(isolated_database, method, path) -> None:
    other_team = _team_id(isolated_database, "other-team-token")
    _seed(isolated_database, other_team)

    def call(client):
        if method == "post":
            return client.post(path.format(MEETING), json={})
        return client.get(path.format(MEETING))

    with TestClient(main_module.app, headers=OTHER_HEADERS) as client:
        assert call(client).status_code == 200  # 本团队自己访问正常
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = call(client)
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "team_forbidden"


def test_self_speaker_survives_missing_report(isolated_database) -> None:
    """报告还没生成时，指认接口应给可读提示，而不是 500。"""
    team_id = _team_id(isolated_database)
    isolated_database.create_meeting(MEETING, team_id, "无报告会议", Path("/tmp/m.wav"))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post(
            "/api/meetings/{}/self-speaker".format(MEETING), json={"local_label": "说话人 1"},
        )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"

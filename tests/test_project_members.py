"""阶段 13（M2）· R-P2-4 项目级声纹库的契约与隔离测试。

红线 13：项目间成员声纹不得串用；A 项目成员不出现在 B 项目。
"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

import app.main as main_module
from app.db import Database
from app.speaker import SpeakerObservation

AUTH_HEADERS = {"X-Access-Token": "test-access-token"}


@pytest.fixture(autouse=True)
def isolated_database(monkeypatch, tmp_path):
    database = Database(tmp_path / "proj-members.db")
    database.initialize(main_module.settings.parsed_team_tokens())
    monkeypatch.setattr(main_module, "database", database)
    monkeypatch.setattr(main_module.speaker_recognizer, "enabled", False)
    return database


def _team(database: Database) -> int:
    team_id = database.authenticate("test-access-token")
    assert team_id is not None
    return team_id


def _observation(label: str, embedding: list) -> SpeakerObservation:
    return SpeakerObservation(
        local_label=label, embedding=embedding, speech_seconds=30.0,
        excerpts=["测试原话"],
    )


def _two_projects_with_same_name_member(database: Database):
    """两个项目各有一个「张总」，声纹不同。"""
    team_id = _team(database)
    project_a = database.create_project("pa", team_id, "A 项目")
    project_b = database.create_project("pb", team_id, "B 项目")
    database.create_meeting("ma", team_id, "A 会", Path("/tmp/a.wav"), project_id=project_a.id)
    database.create_meeting("mb", team_id, "B 会", Path("/tmp/b.wav"), project_id=project_b.id)
    database.save_meeting_speakers("ma", team_id, [_observation("说话人 1", [1.0, 0.0])])
    database.save_meeting_speakers("mb", team_id, [_observation("说话人 1", [0.0, 1.0])])
    a = database.confirm_meeting_speaker("ma", team_id, "说话人 1", "张总", "领导", True, True, "chinese")
    b = database.confirm_meeting_speaker("mb", team_id, "说话人 1", "张总", "领导", True, True, "chinese")
    assert a is not None and b is not None
    database.finalize_meeting_voiceprints("ma", team_id, "chinese", consent_confirmed=True)
    database.finalize_meeting_voiceprints("mb", team_id, "chinese", consent_confirmed=True)
    return team_id, project_a, project_b, a[0], b[0]


# --- 数据层：项目隔离 -------------------------------------------------------


def test_same_name_in_two_projects_are_different_members(isolated_database) -> None:
    database = isolated_database
    _, project_a, project_b, member_a, member_b = _two_projects_with_same_name_member(database)
    assert member_a != member_b
    assert [m.name for m in database.list_project_members(project_a.id, _team(database))] == ["张总"]
    assert [m.name for m in database.list_project_members(project_b.id, _team(database))] == ["张总"]
    assert database.member_project_id(member_a) == project_a.id
    assert database.member_project_id(member_b) == project_b.id


def test_voice_profiles_are_project_scoped(isolated_database) -> None:
    """红线 13：A 项目的声纹不会出现在 B 项目。"""
    database = isolated_database
    team_id, project_a, project_b, _, _ = _two_projects_with_same_name_member(database)
    a_profiles = database.voice_profiles(team_id, "chinese", project_id=project_a.id)
    b_profiles = database.voice_profiles(team_id, "chinese", project_id=project_b.id)
    assert len(a_profiles) == 1 and a_profiles[0]["embedding"] == [1.0, 0.0]
    assert len(b_profiles) == 1 and b_profiles[0]["embedding"] == [0.0, 1.0]
    # 未归类桶为空
    assert database.voice_profiles(team_id, "chinese", project_id=None) == []
    # 旧行为（不传项目）仍返回全部，仅供历史调用
    assert len(database.voice_profiles(team_id, "chinese")) == 2


def test_unclassified_member_moves_with_meeting(isolated_database) -> None:
    """R-P2-4 ④：未归类时成员暂存「未归类」，会议归入项目后随会议迁移。"""
    database = isolated_database
    team_id = _team(database)
    project = database.create_project("px", team_id, "目标项目")
    database.create_meeting("mu", team_id, "未归类会议", Path("/tmp/u.wav"))
    database.save_meeting_speakers("mu", team_id, [_observation("说话人 1", [0.5, 0.5])])
    result = database.confirm_meeting_speaker(
        "mu", team_id, "说话人 1", "李四", "", False, True, "chinese"
    )
    member_id = result[0]
    assert database.member_project_id(member_id) is None
    assert database.move_meeting("mu", team_id, project.id) is True
    assert database.member_project_id(member_id) == project.id


# --- API 层：项目级端点 -----------------------------------------------------


def test_project_member_list_is_scoped(isolated_database) -> None:
    database = isolated_database
    _, project_a, project_b, member_a, _ = _two_projects_with_same_name_member(database)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        a_list = client.get("/api/projects/{}/members".format(project_a.id)).json()
        b_list = client.get("/api/projects/{}/members".format(project_b.id)).json()
    assert [item["id"] for item in a_list] == [member_a]
    assert member_a not in [item["id"] for item in b_list]
    assert all(item["project_id"] == project_a.id for item in a_list)


def test_cannot_touch_member_of_another_project(isolated_database) -> None:
    database = isolated_database
    _, project_a, project_b, member_a, _ = _two_projects_with_same_name_member(database)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        assert client.patch(
            "/api/projects/{}/members/{}".format(project_b.id, member_a),
            json={"name": "改个名"},
        ).status_code == 404
        assert client.delete(
            "/api/projects/{}/members/{}/voiceprint".format(project_b.id, member_a)
        ).status_code == 404
        assert client.post(
            "/api/projects/{}/members/{}/merge".format(project_b.id, member_a),
            json={"target_member_id": member_a},
        ).status_code == 404


def test_project_member_update_and_voiceprint_delete(isolated_database) -> None:
    database = isolated_database
    _, project_a, _, member_a, _ = _two_projects_with_same_name_member(database)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        updated = client.patch(
            "/api/projects/{}/members/{}".format(project_a.id, member_a),
            json={"name": "张总（改名）", "role": "决策人", "is_key_decision_maker": True},
        )
        assert updated.status_code == 200
        assert updated.json()["name"] == "张总（改名）"
        assert updated.json()["project_id"] == project_a.id
        deleted = client.delete(
            "/api/projects/{}/members/{}/voiceprint".format(project_a.id, member_a)
        )
        assert deleted.status_code == 200
    # 只删本项目声纹；成员仍在，历史会议数据不变
    members = database.list_project_members(project_a.id, _team(database))
    assert [m.name for m in members] == ["张总（改名）"]
    assert members[0].has_voiceprint is False


def test_project_member_merge_stays_inside_project(isolated_database) -> None:
    database = isolated_database
    team_id, project_a, _, member_a, _ = _two_projects_with_same_name_member(database)
    database.create_meeting("ma2", team_id, "A 二会", Path("/tmp/a2.wav"), project_id=project_a.id)
    database.save_meeting_speakers("ma2", team_id, [_observation("说话人 1", [1.0, 0.2])])
    dup = database.confirm_meeting_speaker(
        "ma2", team_id, "说话人 1", "张总重复", "", False, True, "chinese"
    )
    dup_id = dup[0]
    assert database.member_project_id(dup_id) == project_a.id
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        merged = client.post(
            "/api/projects/{}/members/{}/merge".format(project_a.id, dup_id),
            json={"target_member_id": member_a},
        )
        assert merged.status_code == 200
    assert all(m.id != dup_id for m in database.list_project_members(project_a.id, team_id))


def test_project_members_requires_project_ownership(isolated_database) -> None:
    database = isolated_database
    other_team = database.authenticate("other-team-token")
    project = database.create_project("other", other_team, "别人的项目")
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        assert client.get("/api/projects/{}/members".format(project.id)).status_code == 403

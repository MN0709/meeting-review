"""阶段 13（M2）· R-P2-11 旧数据迁移的测试。

覆盖：dry-run 不写数据、幂等、数量对账、成员归位、speaker_profiles 跟随、
热词导出后清空、工作区改名、schema_version 落账。
"""

import json
import os
import sqlite3
import time

os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

from app.db import Database
from app.migrate_v2 import SCHEMA_VERSION, migrate


def _legacy_db(tmp_path):
    path = tmp_path / "legacy.db"
    database = Database(path)
    database.initialize({"产品项目组": "token-a"}, owner_name="我", auth_enabled=False)
    team_id = database.authenticate("token-a")
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO projects(id,team_id,name,parent_id,created_at) VALUES('p1',?,'李总',NULL,'2026-01-01')",
            (team_id,),
        )
        connection.execute(
            "INSERT INTO projects(id,team_id,name,parent_id,created_at) VALUES('p2',?,'张总',NULL,'2026-01-01')",
            (team_id,),
        )
        for meeting_id, project in (("m1", "p1"), ("m2", "p1"), ("m3", "p2"), ("m4", None)):
            connection.execute(
                """INSERT INTO meetings(id,team_id,title,project_id,audio_path,status,created_at)
                   VALUES(?,?,?,?,NULL,'完成','2026-01-01')""",
                (meeting_id, team_id, meeting_id, project),
            )
        connection.execute(
            """INSERT INTO members(id,team_id,name,role,is_key_decision_maker,created_at,project_id)
               VALUES(1,?,'马宁','',0,'2026-01-01',NULL)""",
            (team_id,),
        )
        connection.execute(
            """INSERT INTO members(id,team_id,name,role,is_key_decision_maker,created_at,project_id)
               VALUES(2,?,'小王','',0,'2026-01-01',NULL)""",
            (team_id,),
        )
        # 马宁出现在 p1 两场 + p2 一场 → 归 p1；小王只出现在 p2 → 归 p2
        for meeting_id, label, member_id in (
            ("m1", "说话人 1", 1), ("m2", "说话人 1", 1),
            ("m3", "说话人 1", 1), ("m3", "说话人 2", 2),
        ):
            connection.execute(
                """INSERT INTO meeting_speakers(
                       meeting_id,team_id,local_label,member_id,confidence,status,
                       embedding_json,speech_seconds,excerpts_json,created_at,updated_at)
                   VALUES(?,?,?,?,NULL,'已确认','[0.1,0.2]',10.0,'[]','2026-01-01','2026-01-01')""",
                (meeting_id, team_id, label, member_id),
            )
        connection.execute(
            """INSERT INTO speaker_profiles(
                   team_id,member_id,embedding_json,model_version,consented_at,updated_at,project_id)
               VALUES(?,1,'[0.1,0.2]','chinese','2026-01-01','2026-01-01',NULL)""",
            (team_id,),
        )
        connection.execute(
            "INSERT INTO team_terms(team_id,term,note,created_at,updated_at) VALUES(?,?,?,?,?)",
            (team_id, "会脉", "", "2026-01-01", "2026-01-01"),
        )
    return path


def _member_projects(path):
    with sqlite3.connect(path) as connection:
        return {row[0]: row[1] for row in connection.execute("SELECT id, project_id FROM members")}


def _scalar(path, sql):
    with sqlite3.connect(path) as connection:
        return connection.execute(sql).fetchone()[0]


def test_dry_run_changes_nothing(tmp_path) -> None:
    path = _legacy_db(tmp_path)
    before = _member_projects(path)
    result = migrate(str(path), owner_name="我", dry_run=True)
    assert result["dry_run"] is True
    assert result["planned"]["members"]  # 计划里能看到成员要归位
    assert _member_projects(path) == before  # 实际数据未动
    assert _scalar(path, "SELECT COUNT(*) FROM team_terms") == 1
    assert _scalar(path, "SELECT COUNT(*) FROM projects") == 2


def test_migration_assigns_members_and_preserves_counts(tmp_path) -> None:
    path = _legacy_db(tmp_path)
    result = migrate(str(path), owner_name="我", dry_run=False)
    before, after = result["counts_before"], result["counts_after"]
    # 数量对账：会议/报告/转写/成员不丢
    for table in ("meetings", "reports", "transcripts", "members"):
        assert before[table] == after[table]
    # 未归类会议 m4 → 以旧团队名命名的项目（项目数 +1）
    assert after["projects"] == before["projects"] + 1
    projects = _member_projects(path)
    assert projects[1] == "p1"  # 马宁 → 李总（p1）
    assert projects[2] == "p2"  # 小王 → 张总（p2）
    assert _scalar(path, "SELECT project_id FROM speaker_profiles WHERE member_id=1") == "p1"
    # m4 归入「产品项目组」项目；m1/m2/m3 不变
    assert _scalar(path, "SELECT project_id FROM meetings WHERE id='m4'") is not None
    assert _scalar(path, "SELECT name FROM projects WHERE id=(SELECT project_id FROM meetings WHERE id='m4')") == "产品项目组"
    # 热词导出并清空
    assert _scalar(path, "SELECT COUNT(*) FROM team_terms") == 0
    assert result["applied"]["terms_cleared"] == 1
    backup = result["applied"]["terms_backup"]
    assert json.loads(open(backup, encoding="utf-8").read())[0]["term"] == "会脉"
    # 工作区改名 + schema 版本
    assert _scalar(path, "SELECT name FROM teams") == "我"
    with sqlite3.connect(path) as connection:
        version = connection.execute(
            "SELECT value FROM app_state WHERE key='schema_version'"
        ).fetchone()[0]
    assert version == SCHEMA_VERSION
    assert result["backup"] and os.path.exists(result["backup"])


def test_migration_is_idempotent(tmp_path) -> None:
    path = _legacy_db(tmp_path)
    migrate(str(path), owner_name="我")
    first = _member_projects(path)
    counts_first = {
        table: _scalar(path, "SELECT COUNT(*) FROM {}".format(table))
        for table in ("meetings", "members", "projects")
    }
    time.sleep(0.01)
    second = migrate(str(path), owner_name="我")
    assert _member_projects(path) == first
    assert second["planned"]["members"] == []
    assert second["planned"]["projects"] == []
    assert second["planned"]["terms"] == []
    assert second["planned"]["unclassified"]["count"] == 0
    counts_second = {
        table: _scalar(path, "SELECT COUNT(*) FROM {}".format(table))
        for table in ("meetings", "members", "projects")
    }
    assert counts_first == counts_second


def test_migration_without_any_data_is_safe(tmp_path) -> None:
    path = tmp_path / "empty.db"
    Database(path).initialize({}, owner_name="我", auth_enabled=False)
    result = migrate(str(path), owner_name="我")
    assert result["counts_before"]["meetings"] == 0
    assert result["counts_after"]["meetings"] == 0

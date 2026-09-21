"""R-P2-11：旧数据一次性迁移（团队版 → 个人版）。

规则（PRD 2.0 §4.2）：
1. 迁移前整库备份到 `data/meeting-review.db.bak-<YYYYMMDD-HHMMSS>`；
2. 解析个人工作区（`app_state.owner_workspace_id`，没有则用唯一团队）；
3. 旧项目 → 个人项目；同名冲突加「<旧团队名>·」前缀；
4. `project_id` 为空的旧会议 → 归入以旧团队名命名的项目；
5. 成员按其会议出现次数最多的项目归位；判断不出留在「未归类」桶（`project_id` 为空）；
6. `speaker_profiles.project_id` 跟随其成员；
7. 导出并清空 `team_terms`；
8. 写 `app_state.schema_version='2.0'`，并把个人工作区改名为 `OWNER_NAME`。

特性：**幂等**（重复执行不新增项目/成员）、**--dry-run** 只打印将发生的变更。
用法：
    python -m app.migrate_v2 --dry-run
    python -m app.migrate_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "2.0"
COUNTED_TABLES = ("meetings", "reports", "transcripts", "members", "projects", "speaker_profiles")


def _ensure_columns(connection: sqlite3.Connection) -> None:
    """补上 R-P2-4 需要的可空列（与 App 启动时的 _ensure_column 一致，幂等）。"""
    for table, column, ddl in (
        ("members", "project_id", "TEXT"),
        ("speaker_profiles", "project_id", "TEXT"),
    ):
        if not _table_exists(connection, table):
            continue
        existing = {row[1] for row in connection.execute("PRAGMA table_info({})".format(table))}
        if column not in existing:
            connection.execute("ALTER TABLE {} ADD COLUMN {} {}".format(table, column, ddl))


def _counts(connection: sqlite3.Connection) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for table in COUNTED_TABLES:
        try:
            counts[table] = int(connection.execute("SELECT COUNT(*) FROM {}".format(table)).fetchone()[0])
        except sqlite3.OperationalError:
            counts[table] = -1
    return counts


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _app_state(connection: sqlite3.Connection, key: str) -> Optional[str]:
    if not _table_exists(connection, "app_state"):
        return None
    row = connection.execute("SELECT value FROM app_state WHERE key=?", (key,)).fetchone()
    return str(row[0]) if row else None


def _set_app_state(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        """INSERT INTO app_state(key, value, updated_at) VALUES(?,?,?)
           ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
        (key, value, datetime.now().isoformat()),
    )


def resolve_workspace(connection: sqlite3.Connection, owner_name: str, persist: bool = True) -> int:
    recorded = _app_state(connection, "owner_workspace_id")
    if recorded is not None:
        row = connection.execute("SELECT id FROM teams WHERE id=?", (recorded,)).fetchone()
        if row:
            return int(row[0])
    rows = connection.execute("SELECT id FROM teams ORDER BY id LIMIT 2").fetchall()
    if len(rows) == 1:
        workspace = int(rows[0][0])
        if persist:
            _set_app_state(connection, "owner_workspace_id", str(workspace))
        return workspace
    if not persist:
        # dry-run 不得写数据；多团队且未记录时，只用第一个团队做“预演”。
        if rows:
            return int(rows[0][0])
        raise SystemExit("既没有 App 记录的个人工作区，也没有可用的团队；请先启动一次服务")
    cursor = connection.execute(
        "INSERT INTO teams(name, token_hash, created_at) VALUES(?,?,?)",
        (owner_name, "migrated", datetime.now().isoformat()),
    )
    workspace = int(cursor.lastrowid)
    _set_app_state(connection, "owner_workspace_id", str(workspace))
    return workspace


def build_plan(
    connection: sqlite3.Connection, owner_name: str = "我", persist_workspace: bool = True,
) -> Dict[str, Any]:
    """只读地计算迁移计划，供 --dry-run 展示与执行复用。"""
    workspace = resolve_workspace(connection, owner_name, persist=persist_workspace)
    teams = {int(row[0]): str(row[1]) for row in connection.execute("SELECT id,name FROM teams")}
    workspace_old_name = teams.get(workspace, owner_name)
    legacy_name = _app_state(connection, "legacy_team_name") or workspace_old_name

    plan: Dict[str, Any] = {
        "workspace_id": workspace,
        "workspace_old_name": workspace_old_name,
        "legacy_team_name": legacy_name,
        "rename_workspace_to": owner_name,
        "projects": [],
        "unclassified": {"count": 0, "fallback_project": None},
        "members": [],
        "terms": [],
        "already_migrated": _app_state(connection, "schema_version") == SCHEMA_VERSION,
    }

    # 3. 项目归位与重命名
    rows = connection.execute(
        "SELECT id, team_id, name FROM projects ORDER BY team_id, id"
    ).fetchall()
    existing_names = {str(r[2]) for r in rows if int(r[1]) == workspace}
    for project_id, team_id, name in rows:
        team_id, name = int(team_id), str(name)
        new_name = name
        if team_id != workspace and name in existing_names:
            new_name = "{}·{}".format(teams.get(team_id, "旧团队"), name)
        if team_id != workspace or new_name != name:
            plan["projects"].append({
                "project_id": project_id,
                "from_team": team_id,
                "old_name": name,
                "new_name": new_name,
            })
        existing_names.add(new_name)

    # 4. 未归类会议 → 以旧团队名命名的项目
    unclassified = connection.execute(
        "SELECT team_id, COUNT(*) FROM meetings WHERE project_id IS NULL GROUP BY team_id"
    ).fetchall()
    if unclassified:
        plan["unclassified"]["count"] = sum(int(row[1]) for row in unclassified)
        plan["unclassified"]["fallback_project"] = legacy_name
        plan["unclassified"]["by_team"] = [
            {"team_id": int(row[0]), "team_name": teams.get(int(row[0]), "未知团队"), "meetings": int(row[1])}
            for row in unclassified
        ]

    # 5. 成员归位：按其会议出现次数最多的项目
    member_rows = connection.execute(
        "SELECT id, team_id, name, project_id FROM members ORDER BY id"
    ).fetchall()
    for member_id, team_id, name, current_project in member_rows:
        counts = connection.execute(
            """SELECT m.project_id AS pid, COUNT(*) AS n
               FROM meeting_speakers s
               JOIN meetings m ON m.id = s.meeting_id
               WHERE s.member_id=? AND s.team_id=? AND m.project_id IS NOT NULL
               GROUP BY m.project_id
               ORDER BY n DESC, m.project_id""",
            (int(member_id), int(team_id)),
        ).fetchall()
        if not counts:
            target = None
        elif len(counts) > 1 and int(counts[0][1]) == int(counts[1][1]):
            target = None  # 平分秋色 → 不猜，留在未归类桶
        else:
            target = str(counts[0][0])
        if target != current_project:
            plan["members"].append({
                "member_id": int(member_id),
                "name": str(name),
                "from_project": current_project,
                "to_project": target,
            })

    # 7. 热词导出后清空
    if _table_exists(connection, "team_terms"):
        plan["terms"] = [
            {"team_id": int(row[0]), "term": str(row[1]), "note": str(row[2] or "")}
            for row in connection.execute("SELECT team_id, term, note FROM team_terms ORDER BY id")
        ]

    return plan


def _apply(
    connection: sqlite3.Connection, plan: Dict[str, Any], backup_dir: Path,
) -> Dict[str, Any]:
    workspace = plan["workspace_id"]
    applied: Dict[str, Any] = {"projects_moved": 0, "projects_renamed": 0, "unclassified_assigned": 0,
                               "members_assigned": 0, "terms_cleared": 0, "profiles_synced": 0}
    for change in plan["projects"]:
        connection.execute(
            "UPDATE projects SET team_id=?, name=? WHERE id=?",
            (workspace, change["new_name"], change["project_id"]),
        )
        if change["from_team"] != workspace:
            applied["projects_moved"] += 1
        if change["new_name"] != change["old_name"]:
            applied["projects_renamed"] += 1

    # 未归类会议 → 以旧团队名命名的项目（每个旧团队一个）
    if plan["unclassified"]["count"]:
        for entry in plan["unclassified"].get("by_team", []):
            project_name = entry["team_name"]
            row = connection.execute(
                "SELECT id FROM projects WHERE team_id=? AND name=?", (workspace, project_name)
            ).fetchone()
            if row:
                project_id = str(row[0])
            else:
                project_id = "migrated-{}".format(
                    hashlib.sha1(project_name.encode("utf-8")).hexdigest()[:12]
                )
                connection.execute(
                    """INSERT INTO projects(id, team_id, name, parent_id, created_at)
                       VALUES(?,?,?,NULL,?)""",
                    (project_id, workspace, project_name, datetime.now().isoformat()),
                )
            cursor = connection.execute(
                "UPDATE meetings SET project_id=? WHERE project_id IS NULL AND team_id=?",
                (project_id, entry["team_id"]),
            )
            applied["unclassified_assigned"] += cursor.rowcount

    for change in plan["members"]:
        connection.execute(
            "UPDATE members SET project_id=? WHERE id=?",
            (change["to_project"], change["member_id"]),
        )
        applied["members_assigned"] += 1

    # speaker_profiles.project_id 跟随 member
    cursor = connection.execute(
        """UPDATE speaker_profiles SET project_id=(
               SELECT project_id FROM members WHERE members.id = speaker_profiles.member_id)
           WHERE project_id IS NOT (
               SELECT project_id FROM members WHERE members.id = speaker_profiles.member_id)"""
    )
    applied["profiles_synced"] = cursor.rowcount

    if plan["terms"]:
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = backup_dir / "team_terms-{}.json".format(stamp)
        target.write_text(json.dumps(plan["terms"], ensure_ascii=False, indent=2), encoding="utf-8")
        connection.execute("DELETE FROM team_terms")
        applied["terms_cleared"] = len(plan["terms"])
        applied["terms_backup"] = str(target)

    connection.execute(
        "UPDATE teams SET name=? WHERE id=?", (plan["rename_workspace_to"], workspace)
    )
    _set_app_state(connection, "legacy_team_name", plan["legacy_team_name"])
    _set_app_state(connection, "schema_version", SCHEMA_VERSION)
    return applied


def migrate(
    database_path: str, owner_name: str = "我", dry_run: bool = False,
) -> Dict[str, Any]:
    path = Path(database_path)
    if not path.exists():
        raise SystemExit("数据库不存在：{}".format(path))
    backup_path: Optional[Path] = None
    connection = sqlite3.connect(str(path))
    try:
        _ensure_columns(connection)
        before = _counts(connection)
        plan = build_plan(connection, owner_name, persist_workspace=not dry_run)
        result: Dict[str, Any] = {
            "database": str(path), "dry_run": dry_run,
            "workspace_id": plan["workspace_id"],
            "workspace_old_name": plan["workspace_old_name"],
            "legacy_team_name": plan["legacy_team_name"],
            "planned": {
                "projects": plan["projects"],
                "unclassified": plan["unclassified"],
                "members": plan["members"],
                "terms": plan["terms"],
            },
            "counts_before": before,
        }
        if dry_run:
            result["counts_after"] = before
            result["applied"] = {}
            return result

        if not plan["already_migrated"] or plan["projects"] or plan["terms"]:
            backup_path = path.with_name(
                "{}.bak-{}".format(path.name, datetime.now().strftime("%Y%m%d-%H%M%S"))
            )
            shutil.copy2(path, backup_path)
        with connection:
            applied = _apply(connection, plan, path.parent / "backup")
        result["backup"] = str(backup_path) if backup_path else None
        result["applied"] = applied
        result["counts_after"] = _counts(connection)
        return result
    finally:
        connection.close()


def main(argv: Optional[List[str]] = None) -> int:
    from app.config import get_settings

    parser = argparse.ArgumentParser(description="R-P2-11 个人版数据迁移（幂等，可 dry-run）")
    parser.add_argument("--dry-run", action="store_true", help="只打印将发生的变更，不写任何数据")
    parser.add_argument("--database", default=None, help="SQLite 路径；默认取 DATABASE_PATH")
    parser.add_argument("--owner-name", default=None, help="个人工作区名；默认取 OWNER_NAME")
    args = parser.parse_args(argv)
    settings = get_settings()
    result = migrate(
        args.database or settings.database_path,
        owner_name=args.owner_name or settings.owner_name,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

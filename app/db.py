from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

from app.models import (
    ActionStatus, MeetingHistory, MeetingListItem, MeetingSource, ProjectListItem,
    ProjectMemory, ProjectMemoryAction, ProjectMemoryDecision, ProjectMemoryIssue,
    TeamMeetingReport, Transcript, TranscriptSegment,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class ProjectHasActiveMeetingsError(RuntimeError):
    pass


class ProjectHasChildrenError(RuntimeError):
    pass


class Database:
    """Small SQLite repository. Every public resource read is team-scoped."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._tokens: Dict[str, int] = {}

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection, table: str, column: str, definition: str
    ) -> None:
        columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def initialize(self, team_tokens: Dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS teams(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    token_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS members(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id INTEGER NOT NULL REFERENCES teams(id),
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS projects(
                    id TEXT PRIMARY KEY,
                    team_id INTEGER NOT NULL REFERENCES teams(id),
                    name TEXT NOT NULL,
                    parent_id TEXT REFERENCES projects(id),
                    created_at TEXT NOT NULL,
                    UNIQUE(team_id, name)
                );
                CREATE TABLE IF NOT EXISTS meetings(
                    id TEXT PRIMARY KEY,
                    team_id INTEGER NOT NULL REFERENCES teams(id),
                    title TEXT NOT NULL,
                    project_id TEXT REFERENCES projects(id),
                    audio_path TEXT,
                    duration_seconds REAL NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS transcripts(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
                    start REAL NOT NULL,
                    end REAL NOT NULL,
                    speaker_label TEXT,
                    text TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reports(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id TEXT NOT NULL UNIQUE REFERENCES meetings(id) ON DELETE CASCADE,
                    json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS meeting_action_items(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
                    team_id INTEGER NOT NULL REFERENCES teams(id),
                    item_index INTEGER NOT NULL,
                    task TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    deadline TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT '待确认'
                        CHECK(status IN ('待确认','进行中','已完成','已取消')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(meeting_id, item_index)
                );
                """
            )
            self._ensure_column(connection, "meetings", "project_id", "TEXT REFERENCES projects(id)")
            self._ensure_column(connection, "projects", "parent_id", "TEXT REFERENCES projects(id)")
            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_projects_team_created
                    ON projects(team_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_projects_team_parent
                    ON projects(team_id, parent_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_meetings_team_created
                    ON meetings(team_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_meetings_team_project_created
                    ON meetings(team_id, project_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_action_items_team_status
                    ON meeting_action_items(team_id, status, updated_at DESC);
                """
            )
            for name, token in team_tokens.items():
                connection.execute(
                    """INSERT INTO teams(name, token_hash, created_at) VALUES(?,?,?)
                       ON CONFLICT(name) DO UPDATE SET token_hash=excluded.token_hash""",
                    (name, _token_hash(token), _utc_now()),
                )
            rows = connection.execute(
                "SELECT id, name FROM teams WHERE name IN ({})".format(
                    ",".join("?" for _ in team_tokens)
                ),
                tuple(team_tokens),
            ).fetchall()
            by_name = {row["name"]: row["id"] for row in rows}
            self._tokens = {token: by_name[name] for name, token in team_tokens.items()}
            self._backfill_action_items(connection)

    @staticmethod
    def _report_payload(raw_json: str) -> dict:
        payload = json.loads(raw_json)
        if not isinstance(payload, dict):
            raise ValueError("report JSON must be an object")
        # 历史报告可能由旧版本生成；只补展示字段，不伪造会议内容。
        payload.setdefault("overview", "该报告生成于旧版本，暂无会议总览。")
        payload.setdefault("unresolved_issues", [])
        return payload

    def _backfill_action_items(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """SELECT r.meeting_id,r.json,r.created_at,m.team_id
               FROM reports r JOIN meetings m ON m.id=r.meeting_id"""
        ).fetchall()
        for row in rows:
            try:
                report = TeamMeetingReport.model_validate(self._report_payload(row["json"]))
            except (ValueError, TypeError):
                continue
            for index, item in enumerate(report.action_items):
                connection.execute(
                    """INSERT OR IGNORE INTO meeting_action_items(
                           meeting_id,team_id,item_index,task,owner,deadline,status,created_at,updated_at
                       ) VALUES(?,?,?,?,?,?,'待确认',?,?)""",
                    (
                        row["meeting_id"], row["team_id"], index, item.task, item.owner,
                        item.deadline, row["created_at"], row["created_at"],
                    ),
                )

    def authenticate(self, candidate: str) -> Optional[int]:
        import hmac

        for token, team_id in self._tokens.items():
            if hmac.compare_digest(candidate, token):
                return team_id
        return None

    def create_project(
        self, project_id: str, team_id: int, name: str, parent_id: Optional[str] = None
    ) -> ProjectListItem:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO projects(id,team_id,name,parent_id,created_at) VALUES(?,?,?,?,?)",
                (project_id, team_id, name, parent_id, _utc_now()),
            )
        created = self.get_project(project_id, team_id)
        if created is None:
            raise RuntimeError("project was not persisted")
        return created

    def get_project(self, project_id: str, team_id: int) -> Optional[ProjectListItem]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """SELECT p.id,p.name,p.parent_id,p.created_at,COUNT(m.id) AS meeting_count
                   FROM projects p LEFT JOIN meetings m ON m.project_id=p.id
                   WHERE p.id=? AND p.team_id=? GROUP BY p.id""",
                (project_id, team_id),
            ).fetchone()
        return ProjectListItem.model_validate(dict(row)) if row else None

    def project_owner_team_id(self, project_id: str) -> Optional[int]:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT team_id FROM projects WHERE id=?", (project_id,)).fetchone()
        return int(row["team_id"]) if row else None

    def list_projects(self, team_id: int) -> list[ProjectListItem]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT p.id,p.name,p.parent_id,p.created_at,COUNT(m.id) AS meeting_count
                   FROM projects p LEFT JOIN meetings m ON m.project_id=p.id
                   WHERE p.team_id=? GROUP BY p.id ORDER BY p.created_at DESC""",
                (team_id,),
            ).fetchall()
        return [ProjectListItem.model_validate(dict(row)) for row in rows]

    def rename_project(self, project_id: str, team_id: int, name: str) -> Optional[ProjectListItem]:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE projects SET name=? WHERE id=? AND team_id=?", (name, project_id, team_id)
            )
        return self.get_project(project_id, team_id) if cursor.rowcount else None

    def delete_project(self, project_id: str, team_id: int, delete_meetings: bool) -> int:
        with self._lock, self._connect() as connection:
            child = connection.execute(
                "SELECT 1 FROM projects WHERE team_id=? AND parent_id=? LIMIT 1",
                (team_id, project_id),
            ).fetchone()
            if child:
                raise ProjectHasChildrenError()
            rows = connection.execute(
                "SELECT id,status,audio_path FROM meetings WHERE team_id=? AND project_id=?",
                (team_id, project_id),
            ).fetchall()
            if delete_meetings and any(
                row["audio_path"] is not None or row["status"] not in {"完成", "失败"}
                for row in rows
            ):
                raise ProjectHasActiveMeetingsError()
            if delete_meetings:
                connection.execute(
                    "DELETE FROM meetings WHERE team_id=? AND project_id=?", (team_id, project_id)
                )
            else:
                connection.execute(
                    "UPDATE meetings SET project_id=NULL WHERE team_id=? AND project_id=?",
                    (team_id, project_id),
                )
            deleted = connection.execute(
                "DELETE FROM projects WHERE id=? AND team_id=?", (project_id, team_id)
            )
            if deleted.rowcount != 1:
                raise PermissionError("project does not belong to team")
        return len(rows)

    def create_meeting(
        self, meeting_id: str, team_id: int, title: str, audio_path: Path,
        project_id: Optional[str] = None,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO meetings(id,team_id,title,project_id,audio_path,status,created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (meeting_id, team_id, title, project_id, str(audio_path), "排队中", _utc_now()),
            )

    def delete_meeting(self, meeting_id: str, team_id: int) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM meetings WHERE id=? AND team_id=?", (meeting_id, team_id))

    def update_status(self, meeting_id: str, team_id: int, status: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE meetings SET status=? WHERE id=? AND team_id=?", (status, meeting_id, team_id)
            )

    def move_meeting(self, meeting_id: str, team_id: int, project_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE meetings SET project_id=? WHERE id=? AND team_id=?",
                (project_id, meeting_id, team_id),
            )
        return cursor.rowcount == 1

    def clear_audio_path(self, meeting_id: str, team_id: int) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE meetings SET audio_path=NULL WHERE id=? AND team_id=?", (meeting_id, team_id)
            )

    def save_transcript(self, meeting_id: str, team_id: int, transcript: Transcript) -> None:
        with self._lock, self._connect() as connection:
            owned = connection.execute(
                "SELECT 1 FROM meetings WHERE id=? AND team_id=?", (meeting_id, team_id)
            ).fetchone()
            if not owned:
                raise PermissionError("meeting does not belong to team")
            connection.execute("DELETE FROM transcripts WHERE meeting_id=?", (meeting_id,))
            connection.executemany(
                "INSERT INTO transcripts(meeting_id,start,end,speaker_label,text) VALUES(?,?,?,?,?)",
                [(meeting_id, segment.start, segment.end, None, segment.text) for segment in transcript.segments],
            )
            connection.execute(
                "UPDATE meetings SET duration_seconds=? WHERE id=? AND team_id=?",
                (transcript.duration_seconds, meeting_id, team_id),
            )

    def save_report(self, meeting_id: str, team_id: int, report: TeamMeetingReport) -> None:
        payload = report.model_dump_json()
        report_action_items = getattr(report, "action_items", [])
        now = _utc_now()
        with self._lock, self._connect() as connection:
            owned = connection.execute(
                "SELECT 1 FROM meetings WHERE id=? AND team_id=?", (meeting_id, team_id)
            ).fetchone()
            if not owned:
                raise PermissionError("meeting does not belong to team")
            connection.execute(
                """INSERT INTO reports(meeting_id,json,created_at) VALUES(?,?,?)
                   ON CONFLICT(meeting_id) DO UPDATE SET json=excluded.json, created_at=excluded.created_at""",
                (meeting_id, payload, now),
            )
            for index, item in enumerate(report_action_items):
                connection.execute(
                    """INSERT INTO meeting_action_items(
                           meeting_id,team_id,item_index,task,owner,deadline,status,created_at,updated_at
                       ) VALUES(?,?,?,?,?,?,'待确认',?,?)
                       ON CONFLICT(meeting_id,item_index) DO UPDATE SET
                           task=excluded.task,owner=excluded.owner,deadline=excluded.deadline,
                           updated_at=excluded.updated_at""",
                    (meeting_id, team_id, index, item.task, item.owner, item.deadline, now, now),
                )
            connection.execute(
                "DELETE FROM meeting_action_items WHERE meeting_id=? AND item_index>=?",
                (meeting_id, len(report_action_items)),
            )

    def owner_team_id(self, meeting_id: str) -> Optional[int]:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT team_id FROM meetings WHERE id=?", (meeting_id,)).fetchone()
            return int(row["team_id"]) if row else None

    def list_meetings(
        self, team_id: int, project_id: Optional[str] = None, unclassified: bool = False,
        include_children: bool = False,
    ) -> list[MeetingListItem]:
        conditions = ["team_id=?"]
        parameters: list[object] = [team_id]
        if project_id is not None:
            if include_children:
                conditions.append(
                    "(project_id=? OR project_id IN "
                    "(SELECT id FROM projects WHERE team_id=? AND parent_id=?))"
                )
                parameters.extend((project_id, team_id, project_id))
            else:
                conditions.append("project_id=?")
                parameters.append(project_id)
        elif unclassified:
            conditions.append("project_id IS NULL")
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT id,title,project_id,duration_seconds,status,created_at FROM meetings
                   WHERE {} ORDER BY created_at DESC""".format(" AND ".join(conditions)),
                tuple(parameters),
            ).fetchall()
        return [MeetingListItem.model_validate(dict(row)) for row in rows]

    def get_history(self, meeting_id: str, team_id: int) -> Optional[MeetingHistory]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """SELECT m.id,m.title,m.project_id,m.duration_seconds,m.status,m.created_at,r.json
                   FROM meetings m JOIN reports r ON r.meeting_id=m.id
                   WHERE m.id=? AND m.team_id=?""",
                (meeting_id, team_id),
            ).fetchone()
            transcript_rows = connection.execute(
                """SELECT t.start,t.end,t.text FROM transcripts t
                   JOIN meetings m ON m.id=t.meeting_id
                   WHERE t.meeting_id=? AND m.team_id=? ORDER BY t.id""",
                (meeting_id, team_id),
            ).fetchall() if row else []
        if not row:
            return None
        report_payload = self._report_payload(row["json"])
        return MeetingHistory(
            id=row["id"], title=row["title"], project_id=row["project_id"],
            duration_seconds=row["duration_seconds"],
            status=row["status"], created_at=row["created_at"],
            report=TeamMeetingReport.model_validate(report_payload),
            transcript=[TranscriptSegment.model_validate(dict(item)) for item in transcript_rows],
        )

    def get_project_memory(
        self, project_id: str, team_id: int, meeting_limit: int = 3
    ) -> Optional[ProjectMemory]:
        project = self.get_project(project_id, team_id)
        if project is None:
            return None
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT m.id,m.title,m.project_id,m.duration_seconds,m.status,m.created_at,r.json
                   FROM meetings m JOIN reports r ON r.meeting_id=m.id
                   WHERE m.team_id=? AND m.status='完成'
                     AND (m.project_id=? OR m.project_id IN (
                         SELECT id FROM projects WHERE team_id=? AND parent_id=?
                     ))
                   ORDER BY m.created_at DESC LIMIT ?""",
                (team_id, project_id, team_id, project_id, meeting_limit),
            ).fetchall()
            meeting_ids = [row["id"] for row in rows]
            if meeting_ids:
                action_rows = connection.execute(
                    """SELECT id,meeting_id,item_index,status FROM meeting_action_items
                       WHERE team_id=? AND meeting_id IN ({})""".format(
                        ",".join("?" for _ in meeting_ids)
                    ),
                    (team_id, *meeting_ids),
                ).fetchall()
            else:
                action_rows = []

        actions_by_position = {
            (row["meeting_id"], row["item_index"]): row for row in action_rows
        }
        recent_meetings: list[MeetingListItem] = []
        decisions: list[ProjectMemoryDecision] = []
        action_items: list[ProjectMemoryAction] = []
        unresolved_issues: list[ProjectMemoryIssue] = []
        for row in rows:
            recent_meetings.append(MeetingListItem.model_validate({
                key: row[key]
                for key in ("id", "title", "project_id", "duration_seconds", "status", "created_at")
            }))
            source = MeetingSource(id=row["id"], title=row["title"], created_at=row["created_at"])
            report = TeamMeetingReport.model_validate(self._report_payload(row["json"]))
            decisions.extend(
                ProjectMemoryDecision(**item.model_dump(), source=source)
                for item in report.decisions
            )
            unresolved_issues.extend(
                ProjectMemoryIssue(**item.model_dump(), source=source)
                for item in report.unresolved_issues
            )
            for index, item in enumerate(report.action_items):
                action_row = actions_by_position.get((row["id"], index))
                if action_row is None:
                    continue
                action_items.append(ProjectMemoryAction(
                    id=action_row["id"], task=item.task, owner=item.owner,
                    deadline=item.deadline, status=action_row["status"], source=source,
                ))
        return ProjectMemory(
            project_id=project.id,
            project_name=project.name,
            recent_meetings=recent_meetings,
            decisions=decisions,
            action_items=action_items,
            unresolved_issues=unresolved_issues,
        )

    def action_owner_team_id(self, action_id: int) -> Optional[int]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT team_id FROM meeting_action_items WHERE id=?", (action_id,)
            ).fetchone()
        return int(row["team_id"]) if row else None

    def update_action_status(
        self, action_id: int, team_id: int, action_status: ActionStatus
    ) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """UPDATE meeting_action_items SET status=?,updated_at=?
                   WHERE id=? AND team_id=?""",
                (action_status, _utc_now(), action_id, team_id),
            )
        return cursor.rowcount == 1

    def transcript_rows(self, meeting_id: str) -> Iterable[sqlite3.Row]:
        with self._lock, self._connect() as connection:
            return list(connection.execute(
                "SELECT start,end,speaker_label,text FROM transcripts WHERE meeting_id=? ORDER BY id",
                (meeting_id,),
            ))

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.models import (
    ActionStatus, DEFAULT_MEETING_TITLE, MeetingHistory, MeetingListItem, MeetingSource, MeetingSpeaker,
    MemberIdentity, ProjectListItem, SpeakerClip,
    ProjectMemory, ProjectMemoryAction, ProjectMemoryDecision, ProjectMemoryIssue,
    TeamMeetingReport, Transcript, TranscriptSegment,
)


logger = logging.getLogger(__name__)


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
        # FTS5 是否可用（运行环境差异）；不可用时工具层回退 LIKE。
        self._fts5_available = False

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

    @staticmethod
    def _setup_fts5(connection: sqlite3.Connection) -> bool:
        """建立转写全文索引（R-P1-3 search_transcript 的底座）。

        用 external content FTS5：索引只存词项，转写原文仍只存 transcripts 表，
        不产生第二份敏感文本；用触发器保持同步。
        tokenizer 选 trigram：unicode61 会把整段中文当单个 token，导致“搜索中间的词”
        搜不到；trigram 支持中文子串匹配（查询词需 ≥ 3 字，更短的由工具层回退 LIKE）。
        环境不支持 FTS5 时降级返回 False（工具层回退 LIKE），不影响服务启动。
        """
        try:
            connection.executescript(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS transcript_fts USING fts5(
                    text, content='transcripts', content_rowid='id', tokenize='trigram'
                );
                CREATE TRIGGER IF NOT EXISTS transcripts_fts_ai AFTER INSERT ON transcripts BEGIN
                    INSERT INTO transcript_fts(rowid, text) VALUES (new.id, new.text);
                END;
                CREATE TRIGGER IF NOT EXISTS transcripts_fts_ad AFTER DELETE ON transcripts BEGIN
                    INSERT INTO transcript_fts(transcript_fts, rowid, text)
                        VALUES ('delete', old.id, old.text);
                END;
                CREATE TRIGGER IF NOT EXISTS transcripts_fts_au AFTER UPDATE ON transcripts BEGIN
                    INSERT INTO transcript_fts(transcript_fts, rowid, text)
                        VALUES ('delete', old.id, old.text);
                    INSERT INTO transcript_fts(rowid, text) VALUES (new.id, new.text);
                END;
                """
            )
        except sqlite3.OperationalError as exc:
            logger.warning("fts5_unavailable error_type=%s", type(exc).__name__)
            return False
        # 旧库补索引：transcripts 已有内容但索引为空时重建一次（幂等）。
        try:
            existing = connection.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
            if existing:
                indexed = connection.execute(
                    "SELECT COUNT(*) FROM transcript_fts_docsize"
                ).fetchone()[0]
                if indexed != existing:
                    connection.execute(
                        "INSERT INTO transcript_fts(transcript_fts) VALUES('rebuild')"
                    )
        except sqlite3.OperationalError as exc:
            logger.warning("fts5_rebuild_skipped error_type=%s", type(exc).__name__)
        return True

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
                    role TEXT NOT NULL DEFAULT '',
                    is_key_decision_maker INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS speaker_profiles(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id INTEGER NOT NULL REFERENCES teams(id),
                    member_id INTEGER NOT NULL UNIQUE REFERENCES members(id) ON DELETE CASCADE,
                    embedding_json TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    consented_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
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
                CREATE TABLE IF NOT EXISTS meeting_speakers(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
                    team_id INTEGER NOT NULL REFERENCES teams(id),
                    local_label TEXT NOT NULL,
                    member_id INTEGER REFERENCES members(id) ON DELETE SET NULL,
                    confidence REAL,
                    status TEXT NOT NULL CHECK(status IN ('待确认','已识别','已确认','仅本场')),
                    embedding_json TEXT,
                    speech_seconds REAL NOT NULL DEFAULT 0,
                    excerpts_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(meeting_id, local_label)
                );
                CREATE TABLE IF NOT EXISTS speaker_clips(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
                    team_id INTEGER NOT NULL REFERENCES teams(id),
                    local_label TEXT NOT NULL,
                    start REAL NOT NULL,
                    end REAL NOT NULL,
                    text TEXT NOT NULL,
                    mime_type TEXT NOT NULL DEFAULT 'audio/wav',
                    audio BLOB NOT NULL,
                    created_at TEXT NOT NULL
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
                CREATE TABLE IF NOT EXISTS llm_usage(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id INTEGER NOT NULL,
                    meeting_id TEXT NULL,
                    project_id TEXT NULL,
                    stage TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_tokens INTEGER NULL,
                    completion_tokens INTEGER NULL,
                    total_tokens INTEGER NULL,
                    duration_ms INTEGER NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agent_tasks(
                    task_id TEXT PRIMARY KEY,
                    team_id INTEGER NOT NULL,
                    meeting_id TEXT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NULL,
                    progress_json TEXT NULL,
                    message TEXT NULL,
                    request_id TEXT NULL,
                    long_meeting INTEGER NOT NULL DEFAULT 0,
                    audio_path TEXT NULL,
                    error TEXT NULL,
                    error_code INTEGER NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT NULL
                );
                CREATE TABLE IF NOT EXISTS agent_task_deps(
                    task_id TEXT NOT NULL,
                    depends_on_task_id TEXT NOT NULL,
                    UNIQUE(task_id, depends_on_task_id)
                );
                CREATE TABLE IF NOT EXISTS agent_audit(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id INTEGER NOT NULL,
                    meeting_id TEXT NULL,
                    session_id TEXT NOT NULL,
                    step INTEGER NULL,
                    tool_name TEXT NULL,
                    args_digest TEXT NULL,
                    decision TEXT NOT NULL,
                    result_code TEXT NULL,
                    duration_ms INTEGER NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            self._ensure_column(connection, "meetings", "project_id", "TEXT REFERENCES projects(id)")
            self._ensure_column(connection, "meetings", "speaker_consent_at", "TEXT")
            self._ensure_column(connection, "projects", "parent_id", "TEXT REFERENCES projects(id)")
            self._ensure_column(connection, "members", "role", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(
                connection, "members", "is_key_decision_maker", "INTEGER NOT NULL DEFAULT 0"
            )
            self._ensure_column(
                connection, "meeting_speakers", "remember_requested", "INTEGER NOT NULL DEFAULT 0"
            )
            self._ensure_column(connection, "agent_tasks", "message", "TEXT")
            self._ensure_column(connection, "agent_tasks", "request_id", "TEXT")
            self._ensure_column(connection, "agent_tasks", "long_meeting", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(connection, "agent_tasks", "audio_path", "TEXT")
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
                CREATE INDEX IF NOT EXISTS idx_members_team_name ON members(team_id, name);
                CREATE INDEX IF NOT EXISTS idx_speaker_profiles_team ON speaker_profiles(team_id);
                CREATE INDEX IF NOT EXISTS idx_meeting_speakers_team_meeting
                    ON meeting_speakers(team_id, meeting_id);
                CREATE INDEX IF NOT EXISTS idx_speaker_clips_team_meeting
                    ON speaker_clips(team_id, meeting_id, local_label);
                CREATE INDEX IF NOT EXISTS idx_llm_usage_team_meeting
                    ON llm_usage(team_id, meeting_id);
                CREATE INDEX IF NOT EXISTS idx_llm_usage_team_project
                    ON llm_usage(team_id, project_id);
                CREATE INDEX IF NOT EXISTS idx_llm_usage_team_created
                    ON llm_usage(team_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_agent_tasks_team_status
                    ON agent_tasks(team_id, status);
                CREATE INDEX IF NOT EXISTS idx_agent_audit_team_session
                    ON agent_audit(team_id, session_id, id);
                """
            )
            # 转写全文索引（R-P1-3 的底座）；环境不支持 FTS5 时降级，不影响启动。
            self._fts5_available = self._setup_fts5(connection)
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

        # hmac.compare_digest 不接受含非 ASCII 字符的 str，而 Starlette 把请求头按
        # latin-1 解码，任意字节序列都可能出现（例如客户端发来畸形口令）。
        # 统一转成 UTF-8 字节再比较：既保留时序安全比较，也避免把 403 变成 500。
        # 详见 P0 阶段文档第十二节的风险记录。
        try:
            candidate_bytes = candidate.encode("utf-8")
        except UnicodeEncodeError:  # pragma: no cover - 防御性兜底
            return None
        for token, team_id in self._tokens.items():
            if hmac.compare_digest(candidate_bytes, token.encode("utf-8")):
                return team_id
        return None

    # ------------------------------------------------------------------
    # LLM 成本归因（PRD R-P0-2 / §11.1）
    # ------------------------------------------------------------------

    def meeting_context(self, meeting_id: str, team_id: int) -> Optional[Dict[str, Any]]:
        """返回本团队某场会议的成本归因上下文；不属于本团队则返回 None。"""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT id, project_id FROM meetings WHERE id = ? AND team_id = ?",
                (meeting_id, team_id),
            ).fetchone()
        if row is None:
            return None
        return {"meeting_id": row["id"], "project_id": row["project_id"]}

    def record_llm_usage(
        self,
        team_id: int,
        stage: str,
        model: str,
        meeting_id: Optional[str] = None,
        project_id: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        duration_ms: Optional[int] = None,
        created_at: Optional[str] = None,
    ) -> None:
        """写入一次 LLM 调用的用量。只记数值与标识，永不记 prompt / 回复原文。"""
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO llm_usage(
                       team_id,meeting_id,project_id,stage,model,
                       prompt_tokens,completion_tokens,total_tokens,duration_ms,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    team_id, meeting_id, project_id, stage, model,
                    prompt_tokens, completion_tokens, total_tokens, duration_ms,
                    created_at or _utc_now(),
                ),
            )

    def usage_summary(
        self,
        team_id: int,
        meeting_id: Optional[str] = None,
        project_id: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """按 stage / 会议聚合用量，严格限定在 team_id 内。表为空时返回零值与空列表。"""
        conditions = ["team_id = ?"]
        params: List[Any] = [team_id]
        if meeting_id is not None:
            conditions.append("meeting_id = ?")
            params.append(meeting_id)
        if project_id is not None:
            conditions.append("project_id = ?")
            params.append(project_id)
        if date_from is not None:
            conditions.append("created_at >= ?")
            params.append(date_from)
        if date_to is not None:
            conditions.append("created_at <= ?")
            params.append(date_to)
        where = " AND ".join(conditions)

        aggregates = """COUNT(*) AS calls,
                        COALESCE(SUM(prompt_tokens),0) AS prompt_tokens,
                        COALESCE(SUM(completion_tokens),0) AS completion_tokens,
                        COALESCE(SUM(total_tokens),0) AS total_tokens,
                        COALESCE(SUM(duration_ms),0) AS duration_ms"""
        with self._lock, self._connect() as connection:
            totals = connection.execute(
                "SELECT {} FROM llm_usage WHERE {}".format(aggregates, where), params
            ).fetchone()
            by_stage = connection.execute(
                """SELECT stage, model, {}, MIN(created_at) AS first_at, MAX(created_at) AS last_at
                   FROM llm_usage WHERE {}
                   GROUP BY stage, model ORDER BY stage ASC, model ASC""".format(aggregates, where),
                params,
            ).fetchall()
            by_meeting = connection.execute(
                """SELECT meeting_id, project_id, {}, MIN(created_at) AS first_at, MAX(created_at) AS last_at
                   FROM llm_usage WHERE {} AND meeting_id IS NOT NULL
                   GROUP BY meeting_id, project_id ORDER BY MAX(created_at) DESC""".format(
                    aggregates, where
                ),
                params,
            ).fetchall()

        def _row(row: sqlite3.Row) -> Dict[str, Any]:
            return {key: row[key] for key in row.keys()}

        return {
            "totals": _row(totals),
            "by_stage": [_row(row) for row in by_stage],
            "by_meeting": [_row(row) for row in by_meeting],
        }

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
                meeting_ids = [row["id"] for row in rows]
                if meeting_ids:
                    placeholders = ",".join("?" for _ in meeting_ids)
                    for table in ("llm_usage", "agent_audit", "agent_tasks"):
                        connection.execute(
                            "DELETE FROM {} WHERE team_id=? AND meeting_id IN ({})".format(
                                table, placeholders
                            ),
                            (team_id, *meeting_ids),
                        )
                connection.execute(
                    "DELETE FROM meetings WHERE team_id=? AND project_id=?", (team_id, project_id)
                )
            else:
                # 会议移入未分类；成本行保留但解除对已删项目的指向（PRD §10.3）。
                connection.execute(
                    "UPDATE llm_usage SET project_id=NULL WHERE team_id=? AND project_id=?",
                    (team_id, project_id),
                )
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
            # PRD §10.3：删除会议时，Agent 新增的关联数据一并级联清理。
            connection.execute(
                "DELETE FROM llm_usage WHERE meeting_id=? AND team_id=?", (meeting_id, team_id)
            )
            connection.execute(
                "DELETE FROM agent_audit WHERE meeting_id=? AND team_id=?", (meeting_id, team_id)
            )
            connection.execute(
                "DELETE FROM agent_tasks WHERE meeting_id=? AND team_id=?", (meeting_id, team_id)
            )
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

    def update_meeting_title(self, meeting_id: str, team_id: int, title: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE meetings SET title=? WHERE id=? AND team_id=?",
                (title, meeting_id, team_id),
            )
        return cursor.rowcount == 1

    def apply_suggested_title(self, meeting_id: str, team_id: int, suggested_title: str) -> bool:
        """D-027：标题仍是默认值（或空）时采用 AI 建议标题；用户改过的绝不覆盖。"""
        title = (suggested_title or "").strip()[:100]
        if not title or title == DEFAULT_MEETING_TITLE:
            return False
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """UPDATE meetings SET title=?
                   WHERE id=? AND team_id=?
                     AND (title IS NULL OR title='' OR title=?)""",
                (title, meeting_id, team_id, DEFAULT_MEETING_TITLE),
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
                [
                    (meeting_id, segment.start, segment.end, segment.speaker_label, segment.text)
                    for segment in transcript.segments
                ],
            )
            connection.execute(
                "UPDATE meetings SET duration_seconds=? WHERE id=? AND team_id=?",
                (transcript.duration_seconds, meeting_id, team_id),
            )

    def voice_profiles(self, team_id: int, model_version: Optional[str] = None) -> list[dict]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT m.id AS member_id,m.name,p.embedding_json
                   FROM speaker_profiles p JOIN members m ON m.id=p.member_id
                   WHERE p.team_id=? AND m.team_id=?
                     AND (? IS NULL OR p.model_version=?) ORDER BY m.id""",
                (team_id, team_id, model_version, model_version),
            ).fetchall()
        return [
            {
                "member_id": int(row["member_id"]),
                "name": row["name"],
                "embedding": json.loads(row["embedding_json"]),
            }
            for row in rows
        ]

    def save_meeting_speakers(self, meeting_id: str, team_id: int, observations) -> None:
        now = _utc_now()
        with self._lock, self._connect() as connection:
            owned = connection.execute(
                "SELECT 1 FROM meetings WHERE id=? AND team_id=?", (meeting_id, team_id)
            ).fetchone()
            if not owned:
                raise PermissionError("meeting does not belong to team")
            connection.execute("DELETE FROM speaker_clips WHERE meeting_id=?", (meeting_id,))
            connection.execute("DELETE FROM meeting_speakers WHERE meeting_id=?", (meeting_id,))
            connection.executemany(
                """INSERT INTO meeting_speakers(
                       meeting_id,team_id,local_label,member_id,confidence,status,
                       embedding_json,speech_seconds,excerpts_json,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        meeting_id, team_id, item.local_label, item.matched_member_id,
                        item.confidence, "已识别" if item.matched_member_id else "待确认",
                        json.dumps(item.embedding) if item.embedding is not None else None,
                        item.speech_seconds, json.dumps(item.excerpts, ensure_ascii=False), now, now,
                    )
                    for item in observations
                ],
            )
            for item in observations:
                connection.executemany(
                    """INSERT INTO speaker_clips(
                           meeting_id,team_id,local_label,start,end,text,mime_type,audio,created_at
                       ) VALUES(?,?,?,?,?,?,?, ?,?)""",
                    [
                        (
                            meeting_id, team_id, item.local_label, clip.start, clip.end,
                            clip.text, "audio/wav", clip.audio, now,
                        )
                        for clip in item.clips
                    ],
                )

    def list_members(self, team_id: int) -> list[MemberIdentity]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT m.id,m.name,m.role,m.is_key_decision_maker,m.created_at,
                          CASE WHEN p.id IS NULL THEN 0 ELSE 1 END AS has_voiceprint
                   FROM members m LEFT JOIN speaker_profiles p ON p.member_id=m.id
                   WHERE m.team_id=? ORDER BY m.name,m.id""",
                (team_id,),
            ).fetchall()
        return [MemberIdentity(
            id=row["id"], name=row["name"], role=row["role"],
            is_key_decision_maker=bool(row["is_key_decision_maker"]),
            has_voiceprint=bool(row["has_voiceprint"]), created_at=row["created_at"],
        ) for row in rows]

    def member_owner_team_id(self, member_id: int) -> Optional[int]:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT team_id FROM members WHERE id=?", (member_id,)).fetchone()
        return int(row["team_id"]) if row else None

    def update_member(
        self, member_id: int, team_id: int, name: str, role: str,
        is_key_decision_maker: bool,
    ) -> Optional[MemberIdentity]:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """UPDATE members SET name=?,role=?,is_key_decision_maker=?
                   WHERE id=? AND team_id=?""",
                (name, role, int(is_key_decision_maker), member_id, team_id),
            )
            if cursor.rowcount != 1:
                return None
        return next(item for item in self.list_members(team_id) if item.id == member_id)

    def delete_voiceprint(self, member_id: int, team_id: int) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM speaker_profiles WHERE member_id=? AND team_id=?",
                (member_id, team_id),
            )
        return cursor.rowcount == 1

    def merge_members(self, source_id: int, target_id: int, team_id: int) -> Optional[int]:
        if source_id == target_id:
            raise ValueError("source and target must differ")
        now = _utc_now()
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT id,name FROM members WHERE team_id=? AND id IN (?,?)",
                (team_id, source_id, target_id),
            ).fetchall()
            members = {int(row["id"]): row["name"] for row in rows}
            if source_id not in members or target_id not in members:
                return None
            source_profile = connection.execute(
                "SELECT id FROM speaker_profiles WHERE member_id=?", (source_id,)
            ).fetchone()
            target_profile = connection.execute(
                "SELECT id FROM speaker_profiles WHERE member_id=?", (target_id,)
            ).fetchone()
            if source_profile and not target_profile:
                connection.execute(
                    "UPDATE speaker_profiles SET member_id=?,team_id=?,updated_at=? WHERE member_id=?",
                    (target_id, team_id, now, source_id),
                )
            elif source_profile:
                connection.execute("DELETE FROM speaker_profiles WHERE member_id=?", (source_id,))
            meeting_rows = connection.execute(
                "SELECT DISTINCT meeting_id FROM meeting_speakers WHERE team_id=? AND member_id=?",
                (team_id, source_id),
            ).fetchall()
            connection.execute(
                "UPDATE meeting_speakers SET member_id=?,updated_at=? WHERE team_id=? AND member_id=?",
                (target_id, now, team_id, source_id),
            )
            report_rows = connection.execute(
                """SELECT r.meeting_id,r.json FROM reports r
                   JOIN meetings m ON m.id=r.meeting_id WHERE m.team_id=?""",
                (team_id,),
            ).fetchall()
            for row in report_rows:
                payload = json.loads(row["json"])
                changed = False
                for item in payload.get("decisions", []):
                    if item.get("decision_maker") == members[source_id]:
                        item["decision_maker"] = members[target_id]
                        changed = True
                for item in payload.get("action_items", []):
                    if item.get("owner") == members[source_id]:
                        item["owner"] = members[target_id]
                        changed = True
                if changed:
                    connection.execute(
                        "UPDATE reports SET json=?,created_at=? WHERE meeting_id=?",
                        (json.dumps(payload, ensure_ascii=False), now, row["meeting_id"]),
                    )
            connection.execute(
                "UPDATE meeting_action_items SET owner=?,updated_at=? WHERE team_id=? AND owner=?",
                (members[target_id], now, team_id, members[source_id]),
            )
            connection.execute("DELETE FROM members WHERE id=? AND team_id=?", (source_id, team_id))
        return len(meeting_rows)

    def meeting_speaker_owner_team_id(self, meeting_id: str, local_label: str) -> Optional[int]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT team_id FROM meeting_speakers WHERE meeting_id=? AND local_label=?",
                (meeting_id, local_label),
            ).fetchone()
        return int(row["team_id"]) if row else None

    def confirm_meeting_speaker(
        self, meeting_id: str, team_id: int, local_label: str, name: str, role: str,
        is_key_decision_maker: bool, remember_voice: bool, model_version: str,
    ) -> Optional[tuple[int, bool, bool]]:
        """Map a local label; voiceprint persistence is deferred to meeting finalization."""
        now = _utc_now()
        with self._lock, self._connect() as connection:
            speaker = connection.execute(
                """SELECT embedding_json FROM meeting_speakers
                   WHERE meeting_id=? AND team_id=? AND local_label=?""",
                (meeting_id, team_id, local_label),
            ).fetchone()
            if speaker is None:
                return None
            if remember_voice and not speaker["embedding_json"]:
                raise ValueError("voice sample unavailable")
            member = connection.execute(
                "SELECT id FROM members WHERE team_id=? AND name=? ORDER BY id LIMIT 1",
                (team_id, name),
            ).fetchone()
            if member:
                member_id = int(member["id"])
                connection.execute(
                    """UPDATE members SET role=?,is_key_decision_maker=? WHERE id=? AND team_id=?""",
                    (role, int(is_key_decision_maker), member_id, team_id),
                )
            else:
                cursor = connection.execute(
                    """INSERT INTO members(team_id,name,role,is_key_decision_maker,created_at)
                       VALUES(?,?,?,?,?)""",
                    (team_id, name, role, int(is_key_decision_maker), now),
                )
                member_id = int(cursor.lastrowid)
            connection.execute(
                """UPDATE meeting_speakers
                   SET member_id=?,status=?,remember_requested=?,updated_at=?
                   WHERE meeting_id=? AND team_id=? AND local_label=?""",
                (
                    member_id, "已确认" if remember_voice else "仅本场",
                    int(remember_voice), now,
                    meeting_id, team_id, local_label,
                ),
            )
            report_row = connection.execute(
                "SELECT json FROM reports WHERE meeting_id=?", (meeting_id,)
            ).fetchone()
            affected = False
            if report_row:
                payload = json.loads(report_row["json"])
                for item in payload.get("decisions", []):
                    if item.get("decision_maker") == local_label:
                        item["decision_maker"] = name
                        affected = True
                for item in payload.get("action_items", []):
                    if item.get("owner") == local_label:
                        item["owner"] = name
                        affected = True
                if affected:
                    connection.execute(
                        "UPDATE reports SET json=?,created_at=? WHERE meeting_id=?",
                        (json.dumps(payload, ensure_ascii=False), now, meeting_id),
                    )
                    connection.execute(
                        """UPDATE meeting_action_items SET owner=?,updated_at=?
                           WHERE meeting_id=? AND owner=?""",
                        (name, now, meeting_id, local_label),
                    )
        return member_id, False, affected

    def finalize_meeting_voiceprints(
        self, meeting_id: str, team_id: int, model_version: str,
        consent_confirmed: bool = False,
    ) -> Optional[list[str]]:
        now = _utc_now()
        with self._lock, self._connect() as connection:
            owned = connection.execute(
                "SELECT 1 FROM meetings WHERE id=? AND team_id=?", (meeting_id, team_id)
            ).fetchone()
            if not owned:
                return None
            rows = connection.execute(
                """SELECT s.member_id,s.embedding_json,m.name
                   FROM meeting_speakers s JOIN members m ON m.id=s.member_id
                   WHERE s.meeting_id=? AND s.team_id=? AND s.remember_requested=1""",
                (meeting_id, team_id),
            ).fetchall()
            for row in rows:
                if not row["embedding_json"]:
                    raise ValueError("voice sample unavailable")
                connection.execute(
                    """INSERT INTO speaker_profiles(
                           team_id,member_id,embedding_json,model_version,consented_at,updated_at
                       ) VALUES(?,?,?,?,?,?)
                       ON CONFLICT(member_id) DO UPDATE SET
                           embedding_json=excluded.embedding_json,
                           model_version=excluded.model_version,
                           consented_at=excluded.consented_at,updated_at=excluded.updated_at""",
                    (
                        team_id, row["member_id"], row["embedding_json"],
                        model_version, now, now,
                    ),
                )
            connection.execute(
                """UPDATE meeting_speakers SET remember_requested=0,status='已确认',updated_at=?
                   WHERE meeting_id=? AND team_id=? AND remember_requested=1""",
                (now, meeting_id, team_id),
            )
            if consent_confirmed:
                connection.execute(
                    "UPDATE meetings SET speaker_consent_at=? WHERE id=? AND team_id=?",
                    (now, meeting_id, team_id),
                )
        return [row["name"] for row in rows]

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
                """SELECT m.id,m.title,m.project_id,m.duration_seconds,m.status,m.created_at,
                          m.speaker_consent_at,r.json
                   FROM meetings m JOIN reports r ON r.meeting_id=m.id
                   WHERE m.id=? AND m.team_id=?""",
                (meeting_id, team_id),
            ).fetchone()
            transcript_rows = connection.execute(
                """SELECT t.start,t.end,t.speaker_label,t.text FROM transcripts t
                   JOIN meetings m ON m.id=t.meeting_id
                   WHERE t.meeting_id=? AND m.team_id=? ORDER BY t.id""",
                (meeting_id, team_id),
            ).fetchall() if row else []
            speaker_rows = connection.execute(
                """SELECT s.local_label,s.member_id,s.confidence,s.status,s.speech_seconds,
                          s.excerpts_json,s.embedding_json,s.remember_requested,m.name
                   FROM meeting_speakers s LEFT JOIN members m ON m.id=s.member_id
                   WHERE s.meeting_id=? AND s.team_id=? ORDER BY s.id""",
                (meeting_id, team_id),
            ).fetchall() if row else []
            clip_rows = connection.execute(
                """SELECT c.id,c.local_label,c.start,c.end,c.text
                   FROM speaker_clips c JOIN meetings m ON m.id=c.meeting_id
                   WHERE c.meeting_id=? AND c.team_id=? AND m.team_id=? ORDER BY c.id""",
                (meeting_id, team_id, team_id),
            ).fetchall() if row else []
        if not row:
            return None
        report_payload = self._report_payload(row["json"])
        clips_by_label: dict[str, list[SpeakerClip]] = {}
        for clip in clip_rows:
            clips_by_label.setdefault(clip["local_label"], []).append(SpeakerClip(
                id=clip["id"], start=clip["start"], end=clip["end"], text=clip["text"],
            ))
        return MeetingHistory(
            id=row["id"], title=row["title"], project_id=row["project_id"],
            duration_seconds=row["duration_seconds"],
            status=row["status"], created_at=row["created_at"],
            report=TeamMeetingReport.model_validate(report_payload),
            transcript=[TranscriptSegment.model_validate(dict(item)) for item in transcript_rows],
            speaker_consent_confirmed=row["speaker_consent_at"] is not None,
            speakers=[MeetingSpeaker(
                local_label=item["local_label"],
                display_name=item["name"] or item["local_label"],
                member_id=item["member_id"], confidence=item["confidence"],
                status=item["status"], speech_seconds=item["speech_seconds"],
                excerpts=json.loads(item["excerpts_json"]),
                clips=clips_by_label.get(item["local_label"], []),
                has_voice_sample=item["embedding_json"] is not None,
                remember_requested=bool(item["remember_requested"]),
            ) for item in speaker_rows],
        )

    def speaker_clip(self, clip_id: int, team_id: int) -> Optional[tuple[str, bytes]]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """SELECT c.mime_type,c.audio FROM speaker_clips c
                   JOIN meetings m ON m.id=c.meeting_id
                   WHERE c.id=? AND c.team_id=? AND m.team_id=?""",
                (clip_id, team_id, team_id),
            ).fetchone()
        return (row["mime_type"], bytes(row["audio"])) if row else None

    def speaker_clip_owner_team_id(self, clip_id: int) -> Optional[int]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT team_id FROM speaker_clips WHERE id=?", (clip_id,)
            ).fetchone()
        return int(row["team_id"]) if row else None

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

    # ------------------------------------------------------------------
    # Agent 任务与审计（PRD R-P1-8 / R-P1-5 的存储底座）
    # 本阶段只建表与读写；任务队列的替换与重启恢复在阶段 7（P1-E）接线。
    # ------------------------------------------------------------------

    def fts5_available(self) -> bool:
        """转写全文索引是否可用（工具层据此决定用 MATCH 还是 LIKE）。"""
        return self._fts5_available

    def upsert_agent_task(
        self, task_id: str, team_id: int, status: str, *,
        meeting_id: Optional[str] = None, stage: Optional[str] = None,
        progress_json: Optional[str] = None, message: Optional[str] = None,
        error: Optional[str] = None, error_code: Optional[int] = None,
        finished_at: Optional[str] = None, request_id: Optional[str] = None,
        long_meeting: Optional[bool] = None, audio_path: Optional[str] = None,
    ) -> None:
        """写入或更新一条 Agent 任务行（R-P1-8）。

        progress_json / message / request_id / long_meeting / audio_path 用 COALESCE 合并，
        避免「计划写入」与「任务状态写入」互相覆盖。
        """
        now = _utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO agent_tasks(
                       task_id,team_id,meeting_id,status,stage,progress_json,message,
                       request_id,long_meeting,audio_path,error,error_code,
                       created_at,updated_at,finished_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(task_id) DO UPDATE SET
                       status=excluded.status, stage=excluded.stage,
                       progress_json=COALESCE(excluded.progress_json, agent_tasks.progress_json),
                       message=COALESCE(excluded.message, agent_tasks.message),
                       request_id=COALESCE(excluded.request_id, agent_tasks.request_id),
                       long_meeting=COALESCE(excluded.long_meeting, agent_tasks.long_meeting),
                       audio_path=COALESCE(excluded.audio_path, agent_tasks.audio_path),
                       error=excluded.error, error_code=excluded.error_code,
                       updated_at=excluded.updated_at, finished_at=excluded.finished_at""",
                (
                    task_id, team_id, meeting_id, status, stage, progress_json, message,
                    request_id, int(bool(long_meeting)), audio_path, error, error_code,
                    now, now, finished_at,
                ),
            )

    def get_agent_task(self, task_id: str, team_id: int) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_tasks WHERE task_id=? AND team_id=?", (task_id, team_id)
            ).fetchone()
        return dict(row) if row else None

    def list_active_agent_tasks(self) -> List[Dict[str, Any]]:
        """列出未终态任务，供进程重启后恢复（R-P1-8）。

        不按 team 过滤，因为恢复必须覆盖全部团队；但只返回标识与进度字段，
        其中 progress_json 由调用方保证不含转写文本（R-P1-9 计划只存步骤）。
        """
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM agent_tasks WHERE status NOT IN ('完成','失败') "
                "ORDER BY created_at"
            ).fetchall()
        return [dict(row) for row in rows]

    def record_agent_audit(
        self, *, team_id: int, session_id: str, decision: str,
        meeting_id: Optional[str] = None, step: Optional[int] = None,
        tool_name: Optional[str] = None, args_digest: Optional[str] = None,
        result_code: Optional[str] = None, duration_ms: Optional[int] = None,
        created_at: Optional[str] = None,
    ) -> None:
        """写入一次工具调用的审计行（R-P1-5）。

        只记参数摘要（调用方传 digest），**永不记转写文本、引文原文、口令、密钥**。
        meeting_id 用于执行 PRD §10.3 的删除级联。
        """
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO agent_audit(
                       team_id,meeting_id,session_id,step,tool_name,args_digest,
                       decision,result_code,duration_ms,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    team_id, meeting_id, session_id, step, tool_name, args_digest,
                    decision, result_code, duration_ms, created_at or _utc_now(),
                ),
            )

    def list_agent_audit(
        self, team_id: int, session_id: Optional[str] = None, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """按团队（可选按会话）读取审计行，用于回放一次会话的每一步调用。"""
        conditions = ["team_id=?"]
        params: List[Any] = [team_id]
        if session_id is not None:
            conditions.append("session_id=?")
            params.append(session_id)
        params.append(max(1, min(int(limit), 1000)))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM agent_audit WHERE {} ORDER BY id LIMIT ?".format(
                    " AND ".join(conditions)
                ),
                tuple(params),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_agent_audit_for_meeting(
        self, meeting_id: str, team_id: int, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """按会议读取 Agent 步骤轨迹（只读展示用；已按 team_id 隔离）。"""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT session_id, step, tool_name, decision, result_code,
                          duration_ms, created_at
                   FROM agent_audit WHERE meeting_id=? AND team_id=?
                   ORDER BY id LIMIT ?""",
                (meeting_id, team_id, max(1, min(int(limit), 1000))),
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # 只读工具用的查询（PRD §9.1）；全部按 team_id 先做归属校验。
    # ------------------------------------------------------------------

    def get_report(self, meeting_id: str, team_id: int) -> Optional[TeamMeetingReport]:
        """只取报告（不连带转写），供 get_report 工具用。"""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """SELECT r.json FROM reports r JOIN meetings m ON m.id=r.meeting_id
                   WHERE m.id=? AND m.team_id=?""",
                (meeting_id, team_id),
            ).fetchone()
        if row is None:
            return None
        return TeamMeetingReport.model_validate(self._report_payload(row["json"]))

    def load_transcript(self, meeting_id: str, team_id: int) -> Optional[Transcript]:
        """加载一场会议的完整转写（供能力工具分析用；已按 team_id 隔离）。"""
        with self._lock, self._connect() as connection:
            meeting = connection.execute(
                "SELECT duration_seconds FROM meetings WHERE id=? AND team_id=?",
                (meeting_id, team_id),
            ).fetchone()
            if meeting is None:
                return None
            rows = connection.execute(
                "SELECT start,end,speaker_label,text FROM transcripts WHERE meeting_id=? ORDER BY id",
                (meeting_id,),
            ).fetchall()
        return Transcript(
            language="zh",
            duration_seconds=meeting["duration_seconds"],
            segments=[
                TranscriptSegment(
                    start=row["start"], end=row["end"],
                    speaker_label=row["speaker_label"], text=row["text"],
                )
                for row in rows
            ],
        )

    def transcript_summary(self, meeting_id: str, team_id: int) -> Optional[Dict[str, Any]]:
        """转写概览：段数、时间范围、说话人时长分布、要点密度（不返回全文）。"""
        with self._lock, self._connect() as connection:
            owned = connection.execute(
                "SELECT 1 FROM meetings WHERE id=? AND team_id=?", (meeting_id, team_id)
            ).fetchone()
            if not owned:
                return None
            aggregate = connection.execute(
                """SELECT COUNT(*) AS n, COALESCE(MIN(start),0) AS s, COALESCE(MAX(end),0) AS e
                   FROM transcripts WHERE meeting_id=?""",
                (meeting_id,),
            ).fetchone()
            speakers = connection.execute(
                """SELECT COALESCE(speaker_label,'') AS label, COALESCE(SUM(end-start),0) AS seconds
                   FROM transcripts WHERE meeting_id=? GROUP BY label ORDER BY seconds DESC""",
                (meeting_id,),
            ).fetchall()
        span = float(aggregate["e"]) - float(aggregate["s"])
        per_minute = (float(aggregate["n"]) / span * 60) if span > 0 else 0.0
        return {
            "total_segments": int(aggregate["n"]),
            "time_range": {"start": round(float(aggregate["s"]), 3), "end": round(float(aggregate["e"]), 3)},
            "speakers": [
                {"label": row["label"], "seconds": round(float(row["seconds"]), 1)} for row in speakers
            ],
            "segments_per_minute": round(per_minute, 2),
        }

    def transcript_page(
        self, meeting_id: str, team_id: int, offset: int, limit: int
    ) -> Optional[List[Dict[str, Any]]]:
        with self._lock, self._connect() as connection:
            owned = connection.execute(
                "SELECT 1 FROM meetings WHERE id=? AND team_id=?", (meeting_id, team_id)
            ).fetchone()
            if not owned:
                return None
            rows = connection.execute(
                """SELECT start,end,speaker_label,text FROM transcripts
                   WHERE meeting_id=? ORDER BY id LIMIT ? OFFSET ?""",
                (meeting_id, max(1, int(limit)), max(0, int(offset))),
            ).fetchall()
        return [dict(row) for row in rows]

    def read_segment_window(
        self, meeting_id: str, team_id: int, *,
        timestamp: Optional[float] = None, segment_index: Optional[int] = None,
        before: int = 5, after: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """回读某时间点/某段前后的转写片段（PRD §9.1 read_segment_window）。"""
        before = max(0, min(int(before), 50))
        after = max(0, min(int(after), 50))
        with self._lock, self._connect() as connection:
            owned = connection.execute(
                "SELECT 1 FROM meetings WHERE id=? AND team_id=?", (meeting_id, team_id)
            ).fetchone()
            if not owned:
                return None
            if segment_index is not None:
                offset = max(0, int(segment_index))
            elif timestamp is not None:
                anchor = connection.execute(
                    """SELECT id FROM transcripts WHERE meeting_id=? AND start<=?
                       ORDER BY start DESC, id DESC LIMIT 1""",
                    (meeting_id, float(timestamp)),
                ).fetchone()
                if anchor is None:
                    first = connection.execute(
                        "SELECT id FROM transcripts WHERE meeting_id=? ORDER BY id LIMIT 1",
                        (meeting_id,),
                    ).fetchone()
                    anchor_id = first["id"] if first else None
                else:
                    anchor_id = anchor["id"]
                if anchor_id is None:
                    return {"segments": [], "window": {"start": 0.0, "end": 0.0},
                            "total_segments": 0, "anchor_index": 0}
                offset = connection.execute(
                    "SELECT COUNT(*) AS c FROM transcripts WHERE meeting_id=? AND id<?",
                    (meeting_id, anchor_id),
                ).fetchone()["c"]
            else:
                offset = 0
            total = connection.execute(
                "SELECT COUNT(*) AS c FROM transcripts WHERE meeting_id=?", (meeting_id,)
            ).fetchone()["c"]
            start_offset = max(0, offset - before)
            rows = connection.execute(
                """SELECT start,end,speaker_label,text FROM transcripts
                   WHERE meeting_id=? ORDER BY id LIMIT ? OFFSET ?""",
                (meeting_id, before + after + 1, start_offset),
            ).fetchall()
        segments = [dict(row) for row in rows]
        return {
            "segments": segments,
            "window": {
                "start": segments[0]["start"] if segments else 0.0,
                "end": segments[-1]["end"] if segments else 0.0,
            },
            "total_segments": int(total),
            "anchor_index": int(offset),
        }

    def search_transcripts(
        self, team_id: int, query: str, *,
        meeting_id: Optional[str] = None, project_id: Optional[str] = None, limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """跨会议/单会议检索转写，严格限定在 team_id 内。

        中文子串检索用 FTS5(trigram)，查询词 ≥ 3 字；更短的或含特殊字符时回退 LIKE。
        """
        text = (query or "").strip()
        if not text:
            return []
        limit = max(1, min(int(limit), 50))
        conditions = ["m.team_id = ?"]
        params: List[Any] = [team_id]
        if meeting_id is not None:
            conditions.append("t.meeting_id = ?")
            params.append(meeting_id)
        if project_id is not None:
            conditions.append("m.project_id = ?")
            params.append(project_id)
        where = " AND ".join(conditions)
        escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        use_fts = (
            self._fts5_available and len(text) >= 3 and not any(ch in text for ch in '"()*:^-')
        )
        with self._lock, self._connect() as connection:
            if use_fts:
                rows = connection.execute(
                    """SELECT t.meeting_id, m.title AS meeting_title, t.start, t.end,
                              t.speaker_label, t.text
                       FROM transcript_fts f
                       JOIN transcripts t ON t.id = f.rowid
                       JOIN meetings m ON m.id = t.meeting_id
                       WHERE transcript_fts MATCH ? AND {}
                       ORDER BY f.rank LIMIT ?""".format(where),
                    (text, *params, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT t.meeting_id, m.title AS meeting_title, t.start, t.end,
                              t.speaker_label, t.text
                       FROM transcripts t JOIN meetings m ON m.id = t.meeting_id
                       WHERE t.text LIKE ? ESCAPE '\\' AND {}
                       ORDER BY t.meeting_id, t.id LIMIT ?""".format(where),
                    ("%{}%".format(escaped), *params, limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def list_action_items(
        self, team_id: int, *, project_id: Optional[str] = None,
        status: Optional[str] = None, limit: int = 200,
    ) -> List[Dict[str, Any]]:
        conditions = ["a.team_id = ?"]
        params: List[Any] = [team_id]
        if project_id is not None:
            conditions.append("m.project_id = ?")
            params.append(project_id)
        if status is not None:
            conditions.append("a.status = ?")
            params.append(status)
        limit = max(1, min(int(limit), 200))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT a.id,a.task,a.owner,a.deadline,a.status,
                          a.meeting_id,m.title AS meeting_title
                   FROM meeting_action_items a JOIN meetings m ON m.id=a.meeting_id
                   WHERE {} ORDER BY a.updated_at DESC, a.id DESC LIMIT ?""".format(
                    " AND ".join(conditions)
                ),
                (*params, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_meetings_filtered(
        self, team_id: int, *, project_id: Optional[str] = None,
        date_from: Optional[str] = None, date_to: Optional[str] = None, limit: int = 20,
    ) -> List[Dict[str, Any]]:
        conditions = ["team_id = ?"]
        params: List[Any] = [team_id]
        if project_id is not None:
            conditions.append("project_id = ?")
            params.append(project_id)
        if date_from is not None:
            conditions.append("created_at >= ?")
            params.append(date_from)
        if date_to is not None:
            conditions.append("created_at <= ?")
            params.append(date_to)
        limit = max(1, min(int(limit), 100))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT id,title,project_id,duration_seconds,status,created_at FROM meetings
                   WHERE {} ORDER BY created_at DESC LIMIT ?""".format(" AND ".join(conditions)),
                (*params, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # R-P1.5-4（阶段 8／M1）：跨会议搜索的产品查询。
    # 与 Agent 工具用的 `search_transcripts` 分开，避免改动已被工具依赖的方法；
    # 两者共用同一套 FTS5(trigram) + LIKE 回退策略与 team_id 隔离条件。
    # ------------------------------------------------------------------

    def search_meetings(
        self, team_id: int, query: str, *, project_id: Optional[str] = None,
        date_from: Optional[str] = None, date_to: Optional[str] = None,
        limit: int = 20, per_meeting_limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """跨会议检索转写片段（严格限定 team_id）。

        - 查询词 ≥ 3 字且不含 FTS 特殊字符时走 FTS5(trigram)，否则回退 LIKE；
        - 同一场会议最多返回 `per_meeting_limit` 条，避免单场会议淹没结果；
        - 返回按命中顺序（FTS 用 rank，LIKE 用会议时间倒序 + 片段顺序）。
        """
        text = (query or "").strip()
        if not text:
            return []
        limit = max(1, min(int(limit), 50))
        per_meeting_limit = max(1, min(int(per_meeting_limit), limit))
        # 先多取一些候选，再按会议数量裁剪；上限防止长会议产生超大扫描结果。
        fetch_limit = min(max(limit * per_meeting_limit * 4, limit), 500)

        conditions = ["m.team_id = ?"]
        params: List[Any] = [team_id]
        if project_id is not None:
            conditions.append("m.project_id = ?")
            params.append(project_id)
        if date_from is not None:
            conditions.append("m.created_at >= ?")
            params.append(date_from)
        if date_to is not None:
            conditions.append("m.created_at <= ?")
            params.append(date_to)
        where = " AND ".join(conditions)
        escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        use_fts = (
            self._fts5_available and len(text) >= 3 and not any(ch in text for ch in '"()*:^-')
        )
        with self._lock, self._connect() as connection:
            if use_fts:
                rows = connection.execute(
                    """SELECT t.meeting_id, m.title AS meeting_title, m.project_id AS project_id,
                              p.name AS project_name, t.start, t.end, t.speaker_label, t.text
                       FROM transcript_fts f
                       JOIN transcripts t ON t.id = f.rowid
                       JOIN meetings m ON m.id = t.meeting_id
                       LEFT JOIN projects p ON p.id = m.project_id
                       WHERE transcript_fts MATCH ? AND {}
                       ORDER BY f.rank LIMIT ?""".format(where),
                    (text, *params, fetch_limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT t.meeting_id, m.title AS meeting_title, m.project_id AS project_id,
                              p.name AS project_name, t.start, t.end, t.speaker_label, t.text
                       FROM transcripts t
                       JOIN meetings m ON m.id = t.meeting_id
                       LEFT JOIN projects p ON p.id = m.project_id
                       WHERE t.text LIKE ? ESCAPE '\\' AND {}
                       ORDER BY m.created_at DESC, t.start ASC LIMIT ?""".format(where),
                    ("%{}%".format(escaped), *params, fetch_limit),
                ).fetchall()

        hits: List[Dict[str, Any]] = []
        seen_per_meeting: Dict[str, int] = {}
        for row in rows:
            item = dict(row)
            meeting_id = str(item.pop("meeting_id"))
            used = seen_per_meeting.get(meeting_id, 0)
            if used >= per_meeting_limit:
                continue
            seen_per_meeting[meeting_id] = used + 1
            hits.append({**item, "meeting_id": meeting_id})
            if len(hits) >= limit:
                break
        return hits

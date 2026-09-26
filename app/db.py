from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.attribution import attribute_report, display_speaker, speaker_for_evidence
from app.conclusions import dedupe_report
from app.followups import fingerprint
from app.models import (
    ActionStatus, DEFAULT_MEETING_TITLE, MeetingHistory, MeetingListItem, MeetingSource, MeetingSpeaker,
    MemberIdentity, ProjectListItem, SpeakerClip,
    SelfSpeakerResult,
    FollowupItem, FollowupList, FollowupStatus,
    SpeakerDigest, SpeakerItem, SpeakerSection,
    ProjectMemory, ProjectMemoryAction, ProjectMemoryDecision, ProjectMemoryIssue,
    TeamMeetingReport, Transcript, TranscriptSegment,
)


logger = logging.getLogger(__name__)

# 用于区分「未传参数」与「显式传 None」（None = 未归类桶，project_id IS NULL）。
_UNSET = object()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# R-P2.2（阶段 18）：「谁说了什么」——把报告条目按「谁说的」分组。
# 行动项（action）按产品经理 2026-09-22 决定**不进入**这个视图（负责人不准，已收起）；
# 只保留每个人说的关键结论（决策）与他提出/悬而未决的问题（待跟进、紧急）。
_KIND_FIELDS = (
    ("decision", "decisions"),
    ("issue", "unresolved_issues"),
    ("urgent", "urgent_items"),
)


def _sort_sections(
    buckets: Dict[Optional[str], List[SpeakerItem]], key_names: Dict[str, bool],
) -> List[SpeakerSection]:
    """★关键决策人优先 → 其余按条目数 → 「未标注说话人」最后。"""
    sections = [
        SpeakerSection(
            speaker=speaker,
            is_key_decision_maker=bool(speaker and key_names.get(speaker)),
            items=items,
        )
        for speaker, items in buckets.items()
    ]
    sections.sort(key=lambda section: (
        0 if section.is_key_decision_maker else (2 if section.speaker is None else 1),
        -len(section.items), section.speaker or "",
    ))
    return sections


def build_speaker_sections(
    report: TeamMeetingReport, key_names: Dict[str, bool],
    *, meeting_id: Optional[str] = None, meeting_title: Optional[str] = None,
) -> List[SpeakerSection]:
    """把一场报告的四类条目按 `speaker`（已归属）分组；无人称的归入 None。"""
    buckets: Dict[Optional[str], List[SpeakerItem]] = {}
    for kind, attr in _KIND_FIELDS:
        for item in (getattr(report, attr, None) or []):
            speaker = getattr(item, "speaker", None) or None
            text = str(getattr(item, "task", None) or getattr(item, "content", "") or "")
            if not text:
                continue
            buckets.setdefault(speaker, []).append(SpeakerItem(
                kind=kind, text=text, evidence=getattr(item, "evidence", None),
                meeting_id=meeting_id, meeting_title=meeting_title,
            ))
    return _sort_sections(buckets, key_names)


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

    def initialize(
        self, team_tokens: Dict[str, str], owner_name: str = "我", auth_enabled: bool = False,
    ) -> None:
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
                -- R-P1.5-3（阶段 11／M4）：分享链接与访问审计。只存令牌哈希。
                CREATE TABLE IF NOT EXISTS share_links(
                    token_hash TEXT PRIMARY KEY,
                    team_id INTEGER NOT NULL,
                    meeting_id TEXT NOT NULL,
                    scopes_json TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by_token_hash TEXT NULL,
                    revoked_at TEXT NULL,
                    view_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_share_links_meeting
                    ON share_links(meeting_id, team_id);
                CREATE TABLE IF NOT EXISTS share_audit(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    share_token_hash TEXT NOT NULL,
                    team_id INTEGER NOT NULL,
                    meeting_id TEXT NOT NULL,
                    accessed_at TEXT NOT NULL,
                    ip TEXT NULL,
                    result_code TEXT NOT NULL
                );
                -- R-P2-2/3/11：个人模式的键值状态（schema 版本 / 个人工作区 / 本人声纹引导）。
                CREATE TABLE IF NOT EXISTS app_state(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                -- R-P2-2：本人声纹（全局唯一一行，id 固定为 1）。
                CREATE TABLE IF NOT EXISTS owner_voiceprints(
                    id INTEGER PRIMARY KEY CHECK(id=1),
                    team_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    sample_seconds REAL NOT NULL DEFAULT 0,
                    enrolled_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                -- R-P2-6：归类留痕（人工/自动都记，支持撤销）。
                CREATE TABLE IF NOT EXISTS project_assignments(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id INTEGER NOT NULL,
                    meeting_id TEXT NOT NULL,
                    from_project_id TEXT NULL,
                    to_project_id TEXT NULL,
                    source TEXT NOT NULL CHECK(source IN ('manual','auto')),
                    confidence REAL NULL,
                    reason TEXT NULL,
                    created_at TEXT NOT NULL,
                    undone_at TEXT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_project_assignments_meeting
                    ON project_assignments(team_id, meeting_id, id DESC);
                -- R-P2-5：批量上传批次。
                CREATE TABLE IF NOT EXISTS review_batches(
                    batch_id TEXT PRIMARY KEY,
                    team_id INTEGER NOT NULL,
                    total INTEGER NOT NULL,
                    consent_version TEXT NOT NULL,
                    ip TEXT NULL,
                    created_at TEXT NOT NULL
                );
                -- R-P1.5-9（阶段 10-B）：术语热词表。
                -- R-P2-9：本版停用（界面与接口已删除），表保留以兼容历史库，不再读写。
                CREATE TABLE IF NOT EXISTS team_terms(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id INTEGER NOT NULL,
                    term TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(team_id, term)
                );
                CREATE INDEX IF NOT EXISTS idx_team_terms_team ON team_terms(team_id, term);
                -- R-P1.5-6（阶段 10-A）：上传同意留证。
                CREATE TABLE IF NOT EXISTS consent_records(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id TEXT NOT NULL,
                    team_id INTEGER NOT NULL,
                    consent_version TEXT NOT NULL,
                    consented_at TEXT NOT NULL,
                    ip TEXT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_consent_meeting
                    ON consent_records(meeting_id, team_id);
                -- R-P1.5-7（阶段 9-C）：四类交付物各自独立成/败。
                CREATE TABLE IF NOT EXISTS deliverable_status(
                    meeting_id TEXT NOT NULL,
                    team_id INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending','ok','failed','needs_review')),
                    error_code TEXT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (meeting_id, kind)
                );
                -- R-P2.1-7（阶段 17／M2）：待跟进跨会议跟踪。
                -- 红线：项目内合并（UNIQUE(project_id,fingerprint)），跨项目绝不合并；
                -- 状态只能人改（本阶段不实现 AI 建议已解决）。
                CREATE TABLE IF NOT EXISTS followups(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id INTEGER NOT NULL,
                    project_id TEXT NOT NULL REFERENCES projects(id),
                    fingerprint TEXT NOT NULL,
                    text TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open'
                        CHECK(status IN ('open','resolved','dropped')),
                    first_meeting_id TEXT REFERENCES meetings(id) ON DELETE SET NULL,
                    first_speaker TEXT NULL,
                    last_seen_meeting_id TEXT REFERENCES meetings(id) ON DELETE SET NULL,
                    resolved_meeting_id TEXT REFERENCES meetings(id) ON DELETE SET NULL,
                    resolved_evidence_json TEXT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(project_id, fingerprint)
                );
                CREATE INDEX IF NOT EXISTS idx_followups_project_status
                    ON followups(team_id, project_id, status, updated_at DESC);
                -- 一条待跟进出现过的所有场次（含首提与后续重现）。
                CREATE TABLE IF NOT EXISTS followup_meetings(
                    followup_id INTEGER NOT NULL REFERENCES followups(id) ON DELETE CASCADE,
                    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
                    UNIQUE(followup_id, meeting_id)
                );
                """
            )
            self._ensure_column(connection, "meetings", "project_id", "TEXT REFERENCES projects(id)")
            self._ensure_column(connection, "meetings", "speaker_consent_at", "TEXT")
            # R-P1.5-5（阶段 9／M2）：本场「我」的指认。两者均可为空 → 代表未指定。
            self._ensure_column(connection, "meetings", "self_member_id", "INTEGER")
            self._ensure_column(connection, "meetings", "self_label", "TEXT")
            self._ensure_column(connection, "projects", "parent_id", "TEXT REFERENCES projects(id)")
            self._ensure_column(connection, "members", "role", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(
                connection, "members", "is_key_decision_maker", "INTEGER NOT NULL DEFAULT 0"
            )
            self._ensure_column(
                connection, "meeting_speakers", "remember_requested", "INTEGER NOT NULL DEFAULT 0"
            )
            # R-P2-4：声纹库从「团队全局」改为「项目级」。project_id 为空 = 未归类桶。
            self._ensure_column(connection, "members", "project_id", "TEXT REFERENCES projects(id)")
            self._ensure_column(connection, "speaker_profiles", "project_id", "TEXT")
            # R-P2-3：「我」的来源与置信度。
            self._ensure_column(connection, "meetings", "self_source", "TEXT")
            self._ensure_column(connection, "meetings", "self_confidence", "REAL")
            # R-P2-6：归类来源与置信度。
            self._ensure_column(connection, "meetings", "assignment_source", "TEXT")
            self._ensure_column(connection, "meetings", "assignment_confidence", "REAL")
            # R-P2-5：任务所属批次。
            self._ensure_column(connection, "agent_tasks", "batch_id", "TEXT")
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
                CREATE INDEX IF NOT EXISTS idx_members_team_project ON members(team_id, project_id, name);
                CREATE INDEX IF NOT EXISTS idx_speaker_profiles_team ON speaker_profiles(team_id);
                CREATE INDEX IF NOT EXISTS idx_speaker_profiles_team_project
                    ON speaker_profiles(team_id, project_id);
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
            rows = (
                connection.execute(
                    "SELECT id, name FROM teams WHERE name IN ({})".format(
                        ",".join("?" for _ in team_tokens)
                    ),
                    tuple(team_tokens),
                ).fetchall()
                if team_tokens
                else []
            )
            by_name = {row["name"]: row["id"] for row in rows}
            self._tokens = {token: by_name[name] for name, token in team_tokens.items()}
            if auth_enabled:
                # 进阶模式：不会创建额外工作区；仅当只有一个团队时记录为个人工作区。
                existing = connection.execute(
                    "SELECT id FROM teams ORDER BY id LIMIT 2"
                ).fetchall()
                self._owner_workspace_id = int(existing[0]["id"]) if len(existing) == 1 else None
            else:
                self._owner_workspace_id = self._ensure_personal_workspace(connection, owner_name)
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
            # R-P2.1-7：项目删除时其待跟进一并删除（followup_meetings 随 FK 级联）。
            connection.execute(
                "DELETE FROM followups WHERE team_id=? AND project_id=?", (team_id, project_id)
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
            connection.execute(
                "DELETE FROM deliverable_status WHERE meeting_id=? AND team_id=?",
                (meeting_id, team_id),
            )
            connection.execute(
                "DELETE FROM consent_records WHERE meeting_id=? AND team_id=?",
                (meeting_id, team_id),
            )
            connection.execute(
                "DELETE FROM share_links WHERE meeting_id=? AND team_id=?", (meeting_id, team_id),
            )
            connection.execute(
                "DELETE FROM share_audit WHERE meeting_id=? AND team_id=?", (meeting_id, team_id),
            )
            # R-P2.1-7：解除待跟进对被删会议的引用（首提/最近/解决均置空），
            # 出场记录（followup_meetings）一并删除。待跟进本身保留，状态不动。
            connection.execute(
                "UPDATE followups SET first_meeting_id=NULL WHERE first_meeting_id=?",
                (meeting_id,),
            )
            connection.execute(
                "UPDATE followups SET last_seen_meeting_id=NULL WHERE last_seen_meeting_id=?",
                (meeting_id,),
            )
            connection.execute(
                "UPDATE followups SET resolved_meeting_id=NULL WHERE resolved_meeting_id=?",
                (meeting_id,),
            )
            connection.execute(
                "DELETE FROM followup_meetings WHERE meeting_id=?", (meeting_id,)
            )
            connection.execute("DELETE FROM meetings WHERE id=? AND team_id=?", (meeting_id, team_id))

    def update_status(self, meeting_id: str, team_id: int, status: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE meetings SET status=? WHERE id=? AND team_id=?", (status, meeting_id, team_id)
            )

    def move_meeting(
        self, meeting_id: str, team_id: int, project_id: str, *,
        source: str = "manual", confidence: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> bool:
        now = _utc_now()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT project_id FROM meetings WHERE id=? AND team_id=?",
                (meeting_id, team_id),
            ).fetchone()
            if row is None:
                return False
            from_project_id = row["project_id"]
            cursor = connection.execute(
                """UPDATE meetings SET project_id=?, assignment_source=?, assignment_confidence=?
                   WHERE id=? AND team_id=?""",
                (project_id, source if from_project_id != project_id else None,
                 confidence if from_project_id != project_id else None, meeting_id, team_id),
            )
            if cursor.rowcount != 1:
                return False
            # R-P2-6：归类留痕（人工与自动都记）。
            if from_project_id != project_id:
                connection.execute(
                    """INSERT INTO project_assignments(
                           team_id, meeting_id, from_project_id, to_project_id, source,
                           confidence, reason, created_at, undone_at)
                       VALUES(?,?,?,?,?,?,?,?,NULL)""",
                    (team_id, meeting_id, from_project_id, project_id, source,
                     confidence, reason, now),
                )
            # R-P2-4 ④：未归类成员随会议迁入目标项目（只动 project_id 为空的人）。
            connection.execute(
                """UPDATE members SET project_id=?
                   WHERE team_id=? AND project_id IS NULL AND id IN (
                       SELECT member_id FROM meeting_speakers
                       WHERE meeting_id=? AND team_id=? AND member_id IS NOT NULL)""",
                (project_id, team_id, meeting_id, team_id),
            )
            connection.execute(
                """UPDATE speaker_profiles SET project_id=?, updated_at=?
                   WHERE member_id IN (
                       SELECT member_id FROM meeting_speakers
                       WHERE meeting_id=? AND team_id=? AND member_id IS NOT NULL)
                     AND project_id IS NULL""",
                (project_id, now, meeting_id, team_id),
            )
        return True

    def assign_meeting_auto(
        self, meeting_id: str, team_id: int, project_id: str,
        confidence: Optional[float], reason: Optional[str] = None,
    ) -> bool:
        """R-P2-6：只把**未归类**会议自动归入**已存在**项目（永不自动新建）。"""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT project_id FROM meetings WHERE id=? AND team_id=?",
                (meeting_id, team_id),
            ).fetchone()
            owner = connection.execute(
                "SELECT team_id FROM projects WHERE id=?", (project_id,)
            ).fetchone()
        if row is None or owner is None or int(owner["team_id"]) != team_id:
            return False
        if row["project_id"] is not None:
            return False
        return self.move_meeting(
            meeting_id, team_id, project_id, source="auto",
            confidence=confidence, reason=reason,
        )

    def undo_assignment(self, meeting_id: str, team_id: int) -> bool:
        """R-P2-6 ①/红线 16：撤销最近一次自动归类，回到原项目（通常为未归类），并留痕。"""
        now = _utc_now()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """SELECT id, from_project_id FROM project_assignments
                   WHERE meeting_id=? AND team_id=? AND source='auto' AND undone_at IS NULL
                   ORDER BY id DESC LIMIT 1""",
                (meeting_id, team_id),
            ).fetchone()
            if row is None:
                return False
            connection.execute(
                """UPDATE meetings SET project_id=?, assignment_source=NULL,
                          assignment_confidence=NULL WHERE id=? AND team_id=?""",
                (row["from_project_id"], meeting_id, team_id),
            )
            connection.execute(
                "UPDATE project_assignments SET undone_at=? WHERE id=?", (now, row["id"])
            )
        return True

    def list_assignments(self, meeting_id: str, team_id: int) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT id, from_project_id, to_project_id, source, confidence, reason,
                          created_at, undone_at FROM project_assignments
                   WHERE meeting_id=? AND team_id=? ORDER BY id DESC""",
                (meeting_id, team_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def meeting_speaker_embedding(
        self, meeting_id: str, team_id: int, local_label: str,
    ) -> Optional[Dict[str, Any]]:
        """取本场某个说话人的 embedding（用于「认领为我的声纹」）。"""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """SELECT local_label, member_id, embedding_json, speech_seconds
                   FROM meeting_speakers
                   WHERE meeting_id=? AND team_id=? AND local_label=?""",
                (meeting_id, team_id, local_label),
            ).fetchone()
        if row is None or not row["embedding_json"]:
            return None
        return dict(row)

    def create_review_batch(
        self, batch_id: str, team_id: int, total: int, consent_version: str, ip: str,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO review_batches(
                       batch_id, team_id, total, consent_version, ip, created_at)
                   VALUES(?,?,?,?,?,?)""",
                (batch_id, team_id, total, consent_version, ip, _utc_now()),
            )

    def review_batch(self, batch_id: str, team_id: int) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """SELECT batch_id, total, consent_version, created_at
                   FROM review_batches WHERE batch_id=? AND team_id=?""",
                (batch_id, team_id),
            ).fetchone()
            if row is None:
                return None
            task_rows = connection.execute(
                """SELECT task_id, status, stage, message, created_at FROM agent_tasks
                   WHERE team_id=? AND batch_id=? ORDER BY created_at""",
                (team_id, batch_id),
            ).fetchall()
        return {**dict(row), "tasks": [dict(item) for item in task_rows]}

    def set_task_batch(self, task_id: str, batch_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE agent_tasks SET batch_id=? WHERE task_id=?", (batch_id, task_id)
            )

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

    def voice_profiles(
        self, team_id: int, model_version: Optional[str] = None, project_id: Any = _UNSET,
    ) -> list[dict]:
        """按项目返回可用于匹配的声纹。

        R-P2-4：默认（传 `project_id`）只返回**同一项目**的成员声纹；
        `project_id=None` 表示「未归类」桶（`project_id IS NULL`）；
        不传 `project_id`（哨兵 `_UNSET`）时保持旧行为（整团队），仅供历史调用与测试。
        这样 A 项目的声纹不会出现在 B 项目（红线 13）。
        """
        scope_sql, scope_args = "", []
        if project_id is None:
            scope_sql = " AND m.project_id IS NULL"
        elif project_id is not _UNSET:
            scope_sql = " AND m.project_id=?"
            scope_args = [project_id]
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT m.id AS member_id,m.name,p.embedding_json
                   FROM speaker_profiles p JOIN members m ON m.id=p.member_id
                   WHERE p.team_id=? AND m.team_id=?""" + scope_sql + """
                     AND (? IS NULL OR p.model_version=?) ORDER BY m.id""",
                (team_id, team_id, *scope_args, model_version, model_version),
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

    def set_self_speaker(
        self, meeting_id: str, team_id: int, *,
        member_id: Optional[int] = None, label: Optional[str] = None,
    ) -> bool:
        """R-P1.5-5：人工指认本场「我」（两者都为空 = 清除）。

        R-P2-3 ⑥：人工指认优先——写入 source='manual'，后续声纹识别不再覆盖。
        """
        source = "manual" if (member_id is not None or label) else None
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """UPDATE meetings SET self_member_id=?, self_label=?, self_source=?,
                          self_confidence=NULL WHERE id=? AND team_id=?""",
                (member_id, label, source, meeting_id, team_id),
            )
            return cursor.rowcount > 0

    def self_speaker(self, meeting_id: str, team_id: int) -> Optional[Dict[str, Any]]:
        """本场「我」的指认；未指定返回 None。绝不推断。

        - 选了已命名说话人：member_id 有值，self_name 取成员姓名；
        - 选了未命名说话人：只有 label，self_name 取说话人标签（如「说话人 1」）；
        - 两者都空：返回 None（上层一律显示「未指定你自己」）。
        """
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """SELECT m.self_member_id, m.self_label, m.self_source, m.self_confidence,
                          mem.name AS member_name
                   FROM meetings m LEFT JOIN members mem ON mem.id = m.self_member_id
                   WHERE m.id=? AND m.team_id=?""",
                (meeting_id, team_id),
            ).fetchone()
            if row is None:
                return None
            member_id = row["self_member_id"]
            label = row["self_label"]
            member_name = row["member_name"]
            if member_id is None and not label:
                return None
            if not label:
                speaker = connection.execute(
                    """SELECT local_label FROM meeting_speakers
                       WHERE meeting_id=? AND team_id=? AND member_id=? ORDER BY id LIMIT 1""",
                    (meeting_id, team_id, member_id),
                ).fetchone()
                label = speaker["local_label"] if speaker else None
        return {
            "member_id": member_id,
            "local_label": label,
            "member_name": member_name,
            "self_name": member_name or label,
            "source": row["self_source"],
            "confidence": row["self_confidence"],
        }

    # --- R-P2-2/3/12：本人声纹与「我」的识别 --------------------------------
    def owner_voiceprint(self, team_id: int) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """SELECT id, team_id, name, embedding_json, model_version,
                          sample_seconds, enrolled_at, updated_at
                   FROM owner_voiceprints WHERE team_id=? AND id=1""",
                (team_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "name": row["name"],
            "embedding": json.loads(row["embedding_json"]),
            "model_version": row["model_version"],
            "sample_seconds": float(row["sample_seconds"]),
            "enrolled_at": row["enrolled_at"],
            "updated_at": row["updated_at"],
        }

    def save_owner_voiceprint(
        self, team_id: int, name: str, embedding: list, model_version: str,
        sample_seconds: float,
    ) -> None:
        now = _utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO owner_voiceprints(
                       id, team_id, name, embedding_json, model_version,
                       sample_seconds, enrolled_at, updated_at)
                   VALUES(1,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                       team_id=excluded.team_id, name=excluded.name,
                       embedding_json=excluded.embedding_json,
                       model_version=excluded.model_version,
                       sample_seconds=excluded.sample_seconds,
                       updated_at=excluded.updated_at""",
                (team_id, name, json.dumps(embedding), model_version, sample_seconds, now, now),
            )

    def delete_owner_voiceprint(self, team_id: int) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM owner_voiceprints WHERE id=1 AND team_id=?", (team_id,)
            )
        return cursor.rowcount > 0

    def set_owner_identification(
        self, meeting_id: str, team_id: int, label: Optional[str],
        confidence: Optional[float], source: Optional[str],
    ) -> bool:
        """写回本场「我」的声纹识别结果；**人工指认优先，不覆盖**（R-P2-3 ⑥）。"""
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """UPDATE meetings SET self_label=?, self_source=?, self_confidence=?
                   WHERE id=? AND team_id=? AND (self_source IS NULL OR self_source!='manual')""",
                (label, source, confidence, meeting_id, team_id),
            )
            return cursor.rowcount > 0

    def list_backfill_meetings(self, team_id: int, limit: int) -> list[Dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT id, title, self_source FROM meetings
                   WHERE team_id=? AND status='完成'
                   ORDER BY created_at DESC, id DESC LIMIT ?""",
                (team_id, int(limit)),
            ).fetchall()
        return [dict(row) for row in rows]

    def meeting_speaker_candidates(self, meeting_id: str, team_id: int) -> list[Dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT s.local_label, s.embedding_json, s.speech_seconds, s.member_id,
                          s.confidence, mem.name AS member_name
                   FROM meeting_speakers s LEFT JOIN members mem ON mem.id = s.member_id
                   WHERE s.meeting_id=? AND s.team_id=? ORDER BY s.id""",
                (meeting_id, team_id),
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # R-P1.5-7（阶段 9-C）：交付物状态。四类交付物各自独立成/败。
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # R-P1.5-6（阶段 10-A）：上传同意留证。
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # R-P1.5-9（阶段 10-B）：团队术语热词表（手动维护部分）。
    # --- R-P2-1/2/11：个人工作区与 app_state --------------------------------
    def _ensure_personal_workspace(self, connection: sqlite3.Connection, owner_name: str) -> int:
        """解析个人工作区（单行 teams 的 id）。

        优先用 app_state 里记录过的 id；否则复用现有唯一团队（升级自 v1.2 的真实库），
        没有团队时新建一个以 OWNER_NAME 命名的工作区。
        """
        recorded = self._app_state_value(connection, "owner_workspace_id")
        if recorded is not None:
            row = connection.execute("SELECT id FROM teams WHERE id=?", (recorded,)).fetchone()
            if row is not None:
                return int(row["id"])
        existing = connection.execute(
            "SELECT id FROM teams ORDER BY id LIMIT 2"
        ).fetchall()
        if len(existing) == 1:
            workspace_id = int(existing[0]["id"])
        else:
            name = (owner_name or "我").strip() or "我"
            cursor = connection.execute(
                "INSERT INTO teams(name, token_hash, created_at) VALUES(?,?,?)",
                (name, _token_hash(_utc_now()), _utc_now()),
            )
            workspace_id = int(cursor.lastrowid)
        self._app_state_set(connection, "owner_workspace_id", str(workspace_id))
        return workspace_id

    @staticmethod
    def _app_state_value(connection: sqlite3.Connection, key: str) -> Optional[str]:
        row = connection.execute("SELECT value FROM app_state WHERE key=?", (key,)).fetchone()
        return str(row["value"]) if row is not None else None

    @staticmethod
    def _app_state_set(connection: sqlite3.Connection, key: str, value: str) -> None:
        connection.execute(
            """INSERT INTO app_state(key, value, updated_at) VALUES(?,?,?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
            (key, value, _utc_now()),
        )

    def get_app_state(self, key: str) -> Optional[str]:
        with self._lock, self._connect() as connection:
            return self._app_state_value(connection, key)

    def set_app_state(self, key: str, value: str) -> None:
        with self._lock, self._connect() as connection:
            self._app_state_set(connection, key, value)

    def owner_workspace_id(self) -> Optional[int]:
        """个人模式下的固定工作区 id（服务启动时已解析并缓存）。"""
        return getattr(self, "_owner_workspace_id", None)

    # ------------------------------------------------------------------
    # R-P1.5-3（阶段 11／M4）：分享链接与访问审计。
    # 表里只存令牌哈希；原文仅在创建时返回一次。
    # ------------------------------------------------------------------

    def create_share_link(
        self, token_hash: str, team_id: int, meeting_id: str, scopes: List[str],
        expires_at: str, created_by_token_hash: Optional[str] = None,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO share_links(
                       token_hash,team_id,meeting_id,scopes_json,expires_at,created_at,
                       created_by_token_hash,revoked_at,view_count)
                   VALUES(?,?,?,?,?,?,?,NULL,0)""",
                (token_hash, team_id, meeting_id, json.dumps(list(scopes), ensure_ascii=False),
                 expires_at, _utc_now(), created_by_token_hash),
            )

    def share_link(self, token_hash: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """SELECT token_hash,team_id,meeting_id,scopes_json,expires_at,created_at,
                          revoked_at,view_count
                   FROM share_links WHERE token_hash=?""",
                (token_hash,),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["scopes"] = json.loads(item.pop("scopes_json"))
        return item

    def list_share_links(self, meeting_id: str, team_id: int) -> List[Dict[str, Any]]:
        """列表带 `share_id`（SQLite rowid），用于撤销——数据库里没有令牌原文，无法回显。"""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT rowid AS share_id, token_hash, scopes_json, expires_at, created_at,
                          revoked_at, view_count
                   FROM share_links WHERE meeting_id=? AND team_id=? ORDER BY created_at DESC""",
                (meeting_id, team_id),
            ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["scopes"] = json.loads(item.pop("scopes_json"))
            items.append(item)
        return items

    def revoke_share_link(self, token_hash: str, team_id: int) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """UPDATE share_links SET revoked_at=?
                   WHERE token_hash=? AND team_id=? AND revoked_at IS NULL""",
                (_utc_now(), token_hash, team_id),
            )
            return cursor.rowcount > 0

    def revoke_share_link_by_id(self, share_id: int, team_id: int) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """UPDATE share_links SET revoked_at=?
                   WHERE rowid=? AND team_id=? AND revoked_at IS NULL""",
                (_utc_now(), int(share_id), team_id),
            )
            return cursor.rowcount > 0

    def count_share_view(self, token_hash: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE share_links SET view_count=view_count+1 WHERE token_hash=?", (token_hash,),
            )

    def record_share_access(
        self, token_hash: str, team_id: int, meeting_id: str, result_code: str,
        ip: Optional[str] = None,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO share_audit(
                       share_token_hash,team_id,meeting_id,accessed_at,ip,result_code)
                   VALUES(?,?,?,?,?,?)""",
                (token_hash, team_id, meeting_id, _utc_now(), ip, result_code),
            )

    def share_audit_rows(self, meeting_id: str, team_id: int) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT share_token_hash,accessed_at,ip,result_code FROM share_audit
                   WHERE meeting_id=? AND team_id=? ORDER BY id""",
                (meeting_id, team_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_consent(
        self, meeting_id: str, team_id: int, consent_version: str, ip: Optional[str] = None,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO consent_records(meeting_id,team_id,consent_version,consented_at,ip)
                   VALUES(?,?,?,?,?)""",
                (meeting_id, team_id, consent_version, _utc_now(), ip),
            )

    def consent_records(self, meeting_id: str, team_id: int) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT consent_version,consented_at,ip FROM consent_records
                   WHERE meeting_id=? AND team_id=? ORDER BY id""",
                (meeting_id, team_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def set_deliverable_status(
        self, meeting_id: str, team_id: int, kind: str, status: str,
        error_code: Optional[str] = None,
    ) -> None:
        if status not in ("pending", "ok", "failed", "needs_review"):
            raise ValueError("未知交付物状态：{}".format(status))
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO deliverable_status(meeting_id,team_id,kind,status,error_code,updated_at)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(meeting_id,kind) DO UPDATE SET
                       status=excluded.status, error_code=excluded.error_code,
                       updated_at=excluded.updated_at, team_id=excluded.team_id""",
                (meeting_id, team_id, kind, status, error_code, _utc_now()),
            )

    def deliverable_statuses(self, meeting_id: str, team_id: int) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT kind,status,error_code,updated_at FROM deliverable_status
                   WHERE meeting_id=? AND team_id=? ORDER BY kind""",
                (meeting_id, team_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_meeting_action_items(self, meeting_id: str, team_id: int) -> List[Dict[str, Any]]:
        """本场全部行动项（只读），供「我答应的任务」切面使用。"""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT a.id,a.task,a.owner,a.deadline,a.status
                   FROM meeting_action_items a JOIN meetings m ON m.id=a.meeting_id
                   WHERE a.meeting_id=? AND a.team_id=? AND m.team_id=?
                   ORDER BY a.item_index, a.id""",
                (meeting_id, team_id, team_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_members(self, team_id: int) -> list[MemberIdentity]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT m.id,m.name,m.role,m.is_key_decision_maker,m.created_at,m.project_id,
                          CASE WHEN p.id IS NULL THEN 0 ELSE 1 END AS has_voiceprint
                   FROM members m LEFT JOIN speaker_profiles p ON p.member_id=m.id
                   WHERE m.team_id=? ORDER BY m.name,m.id""",
                (team_id,),
            ).fetchall()
        return [MemberIdentity(
            id=row["id"], name=row["name"], role=row["role"],
            is_key_decision_maker=bool(row["is_key_decision_maker"]),
            has_voiceprint=bool(row["has_voiceprint"]), created_at=row["created_at"],
            project_id=row["project_id"],
        ) for row in rows]

    def list_project_members(self, project_id: str, team_id: int) -> list[MemberIdentity]:
        """R-P2-4：只返回本项目成员（项目级声纹库）。"""
        return [
            item for item in self.list_members(team_id) if item.project_id == project_id
        ]

    def member_project_id(self, member_id: int) -> Optional[str]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT project_id FROM members WHERE id=?", (member_id,)
            ).fetchone()
        return row["project_id"] if row else None

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
        """Map a local label; voiceprint persistence is deferred to meeting finalization.

        R-P2-4：成员按（项目 + 姓名）查重并创建——同名的人在不同项目是不同成员。
        """
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
            meeting_row = connection.execute(
                "SELECT project_id FROM meetings WHERE id=? AND team_id=?", (meeting_id, team_id)
            ).fetchone()
            project_id = meeting_row["project_id"] if meeting_row else None
            member = connection.execute(
                """SELECT id FROM members
                   WHERE team_id=? AND name=? AND project_id IS ?
                   ORDER BY id LIMIT 1""",
                (team_id, name, project_id),
            ).fetchone()
            if member:
                member_id = int(member["id"])
                connection.execute(
                    """UPDATE members SET role=?,is_key_decision_maker=? WHERE id=? AND team_id=?""",
                    (role, int(is_key_decision_maker), member_id, team_id),
                )
            else:
                cursor = connection.execute(
                    """INSERT INTO members(team_id,name,role,is_key_decision_maker,created_at,project_id)
                       VALUES(?,?,?,?,?,?)""",
                    (team_id, name, role, int(is_key_decision_maker), now, project_id),
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
                member_project = connection.execute(
                    "SELECT project_id FROM members WHERE id=?", (row["member_id"],)
                ).fetchone()
                profile_project = member_project["project_id"] if member_project else None
                connection.execute(
                    """INSERT INTO speaker_profiles(
                           team_id,member_id,embedding_json,model_version,consented_at,updated_at,project_id
                       ) VALUES(?,?,?,?,?,?,?)
                       ON CONFLICT(member_id) DO UPDATE SET
                           embedding_json=excluded.embedding_json,
                           model_version=excluded.model_version,
                           consented_at=excluded.consented_at,updated_at=excluded.updated_at,
                           project_id=excluded.project_id""",
                    (
                        team_id, row["member_id"], row["embedding_json"],
                        model_version, now, now, profile_project,
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
                """SELECT id,title,project_id,duration_seconds,status,created_at,
                          assignment_source,assignment_confidence FROM meetings
                   WHERE {} ORDER BY created_at DESC""".format(" AND ".join(conditions)),
                tuple(parameters),
            ).fetchall()
        return [MeetingListItem.model_validate(dict(row)) for row in rows]

    def get_history(self, meeting_id: str, team_id: int) -> Optional[MeetingHistory]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """SELECT m.id,m.title,m.project_id,m.duration_seconds,m.status,m.created_at,
                          m.speaker_consent_at,m.assignment_source,m.assignment_confidence,r.json
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
                          s.excerpts_json,s.embedding_json,s.remember_requested,
                          m.name,m.is_key_decision_maker
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
        self_speaker = self.self_speaker(meeting_id, team_id)
        segments = [TranscriptSegment.model_validate(dict(item)) for item in transcript_rows]
        label_to_name = {
            str(item["local_label"]): (item["name"] or str(item["local_label"]))
            for item in speaker_rows
        }
        # R-P2.1：读取时确定性互斥/去重 + 说话人归属（旧报告不改库也能生效）。
        report, _ = dedupe_report(TeamMeetingReport.model_validate(report_payload))
        report = attribute_report(report, segments, label_to_name)
        # R-P2.2（阶段 18）：「谁说了什么」按人分组（★关键决策人由用户标记）。
        key_names = {
            str(item["name"]): bool(item["is_key_decision_maker"])
            for item in speaker_rows if item["name"]
        }
        speaker_digest = SpeakerDigest(sections=build_speaker_sections(
            report, key_names, meeting_id=row["id"], meeting_title=row["title"],
        ))
        clips_by_label: dict[str, list[SpeakerClip]] = {}
        for clip in clip_rows:
            clips_by_label.setdefault(clip["local_label"], []).append(SpeakerClip(
                id=clip["id"], start=clip["start"], end=clip["end"], text=clip["text"],
            ))
        return MeetingHistory(
            id=row["id"], title=row["title"], project_id=row["project_id"],
            duration_seconds=row["duration_seconds"],
            status=row["status"], created_at=row["created_at"],
            assignment_source=row["assignment_source"],
            assignment_confidence=row["assignment_confidence"],
            report=report,
            transcript=segments,
            speaker_consent_confirmed=row["speaker_consent_at"] is not None,
            self_speaker=SelfSpeakerResult(**self_speaker, self_speaker_set=True) if self_speaker else None,
            speaker_digest=speaker_digest,
            speakers=[MeetingSpeaker(
                local_label=item["local_label"],
                display_name=item["name"] or item["local_label"],
                member_id=item["member_id"], confidence=item["confidence"],
                status=item["status"], speech_seconds=item["speech_seconds"],
                excerpts=json.loads(item["excerpts_json"]),
                clips=clips_by_label.get(item["local_label"], []),
                has_voice_sample=item["embedding_json"] is not None,
                remember_requested=bool(item["remember_requested"]),
                is_key_decision_maker=bool(item["is_key_decision_maker"]),
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
                    """SELECT id,meeting_id,item_index,task,status FROM meeting_action_items
                       WHERE team_id=? AND meeting_id IN ({})""".format(
                        ",".join("?" for _ in meeting_ids)
                    ),
                    (team_id, *meeting_ids),
                ).fetchall()
            else:
                action_rows = []

        status_by_task: dict = {}
        for row in action_rows:
            status_by_task.setdefault((row["meeting_id"], row["task"]), []).append(row)
        recent_meetings: list[MeetingListItem] = []
        decisions: list[ProjectMemoryDecision] = []
        action_items: list[ProjectMemoryAction] = []
        unresolved_issues: list[ProjectMemoryIssue] = []
        for row in rows:
            recent_meetings.append(MeetingListItem.model_validate({
                key: row[key]
                for key in ("id", "title", "project_id", "duration_seconds", "status", "created_at")
            }))
            history = self.get_history(row["id"], team_id)
            if history is None:
                continue
            report = history.report  # 已做互斥/去重 + 说话人归属
            source = MeetingSource(id=row["id"], title=row["title"], created_at=row["created_at"])
            decisions.extend(
                ProjectMemoryDecision(**item.model_dump(), source=source)
                for item in report.decisions
            )
            unresolved_issues.extend(
                ProjectMemoryIssue(**item.model_dump(), source=source)
                for item in report.unresolved_issues
            )
            for item in report.action_items:
                bucket = status_by_task.get((row["id"], item.task))
                action_row = bucket.pop(0) if bucket else None
                if action_row is None:
                    continue
                action_items.append(ProjectMemoryAction(
                    id=action_row["id"], task=item.task, owner=item.owner,
                    deadline=item.deadline, status=action_row["status"], source=source,
                    evidence=item.evidence, speaker=item.speaker,
                ))
        return ProjectMemory(
            project_id=project.id,
            project_name=project.name,
            recent_meetings=recent_meetings,
            decisions=decisions,
            action_items=action_items,
            unresolved_issues=unresolved_issues,
        )

    def get_project_speaker_digest(
        self, project_id: str, team_id: int, meeting_limit: int = 20,
    ) -> Optional[SpeakerDigest]:
        """R-P2.2（阶段 18）：这个项目里，谁说了什么（跨会议按人聚合）。

        只统计已完成、且有报告的会议；同一展示名跨会议自动合并；**跨项目不合并**。
        """
        project = self.get_project(project_id, team_id)
        if project is None:
            return None
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT m.id FROM meetings m JOIN reports r ON r.meeting_id=m.id
                   WHERE m.team_id=? AND m.status='完成'
                     AND (m.project_id=? OR m.project_id IN (
                         SELECT id FROM projects WHERE team_id=? AND parent_id=?
                     ))
                   ORDER BY m.created_at DESC LIMIT ?""",
                (team_id, project_id, team_id, project_id, meeting_limit),
            ).fetchall()
        buckets: Dict[Optional[str], List[SpeakerItem]] = {}
        key_flags: Dict[str, bool] = {}
        for row in rows:
            history = self.get_history(row["id"], team_id)
            if history is None:
                continue
            for section in history.speaker_digest.sections:
                buckets.setdefault(section.speaker, []).extend(section.items)
                if section.is_key_decision_maker and section.speaker:
                    key_flags[section.speaker] = True
        return SpeakerDigest(sections=_sort_sections(buckets, key_flags))

    # ------------------------------------------------------------------
    # R-P2.1-7（阶段 17／M2）：待跟进跨会议跟踪
    # ------------------------------------------------------------------

    def upsert_followups(self, meeting_id: str, team_id: int, report: TeamMeetingReport) -> None:
        """报告写库后，把本场 unresolved_issues 按指纹 upsert 到所属项目。

        只跟踪已归类会议（project_id 为空不跟踪）；跨项目隔离由
        UNIQUE(project_id, fingerprint) + team_id 过滤保证。
        状态只在此处**新建 open**，绝不把 resolved/dropped 悄悄改回 open（红线 23）。
        """
        issues = list(getattr(report, "unresolved_issues", []) or [])
        if not issues:
            return
        with self._lock, self._connect() as connection:
            meeting = connection.execute(
                "SELECT project_id FROM meetings WHERE id=? AND team_id=?",
                (meeting_id, team_id),
            ).fetchone()
            if meeting is None or not meeting["project_id"]:
                return
            project_id = meeting["project_id"]
            segments, label_to_name = self._speaker_context(meeting_id, team_id)
            now = _utc_now()
            for issue in issues:
                fp = fingerprint(str(issue.content))
                speaker = display_speaker(
                    speaker_for_evidence(getattr(issue, "evidence", None), segments),
                    label_to_name,
                )
                existing = connection.execute(
                    "SELECT id FROM followups WHERE team_id=? AND project_id=? AND fingerprint=?",
                    (team_id, project_id, fp),
                ).fetchone()
                if existing is None:
                    cursor = connection.execute(
                        """INSERT INTO followups(
                               team_id,project_id,fingerprint,text,status,
                               first_meeting_id,first_speaker,last_seen_meeting_id,
                               created_at,updated_at
                           ) VALUES(?,?,?,?,'open',?,?,?,?,?)""",
                        (team_id, project_id, fp, str(issue.content),
                         meeting_id, speaker, meeting_id, now, now),
                    )
                    followup_id = cursor.lastrowid
                else:
                    followup_id = int(existing["id"])
                    connection.execute(
                        "UPDATE followups SET last_seen_meeting_id=?, updated_at=? WHERE id=?",
                        (meeting_id, now, followup_id),
                    )
                connection.execute(
                    "INSERT OR IGNORE INTO followup_meetings(followup_id, meeting_id) VALUES(?,?)",
                    (followup_id, meeting_id),
                )

    def list_followups(
        self, project_id: str, team_id: int, status: Optional[FollowupStatus] = None,
    ) -> Optional[FollowupList]:
        """列出项目待跟进（默认全部，可按 status 过滤）。返回 None 表示项目不存在。"""
        project = self.get_project(project_id, team_id)
        if project is None:
            return None
        conditions = ["f.team_id=?", "f.project_id=?"]
        params: List[Any] = [team_id, project_id]
        if status is not None:
            conditions.append("f.status=?")
            params.append(status)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT f.id,f.project_id,f.text,f.status,f.first_speaker,
                          f.first_meeting_id,f.last_seen_meeting_id,f.created_at,f.updated_at,
                          fm.meeting_id AS m_id
                   FROM followups f
                   LEFT JOIN followup_meetings fm ON fm.followup_id=f.id
                   WHERE {} ORDER BY f.updated_at DESC, f.id DESC""".format(" AND ".join(conditions)),
                tuple(params),
            ).fetchall()
            meeting_ids = sorted(
                {row["m_id"] for row in rows if row["m_id"]}
                | {row["first_meeting_id"] for row in rows if row["first_meeting_id"]}
                | {row["last_seen_meeting_id"] for row in rows if row["last_seen_meeting_id"]}
            )
            titles: Dict[str, str] = {}
            if meeting_ids:
                title_rows = connection.execute(
                    "SELECT id,title FROM meetings WHERE id IN ({})".format(
                        ",".join("?" for _ in meeting_ids)
                    ),
                    tuple(meeting_ids),
                ).fetchall()
                titles = {row["id"]: row["title"] for row in title_rows}
        items: Dict[int, FollowupItem] = {}
        order: List[int] = []
        for row in rows:
            fid = int(row["id"])
            if fid not in items:
                items[fid] = FollowupItem(
                    id=fid, project_id=row["project_id"], text=row["text"],
                    status=row["status"], first_speaker=row["first_speaker"],
                    first_meeting_id=row["first_meeting_id"],
                    first_meeting_title=titles.get(row["first_meeting_id"])
                    if row["first_meeting_id"] else None,
                    last_seen_meeting_id=row["last_seen_meeting_id"],
                    last_seen_meeting_title=titles.get(row["last_seen_meeting_id"])
                    if row["last_seen_meeting_id"] else None,
                    meeting_ids=[], meeting_count=0,
                    created_at=row["created_at"], updated_at=row["updated_at"],
                )
                order.append(fid)
            if row["m_id"] and row["m_id"] not in items[fid].meeting_ids:
                items[fid].meeting_ids.append(row["m_id"])
        for fid in order:
            items[fid].meeting_count = len(items[fid].meeting_ids)
        result_items = [items[fid] for fid in order]
        open_count = sum(1 for item in result_items if item.status == "open")
        return FollowupList(
            project_id=project_id, total=len(result_items), open=open_count,
            items=result_items,
        )

    def followup_owner_team_id(self, followup_id: int) -> Optional[int]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT team_id FROM followups WHERE id=?", (followup_id,)
            ).fetchone()
        return int(row["team_id"]) if row else None

    def update_followup_status(
        self, followup_id: int, team_id: int, status: FollowupStatus,
        resolved_meeting_id: Optional[str] = None,
    ) -> bool:
        """人工改状态。resolved_meeting_id 仅在本团队内有效时才会被采信。"""
        with self._lock, self._connect() as connection:
            if resolved_meeting_id:
                owned = connection.execute(
                    "SELECT 1 FROM meetings WHERE id=? AND team_id=?",
                    (resolved_meeting_id, team_id),
                ).fetchone()
                if owned is None:
                    resolved_meeting_id = None
            cursor = connection.execute(
                """UPDATE followups SET status=?, resolved_meeting_id=?, updated_at=?
                   WHERE id=? AND team_id=?""",
                (status, resolved_meeting_id, _utc_now(), followup_id, team_id),
            )
        return cursor.rowcount == 1

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
        """只取报告（不连带转写），供 get_report 工具用。

        R-P2.1：读取时先做确定性互斥/去重，再补 `speaker`（不落库）。
        """
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """SELECT r.json FROM reports r JOIN meetings m ON m.id=r.meeting_id
                   WHERE m.id=? AND m.team_id=?""",
                (meeting_id, team_id),
            ).fetchone()
        if row is None:
            return None
        report = TeamMeetingReport.model_validate(self._report_payload(row["json"]))
        report, _ = dedupe_report(report)
        return self.attribute_meeting_report(meeting_id, team_id, report)

    def _speaker_context(self, meeting_id: str, team_id: int):
        """转写片段 + 本地标签→展示名映射（供读取时说话人归属）。"""
        with self._lock, self._connect() as connection:
            transcript_rows = connection.execute(
                """SELECT t.start,t.end,t.speaker_label,t.text FROM transcripts t
                   JOIN meetings m ON m.id=t.meeting_id
                   WHERE t.meeting_id=? AND m.team_id=? ORDER BY t.id""",
                (meeting_id, team_id),
            ).fetchall()
            speaker_rows = connection.execute(
                """SELECT s.local_label,m.name FROM meeting_speakers s
                   LEFT JOIN members m ON m.id=s.member_id
                   WHERE s.meeting_id=? AND s.team_id=? ORDER BY s.id""",
                (meeting_id, team_id),
            ).fetchall()
        segments = [TranscriptSegment.model_validate(dict(row)) for row in transcript_rows]
        label_to_name = {
            str(row["local_label"]): (row["name"] or str(row["local_label"])) for row in speaker_rows
        }
        return segments, label_to_name

    def attribute_meeting_report(
        self, meeting_id: str, team_id: int, report: TeamMeetingReport,
    ) -> TeamMeetingReport:
        segments, label_to_name = self._speaker_context(meeting_id, team_id)
        return attribute_report(report, segments, label_to_name)

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

"""PRD §9.1：8 个只读工具（默认放行）。

全部只做查询：复用 app/db.py 的既有查询能力，不写任何业务表。
所有工具的第一个参数都是 harness 注入的 team_id（信任边界）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from app.agent.tools.contract import ToolResult, failure, success
from app.agent.tools.registry import ToolRegistry, ToolHandler
from app.agent.tools.support import check_meeting, check_project, guard_size, snippet
from app.db import Database

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 50
MAX_LIMIT = 50
MAX_MEMBERSHIP_LIMIT = 200
ACTION_STATUSES = ("待确认", "进行中", "已完成", "已取消")
_OBJECT = {"type": "object", "additionalProperties": False}


def _schema(properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {**_OBJECT, "properties": properties, "required": required}


def _parse_iso(value: str, field: str) -> Optional[ToolResult]:
    try:
        datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return failure("invalid_args", "{} 不是合法的 ISO8601 时间。".format(field))
    return None


# ---------------------------------------------------------------------------
# 1. get_transcript
# ---------------------------------------------------------------------------


def _get_transcript(
    db: Database, team_id: int, meeting_id: str, page: int = 1, page_size: int = DEFAULT_PAGE_SIZE
) -> ToolResult:
    err = check_meeting(db, meeting_id, team_id)
    if err:
        return err
    try:
        page = max(1, int(page))
        page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))
    except (TypeError, ValueError):
        return failure("invalid_args", "page / page_size 必须是整数。")
    summary = db.transcript_summary(meeting_id, team_id)
    if summary is None:
        return failure("not_found", "这场会议还没有可用的转写。")
    total = summary["total_segments"]
    pages = max(1, (total + page_size - 1) // page_size)
    page = min(page, pages)
    segments = db.transcript_page(meeting_id, team_id, (page - 1) * page_size, page_size) or []
    truncated = page < pages
    payload = {
        "meeting_id": meeting_id,
        "total_segments": total,
        "page": page,
        "pages": pages,
        "page_size": page_size,
        "overview": {
            "time_range": summary["time_range"],
            "speakers": summary["speakers"],
            "segments_per_minute": summary["segments_per_minute"],
        },
        "segments": [
            {
                "start": item["start"],
                "end": item["end"],
                "speaker_label": item["speaker_label"],
                "text": item["text"],
            }
            for item in segments
        ],
    }
    hint = "用 page={} 取下一页，或用 search_transcript 定向检索。".format(page + 1) if truncated else None
    return guard_size(
        success(
            "本场共 {} 段转写，当前第 {}/{} 页（{} 段）。".format(total, page, pages, len(segments)),
            payload,
            truncated=truncated,
            next_hint=hint,
        ),
        next_hint="用 page={} 取下一页。".format(page + 1),
    )


# ---------------------------------------------------------------------------
# 2. search_transcript
# ---------------------------------------------------------------------------


def _search_transcript(
    db: Database, team_id: int, query: str,
    meeting_id: Optional[str] = None, project_id: Optional[str] = None, limit: int = 10,
) -> ToolResult:
    if not query or not str(query).strip():
        return failure("invalid_args", "query 不能为空。")
    if meeting_id is not None:
        err = check_meeting(db, meeting_id, team_id)
        if err:
            return err
    if project_id is not None:
        err = check_project(db, project_id, team_id)
        if err:
            return err
    try:
        limit = max(1, min(int(limit), MAX_LIMIT))
    except (TypeError, ValueError):
        return failure("invalid_args", "limit 必须是整数。")
    rows = db.search_transcripts(
        team_id, str(query).strip(), meeting_id=meeting_id, project_id=project_id, limit=limit
    )
    hits = [
        {
            "meeting_id": row["meeting_id"],
            "meeting_title": row["meeting_title"],
            "start": row["start"],
            "end": row["end"],
            "speaker_label": row["speaker_label"],
            "text_snippet": snippet(row["text"]),
        }
        for row in rows
    ]
    truncated = len(hits) >= limit
    hint = "命中已达上限，请用更具体的关键词或缩小到单场会议。" if truncated else None
    return success(
        "命中 {} 条转写（上限 {}）。".format(len(hits), limit),
        {"hits": hits},
        truncated=truncated,
        next_hint=hint,
    )


# ---------------------------------------------------------------------------
# 3. read_segment_window
# ---------------------------------------------------------------------------


def _read_segment_window(
    db: Database, team_id: int, meeting_id: str,
    timestamp: Optional[float] = None, segment_index: Optional[int] = None,
    before: int = 5, after: int = 5,
) -> ToolResult:
    err = check_meeting(db, meeting_id, team_id)
    if err:
        return err
    if timestamp is None and segment_index is None:
        return failure("invalid_args", "必须提供 timestamp 或 segment_index 之一。")
    try:
        if timestamp is not None:
            timestamp = float(timestamp)
        if segment_index is not None:
            segment_index = int(segment_index)
        before = max(0, min(int(before), 50))
        after = max(0, min(int(after), 50))
    except (TypeError, ValueError):
        return failure("invalid_args", "timestamp / segment_index / before / after 类型不合法。")
    window = db.read_segment_window(
        meeting_id, team_id, timestamp=timestamp, segment_index=segment_index,
        before=before, after=after,
    )
    if window is None:
        return failure("not_found", "这场会议还没有可用的转写。")
    segments = window["segments"]
    if not segments:
        return failure("not_found", "该位置附近没有转写片段。")
    return guard_size(
        success(
            "回读 {} 段转写（{:.1f}s–{:.1f}s）。".format(
                len(segments), window["window"]["start"], window["window"]["end"]
            ),
            {
                "meeting_id": meeting_id,
                "window": window["window"],
                "segments": [
                    {
                        "start": item["start"], "end": item["end"],
                        "speaker_label": item["speaker_label"], "text": item["text"],
                    }
                    for item in segments
                ],
                "total_segments": window["total_segments"],
                "anchor_index": window["anchor_index"],
            },
        ),
        next_hint="缩小 before/after 或改用 search_transcript。",
    )


# ---------------------------------------------------------------------------
# 4. get_report
# ---------------------------------------------------------------------------


def _get_report(db: Database, team_id: int, meeting_id: str) -> ToolResult:
    err = check_meeting(db, meeting_id, team_id)
    if err:
        return err
    report = db.get_report(meeting_id, team_id)
    if report is None:
        return failure("not_found", "这场会议还没有生成报告。")
    return success(
        "报告：决策 {} 条、行动项 {} 条、遗留问题 {} 条。".format(
            len(report.decisions), len(report.action_items), len(report.unresolved_issues)
        ),
        report.model_dump(),
    )


# ---------------------------------------------------------------------------
# 5. get_project_memory
# ---------------------------------------------------------------------------


def _get_project_memory(db: Database, team_id: int, project_id: str, meeting_limit: int = 3) -> ToolResult:
    err = check_project(db, project_id, team_id)
    if err:
        return err
    try:
        meeting_limit = max(1, min(int(meeting_limit), 10))
    except (TypeError, ValueError):
        return failure("invalid_args", "meeting_limit 必须是整数。")
    memory = db.get_project_memory(project_id, team_id, meeting_limit=meeting_limit)
    if memory is None:
        return failure("not_found", "没有找到这个项目的连续回顾。")
    return success(
        "项目回顾：{} 场会议、决策 {} 条、行动项 {} 条、遗留问题 {} 条。".format(
            len(memory.recent_meetings), len(memory.decisions),
            len(memory.action_items), len(memory.unresolved_issues),
        ),
        memory.model_dump(),
    )


# ---------------------------------------------------------------------------
# 6. list_action_items
# ---------------------------------------------------------------------------


def _list_action_items(
    db: Database, team_id: int, project_id: Optional[str] = None, status: Optional[str] = None
) -> ToolResult:
    if project_id is not None:
        err = check_project(db, project_id, team_id)
        if err:
            return err
    if status is not None and status not in ACTION_STATUSES:
        return failure("invalid_args", "status 只能是：{}。".format("、".join(ACTION_STATUSES)))
    rows = db.list_action_items(team_id, project_id=project_id, status=status, limit=MAX_MEMBERSHIP_LIMIT)
    items = [
        {
            "id": row["id"],
            "task": row["task"],
            "owner": row["owner"],
            "deadline": row["deadline"],
            "status": row["status"],
            "source_meeting": {"id": row["meeting_id"], "title": row["meeting_title"]},
        }
        for row in rows
    ]
    return success("共 {} 条行动项。".format(len(items)), {"items": items})


# ---------------------------------------------------------------------------
# 7. list_members
# ---------------------------------------------------------------------------


def _list_members(db: Database, team_id: int) -> ToolResult:
    members = db.list_members(team_id)
    payload = {
        "members": [
            {
                "id": item.id,
                "name": item.name,
                "role": item.role,
                "is_key_decision_maker": item.is_key_decision_maker,
                "has_voiceprint": item.has_voiceprint,
            }
            for item in members
        ]
    }
    return success("团队共 {} 位成员。".format(len(members)), payload)


# ---------------------------------------------------------------------------
# 8. list_meetings
# ---------------------------------------------------------------------------


def _list_meetings(
    db: Database, team_id: int, project_id: Optional[str] = None,
    date_from: Optional[str] = None, date_to: Optional[str] = None, limit: int = 20,
) -> ToolResult:
    if project_id is not None:
        err = check_project(db, project_id, team_id)
        if err:
            return err
    for value, field in ((date_from, "date_from"), (date_to, "date_to")):
        if value is not None:
            err = _parse_iso(value, field)
            if err:
                return err
    try:
        limit = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        return failure("invalid_args", "limit 必须是整数。")
    rows = db.list_meetings_filtered(
        team_id, project_id=project_id, date_from=date_from, date_to=date_to, limit=limit
    )
    meetings = [
        {
            "id": row["id"], "title": row["title"], "project_id": row["project_id"],
            "duration_seconds": row["duration_seconds"], "status": row["status"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]
    truncated = len(meetings) >= limit
    return success(
        "共 {} 场会议（上限 {}）。".format(len(meetings), limit),
        {"meetings": meetings},
        truncated=truncated,
        next_hint="缩小时间范围或提高 limit。" if truncated else None,
    )


# ---------------------------------------------------------------------------
# 注册（新增只读工具 = 加一个函数 + 这里一行）
# ---------------------------------------------------------------------------


def _bind(db: Database, func: Callable[..., ToolResult]) -> ToolHandler:
    def handler(team_id: int, **kwargs: Any) -> ToolResult:
        return func(db, team_id, **kwargs)

    return handler


def register_readonly_tools(registry: ToolRegistry, db: Database) -> None:
    registry.register_tool(
        name="get_transcript",
        description="读取一场会议的转写概览与分页片段（默认只给摘要，全文需检索）。",
        input_schema=_schema(
            {
                "meeting_id": {"type": "string"},
                "page": {"type": "integer", "minimum": 1},
                "page_size": {"type": "integer", "minimum": 1, "maximum": MAX_PAGE_SIZE},
            },
            ["meeting_id"],
        ),
        handler=_bind(db, _get_transcript),
    )
    registry.register_tool(
        name="search_transcript",
        description="按关键词在转写中检索（可限定单场会议或项目），返回命中片段。",
        input_schema=_schema(
            {
                "query": {"type": "string"},
                "meeting_id": {"type": "string"},
                "project_id": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT},
            },
            ["query"],
        ),
        handler=_bind(db, _search_transcript),
    )
    registry.register_tool(
        name="read_segment_window",
        description="回读某时间点/某段前后的转写片段，用于核对引文与负责人。",
        input_schema=_schema(
            {
                "meeting_id": {"type": "string"},
                "timestamp": {"type": "number", "minimum": 0},
                "segment_index": {"type": "integer", "minimum": 0},
                "before": {"type": "integer", "minimum": 0, "maximum": 50},
                "after": {"type": "integer", "minimum": 0, "maximum": 50},
            },
            ["meeting_id"],
        ),
        handler=_bind(db, _read_segment_window),
    )
    registry.register_tool(
        name="get_report",
        description="读取某场会议已生成的团队报告。",
        input_schema=_schema({"meeting_id": {"type": "string"}}, ["meeting_id"]),
        handler=_bind(db, _get_report),
    )
    registry.register_tool(
        name="get_project_memory",
        description="读取某项目最近若干场会议的决策、行动项与遗留问题回顾。",
        input_schema=_schema(
            {"project_id": {"type": "string"}, "meeting_limit": {"type": "integer", "minimum": 1, "maximum": 10}},
            ["project_id"],
        ),
        handler=_bind(db, _get_project_memory),
    )
    registry.register_tool(
        name="list_action_items",
        description="列出行动项，可按项目与状态过滤。",
        input_schema=_schema(
            {"project_id": {"type": "string"}, "status": {"type": "string", "enum": list(ACTION_STATUSES)}},
            [],
        ),
        handler=_bind(db, _list_action_items),
    )
    registry.register_tool(
        name="list_members",
        description="列出本团队成员与是否有声纹（team_id 自动注入）。",
        input_schema=_schema({}, []),
        handler=_bind(db, _list_members),
    )
    registry.register_tool(
        name="list_meetings",
        description="列出会议，可按项目与时间范围过滤。",
        input_schema=_schema(
            {
                "project_id": {"type": "string"},
                "date_from": {"type": "string"},
                "date_to": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            [],
        ),
        handler=_bind(db, _list_meetings),
    )

import json
import logging
import math
import re
import shutil
import sqlite3
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set
from uuid import uuid4

from urllib.parse import quote

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from app.agent.builtin_hooks import build_default_hooks
from app.agent.goal import GoalJudge, build_goal_evaluator
from app.agent.harness import AgentHarness
from app.agent.permissions.policy import PermissionPolicy
from app.agent.planning import Planner
from app.agent.session import AgentLimits
from app.agent.tools import default_registry as agent_tool_registry
from app.agent.tools.catalog import register_all_tools
from app.config import get_settings
from app.db import Database, ProjectHasActiveMeetingsError, ProjectHasChildrenError
from app.deliverables.image_minutes import DEFAULT_TEMPLATE, build_parts, render_html
from app.deliverables.render import RendererUnavailable, render_pdf
from app.llm import (
    AnalysisError, LLMAnalyzer, format_timestamp, usage_scope,
    validate_team_action_evidence, validate_team_evidence,
)
from app.models import (
    ActionItemStatusResult, ActionItemStatusUpdate, AgentTrace, AgentTraceStep,
    DEFAULT_MEETING_TITLE,
    LLMUsageFilters, LLMUsageMeetingSummary,
    LLMUsageReport, LLMUsageStageSummary, LLMUsageTotals,
    MeetingHistory, MeetingListItem,
    MeetingFinalizeRequest, MeetingFinalizeResult, MeetingMoveRequest, MeetingTitleUpdate,
    MemberIdentity, MemberMergeRequest, MemberMergeResult, MemberUpdate,
    ProjectCreate, ProjectDeleteResult,
    ProjectListItem, ProjectMemory, ProjectRename, SpeakerConfirmRequest, SpeakerConfirmResult,
    MyTaskItem, MyTasksResult, SearchHit, SearchResponse, SelfSpeakerResult, SelfSpeakerUpdate,
    ImageMinutesItem, ImageMinutesPart, ImageMinutesResult,
    DeliverableState, DeliverablesReport, RetryResult,
    TermCreate, TermItem, TermsPayload,
    ShareCreate, ShareCreated, ShareLinkItem, ShareLinksPayload,
    SuggestedProjectAction,
    TaskAccepted, TaskStatus, TeamMeetingReport,
)
from app.pipeline import build_team_report
from app.security import AdmissionController, AdmissionError, AdmissionReservation, SHANGHAI_TZ
from app import shares, terms
from app.tasks import InMemoryTaskManager, ProgressCallback, TaskAborted, TaskProcessingError, TaskRecord
from app.transcription import WhisperTranscriber, probe_audio_duration
from app.speaker import KnownVoiceProfile, SpeakerRecognizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = Path(tempfile.gettempdir()) / "meeting-review-uploads"
MIN_FREE_DISK_BYTES = 1024**3
ALLOWED_EXTENSIONS: Set[str] = {".mp3", ".m4a", ".wav"}
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

settings = get_settings()
database = Database(settings.database_path)
transcriber = WhisperTranscriber(settings)


def _record_llm_usage(payload: Dict[str, Any]) -> None:
    """把一次 LLM 调用的用量写入 llm_usage（PRD R-P0-2）。

    成本记账是旁路：只记数值与标识，永不记 prompt / 回复原文；
    写库失败不会中断分析主链路（异常已在 LLMAnalyzer 内部捕获并只记日志）。
    """
    database.record_llm_usage(**payload)


analyzer = LLMAnalyzer(settings, usage_recorder=_record_llm_usage)
# 注册 8 个只读工具 + 3 个能力工具（PRD §9.1/§9.2）。pipeline 模式下不会调用它们，
# 仅用于 /api/agent/tools 调试端点枚举。
register_all_tools(agent_tool_registry, database, analyzer)


def _build_meeting_agent() -> AgentHarness:
    """s15 唯一装配点：工具表 + 权限 + 钩子 + 规划 + 完成判定 + 限额 + 动态提示词。"""
    return AgentHarness(
        registry=agent_tool_registry,
        analyzer=analyzer,
        policy=PermissionPolicy(
            tools_enabled=settings.agent_tools_enabled,
            write_tools_enabled=settings.agent_write_tools_enabled,
        ),
        hooks=build_default_hooks(database, audit_enabled=settings.agent_audit_enabled),
        planner=Planner(),
        goal_judge=GoalJudge(build_goal_evaluator(settings.agent_goal_judge, analyzer)),
        limits=AgentLimits(
            max_steps=settings.agent_max_steps,
            step_timeout_seconds=settings.agent_step_timeout_seconds,
            session_timeout_seconds=settings.agent_session_timeout_seconds,
        ),
        context_budget_tokens=settings.agent_context_budget_tokens,
    )


meeting_agent = _build_meeting_agent()
speaker_recognizer = SpeakerRecognizer(settings)
admission = AdmissionController(settings.rate_limit_per_hour, settings.daily_task_limit)


def _on_task_status(record: TaskRecord) -> None:
    if record.team_id:
        database.update_status(record.task_id, record.team_id, record.status)


def _on_audio_deleted(record: TaskRecord) -> None:
    if record.team_id:
        database.clear_audio_path(record.task_id, record.team_id)


async def _process_audio(path: Path, progress: ProgressCallback) -> TeamMeetingReport:
    team_id = database.owner_team_id(path.stem)
    # 成本归因上下文：让每次 LLM 调用都能按会议 / 项目归属（PRD R-P0-2）。
    usage_context: Optional[Dict[str, Any]] = None
    if team_id:
        meeting_context = database.meeting_context(path.stem, team_id)
        if meeting_context is not None:
            usage_context = {"team_id": team_id, **meeting_context}
    if not progress("转写中", "正在转写，长会议可能需要较长时间…"):
        raise TaskAborted()
    speaker_result = None
    # R-P1.5-9：把团队热词作为转写提示（未配置/关闭时为 None，行为与现状一致）。
    term_prompt = terms.build_team_prompt(
        database, team_id,
        enabled=settings.team_terms_enabled, max_chars=settings.term_prompt_max_chars,
    )
    try:
        # 未配置热词时**只用单参数调用**，与引入热词之前逐字节一致（R-P1.5-9 验收要求）。
        if term_prompt:
            transcript = await run_in_threadpool(transcriber.transcribe, path, term_prompt)
        else:
            transcript = await run_in_threadpool(transcriber.transcribe, path)
        duration = transcript.duration_seconds
        if not math.isfinite(duration) or duration > settings.max_audio_minutes * 60:
            raise TaskProcessingError(
                422, f"当前版本支持 {settings.max_audio_minutes:g} 分钟以内的录音"
            )
        if not transcript.segments:
            raise TaskProcessingError(422, "未检测到可转写的语音")
        if not progress("说话人识别中", "转写完成，正在区分说话人并匹配已记住的声音…"):
            raise TaskAborted()
        profiles = [
            KnownVoiceProfile(**item)
            for item in database.voice_profiles(team_id, settings.speaker_model)
        ] if team_id else []
        speaker_result = await run_in_threadpool(
            speaker_recognizer.process, path, transcript, profiles
        )
    finally:
        path.unlink(missing_ok=True)
        if team_id:
            database.clear_audio_path(path.stem, team_id)

    persisted_transcript = speaker_result.transcript if speaker_result else transcript
    analysis_transcript = speaker_result.analysis_transcript if speaker_result else transcript
    if team_id:
        database.save_transcript(path.stem, team_id, persisted_transcript)
        _mark_deliverable(path.stem, team_id, "transcript", "ok")
        if speaker_result:
            database.save_meeting_speakers(path.stem, team_id, speaker_result.observations)
    if not progress("AI 分析中", "转写完成，正在进行 AI 分析…"):
        raise TaskAborted()
    try:
        with usage_scope(usage_context):
            if settings.agent_mode == "agent" and team_id:
                report = await _run_agent_report(path.stem, team_id)
            else:
                # R-P1.5-8：带上当前团队已有项目，让模型优先复用而不是编造新名字。
                # 团队还没有项目时**不传该参数**，调用与之前完全一致。
                team_projects = database.list_projects(team_id)
                if team_projects:
                    report = await build_team_report(
                        analysis_transcript, analyzer, projects=team_projects,
                    )
                else:
                    report = await build_team_report(analysis_transcript, analyzer)
                if settings.agent_mode == "shadow" and team_id:
                    await _shadow_agent_compare(path.stem, team_id, report)
        if speaker_result:
            report = report.model_copy(update={"speaker_stats_note": speaker_result.message})
        if team_id:
            # D-027：标题仍是默认值时，自动采用 AI 建议标题（用户改过的不覆盖）。
            # legacy ReviewReport 没有 suggested_title，用 getattr 保持兼容。
            database.apply_suggested_title(
                path.stem, team_id, getattr(report, "suggested_title", "")
            )
            report = _sanitize_suggested_project(report, team_id)
            database.save_report(path.stem, team_id, report)
            _mark_deliverable(path.stem, team_id, "report", "ok")
            _mark_deliverable(path.stem, team_id, "tasks", "ok")
        return report
    except AnalysisError as exc:
        if team_id:
            _mark_deliverable(path.stem, team_id, "report", "failed", "analysis_failed")
            _mark_deliverable(path.stem, team_id, "tasks", "failed", "analysis_failed")
        raise TaskProcessingError(502, str(exc)) from exc


async def _run_agent_report(meeting_id: str, team_id: int) -> TeamMeetingReport:
    """AGENT_MODE=agent：模型自主编排产出报告，再由确定性代码校验（写库仍由本函数外完成）。"""
    outcome = await meeting_agent.run_meeting(meeting_id=meeting_id, team_id=team_id)
    if outcome.report is None:
        raise TaskProcessingError(
            502, "Agent 未能在限定范围内产出报告（status={}）".format(outcome.status)
        )
    report = TeamMeetingReport.model_validate(outcome.report)
    transcript = database.load_transcript(meeting_id, team_id)
    if transcript is not None:
        try:
            validate_team_evidence(report, transcript.segments)
            # 行动项也必须能回到原话（否则负责人无法核对）
            validate_team_action_evidence(report, transcript.segments)
        except ValueError as exc:
            raise TaskProcessingError(
                502, "Agent 报告的引文校验未通过：{}".format(str(exc)[:150])
            ) from exc
    return report


async def _shadow_agent_compare(
    meeting_id: str, team_id: int, pipeline_report: TeamMeetingReport
) -> None:
    """AGENT_MODE=shadow：Agent 也跑一遍，但**不落库**，只记录与 pipeline 的关键差异。"""
    try:
        outcome = await meeting_agent.run_meeting(
            meeting_id=meeting_id, team_id=team_id, session_id="shadow-{}".format(meeting_id)
        )
    except Exception as exc:
        logger.warning("shadow_failed meeting_id=%s error_type=%s", meeting_id, type(exc).__name__)
        return
    agent_report: Optional[TeamMeetingReport] = None
    if outcome.report is not None:
        try:
            agent_report = TeamMeetingReport.model_validate(outcome.report)
        except Exception:
            agent_report = None
    logger.info(
        "shadow_compare meeting_id=%s status=%s steps=%s tools=%s "
        "pipeline_decisions=%s agent_decisions=%s pipeline_actions=%s agent_actions=%s",
        meeting_id, outcome.status, outcome.steps, len(outcome.records),
        len(pipeline_report.decisions),
        len(agent_report.decisions) if agent_report else -1,
        len(pipeline_report.action_items),
        len(agent_report.action_items) if agent_report else -1,
    )


task_manager = InMemoryTaskManager(
    _process_audio,
    timeout_seconds=settings.processing_timeout_seconds,
    retention_seconds=settings.task_retention_minutes * 60,
    status_callback=_on_task_status,
    audio_deleted_callback=_on_audio_deleted,
    database=database,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    database.initialize(settings.parsed_team_tokens())
    # R-P1-8：先从库里恢复未完成任务，再清理临时目录（保留待续跑任务的音频）。
    task_manager.hydrate()
    prepare_upload_dir(keep=task_manager.resume_paths())
    await task_manager.start()
    logger.info(
        "startup model=%s api_key=%s database=%s",
        settings.openai_model,
        settings.redacted_api_key_state(),
        settings.database_path,
    )
    yield
    await task_manager.stop()


app = FastAPI(title="会脉 · 团队会议记忆", version="0.3.0", lifespan=lifespan)


class APIError(Exception):
    """新增接口（`/api/usage`、后续 `/api/agent/*`）的统一错误结构。

    老 20 个接口继续返回 `{"detail": ...}`（PRD P-6 契约不动）；
    新接口返回 `{"error": {"code", "message"}}`，与 AGENTS.md 底线 3 一致。
    """

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


@app.exception_handler(APIError)
async def api_error_handler(_: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


@app.middleware("http")
async def request_context_and_gates(request: Request, call_next):
    supplied = request.headers.get("X-Request-ID", "")
    request_id = supplied if REQUEST_ID_PATTERN.fullmatch(supplied) else uuid4().hex
    request.state.request_id = request_id
    logger.info("request_started request_id=%s method=%s path=%s", request_id, request.method, request.url.path)
    if request.url.path.startswith("/api/"):
        # R-P1.5-3：分享读取是**独立的受限出口**，不带团队口令也能读；
        # 但只有 GET /api/shares/... 这一条路径豁免，其它 /api/* 仍必须鉴权。
        if _is_public_share_read(request.method, request.url.path):
            request.state.team_id = None
        else:
            team_id = database.authenticate(request.headers.get("X-Access-Token", ""))
            if team_id is None:
                response = JSONResponse(status_code=403, content={"detail": "团队口令错误"})
                response.headers["X-Request-ID"] = request_id
                return response
            request.state.team_id = team_id

    if request.method == "POST" and request.url.path == "/api/review":
        reservation: Optional[AdmissionReservation] = None
        queue_reserved = False
        committed = False
        try:
            reservation = admission.reserve(_client_ip(request))
            if not task_manager.try_reserve_queue_slot(settings.queue_max):
                raise HTTPException(status_code=429, detail="当前排队人数较多，请稍后再试")
            queue_reserved = True
            _ensure_disk_capacity()
            response = await call_next(request)
            committed = response.status_code == status.HTTP_202_ACCEPTED
        except AdmissionError as exc:
            response = JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})
        except HTTPException as exc:
            response = JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        except OSError as exc:
            logger.warning("disk_check_error request_id=%s error_type=%s", request_id, type(exc).__name__)
            response = JSONResponse(status_code=503, content={"detail": "服务器存储空间检查失败，请稍后再试"})
        finally:
            if queue_reserved:
                task_manager.release_queue_reservation()
            if not committed and reservation is not None:
                admission.rollback(reservation)
    else:
        response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info("request_finished request_id=%s status=%s", request_id, response.status_code)
    return response


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/auth/check")
async def auth_check(request: Request) -> dict:
    return {"status": "ok", "team_id": request.state.team_id}


# R-P1-2 调试端点：枚举当前注册的工具。仅 team 内可用，且默认（pipeline）模式下
# 路由不存在（返回 404），确保不改变现有 20 个接口的行为。在阶段 4（P1-B）
# 注册 11 个工具后，这里会列出它们。
if settings.agent_mode != "pipeline":
    @app.get("/api/agent/tools")
    async def agent_tools() -> dict:
        return {
            "mode": settings.agent_mode,
            "goal_judge": settings.agent_goal_judge,
            "system_prompt": meeting_agent.build_prompt(),
            "tools": [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": spec.input_schema,
                    "permission_level": spec.permission_level,
                }
                for spec in agent_tool_registry.list_tools()
            ],
        }


SHARE_READ_PREFIX = "/api/shares/"


def _is_public_share_read(method: str, path: str) -> bool:
    return method == "GET" and path.startswith(SHARE_READ_PREFIX)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def prepare_upload_dir(upload_dir: Path = UPLOAD_DIR, keep: Optional[Set[Path]] = None) -> None:
    upload_dir.mkdir(parents=True, exist_ok=True)
    protected = {path.resolve() for path in (keep or set())}
    for entry in upload_dir.iterdir():
        if (entry.is_file() or entry.is_symlink()) and entry.resolve() not in protected:
            entry.unlink(missing_ok=True)


def _ensure_disk_capacity(upload_dir: Path = UPLOAD_DIR) -> None:
    if shutil.disk_usage(upload_dir).free < MIN_FREE_DISK_BYTES:
        raise HTTPException(status_code=503, detail="服务器存储空间不足，请稍后再试")


async def _save_upload(upload: UploadFile, path: Path) -> None:
    max_bytes = settings.max_upload_mb * 1024 * 1024
    total = 0
    with path.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(status_code=413, detail=f"文件超过 {settings.max_upload_mb} MB 限制")
            output.write(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="上传文件为空")


@app.post("/api/review", response_model=TaskAccepted, status_code=status.HTTP_202_ACCEPTED)
async def review(
    request: Request, file: UploadFile = File(...), title: str = Form(default=""),
    project_id: str = Form(default=""), consent_confirmed: bool = Form(default=False),
) -> TaskAccepted:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="仅支持 mp3、m4a 和 wav 文件")
    clean_title = title.strip()[:100] or DEFAULT_MEETING_TITLE
    clean_project_id = project_id.strip() or None
    if clean_project_id is not None:
        _assert_team_owns_project(clean_project_id, request.state.team_id)
    submitted = False
    created = False
    temp_path: Optional[Path] = None
    task_id = uuid4().hex
    try:
        temp_path = UPLOAD_DIR / f"{task_id}{suffix}"
        await _save_upload(file, temp_path)
        metadata_duration = await run_in_threadpool(probe_audio_duration, temp_path)
        if metadata_duration is not None and metadata_duration > settings.max_audio_minutes * 60:
            actual_minutes = math.ceil(metadata_duration / 6) / 10
            raise HTTPException(status_code=422, detail=f"当前版本支持 {settings.max_audio_minutes:g} 分钟以内的录音，你的录音约 {actual_minutes:.1f} 分钟")
        long_meeting = metadata_duration is not None and metadata_duration > 30 * 60
        # R-P1.5-6：未勾选同意不得上传（红线 10）。放在格式/时长校验之后、
        # 建库与入队之前：拒绝时不留下会议记录，临时文件会在 finally 里清掉。
        if not consent_confirmed:
            raise HTTPException(
                status_code=422,
                detail="请先勾选「我已知晓：音频将上传至本服务用于本次转写，其中包含他人声音」再上传",
            )
        database.create_meeting(
            task_id, request.state.team_id, clean_title, temp_path, clean_project_id
        )
        created = True
        database.save_consent(
            task_id, request.state.team_id, settings.consent_version,
            _client_ip(request),
        )
        accepted = await task_manager.submit(
            temp_path, request.state.request_id, task_id=task_id,
            team_id=request.state.team_id, long_meeting=long_meeting,
        )
        submitted = True
        return accepted
    except HTTPException:
        raise
    except OSError as exc:
        logger.warning("upload_storage_error request_id=%s error_type=%s", request.state.request_id, type(exc).__name__)
        raise HTTPException(status_code=503, detail="服务器暂时无法保存录音，请稍后再试") from exc
    except Exception as exc:
        logger.error("submit_audio_failed request_id=%s error_type=%s", request.state.request_id, type(exc).__name__)
        raise HTTPException(status_code=500, detail="提交音频失败，请稍后重试") from exc
    finally:
        await file.close()
        if not submitted:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            if created:
                database.delete_meeting(task_id, request.state.team_id)


def _assert_team_owns_meeting(meeting_id: str, team_id: int) -> None:
    owner = database.owner_team_id(meeting_id)
    if owner is not None and owner != team_id:
        raise HTTPException(status_code=403, detail="无权访问其他团队的会议")


def _assert_team_owns_project(project_id: str, team_id: int) -> None:
    owner = database.project_owner_team_id(project_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="项目文件夹不存在")
    if owner != team_id:
        raise HTTPException(status_code=403, detail="无权访问其他团队的项目文件夹")


def _assert_team_owns_action(action_id: int, team_id: int) -> None:
    owner = database.action_owner_team_id(action_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="行动项不存在")
    if owner != team_id:
        raise HTTPException(status_code=403, detail="无权访问其他团队的行动项")


@app.get("/api/projects", response_model=list[ProjectListItem])
async def project_list(request: Request) -> list[ProjectListItem]:
    return database.list_projects(request.state.team_id)


@app.get("/api/projects/{project_id}/memory", response_model=ProjectMemory)
async def project_memory(request: Request, project_id: str) -> ProjectMemory:
    _assert_team_owns_project(project_id, request.state.team_id)
    memory = database.get_project_memory(project_id, request.state.team_id, meeting_limit=3)
    if memory is None:
        raise HTTPException(status_code=404, detail="项目文件夹不存在")
    return memory


def _parse_iso8601(value: Optional[str], field_name: str) -> Optional[str]:
    """校验并归一化时间参数；非法值返回 422 而不是 500。"""
    if value is None or value.strip() == "":
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise APIError(422, "invalid_args", "{} 不是合法的 ISO8601 时间，例如 2026-09-18T00:00:00+08:00".format(field_name)) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat()


_SEARCH_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SEARCH_QUERY_MAX_CHARS = 50
SEARCH_SNIPPET_WIDTH = 30


def _parse_search_date(value: Optional[str], field_name: str, *, end_of_day: bool = False) -> Optional[str]:
    """搜索用的日期参数。

    只给到日（`2026-09-18`）时按 Asia/Shanghai 自然日解释：`from` 取当天 00:00:00，
    `to` 取当天 23:59:59.999999，再统一转 UTC 与 `meetings.created_at` 字符串比较；
    给了完整 ISO8601 时与 `/api/usage` 一致，直接用原值。
    """
    if value is None or value.strip() == "":
        return None
    raw = value.strip()
    if not _SEARCH_DATE_ONLY.fullmatch(raw):
        return _parse_iso8601(raw, field_name)
    try:
        day = datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as exc:
        raise APIError(422, "invalid_args", "{} 不是合法日期，例如 2026-09-18".format(field_name)) from exc
    boundary = datetime.combine(day, time.max if end_of_day else time.min, tzinfo=SHANGHAI_TZ)
    return boundary.astimezone(timezone.utc).isoformat()


def _search_snippet(text: str, query: str, width: int = SEARCH_SNIPPET_WIDTH) -> str:
    """从命中片段文本中截出关键词附近的一段（模型/前端都不写 HTML）。"""
    body = (text or "").strip()
    if not body:
        return ""
    index = body.lower().find(query.lower())
    if index < 0:
        index = 0
    start = max(0, index - width)
    end = min(len(body), index + len(query) + width)
    snippet = body[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(body):
        snippet = snippet + "…"
    return snippet[:200]


@app.get("/api/usage", response_model=LLMUsageReport)
async def llm_usage_report(
    request: Request,
    meeting_id: Optional[str] = Query(default=None),
    project_id: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None, alias="from"),
    date_to: Optional[str] = Query(default=None, alias="to"),
) -> LLMUsageReport:
    """按 stage / 会议聚合 LLM 用量与估算成本，严格限定在本团队内（PRD R-P0-2）。"""
    team_id = request.state.team_id
    if meeting_id is not None:
        if database.meeting_context(meeting_id, team_id) is None:
            owner = database.owner_team_id(meeting_id)
            if owner is None:
                raise APIError(404, "not_found", "会议不存在")
            raise APIError(403, "team_forbidden", "无权访问其他团队的会议")
    if project_id is not None:
        owner = database.project_owner_team_id(project_id)
        if owner is None:
            raise APIError(404, "not_found", "项目文件夹不存在")
        if owner != team_id:
            raise APIError(403, "team_forbidden", "无权访问其他团队的项目文件夹")

    parsed_from = _parse_iso8601(date_from, "from")
    parsed_to = _parse_iso8601(date_to, "to")
    summary = database.usage_summary(
        team_id,
        meeting_id=meeting_id,
        project_id=project_id,
        date_from=parsed_from,
        date_to=parsed_to,
    )

    # 金额由可配置单价换算；未配置单价时返回 None，不在代码或数据库里硬编码价格。
    prompt_price = settings.llm_price_prompt_per_1k
    completion_price = settings.llm_price_completion_per_1k
    price_configured = prompt_price is not None or completion_price is not None

    def estimate_cost(row: Dict[str, Any]) -> Optional[float]:
        if not price_configured:
            return None
        cost = 0.0
        if prompt_price is not None:
            cost += (row.get("prompt_tokens") or 0) / 1000 * prompt_price
        if completion_price is not None:
            cost += (row.get("completion_tokens") or 0) / 1000 * completion_price
        return round(cost, 6)

    return LLMUsageReport(
        filters=LLMUsageFilters(
            meeting_id=meeting_id,
            project_id=project_id,
            date_from=parsed_from,
            date_to=parsed_to,
            price_configured=price_configured,
        ),
        totals=LLMUsageTotals(**summary["totals"], cost=estimate_cost(summary["totals"])),
        by_stage=[
            LLMUsageStageSummary(**row, cost=estimate_cost(row)) for row in summary["by_stage"]
        ],
        by_meeting=[
            LLMUsageMeetingSummary(**row, cost=estimate_cost(row)) for row in summary["by_meeting"]
        ],
    )


@app.get("/api/search", response_model=SearchResponse)
async def search_history(
    request: Request,
    q: str = Query(default=""),
    project_id: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None, alias="from"),
    date_to: Optional[str] = Query(default=None, alias="to"),
    limit: Optional[int] = Query(default=None),
) -> SearchResponse:
    """R-P1.5-4：跳会议搜索转写片段（只读，严格 team_id 隔离）。

    与历史查询一致：不占上传限频与每日额度；FTS5 不可用时自动回退 LIKE。
    """
    team_id = request.state.team_id
    text = (q or "").strip()
    if not text or len(text) > SEARCH_QUERY_MAX_CHARS:
        raise APIError(
            422, "invalid_args",
            "搜索词长度需在 1–{} 字之间".format(SEARCH_QUERY_MAX_CHARS),
        )
    if project_id is not None:
        owner = database.project_owner_team_id(project_id)
        if owner is None:
            raise APIError(404, "not_found", "项目文件夹不存在")
        if owner != team_id:
            raise APIError(403, "team_forbidden", "无权访问其他团队的项目文件夹")

    parsed_from = _parse_search_date(date_from, "from")
    parsed_to = _parse_search_date(date_to, "to", end_of_day=True)
    row_limit = 20 if limit is None else max(1, min(int(limit), 50))
    rows = database.search_meetings(
        team_id, text,
        project_id=project_id, date_from=parsed_from, date_to=parsed_to, limit=row_limit,
    )
    hits = [
        SearchHit(
            meeting_id=row["meeting_id"],
            meeting_title=row["meeting_title"],
            project_id=row.get("project_id"),
            project_name=row.get("project_name"),
            start=float(row["start"]),
            end=float(row["end"]),
            timestamp=format_timestamp(float(row["start"])),
            speaker_label=row.get("speaker_label"),
            text_snippet=_search_snippet(str(row["text"]), text) or str(row["text"])[:200],
        )
        for row in rows
    ]
    return SearchResponse(query=text, count=len(hits), hits=hits)


@app.get("/api/meetings/{meeting_id}/my-tasks", response_model=MyTasksResult)
async def my_tasks(request: Request, meeting_id: str) -> MyTasksResult:
    """R-P1.5-5：「我答应的任务」切面。

    - 只收 `owner` 与「我」（成员姓名或本场说话人标签）**完全一致**的行动项；
    - 未指认时返回空 + `self_speaker_set=false`，**绝不推断**（红线 8）。
    """
    team_id = request.state.team_id
    owner_team = database.owner_team_id(meeting_id)
    if owner_team is None:
        raise APIError(404, "not_found", "会议不存在")
    if owner_team != team_id:
        raise APIError(403, "team_forbidden", "无权访问其他团队的会议")

    items = database.list_meeting_action_items(meeting_id, team_id)
    owner_unknown = sum(
        1 for item in items if not (item.get("owner") or "").strip()
        or (item.get("owner") or "").strip() == "未明确"
    )
    current = database.self_speaker(meeting_id, team_id)
    if current is None:
        return MyTasksResult(
            meeting_id=meeting_id, self_speaker_set=False,
            count=0, owner_unknown=owner_unknown, items=[],
        )
    self_name = current["self_name"]
    mine = [
        MyTaskItem(**item) for item in items
        if self_name and (item.get("owner") or "").strip() == self_name
    ]
    return MyTasksResult(
        meeting_id=meeting_id, self_speaker_set=True,
        local_label=current["local_label"], member_id=current["member_id"],
        self_name=self_name, count=len(mine), owner_unknown=owner_unknown, items=mine,
    )


@app.post("/api/meetings/{meeting_id}/self-speaker", response_model=SelfSpeakerResult)
async def set_self_speaker(
    request: Request, meeting_id: str, payload: SelfSpeakerUpdate = Body(...),
) -> SelfSpeakerResult:
    """R-P1.5-5：指定本场哪个说话人是「我」；两者都为空时清除指认。"""
    team_id = request.state.team_id
    owner_team = database.owner_team_id(meeting_id)
    if owner_team is None:
        raise APIError(404, "not_found", "会议不存在")
    if owner_team != team_id:
        raise APIError(403, "team_forbidden", "无权访问其他团队的会议")

    history = database.get_history(meeting_id, team_id)
    if history is None:
        raise APIError(404, "not_found", "这场会议还没有生成报告")

    if payload.member_id is None and not (payload.local_label or "").strip():
        database.set_self_speaker(meeting_id, team_id)
        return SelfSpeakerResult(self_speaker_set=False)

    if payload.member_id is not None:
        member = next(
            (item for item in database.list_members(team_id) if item.id == payload.member_id), None,
        )
        if member is None:
            raise APIError(422, "invalid_args", "团队成员不存在或不属于本团队")
        speaker = next(
            (item for item in history.speakers if item.member_id == member.id), None,
        )
        if speaker is None:
            raise APIError(
                422, "invalid_args",
                "这位成员不在本场会议里；请在本场说话人里指定。",
            )
        database.set_self_speaker(meeting_id, team_id, member_id=member.id, label=speaker.local_label)
        return SelfSpeakerResult(
            self_speaker_set=True, local_label=speaker.local_label,
            member_id=member.id, member_name=member.name, self_name=member.name,
        )

    label = payload.local_label.strip()
    speaker = next((item for item in history.speakers if item.local_label == label), None)
    if speaker is None:
        raise APIError(422, "invalid_args", "本场没有这个说话人：{}".format(label))
    database.set_self_speaker(meeting_id, team_id, member_id=speaker.member_id, label=label)
    member_name = next(
        (item.name for item in database.list_members(team_id) if item.id == speaker.member_id), None,
    )
    return SelfSpeakerResult(
        self_speaker_set=True, local_label=label, member_id=speaker.member_id,
        member_name=member_name, self_name=member_name or label,
    )


DELIVERABLE_KINDS = ("transcript", "report", "tasks", "image_minutes")
DELIVERABLE_LABELS = {
    "transcript": "逐字稿",
    "report": "文字报告（精简纪要）",
    "tasks": "任务单（行动项）",
    "image_minutes": "图片纪要 + PDF",
}
RENDERER_MISSING_CODE = "renderer_unavailable"


def _mark_deliverable(
    meeting_id: str, team_id: Optional[int], kind: str, status: str,
    error_code: Optional[str] = None,
) -> None:
    """写交付物状态。失败不能影响主链路（状态是辅助信息）。"""
    if not team_id:
        return
    try:
        database.set_deliverable_status(meeting_id, team_id, kind, status, error_code)
    except Exception as exc:  # pragma: no cover - 防御性兑底
        logger.warning(
            "deliverable_status_write_failed kind=%s error_type=%s", kind, type(exc).__name__,
        )


def _deliverable_message(kind: str, status: str, error_code: Optional[str]) -> Optional[str]:
    if status == "ok":
        return None
    if status == "pending":
        return "还没生成或还在处理中。"
    if kind == "transcript":
        return "转写失败；原始录音已按隐私策略删除，请重新上传录音。"
    if error_code == RENDERER_MISSING_CODE:
        return "图片纪要已生成，但 PDF 渲染器不可用（可重试）。"
    if kind == "report":
        return "报告生成失败，可重试（重试会重新调用 AI）。"
    return "这一项失败了，可单独重试，不影响其它交付物。"


def _deliverable_state(kind: str, row: Optional[Dict[str, Any]]) -> DeliverableState:
    status = (row or {}).get("status") or "pending"
    error_code = (row or {}).get("error_code")
    retryable = status in ("failed", "needs_review") and kind != "transcript"
    return DeliverableState(
        kind=kind, label=DELIVERABLE_LABELS[kind], status=status,
        error_code=error_code, message=_deliverable_message(kind, status, error_code),
        retryable=retryable, updated_at=(row or {}).get("updated_at"),
    )


def _deliverables_report(meeting_id: str, team_id: int) -> DeliverablesReport:
    rows = {row["kind"]: row for row in database.deliverable_statuses(meeting_id, team_id)}
    # 旧会议（本次升级前创建的）没有状态行；按实际数据推断一个诚实初值，
    # 避免把已经存在的报告/逐字稿显示成「待生成」。
    fallback = _inferred_statuses(meeting_id, team_id)
    items = [
        _deliverable_state(kind, rows.get(kind) or fallback.get(kind))
        for kind in DELIVERABLE_KINDS
    ]
    return DeliverablesReport(
        meeting_id=meeting_id, items=items,
        needs_review=sum(1 for item in items if item.status in ("failed", "needs_review")),
    )


def _inferred_statuses(meeting_id: str, team_id: int) -> Dict[str, Dict[str, Any]]:
    transcript = database.load_transcript(meeting_id, team_id)
    has_transcript = transcript is not None and bool(transcript.segments)
    has_report = database.get_history(meeting_id, team_id) is not None
    inferred: Dict[str, Dict[str, Any]] = {}
    if has_transcript:
        inferred["transcript"] = {"status": "ok", "error_code": None, "updated_at": None}
    if has_report:
        for kind in ("report", "tasks", "image_minutes"):
            inferred[kind] = {"status": "ok", "error_code": None, "updated_at": None}
    return inferred


def _mark_image_minutes_generated(meeting_id: str, team_id: int) -> None:
    """四板块刚生成。

    注意：**不把已有的「待核对」/「失败」改成正常**——看一次报告不应该假装 PDF 问题已修好；
    只有 PDF 真正渲染成功（或用户点重试且真的渲染成功）才算正常。
    """
    current = {
        row["kind"]: row["status"] for row in database.deliverable_statuses(meeting_id, team_id)
    }
    if current.get("image_minutes") in ("failed", "needs_review"):
        return
    _mark_deliverable(meeting_id, team_id, "image_minutes", "ok")


def _image_minutes_html(history, action_rows, self_name: Optional[str]) -> str:
    return render_html(
        report=history.report, action_rows=action_rows, self_name=self_name,
        title="{}｜图片纪要".format(history.title),
        meta=_image_minutes_meta(history, self_name),
        footer=(
            "由「会脉 · 团队会议记忆」生成 ｜ 完整录音已按隐私策略删除，此处不含音频。\n"
            "图片纪要仅用于速览；结论与引文以会脉报告中的原话证据为准。"
        ),
        template=settings.image_minutes_template or DEFAULT_TEMPLATE,
    )


def _image_minutes_inputs(meeting_id: str, team_id: int):
    """图片纪要所需的只读数据：历史（报告/标题/元信息）、行动项状态、本场「我」。"""
    history = database.get_history(meeting_id, team_id)
    if history is None:
        owner = database.owner_team_id(meeting_id)
        if owner is None:
            raise APIError(404, "not_found", "会议不存在")
        if owner != team_id:
            raise APIError(403, "team_forbidden", "无权访问其他团队的会议")
        raise APIError(404, "not_found", "这场会议还没有生成报告，无法生成图片纪要")
    action_rows = database.list_meeting_action_items(meeting_id, team_id)
    current = database.self_speaker(meeting_id, team_id)
    return history, action_rows, (current or {}).get("self_name")


def _image_minutes_meta(history, self_name: Optional[str]) -> str:
    speakers = [item.display_name for item in history.speakers if item.display_name]
    lines = [
        "会议：{}".format(history.title),
        "时间：{} ｜ 时长：{}".format(history.created_at[:10], format_timestamp(history.duration_seconds)),
    ]
    if speakers:
        lines.append("本场说话人：{}".format("、".join(speakers)))
    lines.append(
        "「我」：{}".format(self_name) if self_name else "「我」：未指定你自己（不推测）"
    )
    return "\n".join(lines)


@app.post("/api/meetings/{meeting_id}/image-minutes", response_model=ImageMinutesResult)
async def image_minutes(request: Request, meeting_id: str) -> ImageMinutesResult:
    """R-P1.5-1：生成图片纪要四板块（幂等，只读渲染，不写任何业务数据）。"""
    team_id = request.state.team_id
    history, action_rows, self_name = _image_minutes_inputs(meeting_id, team_id)
    parts = [
        ImageMinutesPart(
            key=part["key"], title=part["title"], subtitle=part["subtitle"],
            empty_note=part["empty_note"],
            items=[
                ImageMinutesItem(
                    text=item["text"],
                    timestamp=item.get("timestamp"), meta=item.get("meta"),
                )
                for item in part["items"]
            ],
        )
        for part in build_parts(report=history.report, action_rows=action_rows, self_name=self_name)
    ]
    _mark_image_minutes_generated(meeting_id, team_id)
    return ImageMinutesResult(
        meeting_id=meeting_id, title=history.title,
        meta=_image_minutes_meta(history, self_name), parts=parts,
    )


@app.get("/api/meetings/{meeting_id}/image-minutes.pdf")
async def image_minutes_pdf(request: Request, meeting_id: str) -> Response:
    """R-P1.5-1：导出图片纪要 PDF。渲染器不可用时返回 503（不返回 500，不影响其它交付物）。"""
    team_id = request.state.team_id
    history, action_rows, self_name = _image_minutes_inputs(meeting_id, team_id)
    html_text = _image_minutes_html(history, action_rows, self_name)
    try:
        pdf_bytes = await render_pdf(html_text, renderer=settings.pdf_renderer)
    except RendererUnavailable as exc:
        # R-P1.5-7：图片纪要本身已生成，只是 PDF 不可用 → 标「待核对」并给重试入口；
        # 其它交付物（文字报告、逐字稿）完全不受影响。
        _mark_deliverable(meeting_id, team_id, "image_minutes", "needs_review", RENDERER_MISSING_CODE)
        raise APIError(503, RENDERER_MISSING_CODE, str(exc)) from exc
    _mark_deliverable(meeting_id, team_id, "image_minutes", "ok")
    filename = "{}-图片纪要.pdf".format(history.title or "会议")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename*=UTF-8''{}".format(
                quote(filename, safe="")
            ),
            "Cache-Control": "no-store",
        },
    )


def _sanitize_suggested_project(report: TeamMeetingReport, team_id: int) -> TeamMeetingReport:
    """R-P1.5-8：项目建议必须落在本团队已有项目上，否则丢弃建议（不影响报告主字段）。

    只丢弃建议，**绝不自动移动会议**——归属永远由人确认。
    """
    suggestion = getattr(report, "suggested_project", None)
    if suggestion is None:
        return report
    allowed = {item.id for item in database.list_projects(team_id)}
    if suggestion.existing_project_id and suggestion.existing_project_id not in allowed:
        logger.info("suggested_project_dropped reason=unknown_project_id")
        return report.model_copy(update={"suggested_project": None})
    if not suggestion.existing_project_id and not suggestion.new_project_name:
        return report.model_copy(update={"suggested_project": None})
    return report


@app.post("/api/meetings/{meeting_id}/suggested-project", response_model=MeetingListItem)
async def resolve_suggested_project(
    request: Request, meeting_id: str, payload: SuggestedProjectAction = Body(...),
) -> MeetingListItem:
    """R-P1.5-8：采纳 / 改名后加入 / 完全不加入。归属与新建都走人确认。"""
    team_id = request.state.team_id
    owner = database.owner_team_id(meeting_id)
    if owner is None:
        raise APIError(404, "not_found", "会议不存在")
    if owner != team_id:
        raise APIError(403, "team_forbidden", "无权访问其他团队的会议")
    history = database.get_history(meeting_id, team_id)
    if history is None:
        raise APIError(404, "not_found", "这场会议还没有生成报告")

    def clear_suggestion() -> None:
        database.save_report(
            meeting_id, team_id,
            history.report.model_copy(update={"suggested_project": None}),
        )

    if payload.action == "dismiss":
        clear_suggestion()
        meeting = next(
            (item for item in database.list_meetings(team_id) if item.id == meeting_id), None,
        )
        if meeting is None:
            raise APIError(404, "not_found", "会议不存在")
        return meeting

    target_project_id = payload.project_id
    if payload.action == "rename":
        name = str(payload.name or "").strip()
        if not name:
            raise APIError(422, "invalid_args", "请填写新的项目名称")
        created = database.create_project(uuid4().hex, team_id, name[:50])
        target_project_id = created.id
    if not target_project_id:
        raise APIError(422, "invalid_args", "请指定要加入的项目")
    owner_team = database.project_owner_team_id(target_project_id)
    if owner_team is None:
        raise APIError(404, "not_found", "项目不存在")
    if owner_team != team_id:
        raise APIError(403, "team_forbidden", "无权访问其他团队的项目")
    if not database.move_meeting(meeting_id, team_id, target_project_id):
        raise APIError(404, "not_found", "会议不存在")
    clear_suggestion()
    meeting = next(
        (item for item in database.list_meetings(team_id) if item.id == meeting_id), None,
    )
    if meeting is None:
        raise APIError(404, "not_found", "会议不存在")
    return meeting


@app.post("/api/meetings/{meeting_id}/share", response_model=ShareCreated)
async def create_share(
    request: Request, meeting_id: str, payload: ShareCreate = Body(...),
) -> ShareCreated:
    """R-P1.5-3：按勾选生成分享链接（链接 3 天有效，可撤销）。"""
    team_id = request.state.team_id
    owner = database.owner_team_id(meeting_id)
    if owner is None:
        raise APIError(404, "not_found", "会议不存在")
    if owner != team_id:
        raise APIError(403, "team_forbidden", "无权访问其他团队的会议")
    if database.get_history(meeting_id, team_id) is None:
        raise APIError(404, "not_found", "这场会议还没有生成报告，暂时无法分享")
    invalid = shares.invalid_scopes(payload.scopes)
    if invalid:
        raise APIError(422, "invalid_args", "未知的分享项：{}".format("、".join(invalid[:5])))
    scopes = shares.normalize_scopes(payload.scopes)
    if not scopes:
        raise APIError(422, "invalid_args", "请至少勾选一项要分享的内容")

    token = shares.new_token()
    deadline = shares.expires_at(settings.share_ttl_hours)
    supplied = request.headers.get("X-Access-Token", "")
    database.create_share_link(
        shares.token_hash(token), team_id, meeting_id, scopes, deadline,
        created_by_token_hash=shares.token_hash(supplied) if supplied else None,
    )
    base = (settings.share_base_url or "").rstrip("/")
    if not base:
        base = "{}://{}".format(request.url.scheme, request.headers.get("host") or "127.0.0.1:8000")
    return ShareCreated(
        url="{}/s/{}".format(base, token), token=token, scopes=scopes, expires_at=deadline,
    )


@app.get("/api/meetings/{meeting_id}/shares", response_model=ShareLinksPayload)
async def list_shares(request: Request, meeting_id: str) -> ShareLinksPayload:
    """列出本场已生成的分享链接（只回显令牌前缀，不回显原文）。"""
    team_id = request.state.team_id
    owner = database.owner_team_id(meeting_id)
    if owner is None:
        raise APIError(404, "not_found", "会议不存在")
    if owner != team_id:
        raise APIError(403, "team_forbidden", "无权访问其他团队的会议")
    items = []
    for row in database.list_share_links(meeting_id, team_id):
        revoked = row.get("revoked_at") is not None
        items.append(ShareLinkItem(
            share_id=int(row["share_id"]),
            token_prefix=str(row["token_hash"])[:8], scopes=row["scopes"],
            expires_at=row["expires_at"], created_at=row["created_at"],
            revoked_at=row.get("revoked_at"), view_count=int(row.get("view_count") or 0),
            active=not revoked and not shares.is_expired(row),
        ))
    return ShareLinksPayload(items=items)


@app.delete("/api/meetings/{meeting_id}/shares/{token_or_id}")
async def revoke_share(request: Request, meeting_id: str, token_or_id: str) -> dict:
    """撤销分享链接（可传列表里的 share_id，或创建时拿到的令牌）；立即失效。

    数据库只存令牌哈希，因此列表接口不回显原文——界面用 `share_id` 撤销。
    """
    team_id = request.state.team_id
    owner = database.owner_team_id(meeting_id)
    if owner is None:
        raise APIError(404, "not_found", "会议不存在")
    if owner != team_id:
        raise APIError(403, "team_forbidden", "无权访问其他团队的会议")
    if token_or_id.isdigit():
        revoked = database.revoke_share_link_by_id(int(token_or_id), team_id)
    else:
        revoked = database.revoke_share_link(shares.token_hash(token_or_id), team_id)
    if not revoked:
        raise APIError(404, "not_found", "分享链接不存在或已撤销")
    return {"status": "ok"}


@app.get("/api/shares/{token}")
async def read_share(request: Request, token: str) -> JSONResponse:
    """分享读取：**无需团队口令**；只返回被勾选的内容；过期/撤销返回 410。"""
    digest = shares.token_hash(token)
    record = database.share_link(digest)
    ip = _client_ip(request)
    if record is None:
        return JSONResponse(status_code=404, content={
            "error": {"code": "not_found", "message": "分享链接不存在"}})
    team_id, meeting_id = int(record["team_id"]), str(record["meeting_id"])
    if record.get("revoked_at") is not None:
        database.record_share_access(digest, team_id, meeting_id, "revoked", ip)
        return JSONResponse(status_code=410, content={
            "error": {"code": "share_revoked", "message": "这条分享链接已被作者撤销"}})
    if shares.is_expired(record):
        database.record_share_access(digest, team_id, meeting_id, "expired", ip)
        return JSONResponse(status_code=410, content={
            "error": {"code": "share_expired", "message": "这条分享链接已超过 3 天有效期"}})
    history = database.get_history(meeting_id, team_id)
    if history is None:
        return JSONResponse(status_code=404, content={
            "error": {"code": "not_found", "message": "会议内容已不可用"}})

    image_payload = None
    if shares.SCOPE_IMAGE_MINUTES in (record.get("scopes") or []):
        action_rows = database.list_meeting_action_items(meeting_id, team_id)
        current = database.self_speaker(meeting_id, team_id)
        image_payload = {
            "parts": [
                {
                    "key": part["key"], "title": part["title"], "subtitle": part["subtitle"],
                    "empty_note": part["empty_note"],
                    "items": [
                        {"text": item["text"], "timestamp": item.get("timestamp"),
                         "meta": item.get("meta")}
                        for item in part["items"]
                    ],
                }
                for part in build_parts(
                    report=history.report, action_rows=action_rows,
                    self_name=(current or {}).get("self_name"),
                )
            ],
        }
    clips = []
    if shares.SCOPE_VOICE in (record.get("scopes") or []):
        for speaker in history.speakers:
            for clip in speaker.clips:
                clips.append({
                    "id": clip.id, "speaker_label": speaker.display_name,
                    "start": clip.start, "end": clip.end,
                    "timestamp": format_timestamp(clip.start), "text": clip.text,
                })
    payload = shares.build_payload(
        record=record, history=history, image_minutes=image_payload, my_tasks=None, clips=clips,
    )
    database.count_share_view(digest)
    database.record_share_access(digest, team_id, meeting_id, "ok", ip)
    return JSONResponse(
        content=payload,
        headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
    )


@app.get("/api/shares/{token}/clips/{clip_id}")
async def read_share_clip(request: Request, token: str, clip_id: int) -> Response:
    """分享范围内的代表语音片段；**必须该链接勾选了「代表语音」**，且片段属于被分享的会议。"""
    digest = shares.token_hash(token)
    record = database.share_link(digest)
    if record is None or record.get("revoked_at") is not None or shares.is_expired(record):
        raise APIError(404, "not_found", "分享链接不可用")
    if shares.SCOPE_VOICE not in (record.get("scopes") or []):
        raise APIError(403, "team_forbidden", "这条分享链接没有包含语音片段")
    team_id, meeting_id = int(record["team_id"]), str(record["meeting_id"])
    history = database.get_history(meeting_id, team_id)
    allowed = {
        clip.id
        for speaker in (history.speakers if history else [])
        for clip in speaker.clips
    }
    if clip_id not in allowed:
        raise APIError(404, "not_found", "语音片段不存在")
    result = database.speaker_clip(clip_id, team_id)
    if result is None:
        raise APIError(404, "not_found", "语音片段不存在")
    mime_type, audio = result
    return Response(content=audio, media_type=mime_type,
                    headers={"Cache-Control": "private, max-age=600"})


@app.get("/s/{token}", include_in_schema=False)
async def share_page(token: str) -> HTMLResponse:
    """最小只读分享页；令牌本身在接口里校验。"""
    return HTMLResponse(shares.render_page())


@app.get("/api/terms", response_model=TermsPayload)
async def list_terms(request: Request) -> TermsPayload:
    """R-P1.5-9：当前团队热词表（手动 + 已确认成员姓名）。"""
    team_id = request.state.team_id
    items = [
        TermItem(
            id=item["id"], term=item["term"], note=item["note"] or "",
            source=item["source"], updated_at=item["updated_at"],
        )
        for item in terms.collect_terms(database, team_id)
    ]
    return TermsPayload(items=items, prompt=terms.build_prompt(
        [item.term for item in items], settings.term_prompt_max_chars,
    ))


@app.post("/api/terms", response_model=TermItem)
async def create_term(request: Request, payload: TermCreate = Body(...)) -> TermItem:
    """新增或更新一个热词（只影响转写质量；不改引文校验规则）。"""
    team_id = request.state.team_id
    term = terms.normalize_term(payload.term)
    if not term:
        raise APIError(422, "invalid_args", "热词不能为空")
    if len(term) > terms.MAX_TERM_CHARS:
        raise APIError(422, "invalid_args", "热词最长 {} 个字".format(terms.MAX_TERM_CHARS))
    term_id = database.upsert_term(team_id, term, str(payload.note or "").strip())
    row = next(
        (item for item in database.list_terms(team_id) if item["id"] == term_id), None,
    )
    return TermItem(
        id=term_id, term=term, note=(row or {}).get("note") or "",
        source="manual", updated_at=(row or {}).get("updated_at"),
    )


@app.delete("/api/terms/{term_id}")
async def delete_term(request: Request, term_id: int) -> dict:
    if not database.delete_term(request.state.team_id, term_id):
        raise APIError(404, "not_found", "热词不存在")
    return {"status": "ok"}


@app.get("/api/meetings/{meeting_id}/deliverables", response_model=DeliverablesReport)
async def meeting_deliverables(request: Request, meeting_id: str) -> DeliverablesReport:
    """R-P1.5-7：四类交付物各自的状态（含「待核对」与可重试标记）。"""
    team_id = request.state.team_id
    owner = database.owner_team_id(meeting_id)
    if owner is None:
        raise APIError(404, "not_found", "会议不存在")
    if owner != team_id:
        raise APIError(403, "team_forbidden", "无权访问其他团队的会议")
    return _deliverables_report(meeting_id, team_id)


@app.post("/api/meetings/{meeting_id}/retry", response_model=RetryResult)
async def retry_deliverable(
    request: Request, meeting_id: str, kind: str = Query(...),
) -> RetryResult:
    """R-P1.5-7：按交付物重试。**不重跑已成功的部分**。"""
    team_id = request.state.team_id
    owner = database.owner_team_id(meeting_id)
    if owner is None:
        raise APIError(404, "not_found", "会议不存在")
    if owner != team_id:
        raise APIError(403, "team_forbidden", "无权访问其他团队的会议")
    if kind not in DELIVERABLE_KINDS:
        raise APIError(422, "invalid_args", "未知交付物：{}".format(kind))

    if kind == "transcript":
        raise APIError(
            409, "not_retryable",
            "转写需要原始录音，而录音已按隐私策略删除；请重新上传录音。",
        )

    if kind == "tasks":
        history = database.get_history(meeting_id, team_id)
        if history is None:
            raise APIError(404, "not_found", "还没有可用的报告，请先重试报告。")
        database.save_report(meeting_id, team_id, history.report)
        _mark_deliverable(meeting_id, team_id, "tasks", "ok")
        return RetryResult(
            meeting_id=meeting_id, kind=kind, status="ok",
            message="已按现有报告重新生成任务单（未重新调用 AI）。",
        )

    if kind == "image_minutes":
        history, action_rows, self_name = _image_minutes_inputs(meeting_id, team_id)
        html_text = _image_minutes_html(history, action_rows, self_name)
        try:
            # 真重试：实际渲染一次 PDF；失败则保持「待核对」，不假装修好。
            await render_pdf(html_text, renderer=settings.pdf_renderer)
        except RendererUnavailable as exc:
            _mark_deliverable(
                meeting_id, team_id, "image_minutes", "needs_review", RENDERER_MISSING_CODE,
            )
            raise APIError(503, RENDERER_MISSING_CODE, str(exc)) from exc
        _mark_deliverable(meeting_id, team_id, "image_minutes", "ok")
        return RetryResult(
            meeting_id=meeting_id, kind=kind, status="ok",
            message="已重新生成图片纪要，并可导出 PDF（未重新调用 AI）。",
        )

    # kind == "report"：唯一会重新调用 AI 的重试
    transcript = database.load_transcript(meeting_id, team_id)
    if transcript is None or not transcript.segments:
        raise APIError(404, "not_found", "没有可用的逐字稿，请先重新上传录音。")
    meeting_context = database.meeting_context(meeting_id, team_id) or {}
    try:
        with usage_scope({"team_id": team_id, **meeting_context}):
            report = await build_team_report(transcript, analyzer)
    except AnalysisError as exc:
        _mark_deliverable(meeting_id, team_id, "report", "failed", "analysis_failed")
        raise APIError(502, "analysis_failed", "重新生成报告失败：{}".format(str(exc)[:150])) from exc
    except Exception as exc:
        _mark_deliverable(meeting_id, team_id, "report", "failed", "analysis_failed")
        raise APIError(502, "analysis_failed", "重新生成报告失败（{}）".format(type(exc).__name__)) from exc
    try:
        validate_team_evidence(report, transcript.segments)
        validate_team_action_evidence(report, transcript.segments)
    except ValueError as exc:
        _mark_deliverable(meeting_id, team_id, "report", "failed", "evidence_failed")
        raise APIError(502, "evidence_failed", "重新生成的报告引文校验未通过") from exc
    database.save_report(meeting_id, team_id, report)
    _mark_deliverable(meeting_id, team_id, "report", "ok")
    _mark_deliverable(meeting_id, team_id, "tasks", "ok")
    return RetryResult(
        meeting_id=meeting_id, kind=kind, status="ok",
        message="已重新生成报告（本次重新调用了 AI）。",
    )


@app.get("/api/meetings/{meeting_id}/agent-trace", response_model=AgentTrace)
async def meeting_agent_trace(request: Request, meeting_id: str) -> AgentTrace:
    """只读展示：这场会议里 Agent 每一步调用了什么工具、判定与耗时（PRD §14.1 M1 验收入口）。

    仅按 team_id 隔离读取 agent_audit；无任何写操作。默认（pipeline）下通常为空。
    """
    _assert_team_owns_meeting(meeting_id, request.state.team_id)
    rows = database.list_agent_audit_for_meeting(meeting_id, request.state.team_id)
    plan: list = []
    task = database.get_agent_task(meeting_id, request.state.team_id)
    if task and task.get("progress_json"):
        try:
            plan = json.loads(task["progress_json"]).get("todo", []) or []
        except (ValueError, TypeError):
            plan = []
    return AgentTrace(
        meeting_id=meeting_id,
        mode=settings.agent_mode,
        audit_enabled=settings.agent_audit_enabled,
        total_calls=len(rows),
        tool_names=sorted({row["tool_name"] for row in rows if row["tool_name"]}),
        steps=[AgentTraceStep(**row) for row in rows],
        plan=[{"task": str(item.get("task", "")), "status": str(item.get("status", "pending"))} for item in plan],
    )


@app.patch("/api/action-items/{action_id}", response_model=ActionItemStatusResult)
async def action_status_update(
    request: Request, action_id: int, payload: ActionItemStatusUpdate = Body(...)
) -> ActionItemStatusResult:
    _assert_team_owns_action(action_id, request.state.team_id)
    if not database.update_action_status(action_id, request.state.team_id, payload.status):
        raise HTTPException(status_code=404, detail="行动项不存在")
    return ActionItemStatusResult(id=action_id, status=payload.status)


@app.post("/api/projects", response_model=ProjectListItem, status_code=status.HTTP_201_CREATED)
async def project_create(request: Request, payload: ProjectCreate = Body(...)) -> ProjectListItem:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="项目文件夹名称不能为空")
    if payload.parent_id is not None:
        _assert_team_owns_project(payload.parent_id, request.state.team_id)
        parent = database.get_project(payload.parent_id, request.state.team_id)
        if parent is None:
            raise HTTPException(status_code=404, detail="上级项目文件夹不存在")
        if parent.parent_id is not None:
            raise HTTPException(status_code=422, detail="项目文件夹最多支持两级")
    try:
        return database.create_project(
            uuid4().hex, request.state.team_id, name, parent_id=payload.parent_id
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="同名项目文件夹已经存在") from exc


@app.patch("/api/projects/{project_id}", response_model=ProjectListItem)
async def project_rename(
    request: Request, project_id: str, payload: ProjectRename = Body(...)
) -> ProjectListItem:
    _assert_team_owns_project(project_id, request.state.team_id)
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="项目文件夹名称不能为空")
    try:
        updated = database.rename_project(project_id, request.state.team_id, name)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="同名项目文件夹已经存在") from exc
    if updated is None:
        raise HTTPException(status_code=404, detail="项目文件夹不存在")
    return updated


@app.delete("/api/projects/{project_id}", response_model=ProjectDeleteResult)
async def project_delete(
    request: Request, project_id: str, delete_meetings: bool = Query(default=False)
) -> ProjectDeleteResult:
    _assert_team_owns_project(project_id, request.state.team_id)
    try:
        affected = database.delete_project(
            project_id, request.state.team_id, delete_meetings=delete_meetings
        )
    except ProjectHasChildrenError as exc:
        raise HTTPException(
            status_code=409,
            detail="该文件夹下还有二级文件夹，请先处理二级文件夹",
        ) from exc
    except ProjectHasActiveMeetingsError as exc:
        raise HTTPException(
            status_code=409,
            detail="文件夹中有正在处理的会议，请处理完成后再选择连同会议删除",
        ) from exc
    return ProjectDeleteResult(
        project_id=project_id,
        affected_meetings=affected,
        meetings_deleted=delete_meetings,
    )


@app.get("/api/tasks/{task_id}", response_model=TaskStatus)
async def task_status(request: Request, task_id: str) -> TaskStatus:
    _assert_team_owns_meeting(task_id, request.state.team_id)
    task = task_manager.get(task_id, request.state.team_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    # R-P1.5-7：按交付物聚合状态（新增字段；老字段与语义不变）。
    try:
        state = _deliverables_report(task_id, request.state.team_id)
        return task.model_copy(update={"deliverables": state.items})
    except Exception as exc:  # pragma: no cover - 状态是辅助信息，不能影响任务查询
        logger.warning("task_deliverables_failed error_type=%s", type(exc).__name__)
        return task


@app.get("/api/meetings", response_model=list[MeetingListItem])
async def meeting_list(
    request: Request, project_id: Optional[str] = Query(default=None),
    unclassified: bool = Query(default=False),
    include_children: bool = Query(default=False),
) -> list[MeetingListItem]:
    if project_id is not None:
        _assert_team_owns_project(project_id, request.state.team_id)
    return database.list_meetings(
        request.state.team_id, project_id, unclassified, include_children
    )


@app.get("/api/meetings/{meeting_id}", response_model=MeetingHistory)
async def meeting_history(request: Request, meeting_id: str) -> MeetingHistory:
    _assert_team_owns_meeting(meeting_id, request.state.team_id)
    history = database.get_history(meeting_id, request.state.team_id)
    if history is None:
        raise HTTPException(status_code=404, detail="历史报告不存在或尚未生成")
    return history


@app.patch("/api/meetings/{meeting_id}/title", response_model=MeetingHistory)
async def meeting_title_update(
    request: Request, meeting_id: str, payload: MeetingTitleUpdate = Body(...),
) -> MeetingHistory:
    _assert_team_owns_meeting(meeting_id, request.state.team_id)
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="会议标题不能为空")
    if not database.update_meeting_title(meeting_id, request.state.team_id, title):
        raise HTTPException(status_code=404, detail="会议不存在")
    history = database.get_history(meeting_id, request.state.team_id)
    if history is None:
        raise HTTPException(status_code=404, detail="历史报告不存在或尚未生成")
    return history


@app.get("/api/members", response_model=list[MemberIdentity])
async def member_list(request: Request) -> list[MemberIdentity]:
    return database.list_members(request.state.team_id)


@app.patch("/api/members/{member_id}", response_model=MemberIdentity)
async def member_update(
    request: Request, member_id: int, payload: MemberUpdate = Body(...),
) -> MemberIdentity:
    owner = database.member_owner_team_id(member_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="成员不存在")
    if owner != request.state.team_id:
        raise HTTPException(status_code=403, detail="无权修改其他团队的成员")
    updated = database.update_member(
        member_id, request.state.team_id, payload.name.strip(), payload.role.strip(),
        payload.is_key_decision_maker,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="成员不存在")
    return updated


@app.delete("/api/members/{member_id}/voiceprint")
async def voiceprint_delete(request: Request, member_id: int) -> dict:
    owner = database.member_owner_team_id(member_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="成员不存在")
    if owner != request.state.team_id:
        raise HTTPException(status_code=403, detail="无权删除其他团队的声纹")
    if not database.delete_voiceprint(member_id, request.state.team_id):
        raise HTTPException(status_code=404, detail="该成员尚未保存声纹")
    return {"status": "ok"}


@app.post("/api/members/{member_id}/merge", response_model=MemberMergeResult)
async def member_merge(
    request: Request, member_id: int, payload: MemberMergeRequest = Body(...),
) -> MemberMergeResult:
    for candidate in (member_id, payload.target_member_id):
        owner = database.member_owner_team_id(candidate)
        if owner is None:
            raise HTTPException(status_code=404, detail="成员不存在")
        if owner != request.state.team_id:
            raise HTTPException(status_code=403, detail="无权合并其他团队的身份")
    try:
        affected = database.merge_members(
            member_id, payload.target_member_id, request.state.team_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="不能将身份合并到自己") from exc
    if affected is None:
        raise HTTPException(status_code=404, detail="成员不存在")
    return MemberMergeResult(
        target_member_id=payload.target_member_id, merged_meetings=affected,
    )


@app.post(
    "/api/meetings/{meeting_id}/speakers/{local_label}/confirm",
    response_model=SpeakerConfirmResult,
)
async def speaker_confirm(
    request: Request, meeting_id: str, local_label: str,
    payload: SpeakerConfirmRequest = Body(...),
) -> SpeakerConfirmResult:
    _assert_team_owns_meeting(meeting_id, request.state.team_id)
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="说话人名称不能为空")
    try:
        result = database.confirm_meeting_speaker(
            meeting_id, request.state.team_id, local_label, name, payload.role.strip(),
            payload.is_key_decision_maker, payload.remember_voice, settings.speaker_model,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail="本场没有可用的声音样本，只能保存本场名称",
        ) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="待确认说话人不存在")
    member_id, saved, affected = result
    return SpeakerConfirmResult(
        member_id=member_id, display_name=name, voiceprint_saved=saved,
        reanalysis_recommended=affected,
    )


@app.post(
    "/api/meetings/{meeting_id}/finalize", response_model=MeetingFinalizeResult,
)
async def meeting_finalize(
    request: Request, meeting_id: str,
    payload: MeetingFinalizeRequest = Body(...),
) -> MeetingFinalizeResult:
    _assert_team_owns_meeting(meeting_id, request.state.team_id)
    history = database.get_history(meeting_id, request.state.team_id)
    pending = [item.display_name for item in history.speakers if item.remember_requested] if history else []
    has_clips = any(item.clips for item in history.speakers) if history else False
    consent_already_recorded = history.speaker_consent_confirmed if history else False
    if (pending or has_clips) and not consent_already_recorded and not payload.consent_confirmed:
        raise HTTPException(
            status_code=422,
            detail="保存代表性声音或声纹前，请统一确认已取得本人同意",
        )
    try:
        members = database.finalize_meeting_voiceprints(
            meeting_id, request.state.team_id, settings.speaker_model,
            payload.consent_confirmed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="存在无法保存的声音样本") from exc
    if members is None:
        raise HTTPException(status_code=404, detail="会议不存在")
    return MeetingFinalizeResult(voiceprints_saved=len(members), members=members)


@app.get("/api/speaker-clips/{clip_id}")
async def speaker_clip(request: Request, clip_id: int) -> Response:
    owner = database.speaker_clip_owner_team_id(clip_id)
    if owner is not None and owner != request.state.team_id:
        raise HTTPException(status_code=403, detail="无权访问其他团队的声音片段")
    result = database.speaker_clip(clip_id, request.state.team_id)
    if result is None:
        raise HTTPException(status_code=404, detail="声音片段不存在")
    mime_type, audio = result
    return Response(
        content=audio, media_type=mime_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.patch("/api/meetings/{meeting_id}/project", response_model=MeetingListItem)
async def meeting_move(
    request: Request, meeting_id: str, payload: MeetingMoveRequest = Body(...)
) -> MeetingListItem:
    _assert_team_owns_meeting(meeting_id, request.state.team_id)
    _assert_team_owns_project(payload.project_id, request.state.team_id)
    current = next(
        (
            item
            for item in database.list_meetings(request.state.team_id)
            if item.id == meeting_id
        ),
        None,
    )
    if current is None:
        raise HTTPException(status_code=404, detail="会议不存在")
    if current.project_id == payload.project_id:
        raise HTTPException(status_code=409, detail="会议已经在该文件夹中")
    if not database.move_meeting(
        meeting_id, request.state.team_id, payload.project_id
    ):
        raise HTTPException(status_code=404, detail="会议不存在")
    return next(
        item
        for item in database.list_meetings(request.state.team_id)
        if item.id == meeting_id
    )

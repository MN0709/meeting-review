import json
import logging
import math
import re
import shutil
import sqlite3
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set
from uuid import uuid4

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response

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
from app.llm import AnalysisError, LLMAnalyzer, usage_scope, validate_team_evidence
from app.models import (
    ActionItemStatusResult, ActionItemStatusUpdate, AgentTrace, AgentTraceStep,
    LLMUsageFilters, LLMUsageMeetingSummary,
    LLMUsageReport, LLMUsageStageSummary, LLMUsageTotals,
    MeetingHistory, MeetingListItem,
    MeetingFinalizeRequest, MeetingFinalizeResult, MeetingMoveRequest, MeetingTitleUpdate,
    MemberIdentity, MemberMergeRequest, MemberMergeResult, MemberUpdate,
    ProjectCreate, ProjectDeleteResult,
    ProjectListItem, ProjectMemory, ProjectRename, SpeakerConfirmRequest, SpeakerConfirmResult,
    TaskAccepted, TaskStatus, TeamMeetingReport,
)
from app.pipeline import build_team_report
from app.security import AdmissionController, AdmissionError, AdmissionReservation
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
    try:
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
        if speaker_result:
            database.save_meeting_speakers(path.stem, team_id, speaker_result.observations)
    if not progress("AI 分析中", "转写完成，正在进行 AI 分析…"):
        raise TaskAborted()
    try:
        with usage_scope(usage_context):
            if settings.agent_mode == "agent" and team_id:
                report = await _run_agent_report(path.stem, team_id)
            else:
                report = await build_team_report(analysis_transcript, analyzer)
                if settings.agent_mode == "shadow" and team_id:
                    await _shadow_agent_compare(path.stem, team_id, report)
        if speaker_result:
            report = report.model_copy(update={"speaker_stats_note": speaker_result.message})
        if team_id:
            database.save_report(path.stem, team_id, report)
        return report
    except AnalysisError as exc:
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
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    database.initialize(settings.parsed_team_tokens())
    prepare_upload_dir()
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


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def prepare_upload_dir(upload_dir: Path = UPLOAD_DIR) -> None:
    upload_dir.mkdir(parents=True, exist_ok=True)
    for entry in upload_dir.iterdir():
        if entry.is_file() or entry.is_symlink():
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
    project_id: str = Form(default=""),
) -> TaskAccepted:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="仅支持 mp3、m4a 和 wav 文件")
    clean_title = title.strip()[:100] or "未命名会议"
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
        database.create_meeting(
            task_id, request.state.team_id, clean_title, temp_path, clean_project_id
        )
        created = True
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

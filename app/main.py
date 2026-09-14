import logging
import math
import re
import shutil
import sqlite3
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Set
from uuid import uuid4

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from app.config import get_settings
from app.db import Database, ProjectHasActiveMeetingsError, ProjectHasChildrenError
from app.llm import AnalysisError, LLMAnalyzer
from app.models import (
    ActionItemStatusResult, ActionItemStatusUpdate, MeetingHistory, MeetingListItem,
    MeetingMoveRequest, MemberIdentity, MemberMergeRequest, MemberMergeResult, MemberUpdate,
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
analyzer = LLMAnalyzer(settings)
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
        report = await build_team_report(analysis_transcript, analyzer)
        if speaker_result and speaker_result.available:
            report = report.model_copy(update={"speaker_stats_note": speaker_result.message})
        if team_id:
            database.save_report(path.stem, team_id, report)
        return report
    except AnalysisError as exc:
        raise TaskProcessingError(502, str(exc)) from exc


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
    yield
    await task_manager.stop()


app = FastAPI(title="会脉 · 团队会议记忆", version="0.3.0", lifespan=lifespan)


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
    if payload.remember_voice and not payload.consent_confirmed:
        raise HTTPException(
            status_code=422,
            detail="记住他人声音前，请确认已取得该参会者同意",
        )
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
    if payload.remember_voice and not saved:
        raise HTTPException(status_code=422, detail="本场没有可用的声音样本，只能保存本场名称")
    return SpeakerConfirmResult(
        member_id=member_id, display_name=name, voiceprint_saved=saved,
        reanalysis_recommended=affected,
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

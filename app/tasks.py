import asyncio
import contextlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Dict, Optional, Set
from uuid import uuid4

from app.models import TaskAccepted, TaskStage, TaskStatus


logger = logging.getLogger(__name__)
TERMINAL_STAGES = {"完成", "失败"}
ProgressCallback = Callable[[TaskStage, str], bool]
TaskProcessor = Callable[[Path, ProgressCallback], Awaitable[Any]]
RecordCallback = Callable[["TaskRecord"], None]


class TaskProcessingError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class TaskAborted(RuntimeError):
    """Raised when a timed-out task reaches its next cancellable boundary."""


@dataclass
class TaskRecord:
    task_id: str
    request_id: str
    audio_path: Path
    status: TaskStage = "上传完成"
    message: str = "上传完成，正在进入处理队列…"
    team_id: int = 0
    long_meeting: bool = False
    report: Optional[Any] = None
    error: Optional[str] = None
    error_code: Optional[int] = None
    updated_at: float = field(default_factory=time.monotonic)
    finished_at: Optional[float] = None
    history: list[TaskStage] = field(default_factory=lambda: ["上传完成"])


class TaskManager:
    """FIFO single-worker queue，状态同时写入 SQLite（R-P1-8）。

    - 传入 `database` 时：每次状态流转落库、`GET /api/tasks/{id}` 在内存过期后读库、
      启动时用 `hydrate()` 恢复未完成任务（音频还在就继续跑，不在就明确失败）。
    - 不传 `database` 时：行为与纯内存队列完全一致（便于单测）。
    - One process must run exactly one Uvicorn worker.
    """

    def __init__(
        self,
        processor: TaskProcessor,
        *,
        timeout_seconds: float,
        retention_seconds: float,
        cleanup_interval_seconds: float = 60,
        status_callback: Optional[RecordCallback] = None,
        audio_deleted_callback: Optional[RecordCallback] = None,
        database: Optional[Any] = None,
    ) -> None:
        self.processor = processor
        self.timeout_seconds = timeout_seconds
        self.retention_seconds = retention_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.status_callback = status_callback
        self.audio_deleted_callback = audio_deleted_callback
        self.database = database
        self.records: Dict[str, TaskRecord] = {}
        self._queue: Deque[str] = deque()
        self._resume_queue: Deque[str] = deque()
        self._queue_event: Optional[asyncio.Event] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._processing_task: Optional[asyncio.Task] = None
        self._current_task_id: Optional[str] = None
        self._queue_reservations = 0

    # ------------------------------------------------------------------
    # R-P1-8：持久化与重启恢复
    # ------------------------------------------------------------------

    def hydrate(self) -> None:
        """启动时从库里读回未完成任务；音频还在就排队续跑，不在就明确失败。"""
        if self.database is None:
            return
        try:
            rows = self.database.list_active_agent_tasks()
        except Exception as exc:
            logger.warning("task_hydrate_failed error_type=%s", type(exc).__name__)
            return
        for row in rows:
            task_id = row["task_id"]
            audio = Path(row["audio_path"]) if row.get("audio_path") else None
            record = TaskRecord(
                task_id=task_id,
                request_id=row.get("request_id") or "",
                audio_path=audio or Path("/nonexistent"),
                team_id=int(row["team_id"]) if row.get("team_id") is not None else 0,
                long_meeting=bool(row.get("long_meeting")),
            )
            record.status = row.get("status") or "排队中"
            record.message = row.get("message") or record.status
            self.records[task_id] = record
            if audio is not None and audio.exists():
                self._resume_queue.append(task_id)
                logger.info("task_resume_queued task_id=%s status=%s", task_id, record.status)
            else:
                record.error = "服务重启，且临时音频已清理，请重新提交"
                record.error_code = 503
                self._transition(record, "失败", record.error)

    def resume_paths(self) -> Set[Path]:
        """需要保留的临时音频（供启动清理时跳过）。"""
        return {
            record.audio_path
            for record in self.records.values()
            if record.status not in TERMINAL_STAGES and record.audio_path.exists()
        }

    def _persist(self, record: TaskRecord) -> None:
        if self.database is None or not record.team_id:
            return
        finished_iso = (
            datetime.now(timezone.utc).isoformat() if record.status in TERMINAL_STAGES else None
        )
        try:
            self.database.upsert_agent_task(
                record.task_id,
                record.team_id,
                record.status,
                meeting_id=record.task_id,
                message=record.message,
                error=record.error,
                error_code=record.error_code,
                finished_at=finished_iso,
                request_id=record.request_id or None,
                long_meeting=record.long_meeting,
                audio_path=str(record.audio_path) if record.audio_path else None,
            )
        except Exception as exc:  # 落库失败不能影响处理主链路
            logger.warning(
                "task_persist_failed task_id=%s error_type=%s", record.task_id, type(exc).__name__
            )

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._worker_task and not self._worker_task.done():
            return
        while self._resume_queue:
            task_id = self._resume_queue.popleft()
            if task_id not in self._queue:
                self._queue.append(task_id)
        self._queue_event = asyncio.Event()
        if self._queue:
            self._queue_event.set()
        self._worker_task = asyncio.create_task(self._worker_loop(), name="meeting-review-worker")
        self._cleanup_task = asyncio.create_task(self._cleanup_loop(), name="meeting-review-cleanup")

    async def stop(self) -> None:
        tasks = [task for task in (self._worker_task, self._cleanup_task) if task]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._processing_task
        self._worker_task = None
        self._cleanup_task = None
        self._processing_task = None
        self._current_task_id = None
        self._queue_reservations = 0
        while self._queue:
            task_id = self._queue.popleft()
            record = self.records.get(task_id)
            if record:
                record.audio_path.unlink(missing_ok=True)
                if self.audio_deleted_callback:
                    self.audio_deleted_callback(record)
                record.error = "服务正在关闭，请重新提交"
                record.error_code = 503
                self._transition(record, "失败", record.error)

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    async def submit(
        self,
        audio_path: Path,
        request_id: str,
        *,
        task_id: Optional[str] = None,
        team_id: int = 0,
        long_meeting: bool = False,
    ) -> TaskAccepted:
        await self.start()
        task_id = task_id or uuid4().hex
        record = TaskRecord(
            task_id=task_id, request_id=request_id, audio_path=audio_path,
            team_id=team_id, long_meeting=long_meeting,
        )
        self.records[task_id] = record
        self._queue.append(task_id)
        self._transition(record, "排队中", "排队中")
        assert self._queue_event is not None
        self._queue_event.set()
        return TaskAccepted(
            task_id=task_id,
            status=record.status,
            queue_position=self._queue_position(task_id),
            message=self._queue_message(task_id),
            long_meeting=long_meeting,
        )

    def try_reserve_queue_slot(self, queue_max: int) -> bool:
        if len(self._queue) + self._queue_reservations >= queue_max:
            return False
        self._queue_reservations += 1
        return True

    def release_queue_reservation(self) -> None:
        if self._queue_reservations > 0:
            self._queue_reservations -= 1

    def get(self, task_id: str, team_id: Optional[int] = None) -> Optional[TaskStatus]:
        record = self.records.get(task_id)
        if record is not None:
            if team_id is not None and record.team_id != team_id:
                return None
            return self._status_from_record(record)
        # 内存已过期（例如重启后）：回落到数据库
        if self.database is not None and team_id is not None:
            row = self.database.get_agent_task(task_id, team_id)
            if row is not None:
                return self._status_from_row(row)
        return None

    def _status_from_record(self, record: TaskRecord) -> TaskStatus:
        queue_position = self._queue_position(record.task_id)
        message = self._queue_message(record.task_id) if record.status == "排队中" else record.message
        return TaskStatus(
            task_id=record.task_id,
            request_id=record.request_id,
            status=record.status,
            queue_position=queue_position,
            message=message,
            report=record.report,
            error=record.error,
            error_code=record.error_code,
            long_meeting=record.long_meeting,
        )

    def _status_from_row(self, row: Dict[str, Any]) -> TaskStatus:
        report = None
        if row.get("status") in TERMINAL_STAGES and self.database is not None:
            try:
                report = self.database.get_report(row["task_id"], int(row["team_id"]))
            except Exception:
                report = None
        return TaskStatus(
            task_id=row["task_id"],
            request_id=row.get("request_id") or "",
            status=row["status"],
            queue_position=0,
            message=row.get("message") or row["status"],
            report=report,
            error=row.get("error"),
            error_code=row.get("error_code"),
            long_meeting=bool(row.get("long_meeting")),
        )

    def cleanup_expired(self, now: Optional[float] = None) -> int:
        current = time.monotonic() if now is None else now
        expired = [
            task_id
            for task_id, record in self.records.items()
            if record.status in TERMINAL_STAGES
            and record.finished_at is not None
            and current - record.finished_at >= self.retention_seconds
        ]
        for task_id in expired:
            self.records.pop(task_id, None)
        return len(expired)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _queue_position(self, task_id: str) -> int:
        record = self.records.get(task_id)
        if record is None or record.status != "排队中":
            return 0
        ahead = 1 if self._current_task_id and self._current_task_id != task_id else 0
        try:
            return ahead + list(self._queue).index(task_id)
        except ValueError:
            return ahead

    def _queue_message(self, task_id: str) -> str:
        return f"排队中，前面还有 {self._queue_position(task_id)} 人"

    def _transition(self, record: TaskRecord, status: TaskStage, message: str) -> bool:
        if record.status in TERMINAL_STAGES:
            return False
        record.status = status
        record.message = message
        record.updated_at = time.monotonic()
        if not record.history or record.history[-1] != status:
            record.history.append(status)
        if status in TERMINAL_STAGES:
            record.finished_at = record.updated_at
        logger.info(
            "task_status request_id=%s task_id=%s status=%s",
            record.request_id,
            record.task_id,
            status,
        )
        self._persist(record)
        if self.status_callback:
            self.status_callback(record)
        return True

    async def _worker_loop(self) -> None:
        assert self._queue_event is not None
        while True:
            if not self._queue:
                self._queue_event.clear()
                await self._queue_event.wait()
                continue
            task_id = self._queue.popleft()
            record = self.records.get(task_id)
            if record is None:
                continue
            self._current_task_id = task_id
            await self._run_record(record)
            self._current_task_id = None

    async def _run_record(self, record: TaskRecord) -> None:
        def progress(status: TaskStage, message: str) -> bool:
            return self._transition(record, status, message)

        processing_task = asyncio.create_task(self.processor(record.audio_path, progress))
        self._processing_task = processing_task
        try:
            report = await asyncio.wait_for(asyncio.shield(processing_task), self.timeout_seconds)
            record.report = report
            self._transition(record, "完成", "复盘完成")
        except asyncio.TimeoutError:
            if self.timeout_seconds >= 60 and self.timeout_seconds % 60 == 0:
                limit = f"{self.timeout_seconds / 60:g} 分钟"
            else:
                limit = f"{self.timeout_seconds:g} 秒"
            record.error = f"处理超过 {limit}，请稍后重试"
            record.error_code = 504
            self._transition(record, "失败", record.error)
            record.audio_path.unlink(missing_ok=True)
            # A cancelled to_thread keeps running. Hold the only worker slot until the
            # underlying operation actually returns, so timed-out work never overlaps.
            with contextlib.suppress(TaskAborted, TaskProcessingError, Exception):
                await processing_task
        except TaskProcessingError as exc:
            record.error = str(exc)
            record.error_code = exc.status_code
            self._transition(record, "失败", record.error)
        except TaskAborted:
            if record.status not in TERMINAL_STAGES:
                record.error = "处理已终止，请重试"
                record.error_code = 500
                self._transition(record, "失败", record.error)
        except Exception as exc:
            logger.error(
                "task_failed request_id=%s task_id=%s error_type=%s",
                record.request_id,
                record.task_id,
                type(exc).__name__,
            )
            record.error = "处理音频失败，请稍后重试"
            record.error_code = 500
            self._transition(record, "失败", record.error)
        finally:
            if not processing_task.done():
                processing_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await processing_task
            record.audio_path.unlink(missing_ok=True)
            if self.audio_deleted_callback:
                self.audio_deleted_callback(record)
            if self._processing_task is processing_task:
                self._processing_task = None

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cleanup_interval_seconds)
            removed = self.cleanup_expired()
            if removed:
                logger.info("expired_tasks_cleaned count=%s", removed)


# 兼容旧名字（既有测试与调用方使用）
InMemoryTaskManager = TaskManager

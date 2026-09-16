import asyncio
import copy
import json
import os
import sqlite3
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import av
from fastapi.testclient import TestClient
from fastapi import HTTPException
from pydantic import ValidationError

os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

import app.main as main_module
from app.config import Settings
from app.llm import LLMAnalyzer, normalize_llm_payload, split_segments, validate_evidence
from app.db import Database
from app.models import ChunkSummary, ReviewReport, SemanticAnalysis, TaskAccepted, TeamMeetingReport, Transcript, TranscriptSegment
from app.pipeline import build_report
from app.security import AdmissionController, AdmissionError, SHANGHAI_TZ
from app.speaker import (
    KnownVoiceProfile, SpeakerObservation, SpeakerRecognitionResult, SpeakerRecognizer,
    best_profile_match,
)
from app.stats import compute_speech_stats
from app.tasks import InMemoryTaskManager
from app.transcription import WhisperTranscriber, probe_audio_duration


FIXTURES = Path(__file__).parent / "fixtures"
AUTH_HEADERS = {"X-Access-Token": "test-access-token"}


@pytest.fixture(autouse=True)
def reset_admission_controller(monkeypatch, tmp_path):
    controller = AdmissionController(
        main_module.settings.rate_limit_per_hour,
        main_module.settings.daily_task_limit,
    )
    monkeypatch.setattr(main_module, "admission", controller)
    test_database = Database(tmp_path / "meeting-review.db")
    test_database.initialize(main_module.settings.parsed_team_tokens())
    monkeypatch.setattr(main_module, "database", test_database)
    # Unit tests use deterministic speaker doubles where recognition matters;
    # never download or load the real model during unrelated endpoint tests.
    monkeypatch.setattr(main_module.speaker_recognizer, "enabled", False)
    main_module.task_manager.records.clear()
    main_module.task_manager._queue.clear()


def load_json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def team_report() -> TeamMeetingReport:
    return TeamMeetingReport(
        overview="会议确认本周上线内测，并明确了发布准备工作的负责人和截止时间。",
        meeting_points=["确定本周上线内测"],
        decisions=[{
            "content": "本周上线内测",
            "decision_maker": "小宁",
            "evidence": {"quote": "我们本周上线内测", "timestamp": "00:00:02"},
        }],
        action_items=[{"task": "准备发布清单", "owner": "小宁", "deadline": "周五"}],
        unresolved_issues=[],
    )


def run_semantic_model_output(payload: dict) -> SemanticAnalysis:
    class StaticOutputAnalyzer(LLMAnalyzer):
        def __init__(self) -> None:
            super().__init__(Settings(OPENAI_API_KEY="test"))

        def _request_sync(self, *args, **kwargs) -> str:
            return json.dumps(payload, ensure_ascii=False)

    transcript = Transcript.model_validate(load_json("mock_transcript.json"))
    result = asyncio.run(
        StaticOutputAnalyzer()._validated_call(
            SemanticAnalysis,
            "system",
            "user",
            "test_normalize",
            transcript.segments,
        )
    )
    return SemanticAnalysis.model_validate(result.model_dump())


class StaticAnalyzer:
    def __init__(self, result: SemanticAnalysis) -> None:
        self.result = result

    async def analyze(self, transcript: Transcript) -> SemanticAnalysis:
        return self.result


def test_health_returns_ok() -> None:
    response = TestClient(main_module.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"]


def test_team_tokens_are_required_and_health_is_exempt(monkeypatch) -> None:
    monkeypatch.delenv("TEAM_TOKENS", raising=False)
    with pytest.raises(ValidationError, match="TEAM_TOKENS"):
        Settings(_env_file=None)

    disk_checked = False

    def unexpected_disk_check():
        nonlocal disk_checked
        disk_checked = True

    monkeypatch.setattr(main_module, "_ensure_disk_capacity", unexpected_disk_check)
    with TestClient(main_module.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/tasks/unknown").status_code == 403
        denied_upload = client.post("/api/review", files={"file": ("private.wav", b"secret", "audio/wav")})
        assert denied_upload.status_code == 403
        assert disk_checked is False
        assert client.get("/api/auth/check", headers={"X-Access-Token": "wrong"}).status_code == 403
        accepted = client.get("/api/auth/check", headers=AUTH_HEADERS)
        assert accepted.status_code == 200
        assert accepted.json()["status"] == "ok"
        assert accepted.json()["team_id"] > 0


def test_team_tokens_are_upserted_on_startup_and_old_token_stops_working(tmp_path) -> None:
    database = Database(tmp_path / "teams.db")
    database.initialize({"甲团队": "old-token"})
    old_id = database.authenticate("old-token")
    assert old_id is not None

    database.initialize({"甲团队": "new-token", "乙团队": "beta-token"})
    assert database.authenticate("old-token") is None
    assert database.authenticate("new-token") == old_id
    assert database.authenticate("beta-token") is not None


@pytest.mark.parametrize("value", ["", "没有分隔符", "甲:a,甲:b", "甲:a,乙:a"])
def test_invalid_team_token_configuration_is_rejected(value) -> None:
    if not value:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, TEAM_TOKENS=value)
    else:
        with pytest.raises(ValueError, match="TEAM_TOKENS"):
            Settings(_env_file=None, TEAM_TOKENS=value).parsed_team_tokens()


def test_service_import_fails_with_clear_message_without_team_tokens(tmp_path) -> None:
    project_dir = Path(__file__).parent.parent
    environment = os.environ.copy()
    environment.pop("TEAM_TOKENS", None)
    environment["PYTHONPATH"] = str(project_dir)

    result = subprocess.run(
        [sys.executable, "-c", "import app.main"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "缺少必填环境变量 TEAM_TOKENS，服务拒绝启动" in result.stderr


def test_hourly_rate_limit_and_rollback() -> None:
    now = datetime(2026, 9, 11, 12, tzinfo=SHANGHAI_TZ)
    controller = AdmissionController(1, 10, clock=lambda: now)
    reservation = controller.reserve("203.0.113.7")
    with pytest.raises(AdmissionError, match="操作过于频繁"):
        controller.reserve("203.0.113.7")

    controller.rollback(reservation)
    controller.reserve("203.0.113.7")


def test_daily_limit_resets_by_asia_shanghai_date() -> None:
    current = [datetime(2026, 9, 11, 15, 59, tzinfo=timezone.utc)]
    controller = AdmissionController(10, 1, clock=lambda: current[0])
    controller.reserve("203.0.113.8")
    with pytest.raises(AdmissionError, match="今日体验名额已用完，明天再来"):
        controller.reserve("203.0.113.9")

    current[0] = datetime(2026, 9, 11, 16, 1, tzinfo=timezone.utc)
    controller.reserve("203.0.113.9")


def test_polling_and_health_do_not_consume_review_rate_limit(monkeypatch) -> None:
    calls = []

    class SpyAdmission:
        def reserve(self, client_ip):
            calls.append(client_ip)
            return SimpleNamespace()

        def rollback(self, reservation):
            pass

    monkeypatch.setattr(main_module, "admission", SpyAdmission())
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/tasks/not-found").status_code == 404
        assert calls == []

        response = client.post("/api/review", files={"file": ("invalid.txt", b"x", "text/plain")})
        assert response.status_code == 415
        assert len(calls) == 1


def test_team_long_meeting_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.max_upload_mb == 300
    assert settings.max_audio_minutes == 240
    assert settings.processing_timeout_seconds == 21600
    assert settings.rate_limit_per_hour == 10
    assert settings.daily_task_limit == 30
    assert settings.queue_max == 5
    assert settings.speaker_recognition_enabled is True
    assert settings.speaker_model == "chinese"
    assert settings.speaker_match_threshold == 0.72
    assert settings.speaker_match_margin == 0.05


def test_duration_probe_reads_metadata_without_decoding(monkeypatch, tmp_path) -> None:
    class FakeContainer:
        duration = 12 * 60 * av.time_base
        streams = SimpleNamespace(audio=[])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def decode(self, *args, **kwargs):
            raise AssertionError("duration probe must not decode audio")

    monkeypatch.setattr(av, "open", lambda *args, **kwargs: FakeContainer())

    assert probe_audio_duration(tmp_path / "metadata.wav") == 720


def test_duration_probe_failure_returns_none_for_transcription_fallback(monkeypatch, tmp_path) -> None:
    def broken_open(*args, **kwargs):
        raise ValueError("damaged container")

    monkeypatch.setattr(av, "open", broken_open)

    assert probe_audio_duration(tmp_path / "damaged.wav") is None


def test_multi_hour_transcription_uses_bounded_chunks_and_preserves_timestamps(
    monkeypatch, tmp_path,
) -> None:
    transcriber = WhisperTranscriber(Settings(_env_file=None))
    calls = []

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            calls.append(audio)
            index = len(calls)
            return iter([
                SimpleNamespace(start=0.2, end=0.8, text=f" 第{index}段 "),
            ]), SimpleNamespace(language="zh")

    monkeypatch.setattr(
        "app.transcription.probe_audio_duration", lambda path: 2 * 60 * 60,
    )
    monkeypatch.setattr(transcriber, "_get_model", lambda: FakeModel())
    monkeypatch.setattr(
        transcriber,
        "_decode_audio_chunks",
        lambda path: iter([[0.0] * 16000, [0.0] * 32000]),
    )

    transcript = transcriber.transcribe(tmp_path / "two-hours.m4a")

    assert len(calls) == 2
    assert transcript.duration_seconds == 2 * 60 * 60
    assert [(item.start, item.end, item.text) for item in transcript.segments] == [
        (0.2, 0.8, "第1段"),
        (1.2, 1.8, "第2段"),
    ]


def test_preflight_rejects_long_audio_before_it_enters_queue(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "probe_audio_duration", lambda path: 242 * 60)
    submitted = False

    async def unexpected_submit(*args, **kwargs):
        nonlocal submitted
        submitted = True
        raise AssertionError("over-limit audio must not enter queue")

    monkeypatch.setattr(main_module.task_manager, "submit", unexpected_submit)

    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post("/api/review", files={"file": ("long.wav", b"fake", "audio/wav")})

    assert response.status_code == 422
    assert response.json()["detail"] == "当前版本支持 240 分钟以内的录音，你的录音约 242.0 分钟"
    assert submitted is False


def test_queue_full_rejects_before_upload_is_saved(monkeypatch) -> None:
    saved = False
    monkeypatch.setattr(main_module.task_manager, "try_reserve_queue_slot", lambda limit: False)

    async def unexpected_save(*args, **kwargs):
        nonlocal saved
        saved = True

    monkeypatch.setattr(main_module, "_save_upload", unexpected_save)

    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post("/api/review", files={"file": ("queued.wav", b"fake", "audio/wav")})

    assert response.status_code == 429
    assert response.json()["detail"] == "当前排队人数较多，请稍后再试"
    assert saved is False


@pytest.mark.parametrize(
    ("message", "status_code"),
    [("操作过于频繁，请稍后再试", 429), ("今日体验名额已用完，明天再来", 429)],
)
def test_quota_rejections_happen_before_upload_save(monkeypatch, message, status_code) -> None:
    saved = False

    class RejectAdmission:
        def reserve(self, client_ip):
            raise AdmissionError(status_code, message)

        def rollback(self, reservation):
            raise AssertionError("a rejected reservation must not be rolled back")

    async def unexpected_save(*args, **kwargs):
        nonlocal saved
        saved = True

    monkeypatch.setattr(main_module, "admission", RejectAdmission())
    monkeypatch.setattr(main_module, "_save_upload", unexpected_save)

    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post("/api/review", files={"file": ("quota.wav", b"fake", "audio/wav")})

    assert response.status_code == status_code
    assert response.json()["detail"] == message
    assert saved is False


def test_low_disk_rejects_before_upload_is_saved(monkeypatch) -> None:
    saved = False

    def reject_disk():
        raise HTTPException(status_code=503, detail="服务器存储空间不足，请稍后再试")

    async def unexpected_save(*args, **kwargs):
        nonlocal saved
        saved = True

    monkeypatch.setattr(main_module, "_ensure_disk_capacity", reject_disk)
    monkeypatch.setattr(main_module, "_save_upload", unexpected_save)

    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post("/api/review", files={"file": ("disk.wav", b"fake", "audio/wav")})

    assert response.status_code == 503
    assert response.json()["detail"] == "服务器存储空间不足，请稍后再试"
    assert saved is False


def test_disk_capacity_requires_at_least_one_gibibyte(monkeypatch, tmp_path) -> None:
    usage = SimpleNamespace(total=2 * 1024**3, used=1024**3 + 1, free=1024**3 - 1)
    monkeypatch.setattr(main_module.shutil, "disk_usage", lambda path: usage)

    with pytest.raises(HTTPException, match="服务器存储空间不足") as exc:
        main_module._ensure_disk_capacity(tmp_path)

    assert exc.value.status_code == 503


def test_client_ip_uses_uvicorn_sanitized_peer_not_raw_forwarded_header() -> None:
    request = SimpleNamespace(
        client=SimpleNamespace(host="198.51.100.20"),
        headers={"X-Forwarded-For": "203.0.113.99"},
    )
    assert main_module._client_ip(request) == "198.51.100.20"

    run_source = (Path(__file__).parent.parent / "run.py").read_text(encoding="utf-8")
    assert "proxy_headers=True" in run_source
    assert "forwarded_allow_ips=settings.forwarded_allow_ips" in run_source


def test_upload_write_error_returns_503_and_leaves_no_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(main_module, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(main_module, "_ensure_disk_capacity", lambda: None)

    async def broken_save(upload, path):
        path.write_bytes(b"partial")
        raise OSError("disk write failed")

    monkeypatch.setattr(main_module, "_save_upload", broken_save)

    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post("/api/review", files={"file": ("write.wav", b"fake", "audio/wav")})

    assert response.status_code == 503
    assert response.json()["detail"] == "服务器暂时无法保存录音，请稍后再试"
    assert list(tmp_path.glob("*.wav")) == []


def test_startup_cleanup_removes_stale_upload_files(tmp_path) -> None:
    stale = tmp_path / "stale.wav"
    stale.write_bytes(b"private audio")

    main_module.prepare_upload_dir(tmp_path)

    assert not stale.exists()


def test_preflight_rounds_just_over_limit_up_in_error_message(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "probe_audio_duration", lambda path: 14400.1)

    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post("/api/review", files={"file": ("long.wav", b"fake", "audio/wav")})

    assert response.status_code == 422
    assert "你的录音约 240.1 分钟" in response.json()["detail"]


@pytest.mark.parametrize(
    ("metadata_duration", "expected_long_meeting"),
    [(5 * 60, False), (4 * 60 * 60, True), (None, False)],
)
def test_preflight_allows_duration_and_randomizes_server_filename(
    monkeypatch, metadata_duration, expected_long_meeting,
) -> None:
    monkeypatch.setattr(main_module, "probe_audio_duration", lambda path: metadata_duration)
    submitted_paths = []

    async def fake_submit(path, request_id, *, task_id=None, team_id=0, long_meeting=False):
        assert path.exists()
        assert path.stem == task_id
        assert "normal" not in path.name
        assert long_meeting is expected_long_meeting
        submitted_paths.append(path)
        path.unlink()
        return TaskAccepted(task_id=task_id, status="排队中", queue_position=0, message="排队中")

    monkeypatch.setattr(main_module.task_manager, "submit", fake_submit)

    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        response = client.post("/api/review", files={"file": ("normal.wav", b"fake", "audio/wav")})

    assert response.status_code == 202
    assert len(response.json()["task_id"]) == 32
    assert len(submitted_paths) == 1


def test_request_id_is_echoed_and_invalid_value_is_replaced() -> None:
    client = TestClient(main_module.app)
    supplied = client.get("/health", headers={"X-Request-ID": "deploy-check-123"})
    assert supplied.headers["X-Request-ID"] == "deploy-check-123"

    invalid = client.get("/missing", headers={"X-Request-ID": "x" * 129})
    assert invalid.status_code == 404
    assert invalid.headers["X-Request-ID"] != "x" * 129
    assert len(invalid.headers["X-Request-ID"]) == 32


def test_fixture_builds_expected_report_without_audio_or_llm() -> None:
    transcript = Transcript.model_validate(load_json("mock_transcript.json"))
    expected = load_json("expected_report.json")
    semantic = SemanticAnalysis.model_validate({key: value for key, value in expected.items() if key != "stats"})
    actual = asyncio.run(build_report(transcript, StaticAnalyzer(semantic)))
    assert actual.model_dump() == expected


def test_overlapping_segments_are_merged_and_boundaries_do_not_form_fillers() -> None:
    transcript = Transcript(
        duration_seconds=10,
        segments=[
            TranscriptSegment(start=0, end=4, text="嗯嗯然"),
            TranscriptSegment(start=3, end=5, text="后呃"),
        ],
    )
    stats = compute_speech_stats(transcript)
    assert stats.transcribed_speech_seconds == 5
    assert stats.filler_counts == {"然后": 0, "就是": 0, "嗯": 2, "呃": 1}


def test_evidence_must_be_exact_original_quote() -> None:
    expected = load_json("expected_report.json")
    semantic = SemanticAnalysis.model_validate({key: value for key, value in expected.items() if key != "stats"})
    transcript = Transcript.model_validate(load_json("mock_transcript.json"))
    validate_evidence(semantic, transcript.segments)
    semantic.performance_analysis[0].evidence[0].quote = "这句话不在原文"
    with pytest.raises(ValueError):
        validate_evidence(semantic, transcript.segments)


def test_evidence_timestamp_must_match_its_segment_with_five_second_tolerance() -> None:
    expected = load_json("expected_report.json")
    transcript = Transcript.model_validate(load_json("mock_transcript.json"))
    semantic = SemanticAnalysis.model_validate({key: value for key, value in expected.items() if key != "stats"})

    semantic.performance_analysis[0].evidence[0].timestamp = "00:00:07"
    validate_evidence(semantic, transcript.segments)

    semantic.performance_analysis[0].evidence[0].timestamp = "00:00:30"
    with pytest.raises(ValueError, match="时间戳"):
        validate_evidence(semantic, transcript.segments)


def test_long_segment_is_split_without_losing_text() -> None:
    original = TranscriptSegment(start=0, end=10, text="甲" * 120)
    chunks = split_segments([original], max_chars=50)
    assert len(chunks) > 1
    assert "".join(segment.text for chunk in chunks for segment in chunk) == original.text


def test_invalid_json_is_retried_then_validated() -> None:
    expected = load_json("expected_report.json")
    valid = json.dumps({key: value for key, value in expected.items() if key != "stats"}, ensure_ascii=False)

    class SequencedAnalyzer(LLMAnalyzer):
        def __init__(self) -> None:
            super().__init__(Settings(OPENAI_API_KEY="test", LLM_MAX_RETRIES=2))
            self.outputs = iter(["not-json", valid])
            self.calls = 0

        def _request_sync(self, *args, **kwargs) -> str:
            self.calls += 1
            return next(self.outputs)

    analyzer = SequencedAnalyzer()
    transcript = Transcript.model_validate(load_json("mock_transcript.json"))
    result = asyncio.run(
        analyzer._validated_call(SemanticAnalysis, "system", "user", "test", transcript.segments)
    )
    assert isinstance(result, SemanticAnalysis)
    assert analyzer.calls == 2


def test_action_item_aliases_missing_fields_and_extras_are_normalized() -> None:
    payload = copy.deepcopy(load_json("expected_report.json"))
    payload.pop("stats")
    payload["meeting_minutes"]["action_items"] = [
        {"item": "用 item 字段的任务", "unexpected": "应丢弃"},
        {"content": "用 content 字段的任务", "owner": "小宁", "extra": 1},
        {"description": "用 description 字段的任务", "deadline": "明天"},
    ]

    result = run_semantic_model_output(payload)

    assert [item.task for item in result.meeting_minutes.action_items] == [
        "用 item 字段的任务",
        "用 content 字段的任务",
        "用 description 字段的任务",
    ]
    assert result.meeting_minutes.action_items[0].owner == "未明确"
    assert result.meeting_minutes.action_items[0].deadline == "未明确"
    assert result.meeting_minutes.action_items[1].deadline == "未明确"
    assert result.meeting_minutes.action_items[2].owner == "未明确"


def test_performance_dict_is_normalized_to_three_findings() -> None:
    payload = copy.deepcopy(load_json("expected_report.json"))
    payload.pop("stats")
    findings = payload["performance_analysis"]
    payload["performance_analysis"] = {
        "clarity": [{"score": findings[0]["score"], "assessment": findings[0]["assessment"], "evidence_quotes": findings[0]["evidence"]}],
        "conclusion_first": {"score": findings[1]["score"], "finding": findings[1]["assessment"], "evidence": findings[1]["evidence"]},
        "logical_structure": {"score": findings[2]["score"], "content": findings[2]["assessment"], "quotes": findings[2]["evidence"]},
        "unknown_dimension": {"assessment": "这一项应丢弃", "evidence": []},
    }

    result = run_semantic_model_output(payload)

    assert [finding.dimension for finding in result.performance_analysis] == ["表达清晰度", "结论先行", "逻辑结构"]
    assert all(finding.evidence for finding in result.performance_analysis)


def test_string_improvement_suggestions_are_normalized_to_objects() -> None:
    payload = copy.deepcopy(load_json("expected_report.json"))
    payload.pop("stats")
    suggestions = [
        "先说结论，再用两点理由展开",
        "每个行动项都说明负责人和时间",
        "用停顿代替然后和就是等衔接词",
    ]
    payload["improvement_suggestions"] = suggestions

    result = run_semantic_model_output(payload)

    assert [item.action for item in result.improvement_suggestions] == suggestions
    assert [item.title for item in result.improvement_suggestions] == [item[:10] for item in suggestions]
    assert all(item.example == "" for item in result.improvement_suggestions)


def test_string_scores_are_normalized_to_integers() -> None:
    payload = copy.deepcopy(load_json("expected_report.json"))
    payload.pop("stats")
    payload["overall_score"] = str(payload["overall_score"])
    for finding in payload["performance_analysis"]:
        finding["score"] = str(finding["score"])

    normalized = normalize_llm_payload(payload, SemanticAnalysis)
    assert isinstance(normalized["overall_score"], int)
    assert all(isinstance(finding["score"], int) for finding in normalized["performance_analysis"])

    result = run_semantic_model_output(payload)

    assert result.overall_score == 7
    assert [finding.score for finding in result.performance_analysis] == [8, 8, 6]
    assert isinstance(result.overall_score, int)
    assert all(isinstance(finding.score, int) for finding in result.performance_analysis)


@pytest.mark.parametrize(
    ("target", "invalid_score"),
    [("overall", 0), ("overall", 11), ("dimension", 0), ("dimension", 11)],
)
def test_score_out_of_range_is_rejected(target, invalid_score) -> None:
    payload = copy.deepcopy(load_json("expected_report.json"))
    payload.pop("stats")
    if target == "overall":
        payload["overall_score"] = invalid_score
    else:
        payload["performance_analysis"][0]["score"] = invalid_score

    with pytest.raises(ValidationError):
        SemanticAnalysis.model_validate(payload)


def test_overall_comment_longer_than_fifty_characters_is_rejected() -> None:
    payload = copy.deepcopy(load_json("expected_report.json"))
    payload.pop("stats")
    payload["overall_comment"] = "过" * 51

    with pytest.raises(ValidationError):
        SemanticAnalysis.model_validate(payload)


def test_json_object_mode_injects_chinese_contract_and_minimal_example() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.request = None

        def create(self, **kwargs):
            self.request = kwargs
            return SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
            )

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    analyzer = LLMAnalyzer(Settings(OPENAI_API_KEY="test"), client=client)

    analyzer._request_sync(SemanticAnalysis, "system", "original prompt", "test", "json_object")

    prompt = completions.request["messages"][1]["content"]
    assert "JSON 字段契约" in prompt
    assert "action_items" in prompt
    assert "performance_analysis" in prompt
    assert "improvement_suggestions" in prompt
    assert "overall_score" in prompt
    assert "overall_comment" in prompt
    assert "dimension/score/assessment/evidence" in prompt
    assert "字段名必须与示例完全一致，不要输出多余字段" in prompt

    analyzer._request_sync(TeamMeetingReport, "system", "team prompt", "test", "json_object")
    team_prompt = completions.request["messages"][1]["content"]
    assert "suggested_title" in team_prompt
    assert "overview(必填字符串，不超过 300 字)" in team_prompt
    assert "unresolved_issues" in team_prompt
    assert '"content":"尚未解决的问题"' in team_prompt


def test_team_payload_normalizes_overview_and_unresolved_issue_shape() -> None:
    payload = {
        "meeting_title": "内测上线安排确认",
        "meeting_overview": "会议讨论内测上线安排。",
        "meeting_points": ["本周上线"],
        "decisions": [{
            "decision": "本周上线",
            "owner": "小宁",
            "evidence": {"quote": "我们本周上线内测", "timestamp": "00:00:02"},
            "extra": "discard",
        }],
        "action_items": [{"item": "准备清单"}],
        "unresolved_issues": [{
            "question": "监控阈值尚未确定",
            "evidence": {"quote": "监控阈值还没定", "timestamp": "00:00:08"},
            "extra": "discard",
        }],
        "extra": "discard",
    }

    normalized = normalize_llm_payload(payload, TeamMeetingReport)
    report = TeamMeetingReport.model_validate(normalized)

    assert report.suggested_title == "内测上线安排确认"
    assert report.overview == "会议讨论内测上线安排。"
    assert report.decisions[0].content == "本周上线"
    assert report.decisions[0].decision_maker == "小宁"
    assert report.action_items[0].owner == "未明确"
    assert report.unresolved_issues[0].content == "监控阈值尚未确定"


def test_frontend_uses_team_report_and_safe_text_rendering() -> None:
    html = (Path(__file__).parent.parent / "static" / "index.html").read_text(encoding="utf-8")
    assert "speaker_stats_note" in html
    assert "说话人识别将于下一版本支持" not in html  # comes from the API contract
    assert "setTimeout(resolve,ms)" in html
    assert "meeting-review-task-id" in html
    assert "meeting-review-project-id" in html
    assert "meeting-review-history-scope" in html
    assert "/api/tasks/" in html
    assert "await sleep(2000)" in html
    assert "最大 300 MB、4 小时" in html
    assert "/api/auth/check" in html
    assert "X-Access-Token" in html
    assert "meeting-review-access-token" in html
    assert "credentials:'omit'" in html
    assert "POLL_LIMIT_MS=6*60*60*1000" in html
    assert "长会议仍在后台处理" in html
    assert "音频转写完成后即删除" in html
    assert "确认并记住此声音" in html
    assert "organizeMeetingDialog" in html
    assert "整理本次会议" in html
    assert "suggestedMeetingTitle" in html
    assert "/title`" in html
    assert "openHistory(id,completedReport,true)" in html
    assert "/speakers/" in html
    assert "/voiceprint" in html
    assert "innerHTML" not in html
    assert "请输入你的团队口令" in html
    assert "location.protocol==='file:'" in html
    assert "打开正确服务地址" in html
    assert "你打开的是本地网页文件，无法连接后端服务" in html
    assert "无法连接后端服务，请确认服务已经启动后再重试" in html
    assert "response.status===403" in html
    assert "项目连续回顾" in html
    assert "/memory`" in html
    assert "/api/action-items/" in html
    assert "['待确认','进行中','已完成','已取消']" in html
    assert "最近 ${memory.recent_meetings.length} 场 · ${openCount} 项未完成" in html
    assert "/api/meetings" in html
    assert "/api/projects" in html
    assert "projectSelect" in html
    assert "data.append('project_id'" in html
    assert "选择项目" in html
    assert "只删除项目，会议移入“未分类”" in html
    assert "删除项目及其中全部会议" in html
    assert "parentProjectSelect" not in html
    assert "最多两级" not in html
    assert "会议元数据、转写稿和报告都会删除，且不可恢复" in html
    assert "confirmPermanentDelete" in html
    assert "$('permanentDeleteDialog').showModal()" in html
    assert "moveMeetingDialog" in html
    assert "会议报告和历史记录会完整保留" in html
    assert "projectTree" in html
    assert 'data-view="home"' in html
    assert 'data-view="meetings"' in html
    assert 'data-view="projects"' in html
    assert html.index('data-view="home"') < html.index('data-view="meetings"') < html.index('data-view="projects"')
    assert 'id="homeView"' in html
    assert 'id="meetingsView"' in html
    assert 'id="projectsView"' in html
    assert 'id="projectDetailView"' in html
    assert "会脉 · 团队会议记忆" in html
    assert '<div class="brand">团队复盘</div>' not in html
    assert "@media(max-width:760px){.app-shell{grid-template-columns:104px minmax(0,1fr)}" in html
    assert ".app-sidebar{position:sticky;z-index:10;top:0;height:100vh" in html
    assert "folder-tree-item" not in html
    assert "history-workspace" not in html
    assert "每个文件夹就是一个项目，不再分二级" in html
    assert "openProject(root.id)" in html
    assert "单层项目文件夹" in html
    assert "showCreateProject" in html
    assert "manageProjectDialog" in html
    assert "saveProjectName" in html
    assert "当前：${project.name}" in html
    assert "include_children=true" in html
    assert "还没有项目，点击上方“＋ 新建项目”创建第一个项目" in html
    assert "还没有会议。回到首页选择项目并上传一段录音" in html
    assert "localStorage.setItem(PROJECT_STORAGE_KEY" in html
    assert "localStorage.setItem(HISTORY_SCOPE_STORAGE_KEY" in html
    assert "① 会议总览" in html
    assert "⑤ 遗留问题" in html
    assert "查看 ${evidence.timestamp} 附近的转写原文" in html
    assert "body.transcript||[]" in html
    assert "textContent=report.overview" in html
    assert "（含子文件夹）" not in html
    assert "projectFilter" not in html
    assert "/project`" in html
    assert "delete_meetings=${deleteMeetings}" in html
    assert "面向产品项目组" not in html
    assert "innerHTML" not in html
    assert html.count("window.fetch(") == 1


def test_fastapi_uses_product_name() -> None:
    assert main_module.app.title == "会脉 · 团队会议记忆"


def test_existing_database_adds_project_column_without_losing_meetings(tmp_path) -> None:
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE teams(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                token_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE meetings(
                id TEXT PRIMARY KEY,
                team_id INTEGER NOT NULL REFERENCES teams(id),
                title TEXT NOT NULL,
                audio_path TEXT,
                duration_seconds REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            INSERT INTO teams(id,name,token_hash,created_at)
            VALUES(1,'旧团队','old-hash','2026-01-01T00:00:00+00:00');
            INSERT INTO meetings(id,team_id,title,audio_path,status,created_at)
            VALUES('legacy-meeting',1,'旧会议',NULL,'完成','2026-01-01T00:00:00+00:00');
            """
        )

    database = Database(path)
    database.initialize({"旧团队": "new-token"})

    meetings = database.list_meetings(1)
    assert [item.id for item in meetings] == ["legacy-meeting"]
    assert meetings[0].project_id is None
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(meetings)")}
        project_columns = {row[1] for row in connection.execute("PRAGMA table_info(projects)")}
    assert "project_id" in columns
    assert "parent_id" in project_columns


def test_project_folders_are_created_listed_and_isolated_by_team() -> None:
    with TestClient(main_module.app) as client:
        created = client.post(
            "/api/projects", headers=AUTH_HEADERS, json={"name": "年度规划"}
        )
        assert created.status_code == 201
        project = created.json()
        assert project["name"] == "年度规划"
        assert project["meeting_count"] == 0

        listed = client.get("/api/projects", headers=AUTH_HEADERS)
        assert [item["id"] for item in listed.json()] == [project["id"]]
        duplicate = client.post(
            "/api/projects", headers=AUTH_HEADERS, json={"name": "年度规划"}
        )
        assert duplicate.status_code == 409

        other_headers = {"X-Access-Token": "other-team-token"}
        assert client.get("/api/projects", headers=other_headers).json() == []
        denied = client.patch(
            f"/api/projects/{project['id']}", headers=other_headers,
            json={"name": "越权改名"},
        )
        assert denied.status_code == 403
        denied_filter = client.get(
            f"/api/meetings?project_id={project['id']}", headers=other_headers
        )
        assert denied_filter.status_code == 403


def test_project_folders_support_two_levels_but_reject_a_third() -> None:
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        root = client.post("/api/projects", json={"name": "客户项目"}).json()
        child_response = client.post(
            "/api/projects", json={"name": "2026年度", "parent_id": root["id"]}
        )
        assert child_response.status_code == 201
        child = child_response.json()
        assert child["parent_id"] == root["id"]

        third = client.post(
            "/api/projects", json={"name": "第三层", "parent_id": child["id"]}
        )
        assert third.status_code == 422
        assert "最多支持两级" in third.json()["detail"]


def test_root_project_filter_can_include_child_folder_meetings() -> None:
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        root = client.post("/api/projects", json={"name": "汇总根目录"}).json()
        child = client.post(
            "/api/projects", json={"name": "二级目录", "parent_id": root["id"]}
        ).json()
        team_id = main_module.database.authenticate("test-access-token")
        assert team_id is not None
        main_module.database.create_meeting(
            "root-meeting", team_id, "一级会议", Path("/tmp/root.wav"), root["id"]
        )
        main_module.database.create_meeting(
            "child-meeting", team_id, "二级会议", Path("/tmp/child.wav"), child["id"]
        )

        direct = client.get(f"/api/meetings?project_id={root['id']}")
        combined = client.get(
            f"/api/meetings?project_id={root['id']}&include_children=true"
        )
        child_only = client.get(f"/api/meetings?project_id={child['id']}")

        assert [item["id"] for item in direct.json()] == ["root-meeting"]
        assert {item["id"] for item in combined.json()} == {
            "root-meeting", "child-meeting",
        }
        assert [item["id"] for item in child_only.json()] == ["child-meeting"]


def test_project_parent_must_belong_to_current_team() -> None:
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        root = client.post("/api/projects", json={"name": "甲团队根目录"}).json()
        response = client.post(
            "/api/projects",
            headers={"X-Access-Token": "other-team-token"},
            json={"name": "越权子目录", "parent_id": root["id"]},
        )
        assert response.status_code == 403


def test_project_with_child_folder_cannot_be_deleted() -> None:
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        root = client.post("/api/projects", json={"name": "有子目录"}).json()
        child = client.post(
            "/api/projects", json={"name": "子目录", "parent_id": root["id"]}
        ).json()

        response = client.delete(f"/api/projects/{root['id']}")

        assert response.status_code == 409
        assert "先处理二级文件夹" in response.json()["detail"]
        team_id = main_module.database.authenticate("test-access-token")
        assert team_id is not None
        assert main_module.database.get_project(root["id"], team_id) is not None
        assert main_module.database.get_project(child["id"], team_id) is not None


def test_review_can_be_assigned_to_project_folder(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "probe_audio_duration", lambda path: 60)

    async def fake_submit(path, request_id, task_id, team_id, long_meeting):
        return TaskAccepted(
            task_id=task_id, status="排队中", queue_position=0,
            message="已进入处理队列", long_meeting=long_meeting,
        )

    monkeypatch.setattr(main_module.task_manager, "submit", fake_submit)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        project = client.post("/api/projects", json={"name": "产品发布"}).json()
        response = client.post(
            "/api/review",
            data={"project_id": project["id"], "title": "发布准备会"},
            files={"file": ("meeting.wav", b"fake", "audio/wav")},
        )
        assert response.status_code == 202

        meetings = client.get(
            f"/api/meetings?project_id={project['id']}"
        ).json()
        assert len(meetings) == 1
        assert meetings[0]["title"] == "发布准备会"
        assert meetings[0]["project_id"] == project["id"]
        assert client.get("/api/meetings?unclassified=true").json() == []


def test_meeting_can_move_between_projects_without_losing_report_or_transcript() -> None:
    transcript = Transcript.model_validate(load_json("mock_transcript.json"))
    report = team_report()
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        source = client.post("/api/projects", json={"name": "原文件夹"}).json()
        target = client.post("/api/projects", json={"name": "目标文件夹"}).json()
        team_id = main_module.database.authenticate("test-access-token")
        assert team_id is not None
        main_module.database.create_meeting(
            "movable-meeting", team_id, "可移动会议", Path("/tmp/movable.wav"), source["id"]
        )
        main_module.database.save_transcript("movable-meeting", team_id, transcript)
        main_module.database.save_report("movable-meeting", team_id, report)
        main_module.database.clear_audio_path("movable-meeting", team_id)
        main_module.database.update_status("movable-meeting", team_id, "完成")

        response = client.patch(
            "/api/meetings/movable-meeting/project",
            json={"project_id": target["id"]},
        )

        assert response.status_code == 200
        assert response.json()["project_id"] == target["id"]
        assert main_module.database.transcript_rows("movable-meeting")
        history = client.get("/api/meetings/movable-meeting")
        assert history.status_code == 200
        assert history.json()["report"] == report.model_dump()


def test_meeting_move_rejects_cross_team_target_project() -> None:
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        source = client.post("/api/projects", json={"name": "本团队目录"}).json()
        other_headers = {"X-Access-Token": "other-team-token"}
        target = client.post(
            "/api/projects", headers=other_headers, json={"name": "其他团队目录"}
        ).json()
        team_id = main_module.database.authenticate("test-access-token")
        assert team_id is not None
        main_module.database.create_meeting(
            "isolated-move", team_id, "不能越权移动", Path("/tmp/isolated.wav"), source["id"]
        )

        response = client.patch(
            "/api/meetings/isolated-move/project", json={"project_id": target["id"]}
        )

        assert response.status_code == 403
        current = main_module.database.list_meetings(team_id, project_id=source["id"])
        assert [item.id for item in current] == ["isolated-move"]


def test_deleting_project_can_move_meetings_to_unclassified() -> None:
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        project = client.post("/api/projects", json={"name": "待归档"}).json()
        team_id = main_module.database.authenticate("test-access-token")
        assert team_id is not None
        main_module.database.create_meeting(
            "move-meeting", team_id, "需要保留的会议", Path("/tmp/move.wav"), project["id"]
        )

        response = client.delete(f"/api/projects/{project['id']}")

        assert response.status_code == 200
        assert response.json() == {
            "project_id": project["id"],
            "affected_meetings": 1,
            "meetings_deleted": False,
        }
        unclassified = client.get("/api/meetings?unclassified=true").json()
        assert [meeting["id"] for meeting in unclassified] == ["move-meeting"]


def test_deleting_project_and_terminal_meetings_cascades_private_data() -> None:
    transcript = Transcript.model_validate(load_json("mock_transcript.json"))
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        project = client.post("/api/projects", json={"name": "全部删除"}).json()
        team_id = main_module.database.authenticate("test-access-token")
        assert team_id is not None
        main_module.database.create_meeting(
            "delete-meeting", team_id, "删除的会议", Path("/tmp/delete.wav"), project["id"]
        )
        main_module.database.save_transcript("delete-meeting", team_id, transcript)
        main_module.database.save_report("delete-meeting", team_id, team_report())
        main_module.database.clear_audio_path("delete-meeting", team_id)
        main_module.database.update_status("delete-meeting", team_id, "完成")

        response = client.delete(
            f"/api/projects/{project['id']}?delete_meetings=true"
        )

        assert response.status_code == 200
        assert response.json()["meetings_deleted"] is True
        assert response.json()["affected_meetings"] == 1
        assert main_module.database.owner_team_id("delete-meeting") is None
        assert main_module.database.transcript_rows("delete-meeting") == []
        assert main_module.database.get_history("delete-meeting", team_id) is None


def test_project_with_processing_meeting_cannot_be_deleted_with_meetings() -> None:
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        project = client.post("/api/projects", json={"name": "处理中"}).json()
        team_id = main_module.database.authenticate("test-access-token")
        assert team_id is not None
        main_module.database.create_meeting(
            "active-meeting", team_id, "正在处理", Path("/tmp/active.wav"), project["id"]
        )

        response = client.delete(
            f"/api/projects/{project['id']}?delete_meetings=true"
        )

        assert response.status_code == 409
        assert "正在处理" in response.json()["detail"]
        assert main_module.database.get_project(project["id"], team_id) is not None


def test_project_delete_is_forbidden_across_teams() -> None:
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        project = client.post("/api/projects", json={"name": "甲团队项目"}).json()
        response = client.delete(
            f"/api/projects/{project['id']}",
            headers={"X-Access-Token": "other-team-token"},
        )

        assert response.status_code == 403


def test_chunk_evidence_outside_chunk_is_retried_without_logging_quotes(caplog) -> None:
    chunk = [TranscriptSegment(start=10, end=20, text="这是该分块中的真实原话。")]
    invalid = {
        "key_points": [],
        "conclusions": [],
        "action_items": [],
        "personal_observations": [],
        "evidence_quotes": [{"quote": "其他分块的话", "timestamp": "00:00:10"}],
    }
    valid = {
        **invalid,
        "evidence_quotes": [{"quote": "真实原话", "timestamp": "00:00:10"}],
    }

    class SequencedChunkAnalyzer(LLMAnalyzer):
        def __init__(self) -> None:
            super().__init__(Settings(OPENAI_API_KEY="test", LLM_MAX_RETRIES=2))
            self.outputs = iter([json.dumps(invalid, ensure_ascii=False), json.dumps(valid, ensure_ascii=False)])
            self.calls = 0

        def _request_sync(self, *args, **kwargs) -> str:
            self.calls += 1
            return next(self.outputs)

    analyzer = SequencedChunkAnalyzer()
    caplog.set_level("WARNING")
    result = asyncio.run(
        analyzer._validated_call(ChunkSummary, "system", "user", "chunk_1", chunk)
    )
    assert isinstance(result, ChunkSummary)
    assert analyzer.calls == 2
    assert "其他分块的话" not in caplog.text
    assert "真实原话" not in caplog.text


def test_chunk_calls_run_concurrently_and_keep_source_order() -> None:
    expected = load_json("expected_report.json")
    semantic = SemanticAnalysis.model_validate({key: value for key, value in expected.items() if key != "stats"})
    transcript = Transcript(
        duration_seconds=30,
        segments=[
            TranscriptSegment(start=0, end=10, text="甲" * 700),
            TranscriptSegment(start=10, end=20, text="乙" * 700),
            TranscriptSegment(start=20, end=30, text="丙" * 700),
        ],
    )

    class ConcurrentAnalyzer(LLMAnalyzer):
        def __init__(self) -> None:
            super().__init__(Settings(OPENAI_API_KEY="test", TRANSCRIPT_CHUNK_CHARS=1200))
            self.active = 0
            self.max_active = 0
            self.merge_prompt = ""

        async def _validated_call(self, model_type, system_prompt, user_prompt, stage, transcript_segments=None):
            if stage.startswith("chunk_"):
                index = int(stage.split("_")[1])
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                try:
                    await asyncio.sleep(0.01 * (4 - index))
                    return ChunkSummary(
                        key_points=[stage],
                        conclusions=[],
                        action_items=[],
                        personal_observations=[],
                        evidence_quotes=[],
                    )
                finally:
                    self.active -= 1
            self.merge_prompt = user_prompt
            return semantic

    analyzer = ConcurrentAnalyzer()
    result = asyncio.run(analyzer.analyze(transcript))
    assert isinstance(result, SemanticAnalysis)
    assert analyzer.max_active == 3
    assert analyzer.merge_prompt.index("chunk_1") < analyzer.merge_prompt.index("chunk_2")
    assert analyzer.merge_prompt.index("chunk_2") < analyzer.merge_prompt.index("chunk_3")


def test_review_endpoint_returns_task_and_reuses_module_level_analyzer(monkeypatch) -> None:
    transcript = Transcript.model_validate(load_json("mock_transcript.json"))
    report = ReviewReport.model_validate(load_json("expected_report.json"))
    received_analyzers = []

    monkeypatch.setattr(main_module.transcriber, "transcribe", lambda path: transcript)

    async def fake_build_report(received_transcript, received_analyzer):
        assert received_transcript is transcript
        received_analyzers.append(received_analyzer)
        return report

    monkeypatch.setattr(main_module, "build_team_report", fake_build_report)
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        for _ in range(2):
            response = client.post("/api/review", files={"file": ("sample.wav", b"fake", "audio/wav")})
            assert response.status_code == 202
            body = response.json()
            assert body["task_id"]
            assert body["status"] == "排队中"

            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                task_response = client.get(f"/api/tasks/{body['task_id']}")
                assert task_response.status_code == 200
                task = task_response.json()
                if task["status"] == "完成":
                    assert task["report"] == report.model_dump()
                    break
                time.sleep(0.01)
            else:
                pytest.fail("task did not complete")

    assert received_analyzers == [main_module.analyzer, main_module.analyzer]


async def _wait_for_task_status(manager, task_id, expected, timeout=1.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        task = manager.get(task_id)
        if task and task.status == expected:
            return task
        await asyncio.sleep(0.001)
    raise AssertionError(f"task {task_id} did not reach {expected}")


def test_task_queue_is_fifo_and_processes_only_one_audio_at_a_time(tmp_path) -> None:
    report = ReviewReport.model_validate(load_json("expected_report.json"))

    async def scenario():
        releases = {"first.wav": asyncio.Event(), "second.wav": asyncio.Event()}
        starts = {name: asyncio.Event() for name in releases}
        start_order = []
        active = 0
        max_active = 0

        async def processor(path, progress):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            start_order.append(path.name)
            starts[path.name].set()
            progress("转写中", "正在转写")
            await releases[path.name].wait()
            progress("AI 分析中", "正在分析")
            active -= 1
            return report

        manager = InMemoryTaskManager(processor, timeout_seconds=2, retention_seconds=1800)
        first_path = tmp_path / "first.wav"
        second_path = tmp_path / "second.wav"
        first_path.write_bytes(b"first")
        second_path.write_bytes(b"second")
        try:
            first = await manager.submit(first_path, "request-1")
            await starts["first.wav"].wait()
            second = await manager.submit(second_path, "request-2")
            queued = manager.get(second.task_id)
            assert queued.status == "排队中"
            assert queued.queue_position == 1
            assert not starts["second.wav"].is_set()

            releases["first.wav"].set()
            await _wait_for_task_status(manager, first.task_id, "完成")
            await starts["second.wav"].wait()
            releases["second.wav"].set()
            await _wait_for_task_status(manager, second.task_id, "完成")

            assert start_order == ["first.wav", "second.wav"]
            assert max_active == 1
            assert not first_path.exists()
            assert not second_path.exists()
        finally:
            await manager.stop()

    asyncio.run(scenario())


def test_task_status_flows_through_all_processing_stages(tmp_path) -> None:
    report = ReviewReport.model_validate(load_json("expected_report.json"))

    async def scenario():
        continue_to_analysis = asyncio.Event()
        finish = asyncio.Event()

        async def processor(path, progress):
            progress("转写中", "正在转写")
            await continue_to_analysis.wait()
            progress("AI 分析中", "正在分析")
            await finish.wait()
            return report

        manager = InMemoryTaskManager(processor, timeout_seconds=2, retention_seconds=1800)
        path = tmp_path / "flow.wav"
        path.write_bytes(b"audio")
        try:
            accepted = await manager.submit(path, "flow-request")
            await _wait_for_task_status(manager, accepted.task_id, "转写中")
            continue_to_analysis.set()
            await _wait_for_task_status(manager, accepted.task_id, "AI 分析中")
            finish.set()
            completed = await _wait_for_task_status(manager, accepted.task_id, "完成")
            assert completed.report == report
            assert completed.error is None
            assert manager.records[accepted.task_id].history == [
                "上传完成", "排队中", "转写中", "AI 分析中", "完成"
            ]
        finally:
            await manager.stop()

    asyncio.run(scenario())


def test_audio_over_duration_limit_fails_before_llm_analysis(monkeypatch, tmp_path) -> None:
    transcript = Transcript(
        duration_seconds=3601,
        segments=[TranscriptSegment(start=0, end=3601, text="超长录音")],
    )
    monkeypatch.setattr(main_module.settings, "max_audio_minutes", 60)
    monkeypatch.setattr(main_module.transcriber, "transcribe", lambda path: transcript)
    analysis_called = False

    async def fake_build_report(*args):
        nonlocal analysis_called
        analysis_called = True

    monkeypatch.setattr(main_module, "build_team_report", fake_build_report)
    path = tmp_path / "long.wav"
    path.write_bytes(b"audio")

    async def scenario():
        manager = InMemoryTaskManager(main_module._process_audio, timeout_seconds=2, retention_seconds=1800)
        try:
            accepted = await manager.submit(path, "duration-request")
            failed = await _wait_for_task_status(manager, accepted.task_id, "失败")
            assert failed.error_code == 422
            assert failed.error == "当前版本支持 60 分钟以内的录音"
            assert failed.report is None
        finally:
            await manager.stop()

    asyncio.run(scenario())
    assert analysis_called is False
    assert not path.exists()


def test_processing_timeout_is_reported_without_releasing_busy_worker(tmp_path) -> None:
    report = ReviewReport.model_validate(load_json("expected_report.json"))

    async def scenario():
        release_first = asyncio.Event()
        second_started = asyncio.Event()

        async def processor(path, progress):
            progress("转写中", "正在转写")
            if path.name == "slow.wav":
                await release_first.wait()
            else:
                second_started.set()
            return report

        manager = InMemoryTaskManager(processor, timeout_seconds=0.02, retention_seconds=1800)
        slow_path = tmp_path / "slow.wav"
        next_path = tmp_path / "next.wav"
        slow_path.write_bytes(b"slow")
        next_path.write_bytes(b"next")
        try:
            slow = await manager.submit(slow_path, "slow-request")
            failed = await _wait_for_task_status(manager, slow.task_id, "失败")
            assert failed.error_code == 504
            assert "处理超过" in failed.error
            assert not slow_path.exists()

            following = await manager.submit(next_path, "next-request")
            await asyncio.sleep(0.01)
            assert not second_started.is_set()
            assert manager.get(following.task_id).status == "排队中"

            release_first.set()
            await second_started.wait()
            await _wait_for_task_status(manager, following.task_id, "完成")
        finally:
            await manager.stop()

    asyncio.run(scenario())


def test_graceful_stop_deletes_running_and_queued_audio(tmp_path) -> None:
    async def scenario():
        started = asyncio.Event()

        async def processor(path, progress):
            started.set()
            await asyncio.Event().wait()

        manager = InMemoryTaskManager(processor, timeout_seconds=10, retention_seconds=1800)
        running_path = tmp_path / "running-stop.wav"
        queued_path = tmp_path / "queued-stop.wav"
        running_path.write_bytes(b"running")
        queued_path.write_bytes(b"queued")
        await manager.submit(running_path, "request-running")
        await started.wait()
        await manager.submit(queued_path, "request-queued")
        await manager.stop()

        assert not running_path.exists()
        assert not queued_path.exists()

    asyncio.run(scenario())


def test_logs_never_contain_transcript_text(monkeypatch, tmp_path, caplog) -> None:
    sensitive_text = "绝不能出现在日志里的私密转写内容"
    transcript = Transcript(
        duration_seconds=10,
        segments=[TranscriptSegment(start=0, end=10, text=sensitive_text)],
    )
    report = ReviewReport.model_validate(load_json("expected_report.json"))
    monkeypatch.setattr(main_module.transcriber, "transcribe", lambda path: transcript)

    async def fake_build_report(*args):
        return report

    monkeypatch.setattr(main_module, "build_team_report", fake_build_report)
    path = tmp_path / "private.wav"
    path.write_bytes(b"audio")
    caplog.set_level("INFO")

    result = asyncio.run(main_module._process_audio(path, lambda status, message: True))

    assert result == report
    assert sensitive_text not in caplog.text


def test_only_terminal_tasks_are_cleaned_after_retention(tmp_path) -> None:
    async def processor(path, progress):
        await asyncio.Event().wait()

    async def scenario():
        manager = InMemoryTaskManager(processor, timeout_seconds=2, retention_seconds=1800)
        running_path = tmp_path / "running.wav"
        running_path.write_bytes(b"audio")
        try:
            accepted = await manager.submit(running_path, "cleanup-request")
            await _wait_for_task_status(manager, accepted.task_id, "排队中")
            record = manager.records[accepted.task_id]
            assert manager.cleanup_expired(now=record.updated_at + 1801) == 0

            record.status = "失败"
            record.finished_at = record.updated_at
            assert manager.cleanup_expired(now=record.finished_at + 1799) == 0
            assert manager.cleanup_expired(now=record.finished_at + 1800) == 1
            assert manager.get(accepted.task_id) is None
        finally:
            await manager.stop()

    asyncio.run(scenario())


def test_team_report_rejects_hallucinated_decision_quote() -> None:
    from app.llm import validate_team_evidence

    transcript = Transcript(
        duration_seconds=10,
        segments=[TranscriptSegment(start=0, end=10, text="我们本周上线内测")],
    )
    report = team_report()
    validate_team_evidence(report, transcript.segments)
    report.decisions[0].evidence.quote = "下周上线"
    with pytest.raises(ValueError):
        validate_team_evidence(report, transcript.segments)


def test_team_report_rejects_hallucinated_unresolved_issue_quote_or_timestamp() -> None:
    from app.llm import validate_team_evidence

    transcript = Transcript(
        duration_seconds=20,
        segments=[TranscriptSegment(
            start=0, end=10, text="我们本周上线内测，但是监控阈值还没定"
        )],
    )
    payload = team_report().model_dump()
    payload["unresolved_issues"] = [{
        "content": "监控阈值尚未确定",
        "evidence": {"quote": "监控阈值还没定", "timestamp": "00:00:08"},
    }]
    report = TeamMeetingReport.model_validate(payload)
    validate_team_evidence(report, transcript.segments)

    report.unresolved_issues[0].evidence.timestamp = "00:00:20"
    with pytest.raises(ValueError):
        validate_team_evidence(report, transcript.segments)


def test_legacy_team_report_gets_safe_defaults_and_transcript_context(tmp_path) -> None:
    database = Database(tmp_path / "legacy-report.db")
    database.initialize({"测试团队": "token"})
    team_id = database.authenticate("token")
    assert team_id is not None
    meeting_id = "legacy-report"
    database.create_meeting(meeting_id, team_id, "旧报告", Path("/tmp/legacy.wav"))
    transcript = Transcript(
        duration_seconds=10,
        segments=[TranscriptSegment(start=0, end=10, text="我们本周上线内测")],
    )
    database.save_transcript(meeting_id, team_id, transcript)
    legacy_payload = team_report().model_dump()
    legacy_payload.pop("overview")
    legacy_payload.pop("unresolved_issues")
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "INSERT INTO reports(meeting_id,json,created_at) VALUES(?,?,?)",
            (meeting_id, json.dumps(legacy_payload, ensure_ascii=False), datetime.now(timezone.utc).isoformat()),
        )

    history = database.get_history(meeting_id, team_id)

    assert history is not None
    assert history.report.overview == "该报告生成于旧版本，暂无会议总览。"
    assert history.report.unresolved_issues == []
    assert [segment.text for segment in history.transcript] == ["我们本周上线内测"]


def test_startup_backfills_action_status_for_existing_reports(tmp_path) -> None:
    path = tmp_path / "action-backfill.db"
    database = Database(path)
    database.initialize({"测试团队": "token"})
    team_id = database.authenticate("token")
    assert team_id is not None
    project = database.create_project("legacy-project", team_id, "旧项目")
    database.create_meeting(
        "legacy-action-meeting", team_id, "旧周会", Path("/tmp/legacy-action.wav"), project.id
    )
    database.update_status("legacy-action-meeting", team_id, "完成")
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO reports(meeting_id,json,created_at) VALUES(?,?,?)",
            (
                "legacy-action-meeting", team_report().model_dump_json(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    restarted = Database(path)
    restarted.initialize({"测试团队": "token"})
    memory = restarted.get_project_memory(project.id, team_id)

    assert memory is not None
    assert len(memory.action_items) == 1
    assert memory.action_items[0].task == "准备发布清单"
    assert memory.action_items[0].status == "待确认"


def test_transcribed_audio_is_deleted_and_transcript_and_report_are_persisted(monkeypatch, tmp_path) -> None:
    team_id = main_module.database.authenticate("test-access-token")
    meeting_id = "a" * 32
    audio_path = tmp_path / f"{meeting_id}.wav"
    audio_path.write_bytes(b"private audio")
    main_module.database.create_meeting(meeting_id, team_id, "周会", audio_path)
    transcript = Transcript(
        duration_seconds=10,
        segments=[TranscriptSegment(start=0, end=10, text="我们本周上线内测")],
    )
    expected = team_report()
    monkeypatch.setattr(main_module.transcriber, "transcribe", lambda path: transcript)

    async def fake_team_report(received, analyzer):
        assert received is transcript
        assert not audio_path.exists(), "Whisper 返回后应在 AI 分析前删除音频"
        return expected

    monkeypatch.setattr(main_module, "build_team_report", fake_team_report)
    actual = asyncio.run(main_module._process_audio(audio_path, lambda status, message: True))

    assert actual == expected
    assert not audio_path.exists()
    rows = list(main_module.database.transcript_rows(meeting_id))
    assert [row["text"] for row in rows] == ["我们本周上线内测"]
    history = main_module.database.get_history(meeting_id, team_id)
    assert history is not None
    assert history.report == expected


def test_speaker_recognition_failure_is_visible_in_report(monkeypatch, tmp_path) -> None:
    team_id = main_module.database.authenticate("test-access-token")
    meeting_id = "speaker-failure-visible"
    audio_path = tmp_path / f"{meeting_id}.m4a"
    audio_path.write_bytes(b"private audio")
    main_module.database.create_meeting(meeting_id, team_id, "声纹失败提示", audio_path)
    transcript = Transcript(
        duration_seconds=5,
        segments=[TranscriptSegment(start=0, end=5, text="这是一段会议发言")],
    )
    monkeypatch.setattr(main_module.transcriber, "transcribe", lambda path: transcript)
    monkeypatch.setattr(
        main_module.speaker_recognizer,
        "process",
        lambda path, received, profiles: SpeakerRecognitionResult(
            received, received, [], False, "说话人识别失败，请稍后重试",
        ),
    )

    async def fake_team_report(received, analyzer):
        return team_report()

    monkeypatch.setattr(main_module, "build_team_report", fake_team_report)
    result = asyncio.run(main_module._process_audio(audio_path, lambda status, message: True))

    assert result.speaker_stats_note == "说话人识别失败，请稍后重试"
    history = main_module.database.get_history(meeting_id, team_id)
    assert history is not None
    assert history.report.speaker_stats_note == result.speaker_stats_note


def test_failed_transcription_also_deletes_audio(monkeypatch, tmp_path) -> None:
    path = tmp_path / ("f" * 32 + ".wav")
    path.write_bytes(b"private audio")

    def fail(_):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(main_module.transcriber, "transcribe", fail)
    with pytest.raises(RuntimeError):
        asyncio.run(main_module._process_audio(path, lambda status, message: True))
    assert not path.exists()


def test_history_and_task_access_are_isolated_by_team() -> None:
    first_id = main_module.database.authenticate("test-access-token")
    second_id = main_module.database.authenticate("other-team-token")
    report = team_report()
    first_meeting = "1" * 32
    second_meeting = "2" * 32
    main_module.database.create_meeting(first_meeting, first_id, "甲团队周会", Path("/tmp/first.wav"))
    main_module.database.create_meeting(second_meeting, second_id, "乙团队周会", Path("/tmp/second.wav"))
    main_module.database.save_report(first_meeting, first_id, report)
    main_module.database.save_report(second_meeting, second_id, report)

    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        listing = client.get("/api/meetings")
        assert listing.status_code == 200
        assert [item["title"] for item in listing.json()] == ["甲团队周会"]
        assert client.get(f"/api/meetings/{first_meeting}").status_code == 200
        assert client.get(f"/api/meetings/{second_meeting}").status_code == 403
        assert client.get(f"/api/tasks/{second_meeting}").status_code == 403


def test_ai_suggested_title_can_be_confirmed_or_edited_with_team_isolation() -> None:
    team_id = main_module.database.authenticate("test-access-token")
    other_team_id = main_module.database.authenticate("other-team-token")
    meeting_id = "title-confirm-meeting"
    main_module.database.create_meeting(
        meeting_id, team_id, "未命名会议", Path("/tmp/title-confirm.wav")
    )
    report = team_report().model_copy(update={"suggested_title": "AI 项目上线决策"})
    main_module.database.save_report(meeting_id, team_id, report)

    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        updated = client.patch(
            f"/api/meetings/{meeting_id}/title", json={"title": "AI 项目上线会"}
        )
        assert updated.status_code == 200
        assert updated.json()["title"] == "AI 项目上线会"
        assert updated.json()["report"]["suggested_title"] == "AI 项目上线决策"

        denied = client.patch(
            f"/api/meetings/{meeting_id}/title",
            headers={"X-Access-Token": "other-team-token"},
            json={"title": "越权修改"},
        )
        assert denied.status_code == 403
        assert main_module.database.owner_team_id(meeting_id) != other_team_id


def test_project_memory_uses_latest_three_meetings_and_updates_action_status() -> None:
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        project = client.post("/api/projects", json={"name": "连续项目"}).json()
        team_id = main_module.database.authenticate("test-access-token")
        assert team_id is not None
        for index in range(4):
            meeting_id = f"memory-meeting-{index}"
            main_module.database.create_meeting(
                meeting_id, team_id, f"第{index + 1}次周会", Path(f"/tmp/{meeting_id}.wav"),
                project["id"],
            )
            payload = team_report().model_dump()
            payload["overview"] = f"第{index + 1}次周会总览"
            payload["action_items"] = [{
                "task": f"任务{index + 1}", "owner": "小宁", "deadline": "周五"
            }]
            main_module.database.save_report(
                meeting_id, team_id, TeamMeetingReport.model_validate(payload)
            )
            main_module.database.update_status(meeting_id, team_id, "完成")
            with sqlite3.connect(main_module.database.path) as connection:
                connection.execute(
                    "UPDATE meetings SET created_at=? WHERE id=?",
                    (f"2026-09-{index + 1:02d}T09:00:00+00:00", meeting_id),
                )

        response = client.get(f"/api/projects/{project['id']}/memory")

        assert response.status_code == 200
        memory = response.json()
        assert [item["title"] for item in memory["recent_meetings"]] == [
            "第4次周会", "第3次周会", "第2次周会"
        ]
        assert [item["task"] for item in memory["action_items"]] == ["任务4", "任务3", "任务2"]
        assert all(item["status"] == "待确认" for item in memory["action_items"])
        assert all(item["source"]["title"].endswith("次周会") for item in memory["decisions"])

        action_id = memory["action_items"][0]["id"]
        updated = client.patch(
            f"/api/action-items/{action_id}", json={"status": "已完成"}
        )
        assert updated.status_code == 200
        assert updated.json() == {"id": action_id, "status": "已完成"}
        refreshed = client.get(f"/api/projects/{project['id']}/memory").json()
        assert refreshed["action_items"][0]["status"] == "已完成"
        assert client.patch(
            f"/api/action-items/{action_id}", json={"status": "假的状态"}
        ).status_code == 422


def test_project_memory_and_action_status_are_isolated_by_team() -> None:
    other_headers = {"X-Access-Token": "other-team-token"}
    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        other_project = client.post(
            "/api/projects", headers=other_headers, json={"name": "乙团队连续项目"}
        ).json()
        other_team_id = main_module.database.authenticate("other-team-token")
        assert other_team_id is not None
        meeting_id = "other-memory-meeting"
        main_module.database.create_meeting(
            meeting_id, other_team_id, "乙团队周会", Path("/tmp/other-memory.wav"),
            other_project["id"],
        )
        main_module.database.save_report(meeting_id, other_team_id, team_report())
        main_module.database.update_status(meeting_id, other_team_id, "完成")
        other_memory = client.get(
            f"/api/projects/{other_project['id']}/memory", headers=other_headers
        ).json()
        action_id = other_memory["action_items"][0]["id"]

        assert client.get(f"/api/projects/{other_project['id']}/memory").status_code == 403
        assert client.patch(
            f"/api/action-items/{action_id}", json={"status": "已完成"}
        ).status_code == 403
        assert client.patch(
            f"/api/action-items/{action_id}", headers=other_headers,
            json={"status": "已完成"},
        ).status_code == 200


def test_voice_match_requires_threshold_and_clear_margin() -> None:
    ning = KnownVoiceProfile(member_id=1, name="宁总", embedding=[1.0, 0.0])
    li = KnownVoiceProfile(member_id=2, name="李总", embedding=[0.0, 1.0])

    match, confidence = best_profile_match([0.99, 0.01], [ning, li], 0.72, 0.05)
    assert match == ning
    assert confidence is not None and confidence > 0.99

    ambiguous, _ = best_profile_match([1.0, 1.0], [ning, li], 0.72, 0.05)
    assert ambiguous is None
    rejected, _ = best_profile_match([-1.0, 0.0], [ning], 0.72, 0.05)
    assert rejected is None


def test_unknown_speaker_model_falls_back_without_stopping_service(monkeypatch, tmp_path) -> None:
    recognizer = SpeakerRecognizer(Settings(_env_file=None))
    monkeypatch.setattr(recognizer, "_load_model", lambda: (_ for _ in ()).throw(SystemExit(1)))
    transcript = Transcript(
        duration_seconds=3,
        segments=[TranscriptSegment(start=0, end=3, text="测试发言")],
    )

    result = recognizer.process(tmp_path / "missing.wav", transcript)

    assert result.available is False
    assert result.transcript == transcript
    assert "声纹模型不可用" in result.message


def test_speaker_input_is_normalized_to_mono_16khz_wav(tmp_path) -> None:
    source = tmp_path / "stereo-44k.wav"
    with wave.open(str(source), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(44100)
        output.writeframes(b"\x00\x00\x00\x00" * 44100)

    recognizer = SpeakerRecognizer(Settings(_env_file=None))
    with recognizer._normalized_wav(source) as normalized:
        assert normalized.exists()
        with wave.open(str(normalized), "rb") as result:
            assert result.getnchannels() == 1
            assert result.getsampwidth() == 2
            assert result.getframerate() == 16000
            assert 15900 <= result.getnframes() <= 16100
    assert not normalized.exists()


def test_speaker_recognition_runs_before_audio_deletion_and_persists_labels(monkeypatch, tmp_path) -> None:
    team_id = main_module.database.authenticate("test-access-token")
    meeting_id = "speaker-processing-meeting"
    audio_path = tmp_path / f"{meeting_id}.wav"
    audio_path.write_bytes(b"private audio")
    main_module.database.create_meeting(meeting_id, team_id, "多人会议", audio_path)
    raw = Transcript(
        duration_seconds=8,
        segments=[TranscriptSegment(start=0, end=4, text="我同意今天上线")],
    )
    labeled = raw.model_copy(update={
        "segments": [raw.segments[0].model_copy(update={"speaker_label": "说话人 1"})]
    })
    monkeypatch.setattr(main_module.transcriber, "transcribe", lambda path: raw)

    def fake_speaker(path, transcript, profiles):
        assert path.exists(), "声纹特征必须在原始音频删除前提取"
        assert transcript is raw
        return SpeakerRecognitionResult(
            transcript=labeled, analysis_transcript=labeled,
            observations=[SpeakerObservation(
                local_label="说话人 1", embedding=[1.0, 0.0], speech_seconds=4,
                excerpts=["我同意今天上线"],
            )],
            available=True, message="已区分 1 位说话人",
        )

    monkeypatch.setattr(main_module.speaker_recognizer, "process", fake_speaker)

    async def fake_report(received, analyzer):
        assert received.segments[0].speaker_label == "说话人 1"
        assert not audio_path.exists()
        return team_report()

    monkeypatch.setattr(main_module, "build_team_report", fake_report)
    result = asyncio.run(main_module._process_audio(audio_path, lambda status, message: True))

    assert result.speaker_stats_note == "已区分 1 位说话人"
    assert not audio_path.exists()
    history = main_module.database.get_history(meeting_id, team_id)
    assert history is not None
    assert history.transcript[0].speaker_label == "说话人 1"
    assert history.speakers[0].status == "待确认"
    assert history.speakers[0].has_voice_sample is True


def test_confirm_voice_once_enables_team_profile_and_enforces_consent_and_isolation() -> None:
    team_id = main_module.database.authenticate("test-access-token")
    meeting_id = "voice-confirm-meeting"
    main_module.database.create_meeting(
        meeting_id, team_id, "声纹确认", Path("/tmp/voice-confirm.wav")
    )
    transcript = Transcript(
        duration_seconds=6,
        segments=[TranscriptSegment(
            start=0, end=6, text="我来做最终决策", speaker_label="说话人 1",
        )],
    )
    main_module.database.save_transcript(meeting_id, team_id, transcript)
    main_module.database.save_meeting_speakers(meeting_id, team_id, [SpeakerObservation(
        local_label="说话人 1", embedding=[0.8, 0.2], speech_seconds=6,
        excerpts=["我来做最终决策"],
    )])
    payload = team_report().model_dump()
    payload["decisions"][0]["decision_maker"] = "说话人 1"
    payload["action_items"][0]["owner"] = "说话人 1"
    main_module.database.save_report(
        meeting_id, team_id, TeamMeetingReport.model_validate(payload)
    )

    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        no_consent = client.post(
            f"/api/meetings/{meeting_id}/speakers/%E8%AF%B4%E8%AF%9D%E4%BA%BA%201/confirm",
            json={"name": "宁总", "remember_voice": True, "consent_confirmed": False},
        )
        assert no_consent.status_code == 422

        confirmed = client.post(
            f"/api/meetings/{meeting_id}/speakers/%E8%AF%B4%E8%AF%9D%E4%BA%BA%201/confirm",
            json={
                "name": "宁总", "role": "老板", "is_key_decision_maker": True,
                "remember_voice": True, "consent_confirmed": True,
            },
        )
        assert confirmed.status_code == 200
        assert confirmed.json()["voiceprint_saved"] is True
        assert confirmed.json()["reanalysis_recommended"] is True

        history = client.get(f"/api/meetings/{meeting_id}").json()
        assert history["speakers"][0]["display_name"] == "宁总"
        assert history["speakers"][0]["status"] == "已确认"
        assert history["report"]["decisions"][0]["decision_maker"] == "宁总"
        members = client.get("/api/members").json()
        assert members[0]["name"] == "宁总"
        assert members[0]["is_key_decision_maker"] is True
        assert members[0]["has_voiceprint"] is True
        profile = main_module.database.voice_profiles(team_id)[0]
        assert profile["embedding"] == [0.8, 0.2]
        future_match, _ = best_profile_match(
            [0.81, 0.19], [KnownVoiceProfile(**profile)], 0.72, 0.05
        )
        assert future_match is not None and future_match.name == "宁总"

        other = client.post(
            f"/api/meetings/{meeting_id}/speakers/%E8%AF%B4%E8%AF%9D%E4%BA%BA%201/confirm",
            headers={"X-Access-Token": "other-team-token"},
            json={"name": "其他人", "remember_voice": False},
        )
        assert other.status_code == 403

        deleted = client.delete(f"/api/members/{members[0]['id']}/voiceprint")
        assert deleted.status_code == 200
        assert client.get("/api/members").json()[0]["has_voiceprint"] is False


def test_duplicate_speaker_identities_can_be_merged_within_team() -> None:
    team_id = main_module.database.authenticate("test-access-token")
    meeting_id = "voice-merge-meeting"
    main_module.database.create_meeting(
        meeting_id, team_id, "身份合并", Path("/tmp/voice-merge.wav")
    )
    main_module.database.save_transcript(meeting_id, team_id, Transcript(
        duration_seconds=8,
        segments=[
            TranscriptSegment(start=0, end=4, text="第一段", speaker_label="说话人 1"),
            TranscriptSegment(start=4, end=8, text="第二段", speaker_label="说话人 2"),
        ],
    ))
    main_module.database.save_meeting_speakers(meeting_id, team_id, [
        SpeakerObservation("说话人 1", [1.0, 0.0], 4, ["第一段"]),
        SpeakerObservation("说话人 2", [0.9, 0.1], 4, ["第二段"]),
    ])
    main_module.database.save_report(meeting_id, team_id, team_report())

    with TestClient(main_module.app, headers=AUTH_HEADERS) as client:
        ids = []
        for label, name in (("说话人 1", "宁总"), ("说话人 2", "宁老师")):
            response = client.post(
                f"/api/meetings/{meeting_id}/speakers/{label.replace(' ', '%20')}/confirm",
                json={"name": name, "remember_voice": False},
            )
            assert response.status_code == 200
            ids.append(response.json()["member_id"])

        merged = client.post(
            f"/api/members/{ids[1]}/merge", json={"target_member_id": ids[0]}
        )
        assert merged.status_code == 200
        assert merged.json()["merged_meetings"] == 1
        assert [item["name"] for item in client.get("/api/members").json()] == ["宁总"]
        speakers = client.get(f"/api/meetings/{meeting_id}").json()["speakers"]
        assert [item["display_name"] for item in speakers] == ["宁总", "宁总"]

        other_team_attempt = client.post(
            f"/api/members/{ids[0]}/merge",
            headers={"X-Access-Token": "other-team-token"},
            json={"target_member_id": ids[0]},
        )
        assert other_team_attempt.status_code == 403

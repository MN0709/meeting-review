import logging
import math
import threading
from pathlib import Path
from typing import Any, Optional

import av

from app.config import Settings
from app.models import Transcript, TranscriptSegment


logger = logging.getLogger(__name__)
WHISPER_SAMPLE_RATE = 16000
LONG_AUDIO_CHUNK_SECONDS = 30 * 60


def probe_audio_duration(audio_path: Path) -> Optional[float]:
    """Read duration from container/stream metadata without decoding audio frames."""
    try:
        with av.open(str(audio_path), mode="r") as container:
            if container.duration is not None:
                seconds = float(container.duration / av.time_base)
                if math.isfinite(seconds) and seconds > 0:
                    return seconds

            stream_durations = [
                float(stream.duration * stream.time_base)
                for stream in container.streams.audio
                if stream.duration is not None and stream.time_base is not None
            ]
            valid_durations = [
                seconds for seconds in stream_durations if math.isfinite(seconds) and seconds > 0
            ]
            return max(valid_durations) if valid_durations else None
    except Exception as exc:
        # 损坏文件或缺失元数据不在这里拒绝；交给 Whisper 的转写后时长检查兜底。
        logger.info("audio_duration_probe_unavailable error_type=%s", type(exc).__name__)
        return None


class WhisperTranscriber:
    """Lazy-load faster-whisper so /health does not trigger a model download."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Optional[Any] = None
        self._lock = threading.Lock()

    def _get_model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from faster_whisper import WhisperModel

                    self._model = WhisperModel(
                        self.settings.whisper_model,
                        device=self.settings.whisper_device,
                        compute_type=self.settings.whisper_compute_type,
                    )
        return self._model

    def transcribe(self, audio_path: Path) -> Transcript:
        model = self._get_model()
        metadata_duration = probe_audio_duration(audio_path)
        if metadata_duration is not None and metadata_duration > LONG_AUDIO_CHUNK_SECONDS:
            return self._transcribe_long_audio(model, audio_path, metadata_duration)
        raw_segments, info = model.transcribe(
            str(audio_path),
            language="zh",
            vad_filter=True,
            beam_size=5,
        )
        segments = [
            TranscriptSegment(start=item.start, end=item.end, text=item.text.strip())
            for item in raw_segments
            if item.text.strip()
        ]
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        if segments:
            duration = max(duration, segments[-1].end)
        return Transcript(
            language=str(getattr(info, "language", "zh") or "zh"),
            duration_seconds=duration,
            segments=segments,
        )

    @staticmethod
    def _decode_audio_chunks(audio_path: Path):
        """Decode at most 30 minutes at once so multi-hour input does not occupy ~1 GB RAM."""
        import numpy as np

        chunk_bytes = LONG_AUDIO_CHUNK_SECONDS * WHISPER_SAMPLE_RATE * 2
        pending = bytearray()
        with av.open(str(audio_path), mode="r") as container:
            if not container.streams.audio:
                return
            resampler = av.AudioResampler(
                format="s16", layout="mono", rate=WHISPER_SAMPLE_RATE,
            )

            def append_frames(frames) -> None:
                for frame in frames or []:
                    pending.extend(frame.to_ndarray().reshape(-1).tobytes())

            for frame in container.decode(audio=0):
                append_frames(resampler.resample(frame))
                while len(pending) >= chunk_bytes:
                    raw = bytes(pending[:chunk_bytes])
                    del pending[:chunk_bytes]
                    yield np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            append_frames(resampler.resample(None))
            if pending:
                yield np.frombuffer(bytes(pending), dtype="<i2").astype(np.float32) / 32768.0

    def _transcribe_long_audio(
        self, model: Any, audio_path: Path, metadata_duration: float,
    ) -> Transcript:
        segments: list[TranscriptSegment] = []
        offset = 0.0
        language = "zh"
        chunk_count = 0
        for audio in self._decode_audio_chunks(audio_path):
            chunk_count += 1
            raw_segments, info = model.transcribe(
                audio,
                language="zh",
                vad_filter=True,
                beam_size=5,
            )
            language = str(getattr(info, "language", language) or language)
            chunk_duration = len(audio) / WHISPER_SAMPLE_RATE
            segments.extend(
                TranscriptSegment(
                    start=offset + item.start,
                    end=offset + item.end,
                    text=item.text.strip(),
                )
                for item in raw_segments
                if item.text.strip()
            )
            offset += chunk_duration
        logger.info(
            "long_audio_transcribed duration_seconds=%.1f chunks=%d",
            metadata_duration,
            chunk_count,
        )
        duration = max(metadata_duration, offset)
        if segments:
            duration = max(duration, segments[-1].end)
        return Transcript(language=language, duration_seconds=duration, segments=segments)

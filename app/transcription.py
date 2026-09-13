import logging
import math
import threading
from pathlib import Path
from typing import Any, Optional

import av

from app.config import Settings
from app.models import Transcript, TranscriptSegment


logger = logging.getLogger(__name__)


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

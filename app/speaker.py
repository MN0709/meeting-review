from __future__ import annotations

import logging
import math
import os
import io
import tempfile
import threading
import wave
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from app.config import Settings
from app.models import Transcript, TranscriptSegment

logger = logging.getLogger(__name__)
_TORCHAUDIO_LOAD_LOCK = threading.RLock()


@dataclass
class KnownVoiceProfile:
    member_id: int
    name: str
    embedding: list[float]


@dataclass
class SpeakerObservation:
    local_label: str
    embedding: Optional[list[float]]
    speech_seconds: float
    excerpts: list[str] = field(default_factory=list)
    matched_member_id: Optional[int] = None
    matched_name: Optional[str] = None
    confidence: Optional[float] = None
    clips: list["SpeakerClipObservation"] = field(default_factory=list)


@dataclass
class SpeakerClipObservation:
    start: float
    end: float
    text: str
    audio: bytes


@dataclass
class SpeakerRecognitionResult:
    transcript: Transcript
    analysis_transcript: Transcript
    observations: list[SpeakerObservation]
    available: bool
    message: str


def cosine_score(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    # WeSpeaker uses this normalized cosine range for recognition.
    return max(0.0, min(1.0, (dot / (left_norm * right_norm) + 1.0) / 2.0))


def best_profile_match(
    embedding: Sequence[float], profiles: Iterable[KnownVoiceProfile],
    threshold: float, margin: float,
) -> tuple[Optional[KnownVoiceProfile], Optional[float]]:
    ranked = sorted(
        ((cosine_score(embedding, profile.embedding), profile) for profile in profiles),
        key=lambda item: item[0], reverse=True,
    )
    if not ranked or ranked[0][0] < threshold:
        return None, ranked[0][0] if ranked else None
    runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
    if ranked[0][0] - runner_up < margin:
        return None, ranked[0][0]
    return ranked[0][1], ranked[0][0]


class SpeakerRecognizer:
    """Lazy local WeSpeaker adapter; absence of optional runtime never invents identities."""

    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.speaker_recognition_enabled
        self.model_name = settings.speaker_model
        self.threshold = settings.speaker_match_threshold
        self.margin = settings.speaker_match_margin
        self.intra_merge_threshold = settings.speaker_intra_merge_threshold
        self._model = None

    def _load_model(self):
        if self._model is None:
            import wespeaker  # type: ignore[import-not-found]

            self._model = wespeaker.load_model(self.model_name)
        return self._model

    @staticmethod
    def _normalize_turns(raw_turns) -> list[tuple[float, float, int]]:
        turns: list[tuple[float, float, int]] = []
        for item in raw_turns or []:
            if len(item) >= 4:
                _, start, end, label = item[:4]
            elif len(item) == 3:
                start, end, label = item
            else:
                continue
            turns.append((float(start), float(end), int(label)))
        return turns

    @staticmethod
    def _assign_label(segment: TranscriptSegment, turns: Sequence[tuple[float, float, int]]) -> Optional[int]:
        overlaps = [
            (max(0.0, min(segment.end, end) - max(segment.start, start)), label)
            for start, end, label in turns
        ]
        if not overlaps:
            return None
        overlap, label = max(overlaps)
        return label if overlap > 0 else None

    @staticmethod
    def _cluster_embedding(model, pcm, sample_rate: int, ranges: Sequence[tuple[float, float]]) -> Optional[list[float]]:
        import torch
        pieces = [
            pcm[:, max(0, int(start * sample_rate)):max(0, int(end * sample_rate))]
            for start, end in ranges if end > start
        ]
        pieces = [piece for piece in pieces if piece.numel()]
        if not pieces:
            return None
        merged = torch.cat(pieces, dim=1)
        if merged.shape[1] < sample_rate:
            return None
        embedding = model.extract_embedding_from_pcm(merged, sample_rate)
        if embedding is None:
            return None
        return [float(value) for value in embedding.detach().cpu().flatten().tolist()]

    @staticmethod
    def _merge_similar_labels(
        embeddings: dict[int, Optional[list[float]]], threshold: float,
    ) -> dict[int, int]:
        labels = sorted(embeddings)
        parent = {label: label for label in labels}

        def find(label: int) -> int:
            while parent[label] != label:
                parent[label] = parent[parent[label]]
                label = parent[label]
            return label

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        for index, left in enumerate(labels):
            for right in labels[index + 1:]:
                left_embedding, right_embedding = embeddings[left], embeddings[right]
                if left_embedding and right_embedding:
                    if cosine_score(left_embedding, right_embedding) >= threshold:
                        union(left, right)
        return {label: find(label) for label in labels}

    @staticmethod
    def _clip_wav(pcm, sample_rate: int, start: float, end: float) -> bytes:
        import torch

        start_index = max(0, int(start * sample_rate))
        end_index = min(pcm.shape[1], int(end * sample_rate))
        samples = pcm[:, start_index:end_index].mean(dim=0)
        samples = samples.clamp(-1, 1).mul(32767).to(torch.int16).cpu().numpy().tobytes()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(samples)
        return buffer.getvalue()

    @staticmethod
    @contextmanager
    def _normalized_wav(audio_path: Path):
        """Give WeSpeaker a predictable mono 16 kHz WAV for MP3/M4A/WAV uploads."""
        import av

        temporary = tempfile.NamedTemporaryFile(
            prefix="meeting-review-speaker-", suffix=".wav", delete=False,
        )
        normalized_path = Path(temporary.name)
        temporary.close()
        try:
            with av.open(str(audio_path), mode="r") as container, wave.open(
                str(normalized_path), "wb"
            ) as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
                for frame in container.decode(audio=0):
                    for converted in resampler.resample(frame) or []:
                        output.writeframes(converted.to_ndarray().reshape(-1).tobytes())
                for converted in resampler.resample(None) or []:
                    output.writeframes(converted.to_ndarray().reshape(-1).tobytes())
            yield normalized_path
        finally:
            normalized_path.unlink(missing_ok=True)

    @staticmethod
    @contextmanager
    def _torchaudio_wav_fallback():
        """Read our PCM WAV even when torchaudio ships without an I/O backend."""
        import torch
        import torchaudio

        with _TORCHAUDIO_LOAD_LOCK:
            original_load = torchaudio.load

            def load_with_wave_fallback(uri, *args, **kwargs):
                try:
                    return original_load(uri, *args, **kwargs)
                except RuntimeError:
                    path = Path(os.fspath(uri))
                    if path.suffix.lower() != ".wav":
                        raise
                    with wave.open(str(path), "rb") as source:
                        if source.getcomptype() != "NONE" or source.getsampwidth() != 2:
                            raise
                        channels = source.getnchannels()
                        sample_rate = source.getframerate()
                        frame_offset = int(kwargs.get("frame_offset", 0))
                        num_frames = int(kwargs.get("num_frames", -1))
                        if frame_offset:
                            source.setpos(min(frame_offset, source.getnframes()))
                        raw = source.readframes(
                            source.getnframes() if num_frames < 0 else num_frames
                        )
                    pcm = torch.frombuffer(bytearray(raw), dtype=torch.int16)
                    if channels > 1:
                        pcm = pcm.reshape(-1, channels).transpose(0, 1).contiguous()
                    else:
                        pcm = pcm.reshape(1, -1)
                    if kwargs.get("normalize", True):
                        pcm = pcm.to(torch.float32).div_(32768.0)
                    if not kwargs.get("channels_first", True):
                        pcm = pcm.transpose(0, 1)
                    return pcm, sample_rate

            torchaudio.load = load_with_wave_fallback
            try:
                yield
            finally:
                torchaudio.load = original_load

    def process(
        self, audio_path: Path, transcript: Transcript,
        profiles: Sequence[KnownVoiceProfile] = (),
    ) -> SpeakerRecognitionResult:
        if not self.enabled:
            return SpeakerRecognitionResult(
                transcript, transcript, [], False, "说话人识别未启用",
            )
        try:
            self._load_model()
        except (ImportError, ModuleNotFoundError):
            logger.warning("speaker_runtime_unavailable model=%s", self.model_name)
            return SpeakerRecognitionResult(
                transcript, transcript, [], False,
                "声纹运行时未安装，本场暂不识别说话人",
            )
        except SystemExit:
            logger.warning("speaker_model_unavailable model=%s", self.model_name)
            return SpeakerRecognitionResult(
                transcript, transcript, [], False,
                "声纹模型不可用，本场暂不识别说话人",
            )
        try:
            with self._normalized_wav(audio_path) as normalized_path:
                with self._torchaudio_wav_fallback():
                    return self._process_supported_audio(normalized_path, transcript, profiles)
        except Exception as exc:
            logger.warning(
                "speaker_audio_normalization_failed error_type=%s", type(exc).__name__,
            )
            return SpeakerRecognitionResult(
                transcript, transcript, [], False,
                "本场音频无法进入说话人识别，已保留完整转写和报告",
            )

    def _process_supported_audio(
        self, audio_path: Path, transcript: Transcript,
        profiles: Sequence[KnownVoiceProfile] = (),
    ) -> SpeakerRecognitionResult:
        try:
            model = self._load_model()
            turns = self._normalize_turns(model.diarize(str(audio_path), audio_path.stem))
        except (ImportError, ModuleNotFoundError):
            logger.warning("speaker_runtime_unavailable model=%s", self.model_name)
            return SpeakerRecognitionResult(
                transcript, transcript, [], False,
                "声纹运行时未安装，本场暂不识别说话人",
            )
        except SystemExit:
            # WeSpeaker's hub exits the process for an unknown built-in model name.
            # Treat configuration/model lookup failures as a safe feature fallback.
            logger.warning("speaker_model_unavailable model=%s", self.model_name)
            return SpeakerRecognitionResult(
                transcript, transcript, [], False,
                "声纹模型不可用，本场暂不识别说话人",
            )
        except Exception as exc:
            logger.warning("speaker_recognition_failed error_type=%s", type(exc).__name__)
            return SpeakerRecognitionResult(
                transcript, transcript, [], False,
                "本场未能稳定区分说话人，已安全回退为未标注转写",
            )
        if not turns:
            return SpeakerRecognitionResult(
                transcript, transcript, [], True, "本场未检测到可分离的说话人",
            )

        raw_labels = sorted({label for _, _, label in turns})
        try:
            import torchaudio

            pcm, sample_rate = torchaudio.load(str(audio_path), normalize=True)
            if pcm.size(0) > 1:
                pcm = pcm.mean(dim=0, keepdim=True)
        except Exception as exc:
            logger.warning("speaker_audio_load_failed error_type=%s", type(exc).__name__)
            pcm = None
            sample_rate = 0
        raw_embeddings: dict[int, Optional[list[float]]] = {}
        for raw_label in raw_labels:
            ranges = [(start, end) for start, end, label in turns if label == raw_label]
            try:
                raw_embeddings[raw_label] = (
                    self._cluster_embedding(model, pcm, sample_rate, ranges)
                    if pcm is not None else None
                )
            except Exception as exc:
                logger.warning("speaker_embedding_failed error_type=%s", type(exc).__name__)
                raw_embeddings[raw_label] = None
        merged_labels = self._merge_similar_labels(
            raw_embeddings, self.intra_merge_threshold,
        )
        turns = [(start, end, merged_labels[label]) for start, end, label in turns]
        final_labels = sorted({label for _, _, label in turns})
        display_labels = {
            raw: f"说话人 {index + 1}" for index, raw in enumerate(final_labels)
        }
        segments: list[TranscriptSegment] = []
        for segment in transcript.segments:
            raw_label = self._assign_label(segment, turns)
            segments.append(segment.model_copy(update={
                "speaker_label": display_labels.get(raw_label) if raw_label is not None else None,
            }))
        labeled = transcript.model_copy(update={"segments": segments})

        observations: list[SpeakerObservation] = []
        for raw_label in final_labels:
            local_label = display_labels[raw_label]
            ranges = [(start, end) for start, end, label in turns if label == raw_label]
            try:
                original_labels = [
                    label for label, canonical in merged_labels.items() if canonical == raw_label
                ]
                embedding = raw_embeddings[original_labels[0]] if len(original_labels) == 1 else (
                    self._cluster_embedding(model, pcm, sample_rate, ranges)
                    if pcm is not None else None
                )
            except Exception as exc:
                logger.warning("speaker_embedding_failed error_type=%s", type(exc).__name__)
                embedding = None
            match, confidence = best_profile_match(
                embedding or [], profiles, self.threshold, self.margin,
            )
            speaker_segments = [
                segment for segment in segments if segment.speaker_label == local_label
            ]
            excerpts = [segment.text for segment in speaker_segments[:3]]
            clips: list[SpeakerClipObservation] = []
            if pcm is not None:
                for segment in speaker_segments[:3]:
                    clip_end = min(segment.end, segment.start + 12.0)
                    if clip_end - segment.start < 0.4:
                        continue
                    clips.append(SpeakerClipObservation(
                        start=segment.start, end=clip_end, text=segment.text,
                        audio=self._clip_wav(pcm, sample_rate, segment.start, clip_end),
                    ))
            observations.append(SpeakerObservation(
                local_label=local_label,
                embedding=embedding,
                speech_seconds=sum(end - start for start, end in ranges),
                excerpts=excerpts,
                matched_member_id=match.member_id if match else None,
                matched_name=match.name if match else None,
                confidence=confidence,
                clips=clips,
            ))

        names = {item.local_label: item.matched_name for item in observations if item.matched_name}
        analysis_segments = [
            segment.model_copy(update={"speaker_label": names.get(segment.speaker_label, segment.speaker_label)})
            for segment in segments
        ]
        return SpeakerRecognitionResult(
            transcript=labeled,
            analysis_transcript=transcript.model_copy(update={"segments": analysis_segments}),
            observations=observations,
            available=True,
            message=f"已区分 {len(observations)} 位说话人",
        )

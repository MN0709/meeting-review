from collections import Counter
from typing import Dict, Iterable, Tuple

from app.models import SpeechStats, Transcript


DEFAULT_FILLERS: Tuple[str, ...] = ("然后", "就是", "嗯", "呃")


def _merged_duration(intervals: Iterable[Tuple[float, float]]) -> float:
    """合并重叠时间段，避免将异常重叠的转写片段重复计时。"""
    ordered = sorted((max(0.0, start), max(0.0, end)) for start, end in intervals if end > start)
    if not ordered:
        return 0.0

    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def compute_speech_stats(
    transcript: Transcript,
    fillers: Tuple[str, ...] = DEFAULT_FILLERS,
) -> SpeechStats:
    # 代码统计 vs LLM 统计：口头禅是可精确复现的字符串计数，交给代码才能保证
    # 同一输入每次结果一致；LLM 适合语义判断，但不适合做精确计数。
    # 按 segment 分别计数，避免两段边界恰好拼出一个词而误计。
    counts: Dict[str, int] = Counter(
        {filler: sum(segment.text.count(filler) for segment in transcript.segments) for filler in fillers}
    )

    speech_seconds = _merged_duration((segment.start, segment.end) for segment in transcript.segments)
    recording_seconds = max(
        transcript.duration_seconds,
        max((segment.end for segment in transcript.segments), default=0.0),
    )
    ratio = speech_seconds / recording_seconds * 100 if recording_seconds else 0.0

    return SpeechStats(
        recording_duration_seconds=round(recording_seconds, 2),
        transcribed_speech_seconds=round(speech_seconds, 2),
        speech_ratio_percent=round(min(ratio, 100.0), 1),
        scope_note="该比例是转写语音片段时长占录音总时长，未做说话人分离",
        filler_counts=dict(counts),
        total_fillers=sum(counts.values()),
    )

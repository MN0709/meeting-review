from typing import Protocol

from app.models import ReviewReport, SemanticAnalysis, TeamMeetingReport, Transcript
from app.stats import compute_speech_stats


class Analyzer(Protocol):
    async def analyze(self, transcript: Transcript) -> SemanticAnalysis:
        ...


async def build_report(transcript: Transcript, analyzer: Analyzer) -> ReviewReport:
    stats = compute_speech_stats(transcript)
    semantic = await analyzer.analyze(transcript)
    return ReviewReport(stats=stats, **semantic.model_dump())


async def build_team_report(transcript: Transcript, analyzer) -> TeamMeetingReport:
    return await analyzer.analyze_team(transcript)

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TranscriptSegment(StrictModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str


class Transcript(StrictModel):
    language: str = "zh"
    duration_seconds: float = Field(ge=0)
    segments: List[TranscriptSegment]

    @property
    def text(self) -> str:
        return "".join(segment.text for segment in self.segments).strip()


class ActionItem(StrictModel):
    task: str
    owner: str
    deadline: str


class MeetingMinutes(StrictModel):
    key_points: List[str]
    conclusions: List[str]
    action_items: List[ActionItem]


class EvidenceQuote(StrictModel):
    quote: str = Field(min_length=1)
    timestamp: str = Field(pattern=r"^\d{2}:\d{2}:\d{2}$")


class PerformanceFinding(StrictModel):
    """Legacy personal-review model; retained for compatibility, not shown by the team UI."""
    dimension: Literal["表达清晰度", "结论先行", "逻辑结构"]
    score: int = Field(ge=1, le=10)
    assessment: str
    evidence: List[EvidenceQuote] = Field(min_length=1)


class ImprovementSuggestion(StrictModel):
    title: str
    action: str
    example: str = ""


class SemanticAnalysis(StrictModel):
    meeting_minutes: MeetingMinutes
    overall_score: int = Field(ge=1, le=10)
    overall_comment: str = Field(max_length=50)
    performance_analysis: List[PerformanceFinding] = Field(min_length=3, max_length=3)
    improvement_suggestions: List[ImprovementSuggestion] = Field(min_length=3, max_length=5)

    @model_validator(mode="after")
    def require_all_dimensions(self) -> "SemanticAnalysis":
        required = {"表达清晰度", "结论先行", "逻辑结构"}
        actual = {item.dimension for item in self.performance_analysis}
        if actual != required:
            raise ValueError("表现分析必须且只能覆盖：表达清晰度、结论先行、逻辑结构")
        return self


class SpeechStats(StrictModel):
    recording_duration_seconds: float
    transcribed_speech_seconds: float
    speech_ratio_percent: float
    scope_note: str
    filler_counts: dict[str, int]
    total_fillers: int


class ReviewReport(SemanticAnalysis):
    """Legacy personal report; retained for tests and existing integrations."""
    stats: SpeechStats


class DecisionItem(StrictModel):
    content: str
    decision_maker: str
    evidence: EvidenceQuote


class TeamMeetingReport(StrictModel):
    meeting_points: List[str]
    decisions: List[DecisionItem]
    action_items: List[ActionItem]
    speaker_stats_note: Literal["说话人识别将于下一版本支持"] = "说话人识别将于下一版本支持"


class TeamChunkSummary(StrictModel):
    meeting_points: List[str]
    decisions: List[DecisionItem]
    action_items: List[ActionItem]


TaskStage = Literal["排队中", "上传完成", "转写中", "AI 分析中", "完成", "失败"]


class TaskAccepted(StrictModel):
    task_id: str
    status: TaskStage
    queue_position: int = Field(ge=0)
    message: str
    long_meeting: bool = False


class TaskStatus(TaskAccepted):
    request_id: str
    report: Optional[Union[TeamMeetingReport, ReviewReport]] = None
    error: Optional[str] = None
    error_code: Optional[int] = None


class ChunkSummary(StrictModel):
    key_points: List[str]
    conclusions: List[str]
    action_items: List[str]
    personal_observations: List[str]
    evidence_quotes: List[EvidenceQuote]


class ProjectCreate(StrictModel):
    name: str = Field(min_length=1, max_length=50)


class ProjectListItem(StrictModel):
    id: str
    name: str
    meeting_count: int = Field(ge=0)
    created_at: str


class MeetingListItem(StrictModel):
    id: str
    title: str
    project_id: Optional[str] = None
    duration_seconds: float
    status: str
    created_at: str


class MeetingHistory(MeetingListItem):
    report: TeamMeetingReport

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TranscriptSegment(StrictModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    speaker_label: Optional[str] = None


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


class UnresolvedIssue(StrictModel):
    content: str
    evidence: EvidenceQuote


class TeamMeetingReport(StrictModel):
    suggested_title: str = Field(default="", max_length=100)
    overview: str = Field(min_length=1, max_length=300)
    meeting_points: List[str]
    decisions: List[DecisionItem]
    action_items: List[ActionItem]
    unresolved_issues: List[UnresolvedIssue]
    speaker_stats_note: str = "说话人识别未启用"


class TeamChunkSummary(StrictModel):
    meeting_points: List[str]
    decisions: List[DecisionItem]
    action_items: List[ActionItem]
    unresolved_issues: List[UnresolvedIssue]


TaskStage = Literal["排队中", "上传完成", "转写中", "说话人识别中", "AI 分析中", "完成", "失败"]


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
    parent_id: Optional[str] = None


class ProjectRename(StrictModel):
    name: str = Field(min_length=1, max_length=50)


class ProjectListItem(StrictModel):
    id: str
    name: str
    parent_id: Optional[str] = None
    meeting_count: int = Field(ge=0)
    created_at: str


class ProjectDeleteResult(StrictModel):
    project_id: str
    affected_meetings: int = Field(ge=0)
    meetings_deleted: bool


class MeetingListItem(StrictModel):
    id: str
    title: str
    project_id: Optional[str] = None
    duration_seconds: float
    status: str
    created_at: str


class MeetingMoveRequest(StrictModel):
    project_id: str


class MeetingTitleUpdate(StrictModel):
    title: str = Field(min_length=1, max_length=100)


class MeetingHistory(MeetingListItem):
    report: TeamMeetingReport
    transcript: List[TranscriptSegment]
    speakers: List["MeetingSpeaker"] = Field(default_factory=list)
    speaker_consent_confirmed: bool = False


SpeakerIdentityStatus = Literal["待确认", "已识别", "已确认", "仅本场"]


class SpeakerClip(StrictModel):
    id: int
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str


class MeetingSpeaker(StrictModel):
    local_label: str
    display_name: str
    member_id: Optional[int] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    status: SpeakerIdentityStatus
    speech_seconds: float = Field(ge=0)
    excerpts: List[str] = Field(default_factory=list)
    clips: List[SpeakerClip] = Field(default_factory=list)
    has_voice_sample: bool = False
    remember_requested: bool = False


class SpeakerConfirmRequest(StrictModel):
    name: str = Field(min_length=1, max_length=50)
    role: str = Field(default="", max_length=50)
    is_key_decision_maker: bool = False
    remember_voice: bool = True


class MeetingFinalizeRequest(StrictModel):
    consent_confirmed: bool = False


class MeetingFinalizeResult(StrictModel):
    voiceprints_saved: int = Field(ge=0)
    members: List[str] = Field(default_factory=list)


class SpeakerConfirmResult(StrictModel):
    member_id: int
    display_name: str
    voiceprint_saved: bool
    reanalysis_recommended: bool


class MemberIdentity(StrictModel):
    id: int
    name: str
    role: str = ""
    is_key_decision_maker: bool = False
    has_voiceprint: bool = False
    created_at: str


class MemberUpdate(StrictModel):
    name: str = Field(min_length=1, max_length=50)
    role: str = Field(default="", max_length=50)
    is_key_decision_maker: bool = False


class MemberMergeRequest(StrictModel):
    target_member_id: int = Field(gt=0)


class MemberMergeResult(StrictModel):
    target_member_id: int
    merged_meetings: int = Field(ge=0)


ActionStatus = Literal["待确认", "进行中", "已完成", "已取消"]


class ActionItemStatusUpdate(StrictModel):
    status: ActionStatus


class ActionItemStatusResult(StrictModel):
    id: int
    status: ActionStatus


class MeetingSource(StrictModel):
    id: str
    title: str
    created_at: str


class ProjectMemoryDecision(DecisionItem):
    source: MeetingSource


class ProjectMemoryAction(ActionItem):
    id: int
    status: ActionStatus
    source: MeetingSource


class ProjectMemoryIssue(UnresolvedIssue):
    source: MeetingSource


class ProjectMemory(StrictModel):
    project_id: str
    project_name: str
    recent_meetings: List[MeetingListItem]
    decisions: List[ProjectMemoryDecision]
    action_items: List[ProjectMemoryAction]
    unresolved_issues: List[ProjectMemoryIssue]


# ---------------------------------------------------------------------------
# LLM 成本归因（PRD R-P0-2）：GET /api/usage 的响应契约
# ---------------------------------------------------------------------------


class LLMUsageTotals(StrictModel):
    calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_ms: int
    # 未配置单价时为 None（不把价格硬编码进代码或数据库）
    cost: Optional[float] = None


class LLMUsageStageSummary(LLMUsageTotals):
    stage: str
    model: str
    first_at: Optional[str] = None
    last_at: Optional[str] = None


class LLMUsageMeetingSummary(LLMUsageTotals):
    meeting_id: str
    project_id: Optional[str] = None
    first_at: Optional[str] = None
    last_at: Optional[str] = None


class LLMUsageFilters(StrictModel):
    meeting_id: Optional[str] = None
    project_id: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    price_configured: bool


class LLMUsageReport(StrictModel):
    filters: LLMUsageFilters
    totals: LLMUsageTotals
    by_stage: List[LLMUsageStageSummary]
    by_meeting: List[LLMUsageMeetingSummary]

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# 会议未填标题时的占位值；D-027 用它判断「标题是否可被 AI 建议替换」。
DEFAULT_MEETING_TITLE = "未命名会议"


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


class EvidenceQuote(StrictModel):
    quote: str = Field(min_length=1)
    timestamp: str = Field(pattern=r"^\d{2}:\d{2}:\d{2}$")


class ActionItem(StrictModel):
    task: str
    owner: str
    deadline: str
    # 新增（可选）：行动项的原话证据。新报告由模型强制提供并严格校验；
    # 旧报告缺失时为 None，保证历史数据仍可读取（D-024 契约向前兼容）。
    evidence: Optional[EvidenceQuote] = None
    # R-P2.1-4：派生字段，读取时按原话定位计算，不落库（None = 未标注说话人）。
    speaker: Optional[str] = None


class MeetingMinutes(StrictModel):
    key_points: List[str]
    conclusions: List[str]
    action_items: List[ActionItem]


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
    # R-P2.1-4：模型不再输出「决策人」；字段保留仅为兼容旧报告，新报告一律为空串。
    decision_maker: str = ""
    evidence: EvidenceQuote
    # R-P2.1-4③：派生字段，读取时按原话定位计算（None = 未标注说话人）。
    speaker: Optional[str] = None
    # R-P2.1-2③：被去重并进决策的行动项负责人，界面显示「落实：X」。
    implementer: Optional[str] = None


class UnresolvedIssue(StrictModel):
    content: str
    evidence: EvidenceQuote
    # R-P2.1-4③：派生字段，读取时按原话定位计算（None = 未标注说话人）。
    speaker: Optional[str] = None


class SuggestedProject(StrictModel):
    """R-P1.5-8：AI 项目归属建议（只建议，不自动移动）。"""

    existing_project_id: Optional[str] = None
    new_project_name: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    reason: str = ""


class SuggestedProjectAction(StrictModel):
    action: Literal["accept", "rename", "dismiss"]
    project_id: Optional[str] = None
    name: Optional[str] = None


class TeamMeetingReport(StrictModel):
    suggested_title: str = Field(default="", max_length=100)
    overview: str = Field(min_length=1, max_length=300)
    meeting_points: List[str]
    decisions: List[DecisionItem]
    action_items: List[ActionItem]
    unresolved_issues: List[UnresolvedIssue]
    # R-P1.5-1（阶段 9-B）：图片纪要的「紧急事项」板块。旧报告无此字段时为空数组。
    urgent_items: List[UnresolvedIssue] = Field(default_factory=list)
    # R-P1.5-8（阶段 10-C）：项目归属建议。旧报告为 None，界面不显示该卡片。
    suggested_project: Optional[SuggestedProject] = None
    speaker_stats_note: str = "说话人识别未启用"


class TeamChunkSummary(StrictModel):
    meeting_points: List[str]
    decisions: List[DecisionItem]
    action_items: List[ActionItem]
    unresolved_issues: List[UnresolvedIssue]
    urgent_items: List[UnresolvedIssue] = Field(default_factory=list)


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
    # R-P1.5-7：按交付物聚合的状态（新增字段，老字段语义不变）。
    deliverables: List["DeliverableState"] = Field(default_factory=list)


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
    # R-P2-6：归类来源（auto / manual / None）与置信度。
    assignment_source: Optional[str] = None
    assignment_confidence: Optional[float] = None


class MeetingMoveRequest(StrictModel):
    project_id: str


class MeetingTitleUpdate(StrictModel):
    title: str = Field(min_length=1, max_length=100)


# ---------------------------------------------------------------------------
# 「我」的指认与我的任务（R-P1.5-5，阶段 9／M2）
# 口径：只做「完全一致」匹配，未指认时一律为空，绝不推断。
# ---------------------------------------------------------------------------


class SelfSpeakerUpdate(StrictModel):
    local_label: Optional[str] = None
    member_id: Optional[int] = None


class SelfSpeakerResult(StrictModel):
    self_speaker_set: bool
    local_label: Optional[str] = None
    member_id: Optional[int] = None
    member_name: Optional[str] = None
    self_name: Optional[str] = None
    # R-P2-3：「我」的来源（voiceprint / manual / None）与置信度。
    source: Optional[str] = None
    confidence: Optional[float] = None


class MyTaskItem(StrictModel):
    id: int
    task: str
    owner: str
    deadline: str
    status: str


class MyTasksResult(StrictModel):
    meeting_id: str
    self_speaker_set: bool
    local_label: Optional[str] = None
    member_id: Optional[int] = None
    self_name: Optional[str] = None
    count: int = Field(ge=0)
    owner_unknown: int = Field(ge=0)
    items: List[MyTaskItem]
    # R-P2-3 ⑤：本人声纹门控状态：skipped / enrolled / not_enrolled；reason 说明未识别原因。
    owner_voiceprint_state: Optional[str] = None
    reason: Optional[str] = None


class OwnerStatus(StrictModel):
    """R-P2-2：本人声纹状态。"""

    enrolled: bool
    skipped: bool
    name: str
    sample_seconds: Optional[float] = None
    enrolled_at: Optional[str] = None
    onboarding_done: bool = False


class OwnerEnrollResult(StrictModel):
    enrolled: bool
    name: str
    sample_seconds: float
    enrolled_at: str
    backfilled: List[Dict[str, Any]] = Field(default_factory=list)


class OwnerBackfillResult(StrictModel):
    requested_limit: int
    scanned: int
    identified: int
    skipped_manual: int
    meetings: List[Dict[str, Any]] = Field(default_factory=list)


class ClaimOwnerVoiceRequest(StrictModel):
    """R-P2-2/3 补充：用某场会议里的一个说话人认领「我」，并把该段声音存成本人声纹。"""

    local_label: str = ""
    consent_confirmed: bool = False


# ---------------------------------------------------------------------------
# R-P2-5 批量上传 / R-P2-6 自动归类
# ---------------------------------------------------------------------------


class BatchReviewItem(StrictModel):
    filename: str
    task_id: Optional[str] = None
    ok: bool
    error: Optional[str] = None


class BatchReviewResult(StrictModel):
    batch_id: str
    total: int
    accepted: int
    items: List[BatchReviewItem]


class BatchTaskProgress(StrictModel):
    task_id: str
    status: str
    stage: Optional[str] = None
    message: Optional[str] = None


class BatchProgress(StrictModel):
    batch_id: str
    total: int
    created_at: str
    accepted: int
    tasks: List[BatchTaskProgress]


class AssignBatchRequest(StrictModel):
    meeting_ids: List[str] = Field(default_factory=list)
    project_id: Optional[str] = None
    # accept_suggestions：逐场采纳各自的 AI 建议（只对既有项目生效）。
    action: Optional[Literal["accept_suggestions"]] = None


class AssignBatchResult(StrictModel):
    assigned: int
    skipped: int
    meetings: List[MeetingListItem] = Field(default_factory=list)


class UndoAssignmentResult(StrictModel):
    meeting_id: str
    undone: bool
    project_id: Optional[str] = None


# ---------------------------------------------------------------------------
# 图片纪要（R-P1.5-1，阶段 9-B）
# 四板块顺序固定；内容只来自报告与行动项，不新增句子。
# ---------------------------------------------------------------------------


class ImageMinutesItem(StrictModel):
    text: str = Field(min_length=1)
    timestamp: Optional[str] = None
    meta: Optional[str] = None
    # R-P2-8 ②：紧急事项并入待办后，用标签区分（不再是独立板块）。
    urgent: bool = False


class ImageMinutesPart(StrictModel):
    key: Literal["core", "todo", "mine", "decisions"]
    title: str
    subtitle: str
    items: List[ImageMinutesItem]
    empty_note: Optional[str] = None


class ImageMinutesResult(StrictModel):
    meeting_id: str
    title: str
    meta: str
    parts: List[ImageMinutesPart]


# ---------------------------------------------------------------------------
# 交付物状态（R-P1.5-7，阶段 9-C）
# ---------------------------------------------------------------------------


class DeliverableState(StrictModel):
    kind: Literal["transcript", "report", "tasks", "image_minutes"]
    label: str
    status: Literal["pending", "ok", "failed", "needs_review"]
    error_code: Optional[str] = None
    message: Optional[str] = None
    retryable: bool = False
    updated_at: Optional[str] = None


class DeliverablesReport(StrictModel):
    meeting_id: str
    items: List[DeliverableState]
    needs_review: int = Field(ge=0)


class RetryResult(StrictModel):
    meeting_id: str
    kind: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# 分享（R-P1.5-3，阶段 11／M4）
# ---------------------------------------------------------------------------


class ShareCreate(StrictModel):
    scopes: List[str] = Field(default_factory=list)


class ShareCreated(StrictModel):
    url: str
    token: str
    scopes: List[str]
    expires_at: str


class ShareLinkItem(StrictModel):
    share_id: int
    token_prefix: str
    scopes: List[str]
    expires_at: str
    created_at: str
    revoked_at: Optional[str] = None
    view_count: int = Field(ge=0, default=0)
    active: bool = True


class ShareLinksPayload(StrictModel):
    items: List[ShareLinkItem]


class MeetingHistory(MeetingListItem):
    report: TeamMeetingReport
    transcript: List[TranscriptSegment]
    speakers: List["MeetingSpeaker"] = Field(default_factory=list)
    speaker_consent_confirmed: bool = False
    self_speaker: Optional[SelfSpeakerResult] = None


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
    # R-P2-4：成员所属项目；None 表示「未归类」桶。
    project_id: Optional[str] = None


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
# R-P2.1-7/8（阶段 17／M2）：待跟进跨会议跟踪
# ---------------------------------------------------------------------------

FollowupStatus = Literal["open", "resolved", "dropped"]


class FollowupItem(StrictModel):
    id: int
    project_id: str
    text: str
    status: FollowupStatus
    first_speaker: Optional[str] = None
    first_meeting_id: Optional[str] = None
    first_meeting_title: Optional[str] = None
    last_seen_meeting_id: Optional[str] = None
    last_seen_meeting_title: Optional[str] = None
    # 该待跟进出现过的所有场次（含首提与后续重现）。
    meeting_ids: List[str] = Field(default_factory=list)
    meeting_count: int = Field(ge=0, default=0)
    created_at: str
    updated_at: str


class FollowupList(StrictModel):
    project_id: str
    total: int = Field(ge=0)
    open: int = Field(ge=0)
    items: List[FollowupItem]


class FollowupStatusUpdate(StrictModel):
    status: FollowupStatus


class FollowupStatusResult(StrictModel):
    id: int
    status: FollowupStatus


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


# ---------------------------------------------------------------------------
# Agent 步骤轨迹（只读展示用；PRD §14.1 M1 验收入口）
# ---------------------------------------------------------------------------


class AgentTraceStep(StrictModel):
    session_id: str
    step: Optional[int] = None
    tool_name: Optional[str] = None
    decision: str
    result_code: Optional[str] = None
    duration_ms: Optional[int] = None
    created_at: str


class AgentPlanItem(StrictModel):
    task: str
    status: str


class AgentTrace(StrictModel):
    meeting_id: str
    mode: str
    audit_enabled: bool
    total_calls: int
    tool_names: List[str]
    steps: List[AgentTraceStep]
    plan: List[AgentPlanItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 跨会议搜索（R-P1.5-4，阶段 8／M1）
# 复用 FTS5 转写索引，只新增只读产品端点，不新增表、不改老契约。
# ---------------------------------------------------------------------------


class SearchHit(StrictModel):
    meeting_id: str
    meeting_title: str
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    timestamp: str = Field(pattern=r"^\d{2}:\d{2}:\d{2}$")
    speaker_label: Optional[str] = None
    text_snippet: str = Field(min_length=1, max_length=200)


class SearchResponse(StrictModel):
    query: str
    count: int = Field(ge=0)
    hits: List[SearchHit]


# TaskStatus.deliverables 引用后定义的 DeliverableState（前向引用）。
TaskStatus.model_rebuild()

from functools import lru_cache
from typing import Dict, Literal, Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# 口令强度下限（PRD R-P0-3）。16 位是阻止"1234"这类可猜口令的最低门槛。
MIN_TEAM_TOKEN_LENGTH = 16


def _token_strength_problem(name: str, token: str) -> Optional[str]:
    """返回口令不合规的原因；合规时返回 None。

    只把"纯 ASCII 数字"和"纯 ASCII 字母"判为弱口令：
    中文长口令（如 16 字以上词组）熵远高于此门槛，不应被误伤。
    """
    if len(token) < MIN_TEAM_TOKEN_LENGTH:
        return f"TEAM_TOKENS 中团队「{name}」的口令至少 {MIN_TEAM_TOKEN_LENGTH} 位（当前 {len(token)} 位）"
    if token.isascii() and token.isdigit():
        return f"TEAM_TOKENS 中团队「{name}」的口令不能是纯数字"
    if token.isascii() and token.isalpha():
        return f"TEAM_TOKENS 中团队「{name}」的口令不能是纯字母"
    return None


class Settings(BaseSettings):
    app_host: str = Field(default="127.0.0.1", alias="APP_HOST")
    app_port: int = Field(default=8000, ge=1, le=65535, alias="APP_PORT")
    forwarded_allow_ips: str = Field(default="127.0.0.1", alias="FORWARDED_ALLOW_IPS")
    # R-P2-1：个人模式（默认）不要求口令；AUTH_ENABLED=true 时才恢复口令校验。
    team_tokens: Optional[SecretStr] = Field(default=None, alias="TEAM_TOKENS")
    auth_enabled: bool = Field(default=False, alias="AUTH_ENABLED")
    owner_name: str = Field(default="我", alias="OWNER_NAME")
    debug_panels_enabled: bool = Field(default=False, alias="DEBUG_PANELS_ENABLED")

    # --- R-P2-2/3/12 本人声纹（比成员匹配更严，误认必须为 0）-------------------
    owner_voiceprint_enabled: bool = Field(default=True, alias="OWNER_VOICEPRINT_ENABLED")
    # 本人声纹阈值（默认 0.80，高于成员阈值）；宁可显示「未识别到你」，不猜。
    owner_match_threshold: float = Field(default=0.80, ge=0, le=1, alias="OWNER_MATCH_THRESHOLD")
    # 该说话人更像某个已知成员且分差达此值时，不算「我」。
    owner_member_priority_margin: float = Field(
        default=0.03, ge=0, le=1, alias="OWNER_MEMBER_PRIORITY_MARGIN"
    )
    # 太短的说话人不算「我」。
    owner_min_speech_seconds: float = Field(default=10, ge=0, alias="OWNER_MIN_SPEECH_SECONDS")
    # 录入本人声纹所需的最少有效语音。
    owner_min_enroll_seconds: float = Field(default=20, ge=0, alias="OWNER_MIN_ENROLL_SECONDS")
    # 录入成功后自动回填的历史会议场次。
    owner_backfill_limit: int = Field(default=2, ge=0, le=50, alias="OWNER_BACKFILL_LIMIT")

    # --- R-P2-5 批量上传 -----------------------------------------------------
    batch_upload_enabled: bool = Field(default=True, alias="BATCH_UPLOAD_ENABLED")
    batch_max_files: int = Field(default=20, ge=1, le=200, alias="BATCH_MAX_FILES")

    # --- R-P2-6 高置信自动归类 -----------------------------------------------
    # 关闭后回到 v1.2 行为：只给建议，不自动移动。
    auto_project_assign_enabled: bool = Field(default=True, alias="AUTO_PROJECT_ASSIGN_ENABLED")
    auto_assign_threshold: float = Field(default=0.75, ge=0, le=1, alias="AUTO_ASSIGN_THRESHOLD")

    # --- R-P2.1 关键结论重构 -------------------------------------------------
    # 内容级去重（互斥与证据级去重始终生效）；关闭只保留证据级。
    conclusion_dedupe_enabled: bool = Field(default=True, alias="CONCLUSION_DEDUPE_ENABLED")
    # 字符二元组 Jaccard 阈值（RK-1 可调）。
    conclusion_dedupe_jaccard: float = Field(
        default=0.6, ge=0, le=1, alias="CONCLUSION_DEDUPE_JACCARD"
    )
    # 二元组包含度阈值（「基本包含」即判重）。
    conclusion_dedupe_containment: float = Field(
        default=0.6, ge=0, le=1, alias="CONCLUSION_DEDUPE_CONTAINMENT"
    )
    # 读取时说话人归属；关闭后一律显示「未标注说话人」。
    speaker_attribution_enabled: bool = Field(default=True, alias="SPEAKER_ATTRIBUTION_ENABLED")
    # --- R-P2.1 M2：待跟进跨会议闭环 ---------------------------------------
    # 关闭后报告写库不再 upsert 待跟进；列表与横幅显示空（不删已存数据）。
    followups_enabled: bool = Field(default=True, alias="FOLLOWUPS_ENABLED")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(default=None, alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    whisper_model: str = Field(default="small", alias="WHISPER_MODEL")
    whisper_device: str = Field(default="cpu", alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="int8", alias="WHISPER_COMPUTE_TYPE")
    speaker_recognition_enabled: bool = Field(default=True, alias="SPEAKER_RECOGNITION_ENABLED")
    speaker_model: str = Field(default="chinese", alias="SPEAKER_MODEL")
    speaker_match_threshold: float = Field(default=0.72, ge=0, le=1, alias="SPEAKER_MATCH_THRESHOLD")
    speaker_match_margin: float = Field(default=0.05, ge=0, le=1, alias="SPEAKER_MATCH_MARGIN")
    # 同场过度切分的合并阈值。2026-09-21 真实录音回归后由 0.78 上调到 0.85：
    # 实测「不同的人」的簇相似度可达 0.789（马宁/胡泊），0.78 会把两个人合成一个。
    speaker_intra_merge_threshold: float = Field(
        default=0.85, ge=0, le=1, alias="SPEAKER_INTRA_MERGE_THRESHOLD"
    )
    max_upload_mb: int = Field(default=300, gt=0, alias="MAX_UPLOAD_MB")
    max_audio_minutes: float = Field(default=240, gt=0, alias="MAX_AUDIO_MINUTES")
    processing_timeout_seconds: float = Field(default=21600, gt=0, alias="PROCESSING_TIMEOUT_SECONDS")
    task_retention_minutes: float = Field(default=30, gt=0, alias="TASK_RETENTION_MINUTES")
    rate_limit_per_hour: int = Field(default=10, gt=0, alias="RATE_LIMIT_PER_HOUR")
    daily_task_limit: int = Field(default=30, gt=0, alias="DAILY_TASK_LIMIT")
    queue_max: int = Field(default=5, gt=0, alias="QUEUE_MAX")
    database_path: str = Field(default="data/meeting-review.db", alias="DATABASE_PATH")
    llm_max_retries: int = Field(default=2, alias="LLM_MAX_RETRIES")
    # R-P1-7 ②：分块分析的并发上限，避免 4 小时会议无条件并发数十次调用
    llm_max_concurrency: int = Field(default=4, ge=1, le=16, alias="LLM_MAX_CONCURRENCY")
    transcript_chunk_chars: int = Field(default=6000, alias="TRANSCRIPT_CHUNK_CHARS")
    # 单价留空时只落 token、不换算金额，避免把价格硬编码进代码或数据库。
    llm_price_prompt_per_1k: Optional[float] = Field(default=None, ge=0, alias="LLM_PRICE_PROMPT_PER_1K")
    llm_price_completion_per_1k: Optional[float] = Field(
        default=None, ge=0, alias="LLM_PRICE_COMPLETION_PER_1K"
    )

    # --- 交付物：图片纪要（R-P1.5-1）----------------------------------------
    # 版式模板名（模板与数据分离；改版式只改模板文件，不改代码）。
    image_minutes_template: str = Field(default="card_v1", alias="IMAGE_MINUTES_TEMPLATE")
    # PDF 渲染器：auto（Playwright 优先，回退本机 Chrome）/ playwright / chrome / none
    pdf_renderer: Literal["auto", "playwright", "chrome", "none"] = Field(
        default="auto", alias="PDF_RENDERER"
    )
    # 上传同意协议版本（R-P1.5-6）。文案变更时递增，旧记录不被覆盖。
    consent_version: str = Field(default="v1", alias="CONSENT_VERSION")
    # 分享链接（R-P1.5-3）：固定 3 天有效；SHARE_BASE_URL 用于内网穿透/公网地址。
    share_ttl_hours: int = Field(default=72, gt=0, alias="SHARE_TTL_HOURS")
    share_base_url: Optional[str] = Field(default=None, alias="SHARE_BASE_URL")
    # --- Agent 层开关（PRD v1.1 §13.1）-------------------------------------
    # 默认 pipeline：不设置任何 AGENT_* 时，行为与 P0 完全一致（零行为变化）。
    agent_mode: Literal["pipeline", "shadow", "agent"] = Field(
        default="pipeline", alias="AGENT_MODE"
    )
    agent_max_steps: int = Field(default=20, ge=1, alias="AGENT_MAX_STEPS")
    agent_context_budget_tokens: int = Field(
        default=60000, ge=1000, alias="AGENT_CONTEXT_BUDGET_TOKENS"
    )
    agent_tools_enabled: str = Field(default="readonly", alias="AGENT_TOOLS_ENABLED")
    agent_write_tools_enabled: bool = Field(default=False, alias="AGENT_WRITE_TOOLS_ENABLED")
    agent_audit_enabled: bool = Field(default=True, alias="AGENT_AUDIT_ENABLED")
    agent_step_timeout_seconds: float = Field(
        default=120, gt=0, alias="AGENT_STEP_TIMEOUT_SECONDS"
    )
    agent_session_timeout_seconds: float = Field(
        default=1800, gt=0, alias="AGENT_SESSION_TIMEOUT_SECONDS"
    )
    # s17 完成判定：rule（默认，零成本）/ model（单独按 stage=goal_judge 计费）
    agent_goal_judge: Literal["rule", "model"] = Field(default="rule", alias="AGENT_GOAL_JUDGE")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    def parsed_team_tokens(self) -> Dict[str, str]:
        teams: Dict[str, str] = {}
        seen_tokens: set[str] = set()
        raw = self.team_tokens.get_secret_value() if self.team_tokens is not None else ""
        if not raw.strip():
            if self.auth_enabled:
                raise ValueError("AUTH_ENABLED=true 时必须配置 TEAM_TOKENS")
            return teams
        for entry in raw.split(","):
            if ":" not in entry:
                raise ValueError("TEAM_TOKENS 格式错误，应为 团队名:口令,团队名:口令")
            name, token = (part.strip() for part in entry.split(":", 1))
            if not name or not token or name in teams or token in seen_tokens:
                raise ValueError("TEAM_TOKENS 中团队名和口令必须非空且不能重复")
            problem = _token_strength_problem(name, token)
            if problem is not None:
                raise ValueError(problem)
            teams[name] = token
            seen_tokens.add(token)
        if not teams and self.auth_enabled:
            raise ValueError("AUTH_ENABLED=true 时需要至少配置一个团队口令")
        return teams

    def redacted_api_key_state(self) -> str:
        """只返回"已配置/未配置"，供启动日志使用，避免密钥原文进入日志。"""
        return "已配置" if self.openai_api_key else "未配置"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    try:
        settings.parsed_team_tokens()
    except ValueError as exc:
        raise RuntimeError(
            "TEAM_TOKENS 不合规，服务拒绝启动：{}\n"
            "修复指引：把每个团队的口令改成不少于 {} 位、且不是纯数字或纯字母的随机字符串，"
            "然后关闭 AUTH_ENABLED 或重新启动服务（口令只写在 .env，不要提交到仓库）。".format(
                exc, MIN_TEAM_TOKEN_LENGTH
            )
        ) from exc
    return settings

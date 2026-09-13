from functools import lru_cache
from typing import Dict, Optional

from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_host: str = Field(default="127.0.0.1", alias="APP_HOST")
    app_port: int = Field(default=8000, ge=1, le=65535, alias="APP_PORT")
    forwarded_allow_ips: str = Field(default="127.0.0.1", alias="FORWARDED_ALLOW_IPS")
    team_tokens: SecretStr = Field(min_length=1, alias="TEAM_TOKENS")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(default=None, alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    whisper_model: str = Field(default="small", alias="WHISPER_MODEL")
    whisper_device: str = Field(default="cpu", alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="int8", alias="WHISPER_COMPUTE_TYPE")
    max_upload_mb: int = Field(default=300, gt=0, alias="MAX_UPLOAD_MB")
    max_audio_minutes: float = Field(default=60, gt=0, alias="MAX_AUDIO_MINUTES")
    processing_timeout_seconds: float = Field(default=5400, gt=0, alias="PROCESSING_TIMEOUT_SECONDS")
    task_retention_minutes: float = Field(default=30, gt=0, alias="TASK_RETENTION_MINUTES")
    rate_limit_per_hour: int = Field(default=10, gt=0, alias="RATE_LIMIT_PER_HOUR")
    daily_task_limit: int = Field(default=30, gt=0, alias="DAILY_TASK_LIMIT")
    queue_max: int = Field(default=5, gt=0, alias="QUEUE_MAX")
    database_path: str = Field(default="data/meeting-review.db", alias="DATABASE_PATH")
    llm_max_retries: int = Field(default=2, alias="LLM_MAX_RETRIES")
    transcript_chunk_chars: int = Field(default=6000, alias="TRANSCRIPT_CHUNK_CHARS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    def parsed_team_tokens(self) -> Dict[str, str]:
        teams: Dict[str, str] = {}
        seen_tokens: set[str] = set()
        for entry in self.team_tokens.get_secret_value().split(","):
            if ":" not in entry:
                raise ValueError("TEAM_TOKENS 格式错误，应为 团队名:口令,团队名:口令")
            name, token = (part.strip() for part in entry.split(":", 1))
            if not name or not token or name in teams or token in seen_tokens:
                raise ValueError("TEAM_TOKENS 中团队名和口令必须非空且不能重复")
            teams[name] = token
            seen_tokens.add(token)
        if not teams:
            raise ValueError("TEAM_TOKENS 至少需要配置一个团队")
        return teams


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        missing_team_tokens = any(error.get("loc") == ("TEAM_TOKENS",) for error in exc.errors())
        if missing_team_tokens:
            raise RuntimeError("缺少必填环境变量 TEAM_TOKENS，服务拒绝启动") from exc
        raise

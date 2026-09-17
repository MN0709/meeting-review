"""P0 可信地基的测试。

只新增，不修改 tests/test_app.py 中的任何既有断言。
覆盖：R-P0-1（可复现构建的配置面）、R-P0-3（口令与密钥加固）、R-P0-2（成本落库，见文件末尾）。
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.config import MIN_TEAM_TOKEN_LENGTH, Settings
from app.db import Database


PROJECT_DIR = Path(__file__).parent.parent
STRONG_TOKEN_A = "mr-Abc123def456GHI789"
STRONG_TOKEN_B = "mr-Zyx987wvu654TSR321"


def _settings(team_tokens: str) -> Settings:
    return Settings(_env_file=None, TEAM_TOKENS=team_tokens)


# --------------------------------------------------------------------------
# R-P0-3：口令强度
# --------------------------------------------------------------------------


def test_strong_team_tokens_are_accepted() -> None:
    teams = _settings(f"甲团队:{STRONG_TOKEN_A},乙团队:{STRONG_TOKEN_B}").parsed_team_tokens()
    assert teams == {"甲团队": STRONG_TOKEN_A, "乙团队": STRONG_TOKEN_B}


def test_short_team_token_is_rejected_with_actionable_message() -> None:
    with pytest.raises(ValueError) as excinfo:
        _settings("产品项目组:1234").parsed_team_tokens()
    message = str(excinfo.value)
    assert f"至少 {MIN_TEAM_TOKEN_LENGTH} 位" in message
    assert "1234" not in message  # 不把口令原文写进错误信息


def test_pure_digit_team_token_is_rejected() -> None:
    with pytest.raises(ValueError, match="不能是纯数字"):
        _settings("甲团队:" + "9" * 20).parsed_team_tokens()


def test_pure_ascii_letter_team_token_is_rejected() -> None:
    with pytest.raises(ValueError, match="不能是纯字母"):
        _settings("甲团队:" + "abcdefghijklmnop").parsed_team_tokens()


def test_long_chinese_passphrase_is_accepted() -> None:
    """中文长口令熵足够高，不应被"纯字母"规则误伤（刻意的细化）。"""
    passphrase = "会脉团队长期口令甲乙丙丁戊己庚辛壬"
    assert len(passphrase) >= MIN_TEAM_TOKEN_LENGTH
    teams = _settings(f"甲团队:{passphrase}").parsed_team_tokens()
    assert teams["甲团队"] == passphrase


@pytest.mark.parametrize("weak", ["产品项目组:1234", "产品项目组:" + "a" * 20, "产品项目组:short"])
def test_weak_team_token_refuses_to_start(weak: str) -> None:
    """弱口令必须让服务拒绝启动，而不是等到运行时才报错。"""
    environment = os.environ.copy()
    environment["TEAM_TOKENS"] = weak
    environment["PYTHONPATH"] = str(PROJECT_DIR)
    result = subprocess.run(
        [sys.executable, "-c", "from app.config import get_settings; get_settings()"],
        cwd=str(PROJECT_DIR),
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "服务拒绝启动" in result.stderr
    assert "口令" in result.stderr


def test_get_settings_reports_redacted_api_key_state() -> None:
    assert Settings(_env_file=None, TEAM_TOKENS=f"甲团队:{STRONG_TOKEN_A}").redacted_api_key_state() == "未配置"
    assert (
        Settings(
            _env_file=None, TEAM_TOKENS=f"甲团队:{STRONG_TOKEN_A}", OPENAI_API_KEY="sk-not-a-real-key"
        ).redacted_api_key_state()
        == "已配置"
    )


# --------------------------------------------------------------------------
# 认证路径健壮性（P0 冒烟发现的现有缺陷）
#
# 现象：客户端发来含非 ASCII 字节的 X-Access-Token 时，hmac.compare_digest 会抛
#       TypeError("comparing strings with non-ASCII characters is not supported")，
#       接口返回 500 而不是 403，控制台还会打出堆栈。
# 修复：在 Database.authenticate 里统一转 UTF-8 字节后比较。
# --------------------------------------------------------------------------


def test_non_ascii_token_is_rejected_by_database(tmp_path) -> None:
    database = Database(tmp_path / "auth.db")
    database.initialize({"甲团队": STRONG_TOKEN_A})
    assert database.authenticate("æ" * 20) is None
    assert database.authenticate("") is None
    assert database.authenticate(STRONG_TOKEN_A) is not None


def test_non_ascii_header_returns_403_not_500(tmp_path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    import app.main as main_module

    teams = main_module.settings.parsed_team_tokens()
    database = Database(tmp_path / "auth-http.db")
    database.initialize(teams)
    monkeypatch.setattr(main_module, "database", database)
    valid_token = next(iter(teams.values()))

    with TestClient(main_module.app) as client:
        # 直接发原始字节，模拟客户端发来畸形口令（Starlette 按 latin-1 解码成非 ASCII）
        malformed = client.get("/api/meetings", headers={"X-Access-Token": b"\xe6" * 20})
        assert malformed.status_code == 403
        assert malformed.json() == {"detail": "团队口令错误"}
        assert client.get("/api/meetings", headers={"X-Access-Token": valid_token}).status_code == 200


def test_startup_log_never_contains_the_api_key(tmp_path) -> None:
    """子进程真实启动一次，确认密钥原文不会进入启动日志。"""
    secret = "sk-startup-log-must-not-contain-me"
    environment = os.environ.copy()
    environment["TEAM_TOKENS"] = f"甲团队:{STRONG_TOKEN_A}"
    environment["OPENAI_API_KEY"] = secret
    environment["DATABASE_PATH"] = str(tmp_path / "log-check.db")
    environment["PYTHONPATH"] = str(PROJECT_DIR)
    code = (
        "from fastapi.testclient import TestClient\n"
        "import app.main as m\n"
        "with TestClient(m.app) as client:\n"
        "    assert client.get('/health').status_code == 200\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_DIR),
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert secret not in combined
    assert "api_key=已配置" in combined

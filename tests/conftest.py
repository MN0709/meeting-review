"""测试进程的公共环境。

历史 385 项测试（v1.2 团队版）大多用「带/不带口令」验证跨工作区隔离。
R-P2-1 后生产默认是个人模式（AUTH_ENABLED=false，无鉴权），此处**把测试默认设为
AUTH_ENABLED=true**，让这些隔离测试继续有意义；个人模式由 `tests/test_personal_mode.py`
单独覆盖（monkeypatch settings.auth_enabled=False），两种模式都有回归。

注意：这两个环境变量必须在导入 `app.main` 之前设置，因此放在 conftest.py。
"""

import os

os.environ.setdefault("AUTH_ENABLED", "true")
os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

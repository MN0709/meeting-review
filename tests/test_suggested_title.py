"""D-027：AI 建议标题在用户未改动时自动生效。

只新增，不修改既有断言。
"""

import os

os.environ.setdefault("TEAM_TOKENS", "测试团队:test-access-token,另一团队:other-team-token")
os.environ.setdefault("DATABASE_PATH", "/tmp/meeting-review-pytest.db")

from app.db import Database
from app.models import DEFAULT_MEETING_TITLE

TEAM = "甲团队"
OTHER = "乙团队"
TOKEN = "test-access-token"
OTHER_TOKEN = "other-team-token"


def _seed(tmp_path, title: str, name: str = "titles.db"):
    path = tmp_path / name
    db = Database(path)
    db.initialize({TEAM: TOKEN, OTHER: OTHER_TOKEN})
    team_id = db.authenticate(TOKEN)
    other_id = db.authenticate(OTHER_TOKEN)
    db.create_meeting("m1", team_id, title, tmp_path / "a.m4a")
    return db, team_id, other_id


def _title(db, team_id) -> str:
    return next(item.title for item in db.list_meetings(team_id) if item.id == "m1")


def test_default_title_is_replaced_by_suggested(tmp_path) -> None:
    db, team_id, _ = _seed(tmp_path, DEFAULT_MEETING_TITLE)
    assert db.apply_suggested_title("m1", team_id, "支付修复与周四上线安排") is True
    assert _title(db, team_id) == "支付修复与周四上线安排"


def test_user_provided_title_is_never_overwritten(tmp_path) -> None:
    db, team_id, _ = _seed(tmp_path, "我自己起的标题", "user-title.db")
    assert db.apply_suggested_title("m1", team_id, "AI 建议标题") is False
    assert _title(db, team_id) == "我自己起的标题"


def test_title_edited_by_user_is_not_overwritten(tmp_path) -> None:
    db, team_id, _ = _seed(tmp_path, DEFAULT_MEETING_TITLE, "edited.db")
    assert db.update_meeting_title("m1", team_id, "用户改过的标题") is True
    assert db.apply_suggested_title("m1", team_id, "AI 建议标题") is False
    assert _title(db, team_id) == "用户改过的标题"


def test_empty_or_default_suggestion_changes_nothing(tmp_path) -> None:
    db, team_id, _ = _seed(tmp_path, DEFAULT_MEETING_TITLE, "empty.db")
    assert db.apply_suggested_title("m1", team_id, "   ") is False
    assert db.apply_suggested_title("m1", team_id, DEFAULT_MEETING_TITLE) is False
    assert _title(db, team_id) == DEFAULT_MEETING_TITLE


def test_suggested_title_is_team_scoped(tmp_path) -> None:
    db, team_id, other_id = _seed(tmp_path, DEFAULT_MEETING_TITLE, "scoped.db")
    # 另一个团队不能改到这场会议的标题
    assert db.apply_suggested_title("m1", other_id, "别的团队写的") is False
    assert _title(db, team_id) == DEFAULT_MEETING_TITLE

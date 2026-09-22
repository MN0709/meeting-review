"""阶段 16（M1）· 前端与交付物文案：不再出现「决策人：」，改为「谁说的」。"""

from pathlib import Path

ROOT = Path(__file__).parent.parent
HTML = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
IMAGE_MINUTES = (ROOT / "app" / "deliverables" / "image_minutes.py").read_text(encoding="utf-8")
SHARES = (ROOT / "app" / "shares.py").read_text(encoding="utf-8")


def test_no_decision_maker_label_anywhere() -> None:
    assert "决策人：" not in HTML
    assert "决策人：" not in IMAGE_MINUTES
    assert "决策人：" not in SHARES


def test_report_uses_speaker_prefix() -> None:
    assert "item.speaker?item.speaker+'：':''" in HTML


def test_three_class_note_present() -> None:
    assert "决策＝已经定了的" in HTML
    assert "待跟进＝还没定、悬着的" in HTML


def test_image_minutes_has_no_decision_maker() -> None:
    assert '"meta": None' in IMAGE_MINUTES or "meta\": None" in IMAGE_MINUTES
    assert "speaker" in IMAGE_MINUTES


def test_share_page_renamed_sections() -> None:
    assert "决策清单" not in SHARES
    assert "关键决策" in SHARES and "待跟进" in SHARES

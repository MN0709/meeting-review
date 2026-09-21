"""阶段 14（M3）· 前端契约测试：本人声纹引导、跳过后果、门控与不认错文案。"""

from pathlib import Path

HTML = (Path(__file__).parent.parent / "static" / "index.html").read_text(encoding="utf-8")


def test_owner_panel_and_recorder_exist() -> None:
    assert 'id="ownerPanel"' in HTML
    assert 'id="ownerRecorder"' in HTML
    assert 'id="ownerStartRecord"' in HTML
    assert 'id="ownerUpload"' in HTML
    assert "MediaRecorder" in HTML
    assert "getUserMedia" in HTML


def test_skip_dialog_warns_consequences() -> None:
    assert 'id="ownerSkipDialog"' in HTML
    assert "「我答应的任务」无法解锁" in HTML
    assert "之后可以在首页随时补录" in HTML


def test_frontend_calls_owner_endpoints() -> None:
    assert "apiFetch('/api/owner'" in HTML or 'apiFetch("/api/owner"' in HTML
    assert "/api/owner/voiceprint" in HTML
    assert "/api/owner/skip" in HTML


def test_my_tasks_is_gated_by_voiceprint_state() -> None:
    assert "owner_voiceprint_state" in HTML
    assert "ovState==='skipped'||ovState==='not_enrolled'" in HTML
    # 手动指认是兜底：已指定本人时即使没录声纹也要显示
    assert "!body.self_speaker_set&&(ovState==='skipped'" in HTML


def test_not_identified_wording_never_guesses() -> None:
    assert "本场未识别到你（不猜测）" in HTML

"""阶段 19：首页视觉重构的静态契约测试。"""

from pathlib import Path


ROOT = Path(__file__).parent.parent
HTML = (ROOT / "static" / "index.html").read_text(encoding="utf-8")


def test_home_has_real_shortcuts_and_recent_meetings() -> None:
    assert 'id="quickImport"' in HTML
    assert 'id="quickBatch"' in HTML
    assert 'id="quickSearch"' in HTML
    assert 'id="recentMeetings"' in HTML
    assert "async function loadRecentMeetings()" in HTML
    assert "apiFetch('/api/meetings'" in HTML
    assert "meetings.slice(0,3)" in HTML


def test_reference_only_inspires_layout_not_unsupported_features() -> None:
    assert "悬浮字幕" not in HTML
    assert ">Chat<" not in HTML
    assert "从录音里，提炼出脉络和新思考" in HTML
    assert "最近文件" in HTML


def test_mobile_uses_bottom_navigation() -> None:
    assert "position:fixed;z-index:40;top:auto;right:0;bottom:0;left:0" in HTML
    assert "grid-template-columns:repeat(4,1fr)" in HTML


def test_unclassified_upload_stays_available() -> None:
    assert "项目可留空" in HTML
    assert "submit.disabled=!hasFile||busy" in HTML


def test_upload_drop_zone_opens_file_picker_on_click() -> None:
    assert "$('dropZone').addEventListener('click'" in HTML
    assert "event.preventDefault();$('audioFile').click()" in HTML


def test_submit_nudges_consent_when_unchecked() -> None:
    assert "if(!$('consentCheck').checked){nudgeConsent();return}" in HTML
    assert "consent-row.consent-nudge" in HTML
    assert "@keyframes consent-nudge" in HTML


def test_failed_meeting_opens_status_and_retry_dialog() -> None:
    assert 'id="meetingStatusDialog"' in HTML
    assert "失败 · 可重试" in HTML
    assert "await showMeetingStatus(id,title" in HTML
    assert "retry?kind=report" in HTML
    assert "重试报告会重新调用 AI（产生费用）" in HTML


def test_missing_speakers_show_explanation_instead_of_hiding_section() -> None:
    assert "section.classList.remove('hidden')" in HTML
    assert "本场未生成可确认的说话人" in HTML
    assert "需使用原录音重新复盘" in HTML


def test_api_key_can_be_configured_after_entering_app() -> None:
    assert 'id="apiKeyBar"' in HTML
    assert 'id="apiKeyInput"' in HTML
    assert 'id="saveApiKey"' in HTML
    assert "window.webkit.messageHandlers.apiKey" in HTML
    assert "renderApiKeyStatus(body.api_key_configured,body.llm_model)" in HTML
    assert "if(!apiKeyConfigured)submit.disabled=true" in HTML

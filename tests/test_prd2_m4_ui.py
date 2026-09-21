"""阶段 15（M4）· 前端契约测试：批量上传、待归类入口、自动归类角标与撤销。"""

from pathlib import Path

HTML = (Path(__file__).parent.parent / "static" / "index.html").read_text(encoding="utf-8")


def test_upload_supports_multiple_files_and_batch_endpoint() -> None:
    assert 'id="audioFile" type="file" multiple' in HTML
    assert 'id="batchProgress"' in HTML
    assert "/api/reviews" in HTML
    assert "submitBatch" in HTML and "pollBatch" in HTML


def test_project_is_optional_for_upload() -> None:
    assert "项目可留空" in HTML
    # 旧的「必须选项目」限制已移除
    assert "!selectedFile||!$('projectSelect').value||activeTaskId" not in HTML


def test_pending_inbox_entry_exists() -> None:
    assert 'id="pendingCard"' in HTML
    assert "loadPendingCard" in HTML
    assert "未归类" in HTML


def test_auto_assignment_badge_and_undo_exist() -> None:
    assert "AI 自动归类" in HTML
    assert "undo-assignment" in HTML
    assert "ai-badge" in HTML


def test_batch_confirm_toolbar_exists() -> None:
    assert 'id="assignProjectSelect"' in HTML
    assert 'id="assignSelected"' in HTML
    assert 'id="acceptSuggestions"' in HTML
    assert "assign-batch" in HTML
    assert "accept_suggestions" in HTML

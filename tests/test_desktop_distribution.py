from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SWIFT = ROOT / "desktop/macos/Sources/MeetingReviewDesktop/main.swift"
BUILD_APP = ROOT / "desktop/macos/build_app.sh"
BUILD_BACKEND = ROOT / "desktop/macos/build_backend.sh"
BACKEND_ENTRY = ROOT / "desktop/macos/backend_entry.py"


def test_desktop_uses_bundled_backend_and_keychain() -> None:
    source = SWIFT.read_text(encoding="utf-8")

    assert "import Security" in source
    assert "SecItemCopyMatching" in source
    assert "SecItemAdd" in source
    assert 'configuration.userContentController.add(self, name: "apiKey")' in source
    assert 'payload["action"] as? String == "save"' in source
    assert 'https://api.deepseek.com/models' in source
    assert 'statusCode == 200' in source
    assert "guard ensureAPIKey()" not in source
    assert 'environment.removeValue(forKey: "OPENAI_API_KEY")' in source
    assert 'Backend/meeting-review-backend' in source
    assert 'Models/faster-whisper-small' in source
    assert 'Models/wespeaker-chinese' in source
    assert 'applicationSupportDirectory' in source
    assert 'MeetingReviewProjectPath' not in source
    assert '.venv/bin/python' not in source


def test_distribution_build_excludes_project_secrets_and_database() -> None:
    source = BUILD_APP.read_text(encoding="utf-8")

    assert 'CFBundleShortVersionString</key><string>0.2.1-beta' in source
    assert 'LSMinimumSystemVersion</key><string>13.0' in source
    assert 'LSArchitecturePriority' in source
    assert '.env' not in source
    assert 'data/meeting-review.db' not in source


def test_frozen_backend_has_multiprocessing_and_model_assets() -> None:
    entry = BACKEND_ENTRY.read_text(encoding="utf-8")
    build = BUILD_BACKEND.read_text(encoding="utf-8")

    assert "multiprocessing.freeze_support()" in entry
    assert "--collect-data silero_vad" in build
    assert "--hidden-import importlib_resources" in build
    assert "--collect-submodules wespeaker" in build

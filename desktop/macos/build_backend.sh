#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h:h}"
PYTHON="$PROJECT_DIR/.venv/bin/python"
DIST_DIR="$SCRIPT_DIR/.backend-dist"
WORK_DIR="$SCRIPT_DIR/.backend-build"

if ! "$PYTHON" -c 'import PyInstaller' >/dev/null 2>&1; then
  echo "缺少打包工具：$PYTHON -m pip install -r $SCRIPT_DIR/packaging-requirements.txt" >&2
  exit 1
fi

rm -rf "$DIST_DIR" "$WORK_DIR"
cd "$PROJECT_DIR"
"$PYTHON" -m PyInstaller \
  --noconfirm \
  --clean \
  --onedir \
  --name meeting-review-backend \
  --distpath "$DIST_DIR" \
  --workpath "$WORK_DIR" \
  --specpath "$WORK_DIR" \
  --paths "$PROJECT_DIR" \
  --additional-hooks-dir "$SCRIPT_DIR/hooks" \
  --add-data "$PROJECT_DIR/static:static" \
  --add-data "$PROJECT_DIR/app/deliverables/templates:app/deliverables/templates" \
  --collect-submodules app \
  --hidden-import faster_whisper \
  --collect-data faster_whisper \
  --collect-data ctranslate2 \
  --collect-submodules wespeaker \
  --collect-data wespeaker \
  --exclude-module wespeaker.frontend \
  --exclude-module s3prl \
  --exclude-module transformers \
  --collect-data silero_vad \
  --hidden-import importlib_resources \
  --hidden-import hdbscan \
  --hidden-import umap \
  --hidden-import torch \
  --hidden-import torchaudio \
  "$SCRIPT_DIR/backend_entry.py"

echo "$DIST_DIR/meeting-review-backend"

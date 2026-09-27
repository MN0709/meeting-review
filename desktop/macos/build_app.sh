#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h:h}"
BUILD_DIR="$SCRIPT_DIR/.build"
BACKEND_DIR="$SCRIPT_DIR/.backend-dist/meeting-review-backend"
APP_DIR="$PROJECT_DIR/dist/会脉.app"
CONTENTS_DIR="$APP_DIR/Contents"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
WHISPER_CACHE="$HOME/.cache/huggingface/hub/models--Systran--faster-whisper-small"
WESPEAKER_CACHE="$HOME/.wespeaker/chinese"
CODESIGN_IDENTITY="${CODESIGN_IDENTITY:--}"

if [[ ! -x "$BACKEND_DIR/meeting-review-backend" ]]; then
  "$SCRIPT_DIR/build_backend.sh"
fi
if [[ ! -d "$WHISPER_CACHE/snapshots" ]]; then
  echo "缺少 faster-whisper small 模型缓存：$WHISPER_CACHE" >&2
  exit 1
fi
if [[ ! -f "$WESPEAKER_CACHE/avg_model.pt" || ! -f "$WESPEAKER_CACHE/config.yaml" ]]; then
  echo "缺少 WeSpeaker chinese 模型缓存：$WESPEAKER_CACHE" >&2
  exit 1
fi

cd "$SCRIPT_DIR"
swift build -c release

rm -rf "$APP_DIR"
mkdir -p "$CONTENTS_DIR/MacOS" "$RESOURCES_DIR/Backend" "$RESOURCES_DIR/Models"
cp "$BUILD_DIR/release/MeetingReviewDesktop" "$CONTENTS_DIR/MacOS/会脉"
ditto "$BACKEND_DIR" "$RESOURCES_DIR/Backend"

WHISPER_SNAPSHOT="$(find "$WHISPER_CACHE/snapshots" -mindepth 1 -maxdepth 1 -type d | head -1)"
cp -RL "$WHISPER_SNAPSHOT" "$RESOURCES_DIR/Models/faster-whisper-small"
mkdir -p "$RESOURCES_DIR/Models/wespeaker-chinese"
cp "$WESPEAKER_CACHE/avg_model.pt" "$WESPEAKER_CACHE/config.yaml" "$RESOURCES_DIR/Models/wespeaker-chinese/"

ICONSET_DIR="$BUILD_DIR/AppIcon.iconset"
rm -rf "$ICONSET_DIR"
mkdir -p "$ICONSET_DIR"
qlmanage -t -s 1024 -o "$BUILD_DIR" "$SCRIPT_DIR/AppIcon.svg" >/dev/null 2>&1
ICON_SOURCE="$BUILD_DIR/AppIcon.svg.png"
for spec in "16 icon_16x16.png" "32 icon_16x16@2x.png" "32 icon_32x32.png" "64 icon_32x32@2x.png" "128 icon_128x128.png" "256 icon_128x128@2x.png" "256 icon_256x256.png" "512 icon_256x256@2x.png" "512 icon_512x512.png" "1024 icon_512x512@2x.png"; do
  size="${spec%% *}"
  name="${spec#* }"
  sips -z "$size" "$size" "$ICON_SOURCE" --out "$ICONSET_DIR/$name" >/dev/null
done
iconutil -c icns "$ICONSET_DIR" -o "$RESOURCES_DIR/AppIcon.icns"

cat > "$CONTENTS_DIR/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleDevelopmentRegion</key><string>zh_CN</string>
  <key>CFBundleDisplayName</key><string>会脉</string>
  <key>CFBundleExecutable</key><string>会脉</string>
  <key>CFBundleIdentifier</key><string>com.huimai.meetingreview</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundleName</key><string>会脉</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.2.1-beta</string>
  <key>CFBundleVersion</key><string>3</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSArchitecturePriority</key><array><string>arm64</string></array>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST

printf 'APPL????' > "$CONTENTS_DIR/PkgInfo"
codesign --force --deep --sign "$CODESIGN_IDENTITY" "$APP_DIR"
codesign --verify --deep --strict --verbose=2 "$APP_DIR"
echo "$APP_DIR"

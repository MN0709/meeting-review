#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h:h}"
APP_DIR="$PROJECT_DIR/dist/会脉.app"
DMG_DIR="$PROJECT_DIR/dist/dmg-root"
DMG_PATH="$PROJECT_DIR/dist/会脉-0.2.1-beta-arm64.dmg"

"$SCRIPT_DIR/build_app.sh"
rm -rf "$DMG_DIR" "$DMG_PATH" "$DMG_PATH.sha256"
mkdir -p "$DMG_DIR"
ditto "$APP_DIR" "$DMG_DIR/会脉.app"
ln -s /Applications "$DMG_DIR/Applications"
hdiutil create -quiet -volname "会脉" -srcfolder "$DMG_DIR" -ov -format UDZO "$DMG_PATH"
rm -rf "$DMG_DIR"
shasum -a 256 "$DMG_PATH" > "$DMG_PATH.sha256"
hdiutil verify "$DMG_PATH"
echo "$DMG_PATH"

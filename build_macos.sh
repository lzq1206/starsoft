#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VERSION="$("$PYTHON_BIN" -c 'from version import APP_VERSION; print(APP_VERSION)' 2>/dev/null || (cd "$ROOT_DIR" && "$PYTHON_BIN" -c 'from version import APP_VERSION; print(APP_VERSION)'))"
MACHINE="$(uname -m)"
case "$MACHINE" in
  arm64) ARCH="arm64" ;;
  x86_64) ARCH="x64" ;;
  *) echo "不支持的 macOS 架构：$MACHINE" >&2; exit 2 ;;
esac

DIST_DIR="$ROOT_DIR/build-macos-dist"
WORK_DIR="$ROOT_DIR/build-macos-work"
PACKAGE_DIR="$ROOT_DIR/build-macos-package"
SOLVER_DIR="$ROOT_DIR/build-macos-solver"
APP_PATH="$DIST_DIR/星点柔焦.app"
ZIP_PATH="$ROOT_DIR/releases/星点柔焦-v${VERSION}-macos-${ARCH}.zip"

"$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements.txt"
rm -rf "$DIST_DIR" "$WORK_DIR" "$PACKAGE_DIR" "$SOLVER_DIR"
mkdir -p "$DIST_DIR" "$WORK_DIR" "$PACKAGE_DIR/星点柔焦-v${VERSION}-macos-${ARCH}"
"$PYTHON_BIN" "$ROOT_DIR/scripts/prepare_astap.py" --output "$SOLVER_DIR" --platform "macos-${ARCH}"
"$PYTHON_BIN" -m PyInstaller \
  --noconfirm --clean --windowed --name "星点柔焦" \
  --distpath "$DIST_DIR" --workpath "$WORK_DIR" \
  --collect-all rawpy --collect-all sep --collect-all tifffile \
  --collect-all seiza --exclude-module astropy.visualization \
  --add-data "$ROOT_DIR/ui/index.html:ui" \
  --add-data "$ROOT_DIR/data/hyg_named_stars.csv:data" "$ROOT_DIR/app.py"

test -d "$APP_PATH"
mkdir -p "$APP_PATH/Contents/Resources"
cp -R "$SOLVER_DIR" "$APP_PATH/Contents/Resources/solver"
codesign --force -s - "$APP_PATH/Contents/Resources/solver/astap_cli"
cp -R "$APP_PATH" "$PACKAGE_DIR/星点柔焦-v${VERSION}-macos-${ARCH}/"
cp "$ROOT_DIR/README.md" "$PACKAGE_DIR/星点柔焦-v${VERSION}-macos-${ARCH}/"
cp "$ROOT_DIR/LICENSE" "$PACKAGE_DIR/星点柔焦-v${VERSION}-macos-${ARCH}/"
cp "$ROOT_DIR/licenses/Seiza-Apache-2.0.txt" "$PACKAGE_DIR/星点柔焦-v${VERSION}-macos-${ARCH}/"
cp "$ROOT_DIR/licenses/THIRD_PARTY_NOTICES.txt" "$PACKAGE_DIR/星点柔焦-v${VERSION}-macos-${ARCH}/"
cp "$ROOT_DIR/licenses/CC-BY-SA-4.0.txt" "$PACKAGE_DIR/星点柔焦-v${VERSION}-macos-${ARCH}/"
cat > "$PACKAGE_DIR/星点柔焦-v${VERSION}-macos-${ARCH}/首次运行说明.txt" <<'EOF'
首次打开时，在 Finder 中按住 Control 并点按“星点柔焦.app”，选择“打开”。
该应用由 GitHub Actions 构建，目前未使用 Apple Developer ID 签名或公证。
程序启动后会打开本地网页界面；也可从界面进入 GitHub Pages 网页版。
程序包已含本机 ASTAP 解算器与 Gaia 派生的 W08 全天天亮星索引（约 G=8 等），无需另行安装解算器或联网查询。
图像、星点坐标与星表匹配均在本机处理；索引提供 0.1 等精度的 G 亮度，不包含 Gaia source_id 或 BP-RP。
EOF
mkdir -p "$ROOT_DIR/releases"
ditto -c -k --sequesterRsrc --keepParent "$PACKAGE_DIR/星点柔焦-v${VERSION}-macos-${ARCH}" "$ZIP_PATH"
shasum -a 256 "$ZIP_PATH" > "$ZIP_PATH.sha256"
echo "Built $ZIP_PATH"

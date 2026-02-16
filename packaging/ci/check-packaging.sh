#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

fail() {
  echo "[packaging-check] ERROR: $*" >&2
  exit 1
}

warn() {
  echo "[packaging-check] WARN: $*" >&2
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || fail "Missing required file: $path"
}

echo "[packaging-check] Validating required packaging files"
require_file packaging/debian/debian/control
require_file packaging/debian/debian/changelog
require_file packaging/debian/debian/rules
require_file packaging/debian/debian/source/format
require_file packaging/arch/PKGBUILD
require_file packaging/fedora/desktop-assistant.spec
require_file packaging/snap/snapcraft.yaml
require_file packaging/flatpak/org.desktopassistant.App.yml
require_file packaging/flatpak/org.desktopassistant.App.desktop
require_file packaging/flatpak/org.desktopassistant.App.metainfo.xml

if [[ ! -x packaging/debian/debian/rules ]]; then
  fail "packaging/debian/debian/rules must be executable"
fi

echo "[packaging-check] Debian metadata checks"
if command -v dpkg-parsechangelog >/dev/null 2>&1; then
  dpkg-parsechangelog -l packaging/debian/debian/changelog >/dev/null
else
  warn "dpkg-parsechangelog not found; skipping Debian changelog parsing"
fi

echo "[packaging-check] Arch metadata checks"
bash -n packaging/arch/PKGBUILD

echo "[packaging-check] Fedora spec checks"
if command -v rpmspec >/dev/null 2>&1; then
  rpmspec -P packaging/fedora/desktop-assistant.spec >/dev/null
else
  warn "rpmspec not found; skipping Fedora spec parse"
fi

echo "[packaging-check] Snap + Flatpak YAML lint"
if command -v yamllint >/dev/null 2>&1; then
  yamllint packaging/snap/snapcraft.yaml packaging/flatpak/org.desktopassistant.App.yml
else
  warn "yamllint not found; skipping YAML lint"
fi

echo "[packaging-check] Flatpak manifest parse"
if command -v flatpak-builder >/dev/null 2>&1; then
  flatpak-builder --show-manifest packaging/flatpak/org.desktopassistant.App.yml >/dev/null
else
  warn "flatpak-builder not found; skipping manifest parse"
fi

echo "[packaging-check] Snap manifest semantic checks"
if command -v python3 >/dev/null 2>&1; then
  python3 - <<'PY'
import pathlib
import sys

snap_path = pathlib.Path("packaging/snap/snapcraft.yaml")
text = snap_path.read_text(encoding="utf-8")
for key in ("name:", "base:", "apps:", "parts:"):
    if key not in text:
        print(f"[packaging-check] ERROR: snapcraft.yaml missing '{key}'", file=sys.stderr)
        sys.exit(1)
print("[packaging-check] snapcraft.yaml contains required top-level sections")
PY
else
  warn "python3 not found; skipping Snap semantic checks"
fi

echo "[packaging-check] Done"

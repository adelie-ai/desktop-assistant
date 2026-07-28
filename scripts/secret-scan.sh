#!/usr/bin/env bash
# The secret-scan step of the gate (#811): gitleaks over the CHECKED-OUT
# FILES - not git history. The key that prompted this issue was never
# committed, so a history-based scan (`gitleaks git`, or the older `gitleaks
# detect`) would have reported clean for the entire time it sat in .env at
# mode 0644. `dir` is gitleaks' filesystem-walk subcommand: it reads whatever
# is on disk, tracked or not, gitignored or not - exactly the case this gate
# exists to catch.
#
# Like scripts/audit.sh, nothing here exits 0 unless a scan actually ran and
# came back clean. An exit status alone is not proof of that (#706's failure
# mode): a fatal gitleaks error - bad config, bad path - exits non-zero and
# writes no report at all, so report-existence is the signal this script
# trusts, not the exit code by itself.
#
# Usage: scripts/secret-scan.sh [DIR]
#   DIR defaults to the repository root. Tests pass a throwaway fixture
#   directory so they can prove detection without touching the real tree.
set -euo pipefail

# Pinned so the rule set - bundled in the gitleaks binary, not fetched per
# run - is the same on every machine that runs this gate (AGENTS.base.md,
# rule 6.2). Checked against the current release before pinning (rule 6.1),
# not remembered: https://github.com/gitleaks/gitleaks/releases/tag/v8.30.1
GITLEAKS_VERSION='8.30.1'

loud() { # loud <headline> <line>...
    local headline="$1"
    shift
    {
        printf '\n'
        printf '  %s\n' '======================================================================'
        printf '  %s\n' "$headline"
        printf '  %s\n' '======================================================================'
        printf '  %s\n' "$@"
        printf '\n'
    } >&2
}

die_loud() {
    loud "$@"
    exit 1
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:-$repo_root}"

if ! command -v gitleaks >/dev/null 2>&1; then
    die_loud 'SECRET SCAN DID NOT RUN: gitleaks is not installed' \
        'The gate promises a working-tree secret scan, so a missing scanner' \
        'fails it instead of quietly passing it.' \
        '' \
        "Install gitleaks ${GITLEAKS_VERSION}:" \
        '    pacman -S gitleaks        # Arch/CachyOS' \
        '    brew install gitleaks     # macOS/Linuxbrew' \
        "    https://github.com/gitleaks/gitleaks/releases/tag/v${GITLEAKS_VERSION}"
fi

installed_version="$(gitleaks version 2>/dev/null | tr -d '[:space:]')"
if [ "$installed_version" != "$GITLEAKS_VERSION" ]; then
    die_loud 'SECRET SCAN DID NOT RUN: gitleaks version does not match the pin' \
        "This gate pins gitleaks ${GITLEAKS_VERSION} so the bundled rule set is" \
        'identical on every machine that runs it (AGENTS.base.md, rule 6.2).' \
        '' \
        "    found:  ${installed_version:-none}" \
        "    pinned: ${GITLEAKS_VERSION}" \
        '' \
        'Install the pinned version, or bump GITLEAKS_VERSION in this script' \
        'deliberately after checking the new release notes (AGENTS.base.md, rule 6.1).'
fi

config="$repo_root/.gitleaks.toml"
ignore="$repo_root/.gitleaksignore"
[ -f "$config" ] || die_loud 'SECRET SCAN DID NOT RUN: missing .gitleaks.toml' \
    "Expected the scan config at $config."
[ -f "$ignore" ] || die_loud 'SECRET SCAN DID NOT RUN: missing .gitleaksignore' \
    "Expected the reviewed-findings baseline at $ignore."

report="$(mktemp)"
scan_out="$(mktemp)"
trap 'rm -f "$report" "$scan_out"' EXIT

# Scanned as `dir .` from inside the target, not `dir <absolute-path>`: gitleaks
# reports each finding's "File" using whatever path style the target argument
# used, and .gitleaksignore fingerprints are repo-relative
# (file:rule-id:line). An absolute target would make every fingerprint in
# .gitleaksignore silently stop matching - not a hypothetical, this is exactly
# how the first version of this script failed its own clean-tree test.
scan_status=0
( cd "$target" && gitleaks dir . \
    --config "$config" \
    --gitleaks-ignore-path "$ignore" \
    --report-format json \
    --report-path "$report" \
    --redact \
    --no-banner \
    --no-color \
    --exit-code 1 \
    -v \
    >"$scan_out" 2>&1 ) || scan_status=$?

# A report is proof the scan ran; an exit status alone is not. Verified
# against the real binary: a fatal error (bad config, bad path) exits
# non-zero WITHOUT writing a report, so report-existence is a reliable signal
# and not an assumption.
if [ ! -s "$report" ] || ! grep -q '^\[' "$report"; then
    die_loud 'SECRET SCAN DID NOT RUN: gitleaks produced no report' \
        "gitleaks exited ${scan_status} without emitting a scan report, so nothing" \
        'was checked.' \
        '' \
        'gitleaks said:' \
        "$(sed 's/^/    /' "$scan_out" | head -40)"
fi

# The findings, read back out of the JSON report rather than trusted from the
# verbose stdout above: `-v` is gitleaks' own formatting and is not guaranteed
# present (e.g. under test, with a mocked binary), while the report file is
# the thing this whole step exists to insist on.
findings_summary() {
    local files rules
    files="$(grep -oE '"File": *"[^"]*"' "$report" | sed -E 's/.*: *"(.*)"$/\1/')"
    rules="$(grep -oE '"RuleID": *"[^"]*"' "$report" | sed -E 's/.*: *"(.*)"$/\1/')"
    paste -d'\t' <(printf '%s\n' "$rules") <(printf '%s\n' "$files") \
        | awk -F'\t' '{printf "    %s  (rule: %s)\n", $2, $1}'
}

finding_count="$(grep -c '"RuleID"' "$report" || true)"

if [ "$scan_status" -ne 0 ] || [ "$finding_count" -gt 0 ]; then
    die_loud 'SECRETS FOUND IN THE WORKING TREE - hard gate failure' \
        'gitleaks matched one or more secrets in checked-out files. Rotate and' \
        'remove any real credential now - deleting the file alone is not enough' \
        'if the value may already be cached or shipped elsewhere. If this is a' \
        'reviewed false positive, add its fingerprint to .gitleaksignore with a' \
        'comment explaining why (AGENTS.md, "Secret scanning") - never delete this' \
        'step or weaken the rule set to make it pass.' \
        '' \
        "$(findings_summary)" \
        '' \
        "$(cat "$scan_out")"
fi

bytes_scanned="$(grep -oE 'scanned ~[0-9]+ bytes[^"]*' "$scan_out" | head -1 || true)"
printf 'secret-scan: clean%s\n' "${bytes_scanned:+ - $bytes_scanned}"

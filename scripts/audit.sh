#!/usr/bin/env bash
# The dependency-scan step of the gate: a RustSec advisory scan of Cargo.lock.
# Run as part of `just check`, before anything compiles, because build scripts
# execute at first build (AGENTS.md, "Security review").
#
# This is a script rather than a one-line recipe because the interesting part
# is the failure classification. A scan that exits 0 because cargo-audit is not
# installed, or because the advisory database could not be fetched, reads in the
# log exactly like "no advisories" - which is the bug #706 was filed about. So
# nothing here exits 0 unless a scan actually ran and came back clean.
#
# Offline: the step fails. If you accept a cached, possibly stale advisory
# database, opt in explicitly with ADELE_AUDIT_ALLOW_STALE=1 and the run says
# so, loudly, in its output.
set -euo pipefail

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

# First 40 lines of the given files, indented for the banner. Never fails: it
# only ever decorates a message that is already on its way out.
indent_files() {
    sed 's/^/    /' "$@" 2>/dev/null | head -40 || true
}

if ! command -v cargo-audit >/dev/null 2>&1; then
    die_loud 'DEPENDENCY SCAN DID NOT RUN: cargo-audit is not installed' \
        'The gate promises a RustSec advisory scan of Cargo.lock, so a missing' \
        'scanner fails it instead of quietly passing it.' \
        '' \
        'Install it, then re-run:' \
        '    cargo install cargo-audit --locked'
fi

audit_args=(--json)
if [ -n "${ADELE_AUDIT_ALLOW_STALE:-}" ]; then
    advisory_db="${CARGO_HOME:-$HOME/.cargo}/advisory-db"
    db_age='unknown age'
    if [ -d "$advisory_db" ]; then
        db_age="last changed $(date -r "$advisory_db" '+%Y-%m-%d %H:%M' 2>/dev/null || echo 'unknown')"
    fi
    loud 'DEPENDENCY SCAN IS USING A STALE ADVISORY DATABASE (opt-in)' \
        'ADELE_AUDIT_ALLOW_STALE is set, so this run does NOT fetch new' \
        'advisories. Anything published since the cached database was last' \
        'updated will not be reported.' \
        '' \
        "    database: $advisory_db ($db_age)" \
        '' \
        'Re-run without ADELE_AUDIT_ALLOW_STALE once you are back online.'
    audit_args+=(--no-fetch --stale)
fi

report="$(mktemp)"
scan_err="$(mktemp)"
trap 'rm -f "$report" "$scan_err"' EXIT

scan_status=0
cargo audit "${audit_args[@]}" >"$report" 2>"$scan_err" || scan_status=$?

# A report is proof the scan ran; an exit status alone is not.
if ! grep -q '"vulnerabilities"' "$report"; then
    die_loud 'DEPENDENCY SCAN DID NOT RUN: cargo-audit produced no report' \
        "cargo audit exited ${scan_status} without emitting a scan report, so nothing" \
        'was checked. The usual cause is that the RustSec advisory database could' \
        'not be fetched: no network, or a proxy in front of github.com.' \
        '' \
        'cargo audit said:' \
        "$(indent_files "$scan_err" "$report")" \
        '' \
        'Offline, and willing to scan against the cached advisory database?' \
        'Re-run with ADELE_AUDIT_ALLOW_STALE=1 (it is loud about what it skipped).'
fi

# Re-print the scan readably for the gate log. The database was just fetched, so
# this is offline and cheap; it is informational, never the pass/fail signal.
print_readable_report() {
    cargo audit --no-fetch --stale >&2 || true
}

if [ "$scan_status" -ne 0 ]; then
    print_readable_report
    die_loud 'DEPENDENCY ADVISORIES FOUND - hard gate failure' \
        "cargo audit exited ${scan_status}. High/critical advisories are blockers: patch" \
        'them in this change, prove the path is unreachable and say why, or file a' \
        'tracked follow-up and reference it from the change (AGENTS.md,' \
        '"Security review"). Never ship past one silently.'
fi

# A configured ignore list makes cargo-audit report "found":false and exit 0
# for an advisory that is really there, which is the #706 shape again: the gate
# says clean because it was told not to look. Suppression is a legitimate,
# reviewed decision, so it does not fail the step - but it is never silent.
suppressed=''
ignored="$(grep -o '"ignore":\[[^]]*\]' "$report" | head -1 | sed 's/^"ignore":\[//; s/\]$//' || true)"
if [ -n "$ignored" ]; then
    loud 'DEPENDENCY SCAN: advisories suppressed by configuration' \
        'cargo-audit was configured to ignore these advisories, so the result' \
        'below does not account for them:' \
        '' \
        "    $ignored" \
        '' \
        'They come from an `[advisories] ignore` list in an audit.toml. Drop the' \
        'entry, or say in the PR why the advisory cannot reach this code.'
    suppressed=', with advisories suppressed by configuration'
fi

# Informational advisories (unmaintained / unsound / yanked) do not fail the
# scan by default, but they are review items, so they do not slip past silently.
outcome='clean'
if ! grep -q '"warnings":{}' "$report"; then
    print_readable_report
    loud 'DEPENDENCY SCAN: informational advisories reported' \
        'No vulnerability blocked the gate, but cargo audit flagged unmaintained,' \
        'unsound or yanked crates (listed above). Treat them as review items.'
    # Not "clean": the summary line must not contradict the banner above it.
    outcome='no vulnerabilities, informational advisories reported'
fi

field() { # field <json-key> - best-effort, cosmetic only
    local value
    value="$(grep -o "\"$1\":[0-9]*" "$report" | head -1 | cut -d: -f2 || true)"
    printf '%s' "${value:-?}"
}
printf 'audit: %s%s - %s dependencies scanned against %s advisories\n' \
    "$outcome" "$suppressed" "$(field dependency-count)" "$(field advisory-count)"

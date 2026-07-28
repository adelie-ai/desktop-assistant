#!/usr/bin/env bash
# Acceptance criteria for the working-tree secret-scan step of the gate (#811).
#
# The key that produced this finding was never committed, so a scanner that
# only walks git history would have reported clean the entire time it was
# exposed. scripts/secret-scan.sh must run gitleaks' filesystem walk (`dir`),
# not its history walk (`git`) - the dedicated test below fails red against a
# script that scans history instead of the checkout.
#
# Wrapper-logic tests (missing tool, version drift, a scan that produced no
# report) use a fake `gitleaks` on PATH, the same mechanism
# audit-gate.test.sh uses for cargo-audit, so the failure classification is
# exercised deterministically without a live scan. The two tests that matter
# most - detecting a real secret shape, and not flagging the clean tree - run
# the real, pinned gitleaks binary: a mocked tool can only prove the
# wrapper's plumbing, never that detection actually works. A scan that scans
# nothing passes every mocked test too, which is the vacuous version of this
# fix.
set -euo pipefail
# shellcheck source=scripts/tests/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SECRET_SCAN_SH="$SCRIPT_TESTS_ROOT/scripts/secret-scan.sh"
PINNED_GITLEAKS_VERSION='8.30.1'

with_fake_gitleaks() {
    mkdir -p "$TEST_TMP/bin"
    cp "$SCRIPT_TESTS_FIXTURES/fake-gitleaks.sh" "$TEST_TMP/bin/gitleaks"
    chmod +x "$TEST_TMP/bin/gitleaks"
    export PATH="$TEST_TMP/bin:$PATH"
    export FAKE_GITLEAKS_LOG="$TEST_TMP/gitleaks.log"
    : >"$FAKE_GITLEAKS_LOG"
}

# A PATH with no gitleaks reachable anywhere, regardless of how THIS machine
# happens to have installed it (~/.cargo/bin, ~/.local/bin, ~/go/bin, a
# distro package in /usr/bin or /usr/local/bin). A naive
# "$TEST_TMP/empty:/usr/bin:/bin" still resolves gitleaks on any machine that
# followed this repo's own `pacman -S gitleaks` instructions, so it never
# actually exercises the missing-binary path it is named for - it happens to
# pass today only because this particular box does not have it there.
# Symlinks in only the coreutils scripts/secret-scan.sh itself needs, so
# nothing outside that directory is ever consulted.
_hermetic_path_without_gitleaks() {
    local bin="$TEST_TMP/hermetic-bin" tool
    mkdir -p "$bin"
    for tool in bash env dirname mktemp grep sed paste awk cat tr rm; do
        ln -sf "$(command -v "$tool")" "$bin/$tool"
    done
    printf '%s' "$bin"
}

CLEAN_REPORT='[]'
LEAK_REPORT='[{"RuleID":"openai-api-key","File":".env","StartLine":4,"Fingerprint":".env:openai-api-key:4","Secret":"REDACTED","Match":"REDACTED"}]'
# gitleaks version, verbatim, from the CachyOS/Arch `pacman -S gitleaks`
# package (confirmed 8.30.1-1.1 - the pinned version - on both plain Arch and
# CachyOS): the distro build does not set the version ldflag, so even the
# exact pinned release reports this instead of a real version number.
UNVERSIONED_OUTPUT='version is set by build process'

# --- wrapper-logic tests (mocked gitleaks) -----------------------------------

secret_scan_passes_on_a_clean_report() {
    with_fake_gitleaks
    export FAKE_GITLEAKS_REPORT="$CLEAN_REPORT" FAKE_GITLEAKS_STATUS=0
    run_cmd "$SECRET_SCAN_SH"
    assert_eq 0 "$RUN_STATUS" "a clean scan must pass the step: $RUN_ERR"
}

secret_scan_fails_when_gitleaks_reports_a_leak() {
    with_fake_gitleaks
    export FAKE_GITLEAKS_REPORT="$LEAK_REPORT" FAKE_GITLEAKS_STATUS=1
    run_cmd "$SECRET_SCAN_SH"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a reported leak must fail the step'
    assert_contains "$RUN_ERR" '.env' 'failure names the offending file'
    assert_contains "$RUN_ERR" 'openai-api-key' 'failure names the rule that matched'
}

secret_scan_fails_loudly_when_gitleaks_is_not_installed() {
    run_cmd env PATH="$(_hermetic_path_without_gitleaks)" "$SECRET_SCAN_SH"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a missing gitleaks must fail the step'
    assert_contains "$RUN_ERR" 'SECRET SCAN DID NOT RUN: gitleaks is not installed' \
        'names the missing-binary failure specifically, not any message that merely mentions gitleaks'
    assert_not_contains "$RUN_OUT" 'clean' 'must not claim a clean scan'
}

secret_scan_fails_when_the_installed_gitleaks_version_does_not_match_the_pin() {
    with_fake_gitleaks
    export FAKE_GITLEAKS_VERSION='7.0.0'
    run_cmd "$SECRET_SCAN_SH"
    [ "$RUN_STATUS" -ne 0 ] || fail 'an unpinned gitleaks version must fail the step'
    assert_contains "$RUN_ERR" 'version does not match the pin' 'names the mismatch failure specifically'
    assert_contains "$RUN_ERR" '7.0.0' 'names the version found'
    assert_contains "$RUN_ERR" "$PINNED_GITLEAKS_VERSION" 'names the pinned version'
    assert_not_contains "$RUN_ERR" 'no parseable version' \
        'a real (if wrong) version number is not the same failure as an unparseable one'
}

secret_scan_fails_when_gitleaks_reports_no_parseable_version() {
    # The CachyOS/Arch package's actual failure mode (confirmed against the
    # real package on both distros): `gitleaks version` prints a sentence,
    # not a version number, for the exact pinned release. This is a
    # packaging defect, not version drift, and needs its own diagnosis - a
    # gate that calls this "version does not match the pin" points the
    # reader at the wrong fix (upgrade/downgrade gitleaks) instead of the
    # right one (install from the release tarball, or opt in).
    with_fake_gitleaks
    export FAKE_GITLEAKS_VERSION="$UNVERSIONED_OUTPUT"
    run_cmd "$SECRET_SCAN_SH"
    [ "$RUN_STATUS" -ne 0 ] || fail 'an unparseable version must fail the step'
    assert_contains "$RUN_ERR" 'no parseable version' 'names the real outcome, distinct from a version mismatch'
    assert_contains "$RUN_ERR" 'packaging defect' 'explains this is a packaging bug, not drift'
    assert_not_contains "$RUN_ERR" 'does not match the pin' \
        'must not be phrased as a mismatch - there is no version to compare'
}

secret_scan_allows_an_unpinned_gitleaks_version_with_explicit_opt_in() {
    # Mirrors ADELE_AUDIT_ALLOW_STALE's shape in scripts/audit.sh: the exact
    # pin stays the default, but nobody should have to edit the gate to
    # unblock unrelated work.
    with_fake_gitleaks
    export FAKE_GITLEAKS_VERSION='7.0.0'
    export FAKE_GITLEAKS_REPORT="$CLEAN_REPORT" FAKE_GITLEAKS_STATUS=0
    export ADELE_SECRET_SCAN_ALLOW_UNPINNED=1
    run_cmd "$SECRET_SCAN_SH"
    assert_eq 0 "$RUN_STATUS" "explicit opt-in must still complete: $RUN_ERR"
    assert_contains "$RUN_ERR" 'ALLOW_UNPINNED' 'opt-in is loud about what it did'
    assert_contains "$RUN_ERR" '7.0.0' 'names the version actually used'
}

secret_scan_allows_an_unparseable_gitleaks_version_with_explicit_opt_in() {
    with_fake_gitleaks
    export FAKE_GITLEAKS_VERSION="$UNVERSIONED_OUTPUT"
    export FAKE_GITLEAKS_REPORT="$CLEAN_REPORT" FAKE_GITLEAKS_STATUS=0
    export ADELE_SECRET_SCAN_ALLOW_UNPINNED=1
    run_cmd "$SECRET_SCAN_SH"
    assert_eq 0 "$RUN_STATUS" "explicit opt-in must still complete even without a parseable version: $RUN_ERR"
    assert_contains "$RUN_ERR" 'ALLOW_UNPINNED' 'opt-in is loud about what it did'
}

secret_scan_fails_when_gitleaks_produces_no_report() {
    with_fake_gitleaks
    # Mirrors #706's failure mode for cargo-audit: an exit status alone is not
    # proof a scan happened. A fatal gitleaks error (bad config, bad path)
    # exits non-zero and writes no report at all - verified empirically
    # against the real binary, not assumed.
    export FAKE_GITLEAKS_NO_REPORT=1 FAKE_GITLEAKS_STATUS=1
    export FAKE_GITLEAKS_STDERR='FTL unable to load gitleaks config'
    run_cmd "$SECRET_SCAN_SH"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a scan that produced no report must fail the step'
    assert_contains "$RUN_ERR" 'DID NOT RUN' 'names the real outcome'
    assert_not_contains "$RUN_OUT" 'clean' 'must not claim a clean scan'
}

secret_scan_uses_the_filesystem_walk_not_the_git_history_walk() {
    with_fake_gitleaks
    export FAKE_GITLEAKS_REPORT="$CLEAN_REPORT" FAKE_GITLEAKS_STATUS=0
    run_cmd "$SECRET_SCAN_SH"
    assert_eq 0 "$RUN_STATUS" "sanity: the clean-report case must still pass: $RUN_ERR"
    local invocation subcommand
    invocation="$(grep -v '^version$' "$FAKE_GITLEAKS_LOG" | head -1)"
    [ -n "$invocation" ] || fail 'gitleaks was never invoked to run a scan'
    subcommand="${invocation%% *}"
    assert_eq 'dir' "$subcommand" 'must scan the filesystem (gitleaks dir), not history (gitleaks git)'
}

check_gate_runs_the_secret_scan() {
    # Deleting the step from `just check` must fail this test.
    local plan
    plan="$(cd "$SCRIPT_TESTS_ROOT" && just -n check 2>&1)"
    assert_contains "$plan" 'scripts/secret-scan.sh' 'the gate runs the secret scan'
}

# --- real-tool tests (the real, pinned gitleaks binary; no mock) ------------
#
# gitleaks is a required gate dependency (scripts/secret-scan.sh fails the
# gate outright if it is missing, same as cargo-audit) - see AGENTS.md,
# "Secret scanning". These tests assume it is on PATH, same as the sqlite-gate
# suite assumes a real cargo/rustc.

secret_scan_detects_a_working_tree_key() {
    local fixture="$TEST_TMP/src"
    mkdir -p "$fixture"
    # Assembled at runtime, from hashes of plain fixture labels, so no line in
    # this committed file is a contiguous, scanner-shaped secret (which would
    # trip this very gate on this file). Shape matches gitleaks' openai-api-key
    # rule (sk-proj-<58 hex>T3BlbkFJ<58 hex>); content is a SHA-256 digest of a
    # label string, never real key material, and was never a working credential.
    local body_a body_b
    body_a="$(printf 'adele-secret-scan-test-fixture-alpha' | sha256sum | cut -c1-58)"
    body_b="$(printf 'adele-secret-scan-test-fixture-beta' | sha256sum | cut -c1-58)"
    printf 'OPENAI_API_KEY=sk-proj-%sT3BlbkFJ%s\n' "$body_a" "$body_b" >"$fixture/.env"

    run_cmd "$SECRET_SCAN_SH" "$fixture"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a synthetic but correctly-shaped key must fail the scan'
    assert_contains "$RUN_ERR" '.env' 'names the file holding the key'
    assert_contains "$RUN_ERR" 'openai-api-key' 'names the rule that matched'
}

secret_scan_does_not_flag_the_clean_tree() {
    # The repo as it stands, including its own real test fixtures (test-only
    # PEM keys under testdata/, a truncated example JWT in the docs) must
    # scan clean under the real config - or the gate is noise from the day it
    # ships.
    run_cmd "$SECRET_SCAN_SH" "$SCRIPT_TESTS_ROOT"
    assert_eq 0 "$RUN_STATUS" "the repo must scan clean: $RUN_ERR$RUN_OUT"
}

secret_scan_detects_a_key_under_claude_worktrees() {
    # .claude/worktrees/ must NOT be allowlisted (AGENTS.md, "Secret
    # scanning"): the #811 incident involved a live .env AND eight stale
    # .claude/worktrees checkouts, so excluding this directory would reopen
    # half the blind spot the gate exists to close. Same reasoning as keeping
    # .flatpak-builder/ in scope, applied to the other directory the
    # incident actually touched.
    local fixture="$TEST_TMP/src"
    mkdir -p "$fixture/.claude/worktrees/some-session"
    local body_a body_b
    body_a="$(printf 'adele-secret-scan-test-fixture-worktrees-alpha' | sha256sum | cut -c1-58)"
    body_b="$(printf 'adele-secret-scan-test-fixture-worktrees-beta' | sha256sum | cut -c1-58)"
    printf 'OPENAI_API_KEY=sk-proj-%sT3BlbkFJ%s\n' "$body_a" "$body_b" \
        >"$fixture/.claude/worktrees/some-session/.env"

    run_cmd "$SECRET_SCAN_SH" "$fixture"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a key under .claude/worktrees/ must still fail the scan'
    assert_contains "$RUN_ERR" '.claude/worktrees' 'names the nested-worktree path holding the key'
}

run_test secret_scan_passes_on_a_clean_report
run_test secret_scan_fails_when_gitleaks_reports_a_leak
run_test secret_scan_fails_loudly_when_gitleaks_is_not_installed
run_test secret_scan_fails_when_the_installed_gitleaks_version_does_not_match_the_pin
run_test secret_scan_fails_when_gitleaks_reports_no_parseable_version
run_test secret_scan_allows_an_unpinned_gitleaks_version_with_explicit_opt_in
run_test secret_scan_allows_an_unparseable_gitleaks_version_with_explicit_opt_in
run_test secret_scan_fails_when_gitleaks_produces_no_report
run_test secret_scan_uses_the_filesystem_walk_not_the_git_history_walk
run_test check_gate_runs_the_secret_scan
run_test secret_scan_detects_a_working_tree_key
run_test secret_scan_does_not_flag_the_clean_tree
run_test secret_scan_detects_a_key_under_claude_worktrees
finish_tests 'secret-scan-gate'

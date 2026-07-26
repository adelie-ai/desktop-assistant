#!/usr/bin/env bash
# Acceptance criteria for the dependency-scan step of the gate (#706).
#
# The point of the step is that it cannot pass by accident: not when
# cargo-audit is missing, and not when the advisory database could not be
# fetched. Those outcomes are driven here by putting a fake `cargo-audit` on
# PATH - the same mechanism cargo itself uses to find the subcommand - so the
# script under test runs unmodified.
set -euo pipefail
# shellcheck source=scripts/tests/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

AUDIT_SH="$SCRIPT_TESTS_ROOT/scripts/audit.sh"

CLEAN_REPORT='{"database":{"advisory-count":1169,"last-commit":null,"last-updated":null},"lockfile":{"dependency-count":496},"vulnerabilities":{"found":false,"count":0,"list":[]},"warnings":{}}'
VULNERABLE_REPORT='{"database":{"advisory-count":1169},"lockfile":{"dependency-count":496},"vulnerabilities":{"found":true,"count":1,"list":[{"advisory":{"id":"RUSTSEC-2020-0071","package":"time"}}]},"warnings":{}}'
WARNING_REPORT='{"database":{"advisory-count":1169},"lockfile":{"dependency-count":496},"vulnerabilities":{"found":false,"count":0,"list":[]},"warnings":{"unmaintained":[{"kind":"unmaintained","package":{"name":"proc-macro-error"}}]}}'

# Put a fake cargo-audit on PATH and point the knobs at this test's temp dir.
with_fake_cargo_audit() {
    mkdir -p "$TEST_TMP/bin"
    cp "$SCRIPT_TESTS_FIXTURES/fake-cargo-audit.sh" "$TEST_TMP/bin/cargo-audit"
    chmod +x "$TEST_TMP/bin/cargo-audit"
    export PATH="$TEST_TMP/bin:$PATH"
    export FAKE_AUDIT_LOG="$TEST_TMP/audit.log"
    : >"$FAKE_AUDIT_LOG"
}

audit_passes_when_no_advisory_is_reported() {
    with_fake_cargo_audit
    export FAKE_AUDIT_STDOUT="$CLEAN_REPORT" FAKE_AUDIT_STATUS=0
    run_cmd "$AUDIT_SH"
    assert_eq 0 "$RUN_STATUS" 'exit status of a clean scan'
    assert_contains "$RUN_OUT$RUN_ERR" '496' 'clean scan reports what it scanned'
}

audit_fails_when_an_advisory_is_reported() {
    with_fake_cargo_audit
    export FAKE_AUDIT_STDOUT="$VULNERABLE_REPORT" FAKE_AUDIT_STATUS=1
    run_cmd "$AUDIT_SH"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a reported advisory must fail the step'
    assert_contains "$RUN_ERR" 'ADVISOR' 'failure names the advisories'
}

audit_fails_loudly_when_cargo_audit_is_not_installed() {
    mkdir -p "$TEST_TMP/empty"
    # A PATH with no cargo-audit anywhere on it. The step must not treat an
    # absent scanner as "nothing to report".
    run_cmd env PATH="$TEST_TMP/empty:/usr/bin:/bin" "$AUDIT_SH"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a missing cargo-audit must fail the step'
    assert_contains "$RUN_ERR" 'cargo install cargo-audit' 'says how to fix it'
    assert_not_contains "$RUN_OUT" 'clean' 'must not claim a clean scan'
}

audit_fails_when_the_advisory_database_cannot_be_fetched() {
    with_fake_cargo_audit
    # Offline: cargo-audit errors out before producing any report.
    export FAKE_AUDIT_STDOUT='' FAKE_AUDIT_STATUS=2
    export FAKE_AUDIT_STDERR='error: couldn'"'"'t fetch advisory database: network unreachable'
    run_cmd "$AUDIT_SH"
    [ "$RUN_STATUS" -ne 0 ] || fail 'an unusable advisory database must fail the step'
    assert_contains "$RUN_ERR" 'advisory database' 'names the advisory database'
    assert_contains "$RUN_ERR" 'network unreachable' 'passes through what cargo-audit said'
    assert_not_contains "$RUN_OUT" 'clean' 'must not claim a clean scan'
}

audit_offline_optin_uses_the_cached_database_and_says_so() {
    with_fake_cargo_audit
    export FAKE_AUDIT_STDOUT="$CLEAN_REPORT" FAKE_AUDIT_STATUS=0
    export ADELE_AUDIT_ALLOW_STALE=1
    run_cmd "$AUDIT_SH"
    assert_eq 0 "$RUN_STATUS" 'explicit offline opt-in still completes'
    assert_contains "$(cat "$FAKE_AUDIT_LOG")" '--no-fetch' 'opt-in skips the fetch'
    assert_contains "$RUN_ERR" 'STALE ADVISORY DATABASE' 'opt-in is loud about what it did'
}

audit_reports_informational_advisories_without_failing() {
    with_fake_cargo_audit
    export FAKE_AUDIT_STDOUT="$WARNING_REPORT" FAKE_AUDIT_STATUS=0
    run_cmd "$AUDIT_SH"
    assert_eq 0 "$RUN_STATUS" 'informational advisories are not a gate failure'
    assert_contains "$RUN_ERR" 'informational' 'but they are surfaced'
    assert_not_contains "$RUN_OUT" 'clean' 'and the summary does not call the scan clean'
}

check_gate_runs_the_dependency_scan_before_the_first_build() {
    # `just -n check` prints the plan it would execute, in order. Build scripts
    # run at first compile - including under clippy - so the scan has to come
    # before any of them, not after `test` (AGENTS.md, "Security review").
    local plan
    plan="$(cd "$SCRIPT_TESTS_ROOT" && just -n check 2>&1)"
    assert_contains "$plan" 'scripts/audit.sh' 'the gate runs the dependency scan'
    local audit_at clippy_at build_at
    audit_at="$(printf '%s\n' "$plan" | grep -n 'scripts/audit.sh' | head -1 | cut -d: -f1)"
    clippy_at="$(printf '%s\n' "$plan" | grep -n 'cargo clippy' | head -1 | cut -d: -f1)"
    build_at="$(printf '%s\n' "$plan" | grep -n 'cargo build --workspace' | head -1 | cut -d: -f1)"
    [ "$audit_at" -lt "$clippy_at" ] || fail "scan (line $audit_at) must precede clippy (line $clippy_at)"
    [ "$audit_at" -lt "$build_at" ] || fail "scan (line $audit_at) must precede build (line $build_at)"
}

run_test audit_passes_when_no_advisory_is_reported
run_test audit_fails_when_an_advisory_is_reported
run_test audit_fails_loudly_when_cargo_audit_is_not_installed
run_test audit_fails_when_the_advisory_database_cannot_be_fetched
run_test audit_offline_optin_uses_the_cached_database_and_says_so
run_test audit_reports_informational_advisories_without_failing
run_test check_gate_runs_the_dependency_scan_before_the_first_build
finish_tests 'audit-gate'

#!/usr/bin/env bash
# Acceptance criteria for the SQLite-adapter step of the gate (#742).
#
# `crates/storage-sqlite` is empty without `--features sqlite`: every module and
# every test file is `#[cfg(feature = "sqlite")]`, and the feature is off by
# default so the daemon build never links the sqlite C library. A workspace
# clippy/test therefore type-checks none of the adapter and runs none of its
# tests, which is why the gate needs a step that names the crate AND the
# feature. These tests read the plan `just -n check` would execute, then run the
# extracted commands, so "the step is in the gate" and "the step actually
# compiles and runs the adapter" are both asserted rather than assumed.
#
# The extracted commands are real cargo invocations. Inside `just check` they
# are warm - the same step has already built those artifacts - but running this
# suite on its own in a cold tree pays a first compile of sqlx + libsqlite3-sys.
set -euo pipefail
# shellcheck source=scripts/tests/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SQLITE_CRATE='desktop-assistant-storage-sqlite'
# A test that exists only inside the feature-gated code, so its presence proves
# the adapter itself was compiled - not merely that some binary ran. This one is
# the migration-registration guard: unregistered migration files compile fine
# and silently never run.
GUARD_TEST='every_migration_is_registered'

# The commands `just check` would run, one per line, in order.
gate_plan() {
    (cd "$SCRIPT_TESTS_ROOT" && just -n check 2>&1)
}

# The single planned command that is `cargo <verb>` against the sqlite crate,
# or the empty string. No match is an expected outcome here (it is the very
# failure these tests describe), so it must not abort the test before the
# assertion that names it.
gate_step_for() { # gate_step_for <cargo-verb>
    gate_plan | { grep -F "cargo $1" || true; } | { grep -F -- "$SQLITE_CRATE" || true; } | head -1
}

# 1-based line number of the first planned command matching a fixed string, or
# the empty string when the plan has no such command.
gate_plan_line_of() { # gate_plan_line_of <needle>
    gate_plan | { grep -n -F -- "$1" || true; } | head -1 | cut -d: -f1
}

check_gate_lints_the_sqlite_adapter_with_its_feature_enabled() {
    local step
    step="$(gate_step_for clippy)"
    [ -n "$step" ] || fail "no 'cargo clippy' step in 'just check' names $SQLITE_CRATE"
    assert_contains "$step" '--features sqlite' 'the lint step enables the feature'
    assert_contains "$step" '--all-targets' 'the lint step type-checks the tests too'
    assert_contains "$step" '-D warnings' 'the lint step treats warnings as errors'
}

check_gate_runs_the_sqlite_adapter_test_suite() {
    local step
    step="$(gate_step_for test)"
    [ -n "$step" ] || fail "no 'cargo test' step in 'just check' names $SQLITE_CRATE"
    assert_contains "$step" '--features sqlite' 'the test step enables the feature'
}

sqlite_gate_steps_run_after_the_dependency_scan() {
    # Enabling the feature pulls in libsqlite3-sys, whose build script compiles C
    # at first build - under clippy as much as under build. The advisory scan has
    # to precede it, for the same reason it precedes the workspace steps.
    local audit_at lint_at test_at
    audit_at="$(gate_plan_line_of 'scripts/audit.sh')"
    lint_at="$(gate_plan_line_of "clippy -p $SQLITE_CRATE")"
    test_at="$(gate_plan_line_of "test -p $SQLITE_CRATE")"
    [ -n "$lint_at" ] || fail "no 'cargo clippy -p $SQLITE_CRATE' step in 'just check'"
    [ -n "$test_at" ] || fail "no 'cargo test -p $SQLITE_CRATE' step in 'just check'"
    [ "$audit_at" -lt "$lint_at" ] || fail "scan (line $audit_at) must precede the sqlite lint (line $lint_at)"
    [ "$audit_at" -lt "$test_at" ] || fail "scan (line $audit_at) must precede the sqlite tests (line $test_at)"
}

the_gates_sqlite_test_step_executes_the_adapters_own_tests() {
    # Not "a cargo test step exists" but "that exact command runs the adapter's
    # tests": run the planned command with `--list`, which builds the same test
    # binaries and prints what they would run without running it.
    local step
    step="$(gate_step_for test)"
    [ -n "$step" ] || fail "no 'cargo test' step in 'just check' names $SQLITE_CRATE"
    # `eval` on a line printed by our own justfile, not on external input.
    run_cmd bash -c "cd '$SCRIPT_TESTS_ROOT' && eval \"\$1 -- --list\"" _ "$step"
    assert_eq 0 "$RUN_STATUS" "the gate's sqlite test step must build: $step"
    local listed
    listed="$(printf '%s\n' "$RUN_OUT" | grep -c ': test$' || true)"
    [ "$listed" -gt 0 ] || fail "the gate's sqlite test step runs 0 tests: $step"
    assert_contains "$RUN_OUT" "$GUARD_TEST" 'the feature-gated adapter tests are among them'
}

the_workspace_test_run_alone_executes_none_of_the_adapters_tests() {
    # The premise of the dedicated step. If this ever fails because the feature
    # became default-on, the extra step is redundant - remove it deliberately
    # rather than leaving two ways to run the same suite.
    run_cmd bash -c "cd '$SCRIPT_TESTS_ROOT' && cargo test -p '$SQLITE_CRATE' -- --list"
    assert_eq 0 "$RUN_STATUS" 'the crate builds with default features'
    local listed
    listed="$(printf '%s\n' "$RUN_OUT" | grep -c ': test$' || true)"
    assert_eq 0 "$listed" 'tests run by a default-feature workspace test'
    assert_not_contains "$RUN_OUT" "$GUARD_TEST" 'the adapter is dark without its feature'
}

run_test check_gate_lints_the_sqlite_adapter_with_its_feature_enabled
run_test check_gate_runs_the_sqlite_adapter_test_suite
run_test sqlite_gate_steps_run_after_the_dependency_scan
run_test the_gates_sqlite_test_step_executes_the_adapters_own_tests
run_test the_workspace_test_run_alone_executes_none_of_the_adapters_tests
finish_tests 'sqlite-gate'

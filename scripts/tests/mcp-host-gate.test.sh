#!/usr/bin/env bash
# Acceptance criteria for the client-side MCP host step of the gate (#910).
#
# `crates/client-common`'s `mcp_host` module is `#[cfg(feature = "mcp-host")]`,
# off by default, and - unlike `storage-sqlite`'s `sqlite` feature - no other
# workspace crate enables it as a dependency either, so nothing pulls it in via
# feature unification. A workspace clippy/test therefore type-checks none of
# the module and runs none of its tests, including the environment-isolation
# coverage for the spawn path a real desktop session uses (as opposed to the
# daemon's own headless fleet). This mirrors sqlite-gate.test.sh's structure:
# these tests read the plan `just -n check` would execute, then run the
# extracted commands, so "the step is in the gate" and "the step actually
# compiles and runs the module" are both asserted rather than assumed.
set -euo pipefail
# shellcheck source=scripts/tests/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

MCP_HOST_CRATE='desktop-assistant-client-common'
# A test that exists only inside the feature-gated module, so its presence
# proves the module itself was compiled - not merely that some binary ran.
# This one is the environment-isolation guard added alongside #910's
# per-server inherit_env mechanism.
GUARD_TEST='client_host_delivers_an_inherit_env_variable_to_the_spawned_child'

# The commands `just check` would run, one per line, in order.
gate_plan() {
    (cd "$SCRIPT_TESTS_ROOT" && just -n check 2>&1)
}

# The single planned command that is `cargo <verb>` against the client-common
# crate with the mcp-host feature, or the empty string. No match is an
# expected outcome here (it is the very failure these tests describe), so it
# must not abort the test before the assertion that names it.
gate_step_for() { # gate_step_for <cargo-verb>
    gate_plan | { grep -F "cargo $1" || true; } | { grep -F -- "$MCP_HOST_CRATE" || true; } | { grep -F -- 'mcp-host' || true; } | head -1
}

# 1-based line number of the first planned command matching a fixed string, or
# the empty string when the plan has no such command.
gate_plan_line_of() { # gate_plan_line_of <needle>
    gate_plan | { grep -n -F -- "$1" || true; } | head -1 | cut -d: -f1
}

check_gate_lints_the_mcp_host_module_with_its_feature_enabled() {
    local step
    step="$(gate_step_for clippy)"
    [ -n "$step" ] || fail "no 'cargo clippy' step in 'just check' names $MCP_HOST_CRATE with mcp-host"
    assert_contains "$step" '--features mcp-host' 'the lint step enables the feature'
    assert_contains "$step" '--all-targets' 'the lint step type-checks the tests too'
    assert_contains "$step" '-D warnings' 'the lint step treats warnings as errors'
}

check_gate_runs_the_mcp_host_test_suite() {
    local step
    step="$(gate_step_for test)"
    [ -n "$step" ] || fail "no 'cargo test' step in 'just check' names $MCP_HOST_CRATE with mcp-host"
    assert_contains "$step" '--features mcp-host' 'the test step enables the feature'
}

mcp_host_gate_steps_run_after_the_dependency_scan() {
    local audit_at lint_at test_at
    audit_at="$(gate_plan_line_of 'scripts/audit.sh')"
    lint_at="$(gate_plan_line_of "clippy -p $MCP_HOST_CRATE --features mcp-host")"
    test_at="$(gate_plan_line_of "test -p $MCP_HOST_CRATE --features mcp-host")"
    [ -n "$lint_at" ] || fail "no 'cargo clippy -p $MCP_HOST_CRATE --features mcp-host' step in 'just check'"
    [ -n "$test_at" ] || fail "no 'cargo test -p $MCP_HOST_CRATE --features mcp-host' step in 'just check'"
    [ "$audit_at" -lt "$lint_at" ] || fail "scan (line $audit_at) must precede the mcp-host lint (line $lint_at)"
    [ "$audit_at" -lt "$test_at" ] || fail "scan (line $audit_at) must precede the mcp-host tests (line $test_at)"
}

the_gates_mcp_host_test_step_executes_the_modules_own_tests() {
    # Not "a cargo test step exists" but "that exact command runs the
    # module's tests": run the planned command with `--list`, which builds
    # the same test binary and prints what it would run without running it.
    local step
    step="$(gate_step_for test)"
    [ -n "$step" ] || fail "no 'cargo test' step in 'just check' names $MCP_HOST_CRATE with mcp-host"
    run_cmd bash -c 'cd "$1" && eval "$2 -- --list"' _ "$SCRIPT_TESTS_ROOT" "$step"
    assert_eq 0 "$RUN_STATUS" "the gate's mcp-host test step must build: $step"
    local listed
    listed="$(printf '%s\n' "$RUN_OUT" | grep -c ': test$' || true)"
    [ "$listed" -gt 0 ] || fail "the gate's mcp-host test step runs 0 tests: $step"
    assert_contains "$RUN_OUT" "$GUARD_TEST" 'the environment-isolation coverage is among them'
}

the_workspace_test_run_alone_executes_none_of_the_mcp_host_modules_tests() {
    # The premise of the dedicated step. Unlike storage-sqlite (a crate that is
    # EMPTY without its feature), client-common has plenty of its own tests
    # outside mcp_host, so the assertion is not "zero tests total" but "none of
    # THIS module's" - the guard test specifically must be absent. If this
    # ever fails because the feature became default-on (or some other crate
    # started depending on it with the feature enabled), the extra step is
    # redundant - remove it deliberately rather than leaving two ways to run
    # the same suite.
    run_cmd bash -c 'cd "$1" && cargo test -p "$2" -- --list' _ "$SCRIPT_TESTS_ROOT" "$MCP_HOST_CRATE"
    assert_eq 0 "$RUN_STATUS" 'the crate builds with default features'
    assert_not_contains "$RUN_OUT" "$GUARD_TEST" 'the mcp-host module is dark without its feature'
}

run_test check_gate_lints_the_mcp_host_module_with_its_feature_enabled
run_test check_gate_runs_the_mcp_host_test_suite
run_test mcp_host_gate_steps_run_after_the_dependency_scan
run_test the_gates_mcp_host_test_step_executes_the_modules_own_tests
run_test the_workspace_test_run_alone_executes_none_of_the_mcp_host_modules_tests
finish_tests 'mcp-host-gate'

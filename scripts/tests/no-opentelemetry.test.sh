#!/usr/bin/env bash
# Acceptance criteria for the no-opentelemetry step of the gate.
#
# Telemetry export is an off-by-default Cargo feature. The promise that goes
# with it is that a default build resolves no opentelemetry crate at all, so a
# desktop install from `cargo install` pays nothing for it: no extra crates, no
# native code, no C toolchain. Nothing in the source says that. One
# `features = ["otel"]` written on a dependency instead of a passthrough turns
# it on for every build, and every other step of the gate stays green.
#
# So the property is checked mechanically, against the resolved tree, and the
# check is held to being able to fail: a `cargo tree` that errors, or that
# reports a tree too small to be this workspace, is a failure and not a pass.
set -euo pipefail
# shellcheck source=scripts/tests/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SCAN_SH="$SCRIPT_TESTS_ROOT/scripts/no-opentelemetry.sh"

# A tree large enough to look real, with nothing telemetry-related in it.
CLEAN_TREE="$(
    for i in $(seq 1 40); do printf 'crate-%02d v1.0.0\n' "$i"; done
    printf 'tracing v0.1.44\ntracing-subscriber v0.3.23\nadelie-telemetry v0.1.0\n'
)"

# Put the fake cargo on PATH and point the script at it.
with_fake_cargo() {
    mkdir -p "$TEST_TMP/bin"
    cp "$SCRIPT_TESTS_FIXTURES/fake-cargo-tree.sh" "$TEST_TMP/bin/cargo"
    chmod +x "$TEST_TMP/bin/cargo"
    export PATH="$TEST_TMP/bin:$PATH"
}

default_build_pulls_no_opentelemetry() {
    # The real workspace, the real cargo. This is the criterion itself.
    run_cmd "$SCAN_SH"
    assert_eq 0 "$RUN_STATUS" 'exit status over this workspace'
    assert_contains "$RUN_OUT$RUN_ERR" 'no opentelemetry' 'the clean run says what it checked'
}

the_scan_fails_when_an_opentelemetry_crate_is_resolved() {
    with_fake_cargo
    export FAKE_TREE_STDOUT="$CLEAN_TREE
opentelemetry v0.32.0
opentelemetry_sdk v0.32.1"
    run_cmd "$SCAN_SH"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a resolved opentelemetry crate must fail the step'
    assert_contains "$RUN_ERR" 'opentelemetry_sdk' 'the failure names the offending crate'
}

the_scan_fails_when_the_tracing_bridge_is_resolved() {
    # `tracing-opentelemetry` does not start with "opentelemetry", so a check
    # anchored at the start of the name would let the whole pipeline through.
    with_fake_cargo
    export FAKE_TREE_STDOUT="$CLEAN_TREE
tracing-opentelemetry v0.33.0"
    run_cmd "$SCAN_SH"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a resolved tracing-opentelemetry must fail the step'
    assert_contains "$RUN_ERR" 'tracing-opentelemetry' 'the failure names the offending crate'
}

the_scan_fails_loudly_when_cargo_cannot_read_the_tree() {
    with_fake_cargo
    export FAKE_TREE_STDOUT='' FAKE_TREE_STATUS=101
    run_cmd "$SCAN_SH"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a cargo that could not resolve must fail the step'
}

the_scan_fails_when_the_tree_is_too_small_to_be_this_workspace() {
    # A `cargo tree` that selects almost nothing exits 0 and prints almost
    # nothing, which reads in the log exactly like "no opentelemetry here".
    with_fake_cargo
    export FAKE_TREE_STDOUT='serde v1.0.0'
    run_cmd "$SCAN_SH"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a tree too small to be this workspace must fail the step'
}

run_test default_build_pulls_no_opentelemetry
run_test the_scan_fails_when_an_opentelemetry_crate_is_resolved
run_test the_scan_fails_when_the_tracing_bridge_is_resolved
run_test the_scan_fails_loudly_when_cargo_cannot_read_the_tree
run_test the_scan_fails_when_the_tree_is_too_small_to_be_this_workspace
finish_tests 'no-opentelemetry'

#!/usr/bin/env bash
# Acceptance criteria for the documentation step of the gate (#1046).
#
# None of `cargo fmt`, `cargo clippy -- -D warnings`, `cargo build` or
# `cargo test` evaluates rustdoc lints, so before this step a doc link to a
# renamed, privatised or deleted item passed the whole gate. Two halves are
# asserted here rather than assumed: that the step is IN `just check` (read out
# of the plan `just -n check` would execute, the way sqlite-gate.test.sh and
# mcp-host-gate.test.sh do), and that the step actually FAILS on the defects it
# exists to catch - proven against throwaway fixture workspaces, so the
# detection is exercised without breaking the real tree.
#
# The fixture workspaces have no dependencies, so each `cargo doc` over one is
# a fraction of a second.
set -euo pipefail
# shellcheck source=scripts/tests/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

DOC_SH="$SCRIPT_TESTS_ROOT/scripts/doc.sh"
SQLITE_CRATE='desktop-assistant-storage-sqlite'
MCP_HOST_CRATE='desktop-assistant-client-common'

# The commands `just check` would run, one per line, in order.
gate_plan() {
    (cd "$SCRIPT_TESTS_ROOT" && just -n check 2>&1)
}

# The single planned command that invokes the doc script with <needle> in it,
# or the empty string. No match is an expected outcome here (it is the very
# failure these tests describe), so it must not abort before the assertion.
gate_doc_step_for() { # gate_doc_step_for <needle>
    gate_plan | { grep -F 'scripts/doc.sh' || true; } | { grep -F -- "$1" || true; } | head -1
}

# 1-based line number of the first planned command matching a fixed string, or
# the empty string when the plan has no such command.
gate_plan_line_of() { # gate_plan_line_of <needle>
    gate_plan | { grep -n -F -- "$1" || true; } | head -1 | cut -d: -f1
}

# Write a single-crate workspace into $TEST_TMP/<name> whose lib.rs is the
# given body, with the same warnings-are-errors lint level the real workspace
# sets. Echoes the workspace directory.
make_fixture_crate() { # make_fixture_crate <name> <lib.rs body>
    local name="$1" body="$2" dir="$TEST_TMP/$1"
    mkdir -p "$dir/src"
    cat >"$dir/Cargo.toml" <<EOF
[package]
name = "$name"
version = "0.0.0"
edition = "2021"

[lints.rust]
warnings = "deny"

[workspace]
EOF
    printf '%s\n' "$body" >"$dir/src/lib.rs"
    printf '%s\n' "$dir"
}

# Put a fake cargo on PATH ahead of the real one.
with_fake_cargo() {
    mkdir -p "$TEST_TMP/bin"
    cp "$SCRIPT_TESTS_FIXTURES/fake-cargo.sh" "$TEST_TMP/bin/cargo"
    chmod +x "$TEST_TMP/bin/cargo"
    export PATH="$TEST_TMP/bin:$PATH"
}

# --- the step is in the gate --------------------------------------------------

check_gate_documents_the_workspace() {
    local step
    step="$(gate_doc_step_for '--workspace')"
    [ -n "$step" ] || fail "no './scripts/doc.sh --workspace' step in 'just check'"
}

check_gate_documents_the_sqlite_adapter_with_its_feature_enabled() {
    # Same premise as lint-sqlite/test-sqlite: without the feature the crate is
    # an empty shell, so the workspace step documents none of it.
    local step
    step="$(gate_doc_step_for "$SQLITE_CRATE")"
    [ -n "$step" ] || fail "no 'scripts/doc.sh' step in 'just check' names $SQLITE_CRATE"
    assert_contains "$step" '--features sqlite' 'the doc step enables the feature'
}

check_gate_documents_the_mcp_host_with_its_feature_enabled() {
    local step
    step="$(gate_doc_step_for "$MCP_HOST_CRATE")"
    [ -n "$step" ] || fail "no 'scripts/doc.sh' step in 'just check' names $MCP_HOST_CRATE"
    assert_contains "$step" '--features mcp-host' 'the doc step enables the feature'
}

documentation_steps_run_after_the_dependency_scan() {
    # cargo doc compiles the dependency graph, so build scripts execute under it
    # exactly as they do under clippy and build. The scans precede it for the
    # same reason they precede those.
    local audit_at secret_at doc_at
    audit_at="$(gate_plan_line_of 'scripts/audit.sh')"
    secret_at="$(gate_plan_line_of 'scripts/secret-scan.sh')"
    doc_at="$(gate_plan_line_of 'scripts/doc.sh')"
    [ -n "$doc_at" ] || fail "no 'scripts/doc.sh' step in 'just check'"
    [ "$audit_at" -lt "$doc_at" ] || fail "scan (line $audit_at) must precede the doc step (line $doc_at)"
    [ "$secret_at" -lt "$doc_at" ] || fail "secret scan (line $secret_at) must precede the doc step (line $doc_at)"
}

# --- the step catches what it is for -----------------------------------------

a_broken_intra_doc_link_fails_the_step() {
    local dir
    dir="$(make_fixture_crate broken_link '
//! A link to [`gone`], which does not exist.
pub fn present() {}
')"
    run_cmd bash -c 'cd "$1" && "$2"' _ "$dir" "$DOC_SH"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a broken intra-doc link must fail the doc step'
    assert_contains "$RUN_ERR" 'RUSTDOC ERRORS' 'the failure is legible'
    assert_contains "$RUN_ERR" 'unresolved link' 'the failure names what rustdoc said'
    assert_not_contains "$RUN_OUT" 'clean' 'must not claim a clean run'
}

a_private_intra_doc_link_fails_the_step() {
    # The class that had been rotting in crates/core/src/context/mod.rs: the
    # link resolves, but only to something the reader of the public docs cannot
    # follow.
    local dir
    dir="$(make_fixture_crate private_link '
//! A link to [`hidden`], which is private.
fn hidden() {}
pub fn present() {}
')"
    run_cmd bash -c 'cd "$1" && "$2"' _ "$dir" "$DOC_SH"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a private intra-doc link must fail the doc step'
    assert_contains "$RUN_ERR" 'private item' 'the failure names the private target'
}

a_clean_crate_passes_the_step() {
    # The control: without it, a step that failed on everything would pass every
    # test above.
    local dir
    dir="$(make_fixture_crate clean_docs '
//! A link to [`present`], which exists and is public.
pub fn present() {}
')"
    run_cmd bash -c 'cd "$1" && "$2"' _ "$dir" "$DOC_SH"
    assert_eq 0 "$RUN_STATUS" "a clean crate must pass: $RUN_ERR"
    assert_contains "$RUN_OUT" 'doc: clean' 'a clean run says what it documented'
}

the_step_fails_when_cargo_produced_no_documentation() {
    # #706's failure mode, in this step's terms: a cargo doc that selects
    # nothing exits 0 and prints almost nothing, which reads in the log exactly
    # like "the docs are fine". Driven with a fake cargo, because a real one
    # will not report success while writing no output.
    with_fake_cargo
    export FAKE_CARGO_TARGET_DIR="$TEST_TMP/target" FAKE_CARGO_PACKAGES='some-crate'
    export FAKE_CARGO_DOC_STATUS=0 FAKE_CARGO_DOC_STDOUT=''
    mkdir -p "$TEST_TMP/target/doc"
    run_cmd "$DOC_SH"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a run that documented nothing must fail the step'
    assert_contains "$RUN_ERR" 'PRODUCED NO DOCUMENTATION' 'the failure says nothing was checked'
    assert_not_contains "$RUN_OUT" 'clean' 'must not claim a clean run'
}

the_step_fails_when_rustdoc_only_warned() {
    # A crate that does not inherit `[lints] workspace = true` gets warnings
    # where every other crate gets errors, so cargo exits 0 and its
    # documentation rots behind a green gate.
    with_fake_cargo
    export FAKE_CARGO_TARGET_DIR="$TEST_TMP/target" FAKE_CARGO_PACKAGES='some-crate'
    export FAKE_CARGO_DOC_STATUS=0
    export FAKE_CARGO_DOC_STDOUT='warning: unresolved link to `gone`'
    mkdir -p "$TEST_TMP/target/doc/some_crate"
    printf '<html></html>' >"$TEST_TMP/target/doc/some_crate/index.html"
    run_cmd "$DOC_SH"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a rustdoc warning must fail the step'
    assert_contains "$RUN_ERR" 'workspace = true' 'the failure says how to fix the lint level'
}

the_step_fails_when_no_package_is_selected() {
    with_fake_cargo
    export FAKE_CARGO_TARGET_DIR="$TEST_TMP/target" FAKE_CARGO_PACKAGES=''
    export FAKE_CARGO_DOC_STATUS=0 FAKE_CARGO_DOC_STDOUT=''
    run_cmd "$DOC_SH"
    [ "$RUN_STATUS" -ne 0 ] || fail 'an empty package selection must fail the step'
    assert_contains "$RUN_ERR" 'no package selected' 'the failure says why'
}

run_test check_gate_documents_the_workspace
run_test check_gate_documents_the_sqlite_adapter_with_its_feature_enabled
run_test check_gate_documents_the_mcp_host_with_its_feature_enabled
run_test documentation_steps_run_after_the_dependency_scan
run_test a_broken_intra_doc_link_fails_the_step
run_test a_private_intra_doc_link_fails_the_step
run_test a_clean_crate_passes_the_step
run_test the_step_fails_when_cargo_produced_no_documentation
run_test the_step_fails_when_rustdoc_only_warned
run_test the_step_fails_when_no_package_is_selected
finish_tests 'doc-gate'

#!/usr/bin/env bash
# Acceptance criteria for the `OTEL` image build argument (#1149).
#
# Both images build the same crates two ways: with the OTLP exporter compiled
# in, and without it. One shell wrapper, scripts/cargo-otel.sh, makes that
# choice, and the images call cargo only through it. So the wrapper is the
# artifact under test here - these tests run the real file with a stub `cargo`
# first on PATH and read the argument list cargo received, rather than
# re-implementing the rule and proving the copy agrees with itself.
#
# The Dockerfile assertions then check the other half: that both images declare
# the argument, copy the wrapper in, and reach cargo through nothing else. A
# correct wrapper that one build site bypasses ships an image with no exporter
# and no complaint.
set -euo pipefail
# shellcheck source=scripts/tests/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

WRAPPER="$SCRIPT_TESTS_ROOT/scripts/cargo-otel.sh"
DAEMON_IMAGE="$SCRIPT_TESTS_ROOT/Dockerfile"
FLEET_IMAGE="$SCRIPT_TESTS_ROOT/Dockerfile.fleet"

# --- harness -----------------------------------------------------------------

# A `cargo` that records the arguments it was given and does nothing else.
_stub_cargo() {
    mkdir -p "$TEST_TMP/bin"
    cat >"$TEST_TMP/bin/cargo" <<'STUB'
#!/bin/sh
printf '%s\n' "$*" >"$CARGO_ARGV"
STUB
    chmod +x "$TEST_TMP/bin/cargo"
}

# Run the wrapper with OTEL set to <value>, or unset when <value> is UNSET.
# Sets RUN_STATUS / RUN_OUT / RUN_ERR (see lib.sh).
_run_wrapper() { # _run_wrapper <value|UNSET> <wrapper args...>
    local otel="$1"
    shift
    _stub_cargo
    rm -f "$TEST_TMP/argv"
    if [ "$otel" = UNSET ]; then
        run_cmd env -u OTEL "PATH=$TEST_TMP/bin:$PATH" "CARGO_ARGV=$TEST_TMP/argv" \
            "$WRAPPER" "$@"
    else
        run_cmd env "OTEL=$otel" "PATH=$TEST_TMP/bin:$PATH" "CARGO_ARGV=$TEST_TMP/argv" \
            "$WRAPPER" "$@"
    fi
}

# The argument list cargo received. Fails when cargo never ran: without this a
# "no --features in the command line" assertion would pass on an empty file,
# which is the vacuous-pass shape this repo keeps finding.
_cargo_argv() {
    [ -s "$TEST_TMP/argv" ] || fail "cargo was never invoked, so this test proved nothing"
    cat "$TEST_TMP/argv"
}

# The fleet image builds a list of servers. Read each of the three places that
# list appears, so a server added to one and forgotten in another fails.
_fleet_copied_servers() {
    grep -oE '^COPY +[a-z-]+-mcp/' "$FLEET_IMAGE" | awk '{print $2}' | tr -d '/' | sort -u
}

_fleet_loop_servers() {
    awk '/for d in/,/; do/' "$FLEET_IMAGE" \
        | tr ' \\;' '\n\n\n' \
        | grep -E '^[a-z-]+-mcp$' \
        | sort -u
}

# --- the wrapper -------------------------------------------------------------

a_build_with_no_otel_argument_asks_cargo_for_no_features() {
    _run_wrapper UNSET build --release --locked -p desktop-assistant-daemon
    assert_eq 0 "$RUN_STATUS" 'wrapper exit status'
    assert_not_contains "$(_cargo_argv)" '--features' 'default cargo invocation'
}

an_explicit_zero_asks_cargo_for_no_features() {
    _run_wrapper 0 build --release --locked -p desktop-assistant-daemon
    assert_eq 0 "$RUN_STATUS" 'wrapper exit status'
    assert_not_contains "$(_cargo_argv)" '--features' 'OTEL=0 cargo invocation'
}

an_otel_build_asks_cargo_for_the_otel_feature() {
    _run_wrapper 1 build --release --locked -p desktop-assistant-daemon
    assert_eq 0 "$RUN_STATUS" 'wrapper exit status'
    assert_contains "$(_cargo_argv)" '--features otel' 'OTEL=1 cargo invocation'
}

an_unrecognised_otel_value_stops_the_build() {
    # The failure this guards against is silent: a value the wrapper does not
    # understand must never build an image that looks instrumented and exports
    # nothing.
    _run_wrapper true build --release
    [ "$RUN_STATUS" -ne 0 ] || fail "OTEL=true was accepted; it must fail loudly instead"
    assert_contains "$RUN_ERR" 'OTEL' 'the failure message must name the argument'
    [ ! -s "$TEST_TMP/argv" ] || fail "cargo ran anyway with an unusable OTEL value"
}

the_wrapper_passes_every_other_argument_through_unchanged() {
    _run_wrapper 1 build --release --locked -p desktop-assistant-daemon
    local argv
    argv="$(_cargo_argv)"
    assert_contains "$argv" 'build --release --locked -p desktop-assistant-daemon' \
        'the caller arguments must reach cargo in order'
}

# --- the two images ----------------------------------------------------------

the_daemon_image_declares_the_argument_off_by_default() {
    assert_contains "$(cat "$DAEMON_IMAGE")" 'ARG OTEL=0' 'Dockerfile'
}

the_fleet_image_declares_the_argument_off_by_default() {
    assert_contains "$(cat "$FLEET_IMAGE")" 'ARG OTEL=0' 'Dockerfile.fleet'
}

the_daemon_image_reaches_cargo_only_through_the_wrapper() {
    local text
    text="$(cat "$DAEMON_IMAGE")"
    assert_contains "$text" '/usr/local/bin/cargo-otel' 'Dockerfile must copy the wrapper in'
    assert_contains "$text" 'cargo-otel build' 'Dockerfile must build through the wrapper'
    assert_not_contains "$text" 'RUN cargo build' 'Dockerfile must not build around the wrapper'
}

the_fleet_image_reaches_cargo_only_through_the_wrapper() {
    local text
    text="$(cat "$FLEET_IMAGE")"
    assert_contains "$text" '/usr/local/bin/cargo-otel' 'Dockerfile.fleet must copy the wrapper in'
    assert_not_contains "$text" 'cargo build' 'Dockerfile.fleet must not build around the wrapper'
}

the_fleet_image_builds_the_daemon_through_the_wrapper() {
    assert_contains "$(cat "$FLEET_IMAGE")" \
        'cargo-otel build --release --locked -p desktop-assistant-daemon' \
        'Dockerfile.fleet daemon build'
}

the_daemon_image_records_whether_it_can_export() {
    # An image is otherwise indistinguishable from the outside, and "is this the
    # instrumented one?" is the first question when telemetry does not arrive.
    assert_contains "$(cat "$DAEMON_IMAGE")" 'LABEL ai.adelie.otel="${OTEL}"' 'Dockerfile'
}

the_fleet_image_records_whether_it_can_export() {
    assert_contains "$(cat "$FLEET_IMAGE")" 'LABEL ai.adelie.otel="${OTEL}"' 'Dockerfile.fleet'
}

the_fleet_image_builds_every_server_it_copies() {
    # A server added to the COPY list but not to the build loop is built by
    # nothing and copied from nowhere; the image then fails late, or worse,
    # ships the previous binary. Derive both lists from the file and compare.
    local copied looped
    copied="$(_fleet_copied_servers)"
    looped="$(_fleet_loop_servers)"
    [ -n "$copied" ] || fail 'found no COPY <server>-mcp/ lines; the file changed shape'
    [ -n "$looped" ] || fail 'found no server list in a build loop; the file changed shape'
    assert_eq "$copied" "$looped" 'the copied servers and the built servers must be the same set'
}

# --- guards on the harness itself --------------------------------------------

the_harness_notices_a_cargo_that_never_ran() {
    # Every "no --features" assertion above reads the recorded argument list. If
    # a broken wrapper silently skipped cargo, that file would be empty and each
    # of those assertions would pass while testing nothing. Prove the reader
    # refuses an empty recording.
    _stub_cargo
    rm -f "$TEST_TMP/argv"
    local status=0
    ( _cargo_argv ) >/dev/null 2>&1 || status=$?
    [ "$status" -ne 0 ] || fail 'the argument-list reader accepted a cargo that never ran'
}

run_test a_build_with_no_otel_argument_asks_cargo_for_no_features
run_test an_explicit_zero_asks_cargo_for_no_features
run_test an_otel_build_asks_cargo_for_the_otel_feature
run_test an_unrecognised_otel_value_stops_the_build
run_test the_wrapper_passes_every_other_argument_through_unchanged
run_test the_daemon_image_declares_the_argument_off_by_default
run_test the_fleet_image_declares_the_argument_off_by_default
run_test the_daemon_image_reaches_cargo_only_through_the_wrapper
run_test the_fleet_image_reaches_cargo_only_through_the_wrapper
run_test the_fleet_image_builds_the_daemon_through_the_wrapper
run_test the_daemon_image_records_whether_it_can_export
run_test the_fleet_image_records_whether_it_can_export
run_test the_fleet_image_builds_every_server_it_copies
run_test the_harness_notices_a_cargo_that_never_ran
finish_tests 'otel-build-arg'

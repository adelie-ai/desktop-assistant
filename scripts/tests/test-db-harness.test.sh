#!/usr/bin/env bash
# Acceptance criteria for the throwaway-Postgres harness behind `just test-db`
# (#662): two sessions must be able to run the DB-gated storage suites at the
# same time, and a run must only ever remove the container it created.
#
# The fast tests drive the harness through a fake container runtime injected
# with CONTAINER_CLI, so naming, port readback, readiness polling and teardown
# are exercised deterministically. The concurrency criterion is not fakeable,
# so it runs two real overlapping invocations against a real runtime.
set -euo pipefail
# shellcheck source=scripts/tests/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

TEST_DB_SH="$SCRIPT_TESTS_ROOT/scripts/test-db.sh"
LEGACY_FIXED_NAME=adele-storage-testdb

with_fake_cli() {
    mkdir -p "$TEST_TMP/bin"
    cp "$SCRIPT_TESTS_FIXTURES/fake-container-cli.sh" "$TEST_TMP/bin/fake-cli"
    chmod +x "$TEST_TMP/bin/fake-cli"
    export CONTAINER_CLI="$TEST_TMP/bin/fake-cli"
    export FAKE_CLI_LOG="$TEST_TMP/cli.log"
    : >"$FAKE_CLI_LOG"
    # Keep the unhappy paths quick; the fake is instant.
    export TEST_DB_READY_TIMEOUT=3
    unset TEST_DB_PORT || true
}

# The container name from the recorded `run` invocation.
created_name() {
    grep -m1 '^run ' "$FAKE_CLI_LOG" | tr ' ' '\n' | grep -A1 -x -- '--name' | tail -1
}

cli_log() { cat "$FAKE_CLI_LOG"; }

test_db_uses_a_unique_container_name_per_invocation() {
    with_fake_cli
    run_cmd "$TEST_DB_SH" start
    assert_eq 0 "$RUN_STATUS" 'first start'
    local first="$RUN_OUT"
    run_cmd "$TEST_DB_SH" start
    assert_eq 0 "$RUN_STATUS" 'second start'
    local second="$RUN_OUT"
    assert_contains "$first" 'ADELE_TEST_DB_CONTAINER' 'start reports its container'
    [ "$first" != "$second" ] || fail "two invocations produced the same settings:
$first"
    assert_not_contains "$first" "$LEGACY_FIXED_NAME" 'no shared fixed name'
}

test_db_exports_the_url_of_the_container_it_created() {
    with_fake_cli
    export FAKE_PORT=15999
    run_cmd "$TEST_DB_SH" run -- sh -c 'printf "%s|%s" "$TEST_DATABASE_URL" "$ADELE_TEST_DB_CONTAINER"'
    assert_eq 0 "$RUN_STATUS" 'payload status'
    assert_contains "$RUN_OUT" '127.0.0.1:15999/postgres' 'url points at the published port'
    assert_contains "$RUN_OUT" "$(created_name)" 'payload sees the container it got'
}

test_db_cleans_up_its_own_container_on_success_and_on_failure() {
    with_fake_cli
    run_cmd "$TEST_DB_SH" run -- true
    assert_eq 0 "$RUN_STATUS" 'a passing payload passes'
    assert_contains "$(cli_log)" "rm -f $(created_name)" 'removes its container after success'

    : >"$FAKE_CLI_LOG"
    run_cmd "$TEST_DB_SH" run -- sh -c 'exit 7'
    assert_eq 7 "$RUN_STATUS" 'the payload status is propagated'
    assert_contains "$(cli_log)" "rm -f $(created_name)" 'removes its container after failure'

    # A readiness timeout is also a failure that must not leak a container.
    : >"$FAKE_CLI_LOG"
    export FAKE_READY_STATUS=1
    run_cmd "$TEST_DB_SH" run -- true
    [ "$RUN_STATUS" -ne 0 ] || fail 'a database that never becomes ready must fail'
    assert_contains "$(cli_log)" "rm -f $(created_name)" 'removes its container after a timeout'
}

test_db_does_not_remove_a_container_it_did_not_create() {
    with_fake_cli
    run_cmd "$TEST_DB_SH" run -- true
    assert_eq 0 "$RUN_STATUS" 'run'
    local removals
    removals="$(cli_log | grep -c '^rm ' || true)"
    assert_eq 1 "$removals" 'exactly one container removed'
    assert_contains "$(cli_log)" "rm -f $(created_name)" 'and it is its own'
    assert_not_contains "$(cli_log)" "$LEGACY_FIXED_NAME" 'never touches the old shared name'

    # Nor on request: `stop` refuses anything it cannot have created.
    : >"$FAKE_CLI_LOG"
    run_cmd "$TEST_DB_SH" stop some-unrelated-container
    [ "$RUN_STATUS" -ne 0 ] || fail 'stop must refuse a foreign container'
    assert_contains "$RUN_ERR" 'refus' 'says it refused'
    assert_eq '' "$(cli_log)" 'and issues no runtime command at all'
}

test_db_reports_a_clear_error_when_no_free_port_is_available() {
    with_fake_cli
    export FAKE_RUN_STATUS=126
    export FAKE_RUN_STDERR='Error: pasta failed with exit code 1: Listen failed for HOST TCP port 127.0.0.1/55432: Address already in use'
    run_cmd "$TEST_DB_SH" run -- touch "$TEST_TMP/payload-ran"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a container that cannot start must fail the run'
    assert_contains "$RUN_ERR" 'port' 'the message is about the port'
    assert_contains "$RUN_ERR" 'Address already in use' 'passes through what the runtime said'
    [ ! -e "$TEST_TMP/payload-ran" ] || fail 'the payload must not run without a database'
    assert_not_contains "$(cli_log)" 'rm ' 'nothing was created, so nothing is removed'
}

test_db_down_removes_only_leftover_harness_containers() {
    with_fake_cli
    export FAKE_PS_NAMES="adele-testdb-1234-abcd
$LEGACY_FIXED_NAME"
    run_cmd "$TEST_DB_SH" prune
    assert_eq 0 "$RUN_STATUS" 'prune'
    assert_contains "$(cli_log)" 'rm -f adele-testdb-1234-abcd' 'sweeps its own leftovers'
    assert_not_contains "$(cli_log)" "rm -f $LEGACY_FIXED_NAME" 'leaves foreign containers alone'
}

two_concurrent_test_db_runs_both_pass() {
    local payload="$SCRIPT_TESTS_FIXTURES/assert-own-database.sh"
    local a_status=0 b_status=0
    ("$TEST_DB_SH" run -- "$payload" >"$TEST_TMP/a.log" 2>&1; echo $? >"$TEST_TMP/a.status") &
    local a_pid=$!
    ("$TEST_DB_SH" run -- "$payload" >"$TEST_TMP/b.log" 2>&1; echo $? >"$TEST_TMP/b.status") &
    local b_pid=$!
    wait "$a_pid" || true
    wait "$b_pid" || true
    a_status="$(cat "$TEST_TMP/a.status")"
    b_status="$(cat "$TEST_TMP/b.status")"
    if [ "$a_status" != 0 ] || [ "$b_status" != 0 ]; then
        printf -- '--- run A (status %s) ---\n' "$a_status" >&2
        cat "$TEST_TMP/a.log" >&2
        printf -- '--- run B (status %s) ---\n' "$b_status" >&2
        cat "$TEST_TMP/b.log" >&2
        fail 'both concurrent invocations must pass'
    fi
    # Different databases, not one shared container.
    local a_name b_name
    a_name="$(grep -o 'adele-testdb-[a-z0-9-]*' "$TEST_TMP/a.log" | head -1)"
    b_name="$(grep -o 'adele-testdb-[a-z0-9-]*' "$TEST_TMP/b.log" | head -1)"
    [ -n "$a_name" ] && [ -n "$b_name" ] || fail 'could not identify both containers'
    [ "$a_name" != "$b_name" ] || fail "both runs used the same container: $a_name"
}

container_runtime_available() {
    if [ -n "${CONTAINER_CLI:-}" ]; then "$CONTAINER_CLI" info >/dev/null 2>&1; return; fi
    podman info >/dev/null 2>&1 || docker info >/dev/null 2>&1
}

run_test test_db_uses_a_unique_container_name_per_invocation
run_test test_db_exports_the_url_of_the_container_it_created
run_test test_db_cleans_up_its_own_container_on_success_and_on_failure
run_test test_db_does_not_remove_a_container_it_did_not_create
run_test test_db_reports_a_clear_error_when_no_free_port_is_available
run_test test_db_down_removes_only_leftover_harness_containers
if container_runtime_available; then
    run_test two_concurrent_test_db_runs_both_pass
else
    skip_test two_concurrent_test_db_runs_both_pass \
        'no reachable podman/docker; the parallel-safety fix is UNVERIFIED here. Start a runtime (or set CONTAINER_CLI) and re-run: just test-scripts'
fi
finish_tests 'test-db-harness'

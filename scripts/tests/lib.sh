# shellcheck shell=bash
# Minimal named-test harness for the shell-level gate scripts under scripts/.
#
# The code under test is shell (the `just check` steps), so the tests are shell
# too. Each acceptance criterion is one function whose name IS the criterion,
# and that name is printed on every result line, so a failing run names the
# unmet requirement instead of a line number.
#
# Sourced by scripts/tests/*.test.sh; run them all with `just test-scripts`.

SCRIPT_TESTS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_TESTS_FIXTURES="$SCRIPT_TESTS_ROOT/scripts/tests/fixtures"

_tests_total=0
_tests_failed=0
_tests_skipped=0
_skip_hints=()

# Run one named test in a subshell with a private temp dir in $TEST_TMP.
# Output is captured and printed only when the test fails.
run_test() {
    local name="$1" out status=0
    _tests_total=$((_tests_total + 1))
    TEST_TMP="$(mktemp -d "${TMPDIR:-/tmp}/adele-script-test.XXXXXXXX")"
    # Deliberately NOT `out="$(...)" || status=$?`: bash suppresses errexit for
    # the whole subshell when it sits in a `||` list, even after an explicit
    # `set -e` inside it, which lets a failed assertion keep going and the test
    # pass on its last line.
    set +e
    out="$( (set -euo pipefail; "$name") 2>&1 )"
    status=$?
    set -e
    rm -rf "$TEST_TMP"
    if [ "$status" -eq 0 ]; then
        printf 'ok       %s\n' "$name"
        return 0
    fi
    _tests_failed=$((_tests_failed + 1))
    printf 'NOT OK   %s\n' "$name"
    printf '%s\n' "$out" | sed 's/^/           /'
}

# Record a test that did not run. The reason and the command that WOULD run it
# are repeated in a loud end-of-suite banner: a quiet skip once read as
# "covered" when nothing had been verified (see crates/storage/tests/support).
skip_test() {
    local name="$1" why="$2"
    _tests_total=$((_tests_total + 1))
    _tests_skipped=$((_tests_skipped + 1))
    _skip_hints+=("$name - $why")
    printf 'SKIPPED  %s\n' "$name"
}

# Print the suite summary; non-zero exit when any test failed.
finish_tests() {
    local suite="$1"
    if [ "${#_skip_hints[@]}" -gt 0 ]; then
        {
            printf '\n'
            printf '!!  %s: %d test(s) did NOT run - this suite verified nothing about them:\n' \
                "$suite" "$_tests_skipped"
            printf '!!    %s\n' "${_skip_hints[@]}"
        } >&2
    fi
    printf '\n%s: %d test(s), %d failed, %d skipped\n' \
        "$suite" "$_tests_total" "$_tests_failed" "$_tests_skipped"
    [ "$_tests_failed" -eq 0 ]
}

# --- assertions --------------------------------------------------------------

# Ends the test immediately. Each test body runs in its own subshell, so an
# exit here cannot be swallowed the way a non-zero return can be.
fail() {
    printf 'assertion failed: %s\n' "$*" >&2
    exit 1
}

assert_eq() { # assert_eq <expected> <actual> [what]
    [ "$1" = "$2" ] || fail "$(printf '%s: expected %q, got %q' "${3:-value}" "$1" "$2")"
}

assert_contains() { # assert_contains <haystack> <needle> [what]
    case "$1" in
        *"$2"*) return 0 ;;
    esac
    fail "$(printf '%s does not contain %q\n--- actual ---\n%s\n--------------' \
        "${3:-output}" "$2" "$1")"
}

assert_not_contains() { # assert_not_contains <haystack> <needle> [what]
    case "$1" in
        *"$2"*)
            fail "$(printf '%s must NOT contain %q\n--- actual ---\n%s\n--------------' \
                "${3:-output}" "$2" "$1")"
            ;;
    esac
}

# Run a command, capturing its streams and status instead of aborting.
# Sets RUN_STATUS, RUN_OUT (stdout) and RUN_ERR (stderr).
run_cmd() {
    RUN_STATUS=0
    "$@" >"$TEST_TMP/stdout" 2>"$TEST_TMP/stderr" || RUN_STATUS=$?
    RUN_OUT="$(cat "$TEST_TMP/stdout")"
    RUN_ERR="$(cat "$TEST_TMP/stderr")"
}

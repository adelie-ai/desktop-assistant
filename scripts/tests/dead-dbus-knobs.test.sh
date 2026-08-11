#!/usr/bin/env bash
# Acceptance criteria for the D-Bus environment knobs (#815, #834).
#
# The cutover (#281/#318/#319) took the in-process D-Bus surface out of the
# daemon. Two environment variables survived the deletion in prose and in
# shipped manifests: `DESKTOP_ASSISTANT_DBUS_INPROCESS` (documented as the way
# to bring the old surface back) and `DESKTOP_ASSISTANT_DBUS_REQUIRED` (set in
# the Dockerfiles and the k8s manifest). No code has read either one since the
# cutover, so both were inert.
#
# An inert knob that reads as a supported control is worse than no knob. An
# operator who follows a documented revert sets the variable, changes the unit
# to `Type=dbus` with `BusName=org.desktopAssistant`, and restarts. Nothing
# claims the name, systemd waits for a bus name that never arrives, and the
# unit fails - so the daemon joins the bridge in being down.
#
# This asserts the property rather than the old names: every
# `DESKTOP_ASSISTANT_DBUS_*` name written anywhere in the repository must be a
# name the Rust code actually reads. A new dead knob is caught without editing
# this file.
set -euo pipefail
# shellcheck source=scripts/tests/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SUITE_FILE='scripts/tests/dead-dbus-knobs.test.sh'
NAME_RE='DESKTOP_ASSISTANT_DBUS_[A-Z0-9_]+'

# Names the code reads. A read is a quoted string literal on a line of Rust
# that is not a comment: `std::env::var("...")`, `env_bool("...")`, or clap's
# `env = "..."`. Comments are stripped first, so a name that appears only in
# prose never counts as a read.
_names_read_by_code() {
    cd "$SCRIPT_TESTS_ROOT"
    git ls-files -z 'crates/*.rs' \
        | xargs -0 sed 's://.*::' \
        | grep -oE "\"$NAME_RE\"" \
        | tr -d '"' \
        | sort -u
}

# Every place a name is written, as `file:line:name`. This suite is excluded
# because it names the dead knobs itself, in the comment above.
_all_mentions() {
    cd "$SCRIPT_TESTS_ROOT"
    git ls-files -z \
        | grep -zv "^$SUITE_FILE\$" \
        | xargs -0 grep -noHIE "$NAME_RE" -- 2>/dev/null \
        | sort -u
}

the_mention_scan_reaches_the_repository() {
    # Without this, a scan that silently returned nothing would report every
    # criterion below as met while verifying none of them.
    local mentions
    mentions="$(_all_mentions)"
    assert_contains "$mentions" 'DESKTOP_ASSISTANT_DBUS_SERVICE' \
        'the mention scan should see the live knob the justfile sets'
}

every_dbus_env_name_written_in_the_repository_is_one_the_code_reads() {
    local read_set offenders name
    read_set="$TEST_TMP/read"
    _names_read_by_code >"$read_set"
    offenders=''
    while IFS= read -r hit; do
        [ -n "$hit" ] || continue
        name="${hit##*:}"
        grep -qxF "$name" "$read_set" || offenders="$offenders
  $hit"
    done < <(_all_mentions)
    [ -z "$offenders" ] || fail "$(printf 'these name a DESKTOP_ASSISTANT_DBUS_* knob no code reads:%s\n--- names the code reads ---\n%s' \
        "$offenders" "$(cat "$read_set")")"
}

run_test the_mention_scan_reaches_the_repository
run_test every_dbus_env_name_written_in_the_repository_is_one_the_code_reads
finish_tests 'dead-dbus-knobs'

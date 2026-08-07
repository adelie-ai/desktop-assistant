#!/usr/bin/env bash
# Acceptance criteria for the shipped units' log level (#776).
#
# Four tracing targets log conversation content verbatim at `debug`:
#   - desktop_assistant_core::service   whole tool results and tool arguments
#   - desktop_assistant_mcp_client      every MCP JSON-RPC request and response,
#                                       in full, plus every search query
#   - desktop_assistant_llm_openai      the head of each assembled prompt
#   - desktop_assistant_storage         each extracted personal fact, in full
#
# The level contract puts conversation content at `debug` on purpose, so this
# list grows whenever a content-carrying target does. A target that logs
# content but is missing here can be raised to `debug` in a shipped unit with
# nothing to catch it.
#
# `just install-service` copies systemd/desktop-assistant-daemon.service verbatim
# into the user's systemd directory, so a `debug` on any of those in a shipped
# unit puts their documents, files and prompts into the journal in cleartext -
# readable by anything that can run journalctl, and shipped to a collector under
# the k8s manifests. That is how #776 happened: the unit carried all three and
# nothing noticed.
#
# These assert the property rather than the old string, so a NEW unit, or a new
# target added to the set below, is covered without editing a test.
set -euo pipefail
# shellcheck source=scripts/tests/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

CONTENT_TARGETS=(
    'desktop_assistant_core::service'
    'desktop_assistant_mcp_client'
    'desktop_assistant_llm_openai'
    'desktop_assistant_storage'
)

_units() { find "$SCRIPT_TESTS_ROOT/systemd" -name '*.service' | sort; }
_rust_log() { grep -h '^Environment=RUST_LOG=' "$1" 2>/dev/null || true; }

no_shipped_unit_logs_conversation_content_to_the_journal() {
    local unit line target offenders=''
    while read -r unit; do
        line="$(_rust_log "$unit")"
        [ -n "$line" ] || continue
        for target in "${CONTENT_TARGETS[@]}"; do
            case "$line" in
                *"$target=debug"*|*"$target=trace"*)
                    offenders="$offenders $(basename "$unit"):$target" ;;
            esac
        done
    done < <(_units)
    [ -z "$offenders" ] || fail "these shipped units put user content in the journal:$offenders"
}

every_shipped_unit_defaults_to_info_or_quieter() {
    local unit line level
    while read -r unit; do
        line="$(_rust_log "$unit")"
        [ -n "$line" ] || continue
        level="${line#Environment=RUST_LOG=}"
        level="${level%%,*}"
        case "$level" in
            error|warn|info) ;;
            *) fail "$(basename "$unit") sets a global RUST_LOG of '$level'; the shipped default must be info or quieter" ;;
        esac
    done < <(_units)
}

the_daemon_unit_shows_how_to_raise_one_target_by_hand() {
    # Without this, the only way a user knows to debug is to turn everything on,
    # which is what the unit used to ship.
    assert_contains "$(cat "$SCRIPT_TESTS_ROOT/systemd/desktop-assistant-daemon.service")" \
        'RUST_LOG=info,desktop_assistant_mcp_client=debug' \
        'the unit should demonstrate the single-target form'
}

run_test no_shipped_unit_logs_conversation_content_to_the_journal
run_test every_shipped_unit_defaults_to_info_or_quieter
run_test the_daemon_unit_shows_how_to_raise_one_target_by_hand
finish_tests 'systemd-logging'

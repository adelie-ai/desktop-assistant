#!/usr/bin/env bash
# Fake `cargo audit` for the audit-gate tests. Installed on PATH under the name
# `cargo-audit`, which is how cargo dispatches the `cargo audit` subcommand, so
# scripts/audit.sh runs its real code path with no network fetch and no live
# advisory database.
#
# Knobs (all optional):
#   FAKE_AUDIT_LOG     append each invocation's arguments to this file
#   FAKE_AUDIT_STDOUT  emit this on stdout (the canned scan report)
#   FAKE_AUDIT_STDERR  emit this on stderr
#   FAKE_AUDIT_STATUS  exit with this status (default 0)
set -uo pipefail

# cargo passes the subcommand name as the first argument (`cargo audit --json`
# execs `cargo-audit audit --json`), so drop it before recording the flags.
[ "${1:-}" != audit ] || shift

[ -z "${FAKE_AUDIT_LOG:-}" ] || printf '%s\n' "$*" >>"$FAKE_AUDIT_LOG"

case "${1:-}" in
    --version | -V)
        printf 'cargo-audit-audit 0.0.0-fake\n'
        exit 0
        ;;
esac

[ -z "${FAKE_AUDIT_STDOUT:-}" ] || printf '%s' "$FAKE_AUDIT_STDOUT"
[ -z "${FAKE_AUDIT_STDERR:-}" ] || printf '%s\n' "$FAKE_AUDIT_STDERR" >&2
exit "${FAKE_AUDIT_STATUS:-0}"

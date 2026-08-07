#!/usr/bin/env bash
# Fake `cargo` for the no-opentelemetry gate tests. Installed on PATH ahead of
# the real cargo so scripts/no-opentelemetry.sh runs its own code path against
# a dependency tree this test chose - including trees the real workspace will
# not produce on demand, which are the ones the guard exists for.
#
# Answers only the one subcommand scripts/no-opentelemetry.sh uses.
#
# Knobs (all optional):
#   FAKE_TREE_STDOUT  emit this from `cargo tree`
#   FAKE_TREE_STATUS  exit status of `cargo tree` (default 0)
set -uo pipefail

case "${1:-}" in
    tree)
        [ -z "${FAKE_TREE_STDOUT:-}" ] || printf '%s\n' "$FAKE_TREE_STDOUT"
        exit "${FAKE_TREE_STATUS:-0}"
        ;;
    *)
        printf 'fake cargo: unexpected subcommand %q\n' "${1:-}" >&2
        exit 127
        ;;
esac

#!/usr/bin/env bash
# Fake `cargo` for the doc-gate tests. Installed on PATH ahead of the real
# cargo so scripts/doc.sh runs its own code path against a cargo that reports
# success while producing nothing - the outcome a real cargo will not produce
# on demand, and the one the "silence is not success" guard exists for.
#
# Answers only the three subcommands scripts/doc.sh uses.
#
# Knobs (all optional):
#   FAKE_CARGO_TARGET_DIR  target_directory reported by `cargo metadata`
#   FAKE_CARGO_PACKAGES    space-separated package names `cargo tree` reports
#   FAKE_CARGO_DOC_STDOUT  emit this from `cargo doc`
#   FAKE_CARGO_DOC_STATUS  exit status of `cargo doc` (default 0)
set -uo pipefail

case "${1:-}" in
    metadata)
        printf '{"target_directory":"%s","packages":[],"workspace_members":[]}\n' \
            "${FAKE_CARGO_TARGET_DIR:-/nonexistent}"
        ;;
    tree)
        for pkg in ${FAKE_CARGO_PACKAGES:-}; do
            printf '%s v0.0.0 (/fake/%s)\n\n' "$pkg" "$pkg"
        done
        ;;
    doc)
        [ -z "${FAKE_CARGO_DOC_STDOUT:-}" ] || printf '%s\n' "$FAKE_CARGO_DOC_STDOUT"
        exit "${FAKE_CARGO_DOC_STATUS:-0}"
        ;;
    *)
        printf 'fake cargo: unexpected subcommand %q\n' "${1:-}" >&2
        exit 127
        ;;
esac

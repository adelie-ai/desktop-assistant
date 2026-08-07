#!/usr/bin/env bash
# Fake `cargo` for the no-opentelemetry gate tests. Installed on PATH ahead of
# the real cargo so scripts/no-opentelemetry.sh runs its own code path against a
# dependency tree this test chose - including trees the real workspace will not
# produce on demand, which are the ones the guard exists for.
#
# The script under test calls `cargo tree` twice, and they answer different
# questions, so this fixture tells them apart by `--depth 0`:
#   without --depth  the full resolved tree
#   with --depth 0   the workspace member list
#
# Knobs (all optional):
#   FAKE_TREE_STDOUT     emit this from the full-tree call
#   FAKE_TREE_STDERR     emit this on stderr from the full-tree call
#   FAKE_TREE_STATUS     exit status of the full-tree call (default 0)
#   FAKE_MEMBERS_STDOUT  emit this from the member-list call
#   FAKE_MEMBERS_STDERR  emit this on stderr from the member-list call
#   FAKE_MEMBERS_STATUS  exit status of the member-list call (default 0)
#
# The stderr knobs carry what real cargo says about itself rather than about
# the tree - "Blocking waiting for file lock on package cache" and the like.
# A caller that parses stdout is entitled to ignore all of it; one that merges
# the two streams reads the first word of each line as a package name.
set -uo pipefail

case "${1:-}" in
    tree) ;;
    *)
        printf 'fake cargo: unexpected subcommand %q\n' "${1:-}" >&2
        exit 127
        ;;
esac

for arg in "$@"; do
    if [ "$arg" = '--depth' ]; then
        [ -z "${FAKE_MEMBERS_STDERR:-}" ] || printf '%s\n' "$FAKE_MEMBERS_STDERR" >&2
        [ -z "${FAKE_MEMBERS_STDOUT:-}" ] || printf '%s\n' "$FAKE_MEMBERS_STDOUT"
        exit "${FAKE_MEMBERS_STATUS:-0}"
    fi
done

[ -z "${FAKE_TREE_STDERR:-}" ] || printf '%s\n' "$FAKE_TREE_STDERR" >&2
[ -z "${FAKE_TREE_STDOUT:-}" ] || printf '%s\n' "$FAKE_TREE_STDOUT"
exit "${FAKE_TREE_STATUS:-0}"

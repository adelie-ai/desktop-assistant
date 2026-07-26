#!/usr/bin/env bash
# Fake podman/docker for the test-db harness tests. Injected through
# CONTAINER_CLI (the same knob operators use), so scripts/test-db.sh runs its
# real control flow - naming, port readback, readiness polling, teardown -
# without a container runtime, a network, or a 10-second Postgres boot.
#
# Every invocation's arguments are appended to $FAKE_CLI_LOG, so a test can
# assert exactly which containers were created and which were removed.
#
# Knobs (all optional):
#   FAKE_RUN_STATUS   exit status for `run` (default 0)
#   FAKE_RUN_STDERR   stderr for `run` (e.g. a port-in-use error)
#   FAKE_PORT         host port reported by `port` (default 15432)
#   FAKE_PORT_STATUS  exit status for `port` (default 0)
#   FAKE_READY_STATUS exit status for `exec` i.e. pg_isready (default 0)
#   FAKE_RUNNING      value reported by `inspect` (default true)
#   FAKE_PS_NAMES     newline-separated container names reported by `ps`
set -uo pipefail

printf '%s\n' "$*" >>"${FAKE_CLI_LOG:?FAKE_CLI_LOG must be set}"

case "${1:-}" in
    info)
        exit "${FAKE_INFO_STATUS:-0}"
        ;;
    run)
        [ -z "${FAKE_RUN_STDERR:-}" ] || printf '%s\n' "$FAKE_RUN_STDERR" >&2
        [ "${FAKE_RUN_STATUS:-0}" = 0 ] || exit "${FAKE_RUN_STATUS}"
        printf '0123456789abcdef\n'
        ;;
    port)
        [ "${FAKE_PORT_STATUS:-0}" = 0 ] || exit "${FAKE_PORT_STATUS}"
        printf '127.0.0.1:%s\n' "${FAKE_PORT:-15432}"
        ;;
    inspect)
        printf '%s\n' "${FAKE_RUNNING:-true}"
        ;;
    exec)
        exit "${FAKE_READY_STATUS:-0}"
        ;;
    logs)
        printf 'fake-container-cli: canned container log\n'
        ;;
    ps)
        [ -z "${FAKE_PS_NAMES:-}" ] || printf '%s\n' "$FAKE_PS_NAMES"
        ;;
    rm) ;;
    *)
        printf 'fake-container-cli: unhandled invocation: %s\n' "$*" >&2
        exit 64
        ;;
esac

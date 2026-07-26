#!/usr/bin/env bash
# Throwaway pgvector container for the DB-gated storage suites (`just test-db`).
#
# Every invocation gets its own container name and its own host port, and only
# ever removes the container it created, so two sessions can run the storage
# suites at the same time (#662). The previous fixed name and fixed port made a
# second run delete the first run's database mid-test, which surfaced as a
# plausible-looking flake in an unrelated test.
#
# Subcommands:
#   start            provision a container, print its settings as shell exports
#   run -- CMD...    provision, run CMD with TEST_DATABASE_URL set, tear down
#   stop NAME...     remove named containers (refuses anything foreign)
#   prune            remove every container of this harness that no live run
#                    owns - a `run` in flight keeps its database, a container
#                    from `start` has no live owner and is swept
#
# Environment:
#   CONTAINER_CLI            podman|docker (auto-detected when unset)
#   TEST_DB_IMAGE            image to run
#   TEST_DB_PORT             fixed host port; unset lets the runtime pick a free one
#   TEST_DB_MAX_CONNECTIONS  Postgres max_connections for the container
#   TEST_DB_READY_TIMEOUT    seconds to wait for Postgres to accept TCP
#   TEST_DB_INITDB           directory of init fixtures to mount
set -euo pipefail

NAME_PREFIX='adele-testdb'
# Marks containers as ours so `prune` can find leftovers without guessing.
LABEL='adele-testdb=1'
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE="${TEST_DB_IMAGE:-docker.io/pgvector/pgvector:pg17}"
INITDB="${TEST_DB_INITDB:-$REPO_ROOT/crates/storage/tests/fixtures/initdb}"
READY_TIMEOUT="${TEST_DB_READY_TIMEOUT:-90}"
# Each DbFixture opens its own pool and the suites run in parallel, so leave
# headroom well above Postgres' default of 100 connections.
MAX_CONNECTIONS="${TEST_DB_MAX_CONNECTIONS:-300}"

# Set only once this invocation has created a container, so teardown can never
# name anything else.
CREATED_NAME=''
CREATED_PORT=''

log() { printf 'test-db: %s\n' "$*" >&2; }

die() {
    printf 'test-db: %s\n' "$1" >&2
    shift
    [ "$#" -eq 0 ] || printf '  %s\n' "$@" >&2
    exit 1
}

remove_created() {
    [ -n "$CREATED_NAME" ] || return 0
    local name="$CREATED_NAME"
    CREATED_NAME=''
    "$CLI" rm -f "$name" >/dev/null 2>&1 || true
    log "removed $name"
}

resolve_cli() {
    if [ -n "${CONTAINER_CLI:-}" ]; then
        CLI="$CONTAINER_CLI"
        return 0
    fi
    if podman info >/dev/null 2>&1; then
        CLI=podman
    elif docker info >/dev/null 2>&1; then
        CLI=docker
    else
        die 'no reachable container runtime' \
            'Start podman or docker, or point CONTAINER_CLI at the one to use.'
    fi
    log "using container runtime '$CLI'"
    # Exported so a payload (and `stop` later on) uses the same runtime.
    export CONTAINER_CLI="$CLI"
}

# Refuse to remove anything this harness could not have created.
assert_ours() {
    case "$1" in
        "$NAME_PREFIX"-*) return 0 ;;
    esac
    die "refusing to remove '$1'" \
        "It is not a test-db container (those are named ${NAME_PREFIX}-<pid>-<random>)." \
        'Remove it yourself if you are sure.'
}

unique_name() {
    printf '%s-%s-%s' "$NAME_PREFIX" "$$" "$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n')"
}

start_container() {
    local name publish
    name="$(unique_name)"
    if [ -n "${TEST_DB_PORT:-}" ]; then
        publish="127.0.0.1:${TEST_DB_PORT}:5432"
    else
        # Let the runtime assign a free host port: asking it is race-free,
        # whereas probing for a free port and then binding it is not.
        publish='127.0.0.1::5432'
    fi

    local run_err
    run_err="$(mktemp)"
    if ! "$CLI" run --rm -d --name "$name" --label "$LABEL" \
        -e POSTGRES_PASSWORD=test -e POSTGRES_DB=postgres \
        -p "$publish" \
        -v "$INITDB:/docker-entrypoint-initdb.d:ro,z" \
        "$IMAGE" -c "max_connections=$MAX_CONNECTIONS" >/dev/null 2>"$run_err"; then
        local detail
        detail="$(sed 's/^/  /' "$run_err" | head -20 || true)"
        rm -f "$run_err"
        local port_note='The runtime could not publish a host port for the test database.'
        if [ -n "${TEST_DB_PORT:-}" ]; then
            port_note="Host port ${TEST_DB_PORT} (from TEST_DB_PORT) could not be published; it is probably in use."
        fi
        die 'could not start the throwaway Postgres container' \
            "$port_note" \
            "$detail" \
            'Unset TEST_DB_PORT to let the runtime pick a free port, or free the port and re-run.'
    fi
    rm -f "$run_err"

    # From here on this invocation owns the container, including on failure.
    CREATED_NAME="$name"
    trap remove_created EXIT
    trap 'remove_created; exit 130' INT TERM

    # Capture the status rather than letting the pipeline carry it: under
    # `set -euo pipefail` a failing `port` aborts at the assignment, which took
    # the diagnostic below with it. That is the case that most needs one - the
    # container died right after `run` and `--rm` already removed it.
    local port_out port_err port_status=0
    port_err="$(mktemp)"
    port_out="$("$CLI" port "$name" 5432/tcp 2>"$port_err")" || port_status=$?
    # First line, then whatever follows the last colon - `podman port` answers
    # with one line per binding, and an IPv6 one has colons of its own. Done
    # with expansions rather than `head | sed`, so a multi-line answer cannot
    # hand the assignment an EPIPE and abort the script instead of reporting.
    CREATED_PORT="${port_out%%$'\n'*}"
    CREATED_PORT="${CREATED_PORT##*:}"
    local bad_port=''
    case "$CREATED_PORT" in
        '' | *[!0-9]*) bad_port=1 ;;
    esac
    if [ "$port_status" -ne 0 ] || [ -n "$bad_port" ]; then
        local port_detail
        port_detail="$( { [ -z "$port_out" ] || printf '%s\n' "$port_out"; cat "$port_err"; } |
            sed 's/^/  /' | head -10 || true)"
        rm -f "$port_err"
        die "could not read the published host port of $name" \
            "'$CLI port $name 5432/tcp' exited $port_status and reported:" \
            "$port_detail" \
            'A container that has already exited is the usual cause; --rm then' \
            'removes it before it can be inspected.'
    fi
    rm -f "$port_err"

    wait_for_postgres "$name"
    log "$name is ready on 127.0.0.1:$CREATED_PORT"
}

wait_for_postgres() {
    local name="$1" i
    for ((i = 1; i <= READY_TIMEOUT; i++)); do
        # -h forces a TCP connection *inside* the container. The image's
        # entrypoint runs the init fixtures against a socket-only temporary
        # server, so TCP accepting connections means initialisation finished.
        if "$CLI" exec "$name" pg_isready -h 127.0.0.1 -U postgres -q >/dev/null 2>&1; then
            return 0
        fi
        if [ "$("$CLI" inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]; then
            break
        fi
        sleep 1
    done
    log "container log for $name:"
    "$CLI" logs --tail 40 "$name" >&2 || true
    die "$name never accepted connections (waited ${READY_TIMEOUT}s)" \
        'The container log is above. Set TEST_DB_READY_TIMEOUT to wait longer.'
}

database_url() {
    printf 'postgres://postgres:test@127.0.0.1:%s/postgres' "$CREATED_PORT"
}

cmd_start() {
    resolve_cli
    start_container
    # The caller keeps the container, so drop the teardown this invocation
    # installed; `just test-db-down` removes it.
    trap - EXIT INT TERM
    # No process owns this container once this one exits, so `prune` will sweep
    # it - along with any other session's. Name it and only it comes down.
    log "tear down with: just test-db-down $CREATED_NAME"
    printf "export ADELE_TEST_DB_CONTAINER='%s'\n" "$CREATED_NAME"
    printf "export TEST_DATABASE_URL='%s'\n" "$(database_url)"
    CREATED_NAME=''
}

cmd_run() {
    [ "$#" -gt 0 ] || die 'run needs a command: scripts/test-db.sh run -- cargo test ...'
    resolve_cli
    start_container
    export ADELE_TEST_DB_CONTAINER="$CREATED_NAME"
    export TEST_DATABASE_URL="$(database_url)"
    local status=0
    "$@" || status=$?
    # remove_created runs from the EXIT trap, for this path and every failure
    # path above it.
    exit "$status"
}

cmd_stop() {
    [ "$#" -gt 0 ] || die 'stop needs a container name (or use `prune`)'
    resolve_cli
    local name
    for name in "$@"; do
        assert_ours "$name"
    done
    for name in "$@"; do
        "$CLI" rm -f "$name" >/dev/null 2>&1 || true
        log "removed $name"
    done
}

cmd_prune() {
    resolve_cli
    local names name pid pruned=0 kept=0
    names="$("$CLI" ps -a --filter "label=$LABEL" --format '{{.Names}}' 2>/dev/null || true)"
    while IFS= read -r name; do
        [ -n "$name" ] || continue
        case "$name" in
            "$NAME_PREFIX"-*) ;;
            *) continue ;;
        esac
        # The name carries the pid of the invocation that created the
        # container, and a live pid means the container is in use rather than
        # left behind. Removing that one is #662 again: a database vanishing
        # mid-test, seen from the other session as a flake in an unrelated
        # suite. Where the check is inexact it errs towards keeping - a
        # recycled pid costs a leftover one more sweep, and unique names mean a
        # leftover never blocks a run. A name with no numeric pid cannot have
        # come from this script, so nothing owns it. (Another user's pid is not
        # signalable, so a shared rootful daemon would still sweep theirs.)
        pid="${name#"$NAME_PREFIX"-}"
        pid="${pid%%-*}"
        if [ -n "$pid" ] && [ -z "${pid//[0-9]/}" ] && kill -0 "$pid" 2>/dev/null; then
            log "keeping $name - its run (pid $pid) is still going"
            kept=$((kept + 1))
            continue
        fi
        "$CLI" rm -f "$name" >/dev/null 2>&1 || true
        log "removed leftover $name"
        pruned=$((pruned + 1))
    done <<<"$names"
    log "pruned $pruned leftover container(s), kept $kept still in use"
}

case "${1:-}" in
    start)
        cmd_start
        ;;
    run)
        shift
        [ "${1:-}" != '--' ] || shift
        cmd_run "$@"
        ;;
    stop)
        shift
        cmd_stop "$@"
        ;;
    prune)
        cmd_prune
        ;;
    *)
        die "unknown subcommand '${1:-}'" 'Expected one of: start, run -- CMD..., stop NAME..., prune'
        ;;
esac

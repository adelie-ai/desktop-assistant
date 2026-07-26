#!/usr/bin/env bash
# Payload for the `just test-db` concurrency test. Stands in for `cargo test`:
# it proves the database this invocation was handed is its own, is fully
# provisioned, and survives for the whole run while another invocation of the
# same harness starts and finishes alongside it.
set -euo pipefail

: "${TEST_DATABASE_URL:?payload needs TEST_DATABASE_URL}"
: "${ADELE_TEST_DB_CONTAINER:?payload needs ADELE_TEST_DB_CONTAINER}"
: "${CONTAINER_CLI:?payload needs CONTAINER_CLI}"

url_port="${TEST_DATABASE_URL##*:}"
url_port="${url_port%%/*}"

published="$("$CONTAINER_CLI" port "$ADELE_TEST_DB_CONTAINER" 5432/tcp | head -1)"
published="${published##*:}"
if [ "$url_port" != "$published" ]; then
    printf 'TEST_DATABASE_URL port %s is not the port of %s (%s)\n' \
        "$url_port" "$ADELE_TEST_DB_CONTAINER" "$published" >&2
    exit 1
fi

query() {
    "$CONTAINER_CLI" exec "$ADELE_TEST_DB_CONTAINER" psql -U postgres -tAc "$1"
}

# The init fixture under crates/storage/tests/fixtures/initdb must have run.
extensions="$(query "select count(*) from pg_extension where extname = 'vector'")"
if [ "${extensions// /}" != 1 ]; then
    printf 'the vector extension is missing from %s\n' "$ADELE_TEST_DB_CONTAINER" >&2
    exit 1
fi

# Hold the database open long enough for a concurrent invocation to provision
# and tear down its own. Under a shared container name this is where the other
# run's teardown deleted this run's database mid-test.
for _ in $(seq 1 "${ASSERT_DB_HOLD_SECONDS:-8}"); do
    if ! query 'select 1' >/dev/null 2>&1; then
        printf 'database %s went away mid-run\n' "$ADELE_TEST_DB_CONTAINER" >&2
        exit 1
    fi
    sleep 1
done

printf '%s stayed healthy on port %s\n' "$ADELE_TEST_DB_CONTAINER" "$url_port"

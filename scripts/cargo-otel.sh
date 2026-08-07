#!/bin/sh
# Run cargo, with the telemetry exporter compiled in or left out.
#
# Both container images build the same crates two ways. Export lives behind an
# off-by-default Cargo feature named `otel`, so a default image is what it has
# always been: console output on stderr, the periodic metrics summary, and no
# opentelemetry crate anywhere in the tree. An image built with the feature on
# can also ship traces, metrics and log records to a collector.
#
# The images call cargo only through this file, so the rule lives in one place
# and can be run outside a container. Usage:
#
#     OTEL=1 cargo-otel.sh build --release --locked -p desktop-assistant-daemon
#
# Every argument is passed to cargo unchanged; `--features otel` is appended
# when OTEL says so.
#
# A value this file does not recognise stops the build. The alternative - treat
# anything unfamiliar as "off" - produces an image the operator believes is
# instrumented and that exports nothing, and nothing later in the build or the
# deployment says otherwise.
set -eu

case "${OTEL:-0}" in
    "" | 0)
        features=''
        ;;
    1)
        features='--features otel'
        ;;
    *)
        echo "cargo-otel: OTEL must be 0 or 1, not '${OTEL}'." >&2
        echo "cargo-otel: 1 compiles the OTLP exporter in; 0 (the default) leaves it out." >&2
        exit 64
        ;;
esac

# Unquoted on purpose: empty must expand to no argument at all, and the value
# is one of the two literals above, never caller input.
# shellcheck disable=SC2086
exec cargo "$@" ${features}

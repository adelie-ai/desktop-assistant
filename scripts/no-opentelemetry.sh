#!/usr/bin/env bash
# The no-opentelemetry step of the gate.
#
# Telemetry export is behind an off-by-default Cargo feature named `otel`. The
# promise that goes with it is that a default build resolves no opentelemetry
# crate at all, so a desktop install from `cargo install` pays nothing for it:
# no extra crates, no native code, no C toolchain.
#
# Nothing in the source states that promise, and one `features = ["otel"]`
# written on a dependency instead of a passthrough turns export on for every
# build while every other step of the gate stays green. So the property is
# checked against the resolved dependency tree.
#
# Like scripts/audit.sh, scripts/secret-scan.sh and scripts/doc.sh, this cannot
# pass by accident. A `cargo tree` that errors, or that reports a tree far too
# small to be this workspace, is a failure and not a clean result.
#
# Usage: scripts/no-opentelemetry.sh
#   Acts on the workspace containing the current directory, so the tests can
#   point it at a throwaway fixture.
set -euo pipefail

# Below this many distinct crates the run proved nothing: this workspace
# resolves several hundred, so a handful means cargo selected almost nothing.
MIN_CRATES=50

loud() { # loud <headline> <line>...
    local headline="$1"
    shift
    {
        printf '\n'
        printf '  %s\n' '======================================================================'
        printf '  %s\n' "$headline"
        printf '  %s\n' '======================================================================'
        printf '  %s\n' "$@"
        printf '\n'
    } >&2
}

die_loud() {
    loud "$@"
    exit 1
}

# `--edges normal,build` is what a build actually compiles: dependencies and
# build scripts. Dev-dependencies are excluded on purpose - a test-only crate
# never reaches a shipped binary, and excluding them keeps the check about what
# `cargo install` produces.
tree_out="$(mktemp)"
trap 'rm -f "$tree_out"' EXIT

tree_status=0
cargo tree --workspace --edges normal,build --prefix none >"$tree_out" 2>&1 || tree_status=$?

if [ "$tree_status" -ne 0 ]; then
    die_loud 'NO-OPENTELEMETRY STEP DID NOT RUN: cargo could not resolve the tree' \
        'The check proves nothing when the tree cannot be read, so this is a failure' \
        'and not a clean result. Fix the resolution error first.' \
        '' \
        "$(sed 's/^/    /' "$tree_out" | head -40)"
fi

mapfile -t crates < <(awk 'NF { print $1 }' "$tree_out" | sort -u)

if [ "${#crates[@]}" -lt "$MIN_CRATES" ]; then
    die_loud 'NO-OPENTELEMETRY STEP RESOLVED ALMOST NOTHING - hard gate failure' \
        "cargo tree reported ${#crates[@]} distinct crate(s); this workspace resolves" \
        "several hundred. Fewer than $MIN_CRATES means the invocation selected almost no" \
        'packages, so a green result here would read as coverage it does not have.'
fi

# Substring, not a prefix: `tracing-opentelemetry` is the layer that bridges
# tracing to the SDK, and a check anchored at the start of the name would let
# the whole pipeline through.
offenders=()
for name in "${crates[@]}"; do
    case "$name" in
        *opentelemetry*) offenders+=("$name") ;;
    esac
done

if [ "${#offenders[@]}" -gt 0 ]; then
    die_loud 'OPENTELEMETRY IN A DEFAULT BUILD - hard gate failure' \
        'A default-feature build must resolve no opentelemetry crate. These were:' \
        '' \
        "$(printf '    %s\n' "${offenders[@]}")" \
        '' \
        'Usually one crate wrote `features = ["otel"]` on its adelie-telemetry' \
        'dependency instead of declaring its own passthrough:' \
        '' \
        '    [features]' \
        '    otel = ["adelie-telemetry/otel"]' \
        '' \
        'Find the path with:' \
        '' \
        '    cargo tree --workspace --edges normal,build --invert opentelemetry'
fi

printf 'no-opentelemetry: clean - %d crate(s) resolved, no opentelemetry among them\n' \
    "${#crates[@]}"

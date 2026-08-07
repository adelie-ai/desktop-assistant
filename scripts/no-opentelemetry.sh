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

# The floor is not a guess at "enough crates". It is read from the workspace
# itself: every member must appear in the tree, or the invocation resolved a
# subset and the clean result is worth nothing. A count alone is too weak - one
# richly-dependent binary crate here resolves 349 of the workspace's 364, so a
# `-p desktop-assistant-daemon` slipped in by a merge would clear any plausible
# count while silently checking none of the other members.

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
# stderr goes to its own file, never into the output that gets parsed. cargo
# writes progress there, and one of those lines appears whenever another cargo
# invocation holds the package-cache lock:
#
#     Blocking waiting for file lock on package cache
#
# Merged into the tree, `Blocking` becomes the first field of a line, and the
# parser below reads first fields as package names. It is still reported on
# every failure path, because cargo's explanation of a resolution error is the
# whole value of the message. Mirrors scripts/audit.sh and scripts/test-db.sh,
# which capture the two streams separately for the same reason.
tree_out="$(mktemp)"
tree_err="$(mktemp)"
trap 'rm -f "$tree_out" "$tree_err"' EXIT

tree_status=0
cargo tree --workspace --edges normal,build --prefix none >"$tree_out" 2>"$tree_err" || tree_status=$?

if [ "$tree_status" -ne 0 ]; then
    die_loud 'NO-OPENTELEMETRY STEP DID NOT RUN: cargo could not resolve the tree' \
        'The check proves nothing when the tree cannot be read, so this is a failure' \
        'and not a clean result. Fix the resolution error first.' \
        '' \
        "$(cat "$tree_err" "$tree_out" | sed 's/^/    /' | head -40)"
fi

# `$2 ~ /^v[0-9]/`: a package line from `cargo tree --prefix none` is
# `<name> v<version>`, optionally followed by a source or a `(*)` de-duplication
# marker. Requiring the version field is a second guard beside the separated
# streams above - anything cargo puts on stdout that is not a package line (a
# `[build-dependencies]` section header, a future addition to the format) is
# ignored rather than counted as a package. If the format ever changes wholesale
# the lists come back empty, and the emptiness checks below fail loudly.
package_names() { # package_names <file>
    awk 'NF && $2 ~ /^v[0-9]/ { print $1 }' "$1" | sort -u
}

mapfile -t crates < <(package_names "$tree_out")

# The workspace members, read from cargo rather than listed here, so a new
# member crate is covered the day it is added.
members_out="$(mktemp)"
members_err="$(mktemp)"
trap 'rm -f "$tree_out" "$tree_err" "$members_out" "$members_err"' EXIT
members_status=0
cargo tree --workspace --depth 0 --prefix none >"$members_out" 2>"$members_err" || members_status=$?
[ "$members_status" -eq 0 ] || die_loud \
    'NO-OPENTELEMETRY STEP DID NOT RUN: cannot list the workspace members' \
    'Without the member list this script cannot tell a full scan from a partial' \
    'one, so it will not report a clean result.' \
    '' \
    "$(cat "$members_err" "$members_out" | sed 's/^/    /' | head -20)"

mapfile -t members < <(package_names "$members_out")
[ "${#members[@]}" -gt 0 ] || die_loud \
    'NO-OPENTELEMETRY STEP DID NOT RUN: the workspace has no members' \
    'cargo reported an empty member list, so nothing was checked.' \
    '' \
    'Either the workspace really has none, or `cargo tree --prefix none` no' \
    'longer answers with `<name> v<version>` lines and the parser needs' \
    'updating. What it printed:' \
    '' \
    "$(cat "$members_err" "$members_out" | sed 's/^/    /' | head -20)"

unchecked=()
for member in "${members[@]}"; do
    found=''
    for name in "${crates[@]}"; do
        [ "$name" = "$member" ] || continue
        found=1
        break
    done
    [ -n "$found" ] || unchecked+=("$member")
done

if [ "${#unchecked[@]}" -gt 0 ]; then
    die_loud 'NO-OPENTELEMETRY STEP CHECKED ONLY PART OF THE WORKSPACE - hard gate failure' \
        "${#unchecked[@]} of the ${#members[@]} workspace member(s) are absent from the tree that" \
        'was scanned, so nothing was proved about them and a green result here' \
        'would read as coverage it does not have.' \
        '' \
        "$(printf '    %s\n' "${unchecked[@]}")"
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

printf 'no-opentelemetry: clean - %d crate(s) resolved across all %d workspace member(s), no opentelemetry among them\n' \
    "${#crates[@]}" "${#members[@]}"

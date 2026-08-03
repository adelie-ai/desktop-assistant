#!/usr/bin/env bash
# The documentation step of the gate (#1046): `cargo doc` over the workspace.
#
# `cargo fmt`, `cargo clippy -D warnings`, `cargo build` and `cargo test` do
# not evaluate rustdoc lints. `cargo doc` is the only step that does, so
# without it `[`some_item`]` can point at an item that was renamed, made
# private, or deleted and every other step stays green. When the gate first
# ran this script the workspace had 100 such errors across 17 crates.
#
# Like scripts/audit.sh and scripts/secret-scan.sh, nothing here exits 0
# unless documentation was actually produced. A `cargo doc` that selects no
# package - a typo in `-p`, a renamed crate, a stale `--exclude` - exits 0 and
# prints almost nothing, which reads in the log exactly like "the docs are
# fine". So this script names the crates the invocation must produce and
# checks each one's `index.html` on disk afterwards.
#
# Usage: scripts/doc.sh [CARGO_DOC_ARG]...
#   No arguments  -> the whole workspace.
#   `-p <package>` -> that package only (used for the feature-gated repeats).
#   `--no-deps` is always added: dependency documentation is not ours to lint,
#   and building it would dominate the runtime of the step.
#   The workspace acted on is the one containing the current directory, so the
#   tests can run this against a throwaway fixture workspace.
set -euo pipefail

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

# Deliberately no `cd`: every cargo invocation below acts on the workspace of
# the current directory. `just` runs recipes from the repository root, so the
# gate documents this workspace, and the tests can point the same script at a
# throwaway fixture workspace by running it from there.
cargo_args=("$@")
[ "${#cargo_args[@]}" -gt 0 ] || cargo_args=(--workspace)

# The crate directory rustdoc writes under target/doc/ is the package name with
# `-` replaced by `_`.
doc_dir_name() { # doc_dir_name <package-name>
    printf '%s\n' "${1//-/_}"
}

# The packages this invocation must document. An explicit `-p` selects one;
# anything else is the whole workspace, read from cargo itself rather than from
# a list in this script, so a new member crate is covered the day it is added.
selected_packages() {
    local i
    for ((i = 0; i < ${#cargo_args[@]}; i++)); do
        case "${cargo_args[i]}" in
            -p | --package)
                printf '%s\n' "${cargo_args[i + 1]:-}"
                return 0
                ;;
            -p=* | --package=*)
                printf '%s\n' "${cargo_args[i]#*=}"
                return 0
                ;;
        esac
    done
    cargo tree --workspace --depth 0 --prefix none 2>/dev/null \
        | awk 'NF { print $1 }'
}

# Where cargo puts build output. Read from cargo rather than assumed, so
# CARGO_TARGET_DIR and a non-default layout both work.
target_dir="$(cargo metadata --format-version 1 --no-deps 2>/dev/null \
    | grep -o '"target_directory":"[^"]*"' | head -1 | cut -d'"' -f4 || true)"
[ -n "$target_dir" ] || die_loud 'DOCUMENTATION STEP DID NOT RUN: cannot locate the target directory' \
    '`cargo metadata` did not report a target_directory, so this script cannot' \
    'check that documentation was produced. Fix the workspace manifest first.'

mapfile -t packages < <(selected_packages)
[ "${#packages[@]}" -gt 0 ] && [ -n "${packages[0]}" ] || die_loud \
    'DOCUMENTATION STEP DID NOT RUN: no package selected' \
    "cargo doc --no-deps ${cargo_args[*]}" \
    '' \
    'selects nothing, so a green run would prove nothing. Check the `-p` name' \
    'against the workspace members.'

doc_out="$(mktemp)"
trap 'rm -f "$doc_out"' EXIT

doc_status=0
cargo doc --no-deps "${cargo_args[@]}" >"$doc_out" 2>&1 || doc_status=$?

if [ "$doc_status" -ne 0 ]; then
    die_loud 'RUSTDOC ERRORS - hard gate failure' \
        'A doc comment points at something rustdoc cannot resolve, or is not the' \
        'markup it looks like. Fix the doc comment - and read the prose beside it,' \
        'because a link to an item that no longer exists usually means the sentence' \
        'around it is stale too. If the target is deliberately private, drop the' \
        'brackets and leave a plain `code span`; do NOT make an item public, and do' \
        'NOT add an #[allow(rustdoc::...)], to satisfy a link.' \
        '' \
        "    cargo doc --no-deps ${cargo_args[*]}" \
        '' \
        "$(sed 's/^/    /' "$doc_out")"
fi

# A warning here means a crate is NOT inheriting `[lints] workspace = true`
# (root Cargo.toml sets `rust.warnings = "deny"`, which is what turns every
# rustdoc lint above into an error). Left alone, that crate's documentation
# rots exactly the way this step exists to prevent, while the step stays green.
if grep -q '^warning:' "$doc_out"; then
    die_loud 'RUSTDOC WARNINGS THAT DID NOT FAIL THE BUILD - hard gate failure' \
        'rustdoc reported a warning instead of an error, so this crate is not' \
        'inheriting the workspace lint levels. Add `[lints] workspace = true` to' \
        'its Cargo.toml, then fix what it reports.' \
        '' \
        "$(grep -A5 '^warning:' "$doc_out" | sed 's/^/    /' | head -40)"
fi

# Proof that documentation exists, not merely that cargo exited 0. Checked
# per selected package: a run that quietly documented a subset (or nothing at
# all) fails here instead of reading as full coverage.
missing=()
for pkg in "${packages[@]}"; do
    index="$target_dir/doc/$(doc_dir_name "$pkg")/index.html"
    [ -s "$index" ] || missing+=("$pkg ($index)")
done

if [ "${#missing[@]}" -gt 0 ]; then
    die_loud 'DOCUMENTATION STEP PRODUCED NO DOCUMENTATION - hard gate failure' \
        "cargo doc exited 0, but ${#missing[@]} of the ${#packages[@]} selected package(s) have" \
        'no index.html on disk. Nothing was checked for them, so this run proves' \
        'nothing about their doc comments.' \
        '' \
        "$(printf '    %s\n' "${missing[@]}")"
fi

printf 'doc: clean - %d crate(s) documented (%s)\n' \
    "${#packages[@]}" "cargo doc --no-deps ${cargo_args[*]}"

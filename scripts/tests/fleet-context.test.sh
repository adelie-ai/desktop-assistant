#!/usr/bin/env bash
# Acceptance criteria for the documented fleet build context (#811).
#
# The fleet image is built from a staged context holding all fourteen repos, so
# its root is the CHECKOUT root, not any one repo. The repo-root .dockerignore
# therefore never applies, and the rsync exclude list in the docs is the only
# control that runs. Two docs carry that list, they had drifted apart, and
# neither excluded .env, .envrc or .claude - so staging this repo copied a live
# .env and eight stale .claude/worktrees checkouts into the image context.
#
# These tests do not compare the docs to a second copy of the list, which would
# only prove the two agree. They EXTRACT the exclude flags from each document
# and run rsync with them against a fixture tree that contains exactly the
# things that must not ship, so a doc whose list stops working fails here.
set -euo pipefail
# shellcheck source=scripts/tests/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

BUILD_DOCS=(
    'docs/k8s-deployment.md'
    'deploy/mcp/README.md'
)

# Anything here must never reach the image context.
MUST_NOT_SHIP=(
    '.env'
    '.envrc'
    '.claude/worktrees/stale/leftover.rs'
    'target/debug/junk'
    '.git/config'
)
# ... and the actual source must still arrive, or the "fix" is just a broken build.
MUST_SHIP=(
    'Cargo.toml'
    'src/main.rs'
)

# Pull the `--exclude <x>` flags out of the rsync invocation in a doc.
_excludes_from() {
    awk '/rsync -a/,/\$CTX/' "$SCRIPT_TESTS_ROOT/$1" \
        | grep -oE -- "--exclude '?[.A-Za-z-]+'?" \
        | sed -E "s/--exclude '?//; s/'?$//"
}

_make_fixture() {
    local root="$1" f
    for f in "${MUST_NOT_SHIP[@]}" "${MUST_SHIP[@]}"; do
        mkdir -p "$root/$(dirname "$f")"
        printf 'fixture\n' > "$root/$f"
    done
}

_stage_with_doc_excludes() {
    local doc="$1" src="$2" dst="$3" args=() ex
    while read -r ex; do [ -n "$ex" ] && args+=(--exclude "$ex"); done < <(_excludes_from "$doc")
    [ "${#args[@]}" -gt 0 ] || fail "$doc: found no --exclude flags; the staging recipe moved or changed shape"
    rsync -aL "${args[@]}" "$src/" "$dst/"
}

_assert_doc_stages_safely() {
    local doc="$1" src="$TEST_TMP/src" dst="$TEST_TMP/ctx" f
    rm -rf "$src" "$dst"; mkdir -p "$src" "$dst"
    _make_fixture "$src"
    _stage_with_doc_excludes "$doc" "$src" "$dst"
    for f in "${MUST_NOT_SHIP[@]}"; do
        [ ! -e "$dst/$f" ] || fail "$doc stages '$f' into the image context"
    done
    for f in "${MUST_SHIP[@]}"; do
        [ -e "$dst/$f" ] || fail "$doc excludes '$f', which the image needs"
    done
}

the_k8s_deployment_guide_stages_no_secrets_or_build_junk() {
    _assert_doc_stages_safely 'docs/k8s-deployment.md'
}

the_mcp_fleet_guide_stages_no_secrets_or_build_junk() {
    _assert_doc_stages_safely 'deploy/mcp/README.md'
}

both_build_guides_document_the_same_exclude_list() {
    # They drifted apart once; a context that is safe in one guide and leaky in
    # the other is how that gets shipped.
    local a b
    a="$(_excludes_from 'docs/k8s-deployment.md' | sort)"
    b="$(_excludes_from 'deploy/mcp/README.md' | sort)"
    assert_eq "$a" "$b" 'the two documented exclude lists must match'
}

run_test the_k8s_deployment_guide_stages_no_secrets_or_build_junk
run_test the_mcp_fleet_guide_stages_no_secrets_or_build_junk
run_test both_build_guides_document_the_same_exclude_list
finish_tests 'fleet-context'

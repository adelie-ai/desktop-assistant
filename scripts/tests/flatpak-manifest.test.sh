#!/usr/bin/env bash
# Acceptance criteria for the flatpak build context (#811).
#
# packaging/flatpak/org.desktopassistant.App.yml declares a "dir" source at
# `path: ../..` - the whole repo checkout, copied byte-for-byte into the
# flatpak build sandbox. A "dir" source copies every file on disk, tracked or
# not, .gitignore or not, so an untracked .env sitting in the working tree
# still reaches the sandbox unless it is named in the source's own `skip:`
# list. That gap is exactly how three live-key copies ended up under
# .flatpak-builder/build/*/.env.
#
# flatpak-builder resolves each `skip:` entry as a path relative to the
# source directory (`g_file_resolve_relative_path`, builder-source-dir.c), so
# these are plain repo-root-relative filenames, not globs.
set -euo pipefail
# shellcheck source=scripts/tests/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

MANIFEST="$SCRIPT_TESTS_ROOT/packaging/flatpak/org.desktopassistant.App.yml"
MUST_SKIP=('.env' '.envrc' 'secrets.toml')

# The `skip:` list nested under the `type: dir` source stanza. Scoped to that
# block (not "any skip: anywhere in the file") so this does not pass by
# coincidence if some other key ever needs a skip list too.
_dir_source_skip_list() {
    awk '
        /^ *- type: dir *$/ { in_dir=1 }
        in_dir && /^ *skip: *$/ { in_skip=1; next }
        in_dir && /^ *-/ && !in_skip { next }
        in_skip && /^ *- / { sub(/^ *- */, ""); print; next }
        in_skip && !/^ *- / { in_skip=0 }
        in_dir && /^ *- type: / && !/type: dir/ { in_dir=0 }
    ' "$MANIFEST"
}

the_flatpak_dir_source_declares_a_skip_list() {
    local list
    list="$(_dir_source_skip_list)"
    [ -n "$list" ] || fail "no skip: list under the 'type: dir' source in $MANIFEST"
}

the_flatpak_dir_source_skip_list_excludes_dotenv_dotenvrc_and_secrets_toml() {
    local list f
    list="$(_dir_source_skip_list)"
    for f in "${MUST_SKIP[@]}"; do
        printf '%s\n' "$list" | grep -qxF "$f" \
            || fail "the flatpak dir source's skip: list does not exclude '$f'"
    done
}

run_test the_flatpak_dir_source_declares_a_skip_list
run_test the_flatpak_dir_source_skip_list_excludes_dotenv_dotenvrc_and_secrets_toml
finish_tests 'flatpak-manifest'

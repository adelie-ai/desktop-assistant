#!/usr/bin/env bash
# Acceptance criteria for the working-tree secret-scan step of the gate (#811).
#
# The key that produced this finding was never committed, so a scanner that
# only walks git history would have reported clean the entire time it was
# exposed. scripts/secret-scan.sh must run gitleaks' filesystem walk (`dir`),
# not its history walk (`git`) - the dedicated test below fails red against a
# script that scans history instead of the checkout.
#
# Wrapper-logic tests (missing tool, version drift, a scan that produced no
# report) use a fake `gitleaks` on PATH, the same mechanism
# audit-gate.test.sh uses for cargo-audit, so the failure classification is
# exercised deterministically without a live scan. The two tests that matter
# most - detecting a real secret shape, and not flagging the clean tree - run
# the real, pinned gitleaks binary: a mocked tool can only prove the
# wrapper's plumbing, never that detection actually works. A scan that scans
# nothing passes every mocked test too, which is the vacuous version of this
# fix.
set -euo pipefail
# shellcheck source=scripts/tests/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SECRET_SCAN_SH="$SCRIPT_TESTS_ROOT/scripts/secret-scan.sh"
PINNED_GITLEAKS_VERSION='8.30.1'

with_fake_gitleaks() {
    mkdir -p "$TEST_TMP/bin"
    cp "$SCRIPT_TESTS_FIXTURES/fake-gitleaks.sh" "$TEST_TMP/bin/gitleaks"
    chmod +x "$TEST_TMP/bin/gitleaks"
    export PATH="$TEST_TMP/bin:$PATH"
    export FAKE_GITLEAKS_LOG="$TEST_TMP/gitleaks.log"
    : >"$FAKE_GITLEAKS_LOG"
}

# A PATH with no gitleaks reachable anywhere, regardless of how THIS machine
# happens to have installed it (~/.cargo/bin, ~/.local/bin, ~/go/bin, a
# distro package in /usr/bin or /usr/local/bin). A naive
# "$TEST_TMP/empty:/usr/bin:/bin" still resolves gitleaks on any machine that
# followed this repo's own `pacman -S gitleaks` instructions, so it never
# actually exercises the missing-binary path it is named for - it happens to
# pass today only because this particular box does not have it there.
# Symlinks in only the coreutils scripts/secret-scan.sh itself needs, so
# nothing outside that directory is ever consulted.
_hermetic_path_without_gitleaks() {
    local bin="$TEST_TMP/hermetic-bin" tool
    mkdir -p "$bin"
    for tool in bash env dirname mktemp grep sed paste awk cat tr rm; do
        ln -sf "$(command -v "$tool")" "$bin/$tool"
    done
    printf '%s' "$bin"
}

CLEAN_REPORT='[]'
LEAK_REPORT='[{"RuleID":"openai-api-key","File":".env","StartLine":4,"Fingerprint":".env:openai-api-key:4","Secret":"REDACTED","Match":"REDACTED"}]'
# gitleaks version, verbatim, from the CachyOS/Arch `pacman -S gitleaks`
# package (confirmed 8.30.1-1.1 - the pinned version - on both plain Arch and
# CachyOS): the distro build does not set the version ldflag, so even the
# exact pinned release reports this instead of a real version number.
UNVERSIONED_OUTPUT='version is set by build process'

# --- wrapper-logic tests (mocked gitleaks) -----------------------------------

secret_scan_passes_on_a_clean_report() {
    with_fake_gitleaks
    export FAKE_GITLEAKS_REPORT="$CLEAN_REPORT" FAKE_GITLEAKS_STATUS=0
    run_cmd "$SECRET_SCAN_SH"
    assert_eq 0 "$RUN_STATUS" "a clean scan must pass the step: $RUN_ERR"
}

secret_scan_fails_when_gitleaks_reports_a_leak() {
    with_fake_gitleaks
    export FAKE_GITLEAKS_REPORT="$LEAK_REPORT" FAKE_GITLEAKS_STATUS=1
    run_cmd "$SECRET_SCAN_SH"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a reported leak must fail the step'
    assert_contains "$RUN_ERR" '.env' 'failure names the offending file'
    assert_contains "$RUN_ERR" 'openai-api-key' 'failure names the rule that matched'
}

secret_scan_fails_loudly_when_gitleaks_is_not_installed() {
    run_cmd env PATH="$(_hermetic_path_without_gitleaks)" "$SECRET_SCAN_SH"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a missing gitleaks must fail the step'
    assert_contains "$RUN_ERR" 'SECRET SCAN DID NOT RUN: gitleaks is not installed' \
        'names the missing-binary failure specifically, not any message that merely mentions gitleaks'
    assert_not_contains "$RUN_OUT" 'clean' 'must not claim a clean scan'
}

secret_scan_fails_when_the_installed_gitleaks_version_does_not_match_the_pin() {
    with_fake_gitleaks
    export FAKE_GITLEAKS_VERSION='7.0.0'
    run_cmd "$SECRET_SCAN_SH"
    [ "$RUN_STATUS" -ne 0 ] || fail 'an unpinned gitleaks version must fail the step'
    assert_contains "$RUN_ERR" 'version does not match the pin' 'names the mismatch failure specifically'
    assert_contains "$RUN_ERR" '7.0.0' 'names the version found'
    assert_contains "$RUN_ERR" "$PINNED_GITLEAKS_VERSION" 'names the pinned version'
    assert_not_contains "$RUN_ERR" 'no parseable version' \
        'a real (if wrong) version number is not the same failure as an unparseable one'
}

secret_scan_fails_when_gitleaks_reports_no_parseable_version() {
    # The CachyOS/Arch package's actual failure mode (confirmed against the
    # real package on both distros): `gitleaks version` prints a sentence,
    # not a version number, for the exact pinned release. This is a
    # packaging defect, not version drift, and needs its own diagnosis - a
    # gate that calls this "version does not match the pin" points the
    # reader at the wrong fix (upgrade/downgrade gitleaks) instead of the
    # right one (install from the release tarball, or opt in).
    with_fake_gitleaks
    export FAKE_GITLEAKS_VERSION="$UNVERSIONED_OUTPUT"
    run_cmd "$SECRET_SCAN_SH"
    [ "$RUN_STATUS" -ne 0 ] || fail 'an unparseable version must fail the step'
    assert_contains "$RUN_ERR" 'no parseable version' 'names the real outcome, distinct from a version mismatch'
    assert_contains "$RUN_ERR" 'packaging defect' 'explains this is a packaging bug, not drift'
    assert_not_contains "$RUN_ERR" 'does not match the pin' \
        'must not be phrased as a mismatch - there is no version to compare'
}

secret_scan_allows_an_unpinned_gitleaks_version_with_explicit_opt_in() {
    # Mirrors ADELE_AUDIT_ALLOW_STALE's shape in scripts/audit.sh: the exact
    # pin stays the default, but nobody should have to edit the gate to
    # unblock unrelated work.
    with_fake_gitleaks
    export FAKE_GITLEAKS_VERSION='7.0.0'
    export FAKE_GITLEAKS_REPORT="$CLEAN_REPORT" FAKE_GITLEAKS_STATUS=0
    export ADELE_SECRET_SCAN_ALLOW_UNPINNED=1
    run_cmd "$SECRET_SCAN_SH"
    assert_eq 0 "$RUN_STATUS" "explicit opt-in must still complete: $RUN_ERR"
    assert_contains "$RUN_ERR" 'ALLOW_UNPINNED' 'opt-in is loud about what it did'
    assert_contains "$RUN_ERR" '7.0.0' 'names the version actually used'
}

secret_scan_allows_an_unparseable_gitleaks_version_with_explicit_opt_in() {
    with_fake_gitleaks
    export FAKE_GITLEAKS_VERSION="$UNVERSIONED_OUTPUT"
    export FAKE_GITLEAKS_REPORT="$CLEAN_REPORT" FAKE_GITLEAKS_STATUS=0
    export ADELE_SECRET_SCAN_ALLOW_UNPINNED=1
    run_cmd "$SECRET_SCAN_SH"
    assert_eq 0 "$RUN_STATUS" "explicit opt-in must still complete even without a parseable version: $RUN_ERR"
    assert_contains "$RUN_ERR" 'ALLOW_UNPINNED' 'opt-in is loud about what it did'
}

secret_scan_fails_when_gitleaks_produces_no_report() {
    with_fake_gitleaks
    # Mirrors #706's failure mode for cargo-audit: an exit status alone is not
    # proof a scan happened. A fatal gitleaks error (bad config, bad path)
    # exits non-zero and writes no report at all - verified empirically
    # against the real binary, not assumed.
    export FAKE_GITLEAKS_NO_REPORT=1 FAKE_GITLEAKS_STATUS=1
    export FAKE_GITLEAKS_STDERR='FTL unable to load gitleaks config'
    run_cmd "$SECRET_SCAN_SH"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a scan that produced no report must fail the step'
    assert_contains "$RUN_ERR" 'DID NOT RUN' 'names the real outcome'
    assert_not_contains "$RUN_OUT" 'clean' 'must not claim a clean scan'
}

secret_scan_uses_the_filesystem_walk_not_the_git_history_walk() {
    with_fake_gitleaks
    export FAKE_GITLEAKS_REPORT="$CLEAN_REPORT" FAKE_GITLEAKS_STATUS=0
    run_cmd "$SECRET_SCAN_SH"
    assert_eq 0 "$RUN_STATUS" "sanity: the clean-report case must still pass: $RUN_ERR"
    local invocation subcommand
    invocation="$(grep -v '^version$' "$FAKE_GITLEAKS_LOG" | head -1)"
    [ -n "$invocation" ] || fail 'gitleaks was never invoked to run a scan'
    subcommand="${invocation%% *}"
    assert_eq 'dir' "$subcommand" 'must scan the filesystem (gitleaks dir), not history (gitleaks git)'
}

check_gate_runs_the_secret_scan() {
    # Deleting the step from `just check` must fail this test.
    local plan
    plan="$(cd "$SCRIPT_TESTS_ROOT" && just -n check 2>&1)"
    assert_contains "$plan" 'scripts/secret-scan.sh' 'the gate runs the secret scan'
}

# --- real-tool tests (the real, pinned gitleaks binary; no mock) ------------
#
# gitleaks is a required gate dependency (scripts/secret-scan.sh fails the
# gate outright if it is missing, same as cargo-audit) - see AGENTS.md,
# "Secret scanning". These tests assume it is on PATH, same as the sqlite-gate
# suite assumes a real cargo/rustc.

secret_scan_detects_a_working_tree_key() {
    local fixture="$TEST_TMP/src"
    mkdir -p "$fixture"
    # Assembled at runtime, from hashes of plain fixture labels, so no line in
    # this committed file is a contiguous, scanner-shaped secret (which would
    # trip this very gate on this file). Shape matches gitleaks' openai-api-key
    # rule (sk-proj-<58 hex>T3BlbkFJ<58 hex>); content is a SHA-256 digest of a
    # label string, never real key material, and was never a working credential.
    local body_a body_b
    body_a="$(printf 'adele-secret-scan-test-fixture-alpha' | sha256sum | cut -c1-58)"
    body_b="$(printf 'adele-secret-scan-test-fixture-beta' | sha256sum | cut -c1-58)"
    printf 'OPENAI_API_KEY=sk-proj-%sT3BlbkFJ%s\n' "$body_a" "$body_b" >"$fixture/.env"

    run_cmd "$SECRET_SCAN_SH" "$fixture"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a synthetic but correctly-shaped key must fail the scan'
    assert_contains "$RUN_ERR" '.env' 'names the file holding the key'
    assert_contains "$RUN_ERR" 'openai-api-key' 'names the rule that matched'
}

secret_scan_does_not_flag_the_clean_tree() {
    # The repo as it stands, including its own real test fixtures (test-only
    # PEM keys under testdata/, a truncated example JWT in the docs) must
    # scan clean under the real config - or the gate is noise from the day it
    # ships.
    run_cmd "$SECRET_SCAN_SH" "$SCRIPT_TESTS_ROOT"
    assert_eq 0 "$RUN_STATUS" "the repo must scan clean: $RUN_ERR$RUN_OUT"
}

secret_scan_detects_a_key_under_claude_worktrees() {
    # .claude/worktrees/ must NOT be allowlisted (AGENTS.md, "Secret
    # scanning"): the #811 incident involved a live .env AND eight stale
    # .claude/worktrees checkouts, so excluding this directory would reopen
    # half the blind spot the gate exists to close. Same reasoning as keeping
    # .flatpak-builder/ in scope, applied to the other directory the
    # incident actually touched.
    local fixture="$TEST_TMP/src"
    mkdir -p "$fixture/.claude/worktrees/some-session"
    local body_a body_b
    body_a="$(printf 'adele-secret-scan-test-fixture-worktrees-alpha' | sha256sum | cut -c1-58)"
    body_b="$(printf 'adele-secret-scan-test-fixture-worktrees-beta' | sha256sum | cut -c1-58)"
    printf 'OPENAI_API_KEY=sk-proj-%sT3BlbkFJ%s\n' "$body_a" "$body_b" \
        >"$fixture/.claude/worktrees/some-session/.env"

    run_cmd "$SECRET_SCAN_SH" "$fixture"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a key under .claude/worktrees/ must still fail the scan'
    assert_contains "$RUN_ERR" '.claude/worktrees' 'names the nested-worktree path holding the key'
}

secret_scan_does_not_flag_a_duplicated_known_fixture_under_claude_worktrees() {
    # The other half of keeping .claude/worktrees/ in scope (review round 3):
    # this repo's own known-fake fixtures (test-only PEM keys) are TRACKED
    # files, so they exist byte-for-byte duplicated inside every nested
    # .claude/worktrees/<session>/ checkout - 27 such duplicates were found
    # scanning the primary checkout (3 fixtures x 9 stale worktrees). A
    # path-EXACT exemption (.gitleaksignore's file:rule:line fingerprint)
    # only ever matches ONE of those paths, so every duplicate reads as a
    # new finding on an otherwise clean tree - exactly the noise that trains
    # people to bypass the gate. The exemption must be shaped so the same
    # tracked fixture is exempt at every path it is checked out to.
    local fixture="$TEST_TMP/src" real_fixture canonical_rel dup_rel
    real_fixture="$SCRIPT_TESTS_ROOT/crates/daemon/src/config/testdata/oidc_test_key1.pem"
    [ -f "$real_fixture" ] || fail "fixture moved: $real_fixture no longer exists"
    canonical_rel='crates/daemon/src/config/testdata/oidc_test_key1.pem'
    dup_rel=".claude/worktrees/some-session/$canonical_rel"

    mkdir -p "$fixture/$(dirname "$canonical_rel")" "$fixture/$(dirname "$dup_rel")"
    cp "$real_fixture" "$fixture/$canonical_rel"
    cp "$real_fixture" "$fixture/$dup_rel"

    run_cmd "$SECRET_SCAN_SH" "$fixture"
    assert_eq 0 "$RUN_STATUS" \
        "a duplicated known-fixture under .claude/worktrees/ must not fail the scan: $RUN_ERR"
}

# --- private information (the real, pinned gitleaks binary; no mock) --------
#
# A second class the gate must catch, next to credentials: private information.
# None of it is a credential, so every one of these strings passed the gate for
# as long as it was in the tree. The scan is split in two layers, and each
# layer has its own tests below:
#
#   Layer 1 - the rules committed in .gitleaks.toml. Generic SHAPES only: an
#             absolute home path, a hostname on a private-network pseudo-TLD.
#             This file is public, so no site-specific value may appear in it.
#   Layer 2 - a host-local rules file outside the repository, holding the
#             site-specific literals (instance names, private domains). It is
#             OPTIONAL: a fresh clone with no such file still scans, and still
#             passes, with layer 1 alone.

# Layer 2 is optional, so a test that does not say which of the two states it
# runs in silently inherits whatever the machine executing it happens to have
# installed at the real location. Every test below states it. This path is
# inside the private per-test temp dir and is never created, so the scan runs
# layer 1 only.
without_private_rules() {
    export ADELE_SECRET_SCAN_PRIVATE_RULES="$TEST_TMP/absent-private-rules.toml"
}

# A layer-2 rules file built inside the throwaway test directory, around an
# invented instance name that names nothing this project or anyone else
# operates. The real host-local file is never read and never depended on: a
# suite that pointed at it would pass or fail according to a file that is not
# in the repository and differs per machine.
INVENTED_SITE_LITERAL='zephyr-prod'
with_invented_private_rules() {
    local file="$TEST_TMP/private-rules.toml"
    cat >"$file" <<'TOML'
[[rules]]
id = "site-invented-instance-name"
description = "Invented instance name; fixture for the gate's own layer-2 tests."
regex = '''\bzephyr-prod\b'''
TOML
    export ADELE_SECRET_SCAN_PRIVATE_RULES="$file"
}

secret_scan_detects_an_absolute_home_path() {
    # One of the audited leaks arrived as pasted terminal output, which carries
    # the operator's home directory with it. Both spellings count: Linux and
    # macOS.
    without_private_rules
    local fixture="$TEST_TMP/src"
    mkdir -p "$fixture/docs"
    printf 'Ran it from /home/mallory/Projects/adelie-ai and it worked.\n' >"$fixture/docs/linux.md"
    printf 'On the laptop the tree is at /Users/mallory/Code/adelie-ai.\n' >"$fixture/docs/macos.md"

    run_cmd "$SECRET_SCAN_SH" "$fixture"
    [ "$RUN_STATUS" -ne 0 ] || fail 'an absolute home path naming an account must fail the scan'
    assert_contains "$RUN_ERR" 'adele-absolute-home-path' 'names the rule that matched'
    assert_contains "$RUN_ERR" 'docs/linux.md' 'names the file holding the Linux home path'
    assert_contains "$RUN_ERR" 'docs/macos.md' 'names the file holding the macOS home path'
}

secret_scan_does_not_flag_documented_home_path_placeholders() {
    # This matters more than the detection above. A rule that fires on the
    # placeholder people are told to write instead gets disabled within a week,
    # and then the real rule is gone too. The first five are the substitutes
    # AGENTS.base.md rule 5.2 names; the rest are the synthetic account names
    # this repository's own fixtures, documentation and container images
    # already use.
    without_private_rules
    local fixture="$TEST_TMP/src"
    mkdir -p "$fixture"
    cat >"$fixture/placeholders.md" <<'EOF'
/home/<user>/.config
/home/user/work
/home/$USER/bin
${HOME}/.local/share
~/.config/adelie-ai
/home/example/.ssh/id_ed25519
/home/assistant/.config
/home/ada/.local/share
/home/ada-client/.cache
/home/peer/notes
/home/x/.agents/skills
/home/{owner}/.local/share/adele
EOF

    run_cmd "$SECRET_SCAN_SH" "$fixture"
    assert_eq 0 "$RUN_STATUS" "documented home-path placeholders must not fail the scan: $RUN_ERR"
}

secret_scan_detects_a_private_network_hostname() {
    # A hostname on a pseudo-TLD that public DNS never delegates is a private
    # machine by construction, whatever it is called - so the rule matches the
    # SHAPE and needs no site-specific name written into the public config.
    without_private_rules
    local fixture="$TEST_TMP/src" tld
    mkdir -p "$fixture"
    for tld in lab lan corp home; do
        printf 'Deployed to daemon.internal-site.%s and it answered.\n' "$tld" \
            >"$fixture/host-$tld.md"
    done

    run_cmd "$SECRET_SCAN_SH" "$fixture"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a private-network hostname must fail the scan'
    assert_contains "$RUN_ERR" 'adele-private-network-hostname' 'names the rule that matched'
    for tld in lab lan corp home; do
        assert_contains "$RUN_ERR" "host-$tld.md" "detects a hostname under .$tld"
    done
}

secret_scan_does_not_flag_public_or_documentation_hostnames() {
    # example.com is the reserved documentation domain and is explicitly fine
    # (AGENTS.base.md rule 5.2). The rest are names this repository already
    # carries on purpose: Kubernetes' own service domain, the cloud metadata
    # host its SSRF policy exists to block, and ordinary member access in Rust
    # that happens to read like a hostname.
    without_private_rules
    local fixture="$TEST_TMP/src"
    mkdir -p "$fixture"
    cat >"$fixture/public.md" <<'EOF'
https://example.com/docs
registry.example.com/adelie/daemon
adele-daemon.default.svc.cluster.local
http://metadata.google.internal/computeMetadata/v1/
let addr = self.local;
let url = cfg.db.internal;
published in the collab.labs directory
EOF

    run_cmd "$SECRET_SCAN_SH" "$fixture"
    assert_eq 0 "$RUN_STATUS" "public and documentation hostnames must not fail the scan: $RUN_ERR"
}

secret_scan_does_not_flag_the_gate_files_that_carry_the_patterns() {
    # The rule file and this suite both have to contain the very strings the
    # rules match, and gitleaks scans the working tree, so without an exemption
    # a clean checkout fails its own gate. That is the noise that trains people
    # to bypass it (the same reasoning as the allowlists already in
    # .gitleaks.toml). Copied to their repo-relative paths inside the fixture,
    # because the exemption is matched by path SUFFIX and has to hold wherever
    # the file is checked out.
    without_private_rules
    local fixture="$TEST_TMP/src"
    mkdir -p "$fixture/scripts/tests"
    cp "$SCRIPT_TESTS_ROOT/.gitleaks.toml" "$fixture/.gitleaks.toml"
    cp "$SCRIPT_TESTS_ROOT/scripts/tests/secret-scan-gate.test.sh" \
        "$fixture/scripts/tests/secret-scan-gate.test.sh"

    run_cmd "$SECRET_SCAN_SH" "$fixture"
    assert_eq 0 "$RUN_STATUS" \
        "the rule file and this suite must not match their own patterns: $RUN_ERR"
}

# --- layer 2: the optional host-local site rules -----------------------------

secret_scan_runs_and_says_so_when_the_host_local_rules_are_absent() {
    # A fresh clone, a new machine and CI all have no host-local file. The scan
    # must still run and still pass on layer 1 alone - and say which layers it
    # ran, because "clean" means two different things in the two states.
    without_private_rules
    local fixture="$TEST_TMP/src"
    mkdir -p "$fixture"
    printf 'Nothing private in here.\n' >"$fixture/README.md"

    run_cmd "$SECRET_SCAN_SH" "$fixture"
    assert_eq 0 "$RUN_STATUS" "a missing host-local rules file must not fail the scan: $RUN_ERR"
    assert_contains "$RUN_OUT" 'clean' 'reports the clean scan'
    assert_contains "$RUN_OUT" 'no host-local rules' 'says the site-specific layer did not run'
}

secret_scan_detects_a_site_literal_when_the_host_local_rules_are_present() {
    with_invented_private_rules
    local fixture="$TEST_TMP/src"
    mkdir -p "$fixture"
    printf 'kubectl -n %s rollout status deploy/adele-daemon\n' "$INVENTED_SITE_LITERAL" \
        >"$fixture/runbook.md"

    run_cmd "$SECRET_SCAN_SH" "$fixture"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a site literal must fail the scan when the host-local rules are present'
    assert_contains "$RUN_ERR" 'site-invented-instance-name' 'names the host-local rule that matched'
    assert_contains "$RUN_ERR" 'runbook.md' 'names the file holding the site literal'
}

secret_scan_does_not_detect_a_site_literal_when_the_host_local_rules_are_absent() {
    # The paired negative. Without it the test above proves only that the scan
    # fails, not that the host-local layer is what made it fail.
    without_private_rules
    local fixture="$TEST_TMP/src"
    mkdir -p "$fixture"
    printf 'kubectl -n %s rollout status deploy/adele-daemon\n' "$INVENTED_SITE_LITERAL" \
        >"$fixture/runbook.md"

    run_cmd "$SECRET_SCAN_SH" "$fixture"
    assert_eq 0 "$RUN_STATUS" \
        "a site literal is layer 2's to catch, so layer 1 alone must pass it: $RUN_ERR"
}

secret_scan_fails_loudly_when_the_host_local_rules_are_malformed() {
    # The host-local file is hand-written, on one machine, outside review. A
    # syntax error in it must not read as a clean scan: gitleaks rejects the
    # config, writes no report, and the existing report-existence check turns
    # that into a hard failure rather than a pass.
    local file="$TEST_TMP/private-rules.toml"
    printf 'this is not valid toml [[[\n' >"$file"
    export ADELE_SECRET_SCAN_PRIVATE_RULES="$file"
    local fixture="$TEST_TMP/src"
    mkdir -p "$fixture"
    printf 'Nothing private in here.\n' >"$fixture/README.md"

    run_cmd "$SECRET_SCAN_SH" "$fixture"
    [ "$RUN_STATUS" -ne 0 ] || fail 'a malformed host-local rules file must fail the step'
    assert_contains "$RUN_ERR" 'DID NOT RUN' 'names the real outcome'
    assert_not_contains "$RUN_OUT" 'clean' 'must not claim a clean scan'
}

run_test secret_scan_passes_on_a_clean_report
run_test secret_scan_fails_when_gitleaks_reports_a_leak
run_test secret_scan_fails_loudly_when_gitleaks_is_not_installed
run_test secret_scan_fails_when_the_installed_gitleaks_version_does_not_match_the_pin
run_test secret_scan_fails_when_gitleaks_reports_no_parseable_version
run_test secret_scan_allows_an_unpinned_gitleaks_version_with_explicit_opt_in
run_test secret_scan_allows_an_unparseable_gitleaks_version_with_explicit_opt_in
run_test secret_scan_fails_when_gitleaks_produces_no_report
run_test secret_scan_uses_the_filesystem_walk_not_the_git_history_walk
run_test check_gate_runs_the_secret_scan
run_test secret_scan_detects_a_working_tree_key
run_test secret_scan_does_not_flag_the_clean_tree
run_test secret_scan_detects_a_key_under_claude_worktrees
run_test secret_scan_does_not_flag_a_duplicated_known_fixture_under_claude_worktrees
run_test secret_scan_detects_an_absolute_home_path
run_test secret_scan_does_not_flag_documented_home_path_placeholders
run_test secret_scan_detects_a_private_network_hostname
run_test secret_scan_does_not_flag_public_or_documentation_hostnames
run_test secret_scan_does_not_flag_the_gate_files_that_carry_the_patterns
run_test secret_scan_runs_and_says_so_when_the_host_local_rules_are_absent
run_test secret_scan_detects_a_site_literal_when_the_host_local_rules_are_present
run_test secret_scan_does_not_detect_a_site_literal_when_the_host_local_rules_are_absent
run_test secret_scan_fails_loudly_when_the_host_local_rules_are_malformed
finish_tests 'secret-scan-gate'

#!/usr/bin/env bash
# The secret-scan step of the gate (#811): gitleaks over the CHECKED-OUT
# FILES - not git history. The key that prompted this issue was never
# committed, so a history-based scan (`gitleaks git`, or the older `gitleaks
# detect`) would have reported clean for the entire time it sat in .env at
# mode 0644. `dir` is gitleaks' filesystem-walk subcommand: it reads whatever
# is on disk, tracked or not, gitignored or not - exactly the case this gate
# exists to catch.
#
# Like scripts/audit.sh, nothing here exits 0 unless a scan actually ran and
# came back clean. An exit status alone is not proof of that (#706's failure
# mode): a fatal gitleaks error - bad config, bad path - exits non-zero and
# writes no report at all, so report-existence is the signal this script
# trusts, not the exit code by itself.
#
# Usage: scripts/secret-scan.sh [DIR]
#   DIR defaults to the repository root. Tests pass a throwaway fixture
#   directory so they can prove detection without touching the real tree.
set -euo pipefail

# Pinned so the rule set - bundled in the gitleaks binary, not fetched per
# run - is the same on every machine that runs this gate (AGENTS.base.md,
# rule 6.2). Checked against the current release before pinning (rule 6.1),
# not remembered: https://github.com/gitleaks/gitleaks/releases/tag/v8.30.1
GITLEAKS_VERSION='8.30.1'

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

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:-$repo_root}"

# The pinned release tarball is the PRIMARY install path, not `pacman -S
# gitleaks`: the CachyOS/Arch package (confirmed 8.30.1-1.1 - the pinned
# version itself - on both plain Arch and CachyOS) never sets the
# build-time version ldflag, so `gitleaks version` prints a sentence instead
# of a version number, and the exact pinned release then fails the version
# check below. The tarball install is user-local (no sudo) and reversible.
tarball_install_instructions() {
    printf '%s\n' \
        "    curl -sL -o gitleaks.tar.gz https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz" \
        '    tar -xzf gitleaks.tar.gz gitleaks && install -m 755 gitleaks ~/.local/bin/gitleaks' \
        '' \
        "    Other platforms/checksums: https://github.com/gitleaks/gitleaks/releases/tag/v${GITLEAKS_VERSION}"
}

if ! command -v gitleaks >/dev/null 2>&1; then
    die_loud 'SECRET SCAN DID NOT RUN: gitleaks is not installed' \
        'The gate promises a working-tree secret scan, so a missing scanner' \
        'fails it instead of quietly passing it.' \
        '' \
        "Install gitleaks ${GITLEAKS_VERSION} from the pinned release tarball" \
        '(user-local, no sudo, and reports a real version - see below for why' \
        'that matters here):' \
        "$(tarball_install_instructions)" \
        '' \
        "\`pacman -S gitleaks\` (Arch/CachyOS) also installs ${GITLEAKS_VERSION}, but that" \
        'package does not set the build-time version ldflag, so `gitleaks version`' \
        'prints "version is set by build process" instead of a real version - the' \
        'pin check below cannot verify that install, so prefer the tarball.'
fi

# Diagnosed as two distinct failures, not one: "gitleaks cannot say what
# version it is" (a packaging defect - confirmed on the CachyOS/Arch package
# above) and "gitleaks says a version, and it is the wrong one" (real drift)
# have different fixes, and conflating them sends the reader at the wrong
# one. A real gitleaks version is a plain X.Y.Z; anything else - including
# empty output, or "version is set by build process" - is unparseable.
installed_version="$(gitleaks version 2>/dev/null | tr -d '[:space:]' || true)"
version_is_parseable=0
case "$installed_version" in
    [0-9]*.[0-9]*.[0-9]*) version_is_parseable=1 ;;
esac

if [ "$version_is_parseable" -eq 1 ] && [ "$installed_version" = "$GITLEAKS_VERSION" ]; then
    : # pinned version, verified - proceed
elif [ -n "${ADELE_SECRET_SCAN_ALLOW_UNPINNED:-}" ]; then
    if [ "$version_is_parseable" -eq 1 ]; then
        loud 'SECRET SCAN: gitleaks version does not match the pin (opt-in)' \
            "found ${installed_version}, pinned ${GITLEAKS_VERSION}. ADELE_SECRET_SCAN_ALLOW_UNPINNED" \
            'is set, so this run scans anyway with whatever rule set that version' \
            'bundles - the scan is not skipped, only the pin check is.'
    else
        loud 'SECRET SCAN: gitleaks reports no parseable version (opt-in)' \
            "\`gitleaks version\` printed \"${installed_version:-<empty>}\", not a version" \
            'number. ADELE_SECRET_SCAN_ALLOW_UNPINNED is set, so this run scans anyway' \
            'without verifying which rule set it is running.'
    fi
elif [ "$version_is_parseable" -eq 1 ]; then
    die_loud 'SECRET SCAN DID NOT RUN: gitleaks version does not match the pin' \
        "This gate pins gitleaks ${GITLEAKS_VERSION} so the bundled rule set is" \
        'identical on every machine that runs it (AGENTS.base.md, rule 6.2).' \
        '' \
        "    found:  ${installed_version}" \
        "    pinned: ${GITLEAKS_VERSION}" \
        '' \
        'Install the pinned version, or bump GITLEAKS_VERSION in this script' \
        'deliberately after checking the new release notes (AGENTS.base.md, rule 6.1).' \
        '' \
        'To unblock unrelated work right now instead of fixing the install:' \
        '    ADELE_SECRET_SCAN_ALLOW_UNPINNED=1 just secret-scan'
else
    die_loud 'SECRET SCAN DID NOT RUN: gitleaks reports no parseable version' \
        "\`gitleaks version\` printed \"${installed_version:-<empty>}\", not a version" \
        'number. This is a known packaging defect, not version drift: the' \
        'CachyOS/Arch pacman gitleaks package (confirmed 8.30.1-1.1, the pinned' \
        'version, on both plain Arch and CachyOS) does not set the build-time' \
        'version ldflag, so even the exact pinned release reports this.' \
        '' \
        'Install the pinned release tarball instead (reports a real version):' \
        "$(tarball_install_instructions)" \
        '' \
        'To unblock unrelated work right now instead of fixing the install:' \
        '    ADELE_SECRET_SCAN_ALLOW_UNPINNED=1 just secret-scan'
fi

# Two configs, because the gate makes two different promises and gitleaks can
# only keep one of them per pass. $config finds CREDENTIALS and extends the
# bundled rule set. $private_info_config finds PRIVATE INFORMATION and extends
# NOTHING: the bundled config's global allowlist discards findings by their
# content, and discards exactly the content those rules look for - see the
# measurement recorded at the top of that file.
config="$repo_root/.gitleaks.toml"
private_info_config="$repo_root/.gitleaks-private-info.toml"
ignore="$repo_root/.gitleaksignore"
[ -f "$config" ] || die_loud 'SECRET SCAN DID NOT RUN: missing .gitleaks.toml' \
    "Expected the scan config at $config."
[ -f "$private_info_config" ] || die_loud \
    'SECRET SCAN DID NOT RUN: missing .gitleaks-private-info.toml' \
    "Expected the private-information rules at $private_info_config." \
    'The gate promises both passes, so a missing rule set fails it rather than' \
    'quietly reducing what gets checked.'
[ -f "$ignore" ] || die_loud 'SECRET SCAN DID NOT RUN: missing .gitleaksignore' \
    "Expected the reviewed-findings baseline at $ignore."

report_credentials="$(mktemp)"
report_private_info="$(mktemp)"
scan_out="$(mktemp)"
merged_config="$(mktemp)"
trap 'rm -f "$report_credentials" "$report_private_info" "$scan_out" "$merged_config"' EXIT

# The scan runs in two layers.
#
# Layer 1 is $private_info_config above, committed and public. It holds generic SHAPES - an
# absolute home path, a hostname on a private-network pseudo-TLD - and no
# site-specific value, because a value written into a public file is the leak
# it was added to prevent.
#
# Layer 2 is this optional file, host-local and outside any repository. It
# holds the literals that belong to one site and must not be published: the
# names of deployed instances, private domains, a personal email domain. It is
# OPTIONAL by design - a fresh clone, a new machine and CI all scan and pass
# on layer 1 alone, and lose only the site-specific checks. See AGENTS.md,
# "Secret scanning", for its shape and what belongs in it.
#
# Merged by concatenation into one temporary config rather than by gitleaks'
# own `[extend] path`: `[extend]` accepts a path OR useDefault, never both, and
# layer 1 needs useDefault to inherit the bundled credential rules. Two scans
# would be the other option, and would mean two reports to reconcile before the
# step can say what it found. Concatenation keeps one scan, one report, and one
# exit status - and the public file never names the private one's contents.
# The private file therefore holds `[[rules]]` blocks and nothing else.
#
# Concatenation has one sharp edge, and it is checked below rather than
# described and hoped for. Redefining something layer 1 already defines is NOT
# uniformly rejected: a second `[extend]` table is a hard TOML error, but a
# second `title` is accepted silently, and a second `[[rules]]` block reusing
# an existing `id` is accepted silently AND REPLACES the original - so a
# copy-paste in the host-local file can switch off a committed rule while the
# scan still prints "clean". Verified against the pinned binary. Reserving a
# rule-id prefix for the host-local layer removes the whole class: `site-`
# cannot collide with layer 1's `adele-` ids, and matches none of the 222
# rules bundled in gitleaks' default set.
#
# HOME is defaulted rather than required: with neither it nor XDG_CONFIG_HOME
# set, the path simply does not exist and the scan runs on layer 1, which is
# the documented state for a machine that has no host-local rules. Letting
# `set -u` abort here would fail the whole gate over an unset variable that
# this step does not actually need.
private_config="${ADELE_SECRET_SCAN_PRIVATE_RULES:-${XDG_CONFIG_HOME:-${HOME:-}/.config}/adelie-ai/gitleaks-private.toml}"

# Rule ids declared in a gitleaks config, one per line. `id` sits at the top
# level of a `[[rules]]` block; allowlist blocks have no such key.
rule_ids_in() { # rule_ids_in <file>
    grep -oE '^[[:space:]]*id[[:space:]]*=[[:space:]]*"[^"]+"' "$1" \
        | sed -E 's/.*"([^"]+)"$/\1/' || true
}

if [ -f "$private_config" ]; then
    # A file that is present but unreadable is NOT the same as an absent one.
    # Absent means "this machine checks layer 1 only", which is a supported
    # state. Unreadable means the site-specific rules were meant to apply and
    # did not, so the scan is missing a layer it was asked for - and a scan
    # missing a layer must never report clean.
    cat "$private_info_config" "$private_config" >"$merged_config" 2>/dev/null || die_loud \
        'SECRET SCAN DID NOT RUN: the host-local rules file could not be read' \
        "Found the file but could not read it, so the site-specific rules were" \
        'not applied. This is not the same as having no such file: the rules' \
        'were meant to run, so the scan is not reported as clean.' \
        '' \
        "    $private_config" \
        '' \
        'Fix the file permissions, or remove the file to scan on the committed' \
        'rules alone.'

    shadowing_ids="$(rule_ids_in "$private_config" | grep -v '^site-' || true)"
    [ -z "$shadowing_ids" ] || die_loud \
        'SECRET SCAN DID NOT RUN: a host-local rule id is not reserved to that layer' \
        'gitleaks resolves rules by id, and a later definition REPLACES an earlier' \
        'one of the same name without saying so - which would let the host-local' \
        'file switch off a committed rule while this step still reported clean.' \
        '' \
        'Host-local rule ids must therefore start with `site-`, which can collide' \
        'neither with the committed rules nor with any rule gitleaks bundles.' \
        '' \
        'Rename these ids in the host-local rules file:' \
        "$(printf '    %s\n' $shadowing_ids)"

    duplicate_ids="$(rule_ids_in "$merged_config" | sort | uniq -d)"
    [ -z "$duplicate_ids" ] || die_loud \
        'SECRET SCAN DID NOT RUN: two rules share an id' \
        'gitleaks keeps one rule per id, so the second definition silently' \
        'replaces the first and the first stops matching anything.' \
        '' \
        'Give each of these its own id:' \
        "$(printf '    %s\n' $duplicate_ids)"

    active_private_info_config="$merged_config"
    layers='private info: shapes + host-local rules'
else
    active_private_info_config="$private_info_config"
    layers="private info: shapes only (no host-local rules at $private_config)"
fi

# Scanned as `dir .` from inside the target, not `dir <absolute-path>`: gitleaks
# reports each finding's "File" using whatever path style the target argument
# used, and .gitleaksignore fingerprints are repo-relative
# (file:rule-id:line). An absolute target would make every fingerprint in
# .gitleaksignore silently stop matching - not a hypothetical, this is exactly
# how the first version of this script failed its own clean-tree test.
run_scan() { # run_scan <config> <report-path>
    local cfg="$1" rep="$2" status=0
    ( cd "$target" && gitleaks dir . \
        --config "$cfg" \
        --gitleaks-ignore-path "$ignore" \
        --report-format json \
        --report-path "$rep" \
        --redact \
        --no-banner \
        --no-color \
        --exit-code 1 \
        -v \
        >>"$scan_out" 2>&1 ) || status=$?
    return "$status"
}

scan_status=0
run_scan "$config" "$report_credentials" || scan_status=$?
private_info_status=0
run_scan "$active_private_info_config" "$report_private_info" || private_info_status=$?

# A report is proof the scan ran; an exit status alone is not. Verified
# against the real binary: a fatal error (bad config, bad path) exits
# non-zero WITHOUT writing a report, so report-existence is a reliable signal
# and not an assumption.
# Checked per pass, not once for both: a pass that produced no report checked
# nothing, and a gate that promises credentials AND private information must
# not report clean when only one of the two actually ran.
check_report() { # check_report <report-path> <what> <status>
    [ -s "$2" ] && grep -q '^\[' "$2" && return 0
    die_loud "SECRET SCAN DID NOT RUN: no report from the $1 pass" \
        "gitleaks exited ${3} without emitting a report for the $1 pass, so" \
        'that half of the scan checked nothing.' \
        '' \
        'gitleaks said:' \
        "$(sed 's/^/    /' "$scan_out" | head -40)"
}
check_report credentials "$report_credentials" "$scan_status"
check_report 'private information' "$report_private_info" "$private_info_status"

# The findings, read back out of the JSON report rather than trusted from the
# verbose stdout above: `-v` is gitleaks' own formatting and is not guaranteed
# present (e.g. under test, with a mocked binary), while the report file is
# the thing this whole step exists to insist on.
findings_summary() {
    local files rules
    files="$(grep -hoE '"File": *"[^"]*"' "$report_credentials" "$report_private_info" \
        | sed -E 's/.*: *"(.*)"$/\1/' || true)"
    rules="$(grep -hoE '"RuleID": *"[^"]*"' "$report_credentials" "$report_private_info" \
        | sed -E 's/.*: *"(.*)"$/\1/' || true)"
    paste -d'\t' <(printf '%s\n' "$rules") <(printf '%s\n' "$files") \
        | awk -F'\t' '{printf "    %s  (rule: %s)\n", $2, $1}'
}

# `|| true` because grep exits 1 when it matches nothing, which is the CLEAN
# case here - without it, `set -e` ends the run at the very moment there is
# nothing to report, and the step prints no summary at all.
finding_count="$(grep -hc '"RuleID"' "$report_credentials" "$report_private_info" \
    | awk '{total += $1} END {print total + 0}' || true)"

if [ "$scan_status" -ne 0 ] || [ "$private_info_status" -ne 0 ] || [ "$finding_count" -gt 0 ]; then
    die_loud 'FOUND IN THE WORKING TREE - hard gate failure' \
        'gitleaks matched a credential or private information in checked-out' \
        'files. Rotate and remove any real credential now - deleting the file' \
        'alone is not enough if the value may already be cached or shipped' \
        'elsewhere. For private information, replace it with a placeholder:' \
        'AGENTS.base.md rule 5.2 lists the substitutes. If this is a reviewed' \
        'false positive, add its fingerprint to .gitleaksignore with a comment' \
        'explaining why (AGENTS.md, "Secret scanning") - never delete this step' \
        'or weaken the rule set to make it pass.' \
        '' \
        "$(findings_summary)" \
        '' \
        "$(cat "$scan_out")"
fi

bytes_scanned="$(grep -oE 'scanned ~[0-9]+ bytes[^"]*' "$scan_out" | head -1 || true)"
# What actually ran is part of the result, not a detail. "clean" means two
# different things with and without the host-local rules, and a machine that
# silently has none otherwise reads as fully checked.
printf 'secret-scan: clean - credentials; %s%s\n' "$layers" "${bytes_scanned:+ - $bytes_scanned}"

#!/usr/bin/env bash
# Fake `gitleaks` for the secret-scan-gate tests. scripts/secret-scan.sh calls
# `gitleaks version` (to check the pin) and `gitleaks dir ... -r <path> ...`
# (to run the scan); this fixture answers both without a real scan, the same
# way fake-cargo-audit.sh stands in for cargo-audit in audit-gate.test.sh.
#
# Knobs (all optional):
#   FAKE_GITLEAKS_LOG      append each invocation's arguments to this file
#   FAKE_GITLEAKS_VERSION  version printed by `gitleaks version` (default: the real pin, 8.30.1)
#   FAKE_GITLEAKS_REPORT   JSON written at the `-r`/`--report-path` argument
#   FAKE_GITLEAKS_NO_REPORT  when set, never write a report - mirrors a real
#                            fatal gitleaks error (bad config, bad path), which
#                            exits non-zero and writes nothing (verified against
#                            the real binary, not assumed)
#   FAKE_GITLEAKS_STDOUT   emit this on stdout
#   FAKE_GITLEAKS_STDERR   emit this on stderr
#   FAKE_GITLEAKS_STATUS   exit with this status (default 0)
set -uo pipefail

[ -z "${FAKE_GITLEAKS_LOG:-}" ] || printf '%s\n' "$*" >>"$FAKE_GITLEAKS_LOG"

if [ "${1:-}" = version ]; then
    printf '%s\n' "${FAKE_GITLEAKS_VERSION:-8.30.1}"
    exit 0
fi

# Find the report path following -r / --report-path.
report_path=''
prev=''
for arg in "$@"; do
    if [ "$prev" = '-r' ] || [ "$prev" = '--report-path' ]; then
        report_path="$arg"
    fi
    prev="$arg"
done

if [ -z "${FAKE_GITLEAKS_NO_REPORT:-}" ] && [ -n "$report_path" ]; then
    printf '%s' "${FAKE_GITLEAKS_REPORT:-[]}" >"$report_path"
fi

[ -z "${FAKE_GITLEAKS_STDOUT:-}" ] || printf '%s' "$FAKE_GITLEAKS_STDOUT"
[ -z "${FAKE_GITLEAKS_STDERR:-}" ] || printf '%s\n' "$FAKE_GITLEAKS_STDERR" >&2
exit "${FAKE_GITLEAKS_STATUS:-0}"

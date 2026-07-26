#!/usr/bin/env bash
# Run every shell-level test suite under scripts/tests/ and aggregate the
# result. All suites run even if an earlier one fails, so one broken criterion
# does not hide the others.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
failed=()

for suite in "$here"/*.test.sh; do
    printf '\n=== %s ===\n' "$(basename "$suite")"
    "$suite" || failed+=("$(basename "$suite")")
done

if [ "${#failed[@]}" -gt 0 ]; then
    printf '\nFAILED suite(s): %s\n' "${failed[*]}" >&2
    exit 1
fi
printf '\nall shell test suites passed\n'

#!/usr/bin/env bash
# Acceptance criteria for the telemetry deploy manifests (#1149), and for the
# checks that guard them.
#
# deploy/k8s/check-telemetry.sh asserts the shape of the collector DaemonSet, its
# pipeline and the daemon's OTEL_* wiring. This suite runs it two ways: against
# the tree as committed, which must pass, and against a copy with one
# requirement broken at a time, which must fail by name.
#
# The second half is the point. A check that cannot fail reads exactly like a
# check that passes, and this repo has shipped four of those. Each entry in the
# table below breaks one requirement and names the check that must notice, and a
# final guard compares the table against the checks the script actually runs, so
# a check added without a matching break fails here rather than passing forever.
#
# Renders are not covered here - `just check-deploy` does that, because it needs
# kubectl. These assertions need only python3, so they run in the main gate.
set -euo pipefail
# shellcheck source=scripts/tests/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

CHECKER="$SCRIPT_TESTS_ROOT/deploy/k8s/check-telemetry.sh"

# One requirement broken, and the check that must notice. The python fragment
# runs with `docs` bound to the YAML documents of the named file.
#
# Format: <check name>|<file under deploy/k8s>|<python fragment>
MUTATIONS=(
"otel_collector_receives_otlp_over_grpc_and_http|base/otel-collector.yaml|ds=[d for d in docs if d['kind']=='DaemonSet'][0]; c=ds['spec']['template']['spec']['containers'][0]; c['ports']=[p for p in c['ports'] if p['containerPort']!=4318]"
"otel_collector_pipes_all_three_signals|base/otel-collector-config.yaml|del docs[0]['service']['pipelines']['logs']"
"otel_collector_batches_before_it_exports|base/otel-collector-config.yaml|docs[0]['service']['pipelines']['traces']['processors']=['memory_limiter']"
"otel_collector_backend_is_an_overlay_placeholder|base/otel-collector-config.yaml|docs[0]['exporters']['otlp']['endpoint']='collector.some-backend.net:4317'"
"otel_collector_is_a_daemonset_registered_in_the_base|base/kustomization.yaml|docs[0]['resources']=[r for r in docs[0]['resources'] if r!='otel-collector.yaml']"
"otel_collector_config_change_rolls_the_pods|base/kustomization.yaml|docs[0]['generatorOptions']={'disableNameSuffixHash': True}"
"daemon_exports_to_the_node_local_collector|base/otel-collector.yaml|svc=[d for d in docs if d['kind']=='Service'][0]; del svc['spec']['internalTrafficPolicy']"
"daemon_labels_its_telemetry_with_namespace_pod_and_node|base/daemon.yaml|dep=[d for d in docs if d['kind']=='Deployment'][0]; c=[x for x in dep['spec']['template']['spec']['containers'] if x['name']=='daemon'][0]; c['env']=[e for e in c['env'] if e['name']!='OTEL_RESOURCE_ATTRIBUTES']"
"daemon_has_time_to_flush_on_termination|base/daemon.yaml|dep=[d for d in docs if d['kind']=='Deployment'][0]; del dep['spec']['template']['spec']['terminationGracePeriodSeconds']"
"telemetry_manifests_name_no_real_environment|base/otel-collector.yaml|[d['metadata'].update(namespace='adele-somewhere') for d in docs]"
)

# --- harness -----------------------------------------------------------------

# A copy of deploy/k8s that a mutation can be applied to.
_copy_tree() {
    local root="$TEST_TMP/tree"
    rm -rf "$root"
    mkdir -p "$root/deploy"
    cp -r "$SCRIPT_TESTS_ROOT/deploy/k8s" "$root/deploy/k8s"
    printf '%s' "$root"
}

# Rewrite a YAML file through a python fragment. Comments are lost, which does
# not matter for a throwaway copy.
_edit_yaml() { # _edit_yaml <file> <python fragment>
    python3 - "$1" "$2" <<'PY'
import sys

import yaml

path, fragment = sys.argv[1], sys.argv[2]
with open(path) as fh:
    docs = [d for d in yaml.safe_load_all(fh) if d]
exec(fragment)  # noqa: S102 - the fragment is this suite's own test data
with open(path, "w") as fh:
    yaml.safe_dump_all(docs, fh)
PY
}

_check_names_from_a_passing_run() {
    run_cmd "$CHECKER"
    [ "$RUN_STATUS" -eq 0 ] || fail "the checker did not pass against the tree as committed:
$RUN_OUT
$RUN_ERR"
    printf '%s\n' "$RUN_OUT" | awk '/^PASS /{print $2}' | sort
}

# --- the manifests themselves ------------------------------------------------

the_telemetry_manifests_pass_their_own_checks() {
    run_cmd "$CHECKER"
    [ "$RUN_STATUS" -eq 0 ] || fail "check-telemetry.sh failed:
$RUN_OUT
$RUN_ERR"
}

# --- the checks are able to fail ---------------------------------------------

every_telemetry_check_fails_when_its_requirement_is_broken() {
    local entry name file fragment root
    for entry in "${MUTATIONS[@]}"; do
        IFS='|' read -r name file fragment <<<"$entry"
        root="$(_copy_tree)"
        _edit_yaml "$root/deploy/k8s/$file" "$fragment"
        run_cmd "$CHECKER" --root "$root"
        [ "$RUN_STATUS" -ne 0 ] || fail \
            "breaking $file did not fail any check; $name cannot fail:
$RUN_OUT"
        assert_contains "$RUN_OUT" "FAIL $name" "breaking $file must fail $name"
    done
}

the_break_table_covers_every_check_that_runs() {
    # Derived from a live passing run rather than from a second hardcoded list,
    # so a check added to the script without a matching break fails here, and a
    # stale entry fails too.
    local running tabled
    running="$(_check_names_from_a_passing_run)"
    tabled="$(printf '%s\n' "${MUTATIONS[@]}" | cut -d'|' -f1 | sort)"
    [ -n "$running" ] || fail 'the checker printed no PASS lines; it changed shape'
    assert_eq "$running" "$tabled" 'every check that runs needs a break that proves it can fail'
}

run_test the_telemetry_manifests_pass_their_own_checks
run_test every_telemetry_check_fails_when_its_requirement_is_broken
run_test the_break_table_covers_every_check_that_runs
finish_tests 'k8s-telemetry'

#!/usr/bin/env bash
# Named-check assertions for the telemetry deploy (deploy/k8s/base/otel-collector.yaml,
# deploy/k8s/base/otel-collector-config.yaml and the OTEL_* wiring in
# deploy/k8s/base/daemon.yaml). These are manifest-shape tests, not a live-cluster
# run: they read the manifests and assert the deploy exports all three signals to a
# node-local collector that an overlay points at a backend. Never contacts the API
# server.
#
# Named checks (legible from output, one requirement each):
#   otel_collector_receives_otlp_over_grpc_and_http
#   otel_collector_pipes_all_three_signals
#   otel_collector_batches_before_it_exports
#   otel_collector_backend_is_an_overlay_placeholder
#   otel_collector_is_a_daemonset_registered_in_the_base
#   otel_collector_config_change_rolls_the_pods
#   daemon_exports_to_the_node_local_collector
#   daemon_labels_its_telemetry_with_namespace_pod_and_node
#   daemon_has_time_to_flush_on_termination
#   telemetry_manifests_name_no_real_environment
#
# Usage: check-telemetry.sh [--root <repo root>]
#   --root points the checks at another copy of the tree, which is how
#   scripts/tests/k8s-telemetry.test.sh proves they are able to fail.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

while [ $# -gt 0 ]; do
    case "$1" in
        --root)
            repo_root="$2"
            shift 2
            ;;
        *)
            echo "usage: $(basename "$0") [--root <repo root>]" >&2
            exit 2
            ;;
    esac
done

base="${repo_root}/deploy/k8s/base"
collector="${base}/otel-collector.yaml"
collector_config="${base}/otel-collector-config.yaml"
daemon="${base}/daemon.yaml"
kustomization="${base}/kustomization.yaml"

for f in "${collector}" "${collector_config}" "${daemon}" "${kustomization}"; do
    if [ ! -f "${f}" ]; then
        echo "FAIL: required file missing: ${f}" >&2
        exit 1
    fi
done

python3 - "${collector}" "${collector_config}" "${daemon}" "${kustomization}" <<'PY'
import re
import sys

import yaml

collector_path, collector_config_path, daemon_path, kustomization_path = sys.argv[1:5]

failures = []


def check(name, ok, reason=""):
    if ok:
        print(f"PASS {name}")
    else:
        print(f"FAIL {name}: {reason}")
        failures.append(name)


def docs(path):
    with open(path) as fh:
        return [d for d in yaml.safe_load_all(fh) if d]


def only(items, what, path):
    if len(items) != 1:
        print(f"FAIL: expected exactly one {what} in {path}, found {len(items)}")
        sys.exit(1)
    return items[0]


collector_docs = docs(collector_path)
daemon_docs = docs(daemon_path)
kustomization = only(docs(kustomization_path), "kustomization", kustomization_path)

daemonset = only(
    [d for d in collector_docs if d.get("kind") == "DaemonSet"], "DaemonSet", collector_path
)
collector_service = only(
    [d for d in collector_docs if d.get("kind") == "Service"], "Service", collector_path
)
collector_pod = daemonset["spec"]["template"]["spec"]
collector_container = only(
    collector_pod.get("containers", []), "collector container", collector_path
)

with open(collector_config_path) as fh:
    pipeline = yaml.safe_load(fh)

daemon_deployment = only(
    [d for d in daemon_docs if d.get("kind") == "Deployment"], "Deployment", daemon_path
)
daemon_pod = daemon_deployment["spec"]["template"]["spec"]
daemon_container = only(
    [c for c in daemon_pod.get("containers", []) if c.get("name") == "daemon"],
    "daemon container",
    daemon_path,
)
daemon_env = {e["name"]: e for e in daemon_container.get("env", [])}


def env_value(name):
    return daemon_env.get(name, {}).get("value", "")


def field_ref(name):
    return (
        daemon_env.get(name, {})
        .get("valueFrom", {})
        .get("fieldRef", {})
        .get("fieldPath", "")
    )


# --- otel_collector_receives_otlp_over_grpc_and_http -------------------------
# A workload picks its transport from OTEL_EXPORTER_OTLP_PROTOCOL at run time,
# so the collector has to answer on both or half the fleet exports into a
# refused connection.
protocols = pipeline.get("receivers", {}).get("otlp", {}).get("protocols", {})
container_ports = {p.get("containerPort") for p in collector_container.get("ports", [])}
service_ports = {p.get("port") for p in collector_service["spec"].get("ports", [])}
check(
    "otel_collector_receives_otlp_over_grpc_and_http",
    "grpc" in protocols
    and "http" in protocols
    and {4317, 4318} <= container_ports
    and {4317, 4318} <= service_ports,
    "the otlp receiver must enable both protocols, and 4317 + 4318 must be open on "
    f"the container (found {sorted(p for p in container_ports if p)}) and on the "
    f"Service (found {sorted(p for p in service_ports if p)})",
)

# --- otel_collector_pipes_all_three_signals ----------------------------------
pipelines = pipeline.get("service", {}).get("pipelines", {})
wired = {
    signal: pipelines.get(signal, {})
    for signal in ("traces", "metrics", "logs")
}
all_three = all(
    "otlp" in (p.get("receivers") or []) and "otlp" in (p.get("exporters") or [])
    for p in wired.values()
)
check(
    "otel_collector_pipes_all_three_signals",
    all_three,
    "traces, metrics and logs must each run from the otlp receiver to the otlp "
    f"exporter; found {sorted(pipelines)}",
)

# --- otel_collector_batches_before_it_exports --------------------------------
check(
    "otel_collector_batches_before_it_exports",
    all("batch" in (p.get("processors") or []) for p in wired.values()),
    "every pipeline must carry the batch processor, or each span is one export call",
)

# --- otel_collector_backend_is_an_overlay_placeholder ------------------------
# The base names no backend. An overlay replaces this file with its own, so the
# endpoint here must be a placeholder rather than a reachable host.
endpoint = str(pipeline.get("exporters", {}).get("otlp", {}).get("endpoint", ""))
check(
    "otel_collector_backend_is_an_overlay_placeholder",
    endpoint.endswith(".example.com:4317") or endpoint.endswith(".example.com:4318"),
    f"the exporter endpoint must be an example.com placeholder, found {endpoint!r}",
)

# --- otel_collector_is_a_daemonset_registered_in_the_base --------------------
# A Deployment would put the collector on one node and every other node's
# telemetry on a cross-node hop. A file absent from `resources:` renders as
# nothing at all.
check(
    "otel_collector_is_a_daemonset_registered_in_the_base",
    daemonset["kind"] == "DaemonSet"
    and "otel-collector.yaml" in (kustomization.get("resources") or []),
    "otel-collector.yaml must hold a DaemonSet and be listed in the base's resources",
)

# --- otel_collector_config_change_rolls_the_pods -----------------------------
# The seed daemon.toml deliberately has a fixed name (it only ever seeds a fresh
# volume). The collector's config is live config: a change to it must produce a
# new ConfigMap name so the DaemonSet rolls, or the edit sits in the cluster
# doing nothing until somebody restarts the pods by hand.
#
# Both places that can take the hash off are read, because kustomize folds a
# global `generatorOptions` into every generator and a local `false` cannot win
# against a global `true` - the two are the same value once folded. A global
# setting therefore silences this quietly, and the manifest still looks right.
generators = kustomization.get("configMapGenerator") or []
collector_generator = [
    g
    for g in generators
    if any("otel-collector-config.yaml" in f for f in (g.get("files") or []))
]
globally_off = (kustomization.get("generatorOptions") or {}).get(
    "disableNameSuffixHash"
) is True
locally_off = bool(collector_generator) and (
    collector_generator[0].get("options", {}).get("disableNameSuffixHash") is True
)
mounts_generated_name = any(
    v.get("configMap", {}).get("name")
    == (collector_generator[0]["name"] if collector_generator else None)
    for v in collector_pod.get("volumes", [])
    if "configMap" in v
)
check(
    "otel_collector_config_change_rolls_the_pods",
    bool(collector_generator)
    and not globally_off
    and not locally_off
    and mounts_generated_name,
    "otel-collector-config.yaml must be a configMapGenerator whose name-suffix "
    "hash is left on - by this generator's own options and by the absence of a "
    "global generatorOptions that disables it - and the DaemonSet must mount that "
    "generated ConfigMap by name",
)

# --- daemon_exports_to_the_node_local_collector ------------------------------
# internalTrafficPolicy: Local is what keeps the hop on the node. A hostPort
# would do the same and cannot: it is a node-wide claim, so a second instance in
# the same cluster never schedules (proved - both its DaemonSet pods stay
# Pending with "didn't have free ports for the requested pod ports").
otlp_endpoint = env_value("OTEL_EXPORTER_OTLP_ENDPOINT")
service_name = collector_service["metadata"]["name"]
check(
    "daemon_exports_to_the_node_local_collector",
    service_name in otlp_endpoint
    and collector_service["spec"].get("internalTrafficPolicy") == "Local",
    f"OTEL_EXPORTER_OTLP_ENDPOINT ({otlp_endpoint!r}) must name the {service_name} "
    "Service, and that Service must set internalTrafficPolicy: Local",
)

# --- daemon_labels_its_telemetry_with_namespace_pod_and_node -----------------
# Without these every instance in the cluster is one undifferentiated service in
# the backend, and "which pod produced this?" has no answer.
attributes = env_value("OTEL_RESOURCE_ATTRIBUTES")
wanted = {
    "k8s.namespace.name": "metadata.namespace",
    "k8s.pod.name": "metadata.name",
    "k8s.node.name": "spec.nodeName",
}
missing = []
for attribute, path in wanted.items():
    reference = re.search(rf"{re.escape(attribute)}=\$\((\w+)\)", attributes)
    if not reference or field_ref(reference.group(1)) != path:
        missing.append(f"{attribute} (from {path})")
check(
    "daemon_labels_its_telemetry_with_namespace_pod_and_node",
    not missing,
    "OTEL_RESOURCE_ATTRIBUTES must carry "
    + ", ".join(wanted)
    + " from the downward API; missing or unsourced: "
    + ", ".join(missing),
)

# --- daemon_has_time_to_flush_on_termination ---------------------------------
# Kubernetes stops every pod with SIGTERM. The daemon handles it and the
# telemetry guard drops after it, flushing what is still buffered - but only
# within the grace period. Stated explicitly so a change to the cluster default
# cannot silence the last window of a rollout.
grace = daemon_pod.get("terminationGracePeriodSeconds")
check(
    "daemon_has_time_to_flush_on_termination",
    isinstance(grace, int) and grace >= 10,
    "the daemon pod must set terminationGracePeriodSeconds explicitly, and leave "
    f"room for the 5-second flush; found {grace!r}",
)

# --- telemetry_manifests_name_no_real_environment ----------------------------
# This repo is public. The base names no namespace, no real host and no real
# registry (see the header of kustomization.yaml).
leaks = []
for path in (collector_path, collector_config_path):
    with open(path) as fh:
        for number, line in enumerate(fh, start=1):
            if re.search(r"^\s*namespace:", line):
                leaks.append(f"{path}:{number} names a namespace")
            for host in re.findall(r"[\w.-]+\.(?:lab|local|internal|int)\b[\w.-]*", line):
                leaks.append(f"{path}:{number} names {host}")
            if re.search(r"\d+\.\d+\.\d+\.\d+", line) and "0.0.0.0" not in line:
                leaks.append(f"{path}:{number} names an address")
check(
    "telemetry_manifests_name_no_real_environment",
    not leaks,
    "; ".join(leaks),
)

if failures:
    print(f"\n{len(failures)} check(s) failed: {', '.join(failures)}", file=sys.stderr)
    sys.exit(1)
print("\nAll telemetry deploy checks passed.")
PY

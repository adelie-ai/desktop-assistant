#!/usr/bin/env bash
# The opt-in property, read from what kustomize actually renders.
#
# check-telemetry.sh beside this file asserts the same property from the
# manifests. That check is sound, and it is not the whole story - a manifest can
# read correctly and render otherwise. This change already produced one of those:
# a global `generatorOptions: disableNameSuffixHash: true` silently overrode a
# local `false`, so the ConfigMap rendered unhashed while the file said it would
# not. So this step renders and reads the objects instead.
#
# The property is "an overlay that does not opt in renders exactly what it
# rendered before telemetry existed". It is asserted as *no telemetry appears in
# the render at all* rather than as a diff against `main`, deliberately:
#
#   - A diff against `main` needs a fetched remote ref, so it fails offline for a
#     reason that has nothing to do with the change under test.
#   - It would fail on every later pull request that changes the base for an
#     unrelated reason, which is most of them, so it would be deleted within the
#     month.
#
# "No telemetry in the render" is the same statement while this branch leaves
# `deploy/k8s/base/` byte-identical to `main` - checked, and 0 files differ - and
# unlike the diff it stays true and stays useful afterwards.
#
# Named checks:
#   rendered_base_carries_no_telemetry
#   rendered_opted_in_overlay_deploys_one_collector
#   rendered_opted_in_daemon_gets_its_telemetry_wiring
#   rendered_overlay_backend_patch_reaches_the_collector
#
# Usage: check-telemetry-render.sh [--root <repo root>]
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

command -v kubectl >/dev/null 2>&1 || {
    echo "FAIL: kubectl is required to render the manifests" >&2
    exit 1
}

base_render="$(kubectl kustomize "${repo_root}/deploy/k8s/base")"
overlay_render="$(kubectl kustomize "${repo_root}/deploy/k8s/overlays/example")"

python3 - "${base_render}" "${overlay_render}" <<'PY'
import re
import sys

import yaml

base_text, overlay_text = sys.argv[1], sys.argv[2]
base = [d for d in yaml.safe_load_all(base_text) if d]
overlay = [d for d in yaml.safe_load_all(overlay_text) if d]

failures = []


def check(name, ok, reason=""):
    if ok:
        print(f"PASS {name}")
    else:
        print(f"FAIL {name}: {reason}")
        failures.append(name)


def names(objects, kind):
    return [d["metadata"]["name"] for d in objects if d.get("kind") == kind]


# A render that produced nothing would pass every emptiness assertion below.
if not base or not overlay:
    print("FAIL: a render produced no objects, so nothing was checked")
    sys.exit(1)

# --- rendered_base_carries_no_telemetry --------------------------------------
# Every way telemetry can reach an install that did not ask for it, read from the
# rendered text rather than from the files: an object of its own, a variable on
# somebody's pod, or a field that exists only to protect a flush.
leaks = []
for pattern, what in (
    (r"otel-collector", "a collector object"),
    (r"OTEL_\w+", "an OTEL_* variable"),
    (r"K8S_(?:NAMESPACE|POD_NAME|NODE_NAME)", "a telemetry downward-API variable"),
    (r"terminationGracePeriodSeconds", "a termination grace period"),
):
    if re.search(pattern, base_text):
        leaks.append(what)
check(
    "rendered_base_carries_no_telemetry",
    not leaks,
    "an overlay that does not opt in must render exactly what it rendered before "
    f"telemetry existed; the base render carries {leaks}",
)

# --- rendered_opted_in_overlay_deploys_one_collector -------------------------
collectors = names(overlay, "DaemonSet")
mounted = [
    volume["configMap"]["name"]
    for d in overlay
    if d.get("kind") == "DaemonSet"
    for volume in d["spec"]["template"]["spec"].get("volumes", [])
    if "configMap" in volume
]
rendered_config = [name for name in names(overlay, "ConfigMap") if "otel" in name]
# A content hash is what makes an edit roll the DaemonSet. The generated name is
# the declared name plus a suffix, so a name equal to the declared one means the
# hash was turned off somewhere between here and the generator.
hashed = [name for name in rendered_config if name != "otel-collector-config"]
check(
    "rendered_opted_in_overlay_deploys_one_collector",
    collectors == ["otel-collector"]
    and len(rendered_config) == 1
    and hashed == rendered_config
    and mounted == rendered_config,
    "an overlay that opts in must render exactly one collector DaemonSet mounting "
    "one hash-suffixed collector ConfigMap; rendered DaemonSets "
    f"{collectors}, ConfigMaps {rendered_config}, mounts {mounted}",
)

# --- rendered_opted_in_daemon_gets_its_telemetry_wiring ----------------------
# The other half of the move: the daemon's variables now arrive by a patch in the
# component, and a patch that fails to apply is silent - the render simply comes
# out without them. Kubernetes expands $(VAR) only from entries earlier in the
# final list, and a strategic merge decides that order, so the order is read here
# rather than assumed.
daemons = [
    d
    for d in overlay
    if d.get("kind") == "Deployment" and d["metadata"]["name"] == "adele-daemon"
]
if len(daemons) != 1:
    print(f"FAIL: expected one adele-daemon Deployment, found {len(daemons)}")
    sys.exit(1)
daemon_pod = daemons[0]["spec"]["template"]["spec"]
container = [c for c in daemon_pod["containers"] if c["name"] == "daemon"][0]
order = [e["name"] for e in container.get("env", [])]
referenced = re.findall(r"\$\((\w+)\)", next(
    (e.get("value", "") for e in container["env"] if e["name"] == "OTEL_RESOURCE_ATTRIBUTES"),
    "",
))
expands = bool(referenced) and all(
    name in order and order.index(name) < order.index("OTEL_RESOURCE_ATTRIBUTES")
    for name in referenced
)
check(
    "rendered_opted_in_daemon_gets_its_telemetry_wiring",
    "OTEL_EXPORTER_OTLP_ENDPOINT" in order
    and "OTEL_RESOURCE_ATTRIBUTES" in order
    and daemon_pod.get("terminationGracePeriodSeconds") == 30
    and expands,
    "opting in must put the OTEL_* variables and a 30s grace period on the daemon, "
    "with every variable OTEL_RESOURCE_ATTRIBUTES interpolates defined before it; "
    f"env order {order}, grace {daemon_pod.get('terminationGracePeriodSeconds')!r}, "
    f"interpolates {referenced}",
)

# --- rendered_overlay_backend_patch_reaches_the_collector --------------------
# The documented way to name a backend is a patch on these two values. The first
# mechanism this change documented - `behavior: replace` on the component's
# generated ConfigMap - does not work at all: kustomize cannot see a component's
# generator to merge against, and the render fails with "does not exist; cannot
# merge or replace". A documented mechanism that nobody exercises is how that
# shipped, so the example overlay uses it and this asserts it lands.
collector_env = {}
for d in overlay:
    if d.get("kind") == "DaemonSet":
        for c in d["spec"]["template"]["spec"]["containers"]:
            collector_env = {e["name"]: e.get("value") for e in c.get("env", [])}
overlay_endpoint = collector_env.get("OTLP_BACKEND_ENDPOINT", "")
check(
    "rendered_overlay_backend_patch_reaches_the_collector",
    overlay_endpoint.endswith(".example.com:4317")
    and overlay_endpoint != "otlp-backend.example.com:4317"
    and "OTLP_BACKEND_INSECURE" in collector_env,
    "the example overlay must name its backend by patching OTLP_BACKEND_ENDPOINT "
    "and OTLP_BACKEND_INSECURE onto the collector, and the patched value must "
    f"reach the render; found {collector_env}",
)

if failures:
    print(f"\n{len(failures)} check(s) failed: {', '.join(failures)}", file=sys.stderr)
    sys.exit(1)
print("\nAll telemetry render checks passed.")
PY

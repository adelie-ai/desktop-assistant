#!/usr/bin/env bash
# Named-check assertions for the #500 RLS-bootstrap deploy (deploy/k8s/base/rls-bootstrap.yaml
# + the `deploy-rls-bootstrap` justfile recipe). These are manifest-shape tests,
# not a live-cluster run: they read the manifest, the justfile recipe, and the
# canonical rls_role.sql and assert the deploy provisions the `adele_query` role
# correctly. Runnable in CI; never contacts the API server.
#
# Named checks (legible from output, one requirement each):
#   rls_bootstrap_manifest_runs_rls_role_sql   - a Job mounts + psql-executes rls_role.sql
#   rls_bootstrap_passes_app_role_adele        - invocation includes `-v app_role=adele`
#   rls_bootstrap_gated_on_postgres_ready      - runs only after `pg_isready`
#   rls_bootstrap_is_rerunnable                - re-applying is idempotent
#   rls_bootstrap_configmap_from_canonical_sql - SQL is generated from the canonical
#                                                file (no hand-copied, rot-prone SQL)
set -euo pipefail

# Resolve repo root from this script's location so it runs from any cwd.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

manifest="${repo_root}/deploy/k8s/base/rls-bootstrap.yaml"
justfile="${repo_root}/justfile"
sql="${repo_root}/crates/storage/bootstrap/rls_role.sql"

for f in "${manifest}" "${justfile}" "${sql}"; do
  if [ ! -f "${f}" ]; then
    echo "FAIL: required file missing: ${f}" >&2
    exit 1
  fi
done

python3 - "${manifest}" "${justfile}" "${sql}" <<'PY'
import re
import sys

import yaml

manifest_path, justfile_path, sql_path = sys.argv[1:4]

failures = []


def check(name, ok, reason=""):
    if ok:
        print(f"PASS {name}")
    else:
        print(f"FAIL {name}: {reason}")
        failures.append(name)


with open(manifest_path) as fh:
    docs = [d for d in yaml.safe_load_all(fh) if d]

jobs = [d for d in docs if d.get("kind") == "Job"]
if len(jobs) != 1:
    print(f"FAIL: expected exactly one Job in {manifest_path}, found {len(jobs)}")
    sys.exit(1)

job = jobs[0]
pod = job["spec"]["template"]["spec"]
containers = pod.get("containers", [])
init_containers = pod.get("initContainers", [])
volumes = pod.get("volumes", [])


def cmd_text(container):
    return " \n ".join(container.get("command", []) + container.get("args", []))


main_cmds = " \n ".join(cmd_text(c) for c in containers)
init_cmds = " \n ".join(cmd_text(c) for c in init_containers)
justfile = open(justfile_path).read()
sql = open(sql_path).read()

# --- rls_bootstrap_manifest_runs_rls_role_sql --------------------------------
runs_psql = re.search(r"psql\b", main_cmds) is not None
runs_file = re.search(r"-f\s+/bootstrap/rls_role\.sql", main_cmds) is not None
configmap_vol_names = {v["name"] for v in volumes if "configMap" in v}
mounts_at_bootstrap = any(
    m.get("mountPath") == "/bootstrap" and m.get("name") in configmap_vol_names
    for c in containers
    for m in c.get("volumeMounts", [])
)
check(
    "rls_bootstrap_manifest_runs_rls_role_sql",
    runs_psql and runs_file and mounts_at_bootstrap,
    "Job must mount a ConfigMap at /bootstrap and run `psql ... -f /bootstrap/rls_role.sql`",
)

# --- rls_bootstrap_passes_app_role_adele -------------------------------------
check(
    "rls_bootstrap_passes_app_role_adele",
    re.search(r"-v\s+app_role=adele\b", main_cmds) is not None,
    "psql invocation must include `-v app_role=adele` (the daemon's connect role)",
)

# --- rls_bootstrap_gated_on_postgres_ready -----------------------------------
check(
    "rls_bootstrap_gated_on_postgres_ready",
    "pg_isready" in init_cmds,
    "an initContainer must gate the Job on postgres readiness via pg_isready",
)

# --- rls_bootstrap_is_rerunnable ---------------------------------------------
# Idempotency has two layers: the SQL swallows a duplicate role and its grants
# self-heal (WITH ADMIN OPTION + ALTER DEFAULT PRIVILEGES), and the deploy
# recipe clears any prior Job before re-applying (a Job's pod template is
# immutable, so a bare re-apply would error). The pod restartPolicy must also be
# a Job-legal value so pod-level retries of the idempotent SQL are safe.
sql_idempotent = (
    "duplicate_object" in sql
    and "WITH ADMIN OPTION" in sql
    and "ALTER DEFAULT PRIVILEGES" in sql
)
restart_ok = pod.get("restartPolicy") in ("OnFailure", "Never")
recipe_clears_prior_job = (
    re.search(r"kubectl\s+delete\s+job\s+rls-bootstrap\b[^\n]*--ignore-not-found", justfile)
    is not None
)
check(
    "rls_bootstrap_is_rerunnable",
    sql_idempotent and restart_ok and recipe_clears_prior_job,
    "SQL must be idempotent (duplicate-role swallow + self-healing grants), pod "
    "restartPolicy must be OnFailure/Never, and the recipe must delete any prior "
    "Job (--ignore-not-found) before re-applying",
)

# --- rls_bootstrap_configmap_from_canonical_sql (anti-drift) -----------------
# The running SQL must be generated from the canonical file, never hand-copied
# into a manifest where it can rot. Assert the recipe generates a ConfigMap
# named rls-bootstrap-sql from crates/storage/bootstrap/rls_role.sql, and that
# the Job mounts exactly that ConfigMap.
recipe_generates = (
    re.search(r"create configmap\s+rls-bootstrap-sql\b", justfile) is not None
    and re.search(
        r"--from-file=rls_role\.sql=crates/storage/bootstrap/rls_role\.sql", justfile
    )
    is not None
)
manifest_mounts_that_cm = any(
    v.get("configMap", {}).get("name") == "rls-bootstrap-sql"
    for v in volumes
    if "configMap" in v
)
check(
    "rls_bootstrap_configmap_from_canonical_sql",
    recipe_generates and manifest_mounts_that_cm,
    "the deploy recipe must generate ConfigMap rls-bootstrap-sql from "
    "crates/storage/bootstrap/rls_role.sql (no hand-copied SQL), and the Job must "
    "mount that ConfigMap",
)

if failures:
    print(f"\n{len(failures)} check(s) failed: {', '.join(failures)}", file=sys.stderr)
    sys.exit(1)
print("\nAll RLS-bootstrap deploy checks passed.")
PY

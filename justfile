set shell := ["bash", "-euo", "pipefail", "-c"]

service_name := "desktop-assistant-daemon"
dev_service_name := "desktop-assistant-daemon-dev"
dev_dbus_service := "org.desktopAssistant.Dev"
service_src := "systemd/desktop-assistant-daemon.service"
dev_service_src := "systemd/desktop-assistant-daemon-dev.service"
service_dst := env_var_or_default("XDG_CONFIG_HOME", env_var("HOME") + "/.config") + "/systemd/user/" + service_name + ".service"
dev_service_dst := env_var_or_default("XDG_CONFIG_HOME", env_var("HOME") + "/.config") + "/systemd/user/" + dev_service_name + ".service"
dbus_service_src := "systemd/org.desktopAssistant.service"
dbus_service_dev_src := "systemd/org.desktopAssistant.Dev.service"
dbus_service_dir := env_var_or_default("XDG_DATA_HOME", env_var("HOME") + "/.local/share") + "/dbus-1/services"
dbus_service_dst := dbus_service_dir + "/org.desktopAssistant.service"
dbus_service_dev_dst := dbus_service_dir + "/org.desktopAssistant.Dev.service"
# D-Bus bridge (the cutover's step 5, #317). The bridge reaches the daemon over
# a peer-cred-authenticated local UDS (#407) — no JWT minter.
bridge_service_name := "adelie-dbus-bridge"
bridge_service_src := "systemd/adelie-dbus-bridge.service"
systemd_user_dir := env_var_or_default("XDG_CONFIG_HOME", env_var("HOME") + "/.config") + "/systemd/user"
bridge_service_dst := systemd_user_dir + "/adelie-dbus-bridge.service"
container_cli := env_var_or_default("CONTAINER_CLI", "docker")
container_security_opts := env_var_or_default("CONTAINER_SECURITY_OPTS", "--security-opt label=disable")
debian_builder_image := env_var_or_default("DEBIAN_BUILDER_IMAGE", "debian:trixie")
rpm_builder_image := env_var_or_default("RPM_BUILDER_IMAGE", "fedora:43")
flatpak_builder_image := env_var_or_default("FLATPAK_BUILDER_IMAGE", "fedora:43")
snap_builder_image := env_var_or_default("SNAP_BUILDER_IMAGE", "docker.io/snapcore/snapcraft:stable")

# List available commands
default: list

@list:
    just --list

# Run backend daemon in foreground (dev)
backend:
    cargo run -p desktop-assistant-daemon

# Run backend daemon on development D-Bus name (coexists with user service)
dev-backend:
    DESKTOP_ASSISTANT_DBUS_SERVICE={{dev_dbus_service}} cargo run -p desktop-assistant-daemon

# Build all workspace crates
build:
    cargo build --workspace

# --- Local verification ("local CI") -----------------------------------------
# We run these locally instead of GitHub Actions. `install-hooks` wires `check`
# into a git pre-push hook so it runs automatically before every push.

# Full local gate: dependency scan, secret scan, formatting, lints, build,
# tests, and the tests for the gate's own shell steps (on the pinned
# toolchain).
# `audit` and `secret-scan` come first on purpose: build scripts execute at
# first compile - under `clippy` as much as under `build` - so both scans have
# to precede either. The `-sqlite` and `-mcp-host` steps are separate because
# that adapter/module is compiled out of the workspace steps entirely; see
# `lint-sqlite` and `lint-mcp-host`.
check: audit secret-scan fmt-check lint lint-sqlite lint-mcp-host build test test-sqlite test-mcp-host test-scripts

# RustSec advisory scan of Cargo.lock. Fails loudly when cargo-audit is missing
# or the advisory database cannot be fetched, rather than passing on silence
# (#706); see scripts/audit.sh for the offline opt-in.
audit:
    ./scripts/audit.sh

# Working-tree secret scan (#811): gitleaks over the checked-out files, not
# git history - the key that prompted this issue was never committed, so a
# history-only scan would have reported clean the whole time it was exposed.
# Fails loudly when gitleaks is missing, is the wrong (unpinned) version, or
# produces no report, rather than passing on silence; see scripts/secret-scan.sh.
secret-scan:
    ./scripts/secret-scan.sh

# Verify formatting without modifying files
fmt-check:
    cargo fmt --all --check

# Apply formatting
fmt:
    cargo fmt --all

# Clippy across the workspace; warnings are errors
lint:
    cargo clippy --workspace --all-targets -- -D warnings

# Clippy the SQLite adapter with its own feature enabled (#742). Every module
# and test file in crates/storage-sqlite is `#[cfg(feature = "sqlite")]`, and
# the feature is off by default so the daemon never links the sqlite C library
# - which means the workspace lint above type-checks an empty crate. Without
# this step a change to a `core::ports` trait can break the adapter with no
# signal at all.
lint-sqlite:
    cargo clippy -p desktop-assistant-storage-sqlite --features sqlite --all-targets -- -D warnings

# Clippy the client-side MCP host with its own feature enabled (#910). The
# `mcp-host` module (crates/client-common/src/mcp_host) is
# `#[cfg(feature = "mcp-host")]`, off by default, and no other workspace crate
# enables it as a dependency - so the workspace lint above type-checks none of
# it. Without this step, a change to this module (including what environment a
# spawned client-side server receives) can break it with no signal at all.
lint-mcp-host:
    cargo clippy -p desktop-assistant-client-common --features mcp-host --all-targets -- -D warnings

# Run the workspace test suite (excludes #[ignore] integration tests)
test:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "${TEST_DATABASE_URL:-}" ]; then
      echo "⚠  TEST_DATABASE_URL unset — storage multi-tenant ISOLATION suites will pass-SKIP." >&2
      echo "   A green run here does NOT verify cross-tenant safety. Run 'just test-db' for that." >&2
    fi
    cargo test --workspace

# Run the SQLite adapter's own suite with its feature enabled (#742). Its test
# files are `#![cfg(feature = "sqlite")]`, so `cargo test --workspace` builds
# them as empty binaries and runs none of them - including the guard that every
# migration file is registered with the runner.
test-sqlite:
    cargo test -p desktop-assistant-storage-sqlite --features sqlite

# Run the client-side MCP host's own suite with its feature enabled (#910).
# Its module is `#[cfg(feature = "mcp-host")]`, so `cargo test --workspace`
# builds it as an empty crate and runs none of its tests - including the
# environment-isolation coverage for the spawn path a real desktop session
# uses.
test-mcp-host:
    cargo test -p desktop-assistant-client-common --features mcp-host

# Real-Secret-Service integration tests (needs a live session bus; mutates + cleans keyring)
test-integration:
    cargo test --workspace -- --ignored

# Named tests for the shell-level gate steps under scripts/ (the gate's own
# plumbing: the dependency scan and the throwaway-Postgres harness).
test-scripts:
    ./scripts/tests/run.sh

# --- DB-gated storage isolation suites (#444) ------------------------------
# The `crates/storage` isolation suites pass-skip when TEST_DATABASE_URL is
# unset, so a bare `cargo test` proves nothing about multi-tenant safety.
# These recipes boot a throwaway pgvector container (the `vector` extension is
# pre-created by the auto-loaded init fixture under
# crates/storage/tests/fixtures/initdb/), so a real isolation run is one
# command. Honors CONTAINER_CLI, else auto-detects podman/docker.
#
# Each invocation provisions its OWN container and its OWN host port and removes
# only what it created, so parallel worktrees can run the suites at the same
# time (#662). The mechanics live in scripts/test-db.sh; see its header for the
# environment knobs (TEST_DB_IMAGE, TEST_DB_PORT, TEST_DB_MAX_CONNECTIONS, ...).

# Boot an ephemeral Postgres, run the storage suite against it, tear it down.
# Extra args pass through to `cargo test` (e.g. `just test-db -- --nocapture`).
test-db *ARGS:
    ./scripts/test-db.sh run -- cargo test -p desktop-assistant-storage {{ARGS}}

# Boot the ephemeral Postgres and leave it running for iterative test runs;
# prints the settings to export. Tear down with `just test-db-down`:
#   eval "$(just test-db-up)"
test-db-up:
    @./scripts/test-db.sh start

# Remove ephemeral Postgres containers started by test-db-up. With no argument,
# sweeps every container this harness left behind (e.g. from a killed run).
test-db-down *NAMES:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "{{NAMES}}" ]; then
      ./scripts/test-db.sh prune
    else
      ./scripts/test-db.sh stop {{NAMES}}
    fi

# Rebase onto latest origin/main then run the gate (catches clean-rebase-but-broken-build)
premerge:
    git fetch origin
    git rebase origin/main
    just check

# --- k8s deploy (deploy/k8s/) ------------------------------------------------
# The deployment is a kustomize base (deploy/k8s/base) plus a per-environment
# overlay that supplies the namespace, image tag, and seed daemon.toml. This
# repo is public, so it ships only deploy/k8s/overlays/example; real overlays
# live outside the repo and point at the base by relative path (see
# deploy/k8s/README.md).
#
# The one step that can't be a static manifest is provisioning the privileged
# `adele_query` RLS role (#500): it must run the canonical
# crates/storage/bootstrap/rls_role.sql WITHOUT hand-copying the SQL into a
# manifest (which would rot). `deploy-rls-bootstrap` generates the ConfigMap
# from that file at deploy time, so the running SQL is always byte-for-byte the
# source. `check-deploy` validates the rendered manifests offline (never
# contacts the cluster).

# Which environment the deploy recipes target. Defaults are the in-repo example;
# point them at your private overlay, e.g.
#   ADELE_K8S_NAMESPACE=my-ns ADELE_K8S_OVERLAY=../my-overlays/prod just deploy-rls-bootstrap
k8s_namespace := env_var_or_default("ADELE_K8S_NAMESPACE", "adele-example")
k8s_overlay := env_var_or_default("ADELE_K8S_OVERLAY", "deploy/k8s/overlays/example")

# Provision the RLS `adele_query` role (#500) in the target namespace.
# Generates the rls-bootstrap-sql ConfigMap from the canonical rls_role.sql
# (single source of truth), then runs the postgres-gated Job that applies it as
# the app role `adele`. Idempotent: the SQL swallows a duplicate role and its
# grants self-heal, and any prior Job is cleared first (a Job's pod template is
# immutable, so a bare re-apply would error). Run after Postgres is up.
deploy-rls-bootstrap:
    kubectl create configmap rls-bootstrap-sql \
        --namespace {{k8s_namespace}} \
        --from-file=rls_role.sql=crates/storage/bootstrap/rls_role.sql \
        --dry-run=client -o yaml | kubectl apply -f -
    kubectl delete job rls-bootstrap --namespace {{k8s_namespace}} --ignore-not-found
    kubectl apply --namespace {{k8s_namespace}} -f deploy/k8s/base/rls-bootstrap.yaml
    kubectl wait --namespace {{k8s_namespace}} --for=condition=complete --timeout=120s job/rls-bootstrap

# Validate the deploy manifests without touching a live cluster: the base and
# the example overlay must both render, the rendered output is schema-validated
# client-side, a dry-run of the generated rls-bootstrap-sql ConfigMap proves the
# canonical SQL path resolves, and the #500 RLS-bootstrap shape/anti-drift
# assertions run. Safe in CI; never contacts the API server.
check-deploy:
    #!/usr/bin/env bash
    set -euo pipefail
    for target in deploy/k8s/base deploy/k8s/overlays/example; do
      echo "kustomize render + dry-run validate: $target"
      kubectl kustomize "$target" | kubectl apply --dry-run=client -f - >/dev/null
    done
    echo "dry-run validate: standalone manifests"
    kubectl apply --dry-run=client -f deploy/k8s/secret.example.yaml >/dev/null
    echo "dry-run validate: generated rls-bootstrap-sql ConfigMap"
    kubectl create configmap rls-bootstrap-sql \
        --namespace {{k8s_namespace}} \
        --from-file=rls_role.sql=crates/storage/bootstrap/rls_role.sql \
        --dry-run=client -o yaml | kubectl apply --dry-run=client -f - >/dev/null
    ./deploy/k8s/check-rls-bootstrap.sh

# Install git hooks (pre-push runs `just check`). Local config; run once per clone.
install-hooks:
    git config core.hooksPath .githooks
    @echo "pre-push hook active — bypass once with: git push --no-verify"

# Install user service file and reload systemd user manager
install-service:
    [ -f "{{service_src}}" ] || (echo "Missing service file: {{service_src}}" >&2; exit 1)
    [ -f "{{dbus_service_src}}" ] || (echo "Missing D-Bus service file: {{dbus_service_src}}" >&2; exit 1)
    mkdir -p "$(dirname '{{service_dst}}')"
    mkdir -p "{{dbus_service_dir}}"
    cp "{{service_src}}" "{{service_dst}}"
    cp "{{dbus_service_src}}" "{{dbus_service_dst}}"
    systemctl --user daemon-reload

# Install user development service file and reload systemd user manager
install-service-dev:
    [ -f "{{dev_service_src}}" ] || (echo "Missing service file: {{dev_service_src}}" >&2; exit 1)
    [ -f "{{dbus_service_dev_src}}" ] || (echo "Missing D-Bus service file: {{dbus_service_dev_src}}" >&2; exit 1)
    mkdir -p "$(dirname '{{dev_service_dst}}')"
    mkdir -p "{{dbus_service_dir}}"
    cp "{{dev_service_src}}" "{{dev_service_dst}}"
    cp "{{dbus_service_dev_src}}" "{{dbus_service_dev_dst}}"
    systemctl --user daemon-reload

# Install only D-Bus activation service files
install-dbus-activation:
    [ -f "{{dbus_service_src}}" ] || (echo "Missing D-Bus service file: {{dbus_service_src}}" >&2; exit 1)
    [ -f "{{dbus_service_dev_src}}" ] || (echo "Missing D-Bus service file: {{dbus_service_dev_src}}" >&2; exit 1)
    mkdir -p "{{dbus_service_dir}}"
    cp "{{dbus_service_src}}" "{{dbus_service_dst}}"
    cp "{{dbus_service_dev_src}}" "{{dbus_service_dev_dst}}"

# Enable + start backend service
backend-enable:
    systemctl --user enable --now {{service_name}}

# Enable + start development backend service
backend-dev-enable:
    systemctl --user enable --now {{dev_service_name}}

# Start backend service
backend-start:
    systemctl --user start {{service_name}}

# Start development backend service
backend-dev-start:
    systemctl --user start {{dev_service_name}}

# Stop backend service
backend-stop:
    systemctl --user stop {{service_name}}

# Stop development backend service
backend-dev-stop:
    systemctl --user stop {{dev_service_name}}

# Restart backend service
backend-restart:
    systemctl --user restart {{service_name}}

# Rebuild + reinstall daemon binary used by systemd service, then restart it
backend-reinstall:
    cargo install --path crates/daemon --force --locked
    systemctl --user restart {{service_name}}
    systemctl --user is-active {{service_name}}

# Restart development backend service
backend-dev-restart:
    systemctl --user restart {{dev_service_name}}

# Show backend service status
backend-status:
    systemctl --user status {{service_name}}

# Show development backend service status
backend-dev-status:
    systemctl --user status {{dev_service_name}}

# Tail backend logs
backend-logs:
    journalctl --user -u {{service_name}} -n 200 -f

# Tail development backend logs
backend-dev-logs:
    journalctl --user -u {{dev_service_name}} -n 200 -f

# Remove user service file and stop the daemon
uninstall-service:
    systemctl --user disable --now {{service_name}} || true
    rm -f "{{service_dst}}"
    rm -f "{{dbus_service_dst}}"
    systemctl --user daemon-reload

# Remove user development service file and stop the dev daemon
uninstall-service-dev:
    systemctl --user disable --now {{dev_service_name}} || true
    rm -f "{{dev_service_dst}}"
    rm -f "{{dbus_service_dev_dst}}"
    systemctl --user daemon-reload

# Remove only D-Bus activation service files
uninstall-dbus-activation:
    rm -f "{{dbus_service_dst}}"
    rm -f "{{dbus_service_dev_dst}}"

# --- D-Bus bridge (cutover step 5, #317) -------------------------------------
# The bridge re-exposes org.desktopAssistant.* by talking to the daemon over a
# peer-cred-authenticated local UDS connection (#407) — no JWT minter.

# Install the bridge user unit and reload. The org.desktopAssistant D-Bus
# activation file (which now points to the bridge) is installed by
# `install-service` / `install-dbus-activation`.
install-bridge:
    [ -f "{{bridge_service_src}}" ] || (echo "Missing service file: {{bridge_service_src}}" >&2; exit 1)
    mkdir -p "{{systemd_user_dir}}"
    cp "{{bridge_service_src}}" "{{bridge_service_dst}}"
    systemctl --user daemon-reload

# Enable + start the bridge
bridge-enable:
    systemctl --user enable --now {{bridge_service_name}}

# Start / stop / restart the bridge
bridge-start:
    systemctl --user start {{bridge_service_name}}
bridge-stop:
    systemctl --user stop {{bridge_service_name}}
bridge-restart:
    systemctl --user restart {{bridge_service_name}}

# Show bridge status
bridge-status:
    systemctl --user status {{bridge_service_name}}

# Tail bridge logs
bridge-logs:
    journalctl --user -u {{bridge_service_name}} -n 200 -f

# Rebuild + reinstall the bridge binary, then restart it
bridge-reinstall:
    cargo install --path crates/dbus-bridge --force --locked
    systemctl --user restart {{bridge_service_name}}
    systemctl --user is-active {{bridge_service_name}}

# Remove the bridge unit (+ activation) and stop it
uninstall-bridge:
    systemctl --user disable --now {{bridge_service_name}} || true
    rm -f "{{bridge_service_dst}}"
    systemctl --user daemon-reload

# Uninstall everything (services)
uninstall:
    just uninstall-service
    just uninstall-service-dev
    just uninstall-bridge

# Clean build artifacts
clean:
    cargo clean

# Validate packaging manifests and metadata
packaging-check:
    ./packaging/ci/check-packaging.sh

# Build packaged binaries and run packaging checks
packaging-ci:
    cargo build --release --package desktop-assistant-daemon
    ./packaging/ci/check-packaging.sh

# Build Debian package artifacts on host
package-deb:
    mkdir -p build/pkg/debian
    rm -rf build/pkg/debian/src
    git archive --format=tar HEAD | tar -xf - -C build/pkg/debian
    cd build/pkg/debian && cp -a packaging/debian/debian ./debian
    cd build/pkg/debian && dpkg-buildpackage -us -uc -b

# Build RPM package artifacts on host
package-rpm:
    mkdir -p build/pkg/rpm
    rm -rf build/pkg/rpm/rpmbuild
    mkdir -p build/pkg/rpm/rpmbuild/SOURCES build/pkg/rpm/rpmbuild/SPECS
    cp packaging/fedora/desktop-assistant.spec build/pkg/rpm/rpmbuild/SPECS/desktop-assistant.spec
    git archive --format=tar.gz --prefix=desktop-assistant-0.1.0/ HEAD > build/pkg/rpm/rpmbuild/SOURCES/desktop-assistant-0.1.0.tar.gz
    rpmbuild --define "_topdir {{invocation_directory()}}/build/pkg/rpm/rpmbuild" -ba build/pkg/rpm/rpmbuild/SPECS/desktop-assistant.spec

# Build Flatpak bundle on host
package-flatpak:
    mkdir -p build/pkg/flatpak
    rm -rf build/pkg/flatpak/src build/pkg/flatpak/build-dir
    mkdir -p build/pkg/flatpak/src
    git archive --format=tar HEAD | tar -xf - -C build/pkg/flatpak/src
    cp packaging/flatpak/org.desktopassistant.App.yml build/pkg/flatpak/src/packaging/flatpak/org.desktopassistant.App.yml
    flatpak-builder --force-clean build/pkg/flatpak/build-dir build/pkg/flatpak/src/packaging/flatpak/org.desktopassistant.App.yml

# Build Snap package on host
package-snap:
    snapcraft --destructive-mode --dir packaging/snap

# Build Debian package artifacts inside Docker
package-deb-docker:
    {{container_cli}} run --rm -t {{container_security_opts}} -v "{{invocation_directory()}}:/work" -w /work {{debian_builder_image}} bash -lc "apt-get update && apt-get install -y --no-install-recommends build-essential dpkg-dev debhelper pkg-config ca-certificates git curl rustc cargo libssl-dev && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable && . /root/.cargo/env && mkdir -p build/pkg/debian && rm -rf build/pkg/debian/src && git ls-files -z | tar --null -T - -cf - | tar -xf - -C build/pkg/debian && cd build/pkg/debian && cp -a packaging/debian/debian ./debian && dpkg-buildpackage -us -uc -b"

# Build RPM package artifacts inside Docker
package-rpm-docker:
    {{container_cli}} run --rm -t {{container_security_opts}} -v "{{invocation_directory()}}:/work" -w /work {{rpm_builder_image}} bash -lc "dnf -y install git tar gzip rust cargo rpm-build rpmdevtools systemd-rpm-macros openssl-devel && mkdir -p build/pkg/rpm && rm -rf build/pkg/rpm/rpmbuild && mkdir -p build/pkg/rpm/rpmbuild/SOURCES build/pkg/rpm/rpmbuild/SPECS && cp packaging/fedora/desktop-assistant.spec build/pkg/rpm/rpmbuild/SPECS/desktop-assistant.spec && git ls-files -z | tar --null -T - --transform 's,^,desktop-assistant-0.1.0/,' -czf build/pkg/rpm/rpmbuild/SOURCES/desktop-assistant-0.1.0.tar.gz && rpmbuild --define '_topdir /work/build/pkg/rpm/rpmbuild' -ba build/pkg/rpm/rpmbuild/SPECS/desktop-assistant.spec"

# Build Flatpak bundle inside Docker
package-flatpak-docker:
    {{container_cli}} run --rm -t {{container_security_opts}} -v "{{invocation_directory()}}:/work" -w /work --privileged {{flatpak_builder_image}} bash -lc "dnf -y install git tar gzip rust cargo flatpak flatpak-builder && flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo && flatpak install -y flathub org.freedesktop.Platform//24.08 org.freedesktop.Sdk//24.08 org.freedesktop.Sdk.Extension.rust-stable//24.08 && mkdir -p build/pkg/flatpak && rm -rf build/pkg/flatpak/src build/pkg/flatpak/build-dir && mkdir -p build/pkg/flatpak/src && git ls-files -z | tar --null -T - -cf - | tar -xf - -C build/pkg/flatpak/src && cp packaging/flatpak/org.desktopassistant.App.yml build/pkg/flatpak/src/packaging/flatpak/org.desktopassistant.App.yml && flatpak-builder --force-clean build/pkg/flatpak/build-dir build/pkg/flatpak/src/packaging/flatpak/org.desktopassistant.App.yml"

# Build Snap package inside Docker
package-snap-docker:
    {{container_cli}} run --rm -t {{container_security_opts}} -v "{{invocation_directory()}}:/work" -w /work/packaging/snap --privileged {{snap_builder_image}} snapcraft --destructive-mode

# Build all package formats that are reliable inside Docker containers
package-all-docker:
    # Snap is intentionally excluded here: core24/snapd runtime requirements
    # are not reliably available in Docker/Podman container builds.
    just package-deb-docker
    just package-rpm-docker
    just package-flatpak-docker

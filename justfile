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
panel_widget := "kde/plasmoid/org.desktopassistant.panelchat"
desktop_widget := "kde/plasmoid/org.desktopassistant.desktopchat"
settings_widget := "kde/plasmoid/org.desktopassistant.settings"
kcm_dir := "kde/kcm/desktop-assistant-settings"
kcm_build_dir := "build/kde-kcm"
panel_widget_id := "org.desktopassistant.panelchat"
desktop_widget_id := "org.desktopassistant.desktopchat"
settings_widget_id := "org.desktopassistant.settings"
shared_chat_module_src := "kde/shared/chat-module"
shared_chat_module_dst := env_var_or_default("XDG_DATA_HOME", env_var("HOME") + "/.local/share") + "/desktop-assistant/chat-module"
container_cli := env_var_or_default("CONTAINER_CLI", "docker")
container_security_opts := env_var_or_default("CONTAINER_SECURITY_OPTS", "--security-opt label=disable")
debian_builder_image := env_var_or_default("DEBIAN_BUILDER_IMAGE", "debian:trixie")
rpm_builder_image := env_var_or_default("RPM_BUILDER_IMAGE", "fedora:43")
flatpak_builder_image := env_var_or_default("FLATPAK_BUILDER_IMAGE", "fedora:43")
snap_builder_image := env_var_or_default("SNAP_BUILDER_IMAGE", "ubuntu:25.10")

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

# Run TUI frontend in foreground (dev)
frontend:
    cargo run -p desktop-assistant-tui

# Run TUI against development D-Bus name
dev-frontend:
    DESKTOP_ASSISTANT_DBUS_SERVICE={{dev_dbus_service}} cargo run -p desktop-assistant-tui

# Build all workspace crates
build:
    cargo build --workspace

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
    cargo install --path crates/daemon --force
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
    journalctl --user -u {{service_name}} -f

# Tail development backend logs
backend-dev-logs:
    journalctl --user -u {{dev_service_name}} -f

# Sync shared chat module to XDG data path
chat-module-sync:
    [ -d "{{shared_chat_module_src}}" ] || (echo "Missing shared module directory: {{shared_chat_module_src}}" >&2; exit 1)
    mkdir -p "$(dirname '{{shared_chat_module_dst}}')"
    rm -rf "{{shared_chat_module_dst}}"
    cp -a "{{shared_chat_module_src}}" "{{shared_chat_module_dst}}"

# Install all KDE Plasma widgets for the current user
widget-install:
    just chat-module-sync
    kpackagetool6 --type Plasma/Applet --install {{panel_widget}}
    kpackagetool6 --type Plasma/Applet --install {{desktop_widget}}
    kpackagetool6 --type Plasma/Applet --install {{settings_widget}}

# Upgrade all KDE Plasma widgets after local changes
widget-upgrade:
    just chat-module-sync
    kpackagetool6 --type Plasma/Applet --upgrade {{panel_widget}}
    kpackagetool6 --type Plasma/Applet --upgrade {{desktop_widget}}
    kpackagetool6 --type Plasma/Applet --upgrade {{settings_widget}}

# Reinstall all KDE Plasma widgets (remove + install)
widget-reinstall:
    just chat-module-sync
    kpackagetool6 --type Plasma/Applet --remove {{panel_widget_id}} || true
    kpackagetool6 --type Plasma/Applet --remove {{desktop_widget_id}} || true
    kpackagetool6 --type Plasma/Applet --remove {{settings_widget_id}} || true
    kpackagetool6 --type Plasma/Applet --install {{panel_widget}}
    kpackagetool6 --type Plasma/Applet --install {{desktop_widget}}
    kpackagetool6 --type Plasma/Applet --install {{settings_widget}}

# Hard refresh KDE widgets (reinstall + restart plasmashell)
widget-hard-refresh:
    just widget-reinstall
    kquitapp6 plasmashell || true
    nohup plasmashell --replace >/tmp/plasmashell-desktop-assistant.log 2>&1 &

# Remove all KDE Plasma widgets
widget-remove:
    kpackagetool6 --type Plasma/Applet --remove {{panel_widget_id}} || true
    kpackagetool6 --type Plasma/Applet --remove {{desktop_widget_id}} || true
    kpackagetool6 --type Plasma/Applet --remove {{settings_widget_id}} || true

# Configure and build KDE System Settings KCM
kcm-build:
    cmake -S {{kcm_dir}} -B {{kcm_build_dir}} -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build {{kcm_build_dir}}

# Install KDE System Settings KCM (user-local prefix)
kcm-install:
    cmake -S {{kcm_dir}} -B {{kcm_build_dir}} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/.local" -DKDE_INSTALL_PLUGINDIR="$HOME/.local/lib64/qt6/plugins"
    cmake --build {{kcm_build_dir}}
    cmake --install {{kcm_build_dir}}

# Install KDE System Settings KCM into system paths (requires sudo)
kcm-install-system:
    cmake -S {{kcm_dir}} -B build/kde-kcm-system -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DKDE_INSTALL_PLUGINDIR=/usr/lib64/qt6/plugins
    cmake --build build/kde-kcm-system
    sudo cmake --install build/kde-kcm-system

# Refresh KDE cache and list Desktop Assistant KCM in current shell
kcm-refresh:
    kbuildsycoca6 || true
    kcmshell6 --list | grep -i kcm_desktopassistant || (if [ -f {{kcm_build_dir}}/prefix.sh ]; then set +u; source {{kcm_build_dir}}/prefix.sh; set -u; kcmshell6 --list | grep -i kcm_desktopassistant || true; fi)

# Open Desktop Assistant KCM with local plugin environment
kcm-open:
    kcmshell6 kcm_desktopassistant || (if [ -f {{kcm_build_dir}}/prefix.sh ]; then set +u; source {{kcm_build_dir}}/prefix.sh; set -u; kcmshell6 kcm_desktopassistant; fi)

# Remove stale KCM plugin copies from legacy plugin paths
kcm-cleanup:
    rm -f "$HOME/.local/lib64/plugins/plasma/kcms/systemsettings/kcm_desktopassistant.so"
    sudo rm -f /usr/lib64/plugins/plasma/kcms/systemsettings/kcm_desktopassistant.so

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

# Uninstall everything (widgets, service, KCM)
uninstall:
    just widget-remove
    just uninstall-service
    just uninstall-service-dev
    just kcm-cleanup

# Clean build artifacts
clean:
    cargo clean
    rm -rf {{kcm_build_dir}} build/kde-kcm-system

# Validate packaging manifests and metadata
packaging-check:
    ./packaging/ci/check-packaging.sh

# Build packaged binaries and run packaging checks
packaging-ci:
    cargo build --release --package desktop-assistant-daemon --package desktop-assistant-tui
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
    {{container_cli}} run --rm -t {{container_security_opts}} -v "{{invocation_directory()}}:/work" -w /work --privileged {{snap_builder_image}} bash -lc "apt-get update && apt-get install -y --no-install-recommends snapcraft && snapcraft --destructive-mode --dir packaging/snap"

# Build all package formats inside Docker containers
package-all-docker:
    just package-deb-docker
    just package-rpm-docker
    just package-flatpak-docker
    just package-snap-docker

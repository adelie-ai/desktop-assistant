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

# Install all KDE Plasma widgets for the current user
widget-install:
    kpackagetool6 --type Plasma/Applet --install {{panel_widget}}
    kpackagetool6 --type Plasma/Applet --install {{desktop_widget}}
    kpackagetool6 --type Plasma/Applet --install {{settings_widget}}

# Upgrade all KDE Plasma widgets after local changes
widget-upgrade:
    kpackagetool6 --type Plasma/Applet --upgrade {{panel_widget}}
    kpackagetool6 --type Plasma/Applet --upgrade {{desktop_widget}}
    kpackagetool6 --type Plasma/Applet --upgrade {{settings_widget}}

# Reinstall all KDE Plasma widgets (remove + install)
widget-reinstall:
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

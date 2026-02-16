set shell := ["bash", "-euo", "pipefail", "-c"]

service_name := "desktop-assistant-daemon"
service_src := "systemd/desktop-assistant-daemon.service"
service_dst := "{{env_var_or_default('XDG_CONFIG_HOME', env_var('HOME') + '/.config')}}/systemd/user/{{service_name}}.service"
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

# Run TUI frontend in foreground (dev)
frontend:
    cargo run -p desktop-assistant-tui

# Build all workspace crates
build:
    cargo build --workspace

# Install user service file and reload systemd user manager
install-service:
    mkdir -p "$(dirname '{{service_dst}}')"
    cp "{{service_src}}" "{{service_dst}}"
    systemctl --user daemon-reload

# Enable + start backend service
backend-enable:
    systemctl --user enable --now {{service_name}}

# Start backend service
backend-start:
    systemctl --user start {{service_name}}

# Stop backend service
backend-stop:
    systemctl --user stop {{service_name}}

# Restart backend service
backend-restart:
    systemctl --user restart {{service_name}}

# Show backend service status
backend-status:
    systemctl --user status {{service_name}}

# Tail backend logs
backend-logs:
    journalctl --user -u {{service_name}} -f

# Install both KDE Plasma widgets for the current user
widget-install:
    kpackagetool6 --type Plasma/Applet --install {{panel_widget}}
    kpackagetool6 --type Plasma/Applet --install {{desktop_widget}}
    kpackagetool6 --type Plasma/Applet --install {{settings_widget}}

# Upgrade both KDE Plasma widgets after local changes
widget-upgrade:
    kpackagetool6 --type Plasma/Applet --upgrade {{panel_widget}}
    kpackagetool6 --type Plasma/Applet --upgrade {{desktop_widget}}
    kpackagetool6 --type Plasma/Applet --upgrade {{settings_widget}}

# Reinstall both KDE Plasma widgets (remove + install)
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

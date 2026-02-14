set shell := ["bash", "-euo", "pipefail", "-c"]

service_name := "desktop-assistant-daemon"
service_src := "systemd/desktop-assistant-daemon.service"
service_dst := "{{env_var_or_default('XDG_CONFIG_HOME', env_var('HOME') + '/.config')}}/systemd/user/{{service_name}}.service"
panel_widget := "kde/plasmoid/org.desktopassistant.panelchat"
desktop_widget := "kde/plasmoid/org.desktopassistant.desktopchat"
panel_widget_id := "org.desktopassistant.panelchat"
desktop_widget_id := "org.desktopassistant.desktopchat"

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

# Upgrade both KDE Plasma widgets after local changes
widget-upgrade:
    kpackagetool6 --type Plasma/Applet --upgrade {{panel_widget}}
    kpackagetool6 --type Plasma/Applet --upgrade {{desktop_widget}}

# Reinstall both KDE Plasma widgets (remove + install)
widget-reinstall:
    kpackagetool6 --type Plasma/Applet --remove {{panel_widget_id}} || true
    kpackagetool6 --type Plasma/Applet --remove {{desktop_widget_id}} || true
    kpackagetool6 --type Plasma/Applet --install {{panel_widget}}
    kpackagetool6 --type Plasma/Applet --install {{desktop_widget}}

# Hard refresh KDE widgets (reinstall + restart plasmashell)
widget-hard-refresh:
    just widget-reinstall
    kquitapp6 plasmashell || true
    nohup plasmashell --replace >/tmp/plasmashell-desktop-assistant.log 2>&1 &

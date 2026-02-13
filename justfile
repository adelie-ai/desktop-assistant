set shell := ["bash", "-euo", "pipefail", "-c"]

service_name := "desktop-assistant-daemon"
service_src := "systemd/desktop-assistant-daemon.service"
service_dst := "{{env_var_or_default('XDG_CONFIG_HOME', env_var('HOME') + '/.config')}}/systemd/user/{{service_name}}.service"

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

# Adele AI Shared Chat Module

This module contains shared chat UI and transport helper logic intended for reuse across multiple shells/frontends.

## Layout

- `ui/ChatView.qml` — reusable chat view
- `code/dbus_client.py` — reusable widget transport helper (WebSocket + D-Bus fallback)
- `images/` — shared avatar assets

## Install location

KDE wrappers load this module from:

- `$XDG_DATA_HOME/desktop-assistant/chat-module`
- fallback: `~/.local/share/desktop-assistant/chat-module`

Use `just chat-module-sync` to copy this module into the XDG data path during local development.

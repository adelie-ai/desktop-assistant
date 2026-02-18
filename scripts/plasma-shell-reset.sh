#!/usr/bin/env bash
set -euo pipefail

# Create a timestamped backup of the Plasma shell config and reset it.
BK="$HOME/.config/plasma-reset-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BK"

cp -a "$HOME/.config/plasma-org.kde.plasma.desktop-appletsrc" "$BK/" 2>/dev/null || true
cp -a "$HOME/.config/plasmashellrc" "$BK/" 2>/dev/null || true

echo "backup=$BK"

rm -f "$HOME/.config/plasma-org.kde.plasma.desktop-appletsrc" "$HOME/.config/plasmashellrc"

# Prefer systemd user restart where available, otherwise fall back to restarting plasmashell
if systemctl --user restart plasma-plasmashell.service >/dev/null 2>&1; then
    :
elif systemctl --user restart plasmashell.service >/dev/null 2>&1; then
    :
else
    kquitapp6 plasmashell >/dev/null 2>&1 || pkill -TERM -x plasmashell || true
    sleep 0.5
    pgrep -x plasmashell >/dev/null && pkill -KILL -x plasmashell || true
    sleep 0.2
    nohup plasmashell --replace >/tmp/plasmashell-desktop-assistant.log 2>&1 &
fi

sleep 1
systemctl --user --no-pager --full status plasma-plasmashell.service 2>/dev/null | sed -n '1,80p' || \
    systemctl --user --no-pager --full status plasmashell.service 2>/dev/null | sed -n '1,80p' || true

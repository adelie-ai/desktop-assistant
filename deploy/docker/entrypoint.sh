#!/bin/sh
# Role-selecting entrypoint for the desktop-assistant image.
#
# The first argument names the role; remaining args are forwarded to the chosen
# binary. With no arguments we default to the daemon.
#
#   (none)              -> desktop-assistant-daemon
#   daemon [args...]    -> desktop-assistant-daemon args...
#   bridge [args...]    -> adelie-dbus-bridge args...
#   <anything> [args]   -> exec <anything> args...   (e.g. run an MCP server
#                          or a debugging shell directly)
set -e

# No-arg case: behave as if "daemon" was requested.
role="${1:-daemon}"
# Drop the role token only when one was actually supplied.
[ "$#" -gt 0 ] && shift

case "$role" in
    daemon)
        exec desktop-assistant-daemon "$@"
        ;;
    bridge)
        exec adelie-dbus-bridge "$@"
        ;;
    *)
        # Unknown role: run it verbatim so e.g. `… fileio-mcp serve --mode stdio`
        # and `… /bin/sh` still work.
        exec "$role" "$@"
        ;;
esac

#!/usr/bin/env python3
import os
import sys
from pathlib import Path


def main() -> int:
    data_home = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    shared = Path(data_home) / "desktop-assistant" / "chat-module" / "code" / "dbus_client.py"
    local_impl = Path(__file__).resolve().with_name("dbus_client_impl.py")

    target = shared if shared.exists() else local_impl
    if not target.exists():
        print('{"error":"shared dbus helper not found"}')
        return 1

    os.execv(sys.executable, [sys.executable, str(target), *sys.argv[1:]])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

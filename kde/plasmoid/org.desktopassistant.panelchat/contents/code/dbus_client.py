#!/usr/bin/env python3
import os
import sys
from pathlib import Path


def main() -> int:
    shared = Path(__file__).resolve().parents[3] / "org.desktopassistant.desktopchat" / "contents" / "code" / "dbus_client.py"
    if not shared.exists():
        print('{"error":"shared dbus helper not found"}')
        return 1

    os.execv(sys.executable, [sys.executable, str(shared), *sys.argv[1:]])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

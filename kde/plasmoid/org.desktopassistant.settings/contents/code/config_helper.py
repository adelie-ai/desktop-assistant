#!/usr/bin/env python3
import argparse
import ast
import json
import re
import subprocess
import sys
from typing import Any

SERVICE = "org.desktopAssistant"
PATH = "/org/desktopAssistant/Settings"
IFACE = "org.desktopAssistant.Settings"


class DbusError(RuntimeError):
    pass


def _run_gdbus(method: str, *args: str) -> Any:
    command = [
        "gdbus",
        "call",
        "--session",
        "--dest",
        SERVICE,
        "--object-path",
        PATH,
        "--method",
        f"{IFACE}.{method}",
        *args,
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise DbusError("gdbus command not found; install glib2 tools") from exc
    except subprocess.CalledProcessError as exc:
        raise DbusError(exc.stderr.strip() or exc.stdout.strip() or "gdbus call failed") from exc

    output = result.stdout.strip()
    normalized = output
    normalized = re.sub(r"@[A-Za-z0-9_(){}\[\],]+\s+", "", normalized)
    normalized = re.sub(r"\b(?:u?int(?:16|32|64)|byte)\s+(-?\d+)", r"\1", normalized)
    normalized = re.sub(r"\btrue\b", "True", normalized)
    normalized = re.sub(r"\bfalse\b", "False", normalized)

    try:
        parsed = ast.literal_eval(normalized)
    except Exception as exc:  # pragma: no cover
        raise DbusError(f"unexpected gdbus output: {output}") from exc
    return parsed


def cmd_load() -> int:
    connector, model, base_url, has_api_key = _run_gdbus("GetLlmSettings")
    response = {
        "connector": connector,
        "model": model,
        "base_url": base_url,
        "has_api_key": bool(has_api_key),
    }
    print(json.dumps(response))
    return 0


def cmd_save(args: argparse.Namespace) -> int:
    _run_gdbus("SetLlmSettings", args.connector, args.model, args.base_url)

    if args.api_key and args.api_key.strip():
        _run_gdbus("SetApiKey", args.api_key)

    print(json.dumps({"ok": True}))
    return 0


def cmd_restart() -> int:
    command = ["systemctl", "--user", "restart", "desktop-assistant-daemon"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(json.dumps({"error": result.stderr.strip() or result.stdout.strip() or "restart failed"}))
        return 1
    print(json.dumps({"ok": True}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("load")

    save = sub.add_parser("save")
    save.add_argument("--connector", default="openai")
    save.add_argument("--model", default="")
    save.add_argument("--base-url", default="")
    save.add_argument("--api-key", default="")

    sub.add_parser("restart")

    args = parser.parse_args()

    try:
        if args.command == "load":
            return cmd_load()
        if args.command == "save":
            return cmd_save(args)
        if args.command == "restart":
            return cmd_restart()

        print(json.dumps({"error": "unknown command"}))
        return 1
    except DbusError as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())

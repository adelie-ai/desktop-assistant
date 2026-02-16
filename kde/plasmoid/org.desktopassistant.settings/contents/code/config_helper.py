#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_SERVICE = "org.desktopAssistant"
DEV_SERVICE = "org.desktopAssistant.Dev"
SETTINGS_PATH = Path.home() / ".config" / "desktop-assistant" / "widget_settings.json"

SERVICE = DEFAULT_SERVICE
PATH = "/org/desktopAssistant/Settings"
IFACE = "org.desktopAssistant.Settings"


class DbusError(RuntimeError):
    pass


def _load_widget_settings() -> dict[str, Any]:
    try:
        payload = json.loads(SETTINGS_PATH.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_widget_service() -> str:
    env_service = os.environ.get("DESKTOP_ASSISTANT_WIDGET_DBUS_SERVICE", "").strip()
    if env_service:
        return env_service

    settings = _load_widget_settings()
    value = str(settings.get("dbus_service", "")).strip()
    return value or DEFAULT_SERVICE


def _save_widget_service(service: str) -> str:
    target = service.strip() or DEFAULT_SERVICE
    settings = _load_widget_settings()
    settings["dbus_service"] = target
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
    return target


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
        "dbus_service": SERVICE,
        "dev_service": DEV_SERVICE,
    }
    print(json.dumps(response))
    return 0


def cmd_save(args: argparse.Namespace) -> int:
    global SERVICE

    service = _save_widget_service(args.dbus_service) if args.dbus_service else SERVICE
    SERVICE = service

    _run_gdbus("SetLlmSettings", args.connector, args.model, args.base_url)

    if args.api_key and args.api_key.strip():
        _run_gdbus("SetApiKey", args.api_key)

    print(json.dumps({"ok": True, "dbus_service": SERVICE}))
    return 0


def cmd_restart() -> int:
    if SERVICE != DEFAULT_SERVICE:
        print(json.dumps({
            "error": "restart is only supported for org.desktopAssistant; dev mode is expected to run via just dev-backend"
        }))
        return 1

    command = ["systemctl", "--user", "restart", "desktop-assistant-daemon"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(json.dumps({"error": result.stderr.strip() or result.stdout.strip() or "restart failed"}))
        return 1
    print(json.dumps({"ok": True}))
    return 0


def main() -> int:
    global SERVICE

    parser = argparse.ArgumentParser()
    parser.add_argument("--service", default="")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("load")

    save = sub.add_parser("save")
    save.add_argument("--connector", default="ollama")
    save.add_argument("--model", default="")
    save.add_argument("--base-url", default="")
    save.add_argument("--api-key", default="")
    save.add_argument("--dbus-service", default="")

    sub.add_parser("restart")

    args = parser.parse_args()
    SERVICE = args.service.strip() or _load_widget_service()

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

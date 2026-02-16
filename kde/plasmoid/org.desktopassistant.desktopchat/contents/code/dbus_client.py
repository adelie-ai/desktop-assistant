#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_SERVICE = "org.desktopAssistant"
DEV_SERVICE = "org.desktopAssistant.Dev"
SETTINGS_PATH = Path.home() / ".config" / "desktop-assistant" / "widget_settings.json"
SERVICE = DEFAULT_SERVICE
PATH = "/org/desktopAssistant/Conversations"
IFACE = "org.desktopAssistant.Conversations"
DBUS_DAEMON_DEST = "org.freedesktop.DBus"
DBUS_DAEMON_PATH = "/org/freedesktop/DBus"
DBUS_DAEMON_IFACE = "org.freedesktop.DBus"


class DbusError(RuntimeError):
    pass


def _load_widget_service() -> str:
    env_service = os.environ.get("DESKTOP_ASSISTANT_WIDGET_DBUS_SERVICE", "").strip()
    if env_service:
        return env_service

    try:
        payload = json.loads(SETTINGS_PATH.read_text())
    except Exception:
        return DEFAULT_SERVICE

    if not isinstance(payload, dict):
        return DEFAULT_SERVICE

    value = str(payload.get("dbus_service", "")).strip()
    return value or DEFAULT_SERVICE


def _parse_gdbus_output(output: str) -> Any:
    normalized = output
    normalized = re.sub(r"@[A-Za-z0-9_(){}\[\],]+\s+", "", normalized)
    normalized = re.sub(r"\b(?:u?int(?:16|32|64)|byte)\s+(-?\d+)", r"\1", normalized)
    normalized = re.sub(r"\btrue\b", "True", normalized)
    normalized = re.sub(r"\bfalse\b", "False", normalized)
    try:
        parsed = ast.literal_eval(normalized)
    except Exception as exc:
        raise DbusError(f"unexpected gdbus output: {output}") from exc
    return parsed


def _run_command(command: list[str], error_hint: str) -> Any:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise DbusError("gdbus command not found; install glib2 tools") from exc
    except subprocess.CalledProcessError as exc:
        raise DbusError(exc.stderr.strip() or exc.stdout.strip() or error_hint) from exc

    return _parse_gdbus_output(result.stdout.strip())


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
    return _run_command(command, "gdbus call failed")


def _name_has_owner(name: str) -> bool:
    command = [
        "gdbus",
        "call",
        "--session",
        "--dest",
        DBUS_DAEMON_DEST,
        "--object-path",
        DBUS_DAEMON_PATH,
        "--method",
        f"{DBUS_DAEMON_IFACE}.NameHasOwner",
        name,
    ]
    parsed = _run_command(command, "NameHasOwner call failed")
    if isinstance(parsed, tuple) and len(parsed) > 0:
        return bool(parsed[0])
    if isinstance(parsed, bool):
        return parsed
    raise DbusError(f"unexpected NameHasOwner response for {name}: {parsed}")


def create_conversation(title: str) -> str:
    response = _run_gdbus("CreateConversation", title)
    return response[0]


def send_prompt(conversation_id: str, prompt: str) -> str:
    response = _run_gdbus("SendPrompt", conversation_id, prompt)
    return response[0]


def get_conversation(conversation_id: str) -> dict[str, Any]:
    response = _run_gdbus("GetConversation", conversation_id)
    conv_id, title, messages = response
    items = []
    for role, content in messages:
        items.append({"role": role, "content": content})
    return {
        "id": conv_id,
        "title": title,
        "messages": items,
    }


def wait_for_assistant_reply(conversation_id: str, initial_count: int, timeout_sec: float, interval_sec: float) -> str:
    deadline = time.monotonic() + timeout_sec
    last_assistant = ""
    while time.monotonic() < deadline:
        conversation = get_conversation(conversation_id)
        messages = conversation["messages"]
        if len(messages) > initial_count:
            for message in reversed(messages):
                if message["role"] == "assistant":
                    return message["content"]
        for message in reversed(messages):
            if message["role"] == "assistant":
                last_assistant = message["content"]
                break
        time.sleep(interval_sec)

    return last_assistant


def list_conversations() -> list[dict[str, Any]]:
    response = _run_gdbus("ListConversations")
    rows = response[0] if isinstance(response, tuple) and len(response) == 1 else response
    return [
        {
            "id": item[0],
            "title": item[1],
            "message_count": item[2],
        }
        for item in rows
    ]


def ensure_conversation(title: str) -> str:
    conversations = list_conversations()
    for conversation in conversations:
        if conversation["title"] == title:
            return str(conversation["id"])
    return create_conversation(title)


def cmd_status() -> int:
    payload: dict[str, Any] = {
        "selected_service": SERVICE,
        "default_service": DEFAULT_SERVICE,
        "dev_service": DEV_SERVICE,
    }

    try:
        payload["production_running"] = _name_has_owner(DEFAULT_SERVICE)
        payload["dev_running"] = _name_has_owner(DEV_SERVICE)
    except DbusError as exc:
        payload["production_running"] = False
        payload["dev_running"] = False
        payload["error"] = str(exc)

    print(json.dumps(payload))
    return 0


def main() -> int:
    global SERVICE

    parser = argparse.ArgumentParser()
    parser.add_argument("--service", default="")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ensure_cmd = subparsers.add_parser("ensure")
    ensure_cmd.add_argument("--title", default="Desktop Chat")

    create_cmd = subparsers.add_parser("create")
    create_cmd.add_argument("--title", default="Desktop Chat")

    subparsers.add_parser("list")

    send_cmd = subparsers.add_parser("send")
    send_cmd.add_argument("conversation_id")
    send_cmd.add_argument("prompt")

    get_cmd = subparsers.add_parser("get")
    get_cmd.add_argument("conversation_id")

    await_cmd = subparsers.add_parser("await")
    await_cmd.add_argument("conversation_id")
    await_cmd.add_argument("--initial-count", type=int, required=True)
    await_cmd.add_argument("--timeout", type=float, default=60.0)
    await_cmd.add_argument("--interval", type=float, default=0.8)

    subparsers.add_parser("status")

    args = parser.parse_args()
    SERVICE = args.service.strip() or _load_widget_service()

    try:
        if args.command == "ensure":
            print(json.dumps({"conversation_id": ensure_conversation(args.title)}))
            return 0
        if args.command == "create":
            print(json.dumps({"conversation_id": create_conversation(args.title)}))
            return 0
        if args.command == "list":
            print(json.dumps({"conversations": list_conversations()}))
            return 0
        if args.command == "send":
            print(json.dumps({"request_id": send_prompt(args.conversation_id, args.prompt)}))
            return 0
        if args.command == "get":
            print(json.dumps(get_conversation(args.conversation_id)))
            return 0
        if args.command == "await":
            content = wait_for_assistant_reply(
                args.conversation_id,
                args.initial_count,
                args.timeout,
                args.interval,
            )
            print(json.dumps({"assistant_reply": content}))
            return 0
        if args.command == "status":
            return cmd_status()
        raise DbusError("unknown command")
    except DbusError as exc:
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())

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
DEFAULT_GDBUS_TIMEOUT_SEC = 12.0


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


def _run_command(command: list[str], error_hint: str, timeout_sec: float = DEFAULT_GDBUS_TIMEOUT_SEC) -> Any:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout_sec)
    except FileNotFoundError as exc:
        raise DbusError("gdbus command not found; install glib2 tools") from exc
    except subprocess.TimeoutExpired as exc:
        raise DbusError(f"{error_hint} (timed out after {timeout_sec:.1f}s)") from exc
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
    return _run_command(command, f"gdbus call failed: {method}")


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
    parsed = _run_command(command, "NameHasOwner call failed", timeout_sec=6.0)
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


def get_conversation(
    conversation_id: str,
    tail: int | None = None,
    after_count: int | None = None,
) -> dict[str, Any]:
    response = _run_gdbus("GetConversation", conversation_id)
    conv_id, title, messages = response
    total_messages = len(messages)
    normalized_after = max(0, int(after_count or 0))
    use_after = after_count is not None

    if use_after:
        visible_messages = messages[normalized_after:] if normalized_after < total_messages else []
        truncated = False
    else:
        # Performance guardrail for widget callers: keep historical back-loads bounded.
        # Large message batches can cause expensive QML layout/render work and freeze
        # the desktop shell, so callers should prefer a small `--tail` value.
        normalized_tail = max(0, int(tail or 0))
        truncated = normalized_tail > 0 and total_messages > normalized_tail
        visible_messages = messages[-normalized_tail:] if truncated else messages

    items = []
    for role, content in visible_messages:
        items.append({"role": role, "content": content})
    return {
        "id": conv_id,
        "title": title,
        "messages": items,
        "message_count": total_messages,
        "truncated": truncated,
        "after_count": normalized_after if use_after else None,
    }


def get_messages(
    conversation_id: str,
    tail: int | None = None,
    after_count: int | None = None,
    include_roles: list[str] | None = None,
) -> dict[str, Any]:
    """Fetch messages via GetMessages, with server-side filtering and pagination.

    - ``tail``: max visible messages to return (applied after filtering); 0 = unlimited.
    - ``after_count``: raw message index to start from; None means use tail mode.
    - ``include_roles``: allowlist of roles to return (e.g. ``["user", "assistant"]``).
      Defaults to ``["user", "assistant"]``.  Pass ``[]`` to receive all roles.

    The returned ``message_count`` is the *total* unfiltered count so callers
    can use it as the next ``after_count`` for incremental fetches.
    """
    tail_arg = str(max(0, int(tail or 0)))
    after_arg = str(max(-1, int(after_count if after_count is not None else -1)))
    roles = include_roles if include_roles is not None else ["user", "assistant"]
    # Build a D-Bus array-of-strings literal: ["role1", "role2"]
    roles_arg = "[" + ", ".join(f'"{r}"' for r in roles) + "]"
    response = _run_gdbus("GetMessages", conversation_id, tail_arg, after_arg, roles_arg)
    total_count, truncated, messages = response
    items = [{"role": role, "content": content} for role, content in messages]
    return {
        "messages": items,
        "message_count": int(total_count),
        "truncated": bool(truncated),
    }


def delete_conversation(conversation_id: str) -> None:
    _run_gdbus("DeleteConversation", conversation_id)


def clear_all_history() -> int:
    response = _run_gdbus("ClearAllHistory")
    if isinstance(response, tuple) and len(response) > 0:
        return int(response[0])
    return int(response)


def wait_for_assistant_reply(conversation_id: str, initial_count: int, timeout_sec: float, interval_sec: float) -> str:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        conversation = get_conversation(conversation_id)
        messages = conversation["messages"]
        if len(messages) > initial_count:
            new_messages = messages[initial_count:]
            latest_new_user_index = -1
            for i, message in enumerate(new_messages):
                if message.get("role") == "user":
                    latest_new_user_index = i

            if latest_new_user_index >= 0:
                for message in new_messages[latest_new_user_index + 1 :]:
                    if message.get("role") == "assistant":
                        return message.get("content", "")
            else:
                # Only accept an assistant reply without a new user inside this
                # window when the boundary is immediately after a user message.
                boundary_is_after_user = initial_count > 0 and initial_count <= len(messages) and (
                    messages[initial_count - 1].get("role") == "user"
                )
                if boundary_is_after_user:
                    for message in new_messages:
                        if message.get("role") == "assistant":
                            return message.get("content", "")
        time.sleep(interval_sec)

    return ""


def list_conversations(max_age_days: int | None = None) -> list[dict[str, Any]]:
    age = max(0, int(max_age_days or 0))
    response = _run_gdbus("ListConversations", str(age))
    rows = response[0] if isinstance(response, tuple) and len(response) == 1 else response
    return [
        {
            "id": item[0],
            "title": item[1],
            "message_count": item[2],
            "updated_at": item[3] if len(item) > 3 else "",
        }
        for item in rows
    ]


def ensure_conversation(title: str) -> str:
    conversations = list_conversations(max_age_days=0)
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

    list_cmd = subparsers.add_parser("list")
    list_cmd.add_argument("--max-age-days", type=int, default=7)

    send_cmd = subparsers.add_parser("send")
    send_cmd.add_argument("conversation_id")
    send_cmd.add_argument("prompt")

    get_cmd = subparsers.add_parser("get")
    get_cmd.add_argument("conversation_id")
    get_cmd.add_argument("--tail", type=int, default=0)
    get_cmd.add_argument("--after-count", type=int)
    get_cmd.add_argument(
        "--roles",
        default="user,assistant",
        help="Comma-separated role allowlist (default: user,assistant). "
             "Pass an empty string to return all roles.",
    )

    delete_cmd = subparsers.add_parser("delete")
    delete_cmd.add_argument("conversation_id")

    subparsers.add_parser("clear")

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
            print(json.dumps({"conversations": list_conversations(args.max_age_days)}))
            return 0
        if args.command == "send":
            print(json.dumps({"request_id": send_prompt(args.conversation_id, args.prompt)}))
            return 0
        if args.command == "get":
            include = [r.strip() for r in args.roles.split(",") if r.strip()]
            print(json.dumps(get_messages(args.conversation_id, args.tail, args.after_count, include)))
            return 0
        if args.command == "delete":
            delete_conversation(args.conversation_id)
            print(json.dumps({"deleted": True, "conversation_id": args.conversation_id}))
            return 0
        if args.command == "clear":
            deleted_count = clear_all_history()
            print(json.dumps({"deleted_count": deleted_count}))
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

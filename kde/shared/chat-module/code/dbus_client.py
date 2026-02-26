#!/usr/bin/env python3
import argparse
import ast
import base64
import hashlib
import json
import os
import re
import secrets
import socket
import ssl
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_SERVICE = "org.desktopAssistant"
DEV_SERVICE = "org.desktopAssistant.Dev"
SETTINGS_PATH = Path.home() / ".config" / "desktop-assistant" / "widget_settings.json"
DEFAULT_CONNECTION_NAME = "local"
DEFAULT_TRANSPORT = "dbus"
DEFAULT_WS_URL = "ws://127.0.0.1:11339/ws"
DEFAULT_WS_SUBJECT = "desktop-widget"
SERVICE = DEFAULT_SERVICE
PATH = "/org/desktopAssistant/Conversations"
IFACE = "org.desktopAssistant.Conversations"
SETTINGS_PATH_DBUS = "/org/desktopAssistant/Settings"
SETTINGS_IFACE = "org.desktopAssistant.Settings"
DBUS_DAEMON_DEST = "org.freedesktop.DBus"
DBUS_DAEMON_PATH = "/org/freedesktop/DBus"
DBUS_DAEMON_IFACE = "org.freedesktop.DBus"
DEFAULT_GDBUS_TIMEOUT_SEC = 12.0
DEFAULT_WS_TIMEOUT_SEC = 12.0
TRANSPORT = DEFAULT_TRANSPORT
WS_URL = DEFAULT_WS_URL
WS_SUBJECT = DEFAULT_WS_SUBJECT
WS_JWT = ""
CONNECTION_NAME = DEFAULT_CONNECTION_NAME
DEFAULT_CONFIG_CONNECTION = DEFAULT_CONNECTION_NAME


class DbusError(RuntimeError):
    pass


class WsError(RuntimeError):
    pass


def _load_widget_settings_payload() -> dict[str, Any]:
    try:
        payload = json.loads(SETTINGS_PATH.read_text())
    except Exception:
        return {}

    if not isinstance(payload, dict):
        return {}
    return payload


def _normalize_transport(value: str) -> str:
    normalized = value.strip().lower()
    return "ws" if normalized == "ws" else "dbus"


def _load_widget_connections(payload: dict[str, Any]) -> tuple[dict[str, dict[str, str]], str]:
    raw_connections = payload.get("connections")
    parsed: dict[str, dict[str, str]] = {}
    default_dbus_service = str(payload.get("dbus_service", "")).strip() or DEFAULT_SERVICE

    if isinstance(raw_connections, list):
        for item in raw_connections:
            if not isinstance(item, dict):
                continue

            name = str(item.get("name", "")).strip()
            if not name or name in parsed:
                continue

            raw_transport = str(item.get("transport", "")).strip()
            if raw_transport:
                transport = _normalize_transport(raw_transport)
            elif name == DEFAULT_CONNECTION_NAME:
                transport = "dbus"
            else:
                transport = "ws"

            dbus_service = str(item.get("dbus_service", "")).strip() or default_dbus_service
            ws_url = str(item.get("ws_url", "")).strip() or DEFAULT_WS_URL
            ws_subject = str(item.get("ws_subject", "")).strip() or DEFAULT_WS_SUBJECT

            parsed[name] = {
                "name": name,
                "transport": transport,
                "dbus_service": dbus_service,
                "ws_url": ws_url,
                "ws_subject": ws_subject,
            }

    default_connection = str(payload.get("default_connection", "")).strip()

    if not isinstance(raw_connections, list) or not raw_connections:
        legacy_transport = str(payload.get("transport", "")).strip().lower()
        legacy_ws_url = str(payload.get("ws_url", "")).strip()
        legacy_ws_subject = str(payload.get("ws_subject", "")).strip()
        use_legacy_ws = legacy_transport == "ws" or bool(legacy_ws_url)
        if use_legacy_ws:
            legacy_name = "legacy-ws"
            parsed[legacy_name] = {
                "name": legacy_name,
                "transport": "ws",
                "dbus_service": default_dbus_service,
                "ws_url": legacy_ws_url or DEFAULT_WS_URL,
                "ws_subject": legacy_ws_subject or DEFAULT_WS_SUBJECT,
            }
            if not default_connection:
                default_connection = legacy_name

    if not parsed:
        parsed[DEFAULT_CONNECTION_NAME] = {
            "name": DEFAULT_CONNECTION_NAME,
            "transport": "dbus",
            "dbus_service": default_dbus_service,
            "ws_url": DEFAULT_WS_URL,
            "ws_subject": DEFAULT_WS_SUBJECT,
        }

    if default_connection not in parsed:
        default_connection = (
            DEFAULT_CONNECTION_NAME
            if DEFAULT_CONNECTION_NAME in parsed
            else next(iter(parsed.keys()))
        )

    return parsed, default_connection


def _load_widget_connection_name(payload: dict[str, Any]) -> str:
    env_name = os.environ.get("DESKTOP_ASSISTANT_WIDGET_CONNECTION", "").strip()
    if env_name:
        return env_name
    value = str(payload.get("connection_name", "")).strip()
    if value:
        return value
    return str(payload.get("connection", "")).strip()


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


def _run_gdbus_settings(method: str, *args: str) -> Any:
    command = [
        "gdbus",
        "call",
        "--session",
        "--dest",
        SERVICE,
        "--object-path",
        SETTINGS_PATH_DBUS,
        "--method",
        f"{SETTINGS_IFACE}.{method}",
        *args,
    ]
    return _run_command(command, f"gdbus settings call failed: {method}")


def _ws_connect(ws_url: str, token: str, timeout_sec: float = DEFAULT_WS_TIMEOUT_SEC) -> socket.socket:
    parsed = urlparse(ws_url)
    scheme = parsed.scheme.lower()
    if scheme not in {"ws", "wss"}:
        raise WsError(f"unsupported websocket URL scheme: {parsed.scheme}")

    host = parsed.hostname or ""
    if not host:
        raise WsError(f"websocket URL missing host: {ws_url}")
    port = parsed.port or (443 if scheme == "wss" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path += f"?{parsed.query}"

    try:
        sock = socket.create_connection((host, port), timeout=timeout_sec)
    except OSError as exc:
        raise WsError(f"failed to connect websocket {ws_url}: {exc}") from exc

    if scheme == "wss":
        context = ssl.create_default_context()
        try:
            sock = context.wrap_socket(sock, server_hostname=host)
        except ssl.SSLError as exc:
            sock.close()
            raise WsError(f"failed TLS handshake for {ws_url}: {exc}") from exc

    key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
    host_header = f"{host}:{port}" if parsed.port else host
    request_lines = [
        f"GET {path} HTTP/1.1",
        f"Host: {host_header}",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Key: {key}",
        "Sec-WebSocket-Version: 13",
        f"Authorization: Bearer {token}",
        "",
        "",
    ]
    request_data = "\r\n".join(request_lines).encode("utf-8")

    try:
        sock.sendall(request_data)
    except OSError as exc:
        sock.close()
        raise WsError(f"failed websocket handshake write: {exc}") from exc

    response = bytearray()
    deadline = time.monotonic() + timeout_sec
    try:
        while b"\r\n\r\n" not in response:
            if time.monotonic() > deadline:
                raise WsError("websocket handshake timed out")
            chunk = sock.recv(4096)
            if not chunk:
                raise WsError("websocket handshake closed unexpectedly")
            response.extend(chunk)
            if len(response) > 65536:
                raise WsError("websocket handshake response too large")
    except Exception:
        sock.close()
        raise

    headers_blob = response.split(b"\r\n\r\n", 1)[0].decode("utf-8", errors="replace")
    lines = headers_blob.split("\r\n")
    if not lines or " 101 " not in lines[0]:
        sock.close()
        raise WsError(f"websocket upgrade failed: {lines[0] if lines else headers_blob}")

    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        key_name, value = line.split(":", 1)
        headers[key_name.strip().lower()] = value.strip()

    expected_accept = base64.b64encode(
        hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
    ).decode("ascii")
    if headers.get("sec-websocket-accept", "") != expected_accept:
        sock.close()
        raise WsError("websocket handshake validation failed (accept header mismatch)")

    sock.settimeout(timeout_sec)
    return sock


def _ws_send_frame(sock: socket.socket, opcode: int, payload: bytes) -> None:
    header = bytearray([0x81])  # FIN + text frame
    header[0] = 0x80 | (opcode & 0x0F)
    length = len(payload)
    mask_key = secrets.token_bytes(4)

    if length < 126:
        header.append(0x80 | length)
    elif length < (1 << 16):
        header.append(0x80 | 126)
        header.extend(length.to_bytes(2, "big"))
    else:
        header.append(0x80 | 127)
        header.extend(length.to_bytes(8, "big"))

    header.extend(mask_key)
    masked = bytearray(payload)
    for idx in range(len(masked)):
        masked[idx] ^= mask_key[idx % 4]

    sock.sendall(bytes(header) + bytes(masked))


def _ws_send_text(sock: socket.socket, text: str) -> None:
    _ws_send_frame(sock, 0x1, text.encode("utf-8"))


def _ws_recv_exact(sock: socket.socket, count: int) -> bytes:
    data = bytearray()
    while len(data) < count:
        chunk = sock.recv(count - len(data))
        if not chunk:
            raise WsError("websocket connection closed")
        data.extend(chunk)
    return bytes(data)


def _ws_recv_frame(sock: socket.socket) -> tuple[int, bytes]:
    first_two = _ws_recv_exact(sock, 2)
    opcode = first_two[0] & 0x0F
    masked = (first_two[1] & 0x80) != 0
    length = first_two[1] & 0x7F

    if length == 126:
        length = int.from_bytes(_ws_recv_exact(sock, 2), "big")
    elif length == 127:
        length = int.from_bytes(_ws_recv_exact(sock, 8), "big")

    mask_key = _ws_recv_exact(sock, 4) if masked else b""
    payload = bytearray(_ws_recv_exact(sock, length))
    if masked:
        for idx in range(len(payload)):
            payload[idx] ^= mask_key[idx % 4]

    return opcode, bytes(payload)


def _ws_resolve_jwt() -> str:
    token = WS_JWT.strip()
    if token:
        return token

    try:
        response = _run_gdbus_settings("GenerateWsJwt", WS_SUBJECT)
    except DbusError as exc:
        raise WsError(
            "no websocket JWT configured and failed to bootstrap via D-Bus GenerateWsJwt"
        ) from exc

    if isinstance(response, tuple) and len(response) > 0:
        token = str(response[0]).strip()
    else:
        token = str(response).strip()

    if not token:
        raise WsError("GenerateWsJwt returned empty token")
    return token


def _ws_request(command: dict[str, Any]) -> Any:
    request_id = f"widget-{secrets.token_hex(8)}"
    request = {
        "id": request_id,
        "command": command,
    }
    token = _ws_resolve_jwt()
    sock = _ws_connect(WS_URL, token)

    try:
        _ws_send_text(sock, json.dumps(request))
        while True:
            opcode, payload = _ws_recv_frame(sock)
            if opcode == 0x1:  # text
                text = payload.decode("utf-8", errors="replace")
                try:
                    frame = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if "result" in frame:
                    envelope = frame["result"]
                    if isinstance(envelope, dict) and envelope.get("id") == request_id:
                        return envelope.get("result")
                    continue
                if "error" in frame:
                    envelope = frame["error"]
                    if isinstance(envelope, dict) and envelope.get("id") == request_id:
                        raise WsError(str(envelope.get("error", "websocket request failed")))
                continue
            if opcode == 0x9:  # ping
                continue
            if opcode == 0x8:  # close
                raise WsError("websocket closed before response")
            # Ignore binary/continuation/control frames we don't use.
    except socket.timeout as exc:
        raise WsError("websocket request timed out") from exc
    finally:
        sock.close()


def _ws_expect_variant(result: Any, variant: str) -> Any:
    if isinstance(result, dict) and variant in result:
        return result[variant]
    raise WsError(f"unexpected websocket result variant, expected '{variant}': {result}")


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
    if TRANSPORT == "ws":
        response = _ws_request({"create_conversation": {"title": title}})
        payload = _ws_expect_variant(response, "conversation_id")
        if isinstance(payload, dict):
            return str(payload.get("id", ""))
        raise WsError(f"unexpected websocket create_conversation payload: {payload}")

    response = _run_gdbus("CreateConversation", title)
    return str(response[0])


def send_prompt(conversation_id: str, prompt: str) -> str:
    if TRANSPORT == "ws":
        response = _ws_request(
            {
                "send_message": {
                    "conversation_id": conversation_id,
                    "content": prompt,
                }
            }
        )
        _ws_expect_variant(response, "ack")
        return ""

    response = _run_gdbus("SendPrompt", conversation_id, prompt)
    return str(response[0])


def get_conversation(
    conversation_id: str,
    tail: int | None = None,
    after_count: int | None = None,
) -> dict[str, Any]:
    if TRANSPORT == "ws":
        response = _ws_request({"get_conversation": {"id": conversation_id}})
        response = _ws_expect_variant(response, "conversation")
        if not isinstance(response, dict):
            raise WsError(f"unexpected websocket get_conversation payload: {response}")
        conv_id = str(response.get("id", conversation_id))
        title = str(response.get("title", ""))
        messages = []
        for item in response.get("messages", []) or []:
            if not isinstance(item, dict):
                continue
            messages.append((str(item.get("role", "")), str(item.get("content", ""))))
    else:
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
    roles = include_roles if include_roles is not None else ["user", "assistant"]

    if TRANSPORT == "ws":
        response = _ws_request({"get_conversation": {"id": conversation_id}})
        response = _ws_expect_variant(response, "conversation")
        if not isinstance(response, dict):
            raise WsError(f"unexpected websocket get_conversation payload: {response}")
        raw_messages = response.get("messages", []) or []
        normalized_raw = [
            {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
            for item in raw_messages
            if isinstance(item, dict)
        ]
        total_count = len(normalized_raw)
        if after_count is not None:
            start = max(0, int(after_count))
            visible = normalized_raw[start:] if start < total_count else []
            truncated = False
        else:
            visible = normalized_raw
            truncated = False

        if roles:
            role_set = {role.strip() for role in roles if role.strip()}
            if role_set:
                visible = [item for item in visible if item.get("role") in role_set]

        normalized_tail = max(0, int(tail or 0))
        if after_count is None and normalized_tail > 0 and len(visible) > normalized_tail:
            visible = visible[-normalized_tail:]
            truncated = True

        return {
            "messages": visible,
            "message_count": int(total_count),
            "truncated": bool(truncated),
        }

    tail_arg = str(max(0, int(tail or 0)))
    # gdbus parses arguments starting with '-' as option flags, so negative
    # sentinel values must be prefixed with the explicit GVariant type.
    raw_after = int(after_count if after_count is not None else -1)
    after_arg = f"int32 {raw_after}" if raw_after < 0 else str(raw_after)
    # Build a GVariant array-of-strings literal for gdbus: ['role1', 'role2']
    # NOTE: GVariant text format uses single-quoted strings, not double-quoted.
    roles_arg = "[" + ", ".join(f"'{r}'" for r in roles) + "]"
    response = _run_gdbus("GetMessages", conversation_id, tail_arg, after_arg, roles_arg)
    total_count, truncated, messages = response
    items = [{"role": role, "content": content} for role, content in messages]
    return {
        "messages": items,
        "message_count": int(total_count),
        "truncated": bool(truncated),
    }


def delete_conversation(conversation_id: str) -> None:
    if TRANSPORT == "ws":
        response = _ws_request({"delete_conversation": {"id": conversation_id}})
        _ws_expect_variant(response, "ack")
        return
    _run_gdbus("DeleteConversation", conversation_id)


def clear_all_history() -> int:
    if TRANSPORT == "ws":
        response = _ws_request({"clear_all_history": {}})
        payload = _ws_expect_variant(response, "cleared")
        if isinstance(payload, dict):
            return int(payload.get("deleted_count", 0))
        raise WsError(f"unexpected websocket clear_all_history payload: {payload}")
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
    if TRANSPORT == "ws":
        payload = {"max_age_days": age if age > 0 else None}
        response = _ws_request({"list_conversations": payload})
        response = _ws_expect_variant(response, "conversations")
        if not isinstance(response, list):
            raise WsError(f"unexpected websocket list_conversations payload: {response}")
        conversations = []
        for item in response:
            if not isinstance(item, dict):
                continue
            conversations.append(
                {
                    "id": str(item.get("id", "")),
                    "title": str(item.get("title", "")),
                    "message_count": int(item.get("message_count", 0)),
                    "updated_at": str(item.get("updated_at", "")),
                }
            )
        return conversations

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
        "selected_connection": CONNECTION_NAME,
        "default_connection": DEFAULT_CONFIG_CONNECTION,
        "transport": TRANSPORT,
        "ws_url": WS_URL if TRANSPORT == "ws" else "",
        "selected_service": SERVICE,
        "default_service": DEFAULT_SERVICE,
        "dev_service": DEV_SERVICE,
    }

    if TRANSPORT == "ws":
        try:
            response = _ws_request({"ping": {}})
            pong = _ws_expect_variant(response, "pong")
            payload["production_running"] = bool(
                isinstance(pong, dict) and str(pong.get("value", "")) == "pong"
            )
            payload["dev_running"] = False
        except WsError as exc:
            payload["production_running"] = False
            payload["dev_running"] = False
            payload["error"] = str(exc)
    else:
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
    global CONNECTION_NAME, DEFAULT_CONFIG_CONNECTION, SERVICE, TRANSPORT, WS_JWT, WS_SUBJECT, WS_URL

    parser = argparse.ArgumentParser()
    parser.add_argument("--connection-name", default="")
    parser.add_argument("--service", default="")
    parser.add_argument("--transport", default="")
    parser.add_argument("--ws-url", default="")
    parser.add_argument("--ws-jwt", default="")
    parser.add_argument("--ws-subject", default="")
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

    subparsers.add_parser("connections")
    subparsers.add_parser("status")

    args = parser.parse_args()
    payload = _load_widget_settings_payload()
    connections, default_connection = _load_widget_connections(payload)
    DEFAULT_CONFIG_CONNECTION = default_connection

    requested_connection = args.connection_name.strip() or _load_widget_connection_name(payload) or default_connection
    if requested_connection not in connections:
        requested_connection = default_connection

    resolved = connections.get(requested_connection)
    if resolved is None:
        resolved = next(iter(connections.values()))
    CONNECTION_NAME = requested_connection
    TRANSPORT = _normalize_transport(str(resolved.get("transport", DEFAULT_TRANSPORT)))
    SERVICE = str(resolved.get("dbus_service", "")).strip() or DEFAULT_SERVICE
    WS_URL = str(resolved.get("ws_url", "")).strip() or DEFAULT_WS_URL
    WS_SUBJECT = str(resolved.get("ws_subject", "")).strip() or DEFAULT_WS_SUBJECT
    WS_JWT = str(payload.get("ws_jwt", "")).strip()

    service_override = args.service.strip() or os.environ.get("DESKTOP_ASSISTANT_WIDGET_DBUS_SERVICE", "").strip()
    if service_override:
        SERVICE = service_override

    transport_override = (args.transport.strip() or os.environ.get("DESKTOP_ASSISTANT_WIDGET_TRANSPORT", "").strip()).lower()
    if transport_override:
        if transport_override not in {"ws", "dbus"}:
            print(json.dumps({"error": f"invalid transport '{transport_override}'"}))
            return 1
        TRANSPORT = transport_override

    ws_url_override = args.ws_url.strip() or os.environ.get("DESKTOP_ASSISTANT_WIDGET_WS_URL", "").strip()
    if ws_url_override:
        WS_URL = ws_url_override

    ws_subject_override = args.ws_subject.strip() or os.environ.get("DESKTOP_ASSISTANT_WIDGET_WS_SUBJECT", "").strip()
    if ws_subject_override:
        WS_SUBJECT = ws_subject_override

    ws_jwt_override = args.ws_jwt.strip() or os.environ.get("DESKTOP_ASSISTANT_WIDGET_WS_JWT", "").strip()
    if ws_jwt_override:
        WS_JWT = ws_jwt_override

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
        if args.command == "connections":
            serialized_connections = []
            for connection in connections.values():
                serialized_connections.append(
                    {
                        "name": str(connection.get("name", "")),
                        "transport": _normalize_transport(str(connection.get("transport", DEFAULT_TRANSPORT))),
                        "dbus_service": str(connection.get("dbus_service", "")).strip(),
                        "ws_url": str(connection.get("ws_url", "")).strip(),
                        "ws_subject": str(connection.get("ws_subject", "")).strip(),
                    }
                )
            print(
                json.dumps(
                    {
                        "selected_connection": CONNECTION_NAME,
                        "default_connection": default_connection,
                        "connections": serialized_connections,
                    }
                )
            )
            return 0
        if args.command == "status":
            return cmd_status()
        raise DbusError("unknown command")
    except (DbusError, WsError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())

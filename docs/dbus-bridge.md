# The D-Bus bridge

`adelie-dbus-bridge` is a per-user binary that exposes the daemon's session-bus
surface (`org.desktopAssistant.*`) **without** linking the daemon's business
logic. Every D-Bus method call is translated into an `api::Command` and shipped
over an authenticated UDS connection to the daemon; signals coming back over
that connection are re-emitted on the bus.

It exists to retire the daemon's weaker *in-process* D-Bus surface
(`crates/dbus-interface`), which bypasses the shared dispatcher — and so misses
idempotency, ack correlation, session scope, cancellation, and live multi-client
sync. Routing D-Bus through the bridge → UDS → dispatch loop → handler makes the
D-Bus transport inherit all of that for free. See the cutover epic
[#281](https://github.com/adelie-ai/desktop-assistant/issues/281) for the full
plan; this page is the operator's view.

```
KDE plasmoid / KCM ─┐
tui/gtk --transport dbus ─┤ session bus  ┌─ adelie-dbus-bridge ─ UDS+JWT ─ daemon
                          └ org.desktopAssistant.* ┘        ▲
                                                  adelie-mint (JWT) ┘
```

## Components

| Unit | Binary | Role |
| --- | --- | --- |
| `desktop-assistant-daemon.service` | `desktop-assistant-daemon` | the daemon; owns the UDS frontend |
| `adelie-mint.service` | `adelie-mint` | mints short-lived HS256 JWTs over a local UDS (`SO_PEERCRED`-authenticated) |
| `adelie-dbus-bridge.service` | `adelie-dbus-bridge` | the bridge; mints a JWT, connects to the daemon over UDS, serves the bus |

The minter's signing key defaults to the same file the daemon uses
(`$XDG_DATA_HOME/desktop-assistant/secrets`), so minted tokens validate against
the daemon with no extra configuration on a single-user desktop. Hardening that
secret's storage is tracked in
[#365](https://github.com/adelie-ai/desktop-assistant/issues/365).

## Install / deploy

User systemd units live in `systemd/`. With binaries installed (`cargo install
--path crates/daemon`, `--path crates/jwt-minter`, `--path crates/dbus-bridge`):

```sh
just install-service     # daemon unit + org.desktopAssistant D-Bus activation
just install-bridge      # bridge + minter units + org.desktopAssistant.Bridge activation
just backend-enable      # enable + start the daemon
just bridge-enable       # enable + start the minter, then the bridge
```

Redeploy after a merged batch (see the deploy-cadence convention) with:

```sh
just bridge-reinstall    # cargo install minter + bridge, restart both
```

Homebrew: the bridge ships as its own formula in the `adelie-ai/homebrew-adelie`
tap (Linux-only, alongside `adelie-mint`). `brew install` lays down the binaries;
the systemd units above are installed separately (Homebrew does not manage user
services). The tap is currently untested end-to-end — see the homebrew-tap notes.

## Default name and the cutover

The bridge binds `org.desktopAssistant.Bridge` by default (`ADELIE_BRIDGE_NAME`)
so it can run **alongside** the daemon's in-process surface during the
transition (Option A, PR #106). The flip to `org.desktopAssistant` — and removal
of the in-process surface plus its `DESKTOP_ASSISTANT_DBUS_SERVICE` /
`DESKTOP_ASSISTANT_DBUS_REQUIRED` env knobs — is the cutover's steps 6 and 7
([#318](https://github.com/adelie-ai/desktop-assistant/issues/318) /
[#319](https://github.com/adelie-ai/desktop-assistant/issues/319)). To run a dev
bridge under a side-name for QA, set `ADELIE_BRIDGE_NAME=org.desktopAssistant.Dev`.

## Failure mode + health

Before the cutover, *daemon up ⇒ D-Bus up*. With the bridge, a missing or
crashed bridge leaves D-Bus clients (KDE) **dark even while the daemon is
healthy** — a new failure mode. The bridge mitigates the common case itself:
since [#316](https://github.com/adelie-ai/desktop-assistant/issues/316) it
reconnects to the daemon (re-minting a fresh token each time), so a daemon
restart does **not** require restarting the bridge. That's also why its unit
uses soft `Wants=`/`After=` rather than `BindsTo=`.

The authoritative "is the bridge up?" fact is the unit's own state:

```sh
systemctl --user is-active adelie-dbus-bridge.service   # active | failed | inactive
busctl --user list | grep org.desktopAssistant.Bridge   # is the name claimed?
just bridge-status                                       # bridge + minter at a glance
```

Surfacing `bridge: down` inside a client's Health/diagnostics page (the
capability model that both gates the UI and explains *why* something is off) is
intentionally **out of scope here** — it belongs to the health-report work,
which is still being scoped. This step provides the underlying fact (the unit
state above) and the unit whose liveness *is* that fact.

## Notes for operators (cutover Q4 / Q5)

- **Q4 — bridge restart cancels in-flight D-Bus turns.** Restarting the bridge
  drops its single UDS connection, which cancels any D-Bus-initiated turn still
  running. This is accepted by design: D-Bus callers fire-and-poll
  (`SendPrompt` then `GetMessages`), and a turn's lifetime is bound to the
  bridge's connection, not the one-shot caller's. Avoid restarting the bridge
  mid-turn if you can; otherwise the caller simply re-sends.
- **Q5 — `DESKTOP_ASSISTANT_DBUS_REQUIRED`.** This daemon env var (fail vs.
  continue when the session bus is unavailable) governs the *in-process* surface
  only. Nothing outside the daemon scripts against it today; its semantics move
  to the bridge unit when the in-process surface is deleted (#319). If you set it
  in a drop-in or a wrapper, plan to move that setting to the bridge unit then.

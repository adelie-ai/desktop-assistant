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
just install-service     # daemon unit + the org.desktopAssistant activation (→ the bridge)
just install-bridge      # bridge + minter units
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

## The name flip (#318)

The bridge binds `org.desktopAssistant` by default (`ADELIE_BRIDGE_NAME`) — it is
the live D-Bus surface. The daemon no longer claims the name: its legacy
in-process surface is **off** by default (`dbus_inprocess = false` /
`DESKTOP_ASSISTANT_DBUS_INPROCESS`), so its systemd unit is `Type=simple` (not
`Type=dbus` with `BusName=`), and the `org.desktopAssistant` D-Bus activation
file points at the bridge.

**Revert** (if the bridge misbehaves): set `DESKTOP_ASSISTANT_DBUS_INPROCESS=true`
on the daemon (env, or `dbus_inprocess = true` under `[transports]` in
daemon.toml), restore its unit to `Type=dbus` + `BusName=org.desktopAssistant`,
stop the bridge, and restart the daemon — the legacy in-process surface returns.
Deleting that surface for good (so the flag goes away) is step 7
([#319](https://github.com/adelie-ai/desktop-assistant/issues/319)).

For a side-by-side QA instance, run a second bridge with
`ADELIE_BRIDGE_NAME=org.desktopAssistant.Bridge` (or `.Dev`).

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
busctl --user list | grep org.desktopAssistant          # is the name claimed (by the bridge)?
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

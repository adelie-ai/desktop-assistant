# adelie-mint — DEPRECATED

`adelie-mint` (this crate, `crates/jwt-minter`) is **deprecated**.

It is a desktop-dev convenience that listens on a Unix domain socket
(`$XDG_RUNTIME_DIR/adelie/mint.sock`) and mints short-lived **HS256** JWTs for
the local OS user identified via `SO_PEERCRED`.

## Why

As part of the containerization epic (#378), the decision in **#383** is that
an **OIDC provider is the sole token issuer**. The daemon already validates
OIDC RS256 tokens over both UDS and WebSocket using one shared validator, so
the local HS256 minter is redundant. See **`docs/oidc-auth.md`** for the full
rationale and configuration.

## What replaces it

An OIDC provider. The recommended FOSS default is **Dex** (see the reference
compose stack in `deploy/compose/`, issue C-2); managed providers
(Cognito, Okta, Auth0) work with the same `[ws_auth.oidc]` config. Configure
`issuer_url`, `client_id`, and `audience` and the daemon auto-discovers the
JWKS. Details in `docs/oidc-auth.md`.

## Removal

Removal is tracked once the **client token path (#384)** lands — i.e. once the
D-Bus bridge and the shared `Connector` obtain OIDC tokens (with auto-refresh)
instead of minting HS256 tokens locally. Until then this crate remains so the
current minter-coupled clients keep working.

See: #378, #383, #384, and `docs/oidc-auth.md`.

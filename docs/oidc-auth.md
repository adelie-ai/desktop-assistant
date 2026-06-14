# OIDC authentication

> Part of the containerization epic (#378). This document records the
> decision (#383) to make an **OIDC provider the sole token issuer** and to
> retire the local HS256 minter `adelie-mint` (`crates/jwt-minter`). The
> remaining client-side work to obtain OIDC tokens instead of minting them is
> tracked in #384 (deferred — see [Outstanding work](#outstanding-work)).

## Decision

The desktop-assistant daemon accepts a bearer JWT on every transport. Going
forward there is **one** way to obtain that token: an OpenID Connect provider
issues an RS256 token, and the daemon validates it against the provider's
JWKS. The local HS256 minter (`adelie-mint`) is **deprecated** and will be
removed once the client token path (#384) lands.

The default FOSS provider for self-hosted / local-first deployments is
**[Dex](https://dexidp.io/)**; managed providers (Cognito, Okta, Auth0) work
with the same config — only the URLs change. See
[Provider parity](#provider-parity).

## Why this is nearly free on the daemon

The daemon already validates OIDC tokens, and it validates **the same bearer
token over both UDS and WebSocket**. An OIDC RS256 token therefore
authenticates over the local UDS socket today with **no daemon code change** —
only configuration.

The relevant facts, from source:

- **One validator, both transports.** The UDS server is handed a
  `WsAsUdsAuth` that wraps the *identical* `ws_auth` validator instance used by
  the WebSocket server — `crates/daemon/src/main.rs:1935`
  (`let ws_auth_for_uds = Arc::clone(&ws_auth);`) feeding
  `WsAsUdsAuth::new(...)` at `crates/daemon/src/main.rs:1945`. The adapter
  itself is `crates/daemon/src/transports.rs:103-126`; its doc comment notes
  it "reuses the WS bearer-token validator for UDS connections so both
  transports honor the same JWT policy (local HS256 + OIDC RS256 fallback)".

- **Local HS256 first, then OIDC RS256.** When OIDC is configured, `ws_auth`
  is an `OidcAwareAuth` (`crates/daemon/src/transports.rs:47-85`): it tries the
  local HS256 mint first and falls back to OIDC RS256 validation
  (`validate_bearer_token`, lines 54-62). Identity extraction follows
  acceptance — the validator that accepted the token is the one that yields its
  `sub` (lines 64-85). So once `adelie-mint` is gone, the HS256 branch simply
  never matches and the OIDC branch carries every request.

- **No signing key required at rest.** The daemon never needs an HS256 signing
  key present to validate OIDC tokens — the WS HS256 key is read lazily and is
  only used by the local-mint path
  (`crates/daemon/src/config/jwt.rs`). An OIDC-only deployment carries no local
  symmetric secret at all.

- **Wiring is config-gated.** `ws_auth` becomes `OidcAwareAuth` only when
  `[ws_auth].methods` contains `"oidc"` *and* `[ws_auth.oidc]` is present;
  otherwise it stays local-only (`crates/daemon/src/main.rs:1751-1780`). The
  default `methods` is `["password"]`
  (`crates/daemon/src/config/mod.rs:242`), so OIDC is strictly opt-in.

## Daemon configuration

OIDC config lives under `[ws_auth]` in `daemon.toml`:

```toml
[ws_auth]
# Must include "oidc" to activate the OIDC validator. (Default: ["password"].)
methods = ["oidc"]

[ws_auth.oidc]
# REQUIRED. The provider's issuer ("iss"). Used both as the OIDC discovery
# base (the daemon fetches `<issuer_url>/.well-known/openid-configuration`)
# and pinned as the accepted `iss` on every token.
issuer_url = "https://idp.example.com"

# REQUIRED by the config schema (used for auth discovery / interactive login;
# the validator itself does not need them to verify a presented token).
authorization_endpoint = "https://idp.example.com/auth"
token_endpoint         = "https://idp.example.com/token"
client_id              = "desktop-assistant"

# OPTIONAL. OAuth scopes advertised to clients.
# Default: "openid profile email".
scopes = "openid profile email"

# OPTIONAL. Explicit JWKS URL. When empty (the default), the daemon
# auto-discovers it from the issuer's `.well-known/openid-configuration`.
jwks_uri = ""

# REQUIRED for validation. The expected `aud` of tokens minted for THIS
# service. Must be non-empty (see security rules below).
audience = "desktop-assistant"
```

Field reference (from `crates/daemon/src/config/mod.rs:292-302` and
`crates/daemon/src/config/oidc.rs`):

| Field | Required | Default | Notes |
|-------|----------|---------|-------|
| `issuer_url` | yes | — | Pinned `iss`; discovery base. https-or-loopback. |
| `authorization_endpoint` | yes (schema) | — | Surfaced for client login/discovery. |
| `token_endpoint` | yes (schema) | — | Surfaced for client login/discovery. |
| `client_id` | yes (schema) | — | OAuth client id. |
| `scopes` | no | `openid profile email` | Advertised scopes. |
| `jwks_uri` | no | `""` → auto-discover | https-or-loopback when set. |
| `audience` | yes (validator) | `""` → **rejected** | Must be non-empty. |

### Security rules enforced by the validator

These are enforced in `OidcValidator::from_config`
(`crates/daemon/src/config/oidc.rs:228-293`) and are not optional:

- **Audience is mandatory and non-empty.** An empty `audience` disables the
  `aud` check, which would accept *any* token the issuer ever minted for any
  other relying party. `from_config` refuses to build a validator when
  `audience` is empty/whitespace (`oidc.rs:235-241`).
- **RS256 only; HMAC disallowed.** The validator is built with
  `Algorithm::RS256` and JWKS entries are filtered to `kty = RSA`,
  `use ∈ {sig, absent}`, `alg ∈ {RS256, absent}`
  (`oidc.rs:184-225, 277`). HMAC algorithms are explicitly *not* accepted,
  defending against the JWKS-substitution-via-`alg=HS256` attack class
  (`oidc.rs:20-22`).
- **HTTPS or explicit loopback only.** `issuer_url`, an explicit `jwks_uri`,
  and the discovered `jwks_uri` must all be `https://` *or* an explicit
  loopback `http://` (`localhost`/`127.0.0.1`/`[::1]`) —
  `require_https_or_loopback` (`oidc.rs:120-141, 247-264`). Plaintext to a
  non-loopback host is rejected so an attacker can't substitute a JWKS.
- **Issuer + expiry pinned.** `iss` is pinned to `issuer_url` and `exp` is
  always validated (`oidc.rs:278-279`).
- **JWKS refresh, not frozen.** An unknown `kid` triggers a rate-limited
  refetch from the *same* pinned `jwks_uri` (min 60s between attempts), so IdP
  key rotation does not lock users out until a restart, and a flood of garbage
  `kid`s can't hammer the IdP (`oidc.rs:27-33, 53, 308-334`). Refresh can only
  *add* keys from the pinned URI; it never relaxes the issuer/audience/algorithm
  rules.
- **Size-capped fetches.** Discovery and JWKS responses are capped at 1 MiB,
  enforced during the streamed read so an over-cap body is never fully buffered
  (`oidc.rs:115-179`).

## Default provider: Dex

[Dex](https://dexidp.io/) is the recommended FOSS identity provider for
self-hosted and local-first deployments. The containerization reference
compose stack (issue C-2) ships a runnable Dex service and a pre-wired daemon:

- `deploy/compose/podman-compose.yml` — the compose stack.
- `deploy/compose/dex-config.yaml` — the Dex configuration.
- `deploy/compose/daemon.toml` — the daemon config with `[ws_auth.oidc]`
  pointed at the in-stack Dex issuer.

See `deploy/compose/README.md` for how to bring the stack up. (That YAML is the
single source of truth for the reference deployment — it is not duplicated
here.) For a local stack the Dex issuer is reachable over loopback, which the
validator's https-or-loopback rule permits for development.

## Provider parity

Any compliant OIDC provider works. In every case you set three fields and the
daemon **auto-discovers the JWKS** from
`<issuer_url>/.well-known/openid-configuration` (leave `jwks_uri` empty):

```toml
[ws_auth]
methods = ["oidc"]

[ws_auth.oidc]
issuer_url = "<provider issuer>"
client_id  = "<your app/client id>"
audience   = "<this service's expected aud>"
# authorization_endpoint / token_endpoint as advertised by the provider
```

| Provider | `issuer_url` shape | Notes |
|----------|--------------------|-------|
| **Dex** (default) | `https://<dex-host>/dex` (or `http://localhost:5556/dex` in dev) | Issuer is the configured `issuer` in `dex-config.yaml`. |
| **AWS Cognito** | `https://cognito-idp.<region>.amazonaws.com/<userPoolId>` | `audience` = the App Client ID (Cognito puts the client id in `aud` for ID tokens; for access tokens validate the `client_id`/resource server scope). |
| **Okta** | `https://<your-org>.okta.com/oauth2/<authServerId>` (default auth server: `https://<your-org>.okta.com/oauth2/default`) | `audience` = the API/authorization-server audience you configured (e.g. `api://default`). |
| **Auth0** | `https://<your-tenant>.<region>.auth0.com/` (note trailing slash) | `audience` = the Auth0 **API Identifier**; request it with the `audience` param at login so Auth0 issues an RS256 access token with that `aud`. |

All four expose standard `.well-known/openid-configuration` discovery and an
RS256 JWKS, so no `jwks_uri` override is needed. Set `audience` to exactly the
`aud` value your provider stamps on tokens minted for this service —
mismatched audiences are rejected (see security rules).

## Outstanding work

Server-side validation is done. The remaining piece is **client-side token
acquisition**, tracked in **#384 (deferred)**:

- The D-Bus bridge (`crates/dbus-bridge`) currently *requires* `adelie-mint`:
  it connects via the shared `Connector` with a `minter_socket`, minting a
  local HS256 JWT (`crates/dbus-bridge/src/main.rs:87-118`). Under
  OIDC-everywhere it must instead obtain an OIDC token
  (client-credentials / refresh-token / resource-owner-password grant) and
  refresh it automatically, then hand that bearer to the `Connector` (which
  already reconnects and replays on expiry).
- The shared `Connector` (`crates/client-common`) needs the same
  token-acquisition + auto-refresh path so gtk/tui inherit it for free.
- The bridge is being reworked elsewhere; #384 should land on top of that
  rework rather than against the current minter-coupled bridge.

Once #384 ships and clients no longer mint, `adelie-mint`
(`crates/jwt-minter`) and `systemd/adelie-mint.service` can be removed.

## Cross-references

- #378 — containerization epic.
- #383 — this decision (OIDC sole issuer; deprecate `adelie-mint`).
- #384 — client OIDC token acquisition (deferred).
- C-2 — reference compose stack with Dex (`deploy/compose/`).
- `crates/jwt-minter/README.md` — deprecation notice for the minter.

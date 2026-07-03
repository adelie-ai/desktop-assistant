# Remote MCP servers

Adele's daemon can use **remote MCP servers** reached over the network
(streamable-HTTP transport), not just local stdio subprocesses. A remote server
is selected by giving its entry an `[servers.http]` table instead of a
`command`. Remote servers may be unauthenticated, use a static bearer token, or
authenticate with **OAuth 2.0** (the daemon mints and refreshes short-lived
access tokens from a stored refresh token).

This folder collects provider-specific how-to guides for wiring Adele up to
particular remote MCP services.

- [Google Workspace (Gmail / Calendar / Drive)](GoogleWorkspace-setup.md)

## How remote MCP works in Adele

- **Transport.** An `[servers.http]` table with a `url` selects the
  streamable-HTTP transport for that server. No `command` is spawned.
- **Auth.** Inside `[servers.http]`:
  - nothing → unauthenticated;
  - `auth_bearer_secret = "<id>"` → send a static `Authorization: Bearer`
    token (value looked up in `secrets.toml` by id);
  - `[servers.http.oauth]` → full OAuth 2.0. The daemon exchanges a stored
    refresh token for access tokens, refreshes them on demand and on `401`.
- **Secrets live apart from config.** `mcp_servers.toml` only ever contains
  secret **references** (ids). The secret **values** (bearer tokens, OAuth
  client secrets, refresh tokens) live in `secrets.toml`, which is written
  `0600`. Never put a secret value in `mcp_servers.toml`.
- **Config locations** (XDG-aware; honor `XDG_CONFIG_HOME`):
  - `~/.config/desktop-assistant/mcp_servers.toml`
  - `~/.config/desktop-assistant/secrets.toml`

## Two ways to configure a remote server

1. **KCM (recommended):** System Settings → Desktop Assistant → **MCP Servers**
   tab → **Add** → Transport **Remote (HTTP)**. The editor writes the config and
   secrets for you and gives each server an honest state
   (`running` / `stopped` / `needs_auth` / `auth_expired` / `error`) plus a
   **Sign in** button for OAuth servers.
2. **By hand:** edit `mcp_servers.toml` / `secrets.toml` directly (examples in
   the provider guides), then restart the daemon or reload its config.

## OAuth sign-in

An OAuth server needs a one-time interactive login to obtain the initial refresh
token. Either:

- click **Sign in** on the server's row in the KCM MCP Servers tab, or
- run the daemon's login command yourself:

  ```
  desktop-assistant --mcp-oauth-login <server-name>
  ```

Both open your browser for consent, then write the refresh token into
`secrets.toml` under the server's `refresh_token_ref`. The server must already
exist in `mcp_servers.toml` (with its `[servers.http.oauth]` block, an
`authorize_url`, and at least one scope) before you sign in.

The login uses the **installed-app loopback flow**: it binds an ephemeral
`http://127.0.0.1:<port>` listener with PKCE, so it needs no publicly reachable
redirect and works headlessly on any desktop. That has implications for the kind
of OAuth client you register — see the provider guides.

# Google Workspace (Gmail / Calendar / Drive) via remote MCP

Google hosts **first-party MCP servers** for Workspace products. Adele connects
to them directly with its remote-HTTP MCP client, authenticating with an OAuth
client **you** own in **your own** Google Cloud project. Nothing here is
Adele-specific plumbing you have to build — it is configuration plus a Google
Cloud setup.

Read [README.md](README.md) first for how remote MCP and secrets work in Adele.

## The endpoints

Each product is a separate MCP server, so each becomes one entry in the MCP
Servers tab / `mcp_servers.toml`.

| Product  | MCP endpoint URL                          |
| -------- | ----------------------------------------- |
| Calendar | `https://calendarmcp.googleapis.com/mcp/v1` |
| Gmail    | `https://gmailmcp.googleapis.com/mcp/v1`    |
| Drive    | `https://drivemcp.googleapis.com/mcp/v1`    |

Fixed OAuth endpoints (same for every product):

- Authorization URL: `https://accounts.google.com/o/oauth2/v2/auth`
- Token URL: `https://oauth2.googleapis.com/token`

## Why your own OAuth client (and not the desktop environment's accounts)

The consent for these MCP servers is a standard Google OAuth authorization. Using
**your own** OAuth client in **your own** Google Cloud project — rather than a
shared client baked into a desktop environment's "online accounts" service —
buys three things:

- **Scope control:** you request exactly the scopes the MCP servers need.
- **Verification immunity:** on a Google **Workspace** domain you can set the
  consent screen to **Internal**, which needs no Google app verification and no
  CASA security assessment, even for sensitive Gmail scopes. A shared,
  externally-published client would need Google to verify it.
- **Portability:** it works headlessly and on any desktop, with no dependency on
  a session account service.

## Part A — Google Cloud Console (one time)

You need a Google Cloud project. On a Workspace domain, an admin may need to
allow the APIs.

1. **Create or select a project** in the [Google Cloud Console](https://console.cloud.google.com/).

2. **Enable the product APIs** (APIs & Services → Library):
   - Gmail API
   - Google Calendar API
   - Google Drive API
   - People API

3. **Enable the MCP services** for the products you want (same Library search):
   - Gmail MCP API
   - Google Calendar MCP API
   - Google Drive MCP API
   - People MCP API

4. **Configure the OAuth consent screen** (APIs & Services → OAuth consent
   screen):
   - **User type: Internal** if this is a Workspace domain — recommended; no
     verification, no CASA review. If Internal is unavailable (personal Google
     account), choose **External** and add yourself as a **Test user**.
   - Fill in an app name and support/contact email.

5. **Create the OAuth client** (APIs & Services → Credentials → Create
   credentials → OAuth client ID):
   - **Application type: Desktop app** — see
     [OAuth client type](#oauth-client-type-desktop-vs-web) for why.
   - Copy the **Client ID** and **Client secret**. You'll give both to Adele.

## Part B — Configure the server in Adele

Do this once per product. The example below is Calendar; Gmail and Drive are
identical except the `url` and `scopes` (see [Scopes](#scopes)).

### Option 1 — KCM (recommended)

System Settings → Desktop Assistant → **MCP Servers** → **Add**:

- **Transport:** Remote (HTTP)
- **URL:** the product endpoint (e.g. `https://calendarmcp.googleapis.com/mcp/v1`)
- **Authentication:** OAuth
- **Client ID:** from Part A
- **Client secret:** paste the value — the editor stores it in `secrets.toml`
  and references it by id; it is never written into `mcp_servers.toml`
- **Authorization URL / Token URL:** the fixed URLs above
- **Scopes:** the product's scopes
- **Account:** the Google account email you'll authorize as

Save. The row appears as **Sign in required**.

### Option 2 — by hand

Add to `~/.config/desktop-assistant/mcp_servers.toml`:

```toml
[[servers]]
name = "google-calendar"      # used as the --mcp-oauth-login argument
namespace = "gcal"            # optional; tools exposed as gcal__<tool>
enabled = true

[servers.http]
url = "https://calendarmcp.googleapis.com/mcp/v1"

[servers.http.oauth]
client_id = "YOUR_CLIENT_ID.apps.googleusercontent.com"
authorize_url = "https://accounts.google.com/o/oauth2/v2/auth"
token_url = "https://oauth2.googleapis.com/token"
# Secret *references* — the values live in secrets.toml, not here.
client_secret_ref = "google_client_secret"
refresh_token_ref = "google_calendar_refresh"   # written by sign-in
scopes = [
  "https://www.googleapis.com/auth/calendar.calendarlist.readonly",
  "https://www.googleapis.com/auth/calendar.events.freebusy",
  "https://www.googleapis.com/auth/calendar.events.readonly",
]
account = "you@example.com"    # token-store key; use the account you authorize
```

Put the **client secret value** in `~/.config/desktop-assistant/secrets.toml`
(create it `0600` if it doesn't exist). The refresh token is added
automatically by sign-in — you don't write it:

```toml
[secrets]
google_client_secret = "PASTE_THE_CLIENT_SECRET_HERE"
```

## Part C — Sign in

The server now exists but has no refresh token, so its state is
**needs_auth**. Authorize it once:

- **KCM:** click **Sign in** on the server's row, or
- **CLI:**

  ```
  desktop-assistant --mcp-oauth-login google-calendar
  ```

Your browser opens for consent. On success the refresh token is written to
`secrets.toml` under `refresh_token_ref`, and on the next reload the server
flips toward **running**. The daemon refreshes access tokens from that refresh
token from then on — no repeat logins unless the token is revoked.

## Scopes

Use the scopes Google's MCP servers expect for each product:

| Product  | Scopes |
| -------- | ------ |
| Calendar | `https://www.googleapis.com/auth/calendar.calendarlist.readonly`<br>`https://www.googleapis.com/auth/calendar.events.freebusy`<br>`https://www.googleapis.com/auth/calendar.events.readonly` |
| Gmail    | `https://www.googleapis.com/auth/gmail.readonly`<br>`https://www.googleapis.com/auth/gmail.compose` |
| Drive    | `https://www.googleapis.com/auth/drive.readonly`<br>`https://www.googleapis.com/auth/drive.file` |

These are read-leaning defaults from Google's setup docs; adjust to your needs.
Broadening scopes later requires signing in again so the new grant is captured.

## OAuth client type: Desktop vs Web

Adele's sign-in uses the **installed-app loopback flow**: it binds an ephemeral
`http://127.0.0.1:<random-port>` listener with PKCE and no client-side redirect
registration. A **Desktop app** OAuth client natively allows loopback on any
port, so it matches this flow.

Google's own MCP documentation shows creating a **Web application** client with a
fixed **Authorized redirect URI**. That suits MCP clients that own a stable
hosted redirect, but a Web client requires an *exact* redirect match, which the
loopback flow's random port does not satisfy. So for Adele, create a **Desktop
app** client. The MCP endpoints validate the access token's *scopes and
project*, not the client type used to obtain it.

## One login for several products (optional)

Each product is its own server entry, and the simplest approach is one sign-in
per server. If you'd rather authorize once for several products, point every
server's `refresh_token_ref` at the same secret id and give them the same
`account`, then sign in on **one** server whose `scopes` list is the **union** of
all the products' scopes. The daemon's token store is keyed by `account`, so the
servers share the cached token.

## Verifying

- The MCP Servers tab shows the server **running** with a tool count.
- Ask Adele to do something that needs it (e.g. "what's on my calendar
  tomorrow?"). The tools appear under the server's `namespace` prefix.

## Troubleshooting

- **Stuck on `needs_auth`.** No refresh token yet — sign in. Confirm the server
  name you passed to `--mcp-oauth-login` matches the `name` in
  `mcp_servers.toml`.
- **`auth_expired`.** The refresh token was revoked or expired (e.g. consent
  removed, or password change on some account types). Sign in again.
- **`error` with an auth message on connect.** Usually a scope/consent mismatch
  or the MCP service not enabled. Re-check Part A step 3, and that the granted
  scopes cover what the endpoint needs.
- **`redirect_uri_mismatch` in the browser.** You created a **Web** client
  instead of a **Desktop** client — the loopback redirect won't match. Recreate
  it as a Desktop app client.
- **`access_denied` / app not verified.** The consent screen is **External** and
  either you aren't listed as a **Test user** or the sensitive scopes need
  verification. Prefer **Internal** on a Workspace domain, or add yourself as a
  test user.

## Security notes

- Secret **values** (client secret, refresh token) only ever live in
  `secrets.toml` (`0600`). `mcp_servers.toml` holds only references.
- Prefer **Internal** consent so no externally-published client is exposed.
- Request the narrowest scopes that do the job; widen only when needed.

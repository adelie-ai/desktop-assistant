# Security Audit — desktop-assistant

**Date:** 2026-03-31
**Scope:** All crates in the `desktop-assistant/` Cargo workspace

---

## Critical / High Severity

### 1. No WebSocket Origin Validation (DOWNGRADED — LOW)

**File:** `crates/ws-interface/src/lib.rs:135-149`

**Status:** Accepted risk (2026-03-31)
**Rationale:** The intended clients are native applications (gtk-client, TUI), not browsers. Native clients do not send `Origin` headers, so CSWSH is not a practical attack vector for this project. Browser-based access is not a supported use case.

The WebSocket handler validates Bearer tokens but does not validate the `Origin` header. In a browser context this would enable Cross-Site WebSocket Hijacking (CSWSH), but since all clients are native, this is low risk.

```rust
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<WsServerState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let Some(token) = extract_bearer_token(&headers) else {
        return (StatusCode::UNAUTHORIZED, "missing bearer token").into_response();
    };
    // No origin validation
    if !state.auth_validator.validate_bearer_token(&token).await {
        return (StatusCode::UNAUTHORIZED, "invalid bearer token").into_response();
    }
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}
```

**Recommendation (defense-in-depth):** Optionally reject connections that *do* carry an `Origin` header, since no legitimate client should send one. Revisit if browser-based clients are ever added.

---

### 2. MCP Server Command Execution Without Path Validation (HIGH)

**File:** `crates/mcp-client/src/lib.rs:55-62`

MCP server commands are spawned directly from user-writable configuration without validation of the command path or arguments.

```rust
pub async fn connect(command: &str, args: &[String]) -> Result<Self, McpError> {
    let mut child = Command::new(command)
        .args(args)
        .stdin(std::process::Stdio::piped())
        .spawn()
        .map_err(McpError::SpawnFailed)?;
```

**Recommendation:**
- Require absolute paths for commands
- Restrict to known directories
- Reject arguments containing shell metacharacters
- Consider sandboxing (e.g. Landlock on Linux)

---

### 3. Raw SQL String Formatting (HIGH)

**File:** `crates/storage/src/database.rs:20-127`

The `execute_database_query` function accepts arbitrary SQL and appends a LIMIT clause via `format!`:

```rust
format!("{sql} LIMIT {limit}")
```

While `sqlx` parameterized queries are used elsewhere, this string formatting bypasses that protection. There are also no query length limits or rate limiting.

**Recommendation:**
- Use parameterized binding for the LIMIT value
- Add query length and complexity limits
- Restrict to read-only operations where appropriate

---

### 4. PAM Unsafe FFI — Potential Dangling Pointer (HIGH)

**File:** `crates/daemon/src/config.rs:1567-1757`

The PAM authentication implementation uses raw FFI with `libc`. The conversation callback stores a raw `*const c_char` to the password in `ConvData` (line 1623). If the backing `CString` is dropped early, this becomes a dangling pointer. Additionally, `libc::strdup()` (line 1680) duplicates the password with manual memory management.

```rust
let conversation = PamConv {
    conv: Some(conversation),
    appdata_ptr: (&conv_data as *const ConvData).cast_mut().cast(),
};
```

**Recommendation:** Consider using the `pam-client` crate, or pin/box the `ConvData` to guarantee its lifetime across the FFI boundary.

---

## Medium Severity

### 5. JWT Signing Key TOCTOU (MEDIUM)

**File:** `crates/daemon/src/config.rs:1036-1069`

The JWT signing key file is written with default permissions, then restricted to `0o600` in a separate call. An attacker could read the key in the window between the two operations.

```rust
std::fs::write(&path, value)?;           // default permissions
std::fs::set_permissions(&path, ...)?;   // then restricted
```

**Recommendation:** Use `OpenOptions::new().write(true).create(true).mode(0o600).open(&path)` to set permissions atomically.

---

### 6. OIDC Discovery — No Timeout or Size Limits (MEDIUM-HIGH)

**File:** `crates/daemon/src/config.rs:1405-1423`

OIDC discovery and JWKS fetches use `reqwest::Client::new()` without explicit timeouts, redirect limits, or response size caps. This exposes the daemon to slow-loris attacks, redirect loops, or memory exhaustion from oversized responses.

**Recommendation:**
- Set explicit connect and read timeouts
- Limit redirect count
- Cap response body size
- Cache JWKS with a TTL

---

### 7. Internal Error Details Leaked to Clients (DOWNGRADED — LOW)

**File:** `crates/ws-interface/src/lib.rs:184, 309-332, 348-371`

**Status:** Accepted risk (2026-03-31)
**Rationale:** All clients are trusted native applications (gtk-client, TUI) running on the same machine or controlled infrastructure — not untrusted browsers. Detailed errors are useful for client-side diagnostics in this context.

Error details from internal handlers are forwarded directly to WebSocket clients and HTTP responses:

```rust
Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, error).into_response(),
```

```rust
out_tx.send(WsFrame::Error { id: req.id, error: e.to_string() })
```

**Recommendation:** Revisit if the WebSocket API is ever exposed to untrusted or browser-based clients.

---

### 8. Hardcoded PAM Service Name (DOWNGRADED — LOW)

**File:** `crates/daemon/src/config.rs:1715`

**Status:** Accepted risk (2026-03-31)
**Rationale:** The `"login"` service works correctly for the current authentication needs. Using a dedicated PAM service is a hygiene improvement (avoids inheriting rules meant for terminal logins like utmp/MOTD) but is not a security vulnerability. Low urgency — revisit if PAM behavior causes unexpected side effects or if the daemon needs different auth policies than terminal login.

The PAM service is hardcoded to `"login"`, which may include modules not intended for a background daemon.

**Recommendation:** Create a dedicated `/etc/pam.d/desktop-assistant` service file when packaging/distribution is addressed.

---

## Low Severity

### 9. No Rate Limiting on WebSocket Messages (LOW)

**File:** `crates/ws-interface/src/lib.rs`

The outbound buffer is capped at 64 messages, but there is no rate limiting on inbound messages. An authenticated client could spam the server. Malformed JSON messages are silently dropped with no feedback or throttling.

**Recommendation:** Implement per-connection rate limiting and send error frames for malformed input.

---

### 10. Server Auth Config via Environment Variables (DOWNGRADED — LOW)

**File:** `crates/daemon/src/main.rs:174-179`

**Status:** Accepted risk (2026-03-31)
**Rationale:** These env vars (`DESKTOP_ASSISTANT_WS_LOGIN_USERNAME`, `DESKTOP_ASSISTANT_WS_LOGIN_PASSWORD`) are *server-side* configuration controlling what credentials the `/login` endpoint accepts — they are not client credentials being passed around. When the daemon is managed by systemd, env vars can be loaded from restricted files via `EnvironmentFile=` or `LoadCredential=`. The *client-side* credential CLI args have been removed from gtk-client and TUI (see gtk-client SECURITY_AUDIT.md #4).

**Recommendation:** Consider migrating to a secrets file or systemd credentials for the static password mode when packaging/distribution is addressed.

---

### 11. FNV-1a Hash for Secret Fingerprinting (LOW)

**File:** `crates/daemon/src/config.rs:1142-1154`

FNV-1a is used to generate log fingerprints for secrets. FNV is not collision-resistant, so two different secrets could produce the same fingerprint.

**Recommendation:** Use SHA-256 or BLAKE3 truncated to 16 bytes.

---

### 12. No Explicit WebSocket Message Size Limit (LOW)

**File:** `crates/ws-interface/src/lib.rs`

WebSocket frame size relies on Axum defaults rather than an explicit configuration.

**Recommendation:** Set an explicit max frame size via `WebSocketUpgrade::max_frame_size()`.

---

## Positive Findings

- JWT uses HS256 with proper secret generation and validation (issuer, audience, expiry)
- Secret files use `0o600` permissions (aside from the TOCTOU gap)
- SQL queries use `sqlx` parameterized binding in most places
- API keys are redacted in logs via `redacted_secret_audit`
- Credential storage integrates with system keyring (KWallet/Secret Service)
- TLS uses `rustls` by default
- No deprecated cryptographic algorithms

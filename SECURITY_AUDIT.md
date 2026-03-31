# Security Audit — desktop-assistant

**Date:** 2026-03-31
**Scope:** All crates in the `desktop-assistant/` Cargo workspace

---

## Accepted Risks

### 1. No WebSocket Origin Validation (ACCEPTED — LOW)

**File:** `crates/ws-interface/src/lib.rs:135-149`

**Status:** Accepted risk (2026-03-31)
**Rationale:** The intended clients are native applications (gtk-client, TUI), not browsers. Native clients do not send `Origin` headers, so CSWSH is not a practical attack vector. Revisit if browser-based clients are ever added.

---

### 2. Internal Error Details Leaked to Clients (ACCEPTED — LOW)

**File:** `crates/ws-interface/src/lib.rs:184, 309-332, 348-371`

**Status:** Accepted risk (2026-03-31)
**Rationale:** All clients are trusted native applications running on the same machine. Detailed errors are useful for client-side diagnostics. Revisit if the WebSocket API is ever exposed to untrusted clients.

---

### 3. Hardcoded PAM Service Name (ACCEPTED — LOW)

**File:** `crates/daemon/src/config.rs:1715`

**Status:** Accepted risk (2026-03-31)
**Rationale:** The `"login"` service works correctly. Using a dedicated PAM service is a hygiene improvement but not a vulnerability. Revisit when packaging/distribution is addressed.

---

### 4. Server Auth Config via Environment Variables (ACCEPTED — LOW)

**File:** `crates/daemon/src/main.rs:174-179`

**Status:** Accepted risk (2026-03-31)
**Rationale:** These env vars are *server-side* configuration controlling what credentials the `/login` endpoint accepts — not client credentials being passed around. When managed by systemd, env vars can be loaded from restricted files via `EnvironmentFile=`.

---

## Remaining Low Severity

### 5. No Rate Limiting on WebSocket Messages (LOW)

**File:** `crates/ws-interface/src/lib.rs`

No rate limiting on inbound messages. An authenticated client could spam the server.

**Recommendation:** Implement per-connection rate limiting and send error frames for malformed input.

---

### 6. FNV-1a Hash for Secret Fingerprinting (LOW)

**File:** `crates/daemon/src/config.rs:1142-1154`

FNV-1a is used to generate log fingerprints for secrets. FNV is not collision-resistant.

**Recommendation:** Use SHA-256 or BLAKE3 truncated to 16 bytes.

---

### 7. No Explicit WebSocket Message Size Limit (LOW)

**File:** `crates/ws-interface/src/lib.rs`

WebSocket frame size relies on Axum defaults rather than an explicit configuration.

**Recommendation:** Set an explicit max frame size via `WebSocketUpgrade::max_frame_size()`.

---

## Resolved (2026-03-31)

- MCP command validation — shell metacharacter rejection added
- SQL LIMIT parameterization — wrapped in subquery with `$1` bind
- PAM FFI lifetime — ConvData boxed with explicit cleanup after pam_end
- JWT signing key TOCTOU — atomic file creation with `mode(0o600)`
- OIDC discovery timeouts — 10s connect, 30s request, 5 redirect limit, 1 MiB cap
- Client CLI credential removal — `--ws-jwt`/`--ws-login-*` removed from TUI

## Positive Findings

- JWT uses HS256 with proper secret generation and validation (issuer, audience, expiry)
- Secret files use `0o600` permissions atomically
- SQL queries use `sqlx` parameterized binding
- API keys are redacted in logs via `redacted_secret_audit`
- Credential storage integrates with system keyring (KWallet/Secret Service)
- TLS uses `rustls` by default
- No deprecated cryptographic algorithms

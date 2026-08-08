# Security Audit — desktop-assistant

**Audited:** 2026-03-31
**Last updated:** 2026-08-08
**Scope:** All crates in the `desktop-assistant/` Cargo workspace

---

## Accepted Risks

### 2. Internal Error Details Leaked to Clients (ACCEPTED — LOW)

**File:** `crates/ws-interface/src/lib.rs:184, 309-332, 348-371`

**Status:** Accepted risk (2026-03-31)
**Rationale:** All clients are trusted native applications running on the same machine. Detailed errors are useful for client-side diagnostics. Revisit if the WebSocket API is ever exposed to untrusted clients.

---

### 3. Hardcoded PAM Service Name (ACCEPTED — LOW)

**File:** `crates/daemon/src/config/pam_auth.rs:184`

**Status:** Accepted risk (2026-03-31)
**Rationale:** The `"login"` service works correctly. Using a dedicated PAM service is a hygiene improvement but not a vulnerability. Revisit when packaging/distribution is addressed.

---

### 4. Server Auth Config via Environment Variables (ACCEPTED — LOW)

**File:** `crates/daemon/src/transports.rs:539-544`

**Status:** Accepted risk (2026-03-31)
**Rationale:** These env vars are *server-side* configuration controlling what credentials the `/login` endpoint accepts — not client credentials being passed around. When managed by systemd, env vars can be loaded from restricted files via `EnvironmentFile=`.

---

## Remaining Low Severity

### 5. No Rate Limiting on WebSocket Messages (LOW)

**File:** `crates/ws-interface/src/lib.rs`

No rate limiting on inbound messages. An authenticated client could spam the server.

**Recommendation:** Implement per-connection rate limiting and send error frames for malformed input.

**Scope note:** This covers the post-authentication message stream only. The
pre-authentication `POST /login` endpoint is a separate concern and is throttled
(`crates/ws-interface/src/login_throttle.rs`): failed attempts are counted per
source address and per username, and the endpoint answers `429` with
`Retry-After` once the budget is spent.

---

### 6. FNV-1a Hash for Secret Fingerprinting (LOW)

**File:** `crates/daemon/src/config/secrets.rs:266-276`

FNV-1a is used to generate log fingerprints for secrets. FNV is not collision-resistant.

**Recommendation:** Use SHA-256 or BLAKE3 truncated to 16 bytes.

---

### 7. No Explicit WebSocket Message Size Limit (RESOLVED — 2026-05-27)

**File:** `crates/ws-interface/src/lib.rs`

**Status:** Resolved by #142 (`feat/issue-142-ws-msg-size`).
The `WebSocketUpgrade` now sets both `.max_message_size(4 << 20)` and `.max_frame_size(4 << 20)` (4 MiB), matching the UDS frame cap in `crates/uds-interface/src/lib.rs` and the D-Bus bridge cap in `crates/dbus-bridge/src/transport.rs`. The handler emits a clean RFC 6455 close (code 1009, "Message Too Big") when the cap is exceeded rather than dropping the TCP connection silently. See `docs/API_TRANSPORT.md` for the cross-transport summary.

---

## Resolved

### 1. No WebSocket Origin Validation (RESOLVED — 2026-08-08)

**File:** `crates/ws-interface/src/lib.rs:297`

**Status:** Resolved. `validate_origin()` checks the `Origin` header against the
`[ws_auth] allowed_origins` list and is enforced on both entry points a browser
can reach: the WebSocket upgrade (`:366`) and `POST /login` (`:487`). An
`Origin` that is not on the list is refused with `403` before any handler runs,
and an empty list refuses every `Origin`, so browser clients are denied by
default rather than allowed by omission.

**Remaining limit, by design:** a request carrying no `Origin` header at all is
allowed. That is the native-client path — tui/gtk send none. This closes the
browser CSWSH vector; it is not an authentication check, and it does not
constrain a non-browser client that simply omits the header.

---

## Positive Findings

- JWT uses HS256 with proper secret generation and validation (issuer, audience, expiry)
- Secret files use `0o600` permissions atomically
- SQL queries use `sqlx` parameterized binding
- API keys are redacted in logs via `redacted_secret_audit`
- Credential storage integrates with system keyring (KWallet/Secret Service)
- TLS uses `rustls` by default
- No deprecated cryptographic algorithms

//! OIDC token validator. Fetches the IdP's JWKS at startup, builds an
//! RS256 validator with the issuer (and optional audience) pinned, then
//! decides locally whether a presented token is valid.
//!
//! Extracted from `config.rs` (#41).
//!
//! Security notes:
//!
//! - JWKS is required to travel over a confidential channel. The
//!   discovery URL and the discovered `jwks_uri` are both required to
//!   be `https://` *or* an explicit loopback `http://`; anything else
//!   is rejected. Plaintext to a non-loopback host would let an
//!   attacker substitute a JWKS and forge tokens.
//! - JWKS keys are filtered: only `kty = RSA` entries with
//!   `use ∈ { sig, absent }` and `alg ∈ { RS256, absent }` are kept,
//!   so a key tagged for encryption (`use = enc`) won't accidentally
//!   verify a signature.
//! - Discovery + JWKS responses are size-capped to 1 MiB, enforced *during*
//!   the streamed read so an over-cap body is never fully buffered (DT-14).
//! - HMAC algorithms are explicitly *not* allowed by the validator
//!   (RS256 only), defending against the JWKS-substitution-via-`alg=HS256`
//!   class of attacks.

use std::collections::HashMap;

use anyhow::anyhow;
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode_header};

use super::OidcConfig;

/// Cached JWKS key set for validating external OIDC tokens.
///
/// Keys with a `kid` are stored in [`Self::keys_by_kid`] for direct
/// lookup; keys without a `kid` go into [`Self::kidless_keys`] and are
/// the fallback iterate-and-try set. A presented token with a `kid`
/// header is matched against `keys_by_kid` first; on miss (or for
/// tokens whose header has no `kid`) we fall through to the kid-less
/// list. The fallback exists because some IdPs serve unkeyed tokens
/// during a key rotation, so a strict kid-only path would briefly
/// reject otherwise valid tokens (#36).
pub struct OidcValidator {
    keys_by_kid: HashMap<String, DecodingKey>,
    kidless_keys: Vec<DecodingKey>,
    validation: Validation,
}

impl OidcValidator {
    /// Build a reqwest client with timeouts suitable for OIDC discovery.
    fn oidc_http_client() -> reqwest::Client {
        reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new())
    }

    /// Maximum response body size for OIDC discovery / JWKS documents (1 MiB).
    const MAX_OIDC_RESPONSE_BYTES: usize = 1_048_576;

    /// Visible to the parent test module so the URL-shape acceptance
    /// tests can exercise it without a live JWKS server.
    pub(super) fn require_https_or_loopback(url: &str, field: &str) -> anyhow::Result<()> {
        let lower = url.trim().to_ascii_lowercase();
        if lower.starts_with("https://") {
            return Ok(());
        }
        if let Some(rest) = lower.strip_prefix("http://") {
            let host = rest
                .split(['/', '?', '#'])
                .next()
                .unwrap_or("")
                .rsplit_once('@')
                .map(|(_, h)| h)
                .unwrap_or(rest.split(['/', '?', '#']).next().unwrap_or(""));
            let host_only = host.rsplit_once(':').map(|(h, _)| h).unwrap_or(host);
            if matches!(host_only, "localhost" | "127.0.0.1" | "[::1]" | "::1") {
                return Ok(());
            }
        }
        Err(anyhow!(
            "OIDC {field} must use https:// (or http://localhost for development); got {url}"
        ))
    }

    /// Fetch a JSON document, enforcing `max_bytes` *during* the read (DT-14).
    ///
    /// The previous implementation called `response.bytes()`, which buffers the
    /// entire body before the size check — a hostile or misconfigured IdP could
    /// stream gigabytes and exhaust memory before we ever reject it. Here we
    /// pull the body chunk-by-chunk and bail the moment the accumulated size
    /// would exceed the cap, so we never hold more than ~`max_bytes` (plus one
    /// chunk) in memory.
    async fn fetch_oidc_json(
        client: &reqwest::Client,
        url: &str,
        max_bytes: usize,
    ) -> anyhow::Result<serde_json::Value> {
        let mut response = client.get(url).send().await?;

        // If the server advertises a content-length over the cap, reject before
        // reading a single byte of body.
        if let Some(len) = response.content_length()
            && len > max_bytes as u64
        {
            return Err(anyhow!(
                "OIDC response from {url} exceeds size limit \
                 ({len} bytes advertised > {max_bytes})"
            ));
        }

        let mut buf: Vec<u8> = Vec::with_capacity(max_bytes.min(8 * 1024));
        while let Some(chunk) = response.chunk().await? {
            if buf.len() + chunk.len() > max_bytes {
                return Err(anyhow!(
                    "OIDC response from {url} exceeds size limit ({max_bytes} bytes)"
                ));
            }
            buf.extend_from_slice(&chunk);
        }
        Ok(serde_json::from_slice(&buf)?)
    }

    /// Fetch JWKS from the IdP and build a validator.
    pub async fn from_config(oidc: &OidcConfig) -> anyhow::Result<Self> {
        let client = Self::oidc_http_client();

        // JWKS must travel over a confidential channel — plaintext fetch lets
        // an attacker swap keys and forge tokens. Permit http only for explicit
        // loopback (development). The jwks_uri override is checked for the
        // same reason.
        Self::require_https_or_loopback(&oidc.issuer_url, "issuer_url")?;
        if !oidc.jwks_uri.is_empty() {
            Self::require_https_or_loopback(&oidc.jwks_uri, "jwks_uri")?;
        }

        let jwks_uri = if oidc.jwks_uri.is_empty() {
            let discovery_url = format!(
                "{}/.well-known/openid-configuration",
                oidc.issuer_url.trim_end_matches('/')
            );
            let discovery =
                Self::fetch_oidc_json(&client, &discovery_url, Self::MAX_OIDC_RESPONSE_BYTES)
                    .await?;
            let resolved = discovery["jwks_uri"]
                .as_str()
                .ok_or_else(|| anyhow!("no jwks_uri in OIDC discovery document"))?
                .to_string();
            Self::require_https_or_loopback(&resolved, "discovered jwks_uri")?;
            resolved
        } else {
            oidc.jwks_uri.clone()
        };

        let jwks = Self::fetch_oidc_json(&client, &jwks_uri, Self::MAX_OIDC_RESPONSE_BYTES).await?;

        let keys = jwks["keys"]
            .as_array()
            .ok_or_else(|| anyhow!("no keys in JWKS response"))?;

        let mut keys_by_kid: HashMap<String, DecodingKey> = HashMap::new();
        let mut kidless_keys: Vec<DecodingKey> = Vec::new();
        for key in keys {
            if key["kty"].as_str() != Some("RSA") {
                continue;
            }
            // JWKS entries optionally declare key usage (`use`) and algorithm
            // (`alg`). Skip keys that are explicitly tagged for encryption or a
            // non-RS256 algorithm — otherwise a key meant for `enc` would be
            // accepted as a token signature.
            if let Some(usage) = key["use"].as_str()
                && usage != "sig"
            {
                continue;
            }
            if let Some(alg) = key["alg"].as_str()
                && alg != "RS256"
            {
                continue;
            }
            let (Some(n), Some(e)) = (key["n"].as_str(), key["e"].as_str()) else {
                continue;
            };
            if n.is_empty() || e.is_empty() {
                continue;
            }
            let Ok(dk) = DecodingKey::from_rsa_components(n, e) else {
                continue;
            };
            match key["kid"].as_str().map(str::to_string) {
                Some(kid) if !kid.is_empty() => {
                    keys_by_kid.insert(kid, dk);
                }
                _ => kidless_keys.push(dk),
            }
        }

        if keys_by_kid.is_empty() && kidless_keys.is_empty() {
            anyhow::bail!("no usable RSA keys found in JWKS");
        }

        let mut validation = Validation::new(Algorithm::RS256);
        validation.validate_exp = true;
        validation.set_issuer(&[&oidc.issuer_url]);
        if !oidc.audience.is_empty() {
            validation.set_audience(&[&oidc.audience]);
        }

        Ok(Self {
            keys_by_kid,
            kidless_keys,
            validation,
        })
    }

    /// Decode and validate `token`, then return the `sub` claim from
    /// it. Returns `None` for tokens this validator would reject.
    /// Mirrors [`Self::validate_token`]'s kid/kidless resolution order
    /// so a token that validates here will also validate there, and a
    /// successfully-decoded payload yields its `sub` string.
    ///
    /// Used by the WS auth path (#105) to map a validated OIDC bearer
    /// token to the `user_id` that scopes storage queries.
    pub fn extract_sub(&self, token: &str) -> Option<String> {
        let header_kid = decode_header(token).ok().and_then(|h| h.kid);

        if let Some(kid) = header_kid.as_deref()
            && let Some(key) = self.keys_by_kid.get(kid)
            && let Ok(data) =
                jsonwebtoken::decode::<serde_json::Value>(token, key, &self.validation)
        {
            return data
                .claims
                .get("sub")
                .and_then(|v| v.as_str())
                .map(String::from);
        }

        for key in &self.kidless_keys {
            if let Ok(data) =
                jsonwebtoken::decode::<serde_json::Value>(token, key, &self.validation)
            {
                return data
                    .claims
                    .get("sub")
                    .and_then(|v| v.as_str())
                    .map(String::from);
            }
        }
        None
    }

    pub fn validate_token(&self, token: &str) -> bool {
        // Resolution order:
        //
        // 1. Parse the JWT header. If it carries a `kid`, look that key
        //    up directly in `keys_by_kid` and try only that one. A
        //    direct hit short-circuits the rest of the verifier set,
        //    so a JWKS with rotated-out keys can't slow validation
        //    down to O(N) and a deliberately-mislabelled `kid` can't
        //    reach a key it isn't authorised against.
        // 2. If the header has no `kid`, OR the `kid` doesn't match
        //    any cached key, fall through to the kid-less keys (the
        //    legacy iterate-and-try set). Some IdPs serve unkeyed
        //    tokens during a brief rotation window, so a strict
        //    kid-only path would briefly reject otherwise-valid
        //    tokens (#36).
        // 3. If header parsing itself fails, the token is malformed —
        //    skip step 1 and fall through to the same fallback set.
        //    `jsonwebtoken::decode` will then reject the malformed
        //    token consistently.
        let header_kid = decode_header(token).ok().and_then(|h| h.kid);

        if let Some(kid) = header_kid.as_deref()
            && let Some(key) = self.keys_by_kid.get(kid)
            && jsonwebtoken::decode::<serde_json::Value>(token, key, &self.validation).is_ok()
        {
            return true;
        }

        for key in &self.kidless_keys {
            if jsonwebtoken::decode::<serde_json::Value>(token, key, &self.validation).is_ok() {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DT-14: an over-cap response body must be rejected by the streamed read
    /// *without first buffering the whole body*. We serve a body well over a
    /// small test cap and assert `fetch_oidc_json` errors with a size message.
    #[tokio::test]
    async fn fetch_oidc_json_rejects_body_over_cap() {
        let server = httpmock::MockServer::start_async().await;
        // 64 KiB body, far above the 1 KiB cap we pass below.
        let big = "x".repeat(64 * 1024);
        let mock = server
            .mock_async(|when, then| {
                when.method(httpmock::Method::GET).path("/jwks");
                then.status(200)
                    .header("content-type", "application/json")
                    .body(big);
            })
            .await;

        let client = OidcValidator::oidc_http_client();
        let url = format!("{}/jwks", server.base_url());
        let err = OidcValidator::fetch_oidc_json(&client, &url, 1024)
            .await
            .expect_err("an over-cap body must be rejected");
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("size") || msg.contains("limit") || msg.contains("exceed"),
            "error should describe the size cap, got: {err}"
        );
        mock.assert_async().await;
    }

    /// A well-formed under-cap document parses as JSON.
    #[tokio::test]
    async fn fetch_oidc_json_parses_under_cap() {
        let server = httpmock::MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(httpmock::Method::GET).path("/disc");
                then.status(200)
                    .header("content-type", "application/json")
                    .body(r#"{"jwks_uri":"https://idp.example.com/keys"}"#);
            })
            .await;

        let client = OidcValidator::oidc_http_client();
        let url = format!("{}/disc", server.base_url());
        let value = OidcValidator::fetch_oidc_json(&client, &url, 1_048_576)
            .await
            .expect("under-cap body should parse");
        assert_eq!(
            value["jwks_uri"].as_str(),
            Some("https://idp.example.com/keys")
        );
        mock.assert_async().await;
    }
}

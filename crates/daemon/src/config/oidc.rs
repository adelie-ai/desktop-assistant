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
    use jsonwebtoken::{EncodingKey, Header};
    use std::time::{SystemTime, UNIX_EPOCH};

    // Two pre-generated 2048-bit RSA keypairs, embedded so the tests don't
    // depend on an `rsa` dev-dep or runtime keygen. `KEY*_PEM` is the PKCS#8
    // private key (for signing test tokens); `KEY*_N`/`KEY*_E` are the
    // base64url JWKS modulus/exponent for the matching public key.
    const KEY1_PEM: &str = include_str!("testdata/oidc_test_key1.pem");
    const KEY2_PEM: &str = include_str!("testdata/oidc_test_key2.pem");
    const KEY1_N: &str = "tsvRqmnpGiItl1UWXeD_w534wbRkd2Z4yfgLTlrWcLo1e-Ch3LrU96iq6aK5bGbqC4LXyict8e90KVflNVbOfjPxwdPIR-bCPZSZUhMmk9HyRkk9paHJ0MJ0_hFHZBSW3ghxtXWui20b9AzVRnlEVQPGZCnLNogoz9zXa9EbNFz_WVePJ9EeyxpJpwdRtGkyi462JbcAYv3Kx6JqXVOPOVTfxo2W9Xps1lkRZvfwr-SH2_JOmV3fct1B4JXu-_-zxNeZOA2VWb9XcoEGXm0Twitf9PJ8bRjBJ6Lwq89pRIkC6JemsF1VzW8Ym3eepdSw_ebvHaRac8vkOeD7gufcNQ";
    const KEY2_N: &str = "6O3wENWDW2t8PKSAivkyqoAjuvd1_OQeuEvbNVPH9wBY4AYZxiwFYyPXr-uebqn-qNJm2Ne_eFb579CKDr0h7oz91ZerapZm2ZqO3VRoY_bhBGSkWUn91IWLP4eDevszqJUdGPZfGG3lJmz7NTkfsTWguaP6KNuFbabOZG3rku3R4B9D2eY_KccPiKHsg_ZsPy-mWdzjWyQQkXdnS8Ajse7fugbTlUXW2F5udTWJ8VdpRKGOZPL2Oqsns6yli1dHdmHPbK3ilVXHfn9y5JdEGpYCdkDcyreCEAiSuX55owlBZU0dc5J3Fs--Gu52jutiayvcImWo2TsJD6GRZhnM5w";
    const KEY_E: &str = "AQAB";

    const TEST_ISSUER: &str = "https://idp.example.com";
    const TEST_AUDIENCE: &str = "desktop-assistant";

    #[derive(serde::Serialize)]
    struct TestClaims {
        sub: String,
        iss: String,
        aud: String,
        exp: u64,
    }

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Mint a signed RS256 token with the given `kid`, issuer, and audience.
    fn mint_token(pem: &str, kid: Option<&str>, iss: &str, aud: &str, exp: u64) -> String {
        let mut header = Header::new(Algorithm::RS256);
        header.kid = kid.map(String::from);
        let claims = TestClaims {
            sub: "user-123".into(),
            iss: iss.into(),
            aud: aud.into(),
            exp,
        };
        let key = EncodingKey::from_rsa_pem(pem.as_bytes()).expect("valid test RSA pem");
        jsonwebtoken::encode(&header, &claims, &key).expect("encode test token")
    }

    /// A single-key JWKS document with the given `kid`, modulus, and exponent.
    fn jwks_body(kid: &str, n: &str, e: &str) -> String {
        serde_json::json!({
            "keys": [{
                "kty": "RSA",
                "use": "sig",
                "alg": "RS256",
                "kid": kid,
                "n": n,
                "e": e,
            }]
        })
        .to_string()
    }

    /// Build a validator bound to `server`'s `jwks_uri`, with audience pinned.
    async fn validator_for(server: &httpmock::MockServer) -> OidcValidator {
        let oidc = OidcConfig {
            issuer_url: TEST_ISSUER.to_string(),
            authorization_endpoint: String::new(),
            token_endpoint: String::new(),
            client_id: "test-client".to_string(),
            scopes: "openid".to_string(),
            jwks_uri: format!("{}/jwks", server.base_url()),
            audience: TEST_AUDIENCE.to_string(),
        };
        OidcValidator::from_config(&oidc)
            .await
            .expect("validator builds against mock JWKS")
    }

    /// DT-2 #268: audience is mandatory. A config with an empty audience must
    /// be rejected at build time — an empty audience would accept any token
    /// the issuer minted, for any relying party.
    #[tokio::test]
    async fn from_config_rejects_empty_audience() {
        let oidc = OidcConfig {
            issuer_url: TEST_ISSUER.to_string(),
            authorization_endpoint: String::new(),
            token_endpoint: String::new(),
            client_id: "test-client".to_string(),
            scopes: "openid".to_string(),
            jwks_uri: "https://idp.example.com/jwks".to_string(),
            audience: "   ".to_string(), // whitespace-only == empty
        };
        let err = OidcValidator::from_config(&oidc)
            .await
            .expect_err("empty audience must be rejected");
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("audience"),
            "error should explain the audience requirement, got: {err}"
        );
    }

    /// DT-2 #268: a token whose `aud` doesn't match the configured audience is
    /// rejected, even when its signature, issuer, and expiry are all valid.
    #[tokio::test]
    async fn token_with_wrong_audience_is_rejected() {
        let server = httpmock::MockServer::start_async().await;
        let jwks = server
            .mock_async(|when, then| {
                when.method(httpmock::Method::GET).path("/jwks");
                then.status(200)
                    .header("content-type", "application/json")
                    .body(jwks_body("key1", KEY1_N, KEY_E));
            })
            .await;

        let validator = validator_for(&server).await;
        jwks.assert_async().await;

        // Correctly signed, correct issuer, not expired — but minted for a
        // *different* audience.
        let token = mint_token(
            KEY1_PEM,
            Some("key1"),
            TEST_ISSUER,
            "some-other-service",
            now_secs() + 3600,
        );
        assert!(
            !validator.validate_token(&token).await,
            "a token for a different audience must be rejected"
        );

        // Sanity: the same key with the *right* audience is accepted, proving
        // the rejection above is the audience check, not a setup error.
        let good = mint_token(
            KEY1_PEM,
            Some("key1"),
            TEST_ISSUER,
            TEST_AUDIENCE,
            now_secs() + 3600,
        );
        assert!(
            validator.validate_token(&good).await,
            "a token for the configured audience must be accepted"
        );
    }

    /// DT-2 #268: after IdP key rotation, a token signed by the *new* key is
    /// picked up via an unknown-`kid` refetch — no daemon restart required.
    #[tokio::test]
    async fn rotated_key_is_picked_up_after_refresh() {
        let server = httpmock::MockServer::start_async().await;

        // Startup JWKS serves only key1.
        let jwks_v1 = server
            .mock_async(|when, then| {
                when.method(httpmock::Method::GET).path("/jwks");
                then.status(200)
                    .header("content-type", "application/json")
                    .body(jwks_body("key1", KEY1_N, KEY_E));
            })
            .await;

        let validator = validator_for(&server).await;
        jwks_v1.assert_async().await;

        // A token signed by the as-yet-unknown rotated key2 is initially
        // unverifiable.
        let rotated = mint_token(
            KEY2_PEM,
            Some("key2"),
            TEST_ISSUER,
            TEST_AUDIENCE,
            now_secs() + 3600,
        );

        // IdP rotates: the JWKS now serves key2 instead. Replace the mock.
        jwks_v1.delete_async().await;
        let jwks_v2 = server
            .mock_async(|when, then| {
                when.method(httpmock::Method::GET).path("/jwks");
                then.status(200)
                    .header("content-type", "application/json")
                    .body(jwks_body("key2", KEY2_N, KEY_E));
            })
            .await;

        // The unknown `kid=key2` triggers a refetch, which now returns key2,
        // so the rotated token validates.
        assert!(
            validator.validate_token(&rotated).await,
            "a token signed by the rotated key must validate after JWKS refresh"
        );
        assert!(
            jwks_v2.hits_async().await >= 1,
            "an unknown kid must trigger a JWKS refetch"
        );
    }

    /// DT-2 #268: an unknown `kid` triggers exactly one refetch; the cache is
    /// updated so a *second* token with the same now-known `kid` does NOT
    /// refetch again.
    #[tokio::test]
    async fn unknown_kid_triggers_refetch_then_caches() {
        let server = httpmock::MockServer::start_async().await;
        let jwks_v1 = server
            .mock_async(|when, then| {
                when.method(httpmock::Method::GET).path("/jwks");
                then.status(200)
                    .header("content-type", "application/json")
                    .body(jwks_body("key1", KEY1_N, KEY_E));
            })
            .await;
        let validator = validator_for(&server).await;
        jwks_v1.assert_async().await;
        jwks_v1.delete_async().await;

        let jwks_v2 = server
            .mock_async(|when, then| {
                when.method(httpmock::Method::GET).path("/jwks");
                then.status(200)
                    .header("content-type", "application/json")
                    .body(jwks_body("key2", KEY2_N, KEY_E));
            })
            .await;

        let token = mint_token(
            KEY2_PEM,
            Some("key2"),
            TEST_ISSUER,
            TEST_AUDIENCE,
            now_secs() + 3600,
        );
        // First use: unknown kid → one refetch → validates.
        assert!(validator.validate_token(&token).await);
        let hits_after_first = jwks_v2.hits_async().await;
        assert_eq!(hits_after_first, 1, "first unknown kid refetches exactly once");

        // Second use of the now-known kid must not refetch again.
        assert!(validator.validate_token(&token).await);
        assert_eq!(
            jwks_v2.hits_async().await,
            hits_after_first,
            "a now-cached kid must not trigger another refetch"
        );
    }

    /// DT-2 #268: a flood of distinct unknown `kid`s must not hammer the IdP —
    /// refetches are rate-limited to at most one per the minimum interval.
    #[tokio::test]
    async fn refetch_is_rate_limited() {
        let server = httpmock::MockServer::start_async().await;
        let jwks = server
            .mock_async(|when, then| {
                when.method(httpmock::Method::GET).path("/jwks");
                then.status(200)
                    .header("content-type", "application/json")
                    .body(jwks_body("key1", KEY1_N, KEY_E));
            })
            .await;
        let validator = validator_for(&server).await;
        // One startup fetch so far.
        let baseline = jwks.hits_async().await;
        assert_eq!(baseline, 1);

        // Hammer with many tokens carrying distinct, never-cached `kid`s. Each
        // is signed by key2 (which the JWKS never serves) so it never validates
        // and every call sees an unknown kid.
        for i in 0..20 {
            let kid = format!("garbage-{i}");
            let token = mint_token(
                KEY2_PEM,
                Some(&kid),
                TEST_ISSUER,
                TEST_AUDIENCE,
                now_secs() + 3600,
            );
            assert!(!validator.validate_token(&token).await);
        }

        // The first unknown kid fired one refetch; the rest were throttled by
        // the rate limiter, so total fetches are well under 20.
        let total = jwks.hits_async().await;
        assert!(
            total <= baseline + 1,
            "rate limiter must cap refetches: saw {total} fetches for 20 unknown kids"
        );
    }

    /// An expired token is rejected even when signed by a cached key.
    #[tokio::test]
    async fn expired_token_is_rejected() {
        let server = httpmock::MockServer::start_async().await;
        let jwks = server
            .mock_async(|when, then| {
                when.method(httpmock::Method::GET).path("/jwks");
                then.status(200)
                    .header("content-type", "application/json")
                    .body(jwks_body("key1", KEY1_N, KEY_E));
            })
            .await;
        let validator = validator_for(&server).await;
        jwks.assert_async().await;

        let expired = mint_token(
            KEY1_PEM,
            Some("key1"),
            TEST_ISSUER,
            TEST_AUDIENCE,
            now_secs().saturating_sub(3600),
        );
        assert!(
            !validator.validate_token(&expired).await,
            "an expired token must be rejected"
        );
    }

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

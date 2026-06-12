//! OIDC token validator. Fetches the IdP's JWKS at startup, builds an
//! RS256 validator with the issuer and (required) audience pinned, then
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
//! - Audience validation is **required** (DT-2 #268). `from_config` refuses
//!   to build a validator when `oidc.audience` is empty, because an empty
//!   audience disables the `aud` check and accepts any token the issuer ever
//!   minted (for any other service). This only ever narrows acceptance.
//! - The JWKS is **refreshed** (DT-2 #268), not frozen at startup, so IdP key
//!   rotation no longer silently locks every user out until a daemon restart:
//!   a token presenting an unknown `kid` triggers a rate-limited refetch from
//!   the *same* configured `jwks_uri`. Refresh only ever *adds* keys from the
//!   pinned URI — the issuer, audience, and algorithm rules are unchanged — so
//!   it can never widen acceptance. The rate limit (a minimum interval between
//!   refetch attempts) means a flood of garbage `kid`s can't hammer the IdP.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::anyhow;
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode_header};
use tokio::sync::{Mutex, RwLock};

use super::OidcConfig;

/// Minimum spacing between unknown-`kid` JWKS refetch attempts.
///
/// An unknown `kid` is the signal "the IdP may have rotated keys"; we refetch
/// so a freshly-rotated key is picked up without a daemon restart. But a flood
/// of tokens carrying garbage `kid`s would otherwise hammer the IdP once per
/// request — so refetch attempts are throttled to at most one per this
/// interval. A genuine rotation is picked up within one interval; garbage
/// `kid`s cost at most one refetch per interval regardless of volume.
const JWKS_REFRESH_MIN_INTERVAL: Duration = Duration::from_secs(60);

/// The decoding keys parsed from a JWKS document, split by whether the source
/// JWK carried a `kid`.
#[derive(Default)]
struct KeySet {
    keys_by_kid: HashMap<String, DecodingKey>,
    kidless_keys: Vec<DecodingKey>,
}

impl KeySet {
    fn is_empty(&self) -> bool {
        self.keys_by_kid.is_empty() && self.kidless_keys.is_empty()
    }
}

/// Tracks the last JWKS refetch attempt so refreshes can be rate-limited.
struct RefreshState {
    /// When the last refetch was *attempted* (success or failure), used to
    /// throttle. `None` until the first refetch.
    last_attempt: Option<Instant>,
    /// Minimum spacing between refetch attempts.
    min_interval: Duration,
}

/// Cached JWKS key set for validating external OIDC tokens.
///
/// Keys with a `kid` are stored in [`KeySet::keys_by_kid`] for direct lookup;
/// keys without a `kid` go into [`KeySet::kidless_keys`] and are the fallback
/// iterate-and-try set. A presented token with a `kid` header is matched
/// against `keys_by_kid` first; on miss (or for tokens whose header has no
/// `kid`) we fall through to the kid-less list. The fallback exists because
/// some IdPs serve unkeyed tokens during a key rotation, so a strict kid-only
/// path would briefly reject otherwise valid tokens (#36).
///
/// The key set lives behind an `RwLock` so the validator can refresh it when a
/// token presents an unknown `kid` (DT-2 #268). Refresh refetches from the
/// pinned [`Self::jwks_uri`] only; the [`Self::validation`] rules never change.
pub struct OidcValidator {
    keys: Arc<RwLock<KeySet>>,
    validation: Validation,
    /// HTTP client reused for refresh refetches.
    client: reqwest::Client,
    /// The resolved (and scheme-validated) JWKS URI. Refresh refetches keys
    /// from *exactly* this URI, so a refresh can never introduce keys from a
    /// different origin.
    jwks_uri: String,
    /// Rate-limit state for unknown-`kid`-triggered refetches.
    refresh_state: Mutex<RefreshState>,
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

    /// Parse a JWKS JSON document into a [`KeySet`], applying the usage /
    /// algorithm / `kty` filters. Errors only when the document has no `keys`
    /// array at all; individual unusable keys are skipped.
    fn parse_jwks(jwks: &serde_json::Value) -> anyhow::Result<KeySet> {
        let keys = jwks["keys"]
            .as_array()
            .ok_or_else(|| anyhow!("no keys in JWKS response"))?;

        let mut set = KeySet::default();
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
                    set.keys_by_kid.insert(kid, dk);
                }
                _ => set.kidless_keys.push(dk),
            }
        }
        Ok(set)
    }

    /// Fetch JWKS from the IdP and build a validator.
    pub async fn from_config(oidc: &OidcConfig) -> anyhow::Result<Self> {
        let client = Self::oidc_http_client();

        // Audience validation is mandatory (DT-2 #268). An empty audience
        // disables the `aud` check entirely, so the validator would accept any
        // token the issuer minted for *any* relying party. Refuse to start in
        // that posture rather than silently widen acceptance.
        if oidc.audience.trim().is_empty() {
            return Err(anyhow!(
                "OIDC audience is required: set `oidc.audience` to this service's \
                 expected `aud` value. An empty audience would accept any token \
                 from the issuer, including tokens minted for other services."
            ));
        }

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
        let set = Self::parse_jwks(&jwks)?;

        if set.is_empty() {
            anyhow::bail!("no usable RSA keys found in JWKS");
        }

        let mut validation = Validation::new(Algorithm::RS256);
        validation.validate_exp = true;
        validation.set_issuer(&[&oidc.issuer_url]);
        // Non-empty (checked above), so this enables the mandatory `aud` check.
        validation.set_audience(&[&oidc.audience]);

        Ok(Self {
            keys: Arc::new(RwLock::new(set)),
            validation,
            client,
            jwks_uri,
            refresh_state: Mutex::new(RefreshState {
                last_attempt: None,
                min_interval: JWKS_REFRESH_MIN_INTERVAL,
            }),
        })
    }

    /// Refetch the JWKS from the pinned [`Self::jwks_uri`] and replace the
    /// cached key set, *if* the rate-limit window has elapsed.
    ///
    /// Returns `Ok(true)` when a refetch was performed (and the cache updated),
    /// `Ok(false)` when the call was throttled (too soon since the last
    /// attempt), and `Err(_)` when a refetch was attempted but failed (network
    /// error, oversized body, no usable keys). A failed refetch leaves the
    /// existing cache untouched, so transient IdP errors never *remove* a key
    /// that was working.
    ///
    /// Hard rule (DT-2 #268): this only ever refetches from the same configured
    /// URI and re-applies the exact same key filters, so it can add keys but
    /// never relax validation.
    async fn refresh(&self) -> anyhow::Result<bool> {
        {
            let mut state = self.refresh_state.lock().await;
            if let Some(last) = state.last_attempt
                && last.elapsed() < state.min_interval
            {
                return Ok(false);
            }
            // Record the attempt *before* the network call so concurrent
            // unknown-`kid` requests don't all fire a refetch at once.
            state.last_attempt = Some(Instant::now());
        }

        let jwks =
            Self::fetch_oidc_json(&self.client, &self.jwks_uri, Self::MAX_OIDC_RESPONSE_BYTES)
                .await?;
        let set = Self::parse_jwks(&jwks)?;
        if set.is_empty() {
            return Err(anyhow!(
                "JWKS refresh from {} returned no usable RSA keys; keeping existing keys",
                self.jwks_uri
            ));
        }

        *self.keys.write().await = set;
        Ok(true)
    }

    /// Decode `token` against the currently-cached key set, returning the
    /// decoded claims on success. Mirrors the kid/kidless resolution order used
    /// by [`Self::validate_token`].
    ///
    /// `header_kid` is the (already-parsed) `kid` from the token header, if any.
    /// `kid_known` is set to whether that `kid` matched a cached key, so the
    /// caller can decide whether an unknown `kid` warrants a refresh.
    async fn try_decode(
        &self,
        token: &str,
        header_kid: Option<&str>,
        kid_known: &mut bool,
    ) -> Option<serde_json::Value> {
        let keys = self.keys.read().await;
        *kid_known = false;

        if let Some(kid) = header_kid
            && let Some(key) = keys.keys_by_kid.get(kid)
        {
            *kid_known = true;
            if let Ok(data) =
                jsonwebtoken::decode::<serde_json::Value>(token, key, &self.validation)
            {
                return Some(data.claims);
            }
        }

        for key in &keys.kidless_keys {
            if let Ok(data) =
                jsonwebtoken::decode::<serde_json::Value>(token, key, &self.validation)
            {
                return Some(data.claims);
            }
        }
        None
    }

    /// Decode and validate `token`, returning its claims, refreshing the JWKS
    /// once (rate-limited) if the token carries a `kid` we don't recognise.
    ///
    /// Resolution order:
    ///
    /// 1. Parse the JWT header. If it carries a `kid`, look that key up
    ///    directly in the cached set and try only that one. A direct hit
    ///    short-circuits the rest of the verifier set, so a JWKS with
    ///    rotated-out keys can't slow validation down to O(N) and a
    ///    deliberately-mislabelled `kid` can't reach a key it isn't authorised
    ///    against.
    /// 2. If the header has no `kid`, OR the `kid` doesn't match any cached
    ///    key, fall through to the kid-less keys (the legacy iterate-and-try
    ///    set). Some IdPs serve unkeyed tokens during a brief rotation window,
    ///    so a strict kid-only path would briefly reject otherwise-valid tokens
    ///    (#36).
    /// 3. If both fail *and* the token presented a `kid` we don't have cached,
    ///    the IdP may have rotated keys: refetch the JWKS (rate-limited) and
    ///    retry once. A token with no `kid`, or a `kid` we already know, does
    ///    not trigger a refetch (a known `kid` that fails is a bad token, not a
    ///    rotation).
    async fn decode_claims(&self, token: &str) -> Option<serde_json::Value> {
        let header_kid = decode_header(token).ok().and_then(|h| h.kid);

        let mut kid_known = false;
        if let Some(claims) = self
            .try_decode(token, header_kid.as_deref(), &mut kid_known)
            .await
        {
            return Some(claims);
        }

        // Only an unknown `kid` is a rotation signal worth a refetch. A token
        // with no `kid`, or with a `kid` we already cache, has already been
        // given every chance above.
        let unknown_kid = header_kid.is_some() && !kid_known;
        if !unknown_kid {
            return None;
        }

        match self.refresh().await {
            Ok(true) => {
                let mut kid_known_after = false;
                self.try_decode(token, header_kid.as_deref(), &mut kid_known_after)
                    .await
            }
            Ok(false) => None, // throttled; no new keys to try
            Err(error) => {
                tracing::warn!(jwks_uri = %self.jwks_uri, %error, "OIDC JWKS refresh failed");
                None
            }
        }
    }

    /// Decode and validate `token`, then return the `sub` claim from it.
    /// Returns `None` for tokens this validator would reject. A
    /// successfully-decoded payload yields its `sub` string.
    ///
    /// Used by the WS auth path (#105) to map a validated OIDC bearer token to
    /// the `user_id` that scopes storage queries.
    pub async fn extract_sub(&self, token: &str) -> Option<String> {
        self.decode_claims(token)
            .await?
            .get("sub")
            .and_then(|v| v.as_str())
            .map(String::from)
    }

    pub async fn validate_token(&self, token: &str) -> bool {
        self.decode_claims(token).await.is_some()
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
        // `OidcValidator` deliberately doesn't derive `Debug` (it holds
        // decoding keys), so match rather than `expect_err`.
        let err = match OidcValidator::from_config(&oidc).await {
            Ok(_) => panic!("empty audience must be rejected"),
            Err(err) => err,
        };
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
            jwks_v2.calls_async().await >= 1,
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
        let hits_after_first = jwks_v2.calls_async().await;
        assert_eq!(
            hits_after_first, 1,
            "first unknown kid refetches exactly once"
        );

        // Second use of the now-known kid must not refetch again.
        assert!(validator.validate_token(&token).await);
        assert_eq!(
            jwks_v2.calls_async().await,
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
        let baseline = jwks.calls_async().await;
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
        let total = jwks.calls_async().await;
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

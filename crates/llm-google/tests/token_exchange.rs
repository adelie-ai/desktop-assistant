//! Integration tests for the GCP service-account JWT-bearer token exchange
//! against an `httpmock` token endpoint. Signs a real RS256 assertion with a
//! throwaway test key; no real GCP is contacted.

use desktop_assistant_llm_google::{ServiceAccountKey, ServiceAccountTokenProvider, TokenProvider};
use httpmock::prelude::*;

/// A throwaway 2048-bit RSA key (never a real credential) used to sign the
/// assertion in tests. Shares the fixture the unit tests use.
const TEST_SA_KEY_PEM: &str = include_str!("../src/testdata/sa_test_key.pem");

fn test_key(token_uri: String) -> ServiceAccountKey {
    ServiceAccountKey {
        client_email: "svc@proj.iam.gserviceaccount.com".into(),
        private_key: TEST_SA_KEY_PEM.into(),
        token_uri,
        private_key_id: Some("kid-123".into()),
        project_id: Some("proj".into()),
    }
}

#[tokio::test]
async fn mints_access_token_via_jwt_bearer_grant() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(POST)
            .path("/token")
            .body_includes("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer")
            .body_includes("assertion=");
        then.status(200)
            .header("content-type", "application/json")
            .body(r#"{"access_token":"minted-abc","expires_in":3600,"token_type":"Bearer"}"#);
    });

    let provider = ServiceAccountTokenProvider::from_key(test_key(server.url("/token")));
    let token = provider.token().await.expect("mint ok");
    assert_eq!(token, "minted-abc");
    m.assert_calls(1);
}

#[tokio::test]
async fn caches_token_across_calls() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(POST).path("/token");
        then.status(200)
            .header("content-type", "application/json")
            .body(r#"{"access_token":"cached-xyz","expires_in":3600}"#);
    });

    let provider = ServiceAccountTokenProvider::from_key(test_key(server.url("/token")));
    let first = provider.token().await.expect("mint ok");
    let second = provider.token().await.expect("cached ok");
    assert_eq!(first, "cached-xyz");
    assert_eq!(second, "cached-xyz");
    // A valid, unexpired token is reused: the endpoint is hit exactly once.
    m.assert_calls(1);
}

#[tokio::test]
async fn token_endpoint_error_surfaces_without_leaking_material() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/token");
        then.status(400)
            .body(r#"{"error":"invalid_grant","error_description":"bad assertion"}"#);
    });

    let provider = ServiceAccountTokenProvider::from_key(test_key(server.url("/token")));
    let err = provider.token().await.expect_err("must fail");
    let detail = err.to_string();
    // The private key must never appear in the error.
    assert!(
        !detail.contains("PRIVATE KEY"),
        "leaked key material: {detail}"
    );
    assert!(
        detail.contains("token exchange") || detail.contains("400"),
        "got {detail}"
    );
}

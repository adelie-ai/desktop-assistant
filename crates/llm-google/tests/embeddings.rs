//! Integration tests for the `EmbeddingClient` impl on both surfaces: Vertex
//! `:predict` (`instances`/`predictions`) and the Gemini API `:embedContent`.

use std::sync::Arc;

use desktop_assistant_core::ports::embedding::EmbeddingClient;
use desktop_assistant_llm_google::{AuthMode, GoogleClient, StaticTokenProvider};
use httpmock::prelude::*;

#[tokio::test]
async fn vertex_predict_embeddings_round_trip() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/projects/test-proj/locations/us-central1/publishers/google/models/text-embedding-004:predict")
            .header("authorization", "Bearer test-token")
            .body_includes("\"instances\"")
            .body_includes("\"content\":\"alpha\"");
        then.status(200).header("content-type", "application/json").body(
            r#"{"predictions":[{"embeddings":{"values":[0.1,0.2,0.3]}},{"embeddings":{"values":[0.4,0.5,0.6]}}]}"#,
        );
    });

    let client = GoogleClient::new(String::new())
        .with_base_url(server.url(""))
        .with_project(Some("test-proj".into()))
        .with_location("us-central1")
        .with_model("text-embedding-004")
        .with_token_provider(Arc::new(StaticTokenProvider::new("test-token")));

    let vectors = client
        .embed(vec!["alpha".into(), "beta".into()])
        .await
        .expect("embeddings ok");
    m.assert_calls(1);
    assert_eq!(vectors.len(), 2);
    assert_eq!(vectors[0], vec![0.1, 0.2, 0.3]);
    assert_eq!(vectors[1], vec![0.4, 0.5, 0.6]);
}

#[tokio::test]
async fn gemini_api_embed_content_round_trip() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(POST)
            .path("/v1beta/models/text-embedding-004:embedContent")
            .header("x-goog-api-key", "my-api-key")
            .body_includes("\"parts\"");
        then.status(200)
            .header("content-type", "application/json")
            .body(r#"{"embedding":{"values":[0.7,0.8]}}"#);
    });

    let client = GoogleClient::new("my-api-key".into())
        .with_auth_mode(AuthMode::ApiKey)
        .with_base_url(server.url(""))
        .with_model("text-embedding-004");

    let vectors = client
        .embed(vec!["one".into(), "two".into()])
        .await
        .expect("embeddings ok");
    // `:embedContent` is single-text, so two inputs -> two calls.
    m.assert_calls(2);
    assert_eq!(vectors.len(), 2);
    assert_eq!(vectors[0], vec![0.7, 0.8]);
    assert_eq!(vectors[1], vec![0.7, 0.8]);
}

#[tokio::test]
async fn model_identifier_returns_the_embedding_model() {
    let client = GoogleClient::new(String::new()).with_model("gemini-embedding-001");
    assert_eq!(
        client.model_identifier().await.unwrap(),
        "gemini-embedding-001"
    );
}

#[tokio::test]
async fn embedding_http_error_surfaces_as_llm_error() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST)
            .path("/v1beta/models/text-embedding-004:embedContent");
        then.status(400).body("bad request");
    });

    let client = GoogleClient::new("k".into())
        .with_auth_mode(AuthMode::ApiKey)
        .with_base_url(server.url(""))
        .with_model("text-embedding-004");
    let err = client.embed(vec!["x".into()]).await.expect_err("must fail");
    assert!(
        err.to_string().contains("embeddings") || err.to_string().contains("400"),
        "got {err:?}"
    );
}

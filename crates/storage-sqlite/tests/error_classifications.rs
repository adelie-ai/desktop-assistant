//! Contract tests for [`SqliteErrorClassificationStore`] (epic #178). This is a
//! global store (no `user_id`).
#![cfg(feature = "sqlite")]

use desktop_assistant_core::ports::store::ErrorClassificationStore;
use desktop_assistant_storage_sqlite::{SqliteErrorClassificationStore, create_memory_pool};

async fn store() -> SqliteErrorClassificationStore {
    let pool = create_memory_pool().await.expect("pool");
    SqliteErrorClassificationStore::new(pool)
}

#[tokio::test]
async fn record_then_lookup_matches_substring() {
    let s = store().await;
    s.record("bedrock", "ThrottlingException", "rate_limited")
        .await
        .expect("record");

    let hit = s
        .lookup("bedrock", "ServiceError: ThrottlingException: slow down")
        .await
        .unwrap()
        .expect("signature is a substring of the message");
    assert_eq!(hit.signature, "ThrottlingException");
    assert_eq!(hit.cause, "rate_limited");
}

#[tokio::test]
async fn lookup_miss_returns_none() {
    let s = store().await;
    s.record("bedrock", "ThrottlingException", "rate_limited")
        .await
        .unwrap();
    assert!(
        s.lookup("bedrock", "totally unrelated error")
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn longest_matching_signature_wins() {
    let s = store().await;
    s.record("openai", "quota", "rate_limited").await.unwrap();
    s.record("openai", "insufficient_quota", "quota_exceeded")
        .await
        .unwrap();

    // Both signatures occur in the message; the more specific (longer) one wins.
    let hit = s
        .lookup("openai", "Error code 429: insufficient_quota for this org")
        .await
        .unwrap()
        .expect("a signature matches");
    assert_eq!(hit.signature, "insufficient_quota");
    assert_eq!(hit.cause, "quota_exceeded");
}

#[tokio::test]
async fn lookup_is_case_insensitive() {
    let s = store().await;
    s.record("ollama", "Model Is Loading", "model_loading")
        .await
        .unwrap();
    let hit = s
        .lookup("ollama", "the model is loading, please wait")
        .await
        .unwrap()
        .expect("case-insensitive substring match");
    assert_eq!(hit.cause, "model_loading");
}

#[tokio::test]
async fn signatures_are_connector_scoped() {
    let s = store().await;
    s.record("bedrock", "overloaded", "rate_limited")
        .await
        .unwrap();
    // A message under a different connector must not match bedrock's signature.
    assert!(
        s.lookup("openai", "the server is overloaded")
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn record_is_idempotent_upsert() {
    let s = store().await;
    s.record("bedrock", "Throttling", "rate_limited")
        .await
        .unwrap();
    // Re-recording the same (connector, signature) updates the cause rather
    // than duplicating or erroring.
    s.record("bedrock", "Throttling", "throttled")
        .await
        .unwrap();

    let hit = s
        .lookup("bedrock", "got a Throttling response")
        .await
        .unwrap()
        .expect("still matches");
    assert_eq!(hit.cause, "throttled");
}

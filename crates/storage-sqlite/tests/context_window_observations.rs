//! Contract tests for [`SqliteLearnedWindowStore`] (issues #343 / #425). Global
//! store (no `user_id`). Exercises the down-only overflow ratchet, the success
//! high-water mark, and their independence.
#![cfg(feature = "sqlite")]

use desktop_assistant_core::ports::store::LearnedWindowStore;
use desktop_assistant_storage_sqlite::{SqliteLearnedWindowStore, create_memory_pool};

async fn store() -> SqliteLearnedWindowStore {
    let pool = create_memory_pool().await.expect("pool");
    SqliteLearnedWindowStore::new(pool)
}

#[tokio::test]
async fn lookup_miss_returns_none() {
    let s = store().await;
    assert!(s.lookup("openai", "gpt-x").await.unwrap().is_none());
}

#[tokio::test]
async fn record_overflow_then_lookup() {
    let s = store().await;
    s.record_overflow("openai", "gpt-x", 1000, 4000)
        .await
        .unwrap();
    let w = s.lookup("openai", "gpt-x").await.unwrap().unwrap();
    assert_eq!(w.observed_limit, Some(1000));
    assert_eq!(w.configured_window, Some(4000));
    assert_eq!(w.max_success_input, None);
}

#[tokio::test]
async fn overflow_ratchets_down_only_within_same_configured_window() {
    let s = store().await;
    s.record_overflow("c", "m", 1000, 4000).await.unwrap();
    // A smaller observed limit wins (ratchet down).
    s.record_overflow("c", "m", 800, 4000).await.unwrap();
    assert_eq!(
        s.lookup("c", "m").await.unwrap().unwrap().observed_limit,
        Some(800)
    );
    // A larger observed limit for the same window is ignored.
    s.record_overflow("c", "m", 900, 4000).await.unwrap();
    assert_eq!(
        s.lookup("c", "m").await.unwrap().unwrap().observed_limit,
        Some(800)
    );
}

#[tokio::test]
async fn changed_configured_window_replaces_the_observation() {
    let s = store().await;
    s.record_overflow("c", "m", 800, 4000).await.unwrap();
    // A different configured window is a deliberate config change: start fresh
    // even though 1200 > 800.
    s.record_overflow("c", "m", 1200, 8000).await.unwrap();
    let w = s.lookup("c", "m").await.unwrap().unwrap();
    assert_eq!(w.observed_limit, Some(1200));
    assert_eq!(w.configured_window, Some(8000));
}

#[tokio::test]
async fn success_keeps_the_high_water_mark() {
    let s = store().await;
    s.record_success("c", "m", 500).await.unwrap();
    // A smaller success does not lower the high-water mark.
    s.record_success("c", "m", 300).await.unwrap();
    assert_eq!(
        s.lookup("c", "m").await.unwrap().unwrap().max_success_input,
        Some(500)
    );
    // A larger success raises it.
    s.record_success("c", "m", 700).await.unwrap();
    assert_eq!(
        s.lookup("c", "m").await.unwrap().unwrap().max_success_input,
        Some(700)
    );
}

#[tokio::test]
async fn overflow_and_success_are_independent_on_one_row() {
    let s = store().await;
    // Success first, then an overflow: the overflow must not wipe the success
    // high-water, and vice-versa.
    s.record_success("c", "m", 5000).await.unwrap();
    s.record_overflow("c", "m", 1000, 4000).await.unwrap();
    let w = s.lookup("c", "m").await.unwrap().unwrap();
    assert_eq!(w.observed_limit, Some(1000));
    assert_eq!(w.configured_window, Some(4000));
    assert_eq!(w.max_success_input, Some(5000));

    // A later success updates only the high-water, leaving the overflow intact.
    s.record_success("c", "m", 6000).await.unwrap();
    let w = s.lookup("c", "m").await.unwrap().unwrap();
    assert_eq!(w.observed_limit, Some(1000));
    assert_eq!(w.max_success_input, Some(6000));
}

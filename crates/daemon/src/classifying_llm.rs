//! Decorator that normalizes opaque backend LLM errors into structured
//! [`CoreError`] variants (epic #178).
//!
//! Connectors surface unrecognized provider failures as `CoreError::Llm(_)` —
//! a bare string the dispatch loop can't act on. This wrapper turns that into
//! the structured variant (`ContextOverflow`, `RateLimited`, `QuotaExceeded`,
//! …) the core recovery ladder and `RetryingLlmClient` already handle, via
//! three tiers:
//!   1. deterministic built-in matchers ([`classify_builtin`]) — pure, no I/O;
//!   2. a learned cache ([`ErrorClassificationStore`]) — local lookup;
//!   3. a cheap classifier LLM that labels genuinely novel errors and persists
//!      the result so tier 2 catches it next time.
//!
//! ## Placement & loop-safety
//! It wraps the **raw connector** (innermost, in `registry::build_llm_client`),
//! so it sees the connector's own error before `Retrying`/recovery upstream.
//! Putting it *inside* `RetryingLlmClient` is deliberate: `is_retryable_error`
//! retries only `RateLimited`, so a billing error mapped to `QuotaExceeded`
//! (terminal) is never retried, and `ContextOverflow` flows to the bounded
//! recovery ladder.
//!
//! The classifier never loops:
//! - tier 1 is pure and single-shot;
//! - tiers 2/3 run at most once per error and are skipped entirely while a
//!   classification call is in flight ([`is_classification_in_progress`]), so
//!   the tier-3 LLM call's own errors can't recurse back into classification;
//! - tier 3 is time-bounded ([`CLASSIFY_TIMEOUT`]) and best-effort — any
//!   failure, timeout, or unusable answer falls back to the original error.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, Role, ToolDefinition, ToolNamespace};
use desktop_assistant_core::error_classify::{
    ErrorContext, NormalizedCause, cause_to_core_error, classify_builtin,
};
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmClient, LlmResponse, ModelInfo, ReasoningConfig, current_context_budget,
    current_model_override, is_classification_in_progress, with_classification_in_progress,
};
use desktop_assistant_core::ports::store::{ErrorClassificationStore, LearnedWindowStore};
use desktop_assistant_core::sanitize::redact_secrets;

/// Per-process dependencies for the learned (tier 2) and LLM (tier 3) tiers.
/// Installed once at daemon startup via [`install_classification_deps`];
/// absent in tests and when no database is configured, in which case only the
/// deterministic tier-1 matchers run.
pub struct ClassificationDeps {
    /// Learned-classification cache (tier 2).
    pub store: Arc<dyn ErrorClassificationStore>,
    /// Cheap LLM used to classify genuinely novel errors (tier 3). `None`
    /// disables tier 3 (tier 2 still runs).
    pub classifier: Option<Arc<dyn LlmClient>>,
    /// Learned context-window cache (issue #343). When a context-overflow
    /// error carries a parsed `max_tokens`, the observed ceiling is persisted
    /// here per `(connector, model)` so the next turn's budget resolution can
    /// cap DOWN to it. `None` disables window learning (classification still
    /// works).
    pub window_store: Option<Arc<dyn LearnedWindowStore>>,
}

static CLASSIFICATION_DEPS: OnceLock<ClassificationDeps> = OnceLock::new();

/// Install the process-wide classification deps. The first call wins; a later
/// one is ignored with a warning.
pub fn install_classification_deps(deps: ClassificationDeps) {
    if CLASSIFICATION_DEPS.set(deps).is_err() {
        tracing::warn!("classification deps already installed; ignoring duplicate");
    }
}

/// Time budget for a single tier-3 classification call. Mirrors the
/// embeddings-timeout philosophy (#195): never hang the user's turn on a
/// best-effort side path — fall back to the original error instead.
const CLASSIFY_TIMEOUT: Duration = Duration::from_secs(5);

/// Minimum length for a learned signature. Below this a substring is too
/// generic to key future errors on, so tier 3 rejects it and the original
/// error surfaces.
const MIN_SIGNATURE_LEN: usize = 8;

/// Wraps an [`LlmClient`] and remaps opaque `CoreError::Llm` errors into the
/// structured variant the classifier recognizes. See module docs.
#[derive(Clone)]
pub struct ClassifyingLlmClient<L> {
    inner: L,
    connector: String,
}

impl<L> ClassifyingLlmClient<L> {
    pub fn new(inner: L, connector: impl Into<String>) -> Self {
        Self {
            inner,
            connector: connector.into(),
        }
    }

    fn ctx<'a>(&'a self, message: &'a str) -> ErrorContext<'a> {
        ErrorContext {
            connector: &self.connector,
            http_status: None,
            provider_code: None,
            message,
        }
    }

    /// Map a recognized cause onto the structured `CoreError`, or surface the
    /// original opaque error when there is no dedicated variant.
    fn apply(&self, cause: NormalizedCause, detail: String) -> Result<LlmResponse, CoreError> {
        match cause_to_core_error(cause, detail.clone()) {
            Some(mapped) => Err(mapped),
            None => Err(CoreError::Llm(detail)),
        }
    }

    /// Production entry point: reclassify using the process-wide deps.
    async fn reclassify(
        &self,
        result: Result<LlmResponse, CoreError>,
    ) -> Result<LlmResponse, CoreError>
    where
        L: LlmClient,
    {
        self.reclassify_with(result, CLASSIFICATION_DEPS.get())
            .await
    }

    /// Reclassification with deps injected explicitly (so tests don't touch the
    /// global `OnceLock`). Tier 1 is pure; tiers 2/3 run only when `deps` is
    /// present and we're not already inside a classification call.
    async fn reclassify_with(
        &self,
        result: Result<LlmResponse, CoreError>,
        deps: Option<&ClassificationDeps>,
    ) -> Result<LlmResponse, CoreError>
    where
        L: LlmClient,
    {
        let detail = match result {
            Err(CoreError::Llm(detail)) => detail,
            other => return other,
        };

        // Tier 1: deterministic, pure — always safe.
        let cause = classify_builtin(&self.ctx(&detail));
        if !matches!(cause, NormalizedCause::Unknown) {
            // Issue #343: when this is a context overflow that surfaced a real
            // provider-reported ceiling, persist it as a DOWN-ONLY learned cap
            // for the next turn. Best-effort and side-channel — never blocks or
            // changes the error that flows on to the recovery ladder.
            self.learn_window(&cause, deps).await;
            return self.apply(cause, detail);
        }

        // Tiers 2/3 need deps and must not run while we're already classifying
        // (reentrancy guard) — otherwise the classifier LLM's own errors would
        // recurse back into classification.
        if is_classification_in_progress() {
            return Err(CoreError::Llm(detail));
        }
        let Some(deps) = deps else {
            return Err(CoreError::Llm(detail));
        };

        // Tier 2: learned cache.
        if let Ok(Some(learned)) = deps.store.lookup(&self.connector, &detail).await
            && let Some(c) = NormalizedCause::from_key(&learned.cause)
        {
            tracing::info!(
                connector = %self.connector,
                cause = %learned.cause,
                "matched learned error classification (tier 2)"
            );
            return self.apply(c, detail);
        }

        // Tier 3: cheap-LLM classification, then persist for next time.
        if let Some(classifier) = deps.classifier.as_deref()
            && let Some((cause, signature)) = self.classify_via_llm(classifier, &detail).await
        {
            if let Err(e) = deps
                .store
                .record(&self.connector, &signature, cause.as_key())
                .await
            {
                tracing::warn!(error = %e, "failed to persist learned classification");
            }
            tracing::info!(
                connector = %self.connector,
                ?cause,
                "classified opaque LLM error via the LLM tier (tier 3)"
            );
            return self.apply(cause, detail);
        }

        Err(CoreError::Llm(detail))
    }

    /// Issue #343: persist an observed context-overflow ceiling as a DOWN-ONLY
    /// learned cap keyed by `(connector, model)`, tagged with the effective
    /// configured window that was in force this turn (for invalidation).
    ///
    /// Best-effort and defensive:
    /// - only fires for `ContextOverflow { max_tokens: Some(n) }` — no parsed
    ///   ceiling means nothing to learn (`max_tokens: None` persists nothing);
    /// - skipped while a classification is in progress, so the tier-3
    ///   classifier's own overflow can't be mislearned (and its model/budget
    ///   task-locals would be wrong);
    /// - requires both a window store and a live `current_context_budget()` —
    ///   without the in-flight budget we have nothing to key invalidation on;
    /// - the persisted value is the raw provider ceiling; the sanity-floor and
    ///   down-only/invalidation rules are enforced at apply time
    ///   (`apply_learned_cap`) and in the store's ratchet, so a garbage parse
    ///   can never pin the budget even if it is written.
    async fn learn_window(&self, cause: &NormalizedCause, deps: Option<&ClassificationDeps>)
    where
        L: LlmClient,
    {
        let NormalizedCause::ContextOverflow {
            max_tokens: Some(observed),
            ..
        } = cause
        else {
            return;
        };
        if is_classification_in_progress() {
            return;
        }
        let Some(store) = deps.and_then(|d| d.window_store.as_deref()) else {
            return;
        };
        // The effective configured window in force for this turn — the key for
        // invalidation. Without it we can't tell a future config bump from the
        // stale observation, so we decline to learn.
        let Some(budget) = current_context_budget() else {
            return;
        };
        let model = current_model_override()
            .or_else(|| self.inner.get_default_model().map(str::to_string))
            .unwrap_or_default();

        if let Err(e) = store
            .record(&self.connector, &model, *observed, budget.max_input_tokens)
            .await
        {
            tracing::warn!(error = %e, "failed to persist learned context window");
        } else {
            tracing::info!(
                connector = %self.connector,
                model = %model,
                observed_limit = *observed,
                configured_window = budget.max_input_tokens,
                "learned observed context-window ceiling (#343)"
            );
        }
    }

    /// Tier 3: ask the cheap classifier LLM to label this error. Best-effort:
    /// secret-scrubbed, time-bounded, and wrapped in the reentrancy guard so
    /// the classifier's own errors can't recurse. Returns the cause plus a
    /// signature substring (validated to occur in the original message), or
    /// `None` on any failure / unusable answer.
    async fn classify_via_llm(
        &self,
        classifier: &dyn LlmClient,
        detail: &str,
    ) -> Option<(NormalizedCause, String)> {
        let prompt = build_classifier_prompt(&self.connector, &redact_secrets(detail));
        let call = with_classification_in_progress(classifier.stream_completion(
            vec![Message::new(Role::User, prompt)],
            &[],
            ReasoningConfig::default(),
            Box::new(|_| true),
        ));
        let resp = match tokio::time::timeout(CLASSIFY_TIMEOUT, call).await {
            Ok(Ok(resp)) => resp,
            Ok(Err(e)) => {
                tracing::debug!(error = %e, "tier-3 classification LLM call failed");
                return None;
            }
            Err(_) => {
                tracing::debug!("tier-3 classification timed out");
                return None;
            }
        };
        parse_classifier_response(&resp.text, detail)
    }
}

/// Build the tier-3 classifier prompt. `message` must already be
/// secret-scrubbed.
fn build_classifier_prompt(connector: &str, message: &str) -> String {
    format!(
        "You classify errors returned by the \"{connector}\" LLM backend so an automated \
         system can react. Reply with ONLY a JSON object and nothing else:\n\
         {{\"cause\": \"<one of: context_overflow, rate_limited, billing_fatal, auth, \
         model_loading, tools_unsupported, transient, unknown>\", \
         \"signature\": \"<a short distinctive substring copied verbatim from the error that \
         identifies this class of error>\"}}\n\
         Use \"billing_fatal\" for quota/billing/credit exhaustion, \"auth\" for \
         authentication or permission failures, and \"unknown\" if unsure. The signature must \
         be an exact substring of the error and must not contain secrets.\n\n\
         Error: {message}"
    )
}

/// Parse the tier-3 response into `(cause, signature)`. Rejects `unknown`, a
/// too-short signature, or a signature that doesn't actually occur in the
/// original (unredacted) message — which would make the learned entry
/// unmatchable and risks the LLM inventing an over-broad pattern.
fn parse_classifier_response(text: &str, original: &str) -> Option<(NormalizedCause, String)> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end < start {
        return None;
    }

    #[derive(serde::Deserialize)]
    struct Parsed {
        cause: String,
        signature: String,
    }
    let parsed: Parsed = serde_json::from_str(text.get(start..=end)?).ok()?;

    let cause = NormalizedCause::from_key(parsed.cause.trim())?;
    if matches!(cause, NormalizedCause::Unknown) {
        return None;
    }
    let signature = parsed.signature.trim();
    if signature.len() < MIN_SIGNATURE_LEN
        || !original
            .to_ascii_lowercase()
            .contains(&signature.to_ascii_lowercase())
    {
        return None;
    }
    Some((cause, signature.to_string()))
}

#[async_trait::async_trait]
impl<L: LlmClient> LlmClient for ClassifyingLlmClient<L> {
    // --- Faithful passthrough: every capability comes from `inner`. The
    // decorator wraps the connector innermost, so masking any of these (e.g.
    // `max_context_tokens`, which drives budget resolution) would regress
    // behavior. Only the two completion methods are intercepted. ---

    fn get_default_model(&self) -> Option<&str> {
        self.inner.get_default_model()
    }

    fn get_default_base_url(&self) -> Option<&str> {
        self.inner.get_default_base_url()
    }

    fn max_context_tokens(&self) -> Option<u64> {
        self.inner.max_context_tokens()
    }

    fn estimate_tokens(&self, text: &str) -> u64 {
        self.inner.estimate_tokens(text)
    }

    fn supports_hosted_tool_search(&self) -> bool {
        self.inner.supports_hosted_tool_search()
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        self.inner.list_models().await
    }

    async fn refresh_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        self.inner.refresh_models().await
    }

    async fn warmup(&self) {
        self.inner.warmup().await
    }

    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let result = self
            .inner
            .stream_completion(messages, tools, reasoning, on_chunk)
            .await;
        self.reclassify(result).await
    }

    async fn stream_completion_with_namespaces(
        &self,
        messages: Vec<Message>,
        core_tools: &[ToolDefinition],
        namespaces: &[ToolNamespace],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let result = self
            .inner
            .stream_completion_with_namespaces(
                messages, core_tools, namespaces, reasoning, on_chunk,
            )
            .await;
        self.reclassify(result).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::ports::store::LearnedClassification;
    use std::sync::Mutex;

    enum Behavior {
        Ok,
        ErrLlm(String),
        ErrRateLimited,
    }

    struct StubClient {
        calls: Mutex<u32>,
        behavior: Behavior,
        max_ctx: Option<u64>,
    }

    impl StubClient {
        fn new(behavior: Behavior) -> Self {
            Self {
                calls: Mutex::new(0),
                behavior,
                max_ctx: Some(131_072),
            }
        }
        fn calls(&self) -> u32 {
            *self.calls.lock().unwrap()
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for StubClient {
        fn max_context_tokens(&self) -> Option<u64> {
            self.max_ctx
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
            Ok(vec![])
        }

        async fn refresh_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
            Ok(vec![])
        }

        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            _on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            *self.calls.lock().unwrap() += 1;
            match &self.behavior {
                Behavior::Ok => Ok(LlmResponse {
                    text: "ok".into(),
                    tool_calls: vec![],
                    usage: None,
                }),
                Behavior::ErrLlm(s) => Err(CoreError::Llm(s.clone())),
                Behavior::ErrRateLimited => Err(CoreError::RateLimited {
                    retry_after: None,
                    detail: "throttled".into(),
                }),
            }
        }

        fn supports_hosted_tool_search(&self) -> bool {
            false
        }
    }

    /// Tier-3 classifier double: returns canned text and counts calls.
    struct MockClassifier {
        text: String,
        ok: bool,
        calls: Mutex<u32>,
    }
    impl MockClassifier {
        fn ok(text: &str) -> Self {
            Self {
                text: text.into(),
                ok: true,
                calls: Mutex::new(0),
            }
        }
        fn failing() -> Self {
            Self {
                text: String::new(),
                ok: false,
                calls: Mutex::new(0),
            }
        }
        fn calls(&self) -> u32 {
            *self.calls.lock().unwrap()
        }
    }
    #[async_trait::async_trait]
    impl LlmClient for MockClassifier {
        async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
            Ok(vec![])
        }
        async fn refresh_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
            Ok(vec![])
        }
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            _on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            *self.calls.lock().unwrap() += 1;
            if self.ok {
                Ok(LlmResponse {
                    text: self.text.clone(),
                    tool_calls: vec![],
                    usage: None,
                })
            } else {
                Err(CoreError::Llm("classifier exploded".into()))
            }
        }
        fn supports_hosted_tool_search(&self) -> bool {
            false
        }
    }

    /// Tier-2 store double: a single canned lookup result + a record log.
    struct MockStore {
        lookup: Option<LearnedClassification>,
        recorded: Mutex<Vec<(String, String, String)>>,
    }
    impl MockStore {
        fn new(lookup: Option<LearnedClassification>) -> Self {
            Self {
                lookup,
                recorded: Mutex::new(vec![]),
            }
        }
        fn recorded(&self) -> Vec<(String, String, String)> {
            self.recorded.lock().unwrap().clone()
        }
    }
    #[async_trait::async_trait]
    impl ErrorClassificationStore for MockStore {
        async fn lookup(
            &self,
            _connector: &str,
            _message: &str,
        ) -> Result<Option<LearnedClassification>, CoreError> {
            Ok(self.lookup.clone())
        }
        async fn record(
            &self,
            connector: &str,
            signature: &str,
            cause: &str,
        ) -> Result<(), CoreError> {
            self.recorded
                .lock()
                .unwrap()
                .push((connector.into(), signature.into(), cause.into()));
            Ok(())
        }
    }

    fn wrapped(behavior: Behavior) -> ClassifyingLlmClient<StubClient> {
        ClassifyingLlmClient::new(StubClient::new(behavior), "bedrock")
    }

    async fn run(c: &ClassifyingLlmClient<StubClient>) -> Result<LlmResponse, CoreError> {
        c.stream_completion(
            vec![Message::new(Role::User, "hi")],
            &[],
            ReasoningConfig::default(),
            Box::new(|_| true),
        )
        .await
    }

    // --- Tier 1 (no deps) ---

    #[tokio::test]
    async fn opaque_overflow_error_is_reclassified() {
        let c = wrapped(Behavior::ErrLlm(
            "Input is too long for requested model.".into(),
        ));
        let err = run(&c).await.expect_err("must be an error");
        assert!(
            matches!(err, CoreError::ContextOverflow { .. }),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn opaque_billing_error_maps_to_quota_exceeded() {
        let c = wrapped(Behavior::ErrLlm(
            "You exceeded your current quota; check your billing details.".into(),
        ));
        let err = run(&c).await.expect_err("must be an error");
        assert!(
            matches!(err, CoreError::QuotaExceeded { .. }),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn unrecognized_opaque_error_passes_through_unchanged() {
        let c = wrapped(Behavior::ErrLlm("a wild unfamiliar failure".into()));
        let err = run(&c).await.expect_err("must be an error");
        match err {
            CoreError::Llm(s) => assert_eq!(s, "a wild unfamiliar failure"),
            other => panic!("expected unchanged Llm error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn already_structured_error_is_not_touched() {
        let c = wrapped(Behavior::ErrRateLimited);
        let err = run(&c).await.expect_err("must be an error");
        assert!(matches!(err, CoreError::RateLimited { .. }), "got {err:?}");
    }

    #[tokio::test]
    async fn ok_passes_through() {
        let c = wrapped(Behavior::Ok);
        assert!(run(&c).await.is_ok());
    }

    #[tokio::test]
    async fn reclassification_is_single_shot_no_loop() {
        let c = wrapped(Behavior::ErrLlm(
            "Input length (479258) exceeds model's maximum context length (131072).".into(),
        ));
        let _ = run(&c).await;
        assert_eq!(
            c.inner.calls(),
            1,
            "inner client must be called exactly once"
        );
    }

    #[test]
    fn delegates_max_context_tokens_to_inner() {
        let c = wrapped(Behavior::Ok);
        assert_eq!(c.max_context_tokens(), Some(131_072));
    }

    // --- Tier 2 (learned cache) ---

    #[tokio::test]
    async fn tier2_learned_cache_hit_is_applied() {
        let deps = ClassificationDeps {
            store: Arc::new(MockStore::new(Some(LearnedClassification {
                signature: "kaboom".into(),
                cause: "billing_fatal".into(),
            }))),
            classifier: None,
            window_store: None,
        };
        let c = wrapped(Behavior::Ok);
        let err = c
            .reclassify_with(
                Err(CoreError::Llm("novel kaboom happened".into())),
                Some(&deps),
            )
            .await
            .expect_err("must be an error");
        assert!(
            matches!(err, CoreError::QuotaExceeded { .. }),
            "got {err:?}"
        );
    }

    // --- Tier 3 (LLM) ---

    #[tokio::test]
    async fn tier3_llm_classifies_and_persists() {
        let classifier = Arc::new(MockClassifier::ok(
            "{\"cause\": \"billing_fatal\", \"signature\": \"dunning notice\"}",
        ));
        let store = Arc::new(MockStore::new(None));
        let deps = ClassificationDeps {
            store: store.clone(),
            classifier: Some(classifier.clone()),
            window_store: None,
        };
        let c = wrapped(Behavior::Ok);
        let err = c
            .reclassify_with(
                Err(CoreError::Llm(
                    "a mysterious dunning notice from the provider".into(),
                )),
                Some(&deps),
            )
            .await
            .expect_err("must be an error");
        assert!(
            matches!(err, CoreError::QuotaExceeded { .. }),
            "got {err:?}"
        );
        assert_eq!(classifier.calls(), 1);
        assert_eq!(
            store.recorded(),
            vec![(
                "bedrock".into(),
                "dunning notice".into(),
                "billing_fatal".into()
            )]
        );
    }

    #[tokio::test]
    async fn tier3_rejects_signature_absent_from_message() {
        // The LLM hallucinates a signature not present in the error; we must
        // reject it (no remap, nothing persisted).
        let classifier = Arc::new(MockClassifier::ok(
            "{\"cause\": \"billing_fatal\", \"signature\": \"not in the message\"}",
        ));
        let store = Arc::new(MockStore::new(None));
        let deps = ClassificationDeps {
            store: store.clone(),
            classifier: Some(classifier.clone()),
            window_store: None,
        };
        let c = wrapped(Behavior::Ok);
        let err = c
            .reclassify_with(
                Err(CoreError::Llm("some opaque failure".into())),
                Some(&deps),
            )
            .await
            .expect_err("must be an error");
        assert!(matches!(err, CoreError::Llm(_)), "got {err:?}");
        assert!(store.recorded().is_empty());
    }

    #[tokio::test]
    async fn tier3_classifier_failure_surfaces_original_no_loop() {
        let classifier = Arc::new(MockClassifier::failing());
        let store = Arc::new(MockStore::new(None));
        let deps = ClassificationDeps {
            store: store.clone(),
            classifier: Some(classifier.clone()),
            window_store: None,
        };
        let c = wrapped(Behavior::Ok);
        let err = c
            .reclassify_with(Err(CoreError::Llm("opaque boom".into())), Some(&deps))
            .await
            .expect_err("must be an error");
        match err {
            CoreError::Llm(s) => assert_eq!(s, "opaque boom"),
            other => panic!("expected original error, got {other:?}"),
        }
        assert_eq!(classifier.calls(), 1, "classifier tried exactly once");
        assert!(store.recorded().is_empty());
    }

    #[tokio::test]
    async fn reentrancy_guard_skips_tiers_2_and_3() {
        // While a classification is in progress, an opaque error must NOT
        // trigger another cache lookup or LLM call — that is the loop break.
        let classifier = Arc::new(MockClassifier::ok(
            "{\"cause\": \"billing_fatal\", \"signature\": \"dunning notice\"}",
        ));
        let store = Arc::new(MockStore::new(Some(LearnedClassification {
            signature: "boom".into(),
            cause: "billing_fatal".into(),
        })));
        let deps = ClassificationDeps {
            store: store.clone(),
            classifier: Some(classifier.clone()),
            window_store: None,
        };
        let c = wrapped(Behavior::Ok);
        let err = with_classification_in_progress(
            c.reclassify_with(Err(CoreError::Llm("opaque boom".into())), Some(&deps)),
        )
        .await
        .expect_err("must be an error");
        assert!(matches!(err, CoreError::Llm(_)), "got {err:?}");
        assert_eq!(classifier.calls(), 0, "tier 3 must be skipped");
        assert!(store.recorded().is_empty());
    }

    // --- Window learning (issue #343) ----------------------------------------

    use desktop_assistant_core::ports::llm::{
        BudgetSource, ContextBudget, with_context_budget, with_model_override,
    };
    use desktop_assistant_core::ports::store::LearnedWindow;

    /// Learned-window store double: records every `record()` call.
    struct MockWindowStore {
        recorded: Mutex<Vec<(String, String, u64, u64)>>,
    }
    impl MockWindowStore {
        fn new() -> Self {
            Self {
                recorded: Mutex::new(vec![]),
            }
        }
        fn recorded(&self) -> Vec<(String, String, u64, u64)> {
            self.recorded.lock().unwrap().clone()
        }
    }
    #[async_trait::async_trait]
    impl LearnedWindowStore for MockWindowStore {
        async fn lookup(
            &self,
            _connector: &str,
            _model: &str,
        ) -> Result<Option<LearnedWindow>, CoreError> {
            Ok(None)
        }
        async fn record(
            &self,
            connector: &str,
            model: &str,
            observed_limit: u64,
            configured_window: u64,
        ) -> Result<(), CoreError> {
            self.recorded.lock().unwrap().push((
                connector.into(),
                model.into(),
                observed_limit,
                configured_window,
            ));
            Ok(())
        }
    }

    fn budget(max: u64) -> ContextBudget {
        ContextBudget {
            max_input_tokens: max,
            source: BudgetSource::ConnectorTable,
        }
    }

    #[tokio::test]
    async fn overflow_with_max_tokens_persists_observed_window() {
        // A `ContextOverflow { max_tokens: Some(n) }` persists `n` for the
        // in-flight `(connector, model)`, keyed by the effective configured
        // window in force this turn.
        let window = Arc::new(MockWindowStore::new());
        let deps = ClassificationDeps {
            store: Arc::new(MockStore::new(None)),
            classifier: None,
            window_store: Some(window.clone()),
        };
        let c = wrapped(Behavior::Ok);
        let _ = with_context_budget(
            budget(8_192),
            with_model_override("qwen2.5".to_string(), async {
                c.reclassify_with(
                    Err(CoreError::Llm(
                        "Input length (479258) exceeds maximum context length (4096).".into(),
                    )),
                    Some(&deps),
                )
                .await
            }),
        )
        .await;
        assert_eq!(
            window.recorded(),
            vec![("bedrock".into(), "qwen2.5".into(), 4_096, 8_192)],
            "observed ceiling + configured window must be persisted"
        );
    }

    #[tokio::test]
    async fn overflow_without_max_tokens_persists_nothing() {
        // No parsed ceiling → nothing to learn. (The overflow phrasing here
        // carries no two numbers, so `max_tokens` is `None`.)
        let window = Arc::new(MockWindowStore::new());
        let deps = ClassificationDeps {
            store: Arc::new(MockStore::new(None)),
            classifier: None,
            window_store: Some(window.clone()),
        };
        let c = wrapped(Behavior::Ok);
        let err = with_context_budget(
            budget(8_192),
            with_model_override("qwen2.5".to_string(), async {
                c.reclassify_with(
                    Err(CoreError::Llm("prompt is too long".into())),
                    Some(&deps),
                )
                .await
            }),
        )
        .await
        .expect_err("must be an error");
        assert!(
            matches!(err, CoreError::ContextOverflow { .. }),
            "got {err:?}"
        );
        assert!(
            window.recorded().is_empty(),
            "max_tokens=None must persist nothing"
        );
    }

    #[tokio::test]
    async fn window_not_learned_without_a_budget_in_flight() {
        // Without `current_context_budget()` there is nothing to key
        // invalidation on, so we decline to learn rather than store a bogus
        // configured_window.
        let window = Arc::new(MockWindowStore::new());
        let deps = ClassificationDeps {
            store: Arc::new(MockStore::new(None)),
            classifier: None,
            window_store: Some(window.clone()),
        };
        let c = wrapped(Behavior::Ok);
        // No `with_context_budget` wrapper.
        let _ = with_model_override("qwen2.5".to_string(), async {
            c.reclassify_with(
                Err(CoreError::Llm(
                    "Input length (479258) exceeds maximum context length (4096).".into(),
                )),
                Some(&deps),
            )
            .await
        })
        .await;
        assert!(window.recorded().is_empty());
    }

    #[tokio::test]
    async fn non_overflow_error_does_not_learn_window() {
        // A rate-limit (or any non-overflow) cause never touches the window
        // store.
        let window = Arc::new(MockWindowStore::new());
        let deps = ClassificationDeps {
            store: Arc::new(MockStore::new(None)),
            classifier: None,
            window_store: Some(window.clone()),
        };
        let c = wrapped(Behavior::Ok);
        let _ = with_context_budget(
            budget(8_192),
            with_model_override("qwen2.5".to_string(), async {
                c.reclassify_with(
                    Err(CoreError::Llm("rate limit exceeded, slow down".into())),
                    Some(&deps),
                )
                .await
            }),
        )
        .await;
        assert!(window.recorded().is_empty());
    }

    #[tokio::test]
    async fn window_not_learned_while_classification_in_progress() {
        // The tier-3 classifier's OWN overflow must not be mislearned (and its
        // model/budget task-locals would be wrong).
        let window = Arc::new(MockWindowStore::new());
        let deps = ClassificationDeps {
            store: Arc::new(MockStore::new(None)),
            classifier: None,
            window_store: Some(window.clone()),
        };
        let c = wrapped(Behavior::Ok);
        let _ = with_context_budget(
            budget(8_192),
            with_model_override(
                "qwen2.5".to_string(),
                with_classification_in_progress(c.reclassify_with(
                    Err(CoreError::Llm(
                        "Input length (479258) exceeds maximum context length (4096).".into(),
                    )),
                    Some(&deps),
                )),
            ),
        )
        .await;
        assert!(window.recorded().is_empty());
    }
}

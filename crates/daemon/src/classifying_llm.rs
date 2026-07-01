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
    ErrorContext, NormalizedCause, OverflowFields, cause_to_core_error, classify_builtin,
    derive_input_ceiling, extract_overflow_fields,
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
    /// Learned context-window cache (issues #343/#425). A derived input ceiling
    /// from a context-overflow error, and the success high-water mark from
    /// completed turns, are persisted here per `(connector, model)` so the next
    /// turn's budget resolution can cap DOWN (and recover). `None` disables
    /// window learning (classification still works).
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

/// Floor for recording a success high-water mark (issue #425). A prompt smaller
/// than this — or smaller than half the configured budget — tells us nothing
/// useful about the ceiling, so we skip the DB write. Also the fallback gate
/// when no per-turn budget is installed (background jobs).
const SUCCESS_HIGH_WATER_MIN_TOKENS: u64 = 8_192;

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
            Ok(resp) => {
                // Issue #425: a successful call is a data point that this model
                // ACCEPTED `usage.input_tokens` — record it as the success
                // high-water mark that floors the learned cap and lets the
                // budget recover. Best-effort and side-channel.
                self.learn_success(&resp, deps).await;
                return Ok(resp);
            }
            other => return other,
        };

        // Tier 1: deterministic, pure — always safe.
        let cause = classify_builtin(&self.ctx(&detail));
        if !matches!(cause, NormalizedCause::Unknown) {
            // Issue #343/#425: when this is a context overflow, derive the real
            // input ceiling from the error and persist it as a DOWN-ONLY learned
            // cap for the next turn. Best-effort and side-channel — never blocks
            // or changes the error that flows on to the recovery ladder.
            self.learn_window(&cause, &detail, deps).await;
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

    /// Issue #343/#425: derive an input-token ceiling from a context-overflow
    /// error and persist it as a DOWN-ONLY learned cap keyed by
    /// `(connector, model)`, tagged with the configured window in force this
    /// turn (for invalidation).
    ///
    /// The ceiling comes from keyword-anchored extraction
    /// ([`extract_overflow_fields`]) first; when that can't yield a number the
    /// fuzzy LLM extractor backfills it (the "let the LLM decipher it" path).
    /// [`derive_input_ceiling`] then turns the numbers into an input budget
    /// (subtracting the output reservation).
    ///
    /// Best-effort and defensive:
    /// - only fires for `ContextOverflow`;
    /// - skipped while a classification is in progress, so the tier-3
    ///   classifier's own overflow can't be mislearned (and its model/budget
    ///   task-locals would be wrong);
    /// - requires both a window store and a live `current_context_budget()` —
    ///   without the in-flight budget we have nothing to key invalidation on;
    /// - the persisted value is un-snapped; snapping, the success floor, and
    ///   the down-only/invalidation rules are enforced at apply time
    ///   (`apply_learned_cap`) and in the store's ratchet, so a garbage parse
    ///   can never pin the budget even if it is written.
    async fn learn_window(
        &self,
        cause: &NormalizedCause,
        detail: &str,
        deps: Option<&ClassificationDeps>,
    ) where
        L: LlmClient,
    {
        if !matches!(cause, NormalizedCause::ContextOverflow { .. }) {
            return;
        }
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

        // Deterministic keyword extraction first; only pay for the LLM extractor
        // when the message is a phrasing we can't anchor (issue #425).
        let mut fields = extract_overflow_fields(detail);
        if derive_input_ceiling(&fields).is_none()
            && let Some(classifier) = deps.and_then(|d| d.classifier.as_deref())
            && let Some(llm_fields) = self.extract_overflow_via_llm(classifier, detail).await
        {
            fields = fields.or(llm_fields);
        }
        let Some(ceiling) = derive_input_ceiling(&fields) else {
            return; // no usable number in the message — nothing to learn
        };

        let model = current_model_override()
            .or_else(|| self.inner.get_default_model().map(str::to_string))
            .unwrap_or_default();

        if let Err(e) = store
            .record_overflow(&self.connector, &model, ceiling, budget.max_input_tokens)
            .await
        {
            tracing::warn!(error = %e, "failed to persist learned context window");
        } else {
            tracing::info!(
                connector = %self.connector,
                model = %model,
                observed_limit = ceiling,
                configured_window = budget.max_input_tokens,
                "learned observed context-window ceiling (#343/#425)"
            );
        }
    }

    /// Issue #425: record the largest provider-measured input-token count this
    /// model has ACCEPTED as the success high-water mark.
    ///
    /// Best-effort and side-channel:
    /// - skipped while classifying, so the tier-3 classifier's own calls (and
    ///   any background classification traffic) aren't miscounted;
    /// - requires the connector to report `usage.input_tokens` (measured, not
    ///   estimated); a connector that omits usage records nothing;
    /// - only records prompts large enough to be a useful floor — a tiny prompt
    ///   succeeding says nothing about the ceiling, and we don't want a DB write
    ///   on every small turn. The gate is half the configured budget (or, absent
    ///   a budget, the smallest ladder rung).
    async fn learn_success(&self, resp: &LlmResponse, deps: Option<&ClassificationDeps>)
    where
        L: LlmClient,
    {
        if is_classification_in_progress() {
            return;
        }
        let Some(store) = deps.and_then(|d| d.window_store.as_deref()) else {
            return;
        };
        let Some(input_tokens) = resp.usage.as_ref().and_then(|u| u.input_tokens) else {
            return;
        };
        let floor = current_context_budget()
            .map(|b| b.max_input_tokens / 2)
            .unwrap_or(SUCCESS_HIGH_WATER_MIN_TOKENS);
        if input_tokens < floor.max(SUCCESS_HIGH_WATER_MIN_TOKENS) {
            return;
        }
        let model = current_model_override()
            .or_else(|| self.inner.get_default_model().map(str::to_string))
            .unwrap_or_default();
        if let Err(e) = store
            .record_success(&self.connector, &model, input_tokens)
            .await
        {
            tracing::warn!(error = %e, "failed to persist context-window success high-water");
        }
    }

    /// Issue #425: fuzzy fallback that asks the classifier LLM to extract the
    /// overflow numbers from a phrasing the deterministic parser couldn't
    /// anchor. Strict JSON, nullable per field. Best-effort: secret-scrubbed,
    /// time-bounded, reentrancy-guarded; any failure yields `None` and the
    /// caller simply learns nothing this turn.
    async fn extract_overflow_via_llm(
        &self,
        classifier: &dyn LlmClient,
        detail: &str,
    ) -> Option<OverflowFields>
    where
        L: LlmClient,
    {
        let prompt = build_overflow_extraction_prompt(&redact_secrets(detail));
        let call = with_classification_in_progress(classifier.stream_completion(
            vec![Message::new(Role::User, prompt)],
            &[],
            ReasoningConfig::default(),
            Box::new(|_| true),
        ));
        let resp = match tokio::time::timeout(CLASSIFY_TIMEOUT, call).await {
            Ok(Ok(resp)) => resp,
            Ok(Err(e)) => {
                tracing::debug!(error = %e, "tier-3 overflow extraction LLM call failed");
                return None;
            }
            Err(_) => {
                tracing::debug!("tier-3 overflow extraction timed out");
                return None;
            }
        };
        parse_overflow_extraction_response(&resp.text)
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

/// Build the tier-3 overflow-number extraction prompt (issue #425). `message`
/// must already be secret-scrubbed. Asks for strict JSON with nullable numeric
/// fields so the model reports only what the error actually states.
fn build_overflow_extraction_prompt(message: &str) -> String {
    format!(
        "A context-window overflow error from an LLM backend is below. Extract the token \
         numbers it states. Reply with ONLY a JSON object and nothing else:\n\
         {{\"max_context_tokens\": <the model's total context window in tokens, or null>, \
         \"prompt_tokens\": <the input/prompt token count, or null>, \
         \"requested_output_tokens\": <the requested/reserved output tokens, or null>}}\n\
         Use null for any field the error does not state. Copy the integers exactly; do not \
         guess, round, or infer a number that is not written in the error.\n\n\
         Error: {message}"
    )
}

/// Parse the tier-3 overflow-extraction response into [`OverflowFields`].
/// Tolerates JSON nulls and missing keys (both become `None`); returns `None`
/// only when there is no parseable JSON object at all.
fn parse_overflow_extraction_response(text: &str) -> Option<OverflowFields> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end < start {
        return None;
    }

    #[derive(serde::Deserialize)]
    struct Parsed {
        #[serde(default)]
        max_context_tokens: Option<u64>,
        #[serde(default)]
        prompt_tokens: Option<u64>,
        #[serde(default)]
        requested_output_tokens: Option<u64>,
    }
    let parsed: Parsed = serde_json::from_str(text.get(start..=end)?).ok()?;
    Some(OverflowFields {
        max_context_tokens: parsed.max_context_tokens,
        prompt_tokens: parsed.prompt_tokens,
        requested_output_tokens: parsed.requested_output_tokens,
    })
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

    /// Learned-window store double: records every `record_overflow` and
    /// `record_success` call so tests can assert what was learned.
    struct MockWindowStore {
        recorded: Mutex<Vec<(String, String, u64, u64)>>,
        successes: Mutex<Vec<(String, String, u64)>>,
    }
    impl MockWindowStore {
        fn new() -> Self {
            Self {
                recorded: Mutex::new(vec![]),
                successes: Mutex::new(vec![]),
            }
        }
        fn recorded(&self) -> Vec<(String, String, u64, u64)> {
            self.recorded.lock().unwrap().clone()
        }
        fn successes(&self) -> Vec<(String, String, u64)> {
            self.successes.lock().unwrap().clone()
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
        async fn record_overflow(
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
        async fn record_success(
            &self,
            connector: &str,
            model: &str,
            input_tokens: u64,
        ) -> Result<(), CoreError> {
            self.successes
                .lock()
                .unwrap()
                .push((connector.into(), model.into(), input_tokens));
            Ok(())
        }
    }

    fn budget(max: u64) -> ContextBudget {
        ContextBudget {
            max_input_tokens: max,
            source: BudgetSource::ConnectorTable,
        }
    }

    /// A successful `LlmResponse` reporting `input_tokens` in its usage (issue
    /// #425 success high-water tests).
    fn make_ok_response(input_tokens: Option<u64>) -> LlmResponse {
        LlmResponse {
            text: "ok".into(),
            tool_calls: vec![],
            usage: Some(desktop_assistant_core::ports::llm::TokenUsage {
                input_tokens,
                output_tokens: Some(10),
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            }),
        }
    }

    #[tokio::test]
    async fn overflow_persists_derived_input_ceiling_not_request_id_digits() {
        // Issue #425: the real Bedrock/Mantle error whose requestId UUID
        // (`f2e534ff`) poisoned the old positional parse to 534. We must now
        // persist the DERIVED input ceiling (202752 window − 8192 output =
        // 194560), keyed by the configured window in force this turn.
        let window = Arc::new(MockWindowStore::new());
        let deps = ClassificationDeps {
            store: Arc::new(MockStore::new(None)),
            classifier: None,
            window_store: Some(window.clone()),
        };
        let c = wrapped(Behavior::Ok);
        let msg = "Mantle streaming error for requestId f2e534ff-436e-461b-8d93-906629545d84: \
                   This model's maximum context length is 202752 tokens. However, you \
                   requested 8192 output tokens and your prompt contains at least 194561 \
                   input tokens, for a total of at least 202753 tokens.";
        let _ = with_context_budget(
            budget(200_000),
            with_model_override("zai.glm-5".to_string(), async {
                c.reclassify_with(Err(CoreError::Llm(msg.into())), Some(&deps))
                    .await
            }),
        )
        .await;
        assert_eq!(
            window.recorded(),
            vec![(
                "bedrock".into(),
                "zai.glm-5".into(),
                202_752 - 8_192,
                200_000
            )],
            "must persist the derived input ceiling, not the requestId digits"
        );
    }

    #[tokio::test]
    async fn overflow_without_numbers_persists_nothing() {
        // No number anywhere in the message → nothing to derive, nothing to
        // learn (and no classifier configured to fall back to).
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
            "no derivable number must persist nothing"
        );
    }

    #[test]
    fn overflow_extraction_response_parses_nulls_and_missing_as_none() {
        // Nulls and absent keys both become None (issue #425: "null if it can't
        // find the field"); prose around the JSON is tolerated.
        let f = parse_overflow_extraction_response(
            "Here you go: {\"max_context_tokens\": 200000, \"prompt_tokens\": null}",
        )
        .expect("parses");
        assert_eq!(f.max_context_tokens, Some(200_000));
        assert_eq!(f.prompt_tokens, None);
        assert_eq!(f.requested_output_tokens, None);
        // No JSON at all → None.
        assert!(parse_overflow_extraction_response("sorry, I can't tell").is_none());
    }

    #[tokio::test]
    async fn overflow_falls_back_to_llm_extraction_for_novel_phrasing() {
        // Issue #425 "let the LLM decipher it": a phrasing the keyword parser
        // can't anchor ("too many tokens" with no adjacent numbers) still
        // classifies as overflow, so we ask the classifier LLM for the numbers
        // and derive the ceiling from its JSON (128000 − 4096 = 123904).
        let classifier = Arc::new(MockClassifier::ok(
            "{\"max_context_tokens\": 128000, \"prompt_tokens\": 150000, \
             \"requested_output_tokens\": 4096}",
        ));
        let window = Arc::new(MockWindowStore::new());
        let deps = ClassificationDeps {
            store: Arc::new(MockStore::new(None)),
            classifier: Some(classifier.clone()),
            window_store: Some(window.clone()),
        };
        let c = wrapped(Behavior::Ok);
        let _ = with_context_budget(
            budget(200_000),
            with_model_override("mystery-model".to_string(), async {
                c.reclassify_with(
                    Err(CoreError::Llm(
                        "Request rejected: too many tokens for this model.".into(),
                    )),
                    Some(&deps),
                )
                .await
            }),
        )
        .await;
        assert_eq!(classifier.calls(), 1, "the LLM extractor must be consulted");
        assert_eq!(
            window.recorded(),
            vec![(
                "bedrock".into(),
                "mystery-model".into(),
                128_000 - 4_096,
                200_000
            )],
            "the derived ceiling from the LLM's numbers must be persisted"
        );
    }

    #[tokio::test]
    async fn successful_turn_records_high_water_mark() {
        // Issue #425: a completed call reports `usage.input_tokens`; when it's a
        // meaningful fraction of the budget it's recorded as the success
        // high-water mark for `(connector, model)`.
        let window = Arc::new(MockWindowStore::new());
        let deps = ClassificationDeps {
            store: Arc::new(MockStore::new(None)),
            classifier: None,
            window_store: Some(window.clone()),
        };
        let c = wrapped(Behavior::Ok);
        let _ = with_context_budget(
            budget(200_000),
            with_model_override("zai.glm-5".to_string(), async {
                c.reclassify_with(Ok(make_ok_response(Some(180_000))), Some(&deps))
                    .await
            }),
        )
        .await;
        assert_eq!(
            window.successes(),
            vec![("bedrock".into(), "zai.glm-5".into(), 180_000)],
        );
        assert!(window.recorded().is_empty(), "no overflow on a success");
    }

    #[tokio::test]
    async fn small_successful_turn_skips_high_water_write() {
        // A prompt well under half the budget says nothing about the ceiling and
        // must not incur a DB write.
        let window = Arc::new(MockWindowStore::new());
        let deps = ClassificationDeps {
            store: Arc::new(MockStore::new(None)),
            classifier: None,
            window_store: Some(window.clone()),
        };
        let c = wrapped(Behavior::Ok);
        let _ = with_context_budget(
            budget(200_000),
            with_model_override("zai.glm-5".to_string(), async {
                c.reclassify_with(Ok(make_ok_response(Some(5_000))), Some(&deps))
                    .await
            }),
        )
        .await;
        assert!(
            window.successes().is_empty(),
            "tiny prompt is not a high-water"
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

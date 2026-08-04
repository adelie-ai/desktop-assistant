//! The Converse API surface: `Converse` and `ConverseStream` on
//! `bedrock-runtime`.
//!
//! This is the text-and-chat surface, and it reaches most of what Bedrock
//! serves for conversation: Anthropic, Amazon Nova, Meta, Mistral, Cohere,
//! DeepSeek and GLM, mostly through cross-region inference profiles. It does
//! not reach embedding models, image and video generation models, or
//! rerankers - none of those is addressable through Converse at all.

use async_trait::async_trait;
use aws_sdk_bedrockruntime::Client;
use aws_sdk_bedrockruntime::types::{
    ContentBlock, Message as BedrockMessage, SystemContentBlock, ToolConfiguration,
};
use aws_smithy_types::Document;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::ToolCall;
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmResponse, ModelInfo, ModelKind, ModelListingReport, TokenUsage,
};
use std::collections::HashSet;
use std::sync::{Arc, RwLock};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::backend::{
    BackendApiCapabilities, BackendTimeouts, BedrockBackend, BedrockRequest, SamplingParams,
};
use crate::sdk::SdkClients;
use crate::{
    ModalityIndex, ToolCallAccumulator, ToolNameMap, application_profiles_notice,
    apply_stream_event, base_model_for, build_additional_model_request_fields,
    cache_point_rejected_detail, cache_point_rejected_detail_stream, convert_messages,
    convert_tools, document_to_json_string, inference_profile_to_model_info,
    inference_profiles_notice, list_inference_profiles_of_type, map_converse_error,
    map_converse_stream_error, map_token_usage, register_profile_base_model,
    restore_tool_call_names, streaming_tools_unsupported_detail, summary_to_model_info,
    supports_configurable_reasoning, supports_prompt_caching, supports_streaming_with_tools,
    truncated_profiles_notice,
};

/// The Converse and `ConverseStream` operations, as a backend.
pub(crate) struct ConverseBackend {
    sdk: Arc<SdkClients>,
    timeouts: BackendTimeouts,
    /// Models discovered at runtime to reject `ConverseStream` with tools.
    /// Populated when the static allowlist (`supports_streaming_with_tools`)
    /// reports `true` but Bedrock returns the specific "doesn't support tool
    /// use in streaming mode" validation error. Per-backend, so each
    /// connection warms its own. (#67)
    non_streaming_tools_models: Mutex<HashSet<String>>,
    /// Models discovered at runtime to reject a `cachePoint` block, although
    /// `supports_prompt_caching` accepts them. That list is read from AWS
    /// documentation which states it covers only the models absent from
    /// "Models at a glance", so it is a best reading and not an enumeration;
    /// this set is how a wrong entry costs one call instead of every turn.
    ///
    /// Written only from a refusal that names the cache field, on a request
    /// that carried a checkpoint. (#1028)
    cache_unsupported_models: Mutex<HashSet<String>>,
    /// Ids the last listing reported as embedding models, so `can_serve` can
    /// answer from AWS's own modality metadata instead of a curated id list.
    ///
    /// A `std::sync::RwLock` and not a `tokio::sync::Mutex` because
    /// `can_serve` is synchronous and nothing awaits under the guard.
    embedding_models: RwLock<HashSet<String>>,
}

impl ConverseBackend {
    pub(crate) fn new(sdk: Arc<SdkClients>, timeouts: BackendTimeouts) -> Self {
        Self {
            sdk,
            timeouts,
            non_streaming_tools_models: Mutex::new(HashSet::new()),
            cache_unsupported_models: Mutex::new(HashSet::new()),
            embedding_models: RwLock::new(HashSet::new()),
        }
    }

    /// Test-only: record `model` as rejecting tools in streaming mode, so a
    /// turn that carries tools takes the non-streaming (`Converse`) path with
    /// no stream attempt first. Exists so a whole turn can be driven against a
    /// `Converse` mock for a model that supports prompt caching; the
    /// `ConverseStream` reply is an AWS event stream, which a plain HTTP mock
    /// cannot produce.
    #[doc(hidden)]
    pub(crate) async fn __force_non_streaming_tools_for_test(&self, model: &str) {
        self.non_streaming_tools_models
            .lock()
            .await
            .insert(model.to_string());
    }

    /// Call `ListFoundationModels` and `ListInferenceProfiles`, and merge them
    /// into one `ModelInfo` list.
    ///
    /// * Foundation models without `OnDemand` support are filtered out - their
    ///   bare ids are uncallable, and surfacing them leads to a runtime
    ///   `ValidationException`. Those models are reached through an inference
    ///   profile instead.
    /// * Inference profiles are merged in with their prefixed ids
    ///   (`us.anthropic.claude-haiku-4-5-...`) so the model picker exposes the
    ///   ids AWS accepts on Converse.
    ///
    /// All three calls go in parallel. A `ListFoundationModels` failure fails
    /// this listing. A `ListInferenceProfiles` failure does not: many existing
    /// IAM policies grant `bedrock:ListFoundationModels` without
    /// `bedrock:ListInferenceProfiles`, and a foundation-model-only picker
    /// beats no picker at all.
    ///
    /// The degradation is reported in the returned `ModelListingReport`
    /// notices as well as logged. Why both: what survives the filter in a
    /// current AWS account is mostly the embedding families, so a caller that
    /// only sees the model list cannot tell a degraded listing from an account
    /// with nothing but embedding models (#648).
    async fn fetch_models(&self) -> Result<ModelListingReport, CoreError> {
        use aws_sdk_bedrock::types::InferenceProfileType;

        let client = self.sdk.control().await?;

        let foundation_fut = client.list_foundation_models().send();
        let system_fut =
            list_inference_profiles_of_type(client, InferenceProfileType::SystemDefined);
        let application_fut =
            list_inference_profiles_of_type(client, InferenceProfileType::Application);

        let (foundation_res, system_res, application_res) =
            tokio::join!(foundation_fut, system_fut, application_fut);

        let foundation = foundation_res
            .map_err(|e| CoreError::Llm(format!("Bedrock ListFoundationModels failed: {e:#}")))?;

        let summaries = foundation.model_summaries();
        // Built before the on-demand filter: the models it drops are exactly
        // the ones the profiles below route to.
        let modalities = ModalityIndex::from_summaries(summaries);
        let mut models: Vec<ModelInfo> =
            summaries.iter().filter_map(summary_to_model_info).collect();
        let mut notices = Vec::new();

        let mut profiles: Vec<aws_sdk_bedrock::types::InferenceProfileSummary> = Vec::new();
        let mut truncated = false;

        match system_res {
            Ok(listing) => {
                truncated |= listing.truncated;
                profiles.extend(listing.summaries);
            }
            Err(error) => {
                use aws_smithy_types::error::metadata::ProvideErrorMetadata;
                tracing::warn!(
                    "Bedrock ListInferenceProfiles failed; model picker will only show \
                     on-demand foundation models. Grant bedrock:ListInferenceProfiles to \
                     surface inference-profile ids (Claude 4.x, Nova Premier, etc.). \
                     Cause: {error:#}"
                );
                notices.push(inference_profiles_notice(error.code(), error.message()));
            }
        }

        match application_res {
            Ok(listing) => {
                truncated |= listing.truncated;
                profiles.extend(listing.summaries);
            }
            Err(error) => {
                use aws_smithy_types::error::metadata::ProvideErrorMetadata;
                tracing::warn!(
                    "Bedrock ListInferenceProfiles for APPLICATION profiles failed. \
                     Cause: {error:#}"
                );
                // Only when the system-defined call worked. When both failed
                // the notice above already says inference profiles are
                // missing, and a second one repeats it without adding a fact.
                if notices.is_empty() {
                    notices.push(application_profiles_notice(error.code(), error.message()));
                }
            }
        }

        if truncated {
            notices.push(truncated_profiles_notice());
        }

        // Register every profile before any of them is turned into a record.
        // The record reads the register, exactly as the dispatch gates do, so
        // the two cannot answer differently.
        for profile in &profiles {
            register_profile_base_model(profile);
        }
        for profile in &profiles {
            if let Some(info) = inference_profile_to_model_info(profile, &modalities) {
                models.push(info);
            }
        }

        // Stable ordering so UIs don't shuffle between refreshes.
        // Defensive dedupe - foundation ids and profile ids don't collide
        // in practice, but keep the merge total just in case.
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models.dedup_by(|a, b| a.id == b.id);

        self.record_embedding_models(&models);

        Ok(ModelListingReport { models, notices })
    }

    /// Remember which of the listed models return vectors, so [`Self::can_serve`]
    /// answers from what AWS reported rather than from an id pattern.
    ///
    /// Entries accumulate rather than replace. A later listing that degrades -
    /// a throttled control-plane call, a narrowed IAM policy - would otherwise
    /// silently make an embedding model look servable by a chat API again, and
    /// the union is the conservative direction: a model that returns vectors
    /// does not stop returning them.
    fn record_embedding_models(&self, models: &[ModelInfo]) {
        let found: Vec<String> = models
            .iter()
            .filter(|m| m.capabilities.kind == ModelKind::Embedding)
            .map(|m| m.id.clone())
            .collect();
        if found.is_empty() {
            return;
        }
        // Poisoning is ignored for the same reason the profile register
        // ignores it: a panic elsewhere must not cost every later turn its
        // routing.
        let mut known = self
            .embedding_models
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        known.extend(found);
    }

    fn build_inference_config(
        &self,
        sampling: SamplingParams,
    ) -> Option<aws_sdk_bedrockruntime::types::InferenceConfiguration> {
        if sampling.temperature.is_none()
            && sampling.top_p.is_none()
            && sampling.max_tokens.is_none()
        {
            return None;
        }
        let mut inference_cfg = aws_sdk_bedrockruntime::types::InferenceConfiguration::builder();
        if let Some(t) = sampling.temperature {
            inference_cfg = inference_cfg.temperature(t as f32);
        }
        if let Some(p) = sampling.top_p {
            inference_cfg = inference_cfg.top_p(p as f32);
        }
        if let Some(m) = sampling.max_tokens {
            inference_cfg = inference_cfg.max_tokens(m as i32);
        }
        Some(inference_cfg.build())
    }

    /// Translate one turn into the Converse request shape.
    ///
    /// `want_cache_checkpoint` is what the caller asks for. Whether a
    /// checkpoint is actually emitted is read back off the built request, so
    /// `BedrockRequestInputs::cache_checkpoint` describes the bytes that go out
    /// rather than the intent behind them. The two differ: a turn with no
    /// system prompt has no prefix to mark, so it sends no checkpoint however
    /// the policy is set.
    ///
    /// Called a second time, with `want_cache_checkpoint` forced to `false`,
    /// when Bedrock refuses the checkpoint. It is a pure translation of the
    /// same turn, so the retry differs from the first attempt in that one field
    /// and nothing else - which is what makes the retry's outcome readable.
    fn build_request_inputs(
        &self,
        request: &BedrockRequest,
        want_cache_checkpoint: bool,
    ) -> Result<BedrockRequestInputs, CoreError> {
        let BedrockRequest {
            messages,
            tools,
            tool_names,
            model,
            reasoning,
            sampling,
            ..
        } = request;

        let (system, api_messages) = convert_messages(messages, tool_names, want_cache_checkpoint)?;
        let tool_config = convert_tools(tools, tool_names)?;

        // Read from the request, never from the request we meant to build.
        let cache_checkpoint = system
            .iter()
            .any(|block| matches!(block, SystemContentBlock::CachePoint(_)));

        let msg_count = api_messages.len();
        let tool_count = tools.len();
        // Count prompt content only. A cache checkpoint is a control marker
        // with no prompt text, so counting its `Debug` form would inflate the
        // reported prompt size on exactly the models that cache.
        let system_chars: usize = system
            .iter()
            .filter(|b| !matches!(b, SystemContentBlock::CachePoint(_)))
            .map(|b| format!("{b:?}").len())
            .sum();
        let msg_chars: usize = api_messages.iter().map(|m| format!("{m:?}").len()).sum();
        tracing::info!(
            msg_chars,
            msg_count,
            tool_count,
            system_chars,
            cache_checkpoint,
            model = %model,
            "LLM request payload"
        );

        Ok(BedrockRequestInputs {
            model: model.to_string(),
            api_messages,
            system,
            tool_config,
            inference_cfg: self.build_inference_config(*sampling),
            additional_request_fields: build_additional_model_request_fields(model, *reasoning),
            tool_names: tool_names.clone(),
            cache_checkpoint,
        })
    }

    /// Dispatch one request, choosing the path and answering the one
    /// per-model restriction that path selection cannot predict.
    ///
    /// Path selection (#67):
    /// - No tools: streaming is always safe; use the streaming path.
    /// - Tools + model on the static deny-list: skip the stream attempt
    ///   and go straight to non-streaming.
    /// - Tools + runtime memo says non-streaming: same.
    /// - Otherwise: try streaming first; on the specific
    ///   "doesn't support tool use in streaming" validation error,
    ///   memoize the model and retry via non-streaming.
    ///
    /// A refused cache checkpoint is **not** answered here. It changes the
    /// request rather than the path, so it goes back to the caller, which owns
    /// the request. (#1028)
    async fn dispatch_attempt(
        &self,
        client: &Client,
        inputs: BedrockRequestInputs,
        on_chunk: ChunkCallback,
        cancellation: &CancellationToken,
        has_tools: bool,
    ) -> Result<LlmResponse, AttemptError> {
        let model = inputs.model.clone();
        let base_model = base_model_for(&model);
        let memo_says_non_streaming = has_tools && {
            let memo = self.non_streaming_tools_models.lock().await;
            memo.contains(&model)
        };
        let allowlist_says_non_streaming = has_tools && !supports_streaming_with_tools(&base_model);
        if memo_says_non_streaming || allowlist_says_non_streaming {
            if allowlist_says_non_streaming {
                tracing::debug!(
                    model = %model,
                    "skipping ConverseStream: model on the non-streaming-with-tools deny-list"
                );
            }
            return self
                .dispatch_non_streaming(client, inputs, on_chunk, cancellation)
                .await;
        }

        match self
            .dispatch_streaming(client, &inputs, on_chunk, cancellation)
            .await
        {
            Ok(response) => Ok(response),
            Err(StreamingDispatchError::StreamingToolsUnsupported { on_chunk, detail }) => {
                tracing::warn!(
                    model = %model,
                    detail,
                    "Bedrock rejected ConverseStream with tools; retrying via Converse \
                     and memoizing the model so future turns skip the stream attempt"
                );
                self.non_streaming_tools_models
                    .lock()
                    .await
                    .insert(model.clone());
                self.dispatch_non_streaming(client, inputs, on_chunk, cancellation)
                    .await
            }
            Err(StreamingDispatchError::CachePointRejected { on_chunk, detail }) => {
                Err(AttemptError::CachePointRejected { on_chunk, detail })
            }
            Err(StreamingDispatchError::Other(err)) => Err(AttemptError::Other(err)),
        }
    }

    /// Attempt the streaming dispatch. The error path tags the specific
    /// "tools-in-streaming-mode" validation error so the caller can
    /// transparently fall back to `Converse`.
    ///
    /// `cancellation` is checked between SDK events via `tokio::select!`
    /// (issue #109) so the body stream is dropped cleanly when the user
    /// cancels mid-stream.
    async fn dispatch_streaming(
        &self,
        client: &Client,
        inputs: &BedrockRequestInputs,
        mut on_chunk: ChunkCallback,
        cancellation: &CancellationToken,
    ) -> Result<LlmResponse, StreamingDispatchError> {
        let mut request = client
            .converse_stream()
            .model_id(inputs.model.clone())
            .set_messages(Some(inputs.api_messages.clone()));
        if let Some(cfg) = inputs.inference_cfg.clone() {
            request = request.inference_config(cfg);
        }
        if !inputs.system.is_empty() {
            request = request.set_system(Some(inputs.system.clone()));
        }
        if let Some(cfg) = inputs.tool_config.clone() {
            request = request.tool_config(cfg);
        }
        if let Some(extra) = inputs.additional_request_fields.clone() {
            request = request.additional_model_request_fields(extra);
        }

        // Bound both the connection handshake and the gap between streamed
        // events so a stalled Bedrock stream fails the turn gracefully instead
        // of hanging forever (#214). `stream.recv()` and `send()` have no
        // built-in timeout; gpt-oss on Bedrock was observed accepting a
        // tool-history follow-up request and then never emitting an event.
        // The budgets default to the values shared with the reqwest connectors
        // (#302) but are overridable per-connection; Bedrock's AWS-SDK stream
        // can't reuse the `tokio_stream`-typed `next_step`, so it applies the
        // budgets directly.
        let connect_timeout = self.timeouts.connect;
        let event_timeout = self.timeouts.event;

        // Race connection establishment against cancellation and a timeout. If
        // the user cancels mid-handshake we drop the in-flight request (the
        // SDK's HTTP body) before it resolves.
        let send_fut = request.send();
        let response = tokio::select! {
            _ = cancellation.cancelled() => {
                return Err(StreamingDispatchError::Other(CoreError::Cancelled));
            }
            _ = tokio::time::sleep(connect_timeout) => {
                tracing::error!(
                    timeout_s = connect_timeout.as_secs(),
                    "Bedrock converse_stream send() timed out (no response headers)"
                );
                return Err(StreamingDispatchError::Other(CoreError::Llm(
                    "Bedrock converse_stream connection timed out".into(),
                )));
            }
            r = send_fut => match r {
                Ok(r) => r,
                Err(e) => {
                    if let Some(detail) = streaming_tools_unsupported_detail(&e) {
                        return Err(StreamingDispatchError::StreamingToolsUnsupported {
                            on_chunk,
                            detail,
                        });
                    }
                    // Asked only of a request that carried a checkpoint: a
                    // refusal of a field this request did not send is a
                    // refusal of something else.
                    if inputs.cache_checkpoint
                        && let Some(detail) = cache_point_rejected_detail_stream(&e)
                    {
                        return Err(StreamingDispatchError::CachePointRejected {
                            on_chunk,
                            detail,
                        });
                    }
                    return Err(StreamingDispatchError::Other(map_converse_stream_error(e)));
                }
            },
        };

        let mut stream = response.stream;
        let mut text = String::new();
        let mut tool_acc = ToolCallAccumulator::default();
        let mut token_usage: Option<TokenUsage> = None;
        let mut event_count: u64 = 0;

        loop {
            // Race the next streaming event against cancellation and a
            // stall timeout. Dropping `stream` closes the underlying HTTP
            // body the same way the SSE adapters do.
            let event_result = tokio::select! {
                _ = cancellation.cancelled() => {
                    tracing::debug!("Bedrock stream cancelled by token");
                    drop(stream);
                    return Err(StreamingDispatchError::Other(CoreError::Cancelled));
                }
                _ = tokio::time::sleep(event_timeout) => {
                    tracing::error!(
                        timeout_s = event_timeout.as_secs(),
                        events_so_far = event_count,
                        "Bedrock converse_stream stalled - no further event"
                    );
                    drop(stream);
                    return Err(StreamingDispatchError::Other(CoreError::Llm(
                        "Bedrock converse_stream stalled (no events)".into(),
                    )));
                }
                ev = stream.recv() => ev,
            };
            event_count += 1;
            let event = match event_result {
                Ok(Some(e)) => e,
                Ok(None) => break,
                Err(e) => {
                    return Err(StreamingDispatchError::Other(CoreError::Llm(format!(
                        "Bedrock stream receive failed: {e}"
                    ))));
                }
            };
            if !apply_stream_event(
                event,
                &mut text,
                &mut tool_acc,
                &mut on_chunk,
                &mut token_usage,
            ) {
                break;
            }
        }

        // Reverse the sanitization: the model echoed back the Bedrock-safe
        // tool name, but the upstream dispatch (and the MCP routing table)
        // keys on the ORIGINAL name. Map each call's name back. The
        // tool_use_id is left untouched.
        let tool_calls = restore_tool_call_names(tool_acc.into_tool_calls(), &inputs.tool_names);
        let mut response = if tool_calls.is_empty() {
            LlmResponse::text(text)
        } else {
            LlmResponse::with_tool_calls(text, tool_calls)
        };
        if let Some(usage) = token_usage {
            response = response.with_usage(usage);
        }
        Ok(response)
    }

    /// Non-streaming dispatch via Bedrock's `Converse` API. Used for
    /// models that reject tools in streaming mode (#67). Synthesises a
    /// single `on_chunk` call with the full text so the upstream
    /// service contract - "the callback fires at least once with the
    /// model's prose output" - is preserved.
    ///
    /// The request is bounded by the connection's non-streaming budget and
    /// raced against `cancellation`, so this path fails a stalled turn and
    /// answers a stop the same way [`Self::dispatch_streaming`] does.
    async fn dispatch_non_streaming(
        &self,
        client: &Client,
        inputs: BedrockRequestInputs,
        mut on_chunk: ChunkCallback,
        cancellation: &CancellationToken,
    ) -> Result<LlmResponse, AttemptError> {
        let cache_checkpoint = inputs.cache_checkpoint;
        let mut request = client
            .converse()
            .model_id(inputs.model.clone())
            .set_messages(Some(inputs.api_messages));
        if let Some(cfg) = inputs.inference_cfg {
            request = request.inference_config(cfg);
        }
        if !inputs.system.is_empty() {
            request = request.set_system(Some(inputs.system));
        }
        if let Some(cfg) = inputs.tool_config {
            request = request.tool_config(cfg);
        }
        if let Some(extra) = inputs.additional_request_fields {
            request = request.additional_model_request_fields(extra);
        }

        // `Converse` answers once, when generation is complete, so one bound
        // covers the whole call. Race it against cancellation as well, so a
        // stop drops the in-flight request instead of waiting the request out.
        let request_timeout = self.timeouts.non_streaming;
        let send_fut = request.send();
        let response = tokio::select! {
            _ = cancellation.cancelled() => {
                tracing::debug!(model = %inputs.model, "Bedrock converse cancelled by token");
                return Err(AttemptError::Other(CoreError::Cancelled));
            }
            _ = tokio::time::sleep(request_timeout) => {
                tracing::error!(
                    model = %inputs.model,
                    timeout_s = request_timeout.as_secs(),
                    "Bedrock converse send() timed out (no response)"
                );
                return Err(AttemptError::Other(CoreError::Llm(format!(
                    "Bedrock converse request timed out after {}s",
                    request_timeout.as_secs()
                ))));
            }
            r = send_fut => match r {
                Ok(r) => r,
                Err(e) => {
                    // Asked only of a request that carried a checkpoint: a
                    // refusal of a field this request did not send is a
                    // refusal of something else.
                    if cache_checkpoint && let Some(detail) = cache_point_rejected_detail(&e) {
                        return Err(AttemptError::CachePointRejected { on_chunk, detail });
                    }
                    return Err(AttemptError::Other(map_converse_error(e)));
                }
            },
        };

        let mut text = String::new();
        let mut tool_calls = Vec::new();
        if let Some(aws_sdk_bedrockruntime::types::ConverseOutput::Message(message)) =
            response.output
        {
            for block in message.content() {
                match block {
                    ContentBlock::Text(s) => text.push_str(s),
                    ContentBlock::ToolUse(tool_use) => {
                        // Reverse the sanitization so upstream dispatch hits
                        // the real tool; the id is left untouched.
                        let original_name =
                            inputs.tool_names.to_original(tool_use.name()).into_owned();
                        tool_calls.push(ToolCall::new(
                            tool_use.tool_use_id().to_string(),
                            original_name,
                            document_to_json_string(tool_use.input()),
                        ));
                    }
                    _ => {}
                }
            }
        }

        // Fire the callback once with the full text so the upstream
        // service treats this as a (degenerate) stream rather than
        // skipping its post-completion processing. Bail without erroring
        // if the callback signals abort - the response is fully built
        // either way.
        if !text.is_empty() {
            let _ = on_chunk(text.clone());
        }

        let token_usage = response.usage.as_ref().map(map_token_usage);

        let mut llm_response = if tool_calls.is_empty() {
            LlmResponse::text(text)
        } else {
            LlmResponse::with_tool_calls(text, tool_calls)
        };
        if let Some(usage) = token_usage {
            llm_response = llm_response.with_usage(usage);
        }
        Ok(llm_response)
    }
}

#[async_trait]
impl BedrockBackend for ConverseBackend {
    fn api_name(&self) -> &'static str {
        "converse"
    }

    fn can_serve(&self, model_id: &str) -> bool {
        // Converse is a text-and-chat API. An embedding model is not
        // addressable through it at all, and the listing is what says which
        // models those are - AWS's own output modalities, not an id pattern.
        //
        // An id no listing described answers `true`. That is the permissive
        // direction on purpose: a turn carrying an unlisted id dispatches
        // exactly as it does today, rather than being refused by a catalogue
        // that never described it.
        //
        // Both the id as it arrived and the foundation model it reduces to,
        // the same contract every other per-model gate here follows. An
        // inference profile routing to an embedding model is that model, and
        // the listing records whichever form it returned - so asking only one
        // form would refuse a bare id and accept its own profile.
        let known = self
            .embedding_models
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        !known.contains(model_id) && !known.contains(base_model_for(model_id).as_ref())
    }

    async fn list_models(&self) -> Result<ModelListingReport, CoreError> {
        self.fetch_models().await
    }

    fn capabilities(&self, model_id: &str) -> BackendApiCapabilities {
        let base = base_model_for(model_id);
        BackendApiCapabilities {
            streaming: true,
            tools: true,
            // Converse carries image content blocks, and `convert_messages`
            // emits text blocks only. Reporting the API's ability rather than
            // this backend's would advertise a capability no request can use.
            vision: false,
            cache_control: supports_prompt_caching(&base),
            // The same function the request builder reads, so a client cannot
            // be offered a control the request path will discard.
            reasoning: supports_configurable_reasoning(&base),
            hosted_tool_search: false,
            embeddings: false,
        }
    }

    async fn stream_completion(
        &self,
        request: BedrockRequest,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let client = self.sdk.runtime().await?;
        let has_tools = !request.tools.is_empty();

        // The caller's half of the answer is on the request: the operator's
        // policy allows a checkpoint and the model accepts one. This is the
        // backend's half - a refusal this surface has already met for this
        // model.
        let cache_checkpoint = request.cache_checkpoint
            && !self
                .cache_unsupported_models
                .lock()
                .await
                .contains(&request.model);

        let inputs = self.build_request_inputs(&request, cache_checkpoint)?;

        match self
            .dispatch_attempt(client, inputs, on_chunk, &request.cancellation, has_tools)
            .await
        {
            Ok(response) => Ok(response),
            Err(AttemptError::CachePointRejected { on_chunk, detail }) => {
                // The refusal named the cache field, on a request that carried
                // a checkpoint. That is the evidence, and it is the whole of
                // it: the memo is written here, from what the service said,
                // and never from a retry that succeeded - a request without
                // the field succeeds whatever the real cause was, so treating
                // that as proof would certify a wrong verdict.
                tracing::warn!(
                    model = %request.model,
                    detail,
                    "Bedrock refused the prompt-cache checkpoint; retrying this turn \
                     without it and omitting it for later turns on this model"
                );
                self.cache_unsupported_models
                    .lock()
                    .await
                    .insert(request.model.clone());

                let retry = self.build_request_inputs(&request, false)?;
                self.dispatch_attempt(client, retry, on_chunk, &request.cancellation, has_tools)
                    .await
                    .map_err(|e| match e {
                        AttemptError::Other(err) => err,
                        // Unreachable in practice: the retry carries no
                        // checkpoint, and a refusal is only classified for a
                        // request that sent one. Reported rather than retried
                        // again, so a second attempt is the last one.
                        AttemptError::CachePointRejected { detail, .. } => {
                            CoreError::Llm(format!("Bedrock converse request failed: {detail}"))
                        }
                    })
            }
            Err(AttemptError::Other(err)) => Err(err),
        }
    }
}

/// Outcome of a `ConverseStream` dispatch attempt. The "streaming with
/// tools is unsupported" arm carries the unconsumed callback so the
/// caller can retry against `Converse` without rebuilding it; a
/// `ChunkCallback` is `FnOnce`-ish in spirit (boxed dyn FnMut) and
/// passing it back avoids forcing a `Clone` bound on the trait.
enum StreamingDispatchError {
    StreamingToolsUnsupported {
        on_chunk: ChunkCallback,
        detail: String,
    },
    /// Bedrock refused the `cachePoint` block this request carried. Same
    /// callback-carrying reason as the arm above. (#1028)
    CachePointRejected {
        on_chunk: ChunkCallback,
        detail: String,
    },
    Other(CoreError),
}

/// Outcome of one complete dispatch attempt, after the streaming ->
/// non-streaming fallback has been resolved inside
/// `ConverseBackend::dispatch_attempt`.
///
/// Only one failure is actionable at that level, and it is the one the caller
/// can answer by changing the request: a refused cache checkpoint, which the
/// caller retries once without it. (#1028)
enum AttemptError {
    CachePointRejected {
        on_chunk: ChunkCallback,
        detail: String,
    },
    Other(CoreError),
}

/// All the per-call parameters that `ConverseStream` and `Converse`
/// share. Built once per attempt and consumed by whichever dispatch path
/// runs (#67).
struct BedrockRequestInputs {
    model: String,
    api_messages: Vec<BedrockMessage>,
    system: Vec<SystemContentBlock>,
    tool_config: Option<ToolConfiguration>,
    inference_cfg: Option<aws_sdk_bedrockruntime::types::InferenceConfiguration>,
    additional_request_fields: Option<Document>,
    /// Sanitized<->original tool-name bijection for this request. Used to map
    /// the (sanitized) name the model returns in a `toolUse` back to the real
    /// tool so the upstream dispatch can execute it. (#198)
    tool_names: ToolNameMap,
    /// Whether `system` carries a `cachePoint` block. Read off `system` itself
    /// in `ConverseBackend::build_request_inputs`, not from the policy that
    /// asked for one, so it cannot claim a checkpoint the request does not
    /// hold - a turn with no system prompt has no prefix to mark.
    ///
    /// This is what makes a refusal attributable. A request that sent no
    /// checkpoint cannot have had one refused, so the dispatch paths do not
    /// even ask whether the failure names the cache field. The retry is built
    /// without one, which is why the retry can never be read as a second
    /// refusal, and why the memo can never rest on evidence the fallback
    /// itself produced. (#1028)
    cache_checkpoint: bool,
}

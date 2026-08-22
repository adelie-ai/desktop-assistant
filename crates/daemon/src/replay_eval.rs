//! Running the replay eval against this daemon's own store and connector
//! (#1209).
//!
//! The measurement itself lives in
//! [`desktop_assistant_core::replay_eval`], which knows nothing about a store
//! or a model - it is handed answers. This is the thin part that fetches the
//! one and calls the other.
//!
//! Invoked as `desktop-assistant --replay-eval <user-id> [model-id]`. It reads
//! that user's
//! conversations, replays each across the ladder, prints the report and the
//! `[context.models."<id>"]` fragment the result implies, and exits without
//! starting the daemon.

use std::sync::Arc;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::Message;
use desktop_assistant_core::ports::auth::{UserId, with_user_id};
use desktop_assistant_core::ports::llm::{LlmClient, ReasoningConfig};
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_core::replay_eval::{WindowMeasurement, run_replay_eval};

/// The window sizes the eval tries, in estimated tokens, smallest first.
///
/// Five rungs spanning two orders of magnitude. The top rung is the reference
/// and cannot be its own evidence, so a model that only ever agrees there is
/// reported as unmeasured rather than as needing exactly this ladder.
const LADDER: [u64; 5] = [1_000, 4_000, 16_000, 64_000, 256_000];

/// Most conversations one run replays.
///
/// The run costs `conversations x rungs` model calls, so it is bounded. What it
/// left out is named in the report rather than counted, because a sample
/// presented as coverage is the thing the report exists to avoid.
///
/// **The sample is the most recently updated ones**, because that is the order
/// the store lists in. That is a bias and the report says so: a daemon whose
/// recent work is all short questions measures a model on short questions.
const MAX_CONVERSATIONS: usize = 40;

/// What the report says about how the sample was chosen.
const SELECTION_NOTE: &str = "sample: the most recently updated conversations, which is a recency bias rather than a random draw";

/// Run the eval for `user` and answer the report to print.
///
/// `model` names the model the answers come from. It is what the result is
/// keyed by, so it has to be the id an operator would write in
/// `[context.models."<id>"]` - a connector that cannot name its own default
/// gets a refusal rather than a fragment keyed to a phrase no config will ever
/// match.
pub async fn run<S>(
    store: &S,
    llm: Arc<dyn LlmClient>,
    user: String,
    model: Option<String>,
) -> anyhow::Result<String>
where
    S: ConversationStore,
{
    let user_id = UserId::new(user.clone());
    let model = match model.or_else(|| llm.get_default_model().map(str::to_string)) {
        Some(model) => model,
        None => {
            return Err(anyhow::anyhow!(
                "this connector does not name its own model, so the result would be keyed to \
                 nothing a config can match. Pass the model id: --replay-eval <user-id> <model-id>"
            ));
        }
    };

    let (conversations, listed, not_read) = with_user_id(user_id.clone(), async {
        let summaries = store.list().await?;
        let listed = summaries.len();
        let mut loaded = Vec::new();
        let mut not_read = Vec::new();
        for summary in summaries.into_iter().take(MAX_CONVERSATIONS) {
            match store.get(&summary.id).await {
                Ok(conv) => loaded.push(conv),
                Err(e) => not_read.push((summary.id.0, e.to_string())),
            }
        }
        Ok::<_, CoreError>((loaded, listed, not_read))
    })
    .await?;

    let read = conversations.len();
    let estimate = |text: &str| llm.estimate_tokens(text);
    let measurement: WindowMeasurement = with_user_id(user_id, async {
        run_replay_eval(
            model.clone(),
            conversations,
            &LADDER,
            &estimate,
            |_id, _rung, prompt: Vec<Message>| {
                let llm = Arc::clone(&llm);
                async move {
                    let answered = llm
                        .stream_completion(
                            prompt,
                            &[],
                            ReasoningConfig::default(),
                            Box::new(|_chunk| true),
                        )
                        .await?;
                    Ok(answered.text)
                }
            },
        )
        .await
    })
    .await;

    let mut out = measurement.report();
    // The report says what it sampled; this says what it never reached, which
    // is the difference between a sample and a claim of coverage.
    out.push_str(&format!(
        "conversations in the store: {listed}\nconversations read: {read}\n{SELECTION_NOTE}\n"
    ));
    for (id, why) in &not_read {
        out.push_str(&format!("could not read {id}: {why}\n"));
    }
    match measurement.config_fragment() {
        Some(fragment) => {
            out.push_str("\nWrite this into daemon.toml:\n\n");
            out.push_str(&fragment);
        }
        None => out.push_str(
            "\nNothing measured, so nothing to write: the model takes the \
             conservative default.\n",
        ),
    }
    Ok(out)
}

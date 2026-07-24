//! Model-kind classification for arbitrary OpenAI-compatible endpoints.
//!
//! A connector pointed at a generic `/v1/models` endpoint (an on-prem gateway,
//! a self-hosted router) gets bare model ids with no capability metadata. It
//! cannot honestly claim to know whether an id is a chat model or an embedding
//! model, so the default is [`ModelKind::Unknown`] (#647): the daemon allows an
//! `Unknown` binding with a warning rather than blocking a working config on a
//! wrong guess.
//!
//! The one signal we *can* trust is an explicit `embed`/`embedding` token in the
//! id -- the near-universal naming convention for embedding models across
//! providers (`text-embedding-3-small`, `nomic-embed-text`, `*-embedding-*`).
//! When present it positively classifies the model as [`ModelKind::Embedding`].
//! Everything else stays `Unknown`; we deliberately do NOT assume "not an embed
//! id" means generative, because a generic endpoint may serve rerankers,
//! moderation, or image models this connector shouldn't bind either.

use desktop_assistant_core::ports::llm::ModelKind;

/// Classify an OpenAI-compatible model id into a [`ModelKind`].
///
/// Returns [`ModelKind::Embedding`] only on a positive `embed` signal in the id;
/// otherwise [`ModelKind::Unknown`]. Never guesses [`ModelKind::Generative`]
/// from the absence of an embed token -- a generic endpoint carries no metadata
/// to support that claim.
pub fn classify_model_kind(id: &str) -> ModelKind {
    let lower = id.to_ascii_lowercase();
    if lower.contains("embed") {
        ModelKind::Embedding
    } else {
        ModelKind::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_compat_reports_unknown_for_unrecognized_models() {
        // Absence of metadata is `Unknown`, not a wrong guess -- an arbitrary
        // chat-looking id and total gibberish both degrade to `Unknown`.
        assert_eq!(classify_model_kind("gpt-4o"), ModelKind::Unknown);
        assert_eq!(
            classify_model_kind("some-custom-model-v2"),
            ModelKind::Unknown
        );
        assert_eq!(classify_model_kind(""), ModelKind::Unknown);
    }

    #[test]
    fn openai_compat_positively_classifies_embed_ids() {
        // The one signal we trust: an explicit embed token in the id.
        assert_eq!(
            classify_model_kind("text-embedding-3-small"),
            ModelKind::Embedding
        );
        assert_eq!(
            classify_model_kind("nomic-embed-text"),
            ModelKind::Embedding
        );
        assert_eq!(classify_model_kind("BGE-Embedding"), ModelKind::Embedding);
    }
}

//! Token-usage parsing: the `chat/completions` `usage` object into the domain
//! [`TokenUsage`], including cache-activity fields.

use desktop_assistant_core::ports::llm::TokenUsage;

/// Parse a `chat/completions` `usage` object into [`TokenUsage`].
///
/// Reads `prompt_tokens` -> `input_tokens`, `completion_tokens` ->
/// `output_tokens`, and the nested `prompt_tokens_details`:
/// `cached_tokens` -> `cache_read_input_tokens` and `cache_write_tokens` ->
/// `cache_creation_input_tokens` (OpenRouter's documented cache-activity
/// fields; Azure reports only `cached_tokens`, leaving the write side `None`).
///
/// Returns `None` when the object carries none of these fields (a missing or
/// empty `usage`), so a caller can distinguish "no usage reported" from
/// "usage reported as zero".
pub fn parse_usage(usage: &serde_json::Value) -> Option<TokenUsage> {
    let _: Option<TokenUsage> = None;
    let _ = usage;
    todo!("implemented in the next commit")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_prompt_and_completion_tokens() {
        let usage = parse_usage(&serde_json::json!({
            "prompt_tokens": 100,
            "completion_tokens": 50,
        }))
        .expect("usage present");
        assert_eq!(usage.input_tokens, Some(100));
        assert_eq!(usage.output_tokens, Some(50));
        assert_eq!(usage.cache_read_input_tokens, None);
        assert_eq!(usage.cache_creation_input_tokens, None);
    }

    #[test]
    fn parses_cached_tokens_into_cache_read() {
        let usage = parse_usage(&serde_json::json!({
            "prompt_tokens": 200,
            "completion_tokens": 10,
            "prompt_tokens_details": { "cached_tokens": 128 },
        }))
        .expect("usage present");
        assert_eq!(usage.cache_read_input_tokens, Some(128));
        assert_eq!(usage.cache_creation_input_tokens, None);
    }

    #[test]
    fn parses_cache_write_tokens_into_cache_creation() {
        let usage = parse_usage(&serde_json::json!({
            "prompt_tokens": 200,
            "completion_tokens": 10,
            "prompt_tokens_details": {
                "cached_tokens": 64,
                "cache_write_tokens": 192,
            },
        }))
        .expect("usage present");
        assert_eq!(usage.cache_read_input_tokens, Some(64));
        assert_eq!(usage.cache_creation_input_tokens, Some(192));
    }

    #[test]
    fn empty_usage_object_is_none() {
        assert!(parse_usage(&serde_json::json!({})).is_none());
    }

    #[test]
    fn null_usage_is_none() {
        assert!(parse_usage(&serde_json::Value::Null).is_none());
    }

    #[test]
    fn partial_usage_with_only_prompt_tokens() {
        let usage = parse_usage(&serde_json::json!({ "prompt_tokens": 7 })).expect("present");
        assert_eq!(usage.input_tokens, Some(7));
        assert_eq!(usage.output_tokens, None);
    }
}

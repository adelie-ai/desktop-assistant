//! Text sanitization helpers shared by the conversation handler.
//!
//! Models occasionally emit `<think>...</think>` blocks that record their
//! internal reasoning. The conversation handler strips those before storing
//! or surfacing assistant text — both for the final response (after the
//! model has produced its complete output) and for live streaming chunks
//! (where the closing tag may not yet have arrived).

/// Strip `<think>...</think>` blocks from a fully-formed assistant response,
/// then trim outer whitespace and collapse triple newlines down to doubles.
///
/// Why a tolerant parser: when a closing `</think>` is missing, the
/// remainder of the message is treated as part of the thinking block and
/// dropped. The goal is to never surface internal reasoning to the user;
/// erring on the side of dropping content matches that goal.
pub(crate) fn sanitize_assistant_text(text: &str) -> String {
    let mut remaining = text;
    let mut output = String::with_capacity(text.len());

    loop {
        let Some(start) = remaining.find("<think>") else {
            output.push_str(remaining);
            break;
        };

        output.push_str(&remaining[..start]);
        let after_start = &remaining[start + "<think>".len()..];

        match after_start.find("</think>") {
            Some(end) => {
                remaining = &after_start[end + "</think>".len()..];
            }
            None => {
                break;
            }
        }
    }

    let mut sanitized = output.trim().to_string();
    while sanitized.contains("\n\n\n") {
        sanitized = sanitized.replace("\n\n\n", "\n\n");
    }
    sanitized
}

/// Stream-friendly sanitizer used while assistant text is still arriving.
///
/// Why a separate variant: the streaming path does not yet know whether a
/// trailing `<` is the start of `<think>` or a literal angle bracket the
/// model produced. Holding back any partial-tag suffix until the next chunk
/// arrives prevents emitting visible characters that would later be
/// retroactively wrapped in a thinking block.
pub(crate) fn sanitize_assistant_text_for_stream(text: &str) -> String {
    let mut remaining = text;
    let mut output = String::with_capacity(text.len());

    loop {
        let Some(start) = remaining.find("<think>") else {
            output.push_str(remaining);
            break;
        };

        output.push_str(&remaining[..start]);
        let after_start = &remaining[start + "<think>".len()..];

        match after_start.find("</think>") {
            Some(end) => {
                remaining = &after_start[end + "</think>".len()..];
            }
            None => {
                break;
            }
        }
    }

    let partial_len = trailing_tag_prefix_len(&output, "<think>");
    if partial_len > 0 {
        output.truncate(output.len() - partial_len);
    }

    output
}

/// Length of the longest non-empty prefix of `tag` that the rendered text
/// currently ends with. Used to hold back a partial open-tag while
/// streaming so the next chunk can complete it.
fn trailing_tag_prefix_len(text: &str, tag: &str) -> usize {
    for len in (1..tag.len()).rev() {
        if text.ends_with(&tag[..len]) {
            return len;
        }
    }
    0
}

/// Redact substrings that look like credentials before an error message is
/// sent to the (possibly remote) classification LLM (epic #178).
///
/// Conservative by design: over-redaction is fine — the goal is to never
/// exfiltrate a secret in a backend error string (which can embed request
/// headers, API keys, or signed URLs). Secret-looking tokens are replaced with
/// `[REDACTED]`. Whitespace is normalized to single spaces, which is harmless
/// here: the result is only used to build the classifier prompt, never to
/// match learned signatures (those run against the original message).
pub fn redact_secrets(text: &str) -> String {
    text.split_whitespace()
        .map(redact_token)
        .collect::<Vec<_>>()
        .join(" ")
}

fn redact_token(token: &str) -> String {
    let core = token.trim_matches(|c: char| !c.is_ascii_alphanumeric());
    if core.is_empty() || !looks_like_secret(core) {
        return token.to_string();
    }
    token.replace(core, "[REDACTED]")
}

/// Heuristic: known credential prefixes, or a long high-entropy run that mixes
/// letters and digits. Requiring both letters and digits avoids redacting long
/// English words or all-digit numbers (e.g. overflow token counts).
fn looks_like_secret(core: &str) -> bool {
    const PREFIXES: [&str; 8] = [
        "AKIA",
        "ASIA",
        "sk-",
        "xoxb-",
        "xoxp-",
        "ghp_",
        "github_pat_",
        "AIza",
    ];
    if PREFIXES.iter().any(|p| core.starts_with(p)) {
        return true;
    }
    if core.len() >= 20 {
        let all_secret_charset = core
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "+/_=-.".contains(c));
        let has_digit = core.chars().any(|c| c.is_ascii_digit());
        let has_alpha = core.chars().any(|c| c.is_ascii_alphabetic());
        if all_secret_charset && has_digit && has_alpha {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod redact_tests {
    use super::redact_secrets;

    #[test]
    fn redacts_aws_key_and_long_token_keeps_prose() {
        let msg = "AccessDenied for AKIAIOSFODNN7EXAMPLE using token \
                   ghp_aB3dE5fG7hI9kL1mN3pQ5rS7tU9vW1xY3zA — try again";
        let out = redact_secrets(msg);
        assert!(out.contains("[REDACTED]"));
        assert!(!out.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(!out.contains("ghp_aB3dE5fG7hI9kL1mN3pQ5rS7tU9vW1xY3zA"));
        // Ordinary words survive.
        assert!(out.contains("AccessDenied"));
        assert!(out.contains("try again"));
    }

    #[test]
    fn keeps_overflow_numbers_and_short_words() {
        let msg = "Input length (479258) exceeds maximum context length (131072).";
        let out = redact_secrets(msg);
        assert!(out.contains("479258"));
        assert!(out.contains("131072"));
        assert!(!out.contains("[REDACTED]"));
    }
}

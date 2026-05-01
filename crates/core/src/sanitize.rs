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

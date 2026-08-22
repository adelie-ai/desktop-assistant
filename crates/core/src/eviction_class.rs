//! What kind of thing a tool result is, when the turn decides whether to drop
//! it (#1205).
//!
//! ## Why byte count is the wrong axis on its own
//!
//! The sweep in [`crate::planning`] selects rows by role and size, and replaces
//! every one it picks with a pointer under
//! [`crate::planning::DistilledTrace::Absent`] - nothing carries the content
//! forward. That is right for a gate log, whose whole content is "it worked",
//! and wrong for a recall.
//!
//! When the model calls `builtin_knowledge_base_get` it is asking for something
//! to be IN working memory. Offering that entry already cost an embedding, a
//! scoring pass and an admission decision. Dropping it three rounds later with
//! no distillate throws all of that away and sends the model back to free
//! recall - the failure the whole `[Recall]` design exists to remove. The
//! pointer even invites a loop, because it tells the model to read back
//! something it already had.
//!
//! ## The rule
//!
//! Evict by information content. A tool result is an envelope around some
//! substance, and the ratio between the two differs by an order of magnitude
//! between kinds of tool:
//!
//! | Class | Example | What eviction does |
//! |---|---|---|
//! | [`EvictionClass::Mechanism`] | a gate run, a build log, a directory listing | Drop it; keep a pointer. |
//! | [`EvictionClass::RecalledMemory`] | `builtin_knowledge_base_get`, `builtin_scratchpad_search` | **Reduce, do not remove.** |
//! | [`EvictionClass::ExternalContent`] | a fetched page, a third-party API reply | Drop it, provenance unchanged. |
//!
//! A reduced result keeps the entry text that was the point of fetching it and
//! loses the envelope around it - scores, timestamps, tags, metadata and the
//! JSON scaffolding. So the turn keeps what it learned, and no round is spent
//! recovering it.
//!
//! ## External content is decided first, and that ordering is load-bearing
//!
//! [`eviction_class`] asks [`result_is_externally_controlled`] before it asks
//! whether the tool is a memory surface. A `builtin_scratchpad_search` result
//! carrying a note stamped [`crate::tool_provenance::EXTERNAL_CONTENT_MARKER`]
//! is therefore [`EvictionClass::ExternalContent`], and evicts exactly as it
//! does today. Reducing it would rewrite a payload whose provenance the turn
//! has already graded, which is how a reduction would come to launder one.
//!
//! ## What is deliberately not a memory surface
//!
//! [`crate::ports::transcript::TRANSCRIPT_GET_TOOL`] re-surfaces a stored tool
//! result, and its provenance is inherited from whatever produced those bytes.
//! It is left as [`EvictionClass::Mechanism`] because its payload is a
//! transcript rather than a set of entries, so there is no substance to keep
//! that this module could recognise. Skills are left out for the same reason
//! plus a second one: a non-local skill body is externally controlled, and the
//! ordering above already sends it to the right place.

use crate::ports::transcript::TRANSCRIPT_GET_TOOL;
use crate::tool_provenance::result_is_externally_controlled;

/// The daemon's own memory surfaces: a call to one of these is a request to
/// put something IN working memory, so what it returned is the thing that was
/// wanted rather than an artefact of how a tool reports.
///
/// Names are matched after any MCP namespace prefix is stripped, the way
/// [`crate::tool_provenance::classify_tool`] matches. All five are daemon
/// built-ins, and every one of them returns entry rows this module's reduction
/// can recognise.
const MEMORY_SURFACES: &[&str] = &[
    "builtin_knowledge_base_get",
    "builtin_knowledge_base_search",
    "builtin_knowledge_base_list",
    "builtin_scratchpad_search",
    "builtin_conversation_search",
];

/// Opening of a reduced recall result. Distinct from
/// [`crate::planning::COMPACTION_POINTER_PREFIX`] because the two say different
/// things to the model - one says the substance is here without its envelope,
/// the other says the substance is elsewhere - and because a round must be able
/// to tell a result it has already reduced from one it has not.
pub(crate) const RECALL_REDUCED_PREFIX: &str = "<recall reduced";

/// How much of a tool result is substance, for the purpose of eviction.
///
/// Three variants, and [`EvictionClass::disposition`] matches them
/// exhaustively, so a fourth kind of result does not compile until somebody has
/// decided what eviction does with it. The call site asks for the disposition
/// rather than comparing against a variant, which is what makes that true - an
/// equality check against one variant would let a fourth fall silently into the
/// other branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EvictionClass {
    /// The content is an account of something having run. Evicting it loses
    /// nothing the turn needs, because the outcome is already in the reply the
    /// model wrote after reading it.
    Mechanism,
    /// The content is memory the model asked to hold. Reduced rather than
    /// removed: see the module header.
    RecalledMemory,
    /// The content came from, or passed through, a party the user does not
    /// control. Evicted as a mechanism result is, with its provenance grading
    /// untouched.
    ExternalContent,
}

/// What eviction does with a result of this class.
///
/// The one exhaustive match on [`EvictionClass`], and the only thing the
/// eviction path asks. See the enum's own doc for why it is a method rather
/// than a comparison at the call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Disposition {
    /// Replace it with a pointer; the bytes stay in the transcript.
    Pointer,
    /// Keep its substance and drop the envelope around it.
    ReduceToSubstance,
}

impl EvictionClass {
    /// What eviction does with a result of this class.
    pub(crate) fn disposition(self) -> Disposition {
        match self {
            // The whole content is "it worked", and the outcome is already in
            // the reply the model wrote after reading it.
            Self::Mechanism => Disposition::Pointer,
            // Nothing else holds what it carried, so removing it sends the
            // model back to free recall.
            Self::RecalledMemory => Disposition::ReduceToSubstance,
            // Evicted as a mechanism result is, with its provenance grading
            // untouched. Rewriting it would rewrite bytes the turn has already
            // graded.
            Self::ExternalContent => Disposition::Pointer,
        }
    }
}

/// Which class `result` belongs to, given the tool that produced it.
///
/// Derived from the producing tool, once, so no call site restates it.
/// `tool_name` is `None` when the turn cannot say which tool produced a row -
/// a result whose request is no longer in view - and an unattributable result
/// is [`EvictionClass::Mechanism`], which is what the sweep already does with
/// it.
pub(crate) fn eviction_class(tool_name: Option<&str>, result: &str) -> EvictionClass {
    let Some(name) = tool_name.filter(|n| !n.is_empty()) else {
        return EvictionClass::Mechanism;
    };
    if result_is_externally_controlled(name, result) {
        return EvictionClass::ExternalContent;
    }
    if MEMORY_SURFACES.contains(&base_tool_name(name)) {
        return EvictionClass::RecalledMemory;
    }
    EvictionClass::Mechanism
}

/// `name` with any MCP namespace prefix removed, so the same tool classifies
/// the same way through either door.
fn base_tool_name(name: &str) -> &str {
    name.rsplit_once("__").map_or(name, |(_, tool)| tool)
}

/// The entry text a recall result was fetched for, with the envelope around it
/// dropped.
///
/// Answers `None` when the payload is not a shape this can recognise - it is
/// not JSON, or it holds no entry rows. The caller then evicts as it would a
/// mechanism result, because a reduction that cannot find the substance must
/// not silently keep the whole payload.
///
/// **The identity of each row is kept and the rest of the envelope is not.**
/// The id is what `builtin_knowledge_base_mark` and a re-read need, and it is
/// tens of bytes against an entry of thousands; without it a reduced result is
/// text the model cannot cite. Scores, timestamps, tags, metadata, the
/// per-page counters and the JSON scaffolding all go.
///
/// `message_id` is named in the header so the whole result is still one
/// [`crate::ports::transcript::TRANSCRIPT_GET_TOOL`] call away. Reduction is
/// not a promise that nothing was lost - the envelope was lost - so the way
/// back is stated rather than implied.
pub(crate) fn reduce_recalled_result(
    tool_name: Option<&str>,
    message_id: &str,
    result: &str,
) -> Option<String> {
    let payload: serde_json::Value = serde_json::from_str(result).ok()?;
    let rows = ROW_ARRAYS
        .iter()
        .find_map(|field| payload.get(*field).and_then(serde_json::Value::as_array))?;

    let mut body = String::new();
    for row in rows {
        let Some(obj) = row.as_object() else { continue };
        let substance = SUBSTANCE_FIELDS
            .iter()
            .find_map(|f| obj.get(*f).and_then(serde_json::Value::as_str))
            .filter(|text| !text.trim().is_empty());
        let Some(text) = substance else { continue };
        if !body.is_empty() {
            body.push('\n');
        }
        if let Some(id) = IDENTITY_FIELDS
            .iter()
            .find_map(|f| obj.get(*f).and_then(serde_json::Value::as_str))
        {
            body.push_str(&format!("[{id}]\n"));
        }
        body.push_str(text);
        body.push('\n');
    }
    if body.is_empty() {
        return None;
    }

    let ran = match tool_name {
        Some(n) if !n.is_empty() => format!(" (ran {n})"),
        _ => String::new(),
    };
    // Kept short deliberately. The saving a reduction makes is the envelope
    // alone - the entry text stays either way - so a wordy header would eat the
    // whole of it and the reduction would decline on every ordinary payload.
    let reduced = format!(
        "{RECALL_REDUCED_PREFIX}{ran}: entry text only, envelope dropped. \
         Whole result: {TRANSCRIPT_GET_TOOL} message_id=\"{message_id}\".>\n{body}"
    );
    // A reduction no smaller than what it replaces makes the prompt bigger,
    // which is the one thing it may not do. Reachable at the small end: one
    // short row, where the header alone costs more than the envelope it
    // replaces.
    (reduced.len() < result.len()).then_some(reduced)
}

/// Where a recall payload keeps its rows. Both names are shipped shapes:
/// `entries` for the knowledge-base reads, `results` for the searches.
const ROW_ARRAYS: &[&str] = &["entries", "results"];

/// What a row calls the thing that was worth fetching, most specific first.
/// `summary` is last because it is a recognition surface rather than the
/// entry - a row that has both keeps the entry.
const SUBSTANCE_FIELDS: &[&str] = &["content", "text", "body", "summary"];

/// What a row calls itself, most specific first. Kept so a reduced result
/// stays citeable.
const IDENTITY_FIELDS: &[&str] = &["id", "key", "name", "conversation_id"];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_provenance::mark_external_content;

    /// A body long enough that dropping the envelope is worth the header the
    /// reduction adds. Every real recall this reaches is past
    /// `COMPACTION_MIN_EVICT_BYTES`, so a fixture below that size would test
    /// a case eviction never sees.
    fn long_body(text: &str) -> String {
        format!(
            "{text}\n{}",
            "context that made the entry worth fetching. ".repeat(12)
        )
    }

    /// Every variant states a disposition, so the exhaustive match in
    /// `disposition` covers the set.
    ///
    /// What stops a fourth variant compiling is that match itself, not this
    /// test - and what keeps the call site honest is that it asks for a
    /// disposition rather than comparing against one variant, which the
    /// compiler cannot enforce and this test does not check.
    #[test]
    fn every_class_states_its_disposition() {
        assert_eq!(EvictionClass::Mechanism.disposition(), Disposition::Pointer);
        assert_eq!(
            EvictionClass::RecalledMemory.disposition(),
            Disposition::ReduceToSubstance
        );
        assert_eq!(
            EvictionClass::ExternalContent.disposition(),
            Disposition::Pointer,
            "external content evicts as a mechanism result does, and its \
             grading is what makes that a different decision rather than the \
             same one"
        );
    }

    #[test]
    fn a_knowledge_base_read_is_recalled_memory() {
        assert_eq!(
            eviction_class(Some("builtin_knowledge_base_get"), r#"{"ok":true}"#),
            EvictionClass::RecalledMemory
        );
        assert_eq!(
            eviction_class(Some("builtin_scratchpad_search"), r#"{"ok":true}"#),
            EvictionClass::RecalledMemory
        );
    }

    /// A trusted tool that is not a memory surface. Most of the mechanism
    /// output a real turn accumulates - a gate run, a build log, a directory
    /// listing - comes from `terminal` or `fileio` and is externally
    /// controlled, so it lands in [`EvictionClass::ExternalContent`] instead.
    /// The two classes evict identically, and they are separate because what
    /// they say about the bytes is not the same.
    #[test]
    fn a_trusted_non_memory_tool_is_mechanism() {
        assert_eq!(
            eviction_class(Some("builtin_tool_search"), r#"{"ok":true,"tools":[]}"#),
            EvictionClass::Mechanism
        );
    }

    #[test]
    fn an_unattributable_result_is_mechanism() {
        assert_eq!(eviction_class(None, "anything"), EvictionClass::Mechanism);
        assert_eq!(
            eviction_class(Some(""), "anything"),
            EvictionClass::Mechanism
        );
    }

    #[test]
    fn a_fetched_page_is_external_content() {
        assert_eq!(
            eviction_class(Some("web_fetch"), "<html>"),
            EvictionClass::ExternalContent
        );
    }

    /// The ordering in [`eviction_class`]: a memory surface whose payload is
    /// externally controlled must never reach the reduction, because reducing
    /// it would rewrite bytes the turn has already graded.
    #[test]
    fn a_memory_surface_carrying_marked_content_is_external_not_recalled() {
        let marked = mark_external_content("a note a subagent wrote after reading a page");
        let payload = serde_json::json!({
            "ok": true,
            "results": [{"key": "finding", "content": marked}],
        })
        .to_string();
        assert_eq!(
            eviction_class(Some("builtin_scratchpad_search"), &payload),
            EvictionClass::ExternalContent
        );
    }

    #[test]
    fn a_namespaced_memory_surface_still_classifies_as_one() {
        assert_eq!(
            eviction_class(Some("daemon__builtin_knowledge_base_get"), r#"{"ok":true}"#),
            EvictionClass::RecalledMemory
        );
    }

    #[test]
    fn reduction_keeps_the_entry_text_and_drops_the_envelope() {
        let payload = serde_json::json!({
            "ok": true,
            "returned": 1,
            "not_found": [],
            "entries": [{
                "id": "kb-42",
                "content": "The deploy key lives in the sealed secret, not in the repo.",
                "summary": "where the deploy key lives",
                "tags": ["ops", "secrets"],
                "metadata": {"source": "dream"},
                "created_at": "2026-08-01T00:00:00Z",
                "updated_at": "2026-08-02T00:00:00Z",
            }],
        })
        .to_string();

        let reduced = reduce_recalled_result(Some("builtin_knowledge_base_get"), "m-9", &payload)
            .expect("a knowledge-base row is a shape the reduction recognises");

        assert!(
            reduced.contains("The deploy key lives in the sealed secret, not in the repo."),
            "the entry text is the substance and must survive: {reduced}"
        );
        assert!(reduced.contains("kb-42"), "the id must survive: {reduced}");
        for envelope in [
            "updated_at",
            "created_at",
            "metadata",
            "\"tags\"",
            "not_found",
            "\"returned\"",
        ] {
            assert!(
                !reduced.contains(envelope),
                "the envelope field {envelope} must not survive: {reduced}"
            );
        }
        assert!(
            reduced.len() < payload.len(),
            "a reduction that does not shrink the payload is not a reduction"
        );
    }

    #[test]
    fn reduction_names_the_way_back_to_the_whole_result() {
        // The shipped row shape, not a thin stand-in: what the reduction saves
        // is the envelope, so a fixture with less envelope than the real one
        // would measure a saving nobody gets.
        let rows: Vec<serde_json::Value> = ["goal", "finding", "next"]
            .into_iter()
            .enumerate()
            .map(|(i, key)| {
                serde_json::json!({
                    "key": key,
                    "content": long_body("the sweep runs from the round loop"),
                    "type": "note",
                    "sequence": i,
                    "done": false,
                    "pinned": false,
                    "knowledge_entry_id": serde_json::Value::Null,
                    "updated_at": "2026-08-02T00:00:00Z",
                })
            })
            .collect();
        let payload = serde_json::json!({"ok": true, "results": rows, "returned": 3}).to_string();
        let reduced = reduce_recalled_result(Some("builtin_scratchpad_search"), "m-3", &payload)
            .expect("a scratchpad row is a shape the reduction recognises");
        assert!(reduced.starts_with(RECALL_REDUCED_PREFIX), "{reduced}");
        assert!(
            reduced.contains(crate::ports::transcript::TRANSCRIPT_GET_TOOL),
            "the header must name the read-back tool: {reduced}"
        );
        assert!(
            reduced.contains("m-3"),
            "the header must name the message id: {reduced}"
        );
    }

    #[test]
    fn a_conversation_search_hit_reduces_to_its_message_text() {
        let rows: Vec<serde_json::Value> = (1..=3)
            .map(|i| {
                serde_json::json!({
                    "conversation_id": format!("c-{i}"),
                    "conversation_title": "the deploy",
                    "ordinal": 12,
                    "role": "user",
                    "snippet": "the deploy key…",
                    "content": long_body("the deploy key is in the sealed secret"),
                    "rank": 0.81,
                    "updated_at": "2026-08-02T00:00:00Z",
                })
            })
            .collect();
        let payload = serde_json::json!({"ok": true, "results": rows}).to_string();
        let reduced = reduce_recalled_result(Some("builtin_conversation_search"), "m-4", &payload)
            .expect("a conversation-search hit is a shape the reduction recognises");
        assert!(
            reduced.contains("the deploy key is in the sealed secret"),
            "{reduced}"
        );
        assert!(!reduced.contains("rank"), "{reduced}");
    }

    #[test]
    fn a_row_with_no_content_falls_back_to_its_summary() {
        let rows: Vec<serde_json::Value> = (1..=3)
            .map(|i| {
                serde_json::json!({
                    "id": format!("kb-{i}"),
                    "summary": long_body("where the deploy key lives"),
                    "tags": ["ops", "secrets"],
                    "updated_at": "2026-08-02T00:00:00Z",
                })
            })
            .collect();
        let payload = serde_json::json!({"ok": true, "entries": rows}).to_string();
        let reduced = reduce_recalled_result(Some("builtin_knowledge_base_list"), "m-5", &payload)
            .expect("a listing row still carries substance");
        assert!(reduced.contains("where the deploy key lives"), "{reduced}");
    }

    #[test]
    fn an_unrecognised_payload_reduces_to_nothing_rather_than_to_itself() {
        assert_eq!(
            reduce_recalled_result(Some("builtin_knowledge_base_get"), "m-1", "not json at all"),
            None
        );
        assert_eq!(
            reduce_recalled_result(
                Some("builtin_knowledge_base_get"),
                "m-1",
                r#"{"ok":false,"error":"knowledge base not configured"}"#
            ),
            None,
            "a payload with no entry rows has no substance to keep"
        );
    }

    #[test]
    fn a_reduction_no_smaller_than_the_payload_is_declined() {
        // One tiny row: the header alone costs more than the envelope it
        // replaces, and a reduction that grows the prompt is the one thing it
        // may not do.
        let payload =
            serde_json::json!({"ok": true, "entries": [{"id": "k", "content": "y"}]}).to_string();
        assert_eq!(
            reduce_recalled_result(Some("builtin_knowledge_base_get"), "m-1", &payload),
            None
        );
    }
}

//! Wiring for the knowledge use log (#698).
//!
//! The log records three acts. Two of them happen inside a tool call and are
//! recorded there, by the built-in knowledge tools. The third - a `[Recall]`
//! block putting entries in front of the model - happens during prompt
//! assembly, which runs no tool. This module closes that gap by decorating the
//! recall lookup: the entries the block will show are recorded as offered when
//! the lookup answers.
//!
//! ## Which entries the decorator records
//!
//! The block shows the candidates that clear the relevance floor and have a
//! line to print, up to its line budget. The decorator applies the same three
//! rules, through the same public items the renderer uses -
//! `RecallRelevance::clears_floor`, `KnowledgeEntry::display_line` and
//! `max_recall_entries` - so the two selections agree by construction rather
//! than by convention.
//!
//! They part on one rule the decorator cannot apply. The renderer also drops an
//! entry that `[Pinned]` is already carrying in full, and whether a pin
//! resolved is decided later, during assembly. On a turn where a pinned note
//! attaches an entry that also ranks near the prompt, the decorator therefore
//! records that entry - which was in front of the model, under another block -
//! and misses the entry that took its place in the budget. That case needs a
//! pinned attachment and a near-prompt rank at the same time, it moves the
//! count by one either way, and it moves it in the conservative direction: an
//! entry recorded as offered but never opened reads as ranking too high, which
//! is a claim against the entry rather than for it.

use std::sync::Arc;

use desktop_assistant_core::ports::knowledge_use::{
    KnowledgeUseLog, OfferScope, record_in_background,
};
use desktop_assistant_core::ports::recall::{RecallCandidates, RecallRequest, RecallSearchFn};
use desktop_assistant_core::recall::{RECALL_ENTRY_MAX_DISTANCE, max_recall_entries};
use desktop_assistant_storage::PgKnowledgeUseLog;

/// The ids a `[Recall]` block built from `candidates` will show.
///
/// Public to the crate so the daemon's own tests can hold it to the renderer's
/// rules; see the module doc for the one rule it cannot apply.
pub(crate) fn offered_entry_ids(candidates: &RecallCandidates) -> Vec<String> {
    candidates
        .entries
        .iter()
        .filter(|hit| hit.relevance.clears_floor(RECALL_ENTRY_MAX_DISTANCE))
        .filter(|hit| !hit.entry.display_line().is_empty())
        .take(max_recall_entries())
        .map(|hit| hit.entry.id.clone())
        .collect()
}

/// Wrap a recall lookup so the entries it will show are recorded as offered.
///
/// The record is written off the turn's path and its failure is dropped, so a
/// use log that cannot be written costs the measurement and never the block -
/// see `desktop_assistant_core::ports::knowledge_use::record_in_background`.
///
/// A lookup that found nothing is recorded too, as an offer of no entries. That
/// is not a wasted write: a recall offer replaces the conversation's standing
/// offers, and it is what ends the previous turn's. Skipping it would leave last
/// turn's offers standing through a prompt with nothing near it - and a fetch
/// two turns later would then read as a taken-up offer.
pub fn with_offer_recording(inner: RecallSearchFn, log: Arc<PgKnowledgeUseLog>) -> RecallSearchFn {
    Arc::new(move |request: RecallRequest| {
        let inner = Arc::clone(&inner);
        let log = Arc::clone(&log);
        let conversation_id = request.conversation_id.clone();
        Box::pin(async move {
            let candidates = inner(request).await?;
            let offered = offered_entry_ids(&candidates);
            let scope = OfferScope::recall(conversation_id);
            record_in_background("recall_offered", async move {
                log.record_offered(scope, offered).await
            });
            Ok(candidates)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::domain::KnowledgeEntry;
    use desktop_assistant_core::ports::recall::{RecallEntry, RecallRelevance};

    fn hit(id: &str, distance: f64, content: &str) -> RecallEntry {
        RecallEntry {
            entry: KnowledgeEntry::new(id, content, vec![]),
            relevance: RecallRelevance::Distance(distance),
        }
    }

    fn candidates(entries: Vec<RecallEntry>) -> RecallCandidates {
        RecallCandidates {
            entries,
            notes: vec![],
            tags: vec![],
        }
    }

    #[test]
    fn only_the_entries_the_block_shows_are_recorded_as_offered() {
        let found = candidates(vec![
            hit("kb-near", 0.10, "the deploy target is the lab cluster"),
            hit("kb-far", RECALL_ENTRY_MAX_DISTANCE + 0.01, "unrelated"),
            hit("kb-blank", 0.10, "   "),
        ]);
        assert_eq!(offered_entry_ids(&found), vec!["kb-near".to_string()]);
    }

    #[test]
    fn the_offer_record_stops_at_the_blocks_line_budget() {
        let entries: Vec<RecallEntry> = (0..max_recall_entries() + 4)
            .map(|i| hit(&format!("kb-{i}"), 0.10, "a fact worth keeping"))
            .collect();
        let recorded = offered_entry_ids(&candidates(entries));
        assert_eq!(recorded.len(), max_recall_entries());
        assert_eq!(recorded[0], "kb-0");
    }

    #[test]
    fn a_lexical_match_is_offered_because_the_block_shows_it() {
        // The degraded full-text arm carries no distance, and the renderer
        // shows those rows. The record must agree.
        let mut found = candidates(vec![hit("kb-1", 0.0, "a fact")]);
        found.entries[0].relevance = RecallRelevance::LexicalMatch;
        assert_eq!(offered_entry_ids(&found), vec!["kb-1".to_string()]);
    }
}

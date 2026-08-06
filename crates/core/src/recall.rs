//! The `[Recall]` block (#1100): candidate memory, offered before the model acts.
//!
//! The assistant reaches its knowledge base only when it decides to - notice a
//! search might help, choose a query, spend a tool round. When it does not
//! notice, the store is memory nobody reads. This block makes memory arrive
//! unasked: a user prompt is embedded once, both indexes that share that
//! embedding space are asked what is near it, and the candidates go in front of
//! the model before its first move.
//!
//! It is a hint and never an assertion. Entry *content* is not injected: one
//! line per entry costs about a tenth as much, and the model keeps its own
//! judgement about whether any of it matters.
//!
//! ## What bounds it
//!
//! - **A relevance floor, not a top-k.** A candidate under the floor is
//!   dropped rather than padded out to fill the budget, so "thanks" and "run
//!   the tests" produce no block at all.
//! - **A line budget.** [`MAX_RECALL_ENTRIES`] entry lines and
//!   [`MAX_RECALL_TAGS`] tag names.
//! - **One round.** The block answers "what might this prompt be about?", and
//!   the user prompt asks that once. `crate::context` renders it on the first
//!   round of a turn only.
//!
//! ## Saying what did not fit
//!
//! A model that sees eight entries cannot tell whether the store holds exactly
//! eight relevant things or four hundred, and those call for different next
//! moves. So the block reports how many cleared the floor and did not fit.
//!
//! That count means something only because the floor defines it. Over a hybrid
//! search every row scores non-zero against any query, so "how many matched" is
//! not a defined quantity; "how many cleared the floor" is. The lookup reads to
//! [`RECALL_ENTRY_SCAN_LIMIT`] and no further, so when the scan fills up the
//! count is a lower bound and says so.

use crate::ports::recall::{RecallCandidates, RecallEntry, RecallTag};

/// How many knowledge lines the block may show.
///
/// Each line is bounded by [`crate::domain::knowledge::SUMMARY_MAX_CHARS`], so
/// the entry half of the block cannot exceed about 1600 characters however
/// long the entries themselves are. Typical summaries are far shorter, which
/// is what keeps the whole block near its 300-token budget.
pub const MAX_RECALL_ENTRIES: usize = 8;

/// How many tag names the block may show.
///
/// Names only, and few of them: the arm exists to hand the model this user's
/// working vocabulary before its first search, not to list the vocabulary.
pub const MAX_RECALL_TAGS: usize = 5;

/// How many knowledge rows one lookup reads before it stops counting.
///
/// The block shows [`MAX_RECALL_ENTRIES`]; it reads this far so that "and N
/// more matched less closely" is a count rather than a guess. Bounding it costs
/// one `LIMIT` rather than a second query, and a scan that fills up makes the
/// count report itself as a lower bound.
pub const RECALL_ENTRY_SCAN_LIMIT: usize = 50;

/// How many tag rows one lookup reads before the floor is applied.
///
/// Enough headroom that the floor can drop weak neighbours without a second
/// query, and no more: no count is reported for tags, so reading further would
/// buy nothing.
pub const RECALL_TAG_SCAN_LIMIT: usize = 20;

/// The relevance floor for the knowledge arm, stated as the cosine distance a
/// candidate must stay at or under.
///
/// pgvector's `<=>` returns cosine distance in `[0, 2]`, and lower is nearer.
/// The registry's near-duplicate threshold is far tighter (0.10) because it
/// asks whether two tags are the *same concept*; this asks the much looser
/// question of whether an entry is *about what was just asked*.
///
/// This value is a deliberately conservative starting point rather than a
/// measured one: it is set to keep the block quiet on an unrelated prompt,
/// which is the failure that costs the user something. Widening it is the safe
/// direction to tune once a real store has been observed.
pub const RECALL_ENTRY_MAX_DISTANCE: f64 = 0.45;

/// The relevance floor for the tag arm, in the same cosine-distance terms as
/// [`RECALL_ENTRY_MAX_DISTANCE`].
///
/// Its own constant because it measures a different text: a registry row
/// embeds `"<name>: <description>"`, which is terse next to an entry's body, so
/// the two distances are not directly comparable. Untuned for the same reason,
/// and conservative for the same reason.
pub const RECALL_TAG_MAX_DISTANCE: f64 = 0.45;

/// The block's opening line. It states that the material may not fit and that
/// ignoring it is correct, because this fires on every prompt and a weak match
/// set that reads as an instruction is worse than no block at all.
const RECALL_HEADER: &str =
    "Memory that may relate to what was just asked. It may not fit; ignore what does not.";

/// Appended to the header when there are entry lines: what to call to read one
/// in full.
const RECALL_ENTRY_HINT: &str =
    "To read one in full, search its wording with builtin_knowledge_base_search.";

/// Label on the tag line.
const RECALL_TAG_LABEL: &str = "Tags near this prompt:";

/// Render the body of the `[Recall]` block, or `None` when nothing cleared a
/// floor.
///
/// The caller prefixes `[Recall] `; the first line returned here is the header
/// sentence, so the block reads as one paragraph followed by its lines.
///
/// Both candidate lists are taken in the order they arrive - nearest first -
/// and are never reordered: a cosine distance and a lexical match are not
/// comparable, and one lookup only ever produces one of the two.
pub(crate) fn render_recall(candidates: &RecallCandidates) -> Option<String> {
    let near_entries: Vec<&RecallEntry> = candidates
        .entries
        .iter()
        .filter(|hit| hit.relevance.clears_floor(RECALL_ENTRY_MAX_DISTANCE))
        .collect();
    let near_tags: Vec<&RecallTag> = candidates
        .tags
        .iter()
        .filter(|tag| tag.relevance.clears_floor(RECALL_TAG_MAX_DISTANCE))
        .take(MAX_RECALL_TAGS)
        .collect();

    if near_entries.is_empty() && near_tags.is_empty() {
        return None;
    }

    let mut header = RECALL_HEADER.to_string();
    if !near_entries.is_empty() {
        header.push(' ');
        header.push_str(RECALL_ENTRY_HINT);
    }
    let mut block = header;

    for hit in near_entries.iter().take(MAX_RECALL_ENTRIES) {
        block.push('\n');
        block.push_str(&entry_line(hit));
    }

    // A scan that filled up and cleared the floor on every row it read knows
    // only that there are "at least this many" beyond the page.
    let scan_filled = candidates.entries.len() >= RECALL_ENTRY_SCAN_LIMIT;
    let capped = scan_filled && near_entries.len() == candidates.entries.len();
    let dropped = near_entries.len().saturating_sub(MAX_RECALL_ENTRIES);
    if let Some(line) = dropped_line(dropped, capped) {
        block.push('\n');
        block.push_str(&line);
    }

    if let Some(line) = tag_line(&near_tags) {
        block.push('\n');
        block.push_str(&line);
    }

    Some(block)
}

/// One entry line: the id, the entry's tags, and the line that stands for it.
///
/// The tags travel even though they cost width: they are what lets the model
/// turn a hit into a better search of its own.
///
/// [`crate::domain::KnowledgeEntry::display_line`] decides what stands for an
/// entry that has no stored summary, and bounds the result to one physical line
/// of at most [`crate::domain::knowledge::SUMMARY_MAX_CHARS`] characters. That
/// fallback is the normal path until the maintenance pass has filled the column
/// in, so nothing here skips an entry for the lack of a summary.
fn entry_line(hit: &RecallEntry) -> String {
    let line = hit.entry.display_line();
    if hit.entry.tags.is_empty() {
        format!("- {} {}", hit.entry.id, line)
    } else {
        format!(
            "- {} [{}] {}",
            hit.entry.id,
            hit.entry.tags.join(", "),
            line
        )
    }
}

/// The "did not fit" line, or `None` when nothing was dropped.
///
/// `capped` renders the count as a lower bound. Reporting a capped number as if
/// it were exact is the dishonesty this line exists to avoid, and "and 0 more"
/// is noise, so both edges answer with no line at all rather than a hedged one.
fn dropped_line(dropped: usize, capped: bool) -> Option<String> {
    if dropped == 0 {
        return None;
    }
    let quantity = if capped {
        format!("{dropped} or more")
    } else {
        format!("{dropped} more")
    };
    Some(format!("...and {quantity} entries matched less closely."))
}

/// The tag line, or `None` when no tag cleared the floor.
fn tag_line(tags: &[&RecallTag]) -> Option<String> {
    if tags.is_empty() {
        return None;
    }
    let names: Vec<&str> = tags.iter().map(|t| t.name.as_str()).collect();
    Some(format!("{RECALL_TAG_LABEL} {}", names.join(", ")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::KnowledgeEntry;
    use crate::ports::recall::{RecallEntry, RecallRelevance, RecallTag};

    /// A knowledge candidate with a stored summary, at a distance that clears
    /// the floor.
    fn hit(id: &str, summary: &str, tags: &[&str], distance: f64) -> RecallEntry {
        let mut entry = KnowledgeEntry::new(
            id,
            "A body long enough that nobody would mistake it for the summary.",
            tags.iter().map(|t| (*t).to_string()).collect(),
        );
        entry.summary = Some(summary.to_string());
        RecallEntry {
            entry,
            relevance: RecallRelevance::Distance(distance),
        }
    }

    fn tag(name: &str, distance: f64) -> RecallTag {
        RecallTag {
            name: name.to_string(),
            relevance: RecallRelevance::Distance(distance),
        }
    }

    /// `n` knowledge candidates, all comfortably inside the floor.
    fn near_hits(n: usize) -> Vec<RecallEntry> {
        (0..n)
            .map(|i| hit(&format!("kb-{i}"), &format!("fact {i}"), &["topic"], 0.10))
            .collect()
    }

    fn entry_lines(block: &str) -> Vec<&str> {
        block.lines().filter(|l| l.starts_with("- ")).collect()
    }

    #[test]
    fn recall_block_lists_knowledge_hits_with_their_summaries() {
        let candidates = RecallCandidates {
            entries: vec![
                hit(
                    "kb-1a2b",
                    "Prefers dark themes in every editor",
                    &["ui"],
                    0.11,
                ),
                hit(
                    "kb-9f31",
                    "The deploy target is the lab cluster",
                    &["infra", "deploy"],
                    0.19,
                ),
            ],
            tags: vec![],
        };

        let block = render_recall(&candidates).expect("two near hits must produce a block");

        assert!(block.contains("kb-1a2b"), "{block}");
        assert!(
            block.contains("Prefers dark themes in every editor"),
            "{block}"
        );
        assert!(block.contains("kb-9f31"), "{block}");
        assert!(
            block.contains("The deploy target is the lab cluster"),
            "{block}"
        );
        assert!(
            block.contains("[infra, deploy]"),
            "an entry's tags travel with it so the model can search on them: {block}"
        );
        assert!(
            block.contains("builtin_knowledge_base_search"),
            "the block must name what to call to read an entry: {block}"
        );
    }

    #[test]
    fn recall_block_renders_an_entry_that_has_no_summary_from_its_content() {
        // Until the maintenance pass has written summaries, almost every entry
        // has none. A block that skipped them would ship showing nothing.
        let entry = KnowledgeEntry::new(
            "kb-nosum",
            "The lab cluster runs on three nodes and the registry is on the storage host.",
            vec!["infra".to_string()],
        );
        assert!(entry.summary.is_none(), "precondition");
        let candidates = RecallCandidates {
            entries: vec![RecallEntry {
                entry,
                relevance: RecallRelevance::Distance(0.12),
            }],
            tags: vec![],
        };

        let block = render_recall(&candidates).expect("an entry with no summary still shows");

        assert!(
            block.contains("The lab cluster runs on three nodes"),
            "the content stands in for the missing summary: {block}"
        );
        assert_eq!(entry_lines(&block).len(), 1);
    }

    #[test]
    fn recall_block_says_its_contents_may_not_fit() {
        // This fires on every prompt, including ones no memory relates to. A
        // block that read as an assertion would pull the model toward a memory
        // that has nothing to do with the ask.
        let candidates = RecallCandidates {
            entries: vec![hit("kb-1", "a fact", &[], 0.10)],
            tags: vec![],
        };

        let block = render_recall(&candidates).expect("a block");

        assert!(
            block.starts_with(RECALL_HEADER),
            "the block opens by saying it may not fit and may be ignored: {block}"
        );
    }

    #[test]
    fn recall_block_lists_tag_names_close_to_the_prompt() {
        let candidates = RecallCandidates {
            entries: vec![hit("kb-1", "a fact", &[], 0.10)],
            tags: vec![tag("project:adele", 0.18), tag("topic:deployment", 0.22)],
        };

        let block = render_recall(&candidates).expect("a block");

        assert!(block.contains("project:adele"), "{block}");
        assert!(block.contains("topic:deployment"), "{block}");
    }

    #[test]
    fn recall_block_renders_when_only_the_tag_arm_has_hits() {
        // The arm's whole point is a working vocabulary before the first
        // search, which is worth handing over even when no entry is near.
        let candidates = RecallCandidates {
            entries: vec![],
            tags: vec![tag("project:adele", 0.20)],
        };

        let block = render_recall(&candidates).expect("a near tag alone still produces a block");

        assert!(block.contains("project:adele"), "{block}");
        assert!(
            entry_lines(&block).is_empty(),
            "no entry lines when the knowledge arm found nothing: {block}"
        );
        assert!(
            !block.contains(RECALL_ENTRY_HINT),
            "nothing to read in full, so do not tell the model how: {block}"
        );
    }

    #[test]
    fn recall_block_is_absent_when_nothing_clears_the_relevance_floor() {
        // "thanks" and "run the tests" must produce silence, not eight
        // irrelevant memories.
        let far = RECALL_ENTRY_MAX_DISTANCE + 0.01;
        let candidates = RecallCandidates {
            entries: (0..8)
                .map(|i| hit(&format!("kb-{i}"), "an unrelated fact", &[], far))
                .collect(),
            tags: vec![tag("topic:unrelated", RECALL_TAG_MAX_DISTANCE + 0.01)],
        };

        assert!(
            render_recall(&candidates).is_none(),
            "a prompt with nothing near it emits no block at all"
        );
    }

    #[test]
    fn recall_block_respects_its_line_budget() {
        let candidates = RecallCandidates {
            entries: near_hits(MAX_RECALL_ENTRIES + 12),
            tags: (0..MAX_RECALL_TAGS + 7)
                .map(|i| tag(&format!("topic:t{i}"), 0.10))
                .collect(),
        };

        let block = render_recall(&candidates).expect("a block");

        assert_eq!(
            entry_lines(&block).len(),
            MAX_RECALL_ENTRIES,
            "the entry budget is a cap, not a suggestion: {block}"
        );
        let tags = block
            .lines()
            .find(|l| l.starts_with(RECALL_TAG_LABEL))
            .expect("a tag line");
        assert_eq!(
            tags.trim_start_matches(RECALL_TAG_LABEL).split(',').count(),
            MAX_RECALL_TAGS,
            "the tag budget is a cap too: {tags}"
        );
    }

    #[test]
    fn recall_block_reports_how_many_hits_it_dropped() {
        let candidates = RecallCandidates {
            entries: near_hits(MAX_RECALL_ENTRIES + 4),
            tags: vec![],
        };

        let block = render_recall(&candidates).expect("a block");

        assert!(
            block.contains("...and 4 more entries matched less closely."),
            "{block}"
        );
    }

    #[test]
    fn recall_block_reports_a_capped_count_as_a_lower_bound() {
        // The scan filled and every row it read cleared the floor, so the
        // remainder is "at least this many" and must not read as a total.
        let candidates = RecallCandidates {
            entries: near_hits(RECALL_ENTRY_SCAN_LIMIT),
            tags: vec![],
        };

        let block = render_recall(&candidates).expect("a block");

        let dropped = RECALL_ENTRY_SCAN_LIMIT - MAX_RECALL_ENTRIES;
        assert!(
            block.contains(&format!(
                "...and {dropped} or more entries matched less closely."
            )),
            "a capped count must read as a lower bound: {block}"
        );
    }

    #[test]
    fn recall_block_reports_an_exact_count_when_the_scan_did_not_fill_with_matches() {
        // The scan filled, but its tail fell below the floor. Rows arrive
        // nearest-first, so nothing beyond the tail could have cleared it
        // either - the count is exact and must not carry the hedge.
        let mut entries = near_hits(MAX_RECALL_ENTRIES + 3);
        entries.extend((0..RECALL_ENTRY_SCAN_LIMIT - entries.len()).map(|i| {
            hit(
                &format!("kb-far-{i}"),
                "an unrelated fact",
                &[],
                RECALL_ENTRY_MAX_DISTANCE + 0.2,
            )
        }));
        assert_eq!(entries.len(), RECALL_ENTRY_SCAN_LIMIT, "precondition");

        let block = render_recall(&RecallCandidates {
            entries,
            tags: vec![],
        })
        .expect("a block");

        assert!(
            block.contains("...and 3 more entries matched less closely."),
            "{block}"
        );
        assert!(
            !block.contains("or more"),
            "a scan that read past the floor knows the exact count: {block}"
        );
    }

    #[test]
    fn recall_block_omits_the_count_line_when_nothing_was_dropped() {
        let candidates = RecallCandidates {
            entries: near_hits(3),
            tags: vec![],
        };

        let block = render_recall(&candidates).expect("a block");

        assert!(
            !block.contains("more entries matched"),
            "\"and 0 more\" is noise: {block}"
        );
    }

    #[test]
    fn recall_block_counts_only_hits_above_the_relevance_floor() {
        // 12 near, 20 far. The count is the 4 that cleared the floor and did
        // not fit, never the 24 that a top-k would have called matches.
        let mut entries = near_hits(MAX_RECALL_ENTRIES + 4);
        entries.extend((0..20).map(|i| {
            hit(
                &format!("kb-far-{i}"),
                "an unrelated fact",
                &[],
                RECALL_ENTRY_MAX_DISTANCE + 0.3,
            )
        }));

        let block = render_recall(&RecallCandidates {
            entries,
            tags: vec![],
        })
        .expect("a block");

        assert!(
            block.contains("...and 4 more entries matched less closely."),
            "{block}"
        );
    }

    #[test]
    fn recall_block_shows_a_lexical_hit_when_the_embedding_was_unavailable() {
        // The degraded path (#195's precedent): no embedding, so the arms fall
        // back to full-text and every returned row has already passed the
        // database's own binary match.
        let mut entry = KnowledgeEntry::new("kb-fts", "body", vec![]);
        entry.summary = Some("Found by its words".to_string());
        let candidates = RecallCandidates {
            entries: vec![RecallEntry {
                entry,
                relevance: RecallRelevance::LexicalMatch,
            }],
            tags: vec![],
        };

        let block = render_recall(&candidates).expect("a lexical hit still produces a block");

        assert!(block.contains("Found by its words"), "{block}");
    }
}

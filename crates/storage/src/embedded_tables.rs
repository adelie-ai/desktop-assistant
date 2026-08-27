//! The tables that hold embeddings, declared in one place.
//!
//! Embedding lifecycle work -- invalidating vectors stamped with a superseded
//! model, restamping a cosmetic model rename, clearing orphaned stamps -- is
//! uniform across every table that stores a vector. Before this registry
//! existed each sweep named its tables inline, so `skill_index` and
//! `tag_registry` (both added after the sweep was written) were never swept:
//! a model change stranded their vectors at the old dimension, and pgvector
//! answers a mismatched comparison with an error rather than a miss (#682).
//!
//! Declaring the set here makes coverage reviewable, and
//! `tests/embedded_table_registry.rs` derives the same set from
//! `information_schema` so a table that grows a vector column fails CI until
//! it is either added below or exempted in [`EMBEDDED_TABLE_EXEMPTIONS`].
//!
//! # Adding a table
//!
//! Append its name. Every entry must have both an `embedding` and an
//! `embedding_model` column; the sweeps assume that shape and nothing else
//! about the table. A table also needs a backfill of its own to refill what a
//! sweep clears -- see `embedding_backfill` -- or its rows go permanently
//! unembedded rather than converging.
//!
//! # When a table must NOT be swept (#1328)
//!
//! `EMBEDDED_TABLES` assumes every row holds exactly one *current* vector
//! that should track the live model, invalidated and refilled like any other
//! cache. Not every vector column fits that shape, and forcing one to would
//! not merely leave it incomplete the way a forgotten registration does -- it
//! would actively destroy data the table exists to keep, or fail outright
//! against the table's own schema. [`EMBEDDED_TABLE_EXEMPTIONS`] is where such
//! a table is declared instead, with the reason written next to the name so
//! the next person to touch either list does not have to reconstruct why.

/// Tables holding embeddings, swept together by
/// [`crate::embedding_backfill::invalidate_stale_embeddings`].
///
/// These are compile-time constants, never external input, which is what makes
/// interpolating them into the sweep's SQL safe.
pub const EMBEDDED_TABLES: &[&str] = &[
    "knowledge_base",
    "tool_definitions",
    "skill_index",
    "tag_registry",
    "scratchpads",
];

/// Tables that hold a vector column but must never be swept by
/// [`crate::embedding_backfill::invalidate_stale_embeddings`], paired with why.
///
/// `tests/embedded_table_registry.rs`'s schema-derived guard treats a vector
/// table found in neither this list nor [`EMBEDDED_TABLES`] as a bug -- the
/// same guard that would otherwise force a table like this one into
/// `EMBEDDED_TABLES` just to keep the gate green, which is exactly the wrong
/// fix for a table the sweep must never touch.
///
/// Each entry is a `(table, reason)` pair rather than a bare name, so an
/// exemption cannot be added without writing down why in the same literal;
/// `an_exemption_reason_is_never_empty` below is the backstop for a reason
/// nobody bothered to write. Keep this list short and specific -- it is a
/// point of active review, not a place to park a table that merely has not
/// been wired up yet.
pub const EMBEDDED_TABLE_EXEMPTIONS: &[(&str, &str)] = &[
    (
        "recall_snapshot_entries",
        "a deliberately frozen corpus (#1328): a snapshot's whole purpose is          that it does not move while an experiment runs, and its embedding          model is part of its identity -- replay refuses to run under a          different one rather than comparing across models. Sweeping this          table would silently clear the vectors it exists to preserve, and          no backfill could ever refill them: the source knowledge_base rows          have moved on, so the original text a snapshot row was embedded          from may no longer exist to re-embed. A model change must make a          snapshot unusable for replay and say so -- not destroy it and let          the loss look like routine maintenance.",
    ),
    (
        "recall_case_embeddings",
        "not a single current vector per row, so the sweep's assumed shape          does not apply (#1328). The primary key is (case_id,          embedding_model): more than one row legitimately exists per case,          one per model that case has ever been embedded under, and every          row is permanently valid for its own model -- none of them is ever          'stale' the way a knowledge_base row's single embedding_model          column can be. The sweep's `SET embedding_model = NULL` would          violate this table's own NOT NULL primary-key column on every row          it touched. Even a version of the sweep corrected for that would          still be wrong to run: deleting an old-model row would strand any          snapshot still frozen under that model with no way to ever replay          it again, because the live embedder can no longer produce a vector          in a retired model's space to replace what was deleted.",
    ),
];

#[cfg(test)]
mod tests {
    use super::{EMBEDDED_TABLE_EXEMPTIONS, EMBEDDED_TABLES};

    /// Acceptance (#717): the scratchpad holds vectors, so it must take part in
    /// the embedding lifecycle. Without the declaration the stale sweep never
    /// clears its vectors, and after a model change every scratchpad search
    /// compares mismatched dimensions -- which pgvector answers with an error,
    /// not a miss.
    ///
    /// The schema-derived companion in `tests/embedded_table_registry.rs` needs
    /// a database; this one runs in the ordinary gate.
    #[test]
    fn scratchpads_appears_in_embedded_tables() {
        assert!(
            EMBEDDED_TABLES.contains(&"scratchpads"),
            "scratchpads holds an embedding column but is absent from EMBEDDED_TABLES, \
             so the lifecycle sweeps skip it: {EMBEDDED_TABLES:?}"
        );
    }

    /// Acceptance (#1328): an exemption without a reason is exactly the
    /// dumping ground this list must not become. The tuple shape already
    /// forces *a* string; this catches one that says nothing.
    #[test]
    fn an_exemption_reason_is_never_empty() {
        for (table, reason) in EMBEDDED_TABLE_EXEMPTIONS {
            assert!(
                reason.trim().len() >= 20,
                "the exemption for `{table}` must state a real reason, not a placeholder: \
                 {reason:?}"
            );
        }
    }

    /// Acceptance (#1328): a table cannot be both swept and exempt from
    /// sweeping -- that would leave `invalidate_stale_embeddings` running
    /// against a table its own exemption says it must never touch.
    #[test]
    fn no_table_is_both_declared_and_exempt() {
        for (table, _reason) in EMBEDDED_TABLE_EXEMPTIONS {
            assert!(
                !EMBEDDED_TABLES.contains(table),
                "`{table}` is in both EMBEDDED_TABLES and EMBEDDED_TABLE_EXEMPTIONS -- pick one"
            );
        }
    }
}

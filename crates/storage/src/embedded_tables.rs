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
//! it is added below.
//!
//! # Adding a table
//!
//! Append its name. Every entry must have both an `embedding` and an
//! `embedding_model` column; the sweeps assume that shape and nothing else
//! about the table. A table also needs a backfill of its own to refill what a
//! sweep clears -- see `embedding_backfill` -- or its rows go permanently
//! unembedded rather than converging.

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

#[cfg(test)]
mod tests {
    use super::EMBEDDED_TABLES;

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
}

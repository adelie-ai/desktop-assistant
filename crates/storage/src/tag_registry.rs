//! Formal tag vocabulary for the knowledge base (issue #108).
//!
//! Tags are categorical: each is a named, described concept rather than a
//! free-form string. The extractor picks from the registry; new tags are
//! proposed with a description and (ideally) examples, and a pre-flight
//! similarity check redirects near-duplicates to the existing tag instead of
//! letting the vocabulary drift.
//!
//! Two paths propose tags. Dreaming extraction proposes them in bulk through
//! [`create_or_match_tag`]. The knowledge-base write tool proposes them one at
//! a time through [`resolve_proposed_tag`], which is the same check with a
//! narrower door: it answers with a tag name rather than a record, and it
//! accepts a proposal with no description, because a model that omits one must
//! not cost the user a memory.
//!
//! A tag name is normalized by [`crate::tag_normalize::normalize_tag`], the
//! same function the knowledge-base write path uses, so a registry key is
//! always exactly the tag written on the rows it describes — including the
//! `facet:value` colon the knowledge-base prompt asks for.
//!
//! Storage shape mirrors migration `014_tag_registry.sql`: name PK,
//! description, examples (jsonb array of strings), `distinguish_from` siblings
//! intended to keep close concepts apart, a single embedding over
//! `name + description` for similarity dedup, and a `deprecated_for_tag`
//! chain so a retired tag can point at its replacement.

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::knowledge::ProposedTag;
use pgvector::Vector;
use sqlx::PgPool;

use crate::embedding_backfill::BackfillEmbedFn;

/// Cosine distance below which a proposed tag is considered the same concept
/// as an existing one. pgvector `<=>` returns cosine distance in `[0, 2]`;
/// lower = more similar. Empirically `0.10` is tight enough that genuinely
/// different concepts pass, while typos and trivial variations get caught.
///
/// The dedup vector is built from `"<name>: <description>"`, so restoring the
/// facet colon changed that string for every facet tag and moved two facet
/// tags of the same facet closer together. Measured on `nomic-embed-text`:
/// `project:adele-gtk` against `project:adele-tui` is `0.129`, down from
/// `0.175` for the colon-stripped names, and a near-duplicate
/// (`project:adelegtk`) is `0.028`. The band still separates them, and
/// `registry_dedup_still_separates_distinct_facet_tags` holds it there.
pub const TAG_DEDUP_DISTANCE_THRESHOLD: f64 = 0.10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagRecord {
    pub name: String,
    pub description: String,
    pub examples: Vec<String>,
    pub distinguish_from: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TagProposal {
    pub name: String,
    pub description: String,
    pub examples: Vec<String>,
    pub distinguish_from: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum CreateTagOutcome {
    Created(TagRecord),
    /// Proposal was redirected to an existing tag that the similarity check
    /// considered the same concept. Callers should use `existing.name` going
    /// forward.
    RedirectedTo {
        proposed_name: String,
        existing: TagRecord,
        distance: f64,
    },
}

/// Load all active (non-deprecated) tags ordered by name.
///
/// The tag registry is per-user (#102 moved the PK to `(user_id, name)`)
/// so the scope reads the task-local user identity. Dreaming, which is
/// the primary consumer, runs per conversation and inherits each
/// conversation's `user_id` via
/// [`desktop_assistant_core::ports::auth::with_user_id`] — see #105 for the
/// threading contract.
pub async fn list_active_tags(pool: &PgPool) -> Result<Vec<TagRecord>, CoreError> {
    let user_id = current_user_id();
    let rows: Vec<(String, String, serde_json::Value, Vec<String>)> = sqlx::query_as(
        "SELECT name, description, examples, distinguish_from \
         FROM tag_registry \
         WHERE user_id = $1 AND deprecated_for_tag IS NULL \
         ORDER BY name ASC",
    )
    .bind(user_id.as_str())
    .fetch_all(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("tag_registry: list failed: {e}")))?;

    Ok(rows.into_iter().map(row_to_record).collect())
}

/// Look up a single tag by name (active or deprecated). Scoped to the
/// current task-local user.
pub async fn get_tag(pool: &PgPool, name: &str) -> Result<Option<TagRecord>, CoreError> {
    let user_id = current_user_id();
    let row: Option<(String, String, serde_json::Value, Vec<String>)> = sqlx::query_as(
        "SELECT name, description, examples, distinguish_from \
         FROM tag_registry WHERE user_id = $1 AND name = $2",
    )
    .bind(user_id.as_str())
    .bind(name)
    .fetch_optional(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("tag_registry: get failed: {e}")))?;

    Ok(row.map(row_to_record))
}

/// Follow a deprecation chain to its terminal active tag.
///
/// Returns the input name if it isn't deprecated. Returns `None` if the chain
/// terminates at a missing tag (shouldn't happen given the FK, but graceful).
/// The chain is followed within a single user's tag partition; cross-user
/// pointers are forbidden by the FK in #102's migration.
pub async fn resolve_active_name(pool: &PgPool, name: &str) -> Result<Option<String>, CoreError> {
    let user_id = current_user_id();
    let mut current = name.to_string();
    for _ in 0..16 {
        let row: Option<(Option<String>,)> = sqlx::query_as(
            "SELECT deprecated_for_tag FROM tag_registry \
             WHERE user_id = $1 AND name = $2",
        )
        .bind(user_id.as_str())
        .bind(&current)
        .fetch_optional(pool)
        .await
        .map_err(|e| CoreError::Storage(format!("tag_registry: resolve failed: {e}")))?;
        match row {
            None => return Ok(None),
            Some((None,)) => return Ok(Some(current)),
            Some((Some(next),)) => current = next,
        }
    }
    Err(CoreError::Storage(
        "tag_registry: deprecation chain too deep (cycle?)".to_string(),
    ))
}

/// Create a new tag, or redirect to an existing similar one.
///
/// Steps:
/// 1. Normalize the proposed name with [`normalize_tag_name`] — the knowledge
///    base's own normalizer, so a facet tag such as `project:adelie-ai` keeps
///    its colon — and check for an exact match; if found, redirect.
/// 2. Embed `name + description` and search the registry for any active
///    tag within `TAG_DEDUP_DISTANCE_THRESHOLD` cosine distance — if found,
///    redirect to that tag.
/// 3. Otherwise insert and return `Created`.
pub async fn create_or_match_tag(
    pool: &PgPool,
    embed_fn: &BackfillEmbedFn,
    embedding_model: &str,
    proposal: TagProposal,
) -> Result<CreateTagOutcome, CoreError> {
    let user_id = current_user_id();
    let normalized = normalize_tag_name(&proposal.name);

    if let Some(existing) = get_tag(pool, &normalized).await? {
        return Ok(CreateTagOutcome::RedirectedTo {
            proposed_name: proposal.name,
            existing,
            distance: 0.0,
        });
    }

    let embed_text = tag_embed_text(&normalized, &proposal.description);
    let embeddings = embed_fn(vec![embed_text])
        .await
        .map_err(CoreError::Storage)?;
    let vector = embeddings
        .into_iter()
        .next()
        .ok_or_else(|| CoreError::Storage("tag_registry: embed returned no vectors".to_string()))?;
    let query_vec = Vector::from(vector);

    // $3 = the model that produced $1. Only rows embedded by that model can be
    // compared against it: a stored vector from another model has another
    // dimension, and `<=>` answers that with an error rather than a distance,
    // which would fail every tag creation instead of degrading the dedup check.
    // Sameness is decided on the digest half of the `<name>@<digest>` stamp
    // wherever both sides carry one, matching
    // `embedding_backfill::invalidate_stale_embeddings`, so a cosmetic rename
    // keeps the vocabulary deduplicating against its own vectors.
    let nearest: Option<(String, String, serde_json::Value, Vec<String>, f64)> = sqlx::query_as(
        "SELECT name, description, examples, distinguish_from, (embedding <=> $1) AS distance \
         FROM tag_registry \
         WHERE user_id = $2 AND deprecated_for_tag IS NULL AND embedding IS NOT NULL \
           AND embedding_model IS NOT NULL \
           AND (embedding_model = $3 \
                OR (split_part($3, '@', 2) <> '' \
                    AND split_part(embedding_model, '@', 2) = split_part($3, '@', 2))) \
         ORDER BY embedding <=> $1 \
         LIMIT 1",
    )
    .bind(&query_vec)
    .bind(user_id.as_str())
    .bind(embedding_model)
    .fetch_optional(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("tag_registry: nearest search failed: {e}")))?;

    if let Some((name, description, examples, distinguish_from, distance)) = nearest
        && distance < TAG_DEDUP_DISTANCE_THRESHOLD
    {
        return Ok(CreateTagOutcome::RedirectedTo {
            proposed_name: proposal.name,
            existing: row_to_record((name, description, examples, distinguish_from)),
            distance,
        });
    }

    let examples_json = serde_json::Value::Array(
        proposal
            .examples
            .iter()
            .map(|e| serde_json::Value::String(e.clone()))
            .collect(),
    );

    sqlx::query(
        "INSERT INTO tag_registry \
            (user_id, name, description, examples, distinguish_from, embedding, embedding_model) \
         VALUES ($1, $2, $3, $4, $5, $6, $7)",
    )
    .bind(user_id.as_str())
    .bind(&normalized)
    .bind(&proposal.description)
    .bind(&examples_json)
    .bind(&proposal.distinguish_from)
    .bind(&query_vec)
    .bind(embedding_model)
    .execute(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("tag_registry: insert failed: {e}")))?;

    Ok(CreateTagOutcome::Created(TagRecord {
        name: normalized,
        description: proposal.description,
        examples: proposal.examples,
        distinguish_from: proposal.distinguish_from,
    }))
}

/// Build the text a tag is embedded from, given its normalized name and its
/// description.
///
/// Why one function: the vector stored on a tag row is compared directly
/// against the vector a new proposal produces, so the creation path and the
/// backfill must embed byte-identical text or the distances between them mean
/// nothing. `backfill_tag_embeddings` reproduces this rule in SQL and
/// `tag_backfill_reproduces_the_creation_embed_text` holds the two together.
///
/// A tag with no description embeds as its name alone. Appending an empty
/// description would put a separator with nothing after it into the vector,
/// which is signal the tag does not have.
pub fn tag_embed_text(normalized_name: &str, description: &str) -> String {
    if description.trim().is_empty() {
        normalized_name.to_string()
    } else {
        format!("{normalized_name}: {description}")
    }
}

/// Resolve one tool-proposed tag to the name the knowledge base should store.
///
/// This is the knowledge-base write tool's door into the tag vocabulary. It
/// answers with the proposed name when the vocabulary accepts the tag as a new
/// concept, and with an existing tag's name when the two are the same concept,
/// so a near duplicate never becomes a second tag that no read can match.
///
/// A proposal with no description is registered under its name alone. That is
/// weaker signal for the dedup, and it is still better than refusing a write:
/// the model omitting a description must never cost the user a memory.
///
/// Repeating the same proposal gives the same answer and registers nothing
/// twice, including when a concurrent write registered the tag first: the
/// registration is keyed by `(user_id, name)`, so a lost race leaves the tag
/// present and this reads it back rather than reporting a failure.
///
/// Errors are the caller's cue to store the tag as written, not to fail the
/// write - see [`desktop_assistant_core::ports::knowledge::KnowledgeTagResolveFn`].
pub async fn resolve_proposed_tag(
    pool: &PgPool,
    embed_fn: &BackfillEmbedFn,
    embedding_model: &str,
    proposed: &ProposedTag,
) -> Result<String, CoreError> {
    let description = proposed.description.clone().unwrap_or_default();
    if description.trim().is_empty() {
        tracing::debug!(
            tag = %proposed.name,
            "no description for a proposed tag; the vocabulary matches on its name alone"
        );
    }

    let outcome = match create_or_match_tag(
        pool,
        embed_fn,
        embedding_model,
        TagProposal {
            name: proposed.name.clone(),
            description,
            examples: Vec::new(),
            distinguish_from: Vec::new(),
        },
    )
    .await
    {
        Ok(outcome) => outcome,
        Err(e) => {
            // A concurrent write can register the same name between this
            // proposal's name check and its insert. Read it back before
            // reporting a failure: the vocabulary holds the tag, so the answer
            // is the same one the winner got.
            let normalized = normalize_tag_name(&proposed.name);
            match get_tag(pool, &normalized).await {
                Ok(Some(existing)) => {
                    tracing::debug!(
                        tag = %existing.name,
                        "a concurrent write registered this tag first; using it"
                    );
                    return Ok(existing.name);
                }
                _ => return Err(e),
            }
        }
    };

    Ok(match outcome {
        CreateTagOutcome::Created(record) => record.name,
        CreateTagOutcome::RedirectedTo {
            proposed_name,
            existing,
            distance,
        } => {
            tracing::debug!(
                proposed = %proposed_name,
                existing = %existing.name,
                distance,
                "tool-path tag redirected to an existing tag"
            );
            existing.name
        }
    })
}

/// Normalize a proposed tag name into the key the registry stores.
///
/// Why this delegates rather than deciding anything itself: a registry key has
/// to be byte-identical to the tag written on the knowledge-base row it
/// describes, or no lookup can connect the two. So the registry uses the
/// knowledge base's own normalizer, [`crate::tag_normalize::normalize_tag`],
/// and holds no rule of its own that could drift from it.
pub fn normalize_tag_name(raw: &str) -> String {
    crate::tag_normalize::normalize_tag(raw)
}

fn row_to_record(row: (String, String, serde_json::Value, Vec<String>)) -> TagRecord {
    let (name, description, examples, distinguish_from) = row;
    let examples: Vec<String> = examples
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    TagRecord {
        name,
        description,
        examples,
        distinguish_from,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tag_normalize::{normalize_tag, normalize_tags};

    /// One table of raw tag names, driven through the registry entry point and
    /// through both knowledge-base entry points.
    ///
    /// Why a shared table: the registry key and the tag written on a
    /// knowledge-base row must be byte-identical, or a lookup cannot connect
    /// them. What guarantees that today is that there is one implementation,
    /// not two that agree; this table is the tripwire that fires if someone
    /// gives the registry a rule of its own again.
    const CROSS_PATH_TAG_INPUTS: &[&str] = &[
        "project:adelie-ai",
        "Project: Adelie AI",
        "PROJECT:Adelie-AI",
        "topic:Deploy",
        "Topic:Release:2026",
        "person:Ada Lovelace",
        "tool:ripgrep",
        "Preference",
        "  Memory  ",
        "INSTRUCTION",
        "multi   word\ttag",
        ":deploy",
        "user_preference",
        "--leading-trailing--",
        "weird!chars@here",
    ];

    #[test]
    fn registry_preserves_a_facet_tag_colon() {
        // The knowledge-base prompt asks for `project:<name>` shaped tags, so a
        // registry that drops the colon holds a key no row can ever carry.
        assert_eq!(normalize_tag_name("project:adelie-ai"), "project:adelie-ai");
        assert_eq!(normalize_tag_name("Project:Adelie-AI"), "project:adelie-ai");
        assert_eq!(normalize_tag_name("topic:deploy"), "topic:deploy");
        assert_eq!(normalize_tag_name("person:ada"), "person:ada");
        assert_eq!(normalize_tag_name("tool:ripgrep"), "tool:ripgrep");
    }

    #[test]
    fn registry_normalises_the_facet_and_value_halves_independently() {
        // Both halves are lowercased and whitespace-collapsed on their own, and
        // only the FIRST colon separates them.
        assert_eq!(
            normalize_tag_name("Project: Adelie AI"),
            "project:adelie-ai"
        );
        assert_eq!(
            normalize_tag_name("  TOPIC : Deploy Steps "),
            "topic:deploy-steps"
        );
        assert_eq!(
            normalize_tag_name("Topic:Release:2026"),
            "topic:release:2026"
        );
        // A leading colon has no facet name, so it is not a facet tag.
        assert_eq!(normalize_tag_name(":deploy"), ":deploy");
    }

    #[test]
    fn registry_and_knowledge_base_normalise_a_tag_identically() {
        for raw in CROSS_PATH_TAG_INPUTS {
            let registry = normalize_tag_name(raw);
            let knowledge_base = normalize_tag(raw);
            assert_eq!(
                registry, knowledge_base,
                "registry and knowledge base disagree on {raw:?}"
            );
            // The knowledge base writes through the plural entry point, so it
            // is held to the same table.
            assert_eq!(
                normalize_tags([raw]),
                vec![registry.clone()],
                "the knowledge-base write path disagrees on {raw:?}"
            );
        }
    }

    #[test]
    fn registry_embed_text_falls_back_to_the_name_without_a_description() {
        // A tag written through the knowledge-base tool can arrive with no
        // description, and the write must not fail over it. The embed text is
        // then the name alone: `"topic:deploy: "` would put a separator with
        // nothing after it into the vector the dedup compares.
        assert_eq!(
            tag_embed_text("topic:deploy", "runs and releases"),
            "topic:deploy: runs and releases"
        );
        assert_eq!(tag_embed_text("topic:deploy", ""), "topic:deploy");
        assert_eq!(tag_embed_text("topic:deploy", "   "), "topic:deploy");
    }

    #[test]
    fn normalize_handles_common_variants() {
        assert_eq!(normalize_tag_name("Project"), "project");
        assert_eq!(normalize_tag_name("  Architecture  "), "architecture");
        assert_eq!(normalize_tag_name("multi word tag"), "multi-word-tag");
    }

    #[test]
    fn normalize_keeps_characters_the_knowledge_base_keeps() {
        // The registry stores exactly what the knowledge base writes on a row,
        // so it no longer edits underscores, punctuation or edge dashes out of
        // a name. Doing so would put a key in the registry that no row carries.
        assert_eq!(normalize_tag_name("user_preference"), "user_preference");
        assert_eq!(normalize_tag_name("weird!chars@here"), "weird!chars@here");
        assert_eq!(
            normalize_tag_name("--leading-trailing--"),
            "--leading-trailing--"
        );
    }
}

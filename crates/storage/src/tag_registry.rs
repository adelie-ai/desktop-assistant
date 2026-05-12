//! Formal tag vocabulary for the knowledge base (issue #108).
//!
//! Tags are categorical: each is a named, described concept rather than a
//! free-form string. The extractor picks from the registry; new tags must be
//! proposed with a description and (ideally) examples, and a pre-flight
//! similarity check redirects near-duplicates to the existing tag instead of
//! letting the vocabulary drift.
//!
//! Storage shape mirrors migration `014_tag_registry.sql`: name PK,
//! description, examples (jsonb array of strings), `distinguish_from` siblings
//! intended to keep close concepts apart, a single embedding over
//! `name + description` for similarity dedup, and a `deprecated_for_tag`
//! chain so a retired tag can point at its replacement.

use pgvector::Vector;
use sqlx::PgPool;

use crate::embedding_backfill::BackfillEmbedFn;

/// Cosine distance below which a proposed tag is considered the same concept
/// as an existing one. pgvector `<=>` returns cosine distance in `[0, 2]`;
/// lower = more similar. Empirically `0.10` is tight enough that genuinely
/// different concepts pass, while typos and trivial variations get caught.
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
pub async fn list_active_tags(pool: &PgPool) -> Result<Vec<TagRecord>, String> {
    let rows: Vec<(String, String, serde_json::Value, Vec<String>)> = sqlx::query_as(
        "SELECT name, description, examples, distinguish_from
         FROM tag_registry
         WHERE deprecated_for_tag IS NULL
         ORDER BY name ASC",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| format!("tag_registry: list failed: {e}"))?;

    Ok(rows.into_iter().map(row_to_record).collect())
}

/// Look up a single tag by name (active or deprecated).
pub async fn get_tag(pool: &PgPool, name: &str) -> Result<Option<TagRecord>, String> {
    let row: Option<(String, String, serde_json::Value, Vec<String>)> = sqlx::query_as(
        "SELECT name, description, examples, distinguish_from
         FROM tag_registry WHERE name = $1",
    )
    .bind(name)
    .fetch_optional(pool)
    .await
    .map_err(|e| format!("tag_registry: get failed: {e}"))?;

    Ok(row.map(row_to_record))
}

/// Follow a deprecation chain to its terminal active tag.
///
/// Returns the input name if it isn't deprecated. Returns `None` if the chain
/// terminates at a missing tag (shouldn't happen given the FK, but graceful).
pub async fn resolve_active_name(pool: &PgPool, name: &str) -> Result<Option<String>, String> {
    let mut current = name.to_string();
    for _ in 0..16 {
        let row: Option<(Option<String>,)> =
            sqlx::query_as("SELECT deprecated_for_tag FROM tag_registry WHERE name = $1")
                .bind(&current)
                .fetch_optional(pool)
                .await
                .map_err(|e| format!("tag_registry: resolve failed: {e}"))?;
        match row {
            None => return Ok(None),
            Some((None,)) => return Ok(Some(current)),
            Some((Some(next),)) => current = next,
        }
    }
    Err("tag_registry: deprecation chain too deep (cycle?)".to_string())
}

/// Create a new tag, or redirect to an existing similar one.
///
/// Steps:
/// 1. Normalize the proposed name (lowercase, dashes preferred over
///    underscores) and check for an exact match — if found, redirect.
/// 2. Embed `name + description` and search the registry for any active
///    tag within `TAG_DEDUP_DISTANCE_THRESHOLD` cosine distance — if found,
///    redirect to that tag.
/// 3. Otherwise insert and return `Created`.
pub async fn create_or_match_tag(
    pool: &PgPool,
    embed_fn: &BackfillEmbedFn,
    embedding_model: &str,
    proposal: TagProposal,
) -> Result<CreateTagOutcome, String> {
    let normalized = normalize_tag_name(&proposal.name);

    if let Some(existing) = get_tag(pool, &normalized).await? {
        return Ok(CreateTagOutcome::RedirectedTo {
            proposed_name: proposal.name,
            existing,
            distance: 0.0,
        });
    }

    let embed_text = format!("{}: {}", normalized, proposal.description);
    let embeddings = embed_fn(vec![embed_text]).await?;
    let vector = embeddings
        .into_iter()
        .next()
        .ok_or_else(|| "tag_registry: embed returned no vectors".to_string())?;
    let query_vec = Vector::from(vector);

    let nearest: Option<(String, String, serde_json::Value, Vec<String>, f64)> = sqlx::query_as(
        "SELECT name, description, examples, distinguish_from, (embedding <=> $1) AS distance
         FROM tag_registry
         WHERE deprecated_for_tag IS NULL AND embedding IS NOT NULL
         ORDER BY embedding <=> $1
         LIMIT 1",
    )
    .bind(&query_vec)
    .fetch_optional(pool)
    .await
    .map_err(|e| format!("tag_registry: nearest search failed: {e}"))?;

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
        "INSERT INTO tag_registry
            (name, description, examples, distinguish_from, embedding, embedding_model)
         VALUES ($1, $2, $3, $4, $5, $6)",
    )
    .bind(&normalized)
    .bind(&proposal.description)
    .bind(&examples_json)
    .bind(&proposal.distinguish_from)
    .bind(&query_vec)
    .bind(embedding_model)
    .execute(pool)
    .await
    .map_err(|e| format!("tag_registry: insert failed: {e}"))?;

    Ok(CreateTagOutcome::Created(TagRecord {
        name: normalized,
        description: proposal.description,
        examples: proposal.examples,
        distinguish_from: proposal.distinguish_from,
    }))
}

/// Lowercase, trim, and prefer dashes over underscores/spaces.
pub fn normalize_tag_name(raw: &str) -> String {
    raw.trim()
        .to_lowercase()
        .replace([' ', '_'], "-")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-')
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

fn row_to_record(
    row: (String, String, serde_json::Value, Vec<String>),
) -> TagRecord {
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

    #[test]
    fn normalize_handles_common_variants() {
        assert_eq!(normalize_tag_name("Project"), "project");
        assert_eq!(normalize_tag_name("user_preference"), "user-preference");
        assert_eq!(normalize_tag_name("  Architecture  "), "architecture");
        assert_eq!(normalize_tag_name("multi word tag"), "multi-word-tag");
        assert_eq!(normalize_tag_name("--leading-trailing--"), "leading-trailing");
        assert_eq!(normalize_tag_name("weird!chars@here"), "weirdcharshere");
    }
}

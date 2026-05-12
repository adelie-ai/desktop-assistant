//! Phase 1: extract new facts from conversation transcripts.
//!
//! Differences from the legacy implementation (issue #108):
//!
//! - Tags are categorical and must come from `tag_registry`. The system
//!   prompt includes the active vocabulary; the LLM either picks from it or
//!   proposes a new tag in-band with name, description, and examples.
//!   Proposed tags pass through `tag_registry::create_or_match_tag`, which
//!   does a pre-flight similarity check and may redirect to an existing
//!   tag.
//! - Each fact carries a structured `scope` (or explicit `null` for
//!   universals) plus the source conversation id, persisted into
//!   `knowledge_base.metadata`.
//! - No extraction-time vector dedup. Same-cycle near-duplicates are left
//!   for consolidation, where scope-aware merging is the right place to
//!   handle them.

use std::collections::BTreeSet;

use desktop_assistant_core::chunking::{CHUNK_MAX_CHARS, CHUNK_OVERLAP, chunk_text};
use pgvector::Vector;
use sqlx::PgPool;

use super::common::{
    extract_json_payload, find_conversations_with_new_messages, get_max_ordinal,
    load_new_transcript, update_watermark,
};
use super::types::{BackfillEmbedFn, DreamingLlmFn};
use crate::kb_metadata::{KbMetadata, KbScope};
use crate::tag_registry::{
    self, CreateTagOutcome, TagProposal, TagRecord, normalize_tag_name,
};

/// Run the extraction phase. Returns the count of new facts written.
pub async fn run_extraction_phase(
    pool: &PgPool,
    llm_fn: &DreamingLlmFn,
    embed_fn: &BackfillEmbedFn,
    embedding_model: &str,
) -> Result<usize, String> {
    let conversations = find_conversations_with_new_messages(pool).await?;
    if conversations.is_empty() {
        tracing::debug!("dreaming: no conversations with new messages");
        return Ok(0);
    }
    tracing::info!(
        "dreaming: extraction scanning {} conversation(s)",
        conversations.len()
    );

    let registry = tag_registry::list_active_tags(pool).await?;
    let registry_names: BTreeSet<String> =
        registry.iter().map(|t| t.name.clone()).collect();

    let mut total_written = 0usize;

    for (conv_id, watermark, context_summary) in conversations {
        let max_ordinal = get_max_ordinal(pool, &conv_id).await?;
        if max_ordinal <= watermark {
            continue;
        }

        let transcript = load_new_transcript(pool, &conv_id, watermark).await?;
        if transcript.is_empty() {
            update_watermark(pool, &conv_id, max_ordinal).await?;
            continue;
        }

        let system_prompt = build_extraction_system_prompt(&registry);
        let user_prompt = build_extraction_user_prompt(&context_summary, &transcript);

        let response = match llm_fn(system_prompt, user_prompt).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("dreaming: extraction LLM call failed for {conv_id}: {e}");
                continue;
            }
        };

        let proposals = parse_extraction_response(&response);

        let mut written_this_conv = 0usize;
        for proposal in proposals {
            match write_extracted_fact(
                pool,
                embed_fn,
                embedding_model,
                &conv_id,
                proposal,
                &registry_names,
            )
            .await
            {
                Ok(true) => written_this_conv += 1,
                Ok(false) => {}
                Err(e) => tracing::warn!("dreaming: write_extracted_fact failed: {e}"),
            }
        }

        update_watermark(pool, &conv_id, max_ordinal).await?;
        total_written += written_this_conv;

        tracing::info!(
            "dreaming: conversation {conv_id} wrote {written_this_conv} fact(s)"
        );
    }

    Ok(total_written)
}

/// One fact as proposed by the LLM, before tag resolution.
#[derive(Debug, Clone)]
struct ExtractedFactProposal {
    content: String,
    tags: Vec<String>,
    new_tags: Vec<TagProposal>,
    scope: Option<KbScope>,
}

fn parse_extraction_response(response: &str) -> Vec<ExtractedFactProposal> {
    let json_str = extract_json_payload(response.trim());

    // Accept either a bare array of facts or `{facts: [...]}`.
    let root: serde_json::Value = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("dreaming: extraction response is not JSON: {e}");
            tracing::debug!("dreaming: raw response: {response}");
            return Vec::new();
        }
    };

    let facts_array = match root {
        serde_json::Value::Array(a) => a,
        serde_json::Value::Object(mut o) => match o.remove("facts") {
            Some(serde_json::Value::Array(a)) => a,
            _ => {
                tracing::warn!("dreaming: extraction response object has no `facts` array");
                return Vec::new();
            }
        },
        _ => {
            tracing::warn!("dreaming: extraction response is neither array nor object");
            return Vec::new();
        }
    };

    facts_array
        .into_iter()
        .filter_map(parse_one_fact)
        .collect()
}

fn parse_one_fact(value: serde_json::Value) -> Option<ExtractedFactProposal> {
    let obj = value.as_object()?;
    let content = obj.get("content")?.as_str()?.trim().to_string();
    if content.is_empty() {
        return None;
    }

    let tags: Vec<String> = obj
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| t.as_str().map(normalize_tag_name))
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();

    let new_tags: Vec<TagProposal> = obj
        .get("new_tags")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(parse_tag_proposal).collect())
        .unwrap_or_default();

    let scope = match obj.get("scope") {
        Some(serde_json::Value::Object(map)) => {
            let mut scope = KbScope::new();
            for (k, v) in map {
                if let Some(s) = v.as_str() {
                    scope = scope.with(k.clone(), s.to_string());
                }
            }
            if scope.is_empty() { None } else { Some(scope) }
        }
        _ => None,
    };

    Some(ExtractedFactProposal {
        content,
        tags,
        new_tags,
        scope,
    })
}

fn parse_tag_proposal(value: &serde_json::Value) -> Option<TagProposal> {
    let obj = value.as_object()?;
    let name = obj.get("name")?.as_str()?.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let description = obj
        .get("description")?
        .as_str()?
        .trim()
        .to_string();
    if description.is_empty() {
        return None;
    }
    let examples: Vec<String> = obj
        .get("examples")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|e| e.as_str().map(|s| s.trim().to_string()))
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let distinguish_from: Vec<String> = obj
        .get("distinguish_from")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|e| e.as_str().map(normalize_tag_name))
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    Some(TagProposal {
        name,
        description,
        examples,
        distinguish_from,
    })
}

async fn write_extracted_fact(
    pool: &PgPool,
    embed_fn: &BackfillEmbedFn,
    embedding_model: &str,
    source_conversation_id: &str,
    proposal: ExtractedFactProposal,
    registry_names: &BTreeSet<String>,
) -> Result<bool, String> {
    let mut final_tags: BTreeSet<String> = BTreeSet::new();

    // Existing tags: keep only those present in the active registry.
    for tag in &proposal.tags {
        if registry_names.contains(tag) {
            final_tags.insert(tag.clone());
        } else {
            tracing::debug!("dreaming: extractor emitted unknown tag '{tag}', dropping");
        }
    }

    // New-tag proposals go through the registry's pre-flight check.
    for tp in proposal.new_tags {
        match tag_registry::create_or_match_tag(pool, embed_fn, embedding_model, tp).await {
            Ok(CreateTagOutcome::Created(TagRecord { name, .. })) => {
                final_tags.insert(name);
            }
            Ok(CreateTagOutcome::RedirectedTo { existing, distance, proposed_name }) => {
                tracing::debug!(
                    "dreaming: tag '{proposed_name}' redirected to '{}' (distance={distance:.3})",
                    existing.name
                );
                final_tags.insert(existing.name);
            }
            Err(e) => tracing::warn!("dreaming: tag proposal failed: {e}"),
        }
    }

    // Always include the source-tag for provenance lookups.
    final_tags.insert("source:dreaming".to_string());

    // Build structured metadata.
    let metadata = KbMetadata {
        scope: proposal.scope,
        source_conversation_id: Some(source_conversation_id.to_string()),
        ..Default::default()
    };

    // Embed content for the row's `embedding` array.
    let chunks = chunk_text(&proposal.content, CHUNK_MAX_CHARS, CHUNK_OVERLAP);
    let embeddings = embed_fn(chunks).await?;
    if embeddings.is_empty() {
        return Err("dreaming: embedding returned no vectors".to_string());
    }
    let embedding_vecs: Vec<Vector> = embeddings.into_iter().map(Vector::from).collect();

    let id = uuid::Uuid::now_v7().to_string();
    let tags_vec: Vec<String> = final_tags.into_iter().collect();

    sqlx::query(
        "INSERT INTO knowledge_base
            (id, content, tags, metadata, embedding, embedding_model)
         VALUES ($1, $2, $3, $4, $5::vector[], $6)",
    )
    .bind(&id)
    .bind(&proposal.content)
    .bind(&tags_vec)
    .bind(metadata.to_json())
    .bind(&embedding_vecs)
    .bind(embedding_model)
    .execute(pool)
    .await
    .map_err(|e| format!("dreaming: insert fact failed: {e}"))?;

    tracing::info!(
        "dreaming: wrote fact id={id} (scope={:?}): {}",
        metadata.effective_scope(),
        &proposal.content[..proposal.content.len().min(80)]
    );

    Ok(true)
}

fn build_extraction_system_prompt(registry: &[TagRecord]) -> String {
    let mut prompt = String::from(
        "You are a fact-extraction assistant. Identify important long-term facts, \
        preferences, and knowledge from a conversation transcript that would be \
        useful to remember in future conversations.\n\
        \n\
        EXTRACT facts about:\n\
        - User preferences (tools, workflows, communication style)\n\
        - Technical decisions and architectural choices\n\
        - Project-specific knowledge (paths, patterns, conventions)\n\
        - Personal context the user has shared\n\
        - Recurring problems and their solutions\n\
        \n\
        DO NOT extract:\n\
        - Transient task details (what the user is doing right now)\n\
        - Obvious or generic information\n\
        - Information only relevant to the current session\n\
        - Code snippets or implementation details\n\
        \n\
        ## Output format\n\
        \n\
        Return a JSON object with a `facts` array. Each fact has:\n\
        - `content` (string): A self-contained prose sentence. If the fact only \
        applies in a specific context (a project, tool, environment), include \
        that context IN THE PROSE — never write a scope-naked fact like \
        \"the project directory is /a\" without naming the project.\n\
        - `tags` (array of strings): Categorical tags from the registry below. \
        Pick only from registered tags. Tags describe WHAT KIND of fact this \
        is (e.g. `preference`, `architecture`), not the specific subject.\n\
        - `new_tags` (array of objects, optional): Only when no existing tag \
        fits. Each new tag needs: `name` (kebab-case), `description` (one \
        sentence: what does this tag mean?), `examples` (2-3 short \
        instances), and optionally `distinguish_from` (existing tag names \
        this should NOT be confused with).\n\
        - `scope` (object or null): If the fact is conditional on something \
        (e.g. specific to one project, tool, or environment), provide a \
        scope object with string-keyed dimensions: `{\"project\": \
        \"adelie-ai\"}` or `{\"tool\": \"vscode\"}`. Use `null` ONLY when \
        the fact is genuinely universal (e.g. a user preference that holds \
        regardless of project).\n\
        \n\
        ## Tag registry\n\
        \n",
    );

    if registry.is_empty() {
        prompt.push_str(
            "(registry is empty — propose new_tags freely with clear descriptions)\n",
        );
    } else {
        for tag in registry {
            prompt.push_str(&format!("- `{}`: {}", tag.name, tag.description));
            if !tag.examples.is_empty() {
                let joined: Vec<String> = tag
                    .examples
                    .iter()
                    .take(3)
                    .map(|e| format!("\"{e}\""))
                    .collect();
                prompt.push_str(&format!(" — examples: {}", joined.join(", ")));
            }
            if !tag.distinguish_from.is_empty() {
                prompt.push_str(&format!(
                    " — distinguish from: {}",
                    tag.distinguish_from.join(", ")
                ));
            }
            prompt.push('\n');
        }
    }

    prompt.push_str(
        "\n\
        ## Example output\n\
        \n\
        ```json\n\
        {\n  \"facts\": [\n    \
        {\"content\": \"The user prefers dark mode in editors.\", \
        \"tags\": [\"preference\"], \"scope\": null},\n    \
        {\"content\": \"The adelie-ai project uses PostgreSQL with pgvector \
        for semantic search.\", \"tags\": [\"architecture\"], \
        \"scope\": {\"project\": \"adelie-ai\"}}\n  \
        ]\n}\n\
        ```\n\
        \n\
        If there are no facts worth extracting, return `{\"facts\": []}`.",
    );

    prompt
}

fn build_extraction_user_prompt(context_summary: &str, transcript: &str) -> String {
    let mut prompt = String::new();
    if !context_summary.is_empty() {
        prompt.push_str("## Conversation context summary\n\n");
        prompt.push_str(context_summary);
        prompt.push_str("\n\n");
    }
    prompt.push_str("## Recent messages\n\n");
    prompt.push_str(transcript);
    prompt.push_str(
        "\n\nExtract any important long-term facts from the above transcript. \
        Return a JSON object with a `facts` array.",
    );
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_facts_array_form() {
        let response = r#"{"facts": [
            {"content": "The user prefers vim.", "tags": ["preference"], "scope": null}
        ]}"#;
        let facts = parse_extraction_response(response);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].content, "The user prefers vim.");
        assert_eq!(facts[0].tags, vec!["preference"]);
        assert!(facts[0].scope.is_none());
    }

    #[test]
    fn parses_legacy_bare_array_form() {
        let response = r#"[
            {"content": "A fact.", "tags": ["x"]}
        ]"#;
        let facts = parse_extraction_response(response);
        assert_eq!(facts.len(), 1);
    }

    #[test]
    fn parses_scope_object() {
        let response = r#"{"facts": [
            {"content": "Project dir is /a.", "tags": [], "scope": {"project": "adelie-ai"}}
        ]}"#;
        let facts = parse_extraction_response(response);
        assert_eq!(facts.len(), 1);
        let scope = facts[0].scope.as_ref().expect("scope set");
        assert_eq!(
            scope.0.get("project").map(String::as_str),
            Some("adelie-ai")
        );
    }

    #[test]
    fn empty_scope_object_becomes_none() {
        let response = r#"{"facts": [
            {"content": "Universal fact.", "tags": [], "scope": {}}
        ]}"#;
        let facts = parse_extraction_response(response);
        assert!(facts[0].scope.is_none());
    }

    #[test]
    fn parses_new_tag_proposals() {
        let response = r#"{"facts": [
            {"content": "x", "tags": [],
             "new_tags": [{"name": "habit", "description": "A recurring user habit.",
                           "examples": ["wakes early"], "distinguish_from": ["preference"]}]}
        ]}"#;
        let facts = parse_extraction_response(response);
        assert_eq!(facts[0].new_tags.len(), 1);
        assert_eq!(facts[0].new_tags[0].name, "habit");
        assert_eq!(facts[0].new_tags[0].distinguish_from, vec!["preference"]);
    }

    #[test]
    fn normalizes_tag_names_on_parse() {
        let response = r#"{"facts": [
            {"content": "x", "tags": ["User_Preference", "Architecture"]}
        ]}"#;
        let facts = parse_extraction_response(response);
        assert_eq!(facts[0].tags, vec!["user-preference", "architecture"]);
    }

    #[test]
    fn drops_facts_with_empty_content() {
        let response = r#"{"facts": [
            {"content": "", "tags": ["x"]},
            {"content": "ok", "tags": []}
        ]}"#;
        let facts = parse_extraction_response(response);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].content, "ok");
    }

    #[test]
    fn invalid_json_returns_empty() {
        let facts = parse_extraction_response("not json at all");
        assert!(facts.is_empty());
    }

    #[test]
    fn prompt_includes_registered_tag_descriptions() {
        let registry = vec![
            TagRecord {
                name: "preference".into(),
                description: "User preferences for tools or workflows.".into(),
                examples: vec!["prefers vim".into()],
                distinguish_from: vec![],
            },
        ];
        let prompt = build_extraction_system_prompt(&registry);
        assert!(prompt.contains("preference"));
        assert!(prompt.contains("User preferences"));
        assert!(prompt.contains("prefers vim"));
    }

    #[test]
    fn prompt_handles_empty_registry() {
        let prompt = build_extraction_system_prompt(&[]);
        assert!(prompt.contains("registry is empty"));
    }
}

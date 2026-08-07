//! Routing an extracted method to the skill catalog instead of the knowledge
//! base (#1155).
//!
//! The extraction pass used to produce knowledge entries and only knowledge
//! entries, so a recurring routine - a sequence of things to do, in order - was
//! filed as a fact. That is procedural content in declarative memory, and it is
//! wrong three ways at once: it reads badly, because a numbered method rendered
//! as a fact is neither a good fact nor a runnable procedure; it surfaces
//! badly, because it competes for the knowledge arm's attention budget as
//! though it were a fact about the world; and it cannot be followed, because
//! nothing can execute it, attach scripts to it, or record that it was run.
//!
//! The rule, stated once and stated the same way everywhere:
//!
//! > Knowledge records what is true. A skill records how to do something. And a
//! > skill's preferences stay knowledge.
//!
//! So this module gives extraction a second destination. What it writes is a
//! **candidate**: unapproved, exactly like a skill the assistant promotes from
//! a completed plan, and inert until a person approves it. Extraction is
//! unattended and reads a transcript rather than a plan that worked, so it has
//! less evidence than the promotion path, not more.

use chrono::Utc;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::skill::{detect_kind, skill_content_hash, validate_skill_name};
use desktop_assistant_core::domain::{IndexedSkill, Locality, TrustTier};
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::skill_index::SkillIndexStore;
use desktop_assistant_core::skill_promotion::{
    MIN_PROMOTABLE_STEPS, PlanStep, PromotablePlan, render_skill_body, render_skill_md,
};
use sqlx::PgPool;

use crate::skill_index::PgSkillIndexStore;

/// `source` recorded on a skill the dream cycle extracted from a transcript,
/// so it is told apart from one the assistant promoted from its own plan and
/// from one a person put on disk.
///
/// Re-exported from `core` rather than declared here: the promotion path's
/// "may I revise this row?" rule recognises this marker, and two spellings of
/// it would let an unattended write reach a person's skill.
pub(crate) use desktop_assistant_core::skill_promotion::EXTRACTED_SOURCE;

/// The part of the extraction system prompt that sends a method to the skill
/// catalog rather than the fact list.
///
/// Kept beside the code that parses the answer, because the two are one
/// contract: a change to the shape asked for here is a change to
/// [`parse_extracted_skills`].
pub(crate) const SKILL_ROUTING_PROMPT: &str = "\
        ## A method is not a fact\n\
        \n\
        Knowledge records what is TRUE. A skill records HOW TO DO something. \
        When what you found is a method - ordered steps, a repeatable how-to - \
        return it in a `skills` array INSTEAD of writing it as a fact. A method \
        filed as a fact reads badly, competes with real facts for attention, \
        and cannot be followed by anything.\n\
        \n\
        Each skill has:\n\
        - `name` (string): a short kebab-case name, e.g. `weekly-status-report`.\n\
        - `description` (string): one or two sentences saying WHEN to use it. \
        This is what a later search matches on.\n\
        - `steps` (array of objects): the method, in order, at least 3 of them. \
        Each step has `goal` (what the step does) and `outcome` (what it \
        produces, or how you know it worked).\n\
        - `tags` (array of strings, optional).\n\
        \n\
        A skill's PREFERENCES stay knowledge. Put the method in the skill; put \
        which sources to use, in what order, what to skip, and what \"done\" \
        looks like for this person in `facts`. Those change independently of \
        the method, and they are exactly what a fact is for.\n\
        \n\
        Return `{\"skills\": []}` when you found no method. A skill written \
        this way is UNAPPROVED and is not followed until a person approves it.\n\
        \n";

/// One method as proposed by the extraction model, before validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExtractedSkill {
    /// Proposed skill name, before slugging.
    pub name: String,
    /// The "when to use" trigger.
    pub description: String,
    /// The method, in order.
    pub steps: Vec<PlanStep>,
    /// Proposed tags.
    pub tags: Vec<String>,
}

/// Read the `skills` array out of a parsed extraction response.
///
/// Tolerant in the same way [`super::extraction`]'s fact parsing is: a missing
/// array, a malformed entry, or a step with no goal is skipped rather than
/// failing the conversation. A model that returns only facts is the normal
/// case, not an error.
pub(crate) fn parse_extracted_skills(root: &serde_json::Value) -> Vec<ExtractedSkill> {
    let Some(items) = root.get("skills").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    items.iter().filter_map(parse_one_skill).collect()
}

fn parse_one_skill(value: &serde_json::Value) -> Option<ExtractedSkill> {
    let name = value.get("name")?.as_str()?.trim();
    let description = value.get("description")?.as_str()?.trim();
    if name.is_empty() || description.is_empty() {
        return None;
    }

    let steps: Vec<PlanStep> = value
        .get("steps")?
        .as_array()?
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            let goal = s.get("goal").and_then(|v| v.as_str()).unwrap_or("").trim();
            if goal.is_empty() {
                return None;
            }
            let outcome = s
                .get("outcome")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|o| !o.is_empty())
                .map(str::to_string);
            Some(PlanStep {
                key: (i + 1).to_string(),
                goal: goal.to_string(),
                outcome,
                abandoned: false,
            })
        })
        .collect();

    let tags = value
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|t| t.as_str())
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();

    Some(ExtractedSkill {
        name: name.to_string(),
        description: description.to_string(),
        steps,
        tags,
    })
}

/// Reduce a proposed name to something that can be a directory name.
///
/// The model writes prose ("Weekly status report"), and a skill name is also a
/// path segment wherever the catalog is later exported, so it is slugged and
/// then held to the same traversal guard the scanner uses.
pub(crate) fn slug_name(name: &str) -> Option<String> {
    let mut slug = String::with_capacity(name.len());
    let mut pending_dash = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            if pending_dash && !slug.is_empty() {
                slug.push('-');
            }
            pending_dash = false;
            slug.push(c.to_ascii_lowercase());
        } else {
            pending_dash = true;
        }
    }
    if slug.is_empty() || validate_skill_name(&slug).is_err() {
        return None;
    }
    Some(slug)
}

/// Turn a proposed method into a catalog row, or reject it.
///
/// Rejected when the name cannot be slugged, or when the method is shorter than
/// [`MIN_PROMOTABLE_STEPS`]. The step floor is the same number the promotion
/// path uses, deliberately: a two-step "method" is a pair of acts, and the
/// value of a skill is the ordering and the reasons.
pub(crate) fn to_indexed_skill(proposed: &ExtractedSkill) -> Option<IndexedSkill> {
    let name = slug_name(&proposed.name)?;
    if proposed.steps.len() < MIN_PROMOTABLE_STEPS {
        return None;
    }
    let plan = PromotablePlan {
        steps: proposed.steps.clone(),
    };
    let body = render_skill_body(&name, None, &plan);
    let skill_md = render_skill_md(&name, &proposed.description, &proposed.tags, &body);
    Some(IndexedSkill {
        name,
        description: proposed.description.clone(),
        kind: detect_kind(&body),
        // Catalog-only: extraction writes no file. The catalog is the
        // authoritative copy (#639), so the procedure still reads and searches.
        disk_path: String::new(),
        owner_user_id: Some(current_user_id().as_str().to_string()),
        locality: Locality::Daemon,
        content_hash: skill_content_hash(skill_md.as_bytes(), &[]),
        // Provenance, not consent. The store forces `approved_at` to NULL.
        trust_tier: TrustTier::Local,
        source: Some(EXTRACTED_SOURCE.to_string()),
        tags: proposed.tags.clone(),
        attachments: Vec::new(),
        body,
        metadata: serde_json::json!({"authored_from": "extraction"}),
        present_on_disk: false,
        last_seen_at: None,
        approved_at: None,
        approved_by: None,
    })
}

/// Write the methods one conversation's extraction found, and report how many
/// landed.
///
/// A method whose name is already held by an **approved** skill is skipped: a
/// person reviewed that body, and an unattended pass must not overwrite it.
/// An unapproved row of the same name is revised instead, which is what stops
/// every cycle adding another near-duplicate.
pub(crate) async fn write_extracted_skills(
    pool: &PgPool,
    proposals: &[ExtractedSkill],
) -> Result<usize, CoreError> {
    if proposals.is_empty() {
        return Ok(0);
    }
    let store = PgSkillIndexStore::new(pool.clone());
    let now = Utc::now();
    let mut written = 0usize;

    for proposed in proposals {
        let Some(skill) = to_indexed_skill(proposed) else {
            tracing::debug!(
                name = %proposed.name,
                steps = proposed.steps.len(),
                "dreaming: extracted method did not clear the skill bar"
            );
            continue;
        };
        match store.get(&skill.name, skill.owner_user_id.as_deref()).await {
            Ok(Some(existing)) if existing.is_approved() => {
                tracing::info!(
                    skill = %skill.name,
                    "dreaming: an approved skill of this name exists; leaving it alone"
                );
                continue;
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(skill = %skill.name, error = %e, "dreaming: skill lookup failed");
                continue;
            }
        }
        match store.write_authored(&skill, now).await {
            Ok(()) => {
                written += 1;
                tracing::info!(
                    skill = %skill.name,
                    steps = proposed.steps.len(),
                    "dreaming: recorded an extracted method as an unapproved skill"
                );
            }
            Err(e) => {
                tracing::warn!(skill = %skill.name, error = %e, "dreaming: skill write failed")
            }
        }
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::domain::SkillKind;

    fn proposal(steps: usize) -> serde_json::Value {
        let steps: Vec<serde_json::Value> = (1..=steps)
            .map(
                |i| serde_json::json!({"goal": format!("step {i}"), "outcome": format!("did {i}")}),
            )
            .collect();
        serde_json::json!({
            "skills": [{
                "name": "Weekly status report",
                "description": "Use when writing the Monday status note.",
                "steps": steps,
                "tags": ["routine"],
            }]
        })
    }

    #[test]
    fn a_response_with_no_skills_array_yields_nothing() {
        let root = serde_json::json!({"facts": [{"content": "a fact"}]});
        assert!(parse_extracted_skills(&root).is_empty());
    }

    #[test]
    fn a_method_is_parsed_into_ordered_steps() {
        let skills = parse_extracted_skills(&proposal(3));
        assert_eq!(skills.len(), 1);
        assert_eq!(
            skills[0].description,
            "Use when writing the Monday status note."
        );
        assert_eq!(skills[0].tags, vec!["routine"]);
        let keys: Vec<&str> = skills[0].steps.iter().map(|s| s.key.as_str()).collect();
        assert_eq!(
            keys,
            vec!["1", "2", "3"],
            "steps keep the order they were given"
        );
        assert_eq!(skills[0].steps[0].outcome.as_deref(), Some("did 1"));
    }

    #[test]
    fn a_step_with_no_goal_is_dropped() {
        let root = serde_json::json!({
            "skills": [{
                "name": "x", "description": "d",
                "steps": [{"goal": "one"}, {"outcome": "orphan"}, {"goal": "two"}]
            }]
        });
        assert_eq!(parse_extracted_skills(&root)[0].steps.len(), 2);
    }

    #[test]
    fn a_method_shorter_than_the_bar_is_not_a_skill() {
        let two = &parse_extracted_skills(&proposal(2))[0];
        assert!(
            to_indexed_skill(two).is_none(),
            "two steps is a pair of acts, not a method"
        );
    }

    #[test]
    fn an_extracted_method_becomes_an_unapproved_workflow() {
        let three = &parse_extracted_skills(&proposal(3))[0];
        let skill = to_indexed_skill(three).expect("three steps clears the bar");
        assert_eq!(
            skill.name, "weekly-status-report",
            "the prose name is slugged"
        );
        assert_eq!(skill.kind, SkillKind::Workflow);
        assert_eq!(skill.trust_tier, TrustTier::Local);
        assert!(!skill.is_approved(), "extraction is unattended authoring");
        assert!(!skill.present_on_disk);
        assert_eq!(skill.source.as_deref(), Some(EXTRACTED_SOURCE));
        assert!(skill.body.contains("step 1") && skill.body.contains("did 1"));
    }

    #[test]
    fn a_name_that_cannot_be_slugged_is_rejected() {
        for bad in ["", "...", "///", "   "] {
            assert_eq!(slug_name(bad), None, "expected {bad:?} to be rejected");
        }
        assert_eq!(slug_name("../etc/passwd"), Some("etc-passwd".to_string()));
        assert_eq!(
            slug_name("Deploy  The   App"),
            Some("deploy-the-app".to_string())
        );
    }

    #[test]
    fn the_routing_prompt_states_the_rule_and_the_shape() {
        // The prompt and the parser are one contract, so the prompt must ask
        // for exactly the keys `parse_extracted_skills` reads.
        for key in [
            "`skills`",
            "`name`",
            "`description`",
            "`steps`",
            "`goal`",
            "`outcome`",
        ] {
            assert!(
                SKILL_ROUTING_PROMPT.contains(key),
                "the prompt must name {key}"
            );
        }
        assert!(SKILL_ROUTING_PROMPT.contains("UNAPPROVED"));
    }
}

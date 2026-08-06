use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::clock::NowSnapshot;
use desktop_assistant_core::domain::{Role, ToolDefinition, ToolRunner};
use desktop_assistant_core::ports::client_tools::current_client_tools;
use desktop_assistant_core::ports::conversation_ctx::current_conversation_id;
use desktop_assistant_core::ports::conversation_search::ConversationSearchFn;
use desktop_assistant_core::ports::database::DbQueryFn;
use desktop_assistant_core::ports::embedding::{EMBED_TIMEOUT, EmbedFn};
use desktop_assistant_core::ports::knowledge::{
    AVAILABLE_TAGS_LIMIT, KNOWLEDGE_TAG_CENSUS_SAMPLE, KnowledgeDeleteFn, KnowledgeGetFn,
    KnowledgeListFn, KnowledgeListQuery, KnowledgeSearchFn, KnowledgeTagResolveFn,
    KnowledgeWriteFn, ListOrder, ListOrderOpt, ProposedTag, ScopeSize,
};
use desktop_assistant_core::ports::notify::{NotifyFn, NotifyUrgency};
use desktop_assistant_core::ports::scratchpad::{
    MAX_KEYS_PER_CALL, MAX_NOTE_BYTES, MAX_NOTES_PER_WRITE, MAX_PINNED_NOTES, MAX_RESULTS_CEILING,
    NewScratchpadNote, RESPONSE_BYTE_BUDGET, ScratchpadClearFn, ScratchpadDeleteManyFn,
    ScratchpadGetManyFn, ScratchpadListFn, ScratchpadSearchFn, ScratchpadSetPinnedFn,
    ScratchpadWriteFn, plan_pin,
};
use desktop_assistant_core::ports::skill_index::{SkillGetFn, SkillSearchFn};
use desktop_assistant_core::ports::tool_registry::{ToolDefinitionFn, ToolSearchFn};
use desktop_assistant_core::ports::transport::{
    current_client_context, current_client_label, current_co_location, current_transport_kind,
};
use desktop_assistant_core::tag_normalize::normalize_tag;

use crate::executor::McpControlHandle;

/// How long one `builtin_knowledge_base_write` call may spend **inside** the
/// tag vocabulary, added up across every tag in the call.
///
/// Why a budget at all: the caller chooses how many tags one write carries, and
/// each tag the vocabulary has not seen before costs an embedding. Without a
/// ceiling the wait a person sits through grows with the tag count, inside a
/// live turn. Once it is spent the remaining tags are stored as written, which
/// is the same fallback every other absent-vocabulary state uses.
///
/// Why time spent rather than a wall-clock deadline: a write also reads an
/// existing entry and stores each entry, and neither is a consultation. Against
/// a deadline a batch of ten entries with slow stores loses the vocabulary on
/// its last entries with nothing slow about the vocabulary, and the re-tag path
/// is the most exposed because it reads first.
///
/// It gates the start of a consultation, not its end, so one call already in
/// flight when the budget runs out still finishes. Cutting a consultation off
/// mid-flight would spend the embedding and then throw the answer away. The
/// vocabulary's whole share of a write is therefore this plus at most one
/// [`TAG_RESOLVE_CALL_CEILING`].
const TAG_RESOLVE_BUDGET: std::time::Duration = std::time::Duration::from_secs(15);

/// How long one consultation of the tag vocabulary may take before the write
/// gives up on it and stores that tag as written.
///
/// Why the whole call and not only its embedding: a consultation reads the
/// vocabulary, embeds, searches for a near neighbour, and registers. The
/// embedding is bounded on its own, and the database round trips around it are
/// bounded by the connection pool's acquire timeout, which is measured in tens
/// of seconds and is paid once per round trip. A saturated pool therefore held
/// a live turn far longer than the embedding timeout suggests. This bounds the
/// consultation as a whole, whatever inside it is slow.
///
/// The value leaves the embedding timeout its full 5 seconds and 5 more for the
/// round trips around it.
const TAG_RESOLVE_CALL_CEILING: std::time::Duration = std::time::Duration::from_secs(10);

/// How long one tag description may be, in characters.
///
/// Why bounded at all: a description is written once and read forever. It is
/// stored on the registry row, and `build_extraction_system_prompt` renders
/// every active tag's description into the dreaming extraction prompt in full.
/// Nothing deletes a registry row, so an oversized description is a permanent
/// charge on every later extraction. Before the tool path was gated, dreaming
/// was the only writer; now every knowledge write is one, so the rate that
/// surface grows at changes by orders of magnitude.
///
/// Why this size: the field is advertised as one line, and
/// [`desktop_assistant_core::ports::knowledge::AVAILABLE_TAGS_LIMIT`] already
/// treats 50 tags as the working size of a vocabulary. Fifty descriptions of
/// 200 characters is about 10 kB of prompt - bounded, and still generous for
/// one line. A longer description is truncated, never refused: refusing it
/// would cost the tag its description and drop the dedup back to matching bare
/// names, which is the weak option #1070 rejected.
///
/// Bounding how many tags that prompt renders is the other half of the same
/// surface, and is #1103.
const TAG_DESCRIPTION_MAX_CHARS: usize = 200;

/// The tag vocabulary's share of one `builtin_knowledge_base_write` call.
///
/// It carries the time already spent consulting, the reason the vocabulary
/// stopped being consulted once anything has stopped it, and whether any tag
/// went into the store without a vocabulary answer. Every stopping condition -
/// no vocabulary wired, a failure, a spent budget - means the same thing for
/// the rest of the call: store the tags as the caller wrote them.
///
/// Why it spans the call rather than one entry: a batch write is one tool call
/// to the person waiting on it, so a per-entry budget would bound nothing that
/// they can feel.
#[derive(Default)]
struct TagGateBudget {
    spent: std::time::Duration,
    stopped: Option<String>,
    unchecked_tags: usize,
}

impl TagGateBudget {
    /// Start the budget for one write call.
    fn new() -> Self {
        Self::default()
    }

    /// Whether the vocabulary may still be consulted.
    fn is_open(&self) -> bool {
        self.stopped.is_none() && self.spent < TAG_RESOLVE_BUDGET
    }

    /// Charge one finished consultation against the budget.
    fn charge(&mut self, elapsed: std::time::Duration) {
        self.spent = self.spent.saturating_add(elapsed);
    }

    /// Record that `count` tags were stored without a vocabulary answer.
    fn unchecked(&mut self, count: usize) {
        self.unchecked_tags += count;
    }

    /// Stop consulting the vocabulary for the rest of this call, keeping the
    /// first reason. A later reason is a consequence of the first, so it would
    /// only bury it.
    fn stop(&mut self, reason: impl Into<String>) {
        if self.stopped.is_none() {
            self.stopped = Some(reason.into());
        }
    }

    /// The value of the write response's `tag_check` field, or `None` when the
    /// field is left out because the vocabulary answered for every tag.
    ///
    /// Why the caller is told at all: a degraded write stores whatever the
    /// model wrote, so its response would otherwise be byte-identical to a
    /// checked one. The model would then read its own drift back as accepted
    /// vocabulary, which is the failure the vocabulary exists to prevent. This
    /// mirrors `builtin_knowledge_base_search` reporting `scope_size` as
    /// `UNKNOWN` when its census did not run: across both tools, not measured
    /// never reads as measured.
    fn tag_check(&self) -> Option<&'static str> {
        (self.unchecked_tags > 0).then_some(TAG_CHECK_UNKNOWN)
    }

    /// Say once, for the whole call, that the tags were stored as written.
    fn report(&self) {
        if let Some(reason) = &self.stopped {
            tracing::warn!(
                reason = %reason,
                unchecked_tags = self.unchecked_tags,
                "the tag vocabulary was not consulted for the rest of this write; \
                 those tags are stored as written"
            );
        }
    }
}

/// The one value the write response's `tag_check` field takes: at least one tag
/// on this write went to the store without the vocabulary answering for it.
const TAG_CHECK_UNKNOWN: &str = "UNKNOWN";

/// Machine label used until the daemon supplies its own hostname, so a search
/// result is coherent in tests and in a build that never called
/// `BuiltinToolService::with_topology`.
const DEFAULT_DAEMON_HOST: &str = "this machine";

/// How many daemon-side hits the tool registry is asked for.
const REGISTRY_SEARCH_LIMIT: usize = 10;

/// How many client-registered tools one search may return.
///
/// A connection registers tens of tools at most, not thousands, so this is a
/// backstop rather than a working limit. A search that hits it reports the
/// number it dropped.
const DEVICE_SEARCH_LIMIT: usize = 10;

/// Shortest word the client-tool matcher will consider, in characters.
///
/// Below this a "term" is a preposition or an article, which matches nearly
/// every description and therefore separates nothing.
const MIN_MATCH_TERM_CHARS: usize = 3;

/// Split text into lowercase alphanumeric words worth matching on.
fn match_terms(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= MIN_MATCH_TERM_CHARS)
        .map(|w| w.to_lowercase())
        .collect()
}

/// Whether two words describe the same thing closely enough to count as a hit.
///
/// A prefix match in either direction, so "file" finds "files" and "run" finds
/// "running". Deliberately cruder than a stemmer: the set being searched is
/// small, and a wrong extra hit costs the model one line to read, while a
/// missed hit costs it the only tool that acts on the user's own machine.
fn terms_agree(a: &str, b: &str) -> bool {
    a.starts_with(b) || b.starts_with(a)
}

/// Rank client-registered tools against a search query.
///
/// Scores each tool by how many distinct query terms appear in its name or its
/// description, keeps those that match at least one, and orders by score then
/// by name so the result never depends on registration order. Returns the kept
/// tools and the number dropped by `limit`, because a truncated set presented
/// as the whole answer reads as "nothing else matched".
///
/// Lexical rather than vector-based: client tools are registered per connection
/// and never indexed, so no embedding exists for them, and computing one per
/// search would put an embedding round-trip in front of every discovery.
fn match_client_tools<'a>(
    query: &str,
    tools: &'a [ToolDefinition],
    limit: usize,
) -> (Vec<&'a ToolDefinition>, usize) {
    let query_terms = match_terms(query);
    if query_terms.is_empty() {
        return (Vec::new(), 0);
    }
    let mut scored: Vec<(usize, &'a ToolDefinition)> = tools
        .iter()
        .filter_map(|tool| {
            let haystack = match_terms(&format!("{} {}", tool.name, tool.description));
            let score = query_terms
                .iter()
                .filter(|q| haystack.iter().any(|h| terms_agree(q, h)))
                .count();
            (score > 0).then_some((score, tool))
        })
        .collect();
    // Descending score, then ascending name: a stable, explainable order that
    // does not depend on how the client happened to register its tools.
    scored.sort_by(|(a_score, a), (b_score, b)| b_score.cmp(a_score).then(a.name.cmp(&b.name)));
    let dropped = scored.len().saturating_sub(limit);
    scored.truncate(limit);
    (scored.into_iter().map(|(_, tool)| tool).collect(), dropped)
}

const TOOL_KB_WRITE: &str = "builtin_knowledge_base_write";
const TOOL_KB_SEARCH: &str = "builtin_knowledge_base_search";
/// Largest `limit` `builtin_knowledge_base_search` will honour, matching the
/// cap `builtin_knowledge_base_list` already advertises.
///
/// Why a cap at all: the storage layer over-fetches with `limit * 2` to feed
/// the RRF fusion, so an unbounded caller-supplied limit overflows there rather
/// than merely returning a large page.
const KB_SEARCH_MAX_LIMIT: u64 = 500;

const TOOL_KB_DELETE: &str = "builtin_knowledge_base_delete";
const TOOL_KB_LIST: &str = "builtin_knowledge_base_list";
const TOOL_SEARCH: &str = "builtin_tool_search";
const TOOL_NOTIFY: &str = "builtin_notify";
const TOOL_SYS_PROPS: &str = "builtin_sys_props";
const TOOL_DB_QUERY: &str = "builtin_db_query";
const TOOL_MCP_CONTROL: &str = "builtin_mcp_control";
const TOOL_CONV_SEARCH: &str = "builtin_conversation_search";
const TOOL_SCRATCHPAD_WRITE: &str = "builtin_scratchpad_write";
const TOOL_SCRATCHPAD_SEARCH: &str = "builtin_scratchpad_search";
const TOOL_SCRATCHPAD_DELETE: &str = "builtin_scratchpad_delete";
const TOOL_SCRATCHPAD_PIN: &str = "builtin_scratchpad_pin";
const TOOL_SKILL_SEARCH: &str = "builtin_skill_search";
const TOOL_SKILL_GET: &str = "builtin_skill_get";

/// Marker passed as `SkillGetFn`'s `owner` argument to mean "the caller's
/// own scope". Per the port contract (#911), every implementation --
/// `PgSkillIndexStore::get`, `SqliteSkillIndexStore::get`, and the in-memory
/// reference implementation -- resolves "the caller's own" from
/// `current_user_id()`, never from this string, so its literal content is
/// irrelevant; only its `Some`-ness is. It exists purely so
/// [`BuiltinToolService::skill_get`]'s call site reads as intent, not a
/// magic string.
const SKILL_GET_OWN_SCOPE: &str = "self";

/// The active embedding backend: the function that turns a query into a vector,
/// together with the identifier of the model behind it.
///
/// One field rather than two because the searches scope their vector arm to the
/// model that produced the query vector, so a vector paired with another
/// model's identifier searches the wrong rows -- and across a dimension change,
/// raises. Holding the pair in one place means it can only ever be replaced as
/// a unit.
struct EmbeddingBackend {
    embed: EmbedFn,
    model: String,
}

pub struct BuiltinToolService {
    embedding: Option<EmbeddingBackend>,
    kb_write_fn: Option<KnowledgeWriteFn>,
    kb_search_fn: Option<KnowledgeSearchFn>,
    kb_delete_fn: Option<KnowledgeDeleteFn>,
    kb_list_fn: Option<KnowledgeListFn>,
    kb_get_fn: Option<KnowledgeGetFn>,
    kb_tag_resolve_fn: Option<KnowledgeTagResolveFn>,
    tool_search_fn: Option<ToolSearchFn>,
    #[allow(dead_code)]
    tool_definition_fn: Option<ToolDefinitionFn>,
    db_query_fn: Option<DbQueryFn>,
    mcp_handle: Option<McpControlHandle>,
    conversation_search_fn: Option<ConversationSearchFn>,
    scratchpad_write_fn: Option<ScratchpadWriteFn>,
    scratchpad_get_many_fn: Option<ScratchpadGetManyFn>,
    scratchpad_list_fn: Option<ScratchpadListFn>,
    scratchpad_search_fn: Option<ScratchpadSearchFn>,
    scratchpad_delete_many_fn: Option<ScratchpadDeleteManyFn>,
    scratchpad_clear_fn: Option<ScratchpadClearFn>,
    scratchpad_set_pinned_fn: Option<ScratchpadSetPinnedFn>,
    notify_fn: Option<NotifyFn>,
    skill_search_fn: Option<SkillSearchFn>,
    skill_get_fn: Option<SkillGetFn>,
    /// The daemon's own machine, as named to the model in a tool-search result.
    daemon_host: String,
    /// Whether that machine is the user's own workstation. When it is not, a
    /// search result says so, because a daemon-side file tool then acts on a
    /// machine the user is not sitting at.
    daemon_on_workstation: bool,
}

impl Default for BuiltinToolService {
    fn default() -> Self {
        Self::new()
    }
}

impl BuiltinToolService {
    /// Create a minimal BuiltinToolService with no backing stores.
    /// KB and tool_search calls will return errors until closures are configured.
    pub fn new() -> Self {
        Self {
            embedding: None,
            kb_write_fn: None,
            kb_search_fn: None,
            kb_delete_fn: None,
            kb_list_fn: None,
            kb_get_fn: None,
            kb_tag_resolve_fn: None,
            tool_search_fn: None,
            tool_definition_fn: None,
            db_query_fn: None,
            mcp_handle: None,
            conversation_search_fn: None,
            scratchpad_write_fn: None,
            scratchpad_get_many_fn: None,
            scratchpad_list_fn: None,
            scratchpad_search_fn: None,
            scratchpad_delete_many_fn: None,
            scratchpad_clear_fn: None,
            scratchpad_set_pinned_fn: None,
            notify_fn: None,
            skill_search_fn: None,
            skill_get_fn: None,
            daemon_host: DEFAULT_DAEMON_HOST.to_string(),
            daemon_on_workstation: true,
        }
    }

    /// Name the machine the daemon runs on, and say whether it is the user's
    /// own workstation (issue #1082).
    ///
    /// Why: a tool-search result tells the model what each hit acts on. A
    /// daemon-side file tool acts on the user's files when the daemon runs on
    /// their computer, and on a container's files when it does not. Without
    /// this the result can only say "the daemon", which the model reads as
    /// "here".
    pub fn with_topology(mut self, host: impl Into<String>, on_workstation: bool) -> Self {
        self.daemon_host = host.into();
        self.daemon_on_workstation = on_workstation;
        self
    }

    /// Configure the embedding function for generating query vectors, and the
    /// identifier of the model behind it.
    ///
    /// The two are set together because every vector search scopes itself to
    /// the model that produced its query vector; a function paired with another
    /// model's identifier would silently search the wrong rows.
    pub fn with_embedding(mut self, embed_fn: EmbedFn, model_identifier: String) -> Self {
        self.embedding = Some(EmbeddingBackend {
            embed: embed_fn,
            model: model_identifier,
        });
        self
    }

    /// Configure the desktop-notification closure (#`builtin_notify`).
    ///
    /// Capability-gated: the daemon only calls this when a notification
    /// service is present on the session bus, so the tool is simply absent on a
    /// headless host rather than failing at call time.
    pub fn with_notify(mut self, notify_fn: NotifyFn) -> Self {
        self.notify_fn = Some(notify_fn);
        self
    }

    /// Configure the skill-library closures (`builtin_skill_search` /
    /// `builtin_skill_get`). Wired only when a skill index is available, so the
    /// tools are simply absent otherwise (capability-gated like `builtin_notify`).
    pub fn with_skills(mut self, search_fn: SkillSearchFn, get_fn: SkillGetFn) -> Self {
        self.skill_search_fn = Some(search_fn);
        self.skill_get_fn = Some(get_fn);
        self
    }

    /// Configure knowledge base store closures.
    pub fn with_knowledge_base(
        mut self,
        write_fn: KnowledgeWriteFn,
        search_fn: KnowledgeSearchFn,
        delete_fn: KnowledgeDeleteFn,
        list_fn: KnowledgeListFn,
        get_fn: KnowledgeGetFn,
    ) -> Self {
        self.kb_write_fn = Some(write_fn);
        self.kb_search_fn = Some(search_fn);
        self.kb_delete_fn = Some(delete_fn);
        self.kb_list_fn = Some(list_fn);
        self.kb_get_fn = Some(get_fn);
        self
    }

    /// Put the formal tag vocabulary in front of the knowledge-base write path,
    /// so a tag the model writes is checked against the tags that already exist
    /// before it is stored.
    ///
    /// Why capability-gated: the vocabulary lives in the database and needs an
    /// embedding backend to recognise a near duplicate. Without this the write
    /// path keeps its prior behaviour and stores the tag as written, which is a
    /// weaker knowledge base rather than a broken one.
    pub fn with_tag_registry(mut self, resolve_fn: KnowledgeTagResolveFn) -> Self {
        self.kb_tag_resolve_fn = Some(resolve_fn);
        self
    }

    /// Configure tool registry closures.
    pub fn with_tool_registry(
        mut self,
        search_fn: ToolSearchFn,
        definition_fn: ToolDefinitionFn,
    ) -> Self {
        self.tool_search_fn = Some(search_fn);
        self.tool_definition_fn = Some(definition_fn);
        self
    }

    /// Configure the database-query closure for the `builtin_db_query`
    /// tool.
    ///
    /// ## Security posture (issue #141)
    ///
    /// The closure runs *arbitrary* LLM-supplied SQL. The implementation
    /// behind it (see `desktop_assistant_storage::execute_database_query`)
    /// enforces the following invariants before any text reaches the
    /// pool, so it is safe to wire the tool against the same pool used
    /// for ordinary application traffic:
    ///
    /// - **SELECT-only on the read path.** Only single-statement
    ///   `SELECT` / `WITH` / `TABLE` / `VALUES` / `EXPLAIN` queries
    ///   are accepted; everything else is parsed-and-rejected.
    /// - **Per-user (`user_id`) scoping by AST rewrite.** Every
    ///   reference to a personal-data table (`conversations`,
    ///   `messages`, `knowledge_base`, etc.) has a
    ///   `<table>.user_id = $N` predicate grafted into its `WHERE`
    ///   clause, bound to the caller's task-local `UserId`. An
    ///   LLM-supplied predicate naming a different user_id is AND'd
    ///   with the grafted one, so the intersection is empty.
    /// - **Compound statements rejected.** `SELECT 1; DROP TABLE …`
    ///   produces two statements at parse time and is refused.
    /// - **Writes confined to scratch.** The write path is an
    ///   allowlist in both dimensions: the statement kind must be one
    ///   of INSERT / UPDATE / DELETE / TRUNCATE / CREATE
    ///   TABLE|VIEW|INDEX / ALTER TABLE / DROP TABLE|VIEW|INDEX /
    ///   COMMENT (anything else, `CREATE FUNCTION` and `CREATE
    ///   SCHEMA` included, is refused), and every object it names —
    ///   target, source query, subquery, CTE or function, at any
    ///   depth — must be unqualified or `scratch`-qualified. The
    ///   statement then runs as the un-privileged `adele_query` role
    ///   with `search_path` pinned to `scratch` alone, so neither a
    ///   qualified name nor an unqualified one can reach `public`.
    ///
    /// Pre-#141 this docstring contained a single-line "read-only"
    /// claim — which the implementation did not enforce. The audit
    /// test `comment_in_builtin_rs_matches_actual_security_posture`
    /// in this file pins the wording against that regression.
    pub fn with_database(mut self, query_fn: DbQueryFn) -> Self {
        self.db_query_fn = Some(query_fn);
        self
    }

    /// Configure the past-conversation full-text search closure (#71).
    /// When unset, `builtin_conversation_search` returns a clear error
    /// rather than silently no-op-ing.
    pub fn with_conversation_search(mut self, search_fn: ConversationSearchFn) -> Self {
        self.conversation_search_fn = Some(search_fn);
        self
    }

    /// Configure the per-conversation scratchpad store closures (#184). The
    /// builtin tools resolve the active conversation from the task-local
    /// installed by the service dispatch loop; these closures forward to the
    /// store. When unset, the scratchpad tools return a clear error.
    #[allow(clippy::too_many_arguments)]
    pub fn with_scratchpad(
        mut self,
        write_fn: ScratchpadWriteFn,
        get_many_fn: ScratchpadGetManyFn,
        list_fn: ScratchpadListFn,
        search_fn: ScratchpadSearchFn,
        delete_many_fn: ScratchpadDeleteManyFn,
        clear_fn: ScratchpadClearFn,
    ) -> Self {
        self.scratchpad_write_fn = Some(write_fn);
        self.scratchpad_get_many_fn = Some(get_many_fn);
        self.scratchpad_list_fn = Some(list_fn);
        self.scratchpad_search_fn = Some(search_fn);
        self.scratchpad_delete_many_fn = Some(delete_many_fn);
        self.scratchpad_clear_fn = Some(clear_fn);
        self
    }

    /// Wire the pin/unpin write (#597). Additive and separate from
    /// [`Self::with_scratchpad`] so an embedder that predates pinning keeps
    /// compiling; without it, the pin tool is simply not advertised.
    pub fn with_scratchpad_pin(mut self, set_pinned_fn: ScratchpadSetPinnedFn) -> Self {
        self.scratchpad_set_pinned_fn = Some(set_pinned_fn);
        self
    }

    /// Set the MCP control handle (used by builtin_mcp_control tool).
    pub fn set_mcp_control(&mut self, handle: McpControlHandle) {
        self.mcp_handle = Some(handle);
    }

    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        let mut defs = vec![
            ToolDefinition::new(
                TOOL_KB_WRITE,
                "Write or update knowledge base entries. Use for storing preferences, facts, \
                 instructions, project context, or any durable information the user wants remembered. \
                 Content should be self-contained prose that describes both the context (when/why \
                 this information is useful) and the information itself. Provide either a single \
                 entry (top-level `content`/`tags`/`id`) or a batch via `entries`. A write that \
                 gives the `id` of an existing entry keeps the content, the tags and the stored \
                 metadata it leaves out, so send only what changes: `id` plus `content` rewrites \
                 the text and keeps the tags, and `id` plus `tags` re-tags the entry and keeps \
                 the text. To clear the tags, send an empty `tags` list. One field is not kept: \
                 every write through this tool records the entry's provenance as 'explicit', so \
                 an entry that dreaming had extracted or consolidated counts as explicitly saved \
                 from then on, and `builtin_knowledge_base_list` with that `source` filter stops \
                 listing it. Tags are checked against the tag vocabulary, which holds \
                 the tags registered so far and grows as tags are written; it starts empty, so \
                 early writes have little to match against. A tag that means the same as one \
                 already registered is stored under the registered name, so the response \
                 reports the tags actually stored, which may differ from the ones you sent. \
                 Describe any tag you believe is new in `new_tag_descriptions` — that \
                 description is what the check compares. When the response carries \
                 `tag_check: \"UNKNOWN\"`, at least one tag on that write was stored without \
                 being checked, so treat those tags as your own wording and not as established \
                 vocabulary.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Self-contained prose describing the context and information. \
                                            Write naturally, e.g. 'The user lives at 123 Main St, Springfield. \
                                            Use this as their default location for weather, directions, and local searches.' \
                                            Do not use key-value format. Optional when `id` is given (tags-only update)."
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Two-level tags. Give a coarse KIND ('preference', 'memory', or 'instruction') PLUS at least one SPECIFIC facet: 'project:<name>', 'tool:<name>', 'topic:<subject>', or 'person:<name>'. Prefer specific over generic. Good: ['instruction', 'project:adelie-ai', 'topic:deploy']. Too generic: ['instruction']."
                        },
                        "new_tag_descriptions": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                            "description": "Optional map from a tag in `tags` to a one-line description of what that tag means, e.g. {'topic:embeddings': 'Vector embedding generation, models, and backfill'}. Needed only for a tag you believe is new: it is what decides whether your tag means the same as one already in use. A tag already in use ignores its entry here. Omitting a description never fails the write, it only makes the check weaker. Keep it to one line: a description longer than 200 characters is truncated to 200, never rejected."
                        },
                        "id": {
                            "type": "string",
                            "description": "Optional ID. Omit to create a new entry with a generated \
                                            id, which is what almost every write should do. Give \
                                            the id of an existing entry to update it, and the \
                                            fields you leave out keep the values that entry \
                                            already holds. An id no entry holds creates the entry \
                                            at that id, so a write that carries `content` can be \
                                            repeated safely. An id whose entry was retired is \
                                            refused instead: the write does not revive it, so \
                                            store the text as a new entry with no id. An id you \
                                            make up yourself must be a fresh random identifier \
                                            such as a UUID, never a readable name like \
                                            'user-coffee-preference': ids are not yours to name, \
                                            so a readable one may already be taken, and the write \
                                            then fails instead of storing anything."
                        },
                        "entries": {
                            "type": "array",
                            "description": "Batch form: a list of {content?, tags?, id?, new_tag_descriptions?} objects. When present, every top-level field is ignored — content, tags, id and new_tag_descriptions alike — so describe each entry's new tags inside that entry.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string"},
                                    "tags": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Two-level tags: a coarse KIND ('preference'/'memory'/'instruction') PLUS at least one SPECIFIC facet ('project:<name>', 'tool:<name>', 'topic:<subject>', 'person:<name>'). Prefer specific over generic."
                                    },
                                    "new_tag_descriptions": {
                                        "type": "object",
                                        "additionalProperties": {"type": "string"},
                                        "description": "Per-entry map from a tag in this entry's `tags` to a one-line description of what it means. Needed only for a tag you believe is new. A description longer than 200 characters is truncated to 200, never rejected."
                                    },
                                    "id": {"type": "string"}
                                }
                            }
                        }
                    }
                }),
            ),
            ToolDefinition::new(
                TOOL_KB_SEARCH,
                format!(
                    "Search the knowledge base for preferences, memories, and stored context. \
                     Uses hybrid vector + full-text search. Returns `results`, `returned` (how \
                     many entries are in this page), `scope_size`, and `available_tags`; plus \
                     `truncated` and a `message` when the page filled up and entries were left \
                     behind - a full page under FEW carries neither, because FEW already means \
                     you have the whole scope. `scope_size` is NONE \
                     (no entry passes the filters you supplied - retry without them, the store \
                     may still hold plenty), FEW (the scope is no larger than this page, so a \
                     `builtin_knowledge_base_list` sweep would show all of it), MANY (the \
                     scope holds more than this page could show), or UNKNOWN (the scope could \
                     not be measured this time - treat it as NO INFORMATION about the store, \
                     never as an empty one, and judge the page on `results` alone). FEW does \
                     not mean you have seen everything: the page holds what matched the query, \
                     the scope is what passed the filters, so a query that matched nothing \
                     still reports FEW when the scope is small. `available_tags` lists tag \
                     names most frequent first, without counts, and is empty under UNKNOWN. \
                     Both describe the SCOPE - the entries that pass the `tags` \
                     and `exclude_tags` filters - and never the number of entries that matched \
                     the query, which this search cannot count. `available_tags` reports at most \
                     {AVAILABLE_TAGS_LIMIT} tags, counted over at most the \
                     {KNOWLEDGE_TAG_CENSUS_SAMPLE} most recent entries in scope, so a tag \
                     carried only by older entries can be missing from it."
                ),
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Only return entries carrying at least one of these tags"
                        },
                        "exclude_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude entries carrying any of these tags"
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": KB_SEARCH_MAX_LIMIT,
                            "description": "Max results (default 10)"
                        }
                    },
                    "required": ["query"]
                }),
            ),
            ToolDefinition::new(
                TOOL_KB_DELETE,
                "Delete knowledge base entries by ID. Accepts a single `id` or a list of `ids`.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "ID of a single entry to delete"
                        },
                        "ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "IDs of multiple entries to delete in one call"
                        }
                    }
                }),
            ),
            ToolDefinition::new(
                TOOL_KB_LIST,
                "List knowledge base entries without a search query — a straight paginated \
                 enumeration for audits and review. Returns entries plus a `next_cursor`; pass it \
                 back as `cursor` to fetch the next page (null when there are no more).",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 500,
                            "description": "Max entries per page (default 50)"
                        },
                        "cursor": {
                            "type": "string",
                            "description": "Opaque pagination cursor from a previous page's next_cursor. Omit for the first page."
                        },
                        "order": {
                            "type": "string",
                            "enum": ["newest_first", "oldest_first"],
                            "description": "Sort direction by creation time (default newest_first)"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Only include entries carrying at least one of these tags"
                        },
                        "exclude_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude entries carrying any of these tags"
                        },
                        "source": {
                            "type": "string",
                            "description": "Only include entries with this provenance: 'extraction', 'consolidation', or 'explicit'"
                        }
                    }
                }),
            ),
            ToolDefinition::new(
                TOOL_SEARCH,
                "Search for available tools by description. Use this when the user's request \
                 might require a tool that isn't in your current set. Returns tool names and \
                 descriptions; matched tools become available automatically.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What kind of tool are you looking for?"
                        }
                    },
                    "required": ["query"]
                }),
            ),
            ToolDefinition::new(
                TOOL_SYS_PROPS,
                "Return a compact property sheet with basic runtime/system context",
                serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
            ),
            ToolDefinition::new(
                TOOL_DB_QUERY,
                "Execute a SQL query against the assistant's PostgreSQL database. \
                 Use this to inspect your own conversations, messages, knowledge base \
                 entries, tool definitions, and other stored data.\n\n\
                 Reads: SELECT/WITH/TABLE/VALUES/EXPLAIN run in a read-only \
                 transaction and are automatically scoped to the current user. Every \
                 schema is readable.\n\n\
                 Writes: confined to the `scratch` schema, which is your staging area \
                 for intermediate joins, working sets and materializations. Unqualified \
                 writes land there; `scratch.<name>` is equivalent. Permitted \
                 statements are INSERT, UPDATE, DELETE, TRUNCATE, CREATE TABLE / VIEW / \
                 MATERIALIZED VIEW / INDEX, ALTER TABLE, DROP TABLE / VIEW / INDEX, and \
                 COMMENT ON; they run in a normal transaction and are committed.\n\n\
                 Refused: any write that names an object outside `scratch` (including \
                 in a subquery, CTE or source SELECT — read application data in a \
                 separate SELECT instead), and every other statement kind, such as \
                 CREATE SCHEMA, CREATE FUNCTION, GRANT and COPY. See the database \
                 design section of your system prompt for conventions (naming, COMMENT \
                 ON, what belongs in the knowledge base instead).",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query to execute"
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 500,
                            "description": "Maximum rows to return for SELECT queries (default 100). Ignored for write queries."
                        }
                    },
                    "required": ["query"]
                }),
            ),
            ToolDefinition::new(
                TOOL_CONV_SEARCH,
                "Search past conversations by full-text query. Useful for \
                 recalling what was discussed, what decisions were made, or \
                 finding a specific exchange. Returns matching messages \
                 with conversation title, ordinal, role, content, a \
                 highlighted snippet around the match, and a relevance \
                 rank. Hits where the conversation title or summary \
                 matches surface even if no individual message text does. \
                 Use this when the user asks about prior conversations \
                 (\"what did we discuss about X\", \"find where we talked \
                 about Y\").",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Full-text search query (English tsvector). Multi-word phrases are AND-ed."
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                            "description": "Max hits to return (default 10)."
                        },
                        "role": {
                            "type": "string",
                            "enum": ["user", "assistant"],
                            "description": "Restrict matches to a specific role (omit to search all)."
                        }
                    },
                    "required": ["query"]
                }),
            ),
            ToolDefinition::new(
                TOOL_MCP_CONTROL,
                "Check status, start, stop, or restart MCP (Model Context Protocol) \
                 servers. Use this when a tool call fails because an MCP server is \
                 disconnected, or to inspect what servers are available.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["status", "start", "stop", "restart"],
                            "description": "Action to perform"
                        },
                        "server": {
                            "type": "string",
                            "description": "Server name (omit for all servers)"
                        }
                    },
                    "required": ["action"]
                }),
            ),
            ToolDefinition::new(
                TOOL_SCRATCHPAD_WRITE,
                "Add or update notes in this conversation's scratchpad — an ephemeral, \
                 per-conversation working store for facts you want to keep high in context \
                 right now (an evolving plan, open questions, a working set of IDs). Use it \
                 SELECTIVELY: only when you need to carry information forward across a large or \
                 multi-step task (a multi-step plan, investigation notes you'll reference later, \
                 intermediate results held across many turns). For small one-shot tasks — a \
                 single question, quick lookup, or one-line action — don't write here; just \
                 answer or act. Notes are \
                 keyed; writing the same key again replaces it. Pass `notes` to upsert several \
                 at once. Use the reserved key 'goal' for the current objective: it is \
                 auto-surfaced as your task anchor every turn (so it survives compaction), and \
                 you should evolve it as the goal shifts and delete it when done. Each note has \
                 a `type` (default \"note\") and an optional integer `sequence` (same-type notes \
                 sort by it). For working a multi-step task, prefer the begin_step / complete_step \
                 tools — they record and number your plan as todos for you and compact each finished \
                 step's raw work into a note — rather than hand-managing `todo` notes here. The \
                 scratchpad is discarded when the conversation is deleted and is NOT durable across \
                 conversations — promote anything worth keeping to the knowledge base with \
                 builtin_knowledge_base_write, then delete the note here.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "notes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "key": {"type": "string", "description": "Short handle for the note; upserts by key."},
                                    "content": {"type": "string", "description": "The note body (keep it small and high-signal)."},
                                    "type": {"type": "string", "description": "Category, e.g. \"todo\"/\"note\"/\"other\". Defaults to \"note\". Used for filtering/grouping; same-type notes sort by `sequence`."},
                                    "sequence": {"type": "integer", "description": "Optional ordering hint within the type (ascending). Use for ordered todos."},
                                    "done": {"type": "boolean", "description": "Whether this note (e.g. a todo) is checked off. Defaults to false."}
                                },
                                "required": ["key", "content"]
                            },
                            "description": "One or more notes to add/update in a single call."
                        },
                        "key": {"type": "string", "description": "Single-note convenience: the note key (use with `content`)."},
                        "content": {"type": "string", "description": "Single-note convenience: the note body (use with `key`)."},
                        "type": {"type": "string", "description": "Single-note convenience: the note type (default \"note\")."},
                        "sequence": {"type": "integer", "description": "Single-note convenience: ordering hint within the type."},
                        "done": {"type": "boolean", "description": "Single-note convenience: checked-off flag."}
                    }
                }),
            ),
            ToolDefinition::new(
                TOOL_SCRATCHPAD_SEARCH,
                "Read this conversation's scratchpad. Omit `query` and `keys` to list all notes \
                 (ordered by type, then `sequence`); pass `query` to search note keys and \
                 content by meaning as well as by wording, so a note is findable even when you \
                 have since rephrased what it says; pass `keys` to fetch specific notes. Pass \
                 `type` to restrict a list/search to one category, e.g. `type: \"todo\"` for \
                 just your plan. Each returned note includes its `type`, `sequence`, and \
                 `done`. `max_results` is required. Results are bounded — if the response is \
                 truncated you'll get `truncated: true` and should narrow with a `query`, a \
                 `type`, or a smaller key set.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search note keys + content. Matches on meaning as well as on exact words, so describing what you are looking for works as well as quoting it. Omit to list all notes."},
                        "keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Fetch specific notes by key. Takes precedence over `query`."
                        },
                        "type": {"type": "string", "description": "Restrict a list/search to one note type, e.g. \"todo\". Ignored when `keys` is given."},
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Maximum notes to return (required; clamped to 100)."
                        }
                    },
                    "required": ["max_results"]
                }),
            ),
            ToolDefinition::new(
                TOOL_SCRATCHPAD_DELETE,
                "Delete notes from this conversation's scratchpad. Pass `keys` to delete \
                 specific notes, or `all: true` to clear the whole pad. Exactly one of the two \
                 must be supplied.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keys of notes to delete."
                        },
                        "all": {
                            "type": "boolean",
                            "description": "Delete every note in this scratchpad. Mutually exclusive with `keys`."
                        }
                    }
                }),
            ),
        ];

        // Capability-gated: only advertise pinning when the write is wired, so
        // an embedder that predates #597 never sees a tool it cannot service.
        if self.scratchpad_set_pinned_fn.is_some() {
            defs.push(ToolDefinition::new(
                TOOL_SCRATCHPAD_PIN,
                "Pin (or unpin) scratchpad notes by key. A pinned note's full content is \
                 repeated back to you every turn under [Pinned], so you never spend a search \
                 to re-read it and can never work from a stale copy. Pin sparingly: only a \
                 fact you will need repeatedly for the REST of this task and that would be \
                 harmful to get wrong — not something merely interesting. Unpin the moment it \
                 stops mattering; a stale pin costs you context every single turn. At most 5 \
                 notes can be pinned at once, and hitting that is the signal to unpin \
                 something, not to shuffle.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keys of existing notes to pin or unpin."
                        },
                        "pinned": {
                            "type": "boolean",
                            "description": "true to pin, false to unpin. Defaults to true."
                        }
                    },
                    "required": ["keys"]
                }),
            ));
        }

        // Capability-gated: only advertise the notification tool when a
        // notification service was wired (present on the session bus).
        if self.notify_fn.is_some() {
            defs.push(ToolDefinition::new(
                TOOL_NOTIFY,
                "Show a desktop notification to the user via the system notification service. \
                 Use to surface something the user should see now — e.g. a long-running task \
                 finished, or a time-sensitive finding worth interrupting for. Prefer the normal \
                 reply for ordinary output; reserve notifications for things that warrant the \
                 user's attention away from the chat. Only available when a desktop notification \
                 service is present.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Short title line (a few words)."
                        },
                        "body": {
                            "type": "string",
                            "description": "Optional longer detail shown under the title."
                        },
                        "urgency": {
                            "type": "string",
                            "enum": ["low", "normal", "critical"],
                            "description": "Urgency (default normal). 'critical' stays on screen until dismissed; use sparingly."
                        }
                    },
                    "required": ["summary"]
                }),
            ));
        }

        // Capability-gated: only advertise the skill tools when a skill index
        // was wired (a Postgres pool + configured roots).
        if self.skill_search_fn.is_some() {
            defs.push(ToolDefinition::new(
                TOOL_SKILL_SEARCH,
                "Search the on-disk skill library — reusable how-to playbooks and workflows — by \
                 meaning. Call this before a recurring or procedural task to check whether an \
                 established skill already covers it, then read the full body with \
                 builtin_skill_get before following one.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What you are trying to do."},
                        "kind": {
                            "type": "string",
                            "enum": ["skill", "workflow"],
                            "description": "Optional filter: only plain skills or only workflows."
                        },
                        "limit": {"type": "integer", "description": "Max results (default 5)."}
                    },
                    "required": ["query"]
                }),
            ));
            defs.push(ToolDefinition::new(
                TOOL_SKILL_GET,
                "Fetch one skill by name: its full markdown body, on-disk path, attachment \
                 filenames, kind, trust tier, and whether its files are still present on disk. \
                 Use after builtin_skill_search to read a skill before following it. When \
                 present_on_disk is false the procedure is still good, but the skill's files are \
                 gone: its path and attachments no longer resolve, so don't try to run its \
                 bundled scripts. Returns your own user-scoped copy of this skill if you have \
                 one, otherwise the shared global one -- there is no way to address another \
                 user's copy.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The skill name."}
                    },
                    "required": ["name"]
                }),
            ));
        }

        defs
    }

    /// Whether `name` is a built-in tool, i.e. whether the caller must dispatch
    /// it to [`Self::execute_tool`] instead of looking for an MCP server that
    /// owns it. This is the gate `McpToolExecutor` consults first.
    ///
    /// Derived from [`Self::ALL_TOOL_NAMES`] rather than restating the set, so
    /// the two cannot drift: a builtin that is advertised but not claimed here
    /// is routed to MCP and fails every call with "unknown tool".
    ///
    /// Capability-gated tools are claimed whether or not their closure is
    /// wired. Absence is expressed by not advertising the tool, and a call that
    /// arrives anyway gets the tool's own "not configured" error rather than
    /// being routed somewhere it does not belong.
    pub fn supports_tool(name: &str) -> bool {
        Self::ALL_TOOL_NAMES.contains(&name)
    }

    /// The builtin provider groups (Phase 1): a stable group id plus the authored
    /// blurb that seeds the group's synthetic `provider:<id>` row. Built-ins are
    /// surfaced to tool-search by the SAME provider mechanism as external MCP
    /// servers — this classification is what unifies them.
    pub const PROVIDER_GROUPS: &'static [(&'static str, &'static str)] = &[
        (
            "knowledge",
            "Long-term memory: store and recall the user's preferences, facts, \
             instructions, and project context as durable tagged entries, via hybrid \
             vector + full-text search.",
        ),
        (
            "scratchpad",
            "Ephemeral per-conversation working notes: hold a plan, findings, and \
             intermediate results across a multi-step task; discarded when the \
             conversation ends.",
        ),
        (
            "database",
            "Run SQL against the assistant's own PostgreSQL database to inspect or \
             modify its conversations, messages, knowledge, and tool data, and to \
             build your own schemas/views.",
        ),
        (
            "recall",
            "Search past conversations by full-text query to recall what was \
             discussed or decided.",
        ),
        (
            "system",
            "System and desktop touchpoints: read runtime/system context and raise \
             desktop notifications for things that need attention now.",
        ),
        (
            "tool-meta",
            "Discover additional tools by description and manage the MCP servers that \
             provide them (status/start/stop/restart).",
        ),
        (
            "skills",
            "Reusable how-to playbooks and workflows on disk: find an established skill \
             for a recurring or procedural task and read its steps before acting.",
        ),
    ];

    /// Every builtin tool name - the one list the routing surfaces are held to.
    ///
    /// [`Self::supports_tool`] answers straight out of it, and the tests hold
    /// [`Self::tool_definitions`], [`Self::execute_tool`] and
    /// [`Self::provider_group`] to it, so a new builtin cannot be advertised
    /// without being routable, classified, and dispatchable.
    ///
    /// The capability-gated tools (`builtin_notify`, `builtin_scratchpad_pin`,
    /// the skill tools) are listed here even though a given runtime may not
    /// advertise them: routing is by name, and an unwired tool answers with its
    /// own "not configured" error.
    pub const ALL_TOOL_NAMES: &'static [&'static str] = &[
        TOOL_KB_WRITE,
        TOOL_KB_SEARCH,
        TOOL_KB_DELETE,
        TOOL_KB_LIST,
        TOOL_SEARCH,
        TOOL_NOTIFY,
        TOOL_SYS_PROPS,
        TOOL_DB_QUERY,
        TOOL_MCP_CONTROL,
        TOOL_CONV_SEARCH,
        TOOL_SCRATCHPAD_WRITE,
        TOOL_SCRATCHPAD_SEARCH,
        TOOL_SCRATCHPAD_DELETE,
        TOOL_SCRATCHPAD_PIN,
        TOOL_SKILL_SEARCH,
        TOOL_SKILL_GET,
    ];

    /// Classify a builtin tool name into its provider group, or `None` when the
    /// name is not a known builtin. Callers that must register every builtin
    /// (never drop one) fall back to a generic group on `None`; the
    /// `builtin_provider_map_is_exhaustive` test ensures no known builtin relies
    /// on that fallback.
    pub fn provider_group(tool_name: &str) -> Option<&'static str> {
        match tool_name {
            TOOL_KB_WRITE | TOOL_KB_SEARCH | TOOL_KB_DELETE | TOOL_KB_LIST => Some("knowledge"),
            TOOL_SCRATCHPAD_WRITE
            | TOOL_SCRATCHPAD_SEARCH
            | TOOL_SCRATCHPAD_DELETE
            | TOOL_SCRATCHPAD_PIN => Some("scratchpad"),
            TOOL_DB_QUERY => Some("database"),
            TOOL_CONV_SEARCH => Some("recall"),
            TOOL_SYS_PROPS | TOOL_NOTIFY => Some("system"),
            TOOL_SEARCH | TOOL_MCP_CONTROL => Some("tool-meta"),
            TOOL_SKILL_SEARCH | TOOL_SKILL_GET => Some("skills"),
            _ => None,
        }
    }

    /// The authored blurb for a provider group id, or `None` if unknown.
    pub fn provider_blurb(provider: &str) -> Option<&'static str> {
        Self::PROVIDER_GROUPS
            .iter()
            .find(|(id, _)| *id == provider)
            .map(|(_, blurb)| *blurb)
    }

    /// Dispatch a builtin by name. Callers gate on [`Self::supports_tool`]
    /// first, so the catch-all arm is only reached when a name in
    /// [`Self::ALL_TOOL_NAMES`] has no dispatch arm - the
    /// `every_advertised_builtin_is_routable` guard reads its wording, so
    /// reword it there in the same change.
    pub async fn execute_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, CoreError> {
        match name {
            TOOL_KB_WRITE => self.kb_write(arguments).await,
            TOOL_KB_SEARCH => self.kb_search(arguments).await,
            TOOL_KB_DELETE => self.kb_delete(arguments).await,
            TOOL_KB_LIST => self.kb_list(arguments).await,
            TOOL_SEARCH => self.tool_search(arguments).await,
            TOOL_NOTIFY => self.notify(arguments).await,
            TOOL_SYS_PROPS => Ok(self.sys_props()),
            TOOL_DB_QUERY => self.db_query(arguments).await,
            TOOL_MCP_CONTROL => self.mcp_control(arguments).await,
            TOOL_CONV_SEARCH => self.conversation_search(arguments).await,
            TOOL_SCRATCHPAD_WRITE => self.scratchpad_write(arguments).await,
            TOOL_SCRATCHPAD_SEARCH => self.scratchpad_search(arguments).await,
            TOOL_SCRATCHPAD_DELETE => self.scratchpad_delete(arguments).await,
            TOOL_SCRATCHPAD_PIN => self.scratchpad_pin(arguments).await,
            TOOL_SKILL_SEARCH => self.skill_search(arguments).await,
            TOOL_SKILL_GET => self.skill_get(arguments).await,
            _ => Err(CoreError::ToolExecution(format!(
                "unknown built-in tool: {name}"
            ))),
        }
    }

    fn sys_props(&self) -> String {
        let now = NowSnapshot::now();

        // Prefer the CONNECTING CLIENT's self-reported context (#549/#558) for
        // the user + device identity fields. The daemon may be remote or
        // containerized, so its own host env is NOT the user's environment —
        // reporting the daemon host AS the user (the pre-#558 bug) is wrong.
        // We fall back to daemon-host detection only when the client sent no
        // context, and label the source with `identity_source` so daemon-host
        // values are never mistaken for the client's. An empty context counts
        // as absent (fail-closed), and fields the client omitted stay null
        // rather than borrowing the daemon's — a partial client context never
        // leaks a daemon-host value as if it were the user's.
        let client = current_client_context().filter(|c| !c.is_empty());

        let identity = match &client {
            Some(c) => IdentityFields {
                source: "client",
                real_name: c.real_name.clone(),
                username: c.username.clone(),
                home_dir: c.home_dir.clone(),
                hostname: c.hostname.clone(),
                os: c.os.clone(),
                timezone: c.timezone.clone(),
            },
            None => IdentityFields {
                source: "daemon_host_fallback",
                // The daemon host has no notion of the user's real name.
                real_name: None,
                username: detect_username(),
                home_dir: detect_home_dir(),
                hostname: detect_hostname(),
                os: Some(std::env::consts::OS.to_string()),
                timezone: Some(now.timezone()),
            },
        };

        serde_json::json!({
            "ok": true,
            "props": {
                "note": "`identity_source` says where the user/device fields came \
                         from: \"client\" = the connecting client reported them; \
                         \"daemon_host_fallback\" = the client sent none, so these \
                         are the daemon host's own values and may not be the \
                         user's. Server-side tools (file, terminal) run on the \
                         daemon host: relative paths resolve from `daemon_host.cwd`, \
                         not the client's home.",
                "generated_at_epoch": now.epoch_secs(),
                "generated_at_utc": now.utc_rfc3339(),
                "identity_source": identity.source,
                "real_name": identity.real_name,
                "username": identity.username,
                "home_dir": identity.home_dir,
                "hostname": identity.hostname,
                "os": identity.os,
                "timezone": identity.timezone,
                "daemon_host": {
                    "cwd": detect_daemon_cwd(),
                    "generated_at_local": now.local_rfc3339(),
                    "timezone": now.timezone(),
                    "hostname": detect_hostname(),
                    "os": std::env::consts::OS,
                    "arch": std::env::consts::ARCH,
                    "os_version": detect_os_version(),
                    "username": detect_username(),
                    "home_dir": detect_home_dir(),
                    "xdg_dirs": detect_xdg_dirs(),
                    "shell": detect_shell(),
                    "locale": detect_locale(),
                    "session_type": detect_session_type(),
                },
            },
        })
        .to_string()
    }

    async fn kb_write(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let write_fn = self
            .kb_write_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("knowledge base not configured".to_string()))?;

        // Batch form (`entries`) takes precedence over the single top-level
        // form. Each spec is one {content?, tags?, id?} object.
        let specs: Vec<serde_json::Value> = match arguments.get("entries") {
            Some(serde_json::Value::Array(items)) => items.clone(),
            _ => vec![arguments.clone()],
        };

        let mut saved_out = Vec::with_capacity(specs.len());
        // One budget for the whole call, so a batch cannot spend it per entry.
        let mut tag_budget = TagGateBudget::new();
        // A spec that fails ends the call, but the budget still reports what it
        // did first: an entry that failed for its own reason is exactly when an
        // operator wants to know the vocabulary had already stopped answering.
        let mut failure: Option<CoreError> = None;
        for spec in &specs {
            let entry = match self.build_write_entry(spec, &mut tag_budget).await {
                Ok(entry) => entry,
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            };
            // Embedding generation is decoupled from the write: the entry lands
            // immediately (NULL embedding on create, stale embedding left in
            // place on update) and the background embedding-backfill task
            // generates the vector within its next pass. The row is
            // keyword-searchable (FTS) right away; semantic recall follows.
            let saved = match write_fn(entry).await {
                Ok(saved) => saved,
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            };
            saved_out.push(serde_json::json!({
                "id": saved.id,
                // The tags actually stored, which the vocabulary check may have
                // redirected away from the ones the caller sent. Reporting them
                // stops the model believing an entry carries a tag it does not,
                // and then searching for that tag and finding nothing.
                "tags": saved.tags,
                "created_at": saved.created_at,
                "updated_at": saved.updated_at,
            }));
        }
        tag_budget.report();
        if let Some(e) = failure {
            return Err(e);
        }

        let mut response = serde_json::json!({
            "ok": true,
            "count": saved_out.len(),
            "entries": saved_out,
        });
        // Present only when a tag went to the store without the vocabulary
        // answering for it, so a checked write and a degraded one never read
        // the same. See `TagGateBudget::tag_check`.
        if let Some(check) = tag_budget.tag_check()
            && let Some(obj) = response.as_object_mut()
        {
            obj.insert("tag_check".to_string(), serde_json::json!(check));
        }
        Ok(response.to_string())
    }

    /// Build a [`KnowledgeEntry`] from one write spec.
    ///
    /// One rule governs every optional field: a field the write does not
    /// mention keeps the value the stored entry holds, and a field the write
    /// does mention takes what the write gave, including an empty value, which
    /// clears the field. So `tags` absent preserves the entry's tags, and
    /// `"tags": []` clears them. A write with no `id`, or with an `id` no entry
    /// holds, has nothing to fall back to, so every field it omits starts
    /// empty.
    ///
    /// A field is therefore read out of the spec as "the caller supplied this"
    /// or "the caller said nothing", and resolved against
    /// [`Self::existing_entry`] in one line. A new field joins by doing the
    /// same, without restating the rule.
    ///
    /// Tool-authored writes always carry `source = "explicit"`.
    async fn build_write_entry(
        &self,
        spec: &serde_json::Value,
        tag_budget: &mut TagGateBudget,
    ) -> Result<desktop_assistant_core::domain::KnowledgeEntry, CoreError> {
        use desktop_assistant_core::domain::KnowledgeEntry;

        let content_opt = optional_string(spec, "content");
        let id_opt = optional_string(spec, "id");
        // `Some` exactly when the write asked to set the tags, so a
        // present-but-empty list stays distinct from an absent one. A `tags`
        // this cannot read is refused rather than stored as empty, because an
        // empty list now means "clear them".
        let supplied_tags = supplied_string_array(spec, "tags")?;

        let existing = self
            .existing_entry(id_opt.as_deref(), content_opt.is_some())
            .await?;

        let id = id_opt.unwrap_or_else(|| uuid::Uuid::now_v7().to_string());
        // `existing_entry` already refused the one case this cannot resolve —
        // no content and nothing stored — so this guard only keeps the failure
        // legible if that ever stops holding.
        let content = content_opt
            .or_else(|| existing.as_ref().map(|e| e.content.clone()))
            .ok_or_else(|| {
                CoreError::ToolExecution("knowledge_base write requires content".into())
            })?;
        // Tags the caller supplied go through the formal vocabulary; tags
        // carried over from the stored entry are already in it, so a write that
        // does not mention tags re-registers nothing.
        let supplied_tags = match supplied_tags {
            Some(supplied) => Some(self.resolve_tags(supplied, spec, tag_budget).await),
            None => None,
        };
        let tags = supplied_tags
            .or_else(|| existing.as_ref().map(|e| e.tags.clone()))
            .unwrap_or_default();
        // `metadata` has no argument of its own, so a write only ever carries
        // the stored value forward. Its empty value is an object, not JSON
        // null, because the provenance stamp below writes into it.
        let mut metadata = existing
            .as_ref()
            .map(|e| e.metadata.clone())
            .unwrap_or_else(|| serde_json::json!({}));

        // Provenance (#240): stamp the originating conversation so a tool-saved
        // finding is traceable back to where it was learned. Only when a
        // conversation scope is active and it isn't already set.
        if let Some(conv) = current_conversation_id()
            && let Some(obj) = metadata.as_object_mut()
            && !obj.contains_key("source_conversation_id")
        {
            obj.insert(
                "source_conversation_id".to_string(),
                serde_json::Value::String(conv.0),
            );
        }

        Ok(KnowledgeEntry {
            id,
            content,
            tags,
            metadata,
            created_at: String::new(),
            updated_at: String::new(),
            source: Some("explicit".to_string()),
            // The write tool takes no summary yet. `None` preserves whatever
            // the row already holds; it never clears it.
            summary: None,
        })
    }

    /// Load the entry a write falls back to for the fields it does not mention,
    /// or `None` when the write starts from nothing.
    ///
    /// `id` is what the write named, and `has_content` says whether the write
    /// carries content of its own. Together they decide the three cases:
    ///
    /// - No `id`. A create, so there is nothing to fall back to. Without
    ///   content there is also nothing to store, which is an error.
    /// - An `id` an entry holds. An update, and that entry is the fallback.
    /// - An `id` no entry holds. A create at an id the caller chose, which is
    ///   how a caller makes a write idempotent under retry: the retry lands on
    ///   the row the first attempt created instead of a duplicate. This is an
    ///   error only without content, because then the write carries nothing to
    ///   create the entry from.
    ///
    /// The lookup runs for every write that names an `id`, not only for one
    /// without content. A content update that skipped it had no fallback, so it
    /// stored the entry with no tags and no metadata (#1093).
    async fn existing_entry(
        &self,
        id: Option<&str>,
        has_content: bool,
    ) -> Result<Option<desktop_assistant_core::domain::KnowledgeEntry>, CoreError> {
        let Some(id) = id else {
            return if has_content {
                Ok(None)
            } else {
                Err(CoreError::ToolExecution(
                    "knowledge_base write requires `content`, or an `id` of an existing entry to \
                     update its tags"
                        .to_string(),
                ))
            };
        };
        let get_fn = self
            .kb_get_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("knowledge base not configured".to_string()))?;
        let found = get_fn(id.to_string()).await?;
        if found.is_none() && !has_content {
            return Err(CoreError::ToolExecution(format!(
                "no knowledge entry with id {id}"
            )));
        }
        Ok(found)
    }

    /// Put every tag the caller supplied through the formal tag vocabulary and
    /// return the names to store.
    ///
    /// Each tag is offered with its entry from the spec's
    /// `new_tag_descriptions` map, which is what lets the vocabulary tell one
    /// short facet tag from another. A tag the vocabulary already holds matches
    /// on its name and costs no embedding, so sending a description for every
    /// tag is free.
    ///
    /// A description is matched to its tag on the normalized name, on both
    /// sides. The model writes what reads well - `"Topic: Embeddings"` in one
    /// field and `"topic:embeddings"` in the other - and the vocabulary keys on
    /// the normalized name, so matching the two raw strings drops the
    /// description. A tag that loses its description embeds as its bare name,
    /// which carries almost no signal, so the check that this whole path exists
    /// for would mostly not fire.
    ///
    /// Why nothing here can fail the write: the vocabulary is optional. It is
    /// absent when no database or embedding backend is wired, and it can fail
    /// per call when the embedding backend is unreachable. Both degrade to the
    /// prior behaviour - store the tag as the caller wrote it.
    ///
    /// Two things stop the vocabulary being consulted again for the rest of the
    /// write call: its first failure, and a spent [`TagGateBudget`]. A
    /// vocabulary that just failed will not answer the next tag either, and
    /// asking it again pays the whole ceiling for nothing. The caller reports
    /// both once for the call, not once per tag.
    ///
    /// One consultation is bounded by [`TAG_RESOLVE_CALL_CEILING`], and only
    /// the time inside it is charged against the budget, so a slow store or a
    /// slow read elsewhere in the write never spends the vocabulary's share.
    async fn resolve_tags(
        &self,
        tags: Vec<String>,
        spec: &serde_json::Value,
        budget: &mut TagGateBudget,
    ) -> Vec<String> {
        let Some(resolve_fn) = self.kb_tag_resolve_fn.as_ref() else {
            // No vocabulary is wired, so no tag on this write was checked
            // against one. The caller says so rather than answering as though
            // it had been.
            budget.unchecked(tags.len());
            return tags;
        };
        let descriptions = normalized_tag_descriptions(spec);

        let mut resolved = Vec::with_capacity(tags.len());
        for tag in tags {
            if !budget.is_open() {
                budget.stop("the tag vocabulary budget for this write was spent");
                budget.unchecked(1);
                resolved.push(tag);
                continue;
            }
            let description = descriptions.get(&normalize_tag(&tag)).cloned();
            let proposed = ProposedTag {
                name: tag.clone(),
                description,
            };
            let started = tokio::time::Instant::now();
            let answer = tokio::time::timeout(TAG_RESOLVE_CALL_CEILING, resolve_fn(proposed)).await;
            budget.charge(started.elapsed());
            match answer {
                Ok(Ok(name)) => resolved.push(name),
                Ok(Err(e)) => {
                    budget.stop(e.to_string());
                    budget.unchecked(1);
                    resolved.push(tag);
                }
                Err(_) => {
                    budget.stop("the tag vocabulary did not answer within its per-tag ceiling");
                    budget.unchecked(1);
                    resolved.push(tag);
                }
            }
        }
        resolved
    }

    async fn kb_search(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let search_fn = self
            .kb_search_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("knowledge base not configured".to_string()))?;

        let query = required_string(&arguments, "query")?;
        let tags = optional_string_array_nonempty(&arguments, "tags");
        let exclude_tags = optional_string_array_nonempty(&arguments, "exclude_tags");
        // The schema advertises a minimum of 1 and a maximum of
        // `KB_SEARCH_MAX_LIMIT`. Honour both here rather than trusting them: a
        // limit of 0 would return nothing and then report `truncated`, because
        // `results.len() >= limit` holds vacuously, and an unbounded limit
        // reaches `limit * 2` in the storage layer, which overflows.
        let limit = arguments
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(10)
            .clamp(1, KB_SEARCH_MAX_LIMIT) as usize;

        tracing::info!(query = %query, ?tags, ?exclude_tags, limit, "knowledge base search");

        let (query_embedding, embedding_model) = self.embed_query(&query).await;

        let page = search_fn(
            query,
            query_embedding,
            embedding_model,
            tags,
            exclude_tags,
            limit,
        )
        .await?;

        let scope_size = page.scope_size;
        // The cap is a context budget: the list travels to the model inside
        // this tool result. Enforce it here, at the point of serialization,
        // whatever the store hands over.
        let mut available_tags = page.available_tags;
        available_tags.truncate(AVAILABLE_TAGS_LIMIT);

        let items: Vec<serde_json::Value> = page
            .entries
            .into_iter()
            .map(|entry| {
                serde_json::json!({
                    "id": entry.id,
                    "content": entry.content,
                    "summary": entry.summary,
                    "tags": entry.tags,
                    "metadata": entry.metadata,
                    "updated_at": entry.updated_at,
                })
            })
            .collect();

        tracing::info!(
            result_count = items.len(),
            scope_size = scope_size.as_str(),
            available_tag_count = available_tags.len(),
            "knowledge base search results"
        );
        tracing::debug!(results = %serde_json::to_string(&items).unwrap_or_default(), "knowledge base search response");

        // A full page is evidence that entries were left behind, so say so and
        // say what to do about it — the same shape `builtin_scratchpad_search`
        // uses. An absent `truncated` is the claim that nothing was dropped.
        //
        // Why `Few` overrides it: `Few` means the scope is no larger than the
        // page, and what matched is a subset of the scope, so a page that
        // filled up under `Few` holds the whole scope. Claiming truncation
        // there sends the caller narrowing a search that already returned
        // everything there was.
        //
        // Only `Few` earns that. `Unknown` says the scope was never measured,
        // so it proves nothing about what the page holds and must leave the
        // truncation claim standing.
        let truncated = items.len() >= limit && scope_size != ScopeSize::Few;
        let mut response = serde_json::json!({
            "ok": true,
            "results": items,
            "returned": items.len(),
            "scope_size": scope_size.as_str(),
            "available_tags": available_tags,
        });
        if truncated {
            response["truncated"] = serde_json::Value::Bool(true);
            response["message"] = serde_json::json!(
                "results were truncated; narrow with a more specific `query`, a `tags` \
                 filter drawn from `available_tags`, or `exclude_tags`"
            );
        }
        Ok(response.to_string())
    }

    async fn skill_search(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let search_fn = self
            .skill_search_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("skill library not configured".to_string()))?;

        let query = required_string(&arguments, "query")?;
        let kind_filter = arguments
            .get("kind")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        let limit = arguments
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(5) as usize;

        tracing::info!(query = %query, ?kind_filter, limit, "skill search");

        let (query_embedding, embedding_model) = self.embed_query(&query).await;
        // Over-fetch when filtering by kind, then trim to the requested limit.
        let fetch = if kind_filter.is_some() {
            limit.saturating_mul(3)
        } else {
            limit
        };
        let mut results = search_fn(query, query_embedding, embedding_model, fetch).await?;
        if let Some(kind) = &kind_filter {
            results.retain(|s| s.kind.as_str() == kind);
        }
        results.truncate(limit);

        let items: Vec<serde_json::Value> = results
            .into_iter()
            .map(|s| {
                serde_json::json!({
                    "name": s.name,
                    "description": s.description,
                    "kind": s.kind.as_str(),
                    "trust_tier": s.trust_tier.as_str(),
                    "disk_path": s.disk_path,
                    "attachments": s.attachments,
                    "present_on_disk": s.present_on_disk,
                })
            })
            .collect();

        Ok(serde_json::json!({ "ok": true, "results": items }).to_string())
    }

    async fn skill_get(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let get_fn = self
            .skill_get_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("skill library not configured".to_string()))?;

        let name = required_string(&arguments, "name")?;

        // No caller-supplied scope on the wire (#911): the schema advertises
        // only `name`, so there is nothing here to trust or reject. Prefer
        // the caller's own user-scoped skill of this name and fall back to
        // the global one, the same "global plus mine, mine wins" view
        // `builtin_skill_search` presents without asking the caller to pick
        // a scope either.
        //
        // A live personal row wins outright, with no second lookup. A
        // personal row that is a TOMBSTONE (`present_on_disk: false` -- its
        // files are gone, but the append-only catalog keeps it) must not
        // permanently shadow a live global skill of the same name: there is
        // no `owner` argument any more for the caller to reach past it with,
        // so the fallback has to reach past a dead personal row on its own.
        // Only when the global lookup also comes up empty does the personal
        // tombstone stand, since it is then the only record that ever
        // existed for this name.
        let mine = get_fn(name.clone(), Some(SKILL_GET_OWN_SCOPE.to_string())).await?;
        let found = if mine.as_ref().is_some_and(|s| s.present_on_disk) {
            mine
        } else {
            match get_fn(name.clone(), None).await? {
                Some(global) => Some(global),
                None => mine,
            }
        };

        match found {
            Some(s) => Ok(serde_json::json!({
                "ok": true,
                "name": s.name,
                "description": s.description,
                "kind": s.kind.as_str(),
                "trust_tier": s.trust_tier.as_str(),
                "disk_path": s.disk_path,
                "attachments": s.attachments,
                "present_on_disk": s.present_on_disk,
                "last_seen_at": s.last_seen_at.map(|ts| ts.to_rfc3339()),
                "tags": s.tags,
                "body": s.body,
            })
            .to_string()),
            None => Ok(
                serde_json::json!({ "ok": false, "reason": format!("no skill named {name}") })
                    .to_string(),
            ),
        }
    }

    async fn conversation_search(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let search_fn = self.conversation_search_fn.as_ref().ok_or_else(|| {
            CoreError::ToolExecution("conversation search not configured".to_string())
        })?;

        let query = required_string(&arguments, "query")?;
        let limit = arguments
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(10) as usize;
        let role_filter = arguments
            .get("role")
            .and_then(serde_json::Value::as_str)
            .and_then(|s| match s {
                "user" => Some(Role::User),
                "assistant" => Some(Role::Assistant),
                // Reject other roles at the boundary so the SQL layer
                // doesn't have to defend against arbitrary text.
                _ => None,
            });

        tracing::info!(query = %query, limit, ?role_filter, "conversation search");

        let hits = search_fn(query, limit, role_filter).await?;

        let items: Vec<serde_json::Value> = hits
            .into_iter()
            .map(|h| {
                serde_json::json!({
                    "conversation_id": h.conversation_id,
                    "conversation_title": h.conversation_title,
                    "ordinal": h.ordinal,
                    "role": match h.role {
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        Role::System => "system",
                        Role::Tool => "tool",
                    },
                    "snippet": h.snippet,
                    "content": h.content,
                    "rank": h.rank,
                    "updated_at": h.updated_at,
                })
            })
            .collect();

        tracing::info!(result_count = items.len(), "conversation search results");

        Ok(serde_json::json!({
            "ok": true,
            "results": items,
        })
        .to_string())
    }

    async fn kb_delete(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let delete_fn = self
            .kb_delete_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("knowledge base not configured".to_string()))?;

        // Accept either a single `id` or a list of `ids`.
        let mut ids = optional_string_array(&arguments, "ids");
        if let Some(id) = optional_string(&arguments, "id") {
            ids.push(id);
        }
        if ids.is_empty() {
            return Err(CoreError::ToolExecution(
                "knowledge_base delete requires `id` or `ids`".to_string(),
            ));
        }

        let deleted = delete_fn(ids.clone()).await?;

        Ok(serde_json::json!({
            "ok": true,
            "deleted": deleted,
            "ids": ids,
        })
        .to_string())
    }

    async fn kb_list(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let list_fn = self
            .kb_list_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("knowledge base not configured".to_string()))?;

        let limit = arguments
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(50) as usize;
        let order = match arguments.get("order").and_then(serde_json::Value::as_str) {
            Some("oldest_first") => ListOrder::OldestFirst,
            _ => ListOrder::NewestFirst,
        };
        let query = KnowledgeListQuery {
            limit,
            after: optional_string(&arguments, "cursor"),
            order: ListOrderOpt(order),
            tags: optional_string_array_nonempty(&arguments, "tags"),
            exclude_tags: optional_string_array_nonempty(&arguments, "exclude_tags"),
            source: optional_string(&arguments, "source"),
        };

        let page = list_fn(query).await?;

        let items: Vec<serde_json::Value> = page
            .entries
            .into_iter()
            .map(|entry| {
                serde_json::json!({
                    "id": entry.id,
                    "content": entry.content,
                    "summary": entry.summary,
                    "tags": entry.tags,
                    "metadata": entry.metadata,
                    "source": entry.source,
                    "created_at": entry.created_at,
                    "updated_at": entry.updated_at,
                })
            })
            .collect();

        Ok(serde_json::json!({
            "ok": true,
            "count": items.len(),
            "entries": items,
            "next_cursor": page.next_cursor,
        })
        .to_string())
    }

    async fn notify(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let notify_fn = self.notify_fn.as_ref().ok_or_else(|| {
            CoreError::ToolExecution("desktop notifications are not available".to_string())
        })?;

        let summary = required_string(&arguments, "summary")?;
        let body = optional_string(&arguments, "body").unwrap_or_default();
        let urgency =
            NotifyUrgency::parse(arguments.get("urgency").and_then(serde_json::Value::as_str));

        match notify_fn(summary, body, urgency).await? {
            Some(id) => Ok(serde_json::json!({ "ok": true, "shown": true, "id": id }).to_string()),
            // Suppressed by rate-limiting (e.g. an identical notification just
            // fired) — report it without making it an error.
            None => Ok(serde_json::json!({
                "ok": true,
                "shown": false,
                "reason": "suppressed (duplicate of a recent notification)"
            })
            .to_string()),
        }
    }

    async fn tool_search(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let search_fn = self
            .tool_search_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("tool registry not configured".to_string()))?;

        let query = required_string(&arguments, "query")?;
        tracing::info!(query = %query, "tool search");

        let query_embedding = self.embed_text(&query).await.unwrap_or_default();

        let results = search_fn(query.clone(), query_embedding, REGISTRY_SEARCH_LIMIT).await?;

        // Classify each registry hit. Everything in the registry is reached
        // from the daemon, but a server behind HTTP acts on a third-party
        // service rather than on the daemon's own files, and the model has to
        // be able to tell those apart.
        let daemon_names: Vec<&str> = results.iter().map(|t| t.name.as_str()).collect();
        let routed = match &self.mcp_handle {
            Some(handle) => handle.tool_runners(&daemon_names).await,
            // No MCP executor wired: every registry row is a built-in, which
            // runs inside the daemon process.
            None => std::collections::HashMap::new(),
        };
        let daemon_name_set: HashSet<&str> = daemon_names.iter().copied().collect();

        // The connected client's own tools are never in the registry - they are
        // registered per connection, not indexed - so a search that consulted
        // only the registry could never offer the option that acts on the
        // user's own machine.
        let client_defs = match current_client_tools() {
            Some(port) => port.tool_definitions().await,
            None => Vec::new(),
        };
        let same_machine =
            current_co_location().unwrap_or_else(|| current_transport_kind().is_co_located());
        let (device_hits, device_matches_dropped) =
            match_client_tools(&query, &client_defs, DEVICE_SEARCH_LIMIT);
        // On one machine a client tool and a daemon tool of the same name are
        // the same capability, so offering both would be a choice with no
        // difference. The daemon entry is kept, matching how the turn loop
        // resolves that collision.
        let device_hits: Vec<&ToolDefinition> = device_hits
            .into_iter()
            .filter(|t| !(same_machine && daemon_name_set.contains(t.name.as_str())))
            .collect();

        let mut tools: Vec<serde_json::Value> =
            Vec::with_capacity(results.len() + device_hits.len());
        let mut runners_present: HashSet<ToolRunner> = HashSet::new();
        for tool in &results {
            let runner = routed
                .get(tool.name.as_str())
                .copied()
                .unwrap_or(ToolRunner::Daemon);
            runners_present.insert(runner);
            tools.push(serde_json::json!({
                "name": tool.name,
                "description": tool.description,
                "runs_on": runner.as_str(),
            }));
        }
        for tool in &device_hits {
            runners_present.insert(ToolRunner::Device);
            tools.push(serde_json::json!({
                "name": tool.name,
                "description": tool.description,
                "runs_on": ToolRunner::Device.as_str(),
            }));
        }

        let tool_names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
        tracing::info!(
            result_count = tools.len(),
            device_count = device_hits.len(),
            device_matches_dropped,
            same_machine,
            ?tool_names,
            "tool search results"
        );

        let mut response = serde_json::json!({
            "ok": true,
            "same_machine": same_machine,
            "runs_on": self.runner_legend(&runners_present),
            "tools": tools,
        });
        // Never present a truncated match set as the whole answer.
        if device_matches_dropped > 0 {
            response["more_device_tools_matched"] = serde_json::json!(device_matches_dropped);
        }
        Ok(response.to_string())
    }

    /// Explain each runner value that appears in a search result, once per
    /// response rather than once per hit.
    ///
    /// Only the values actually present are described. A legend for a runner
    /// that produced no hit spends context describing a choice the model does
    /// not have.
    fn runner_legend(&self, present: &HashSet<ToolRunner>) -> serde_json::Value {
        let mut legend = serde_json::Map::new();
        if present.contains(&ToolRunner::Daemon) {
            let kind = if self.daemon_on_workstation {
                String::new()
            } else {
                " (a container or a server, not the user's own computer)".to_string()
            };
            legend.insert(
                ToolRunner::Daemon.as_str().to_string(),
                serde_json::json!(format!(
                    "the daemon's own machine, \"{}\"{kind}. Acts on that machine's files \
                     and processes.",
                    self.daemon_host
                )),
            );
        }
        if present.contains(&ToolRunner::RemoteService) {
            legend.insert(
                ToolRunner::RemoteService.as_str().to_string(),
                serde_json::json!(
                    "a service the daemon calls over the network. Acts on that service, \
                     and on no local files at all."
                ),
            );
        }
        if present.contains(&ToolRunner::Device) {
            let label = match current_client_label().filter(|l| !l.trim().is_empty()) {
                Some(label) => format!(", \"{label}\""),
                None => String::new(),
            };
            legend.insert(
                ToolRunner::Device.as_str().to_string(),
                serde_json::json!(format!(
                    "the user's own machine{label}. Acts on the user's own files."
                )),
            );
        }
        serde_json::Value::Object(legend)
    }

    async fn db_query(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let query_fn = self
            .db_query_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("database query not configured".to_string()))?;

        let query = required_string(&arguments, "query")?;
        let limit = arguments
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(100) as usize;

        tracing::info!(limit, "executing db query");
        tracing::debug!(sql = %query, "db query SQL");

        let result = query_fn(query, limit).await?;

        Ok(serde_json::json!({
            "ok": true,
            "result": result,
        })
        .to_string())
    }

    async fn mcp_control(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let handle = self
            .mcp_handle
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("MCP control not configured".to_string()))?;

        let action = required_string(&arguments, "action")?;
        let server = optional_string(&arguments, "server");
        let server_ref = server.as_deref();

        match action.as_str() {
            "status" => {
                let statuses = handle.status(server_ref).await;
                Ok(serde_json::json!({
                    "ok": true,
                    "servers": statuses,
                })
                .to_string())
            }
            "start" => {
                let result = handle
                    .start_server(server_ref)
                    .await
                    .map_err(|e| CoreError::ToolExecution(format!("start failed: {e}")))?;
                let statuses = handle.status(server_ref).await;
                Ok(serde_json::json!({
                    "ok": true,
                    "message": result,
                    "servers": statuses,
                })
                .to_string())
            }
            "stop" => {
                let result = handle
                    .stop_server(server_ref)
                    .await
                    .map_err(|e| CoreError::ToolExecution(format!("stop failed: {e}")))?;
                let statuses = handle.status(server_ref).await;
                Ok(serde_json::json!({
                    "ok": true,
                    "message": result,
                    "servers": statuses,
                })
                .to_string())
            }
            "restart" => {
                let result = handle
                    .restart_server(server_ref)
                    .await
                    .map_err(|e| CoreError::ToolExecution(format!("restart failed: {e}")))?;
                let statuses = handle.status(server_ref).await;
                Ok(serde_json::json!({
                    "ok": true,
                    "message": result,
                    "servers": statuses,
                })
                .to_string())
            }
            _ => Err(CoreError::ToolExecution(format!(
                "unknown MCP control action: {action}"
            ))),
        }
    }

    /// Resolve the conversation the scratchpad tools operate on from the
    /// task-local installed by the service dispatch loop. Errors clearly when
    /// no conversation scope is active (e.g. a non-conversation tool call).
    fn scratchpad_conversation() -> Result<String, CoreError> {
        current_conversation_id().map(|c| c.0).ok_or_else(|| {
            CoreError::ToolExecution("scratchpad requires an active conversation".to_string())
        })
    }

    async fn scratchpad_write(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let conversation_id = Self::scratchpad_conversation()?;
        let write_fn = self
            .scratchpad_write_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("scratchpad not configured".to_string()))?;

        // Accept either a `notes` array or a single `key`+`content`. Each note
        // may carry an optional `type` (default "note"), `sequence`, and `done`.
        let raw: Vec<NewScratchpadNote> =
            if let Some(arr) = arguments.get("notes").and_then(serde_json::Value::as_array) {
                arr.iter().filter_map(parse_new_note).collect()
            } else if arguments.get("key").is_some() || arguments.get("content").is_some() {
                match parse_new_note(&arguments) {
                    Some(note) => vec![note],
                    None => Vec::new(),
                }
            } else {
                return Err(CoreError::ToolExecution(
                "scratchpad_write requires `notes: [{key, content}]` or a single `key` + `content`"
                    .to_string(),
            ));
            };

        if raw.is_empty() {
            return Err(CoreError::ToolExecution(
                "scratchpad_write: no notes provided".to_string(),
            ));
        }

        // Validate each note, then dedupe repeated keys last-wins (a single
        // INSERT can't carry a duplicate ON CONFLICT target). Invalid notes
        // are reported individually rather than failing the whole call.
        let mut rejected: Vec<serde_json::Value> = Vec::new();
        let mut accepted: Vec<NewScratchpadNote> = Vec::new();
        for note in raw {
            if note.key.is_empty() {
                rejected.push(serde_json::json!({"key": note.key, "reason": "empty key"}));
                continue;
            }
            if note.content.len() > MAX_NOTE_BYTES {
                rejected.push(serde_json::json!({
                    "key": note.key,
                    "reason": format!("content exceeds {MAX_NOTE_BYTES} bytes")
                }));
                continue;
            }
            if let Some(existing) = accepted.iter_mut().find(|n| n.key == note.key) {
                *existing = note;
            } else {
                accepted.push(note);
            }
        }

        // Bound the batch: anything past the per-call cap is reported as skipped.
        let mut truncated = false;
        let mut skipped: Vec<String> = Vec::new();
        if accepted.len() > MAX_NOTES_PER_WRITE {
            truncated = true;
            skipped = accepted
                .split_off(MAX_NOTES_PER_WRITE)
                .into_iter()
                .map(|n| n.key)
                .collect();
        }

        let saved = if accepted.is_empty() {
            Vec::new()
        } else {
            write_fn(conversation_id, accepted).await?
        };

        let written: Vec<serde_json::Value> = saved
            .iter()
            .map(|n| serde_json::json!({"key": n.key, "id": n.id, "updated_at": n.updated_at}))
            .collect();

        let mut response = serde_json::json!({"ok": true, "written": written});
        if !rejected.is_empty() {
            response["rejected"] = serde_json::Value::Array(rejected);
        }
        if truncated {
            response["truncated"] = serde_json::Value::Bool(true);
            response["skipped"] = serde_json::json!(skipped);
            response["message"] = serde_json::json!(format!(
                "only the first {MAX_NOTES_PER_WRITE} notes were written; call again with the rest"
            ));
        }
        Ok(response.to_string())
    }

    async fn scratchpad_search(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let conversation_id = Self::scratchpad_conversation()?;

        // `max_results` is required and clamped so a single read is bounded.
        let max_results = arguments
            .get("max_results")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                CoreError::ToolExecution("scratchpad_search requires `max_results`".to_string())
            })? as usize;
        let limit = max_results.clamp(1, MAX_RESULTS_CEILING);

        let keys = optional_string_array(&arguments, "keys");
        let query = optional_string(&arguments, "query");
        // Optional structured filter restricting list/search to one note_type
        // (e.g. only `todo`s). Ignored on the by-keys path (keys are explicit).
        let note_type = optional_string(&arguments, "type");

        // Mode precedence: keys -> query -> list-all. Each path is bounded.
        let mut keys_truncated = false;
        let results =
            if !keys.is_empty() {
                let get_many = self.scratchpad_get_many_fn.as_ref().ok_or_else(|| {
                    CoreError::ToolExecution("scratchpad not configured".to_string())
                })?;
                let mut keys = keys;
                if keys.len() > MAX_KEYS_PER_CALL {
                    keys_truncated = true;
                    keys.truncate(MAX_KEYS_PER_CALL);
                }
                get_many(conversation_id, keys, limit).await?
            } else if let Some(query) = query {
                let search = self.scratchpad_search_fn.as_ref().ok_or_else(|| {
                    CoreError::ToolExecution("scratchpad not configured".to_string())
                })?;
                // The pad is the agent's own working memory, and it
                // re-summarizes as it goes -- so the words it searches with are
                // often not the words it wrote. Embed the query so the store's
                // vector arm can run; an empty vector (no backend, or one that
                // stalled inside `EMBED_TIMEOUT`) reads as "full text only"
                // (#717).
                let (query_embedding, embedding_model) = self.embed_query(&query).await;
                search(
                    conversation_id,
                    query,
                    query_embedding,
                    embedding_model,
                    note_type,
                    limit,
                )
                .await?
            } else {
                let list = self.scratchpad_list_fn.as_ref().ok_or_else(|| {
                    CoreError::ToolExecution("scratchpad not configured".to_string())
                })?;
                list(conversation_id, note_type, limit).await?
            };

        let hit_limit = results.len() >= limit;

        // Enforce the response byte budget so one read can't blow out context.
        // Always include at least one entry even if it alone is large.
        let mut items: Vec<serde_json::Value> = Vec::new();
        let mut bytes = 0usize;
        let mut budget_truncated = false;
        for note in &results {
            let entry = serde_json::json!({
                "key": note.key,
                "content": note.content,
                "type": note.note_type,
                "sequence": note.sequence,
                "done": note.done,
                // So the model can see what it has already pinned without a
                // second call — the cap makes that worth knowing (#597).
                "pinned": note.pinned,
                "updated_at": note.updated_at,
            });
            let size = entry.to_string().len();
            if !items.is_empty() && bytes + size > RESPONSE_BYTE_BUDGET {
                budget_truncated = true;
                break;
            }
            bytes += size;
            items.push(entry);
        }

        let truncated = keys_truncated || budget_truncated || hit_limit;
        let mut response =
            serde_json::json!({"ok": true, "results": items.clone(), "returned": items.len()});
        if truncated {
            response["truncated"] = serde_json::Value::Bool(true);
            response["message"] = serde_json::json!(
                "results were truncated; narrow with a `query`, fewer `keys`, or a smaller scope"
            );
        }
        Ok(response.to_string())
    }

    /// Pin / unpin notes by key (#597).
    ///
    /// The cap is enforced against storage's current state, not against what
    /// this call asks for, so a model that has lost track of its own pins still
    /// gets an accurate refusal naming what is already pinned.
    async fn scratchpad_pin(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let conversation_id = Self::scratchpad_conversation()?;
        let set_pinned = self
            .scratchpad_set_pinned_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("scratchpad not configured".to_string()))?;
        let list = self
            .scratchpad_list_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("scratchpad not configured".to_string()))?;

        // Default to pinning: `pinned` is the less-used half of the tool, and
        // an omitted flag reads as "pin these".
        let pinned = arguments
            .get("pinned")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);
        let keys = optional_string_array(&arguments, "keys");
        if keys.is_empty() {
            return Err(CoreError::ToolExecution(
                "scratchpad_pin requires `keys: [...]`".to_string(),
            ));
        }
        let requested = keys.len();
        let mut keys = keys;
        let mut truncated = false;
        if keys.len() > MAX_KEYS_PER_CALL {
            truncated = true;
            keys.truncate(MAX_KEYS_PER_CALL);
        }

        // Read current state for both the cap check and unknown-key reporting.
        // Bounded by the same ceiling every scratchpad read uses.
        let existing = list(conversation_id.clone(), None, MAX_RESULTS_CEILING).await?;
        let known: Vec<&str> = existing.iter().map(|n| n.key.as_str()).collect();
        let unknown: Vec<String> = keys
            .iter()
            .filter(|k| !known.contains(&k.as_str()))
            .cloned()
            .collect();
        let currently_pinned: Vec<String> = existing
            .iter()
            .filter(|n| n.pinned)
            .map(|n| n.key.clone())
            .collect();

        // Only keys that actually exist count toward the cap — asking to pin a
        // typo'd key must not consume budget or trip the limit.
        let effective: Vec<String> = keys
            .iter()
            .filter(|k| known.contains(&k.as_str()))
            .cloned()
            .collect();
        if effective.is_empty() {
            return Ok(serde_json::json!({
                "ok": true,
                "changed": 0,
                "requested": requested,
                "unknown_keys": unknown,
                "message": "no matching notes; nothing was pinned or unpinned",
            })
            .to_string());
        }
        let to_set = plan_pin(&currently_pinned, &effective, pinned)
            .map_err(|e| CoreError::ToolExecution(format!("scratchpad_pin: {e}")))?;

        let changed = set_pinned(conversation_id, to_set, pinned).await?;
        let mut response = serde_json::json!({
            "ok": true,
            "pinned": pinned,
            "changed": changed,
            "requested": requested,
            "max_pinned": MAX_PINNED_NOTES,
        });
        if !unknown.is_empty() {
            // Reported, not an error: a key that no longer exists is worth
            // knowing about but must not fail the pins that did apply.
            response["unknown_keys"] = serde_json::json!(unknown);
        }
        if truncated {
            response["truncated"] = serde_json::Value::Bool(true);
            response["message"] = serde_json::json!(format!(
                "only the first {MAX_KEYS_PER_CALL} keys were processed; call again for the rest"
            ));
        }
        Ok(response.to_string())
    }

    async fn scratchpad_delete(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let conversation_id = Self::scratchpad_conversation()?;

        let all = arguments
            .get("all")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let keys = optional_string_array(&arguments, "keys");

        // Exactly one mode: refuse both/neither so a stray arg can't mass-delete.
        if all && !keys.is_empty() {
            return Err(CoreError::ToolExecution(
                "scratchpad_delete: pass either `keys` or `all`, not both".to_string(),
            ));
        }
        if !all && keys.is_empty() {
            return Err(CoreError::ToolExecution(
                "scratchpad_delete requires `keys: [...]` or `all: true`".to_string(),
            ));
        }

        if all {
            let clear = self
                .scratchpad_clear_fn
                .as_ref()
                .ok_or_else(|| CoreError::ToolExecution("scratchpad not configured".to_string()))?;
            let deleted = clear(conversation_id).await?;
            return Ok(serde_json::json!({"ok": true, "deleted": deleted}).to_string());
        }

        let delete_many = self
            .scratchpad_delete_many_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("scratchpad not configured".to_string()))?;
        let requested = keys.len();
        let mut keys = keys;
        let mut truncated = false;
        if keys.len() > MAX_KEYS_PER_CALL {
            truncated = true;
            keys.truncate(MAX_KEYS_PER_CALL);
        }
        let deleted = delete_many(conversation_id, keys).await?;

        let mut response =
            serde_json::json!({"ok": true, "deleted": deleted, "requested": requested});
        if truncated {
            response["truncated"] = serde_json::Value::Bool(true);
            response["message"] = serde_json::json!(format!(
                "only the first {MAX_KEYS_PER_CALL} keys were processed; call again for the rest"
            ));
        }
        Ok(response.to_string())
    }

    /// Embed a query and report which model produced the vector.
    ///
    /// Both come from one read of the active backend, so the vector and the
    /// model identifier a search filters on always describe the same backend.
    /// The vector is empty when embeddings are unavailable or the backend
    /// stalled, which every search reads as "take the full-text path"; the
    /// model identifier is then unused.
    async fn embed_query(&self, text: &str) -> (Vec<f32>, String) {
        let Some(backend) = self.embedding.as_ref() else {
            return (Vec::new(), String::new());
        };
        let vector = Self::embed_with(&backend.embed, text)
            .await
            .unwrap_or_default();
        (vector, backend.model.clone())
    }

    /// Embed a single text string, returning None if embeddings are unavailable.
    /// Used for search queries which are always short and don't need chunking.
    async fn embed_text(&self, text: &str) -> Option<Vec<f32>> {
        Self::embed_with(&self.embedding.as_ref()?.embed, text).await
    }

    /// Embed one short text through `embed`, bounded by [`EMBED_TIMEOUT`].
    async fn embed_with(embed: &EmbedFn, text: &str) -> Option<Vec<f32>> {
        match tokio::time::timeout(EMBED_TIMEOUT, embed(vec![text.to_string()])).await {
            Ok(Ok(mut vecs)) => vecs.pop(),
            Ok(Err(e)) => {
                tracing::warn!("failed to embed text: {e}");
                None
            }
            Err(_) => {
                tracing::warn!(
                    timeout = ?EMBED_TIMEOUT,
                    "embedding timed out; falling back to full-text search"
                );
                None
            }
        }
    }
}

fn required_string(args: &serde_json::Value, key: &str) -> Result<String, CoreError> {
    args.get(key)
        .and_then(serde_json::Value::as_str)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| CoreError::ToolExecution(format!("missing required string argument: {key}")))
}

fn optional_string(args: &serde_json::Value, key: &str) -> Option<String> {
    args.get(key)
        .and_then(serde_json::Value::as_str)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn optional_string_array(args: &serde_json::Value, key: &str) -> Vec<String> {
    args.get(key)
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

/// Read a string list the caller may or may not have supplied, for a field
/// where "not supplied" and "supplied empty" mean different things.
///
/// `None` means the write said nothing about the field. `Some` means the write
/// asked to set it, and an empty list then asks to clear it. On the
/// knowledge-base write path that distinction decides whether a stored entry
/// keeps its tags, so the three shapes that are not a list of strings cannot be
/// folded into an empty list the way [`optional_string_array`] folds them:
///
/// - `null` reads as absent. Several model providers encode "I am not setting
///   this field" as an explicit null, and reading that as an empty list clears
///   the field on every write they send.
/// - A value that is not a list is an error. The caller asked to set the field
///   and named something this cannot read, so storing an empty list would
///   answer a set with a wipe, and report success.
/// - A list holding a value that is not a string is an error, for the same
///   reason applied to one element: the caller named that tag.
///
/// A blank or whitespace-only string is dropped rather than refused. An empty
/// tag is not a tag, so trimming it away loses no intent.
fn supplied_string_array(
    args: &serde_json::Value,
    key: &str,
) -> Result<Option<Vec<String>>, CoreError> {
    let Some(value) = args.get(key) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let items = value.as_array().ok_or_else(|| {
        CoreError::ToolExecution(format!(
            "`{key}` must be a list of strings, and this one is {}",
            json_shape(value)
        ))
    })?;
    let mut supplied = Vec::with_capacity(items.len());
    for item in items {
        let text = item.as_str().ok_or_else(|| {
            CoreError::ToolExecution(format!(
                "every tag in `{key}` must be a string, and one of them is {}",
                json_shape(item)
            ))
        })?;
        let text = text.trim();
        if !text.is_empty() {
            supplied.push(text.to_string());
        }
    }
    Ok(Some(supplied))
}

/// Name the JSON shape of `value`, so an error can tell the caller what it
/// actually sent rather than only what was wanted.
fn json_shape(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "a list",
        serde_json::Value::Object(_) => "an object",
    }
}

/// Read one write spec's `new_tag_descriptions` map, keyed by the normalized
/// tag name.
///
/// Why normalized: the tag vocabulary keys on the normalized name, and the
/// model writes each field in whatever shape reads well - `"Topic: Embeddings"`
/// as the description key beside `"topic:embeddings"` in `tags`, or the
/// reverse. Matching the two raw strings drops the description in both
/// directions, and a tag with no description embeds as its bare name.
///
/// Each description is capped at [`TAG_DESCRIPTION_MAX_CHARS`]. It is truncated
/// rather than refused, because refusing it would cost the tag its description
/// and drop the check back to matching bare names.
///
/// Where two keys normalize together, one of them wins. Which one is
/// deterministic but arbitrary: this workspace does not enable `serde_json`'s
/// `preserve_order`, so a `Value::Object` iterates in byte order rather than in
/// the order the model wrote its keys. Two keys for one tag is a malformed
/// argument, not a case worth a rule.
fn normalized_tag_descriptions(spec: &serde_json::Value) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let Some(map) = spec
        .get("new_tag_descriptions")
        .and_then(serde_json::Value::as_object)
    else {
        return out;
    };
    for (name, value) in map {
        let Some(description) = value
            .as_str()
            .map(str::trim)
            .filter(|d| !d.is_empty())
            .map(cap_tag_description)
        else {
            continue;
        };
        let key = normalize_tag(name);
        if key.is_empty() {
            continue;
        }
        out.entry(key).or_insert(description);
    }
    out
}

/// Cut one tag description down to [`TAG_DESCRIPTION_MAX_CHARS`] characters.
///
/// Counted in characters, not bytes, so the cut always lands on a character
/// boundary and a description in any script gets the same allowance. Trailing
/// whitespace left by the cut goes, so the stored text never ends mid-space.
fn cap_tag_description(description: &str) -> String {
    if description.chars().count() <= TAG_DESCRIPTION_MAX_CHARS {
        return description.to_string();
    }
    description
        .chars()
        .take(TAG_DESCRIPTION_MAX_CHARS)
        .collect::<String>()
        .trim_end()
        .to_string()
}

fn optional_string_array_nonempty(args: &serde_json::Value, key: &str) -> Option<Vec<String>> {
    let values = optional_string_array(args, key);
    if values.is_empty() {
        None
    } else {
        Some(values)
    }
}

/// Parse one scratchpad note object (`{key, content, type?, sequence?, done?}`)
/// into a [`NewScratchpadNote`]. Returns `None` when `key` or `content` is
/// absent (the caller treats that as a malformed note). `type` defaults to
/// [`DEFAULT_NOTE_TYPE`]; the key is trimmed (emptiness is validated upstream).
fn parse_new_note(obj: &serde_json::Value) -> Option<NewScratchpadNote> {
    let key = obj.get("key").and_then(serde_json::Value::as_str)?;
    let content = obj.get("content").and_then(serde_json::Value::as_str)?;
    let note_type = obj
        .get("type")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(desktop_assistant_core::domain::DEFAULT_NOTE_TYPE)
        .to_string();
    let sequence = obj
        .get("sequence")
        .and_then(serde_json::Value::as_i64)
        .map(|v| v as i32);
    let done = obj
        .get("done")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    Some(NewScratchpadNote {
        key: key.trim().to_string(),
        content: content.to_string(),
        note_type,
        sequence,
        done,
        // Filled in by the write closure, the one place every scratchpad write
        // passes through (#717).
        embedding: None,
    })
}

/// The user/device identity fields of `builtin_sys_props`, resolved once from
/// either the connecting client's self-reported context or the daemon-host
/// fallback. A named struct (rather than a bare tuple) keeps the two resolution
/// arms and the JSON assembly legible about which value is which (#558).
struct IdentityFields {
    /// Where the fields came from: `"client"` or `"daemon_host_fallback"`.
    source: &'static str,
    real_name: Option<String>,
    username: Option<String>,
    home_dir: Option<String>,
    hostname: Option<String>,
    os: Option<String>,
    timezone: Option<String>,
}

fn detect_username() -> Option<String> {
    ["USER", "LOGNAME", "USERNAME"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .map(|v| v.trim().to_string())
        .find(|v| !v.is_empty())
}

fn detect_home_dir() -> Option<String> {
    ["HOME", "USERPROFILE"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .map(|v| v.trim().to_string())
        .find(|v| !v.is_empty())
}

fn detect_daemon_cwd() -> Option<String> {
    std::env::current_dir()
        .ok()
        .map(|p| p.display().to_string())
        .filter(|s| !s.is_empty())
}

fn detect_xdg_dirs() -> serde_json::Value {
    let home = detect_home_dir();
    let fallback_base = home
        .as_ref()
        .map(|h| PathBuf::from(h).join(".local"))
        .unwrap_or_else(|| PathBuf::from(".local"));

    let config = std::env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| fallback_base.join("config").display().to_string());
    let data = std::env::var("XDG_DATA_HOME")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| fallback_base.join("share").display().to_string());
    let state = std::env::var("XDG_STATE_HOME")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| fallback_base.join("state").display().to_string());
    let cache = std::env::var("XDG_CACHE_HOME")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| fallback_base.join("cache").display().to_string());
    let runtime = std::env::var("XDG_RUNTIME_DIR")
        .ok()
        .filter(|s| !s.trim().is_empty());

    serde_json::json!({
        "config": config,
        "data": data,
        "state": state,
        "cache": cache,
        "runtime": runtime,
    })
}

fn detect_shell() -> Option<String> {
    ["SHELL", "COMSPEC"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .map(|v| v.trim().to_string())
        .find(|v| !v.is_empty())
}

fn detect_locale() -> Option<String> {
    ["LC_ALL", "LANG"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .map(|v| v.trim().to_string())
        .find(|v| !v.is_empty())
}

fn detect_session_type() -> Option<String> {
    std::env::var("XDG_SESSION_TYPE")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn detect_hostname() -> Option<String> {
    if let Ok(hostname) = std::env::var("HOSTNAME") {
        let trimmed = hostname.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    if let Ok(contents) = fs::read_to_string("/etc/hostname") {
        let trimmed = contents.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    None
}

fn detect_os_version() -> Option<String> {
    if std::env::consts::OS != "linux" {
        return None;
    }

    let contents = fs::read_to_string("/etc/os-release").ok()?;
    parse_os_release_field(&contents, "PRETTY_NAME")
        .or_else(|| parse_os_release_field(&contents, "VERSION"))
        .or_else(|| parse_os_release_field(&contents, "VERSION_ID"))
}

fn parse_os_release_field(contents: &str, key: &str) -> Option<String> {
    contents.lines().find_map(|line| {
        let (line_key, raw_value) = line.split_once('=')?;
        if line_key.trim() != key {
            return None;
        }
        let value = raw_value.trim().trim_matches('"').trim_matches('\'');
        if value.is_empty() {
            None
        } else {
            Some(value.to_string())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::ports::knowledge::{KnowledgeSearchPage, ScopeSize};

    #[test]
    fn builtin_provider_map_is_exhaustive() {
        // Every known builtin tool must classify into one of the authored
        // PROVIDER_GROUPS — so a NEW builtin added without a mapping fails here
        // instead of silently registering unclassified (spec requirement).
        let group_ids: Vec<&str> = BuiltinToolService::PROVIDER_GROUPS
            .iter()
            .map(|(id, _)| *id)
            .collect();
        for name in BuiltinToolService::ALL_TOOL_NAMES {
            let group = BuiltinToolService::provider_group(name).unwrap_or_else(|| {
                panic!("builtin '{name}' has no provider group — classify it in provider_group()")
            });
            assert!(
                group_ids.contains(&group),
                "builtin '{name}' maps to '{group}', which is not an authored PROVIDER_GROUP"
            );
        }
        // The same must hold for what a runtime actually advertises, including
        // the capability-gated tools: an unclassified builtin registers under
        // the generic fallback group instead of its own.
        for def in fully_wired_service().tool_definitions() {
            assert!(
                BuiltinToolService::provider_group(&def.name).is_some(),
                "runtime builtin '{}' is unclassified",
                def.name
            );
        }
    }

    /// The pre-#141 docstring on `with_database` claimed "read-only SQL
    /// access" — which the implementation did not enforce. Comment-vs-
    /// behaviour drift on a security-relevant surface is a real bug;
    /// the audit pass in #141 surfaced exactly this kind of drift on
    /// the `execute_database_query` tool.
    ///
    /// This test pins the docstring against the post-#141 contract.
    /// If you change the wording, update this test in the same commit
    /// so the assertion still describes what the code actually does.
    ///
    /// The check reads the source file at compile time via
    /// `include_str!` so we're asserting against the *literal* text
    /// the reviewer will see, not against something the compiler
    /// could fold away.
    #[test]
    fn comment_in_builtin_rs_matches_actual_security_posture() {
        const SRC: &str = include_str!("builtin.rs");

        // Locate the doc-comment block immediately preceding
        // `pub fn with_database(`. The block is the contiguous run of
        // `///` lines above the function signature.
        let fn_pos = SRC
            .find("pub fn with_database(")
            .expect("with_database fn declaration must exist");
        let preceding = &SRC[..fn_pos];
        let doc_block: String = preceding
            .lines()
            .rev()
            .take_while(|l| {
                let t = l.trim_start();
                t.starts_with("///") || t.is_empty()
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n")
            .to_ascii_lowercase();

        // Forbidden: the misleading "read-only" claim from before
        // #141. It's misleading in two ways — the tool *did* allow
        // writes (to the scratch namespace and, footgun, to qualified
        // public tables), and even the "read-only" reads were
        // unscoped across tenants.
        assert!(
            !doc_block.contains("read-only sql access"),
            "with_database docstring still claims `read-only SQL access`; \
             pre-#141 wording is back. Current block:\n---\n{doc_block}\n---"
        );

        // Required: the doc must surface the two facts the LLM-
        // exposed tool actually enforces post-#141 — SELECT-only and
        // per-user scoping. Word choice is flexible (`scoped` /
        // `tenant` / `user_id` all read as the same thing); the test
        // just refuses an empty mention.
        assert!(
            doc_block.contains("select"),
            "with_database docstring must mention SELECT-only enforcement. \
             Current block:\n---\n{doc_block}\n---"
        );
        assert!(
            doc_block.contains("user_id")
                || doc_block.contains("per-user")
                || doc_block.contains("tenant"),
            "with_database docstring must mention per-user / user_id / tenant scoping. \
             Current block:\n---\n{doc_block}\n---"
        );
    }

    #[test]
    fn builtins_expose_expected_tools() {
        let service = BuiltinToolService::new();
        let names: Vec<String> = service
            .tool_definitions()
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert!(names.contains(&TOOL_KB_WRITE.to_string()));
        assert!(names.contains(&TOOL_KB_SEARCH.to_string()));
        assert!(names.contains(&TOOL_KB_DELETE.to_string()));
        assert!(names.contains(&TOOL_SEARCH.to_string()));
        assert!(names.contains(&TOOL_SYS_PROPS.to_string()));
        assert!(names.contains(&TOOL_DB_QUERY.to_string()));
        assert!(names.contains(&TOOL_MCP_CONTROL.to_string()));
        assert!(names.contains(&TOOL_CONV_SEARCH.to_string()));
        assert!(names.contains(&TOOL_SCRATCHPAD_WRITE.to_string()));
        assert!(names.contains(&TOOL_SCRATCHPAD_SEARCH.to_string()));
        assert!(names.contains(&TOOL_SCRATCHPAD_DELETE.to_string()));
    }

    #[test]
    fn kb_write_tags_description_urges_specific_facets() {
        // Generic tags ("instruction", "memory") make KB entries fragment and
        // over-surface. Both the single-write and the batch `tags` schema
        // descriptions must push the two-level rule (a specific facet, not just
        // a bare kind) so the in-schema hint matches the system-prompt guidance.
        let service = BuiltinToolService::new();
        let def = service
            .tool_definitions()
            .into_iter()
            .find(|t| t.name == TOOL_KB_WRITE)
            .expect("kb_write tool is advertised");
        let props = &def.parameters["properties"];

        let single = props["tags"]["description"]
            .as_str()
            .expect("single-write tags has a description");
        assert!(
            single.to_lowercase().contains("specific"),
            "single-write tags description must urge a specific facet: {single}"
        );
        assert!(
            single.contains("topic:") && single.contains("tool:"),
            "single-write tags description must list facet examples: {single}"
        );

        let batch = props["entries"]["items"]["properties"]["tags"]["description"]
            .as_str()
            .expect("batch tags must carry a description too");
        assert!(
            batch.to_lowercase().contains("specific"),
            "batch tags description must urge a specific facet: {batch}"
        );
    }

    // --- Scratchpad tools (#184) ---

    use std::sync::Arc;

    use desktop_assistant_core::domain::{ConversationId, ScratchpadNote};
    use desktop_assistant_core::ports::conversation_ctx::with_conversation_id;

    /// Build a BuiltinToolService whose scratchpad closures share one
    /// in-memory note store, so write/search/delete round-trips are testable
    /// without Postgres. Returns the service and a handle to the store.
    fn scratchpad_service() -> (
        BuiltinToolService,
        Arc<std::sync::Mutex<Vec<ScratchpadNote>>>,
    ) {
        scratchpad_service_with_search(None)
    }

    /// As [`scratchpad_service`], but with `search` standing in for the
    /// in-memory substring search, so a test can observe exactly what the tool
    /// hands the store.
    fn scratchpad_service_with_search(
        search: Option<ScratchpadSearchFn>,
    ) -> (
        BuiltinToolService,
        Arc<std::sync::Mutex<Vec<ScratchpadNote>>>,
    ) {
        use std::pin::Pin;
        let store: Arc<std::sync::Mutex<Vec<ScratchpadNote>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));

        let w = Arc::clone(&store);
        let write_fn: ScratchpadWriteFn =
            Arc::new(move |conv: String, notes: Vec<NewScratchpadNote>| {
                let store = Arc::clone(&w);
                Box::pin(async move {
                    let mut guard = store.lock().unwrap();
                    let mut saved = Vec::new();
                    for (i, note) in notes.into_iter().enumerate() {
                        if let Some(existing) = guard
                            .iter_mut()
                            .find(|n| n.conversation_id == conv && n.key == note.key)
                        {
                            existing.content = note.content;
                            existing.note_type = note.note_type;
                            existing.sequence = note.sequence;
                            existing.done = note.done;
                            existing.updated_at = "t1".into();
                            saved.push(existing.clone());
                        } else {
                            let mut n = ScratchpadNote::new(
                                format!("id-{i}-{}", note.key),
                                &conv,
                                &note.key,
                                &note.content,
                            );
                            n.note_type = note.note_type;
                            n.sequence = note.sequence;
                            n.done = note.done;
                            n.updated_at = "t0".into();
                            guard.push(n.clone());
                            saved.push(n);
                        }
                    }
                    Ok(saved)
                })
                    as Pin<
                        Box<
                            dyn std::future::Future<Output = Result<Vec<ScratchpadNote>, CoreError>>
                                + Send,
                        >,
                    >
            });

        let g = Arc::clone(&store);
        let get_many_fn: ScratchpadGetManyFn =
            Arc::new(move |conv: String, keys: Vec<String>, limit: usize| {
                let store = Arc::clone(&g);
                Box::pin(async move {
                    let guard = store.lock().unwrap();
                    Ok(guard
                        .iter()
                        .filter(|n| n.conversation_id == conv && keys.contains(&n.key))
                        .take(limit)
                        .cloned()
                        .collect())
                })
            });

        let l = Arc::clone(&store);
        let list_fn: ScratchpadListFn = Arc::new(
            move |conv: String, note_type: Option<String>, limit: usize| {
                let store = Arc::clone(&l);
                Box::pin(async move {
                    let guard = store.lock().unwrap();
                    let mut notes: Vec<ScratchpadNote> = guard
                        .iter()
                        .filter(|n| n.conversation_id == conv)
                        .filter(|n| note_type.as_deref().is_none_or(|t| n.note_type == t))
                        .cloned()
                        .collect();
                    // Mirror the store ordering: type, then sequence ascending
                    // (nulls last), then recency (timestamps omitted here).
                    notes.sort_by(|a, b| {
                        a.note_type
                            .cmp(&b.note_type)
                            .then_with(|| match (a.sequence, b.sequence) {
                                (Some(x), Some(y)) => x.cmp(&y),
                                (Some(_), None) => std::cmp::Ordering::Less,
                                (None, Some(_)) => std::cmp::Ordering::Greater,
                                (None, None) => std::cmp::Ordering::Equal,
                            })
                    });
                    notes.truncate(limit);
                    Ok(notes)
                })
            },
        );

        let s = Arc::clone(&store);
        let search_fn: ScratchpadSearchFn = search.unwrap_or_else(|| {
            Arc::new(
                move |conv: String,
                      query: String,
                      _query_embedding: Vec<f32>,
                      _embedding_model: String,
                      note_type: Option<String>,
                      limit: usize| {
                    let store = Arc::clone(&s);
                    Box::pin(async move {
                        let guard = store.lock().unwrap();
                        Ok(guard
                            .iter()
                            .filter(|n| {
                                n.conversation_id == conv
                                    && (n.content.contains(&query) || n.key.contains(&query))
                                    && note_type.as_deref().is_none_or(|t| n.note_type == t)
                            })
                            .take(limit)
                            .cloned()
                            .collect())
                    })
                },
            )
        });

        let d = Arc::clone(&store);
        let delete_many_fn: ScratchpadDeleteManyFn =
            Arc::new(move |conv: String, keys: Vec<String>| {
                let store = Arc::clone(&d);
                Box::pin(async move {
                    let mut guard = store.lock().unwrap();
                    let before = guard.len();
                    guard.retain(|n| !(n.conversation_id == conv && keys.contains(&n.key)));
                    Ok((before - guard.len()) as u64)
                })
            });

        let c = Arc::clone(&store);
        let clear_fn: ScratchpadClearFn = Arc::new(move |conv: String| {
            let store = Arc::clone(&c);
            Box::pin(async move {
                let mut guard = store.lock().unwrap();
                let before = guard.len();
                guard.retain(|n| n.conversation_id != conv);
                Ok((before - guard.len()) as u64)
            })
        });

        // #597 pin/unpin against the same in-memory pad. Mirrors the store's
        // semantics: only notes that exist change, and the count returned is
        // of notes actually CHANGED so a re-pin reports 0.
        let sp = Arc::clone(&store);
        let set_pinned_fn: ScratchpadSetPinnedFn =
            Arc::new(move |conv: String, keys: Vec<String>, pinned: bool| {
                let store = Arc::clone(&sp);
                Box::pin(async move {
                    let mut guard = store.lock().unwrap();
                    let mut changed = 0u64;
                    for note in guard
                        .iter_mut()
                        .filter(|n| n.conversation_id == conv && keys.contains(&n.key))
                    {
                        if note.pinned != pinned {
                            note.pinned = pinned;
                            changed += 1;
                        }
                    }
                    Ok(changed)
                })
                    as Pin<Box<dyn std::future::Future<Output = Result<u64, CoreError>> + Send>>
            });

        let service = BuiltinToolService::new()
            .with_scratchpad(
                write_fn,
                get_many_fn,
                list_fn,
                search_fn,
                delete_many_fn,
                clear_fn,
            )
            .with_scratchpad_pin(set_pinned_fn);
        (service, store)
    }

    fn parse(s: &str) -> serde_json::Value {
        serde_json::from_str(s).unwrap()
    }

    #[tokio::test]
    async fn scratchpad_requires_active_conversation() {
        // Closures configured, but no conversation scope installed.
        let (service, _store) = scratchpad_service();
        for (tool, args) in [
            (
                TOOL_SCRATCHPAD_WRITE,
                serde_json::json!({"key": "k", "content": "v"}),
            ),
            (
                TOOL_SCRATCHPAD_SEARCH,
                serde_json::json!({"max_results": 10}),
            ),
            (TOOL_SCRATCHPAD_DELETE, serde_json::json!({"all": true})),
        ] {
            let result = service.execute_tool(tool, args).await;
            assert!(
                matches!(&result, Err(CoreError::ToolExecution(m)) if m.contains("active conversation")),
                "{tool} must require an active conversation, got {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn pin_marks_the_note_and_shows_up_in_search() {
        let (service, store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            service
                .execute_tool(
                    TOOL_SCRATCHPAD_WRITE,
                    serde_json::json!({"key": "deploy-target", "content": "k3s at 192.168.1.2"}),
                )
                .await
                .unwrap();
            let out = parse(
                &service
                    .execute_tool(
                        TOOL_SCRATCHPAD_PIN,
                        serde_json::json!({"keys": ["deploy-target"]}),
                    )
                    .await
                    .unwrap(),
            );
            assert_eq!(out["ok"], true);
            assert_eq!(out["changed"], 1);
            assert_eq!(out["pinned"], true, "`pinned` defaults to true");
            assert!(store.lock().unwrap()[0].pinned);

            // The model must be able to see what it has already pinned.
            let found = parse(
                &service
                    .execute_tool(
                        TOOL_SCRATCHPAD_SEARCH,
                        serde_json::json!({"max_results": 10}),
                    )
                    .await
                    .unwrap(),
            );
            assert_eq!(found["results"][0]["pinned"], true);
        })
        .await;
    }

    #[tokio::test]
    async fn unpin_clears_the_flag() {
        let (service, store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            service
                .execute_tool(
                    TOOL_SCRATCHPAD_WRITE,
                    serde_json::json!({"key": "k", "content": "v"}),
                )
                .await
                .unwrap();
            service
                .execute_tool(TOOL_SCRATCHPAD_PIN, serde_json::json!({"keys": ["k"]}))
                .await
                .unwrap();
            let out = parse(
                &service
                    .execute_tool(
                        TOOL_SCRATCHPAD_PIN,
                        serde_json::json!({"keys": ["k"], "pinned": false}),
                    )
                    .await
                    .unwrap(),
            );
            assert_eq!(out["changed"], 1);
            assert!(!store.lock().unwrap()[0].pinned);
        })
        .await;
    }

    #[tokio::test]
    async fn rewriting_a_pinned_note_keeps_the_pin() {
        // The footgun this design exists to avoid: `write` is an upsert, so if
        // it carried `pinned` an ordinary content update would silently unpin.
        let (service, store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            service
                .execute_tool(
                    TOOL_SCRATCHPAD_WRITE,
                    serde_json::json!({"key": "k", "content": "first"}),
                )
                .await
                .unwrap();
            service
                .execute_tool(TOOL_SCRATCHPAD_PIN, serde_json::json!({"keys": ["k"]}))
                .await
                .unwrap();
            service
                .execute_tool(
                    TOOL_SCRATCHPAD_WRITE,
                    serde_json::json!({"key": "k", "content": "updated"}),
                )
                .await
                .unwrap();
            let guard = store.lock().unwrap();
            assert_eq!(guard[0].content, "updated");
            assert!(guard[0].pinned, "updating a note must not clear its pin");
        })
        .await;
    }

    #[tokio::test]
    async fn pin_respects_max_pinned_notes_at_the_tool_boundary() {
        let (service, store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            for i in 0..=MAX_PINNED_NOTES {
                service
                    .execute_tool(
                        TOOL_SCRATCHPAD_WRITE,
                        serde_json::json!({"key": format!("k{i}"), "content": "v"}),
                    )
                    .await
                    .unwrap();
            }
            for i in 0..MAX_PINNED_NOTES {
                service
                    .execute_tool(
                        TOOL_SCRATCHPAD_PIN,
                        serde_json::json!({"keys": [format!("k{i}")]}),
                    )
                    .await
                    .unwrap();
            }
            let over = service
                .execute_tool(
                    TOOL_SCRATCHPAD_PIN,
                    serde_json::json!({"keys": [format!("k{MAX_PINNED_NOTES}")]}),
                )
                .await;
            assert!(
                matches!(&over, Err(CoreError::ToolExecution(m)) if m.contains("at most 5")),
                "pinning past the cap must be refused, got {over:?}"
            );
            // ...and must change nothing.
            let guard = store.lock().unwrap();
            assert_eq!(guard.iter().filter(|n| n.pinned).count(), MAX_PINNED_NOTES);
            assert!(
                !guard
                    .iter()
                    .any(|n| n.key == format!("k{MAX_PINNED_NOTES}") && n.pinned)
            );
        })
        .await;
    }

    #[tokio::test]
    async fn pinning_an_unknown_key_is_reported_not_an_error() {
        let (service, _store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            let out = parse(
                &service
                    .execute_tool(TOOL_SCRATCHPAD_PIN, serde_json::json!({"keys": ["nope"]}))
                    .await
                    .unwrap(),
            );
            assert_eq!(out["ok"], true);
            assert_eq!(out["changed"], 0);
            assert_eq!(out["unknown_keys"][0], "nope");
        })
        .await;
    }

    #[tokio::test]
    async fn pin_requires_keys() {
        let (service, _store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            let out = service
                .execute_tool(TOOL_SCRATCHPAD_PIN, serde_json::json!({}))
                .await;
            assert!(
                matches!(&out, Err(CoreError::ToolExecution(m)) if m.contains("requires `keys")),
                "got {out:?}"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn deleting_a_pinned_note_takes_the_pin_with_it() {
        let (service, store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            service
                .execute_tool(
                    TOOL_SCRATCHPAD_WRITE,
                    serde_json::json!({"key": "k", "content": "v"}),
                )
                .await
                .unwrap();
            service
                .execute_tool(TOOL_SCRATCHPAD_PIN, serde_json::json!({"keys": ["k"]}))
                .await
                .unwrap();
            service
                .execute_tool(TOOL_SCRATCHPAD_DELETE, serde_json::json!({"keys": ["k"]}))
                .await
                .unwrap();
            assert!(
                store.lock().unwrap().is_empty(),
                "the note is gone, so nothing can still be pinned"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn pin_tool_is_only_advertised_when_wired() {
        // Capability gating: an embedder that predates #597 must not see a tool
        // the service cannot service.
        let (wired, _store) = scratchpad_service();
        assert!(
            wired
                .tool_definitions()
                .iter()
                .any(|d| d.name == TOOL_SCRATCHPAD_PIN),
            "pin must be advertised once its write is wired"
        );
        assert!(
            !BuiltinToolService::new()
                .tool_definitions()
                .iter()
                .any(|d| d.name == TOOL_SCRATCHPAD_PIN),
            "pin must not be advertised without a wired write"
        );
    }

    #[tokio::test]
    async fn scratchpad_write_search_delete_roundtrip() {
        let (service, _store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            // Batch write two notes.
            let written = service
                .execute_tool(
                    TOOL_SCRATCHPAD_WRITE,
                    serde_json::json!({"notes": [
                        {"key": "goal", "content": "ship the scratchpad"},
                        {"key": "q", "content": "which database to use"}
                    ]}),
                )
                .await
                .unwrap();
            assert_eq!(parse(&written)["written"].as_array().unwrap().len(), 2);

            // List (no query) returns both.
            let listed = service
                .execute_tool(
                    TOOL_SCRATCHPAD_SEARCH,
                    serde_json::json!({"max_results": 10}),
                )
                .await
                .unwrap();
            assert_eq!(parse(&listed)["results"].as_array().unwrap().len(), 2);

            // Search by query matches one.
            let hit = service
                .execute_tool(
                    TOOL_SCRATCHPAD_SEARCH,
                    serde_json::json!({"query": "database", "max_results": 10}),
                )
                .await
                .unwrap();
            let results = parse(&hit);
            assert_eq!(results["results"].as_array().unwrap().len(), 1);
            assert_eq!(results["results"][0]["key"], "q");

            // Fetch by keys.
            let by_key = service
                .execute_tool(
                    TOOL_SCRATCHPAD_SEARCH,
                    serde_json::json!({"keys": ["goal"], "max_results": 10}),
                )
                .await
                .unwrap();
            assert_eq!(parse(&by_key)["results"][0]["key"], "goal");

            // Upsert by key updates content, not count.
            service
                .execute_tool(
                    TOOL_SCRATCHPAD_WRITE,
                    serde_json::json!({"key": "goal", "content": "ship it well"}),
                )
                .await
                .unwrap();
            let after = service
                .execute_tool(
                    TOOL_SCRATCHPAD_SEARCH,
                    serde_json::json!({"max_results": 10}),
                )
                .await
                .unwrap();
            assert_eq!(parse(&after)["results"].as_array().unwrap().len(), 2);

            // Delete one key.
            let del = service
                .execute_tool(TOOL_SCRATCHPAD_DELETE, serde_json::json!({"keys": ["q"]}))
                .await
                .unwrap();
            assert_eq!(parse(&del)["deleted"], 1);

            // Delete all.
            let cleared = service
                .execute_tool(TOOL_SCRATCHPAD_DELETE, serde_json::json!({"all": true}))
                .await
                .unwrap();
            assert_eq!(parse(&cleared)["deleted"], 1);
        })
        .await;
    }

    #[tokio::test]
    async fn scratchpad_write_rejects_empty_key_and_oversize_content() {
        let (service, _store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            let huge = "x".repeat(MAX_NOTE_BYTES + 1);
            let result = service
                .execute_tool(
                    TOOL_SCRATCHPAD_WRITE,
                    serde_json::json!({"notes": [
                        {"key": "", "content": "no key"},
                        {"key": "big", "content": huge},
                        {"key": "ok", "content": "fine"}
                    ]}),
                )
                .await
                .unwrap();
            let json = parse(&result);
            assert_eq!(
                json["written"].as_array().unwrap().len(),
                1,
                "only the valid note is written"
            );
            assert_eq!(json["written"][0]["key"], "ok");
            assert_eq!(json["rejected"].as_array().unwrap().len(), 2);
        })
        .await;
    }

    #[tokio::test]
    async fn scratchpad_write_truncates_over_cap() {
        let (service, _store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            let notes: Vec<serde_json::Value> = (0..MAX_NOTES_PER_WRITE + 5)
                .map(|i| serde_json::json!({"key": format!("k{i}"), "content": "v"}))
                .collect();
            let result = service
                .execute_tool(TOOL_SCRATCHPAD_WRITE, serde_json::json!({"notes": notes}))
                .await
                .unwrap();
            let json = parse(&result);
            assert_eq!(json["truncated"], true);
            assert_eq!(
                json["written"].as_array().unwrap().len(),
                MAX_NOTES_PER_WRITE
            );
            assert_eq!(json["skipped"].as_array().unwrap().len(), 5);
        })
        .await;
    }

    #[tokio::test]
    async fn scratchpad_search_requires_max_results() {
        let (service, _store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            let result = service
                .execute_tool(TOOL_SCRATCHPAD_SEARCH, serde_json::json!({"query": "x"}))
                .await;
            assert!(
                matches!(&result, Err(CoreError::ToolExecution(m)) if m.contains("max_results")),
                "search must require max_results, got {result:?}"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn scratchpad_delete_requires_exactly_one_mode() {
        let (service, _store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            // Neither.
            let neither = service
                .execute_tool(TOOL_SCRATCHPAD_DELETE, serde_json::json!({}))
                .await;
            assert!(matches!(neither, Err(CoreError::ToolExecution(_))));
            // Both.
            let both = service
                .execute_tool(
                    TOOL_SCRATCHPAD_DELETE,
                    serde_json::json!({"keys": ["a"], "all": true}),
                )
                .await;
            assert!(matches!(both, Err(CoreError::ToolExecution(_))));
        })
        .await;
    }

    #[tokio::test]
    async fn scratchpad_search_byte_budget_truncates() {
        let (service, _store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            // Write enough near-max notes that the serialized list exceeds the
            // response byte budget, forcing truncation.
            let big = "y".repeat(MAX_NOTE_BYTES - 100);
            let count = (RESPONSE_BYTE_BUDGET / MAX_NOTE_BYTES) + 3;
            let notes: Vec<serde_json::Value> = (0..count)
                .map(|i| serde_json::json!({"key": format!("k{i}"), "content": big}))
                .collect();
            // Cap is MAX_NOTES_PER_WRITE; write in chunks if needed. count is
            // small (< cap for 20KB/8KB), so a single call suffices.
            service
                .execute_tool(TOOL_SCRATCHPAD_WRITE, serde_json::json!({"notes": notes}))
                .await
                .unwrap();

            let listed = service
                .execute_tool(
                    TOOL_SCRATCHPAD_SEARCH,
                    serde_json::json!({"max_results": 100}),
                )
                .await
                .unwrap();
            let json = parse(&listed);
            assert_eq!(
                json["truncated"], true,
                "oversized list must signal truncation"
            );
            let returned = json["results"].as_array().unwrap().len();
            assert!(
                returned < count,
                "fewer than all notes are returned under the byte budget"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn scratchpad_write_persists_type_sequence_done() {
        let (service, _store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            service
                .execute_tool(
                    TOOL_SCRATCHPAD_WRITE,
                    serde_json::json!({
                        "key": "t1", "content": "wire the migration",
                        "type": "todo", "sequence": 2, "done": false
                    }),
                )
                .await
                .unwrap();

            let by_key = service
                .execute_tool(
                    TOOL_SCRATCHPAD_SEARCH,
                    serde_json::json!({"keys": ["t1"], "max_results": 10}),
                )
                .await
                .unwrap();
            let note = &parse(&by_key)["results"][0];
            assert_eq!(note["type"], "todo");
            assert_eq!(note["sequence"], 2);
            assert_eq!(note["done"], false);

            // Re-writing the same key flips `done` (the check-off path).
            service
                .execute_tool(
                    TOOL_SCRATCHPAD_WRITE,
                    serde_json::json!({
                        "key": "t1", "content": "wire the migration",
                        "type": "todo", "sequence": 2, "done": true
                    }),
                )
                .await
                .unwrap();
            let after = service
                .execute_tool(
                    TOOL_SCRATCHPAD_SEARCH,
                    serde_json::json!({"keys": ["t1"], "max_results": 10}),
                )
                .await
                .unwrap();
            assert_eq!(parse(&after)["results"][0]["done"], true);
        })
        .await;
    }

    #[tokio::test]
    async fn scratchpad_search_filters_by_type() {
        let (service, _store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            service
                .execute_tool(
                    TOOL_SCRATCHPAD_WRITE,
                    serde_json::json!({"notes": [
                        {"key": "t1", "content": "do a thing", "type": "todo", "sequence": 1},
                        {"key": "n1", "content": "a plain note", "type": "note"}
                    ]}),
                )
                .await
                .unwrap();

            let todos = service
                .execute_tool(
                    TOOL_SCRATCHPAD_SEARCH,
                    serde_json::json!({"type": "todo", "max_results": 10}),
                )
                .await
                .unwrap();
            let results = parse(&todos);
            assert_eq!(results["results"].as_array().unwrap().len(), 1);
            assert_eq!(results["results"][0]["key"], "t1");
        })
        .await;
    }

    /// Somewhere to record the `(query vector, model)` pair the search tool
    /// hands the store.
    type QueryRecorder = Arc<std::sync::Mutex<Option<(Vec<f32>, String)>>>;

    fn query_recorder() -> QueryRecorder {
        Arc::new(std::sync::Mutex::new(None))
    }

    /// A scratchpad search that records what it was handed and returns nothing.
    fn recording_search(recorder: &QueryRecorder) -> ScratchpadSearchFn {
        let recorder = Arc::clone(recorder);
        Arc::new(
            move |_conv: String,
                  _query: String,
                  embedding: Vec<f32>,
                  model: String,
                  _note_type: Option<String>,
                  _limit: usize| {
                let recorder = Arc::clone(&recorder);
                Box::pin(async move {
                    *recorder.lock().unwrap() = Some((embedding, model));
                    Ok(Vec::new())
                })
            },
        )
    }

    /// Acceptance (#717): the tool embeds the query and hands the vector to the
    /// store together with the model that produced it. Without the pair the
    /// store's vector arm can never run, and a vector paired with the wrong
    /// model name searches rows of another dimension -- which pgvector answers
    /// with an error, not a miss.
    #[tokio::test]
    async fn scratchpad_search_passes_the_query_embedding_and_its_model_to_the_store() {
        let seen = query_recorder();
        let embed: EmbedFn = Arc::new(|_| Box::pin(async { Ok(vec![vec![0.5_f32, 0.25]]) }));

        let (base, _store) = scratchpad_service_with_search(Some(recording_search(&seen)));
        let service = base.with_embedding(embed, "nomic-embed-text@abc".to_string());

        with_conversation_id(ConversationId::from("c1"), async {
            service
                .execute_tool(
                    TOOL_SCRATCHPAD_SEARCH,
                    serde_json::json!({"query": "stay hydrated", "max_results": 10}),
                )
                .await
                .unwrap();
        })
        .await;

        let recorded = seen.lock().unwrap().clone();
        assert_eq!(
            recorded,
            Some((vec![0.5_f32, 0.25], "nomic-embed-text@abc".to_string())),
            "the query vector and its model must both reach the store"
        );
    }

    /// With no embedding backend the tool still searches, handing over an empty
    /// vector -- the store reads that as "take the full-text path".
    #[tokio::test]
    async fn scratchpad_search_without_an_embedding_backend_passes_an_empty_vector() {
        let seen = query_recorder();
        let (service, _store) = scratchpad_service_with_search(Some(recording_search(&seen)));

        with_conversation_id(ConversationId::from("c1"), async {
            service
                .execute_tool(
                    TOOL_SCRATCHPAD_SEARCH,
                    serde_json::json!({"query": "deploy", "max_results": 10}),
                )
                .await
                .unwrap();
        })
        .await;

        let recorded = seen.lock().unwrap().clone();
        assert_eq!(recorded, Some((Vec::new(), String::new())));
    }

    #[tokio::test]
    async fn scratchpad_list_orders_todos_by_sequence() {
        let (service, _store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            // Written out of order; expect list to return them sorted by `seq`.
            service
                .execute_tool(
                    TOOL_SCRATCHPAD_WRITE,
                    serde_json::json!({"notes": [
                        {"key": "c", "content": "third",  "type": "todo", "sequence": 3},
                        {"key": "a", "content": "first",  "type": "todo", "sequence": 1},
                        {"key": "b", "content": "second", "type": "todo", "sequence": 2}
                    ]}),
                )
                .await
                .unwrap();

            let listed = service
                .execute_tool(
                    TOOL_SCRATCHPAD_SEARCH,
                    serde_json::json!({"type": "todo", "max_results": 10}),
                )
                .await
                .unwrap();
            let results = parse(&listed);
            let keys: Vec<String> = results["results"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n["key"].as_str().unwrap().to_string())
                .collect();
            assert_eq!(keys, vec!["a", "b", "c"], "todos sort by sequence");
        })
        .await;
    }

    #[tokio::test]
    async fn conversation_search_without_store_returns_error() {
        let service = BuiltinToolService::new();
        let result = service
            .execute_tool(TOOL_CONV_SEARCH, serde_json::json!({"query": "test"}))
            .await;
        assert!(matches!(result, Err(CoreError::ToolExecution(_))));
    }

    #[tokio::test]
    async fn conversation_search_with_closure_returns_results() {
        use desktop_assistant_core::ports::conversation_search::{
            ConversationSearchFn, MessageHit,
        };
        use std::sync::Arc;

        let search_fn: ConversationSearchFn = Arc::new(move |query, limit, role_filter| {
            let q = query.clone();
            Box::pin(async move {
                assert_eq!(q, "deploy");
                assert_eq!(limit, 5);
                assert!(matches!(role_filter, Some(Role::Assistant)));
                Ok(vec![MessageHit {
                    conversation_id: "c-1".into(),
                    conversation_title: "Deploy timeline".into(),
                    ordinal: 4,
                    role: Role::Assistant,
                    content: "We can deploy on Friday".into(),
                    snippet: "We can <mark>deploy</mark> on Friday".into(),
                    rank: 0.42,
                    updated_at: "2026-05-02T13:00:00+00:00".into(),
                }])
            })
        });

        let service = BuiltinToolService::new().with_conversation_search(search_fn);
        let response = service
            .execute_tool(
                TOOL_CONV_SEARCH,
                serde_json::json!({"query": "deploy", "limit": 5, "role": "assistant"}),
            )
            .await
            .expect("search succeeds");

        let json: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(json["ok"], serde_json::json!(true));
        let results = json["results"].as_array().expect("results array");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["conversation_id"], "c-1");
        assert_eq!(results[0]["ordinal"], 4);
        assert_eq!(results[0]["role"], "assistant");
        assert!(results[0]["snippet"].as_str().unwrap().contains("<mark>"));
    }

    #[tokio::test]
    async fn conversation_search_rejects_unknown_role() {
        // Unknown roles must not reach the search closure: the boundary
        // strips them rather than passing through arbitrary text.
        use desktop_assistant_core::ports::conversation_search::ConversationSearchFn;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let saw_role_filter = Arc::new(AtomicBool::new(false));
        let saw_clone = Arc::clone(&saw_role_filter);
        let search_fn: ConversationSearchFn = Arc::new(move |_q, _l, role_filter| {
            if role_filter.is_some() {
                saw_clone.store(true, Ordering::SeqCst);
            }
            Box::pin(async { Ok(Vec::new()) })
        });

        let service = BuiltinToolService::new().with_conversation_search(search_fn);
        let _ = service
            .execute_tool(
                TOOL_CONV_SEARCH,
                serde_json::json!({"query": "x", "role": "robot"}),
            )
            .await
            .unwrap();
        assert!(
            !saw_role_filter.load(Ordering::SeqCst),
            "unknown role values must not propagate to the search closure"
        );
    }

    #[tokio::test]
    async fn sys_props_returns_compact_property_sheet() {
        let service = BuiltinToolService::new();

        let response = service
            .execute_tool("builtin_sys_props", serde_json::json!({}))
            .await
            .unwrap();

        let json: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(
            json.get("ok").and_then(serde_json::Value::as_bool),
            Some(true)
        );
        let props = json
            .get("props")
            .and_then(serde_json::Value::as_object)
            .expect("props object");
        assert!(
            props
                .get("generated_at_epoch")
                .and_then(serde_json::Value::as_u64)
                .is_some()
        );
        assert!(
            props
                .get("os")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|s| !s.is_empty())
        );
    }

    #[tokio::test]
    async fn sys_props_prefers_client_context_over_daemon_env() {
        // #558: the daemon may be remote/containerized, so its host env is NOT
        // the user's. When the connecting client reported a context, the
        // identity fields must be the CLIENT's, and the source is labeled
        // `client`.
        use desktop_assistant_core::ports::transport::{ClientContext, with_client_context};

        let service = BuiltinToolService::new();
        let ctx = ClientContext {
            real_name: Some("Ada Lovelace".into()),
            username: Some("ada-client".into()),
            home_dir: Some("/home/ada-client".into()),
            hostname: Some("analytical-engine".into()),
            timezone: Some("Europe/London".into()),
            os: Some("TestOS 9000".into()),
        };
        let response = with_client_context(Some(ctx), async {
            service
                .execute_tool("builtin_sys_props", serde_json::json!({}))
                .await
        })
        .await
        .unwrap();

        let json: serde_json::Value = serde_json::from_str(&response).unwrap();
        let props = json
            .get("props")
            .and_then(serde_json::Value::as_object)
            .expect("props object");

        let s = |k: &str| props.get(k).and_then(serde_json::Value::as_str);
        assert_eq!(s("identity_source"), Some("client"));
        assert_eq!(s("real_name"), Some("Ada Lovelace"));
        assert_eq!(s("username"), Some("ada-client"));
        assert_eq!(s("home_dir"), Some("/home/ada-client"));
        assert_eq!(s("hostname"), Some("analytical-engine"));
        assert_eq!(s("timezone"), Some("Europe/London"));
        assert_eq!(s("os"), Some("TestOS 9000"));

        // The daemon host is still reported, but under a clearly-labeled block —
        // never AS the client's identity. Its username is the real daemon env
        // user (or absent), which is never our synthetic client username.
        let daemon = props
            .get("daemon_host")
            .and_then(serde_json::Value::as_object)
            .expect("daemon_host object");
        assert!(daemon.contains_key("cwd"), "daemon working dir is labeled");
        assert_ne!(
            daemon.get("username").and_then(serde_json::Value::as_str),
            Some("ada-client"),
            "daemon-host username must never be the client's value"
        );
    }

    #[tokio::test]
    async fn sys_props_partial_client_context_does_not_borrow_daemon_identity() {
        // #558: a client that reports only its timezone must not have the OTHER
        // identity fields silently filled from the daemon host — that would
        // present daemon-host values AS the client's. Absent client fields stay
        // null under the `client` source.
        use desktop_assistant_core::ports::transport::{ClientContext, with_client_context};

        let service = BuiltinToolService::new();
        let ctx = ClientContext {
            timezone: Some("America/New_York".into()),
            ..ClientContext::default()
        };
        let response = with_client_context(Some(ctx), async {
            service
                .execute_tool("builtin_sys_props", serde_json::json!({}))
                .await
        })
        .await
        .unwrap();

        let json: serde_json::Value = serde_json::from_str(&response).unwrap();
        let props = json
            .get("props")
            .and_then(serde_json::Value::as_object)
            .expect("props object");

        assert_eq!(
            props
                .get("identity_source")
                .and_then(serde_json::Value::as_str),
            Some("client")
        );
        assert_eq!(
            props.get("timezone").and_then(serde_json::Value::as_str),
            Some("America/New_York")
        );
        assert!(
            props
                .get("username")
                .expect("username key present")
                .is_null(),
            "absent client username must stay null, not borrow the daemon's"
        );
        assert!(
            props
                .get("home_dir")
                .expect("home_dir key present")
                .is_null(),
            "absent client home_dir must stay null, not borrow the daemon's"
        );
        assert!(
            props
                .get("hostname")
                .expect("hostname key present")
                .is_null(),
            "absent client hostname must stay null, not borrow the daemon's"
        );
    }

    #[tokio::test]
    async fn sys_props_without_client_context_labels_daemon_host_fallback() {
        // #558: with no client context installed (the common unset case) the
        // identity fields fall back to the daemon host, explicitly labeled so a
        // reader never mistakes them for the connecting client's values.
        let service = BuiltinToolService::new();
        let response = service
            .execute_tool("builtin_sys_props", serde_json::json!({}))
            .await
            .unwrap();

        let json: serde_json::Value = serde_json::from_str(&response).unwrap();
        let props = json
            .get("props")
            .and_then(serde_json::Value::as_object)
            .expect("props object");

        assert_eq!(
            props
                .get("identity_source")
                .and_then(serde_json::Value::as_str),
            Some("daemon_host_fallback")
        );
        // `real_name` has no daemon-host equivalent, so it stays null in fallback.
        assert!(
            props
                .get("real_name")
                .expect("real_name key present")
                .is_null(),
            "daemon host has no real_name to report"
        );
        // The daemon host block is present with the daemon's own os.
        let daemon = props
            .get("daemon_host")
            .and_then(serde_json::Value::as_object)
            .expect("daemon_host object");
        assert!(
            daemon
                .get("os")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|s| !s.is_empty())
        );
    }

    #[tokio::test]
    async fn kb_write_without_store_returns_error() {
        let service = BuiltinToolService::new();
        let result = service
            .execute_tool(TOOL_KB_WRITE, serde_json::json!({"content": "test"}))
            .await;
        assert!(matches!(result, Err(CoreError::ToolExecution(_))));
    }

    #[tokio::test]
    async fn kb_search_without_store_returns_error() {
        let service = BuiltinToolService::new();
        let result = service
            .execute_tool(TOOL_KB_SEARCH, serde_json::json!({"query": "test"}))
            .await;
        assert!(matches!(result, Err(CoreError::ToolExecution(_))));
    }

    #[tokio::test]
    async fn db_query_without_database_returns_error() {
        let service = BuiltinToolService::new();
        let result = service
            .execute_tool(TOOL_DB_QUERY, serde_json::json!({"query": "SELECT 1"}))
            .await;
        assert!(matches!(result, Err(CoreError::ToolExecution(_))));
    }

    #[tokio::test]
    async fn db_query_with_closure() {
        use desktop_assistant_core::ports::database::DbQueryFn;
        use std::sync::Arc;

        let query_fn: DbQueryFn = Arc::new(|_sql, _limit| {
            Box::pin(async {
                Ok(serde_json::json!({
                    "columns": ["count"],
                    "rows": [[42]],
                    "row_count": 1
                }))
            })
        });

        let service = BuiltinToolService::new().with_database(query_fn);

        let result = service
            .execute_tool(
                TOOL_DB_QUERY,
                serde_json::json!({"query": "SELECT count(*) FROM conversations"}),
            )
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(json["result"]["row_count"], 1);
        assert_eq!(json["result"]["rows"][0][0], 42);
    }

    #[tokio::test]
    async fn tool_search_without_registry_returns_error() {
        let service = BuiltinToolService::new();
        let result = service
            .execute_tool(TOOL_SEARCH, serde_json::json!({"query": "file operations"}))
            .await;
        assert!(matches!(result, Err(CoreError::ToolExecution(_))));
    }

    #[tokio::test]
    async fn kb_write_and_search_with_closures() {
        use desktop_assistant_core::domain::KnowledgeEntry;
        use std::sync::{Arc, Mutex};

        let store: Arc<Mutex<Vec<KnowledgeEntry>>> = Arc::new(Mutex::new(Vec::new()));

        let write_store = Arc::clone(&store);
        let write_fn: KnowledgeWriteFn = Arc::new(move |mut entry| {
            let s = Arc::clone(&write_store);
            Box::pin(async move {
                entry.created_at = "2024-01-01".to_string();
                entry.updated_at = "2024-01-01".to_string();
                // Upsert by id, mirroring the store's ON CONFLICT semantics.
                let mut g = s.lock().unwrap();
                g.retain(|e| e.id != entry.id);
                g.push(entry.clone());
                Ok(entry)
            })
        });

        let search_store = Arc::clone(&store);
        let search_fn: KnowledgeSearchFn =
            Arc::new(move |_query, _emb, _model, _tags, _exclude_tags, limit| {
                let s = Arc::clone(&search_store);
                Box::pin(async move {
                    let entries = s.lock().unwrap();
                    Ok(KnowledgeSearchPage {
                        entries: entries.iter().take(limit).cloned().collect(),
                        scope_size: ScopeSize::Few,
                        available_tags: Vec::new(),
                    })
                })
            });

        let delete_store = Arc::clone(&store);
        let delete_fn: KnowledgeDeleteFn = Arc::new(move |ids| {
            let s = Arc::clone(&delete_store);
            Box::pin(async move {
                let mut g = s.lock().unwrap();
                let before = g.len();
                g.retain(|e| !ids.contains(&e.id));
                Ok(before - g.len())
            })
        });

        let list_store = Arc::clone(&store);
        let list_fn: KnowledgeListFn = Arc::new(move |q| {
            let s = Arc::clone(&list_store);
            Box::pin(async move {
                let g = s.lock().unwrap();
                let entries = g.iter().take(q.limit.max(1)).cloned().collect();
                Ok(
                    desktop_assistant_core::ports::knowledge::KnowledgeListPage {
                        entries,
                        next_cursor: None,
                    },
                )
            })
        });

        let get_store = Arc::clone(&store);
        let get_fn: KnowledgeGetFn = Arc::new(move |id| {
            let s = Arc::clone(&get_store);
            Box::pin(async move { Ok(s.lock().unwrap().iter().find(|e| e.id == id).cloned()) })
        });

        let service = BuiltinToolService::new()
            .with_knowledge_base(write_fn, search_fn, delete_fn, list_fn, get_fn);

        // Write
        let write_result = service
            .execute_tool(
                TOOL_KB_WRITE,
                serde_json::json!({
                    "content": "User prefers dark mode",
                    "tags": ["preference"]
                }),
            )
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&write_result).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(json["count"], 1);
        assert!(json["entries"][0]["id"].as_str().is_some());

        // Search
        let search_result = service
            .execute_tool(TOOL_KB_SEARCH, serde_json::json!({"query": "dark mode"}))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&search_result).unwrap();
        assert_eq!(json["ok"], true);
        let results = json["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert!(
            results[0]["content"]
                .as_str()
                .unwrap()
                .contains("dark mode")
        );

        // List surfaces the entry with its provenance ('explicit' for a
        // tool-authored write) and an id we can operate on.
        let list_result = service
            .execute_tool(TOOL_KB_LIST, serde_json::json!({"limit": 10}))
            .await
            .unwrap();
        let lj: serde_json::Value = serde_json::from_str(&list_result).unwrap();
        assert_eq!(lj["count"], 1);
        assert_eq!(lj["entries"][0]["source"], "explicit");
        let id = lj["entries"][0]["id"].as_str().unwrap().to_string();

        // Partial update: tags only, `content` omitted — existing content is
        // preserved.
        service
            .execute_tool(
                TOOL_KB_WRITE,
                serde_json::json!({"id": id, "tags": ["preference", "retagged"]}),
            )
            .await
            .unwrap();
        {
            let g = store.lock().unwrap();
            assert_eq!(g.len(), 1);
            assert_eq!(g[0].content, "User prefers dark mode");
            assert!(g[0].tags.iter().any(|t| t == "retagged"));
        }

        // Bulk delete by ids.
        let del = service
            .execute_tool(TOOL_KB_DELETE, serde_json::json!({"ids": [id]}))
            .await
            .unwrap();
        let dj: serde_json::Value = serde_json::from_str(&del).unwrap();
        assert_eq!(dj["deleted"], 1);
        assert!(store.lock().unwrap().is_empty());
    }

    // -- knowledge-base search: scope reporting (#1068) ----------------------

    /// What the store saw on the last search, captured so a test can pin the
    /// arguments the tool passed down as well as the response it built.
    #[derive(Default)]
    struct SearchProbe {
        tags: Option<Vec<String>>,
        exclude_tags: Option<Vec<String>>,
        limit: usize,
        /// True when the tool passed no query embedding, which is what makes
        /// the store take its full-text-only path.
        embedding_was_empty: bool,
    }

    /// A knowledge-base service whose store answers every search with `page`,
    /// and records what the tool asked for in the returned probe.
    ///
    /// The store is the component that computes the scope. The tool's own
    /// contract is that it passes the filters down unchanged and reports what
    /// the store said, without reordering, re-counting, or dropping it.
    fn kb_service_reporting(
        page: KnowledgeSearchPage,
    ) -> (
        BuiltinToolService,
        std::sync::Arc<std::sync::Mutex<SearchProbe>>,
    ) {
        use std::sync::{Arc, Mutex};

        let probe = Arc::new(Mutex::new(SearchProbe::default()));
        let probe_for_fn = Arc::clone(&probe);
        let search_fn: KnowledgeSearchFn = Arc::new(
            move |_query, emb: Vec<f32>, _model, tags, exclude_tags, limit| {
                let page = page.clone();
                let probe = Arc::clone(&probe_for_fn);
                Box::pin(async move {
                    *probe.lock().unwrap() = SearchProbe {
                        tags,
                        exclude_tags,
                        limit,
                        embedding_was_empty: emb.is_empty(),
                    };
                    Ok(page)
                })
            },
        );
        let write_fn: KnowledgeWriteFn = Arc::new(|entry| Box::pin(async move { Ok(entry) }));
        let delete_fn: KnowledgeDeleteFn = Arc::new(|_ids| Box::pin(async { Ok(0) }));
        let list_fn: KnowledgeListFn = Arc::new(|_q| {
            Box::pin(async {
                Ok(
                    desktop_assistant_core::ports::knowledge::KnowledgeListPage {
                        entries: Vec::new(),
                        next_cursor: None,
                    },
                )
            })
        });
        let get_fn: KnowledgeGetFn = Arc::new(|_id| Box::pin(async { Ok(None) }));
        let service = BuiltinToolService::new()
            .with_knowledge_base(write_fn, search_fn, delete_fn, list_fn, get_fn);
        (service, probe)
    }

    /// Run `builtin_knowledge_base_search` and parse its response.
    async fn kb_search_response(
        service: &BuiltinToolService,
        arguments: serde_json::Value,
    ) -> serde_json::Value {
        let raw = service
            .execute_tool(TOOL_KB_SEARCH, arguments)
            .await
            .expect("knowledge base search succeeds");
        serde_json::from_str(&raw).expect("search response is JSON")
    }

    fn kb_entry(id: &str, tags: &[&str]) -> desktop_assistant_core::domain::KnowledgeEntry {
        desktop_assistant_core::domain::KnowledgeEntry::new(
            id,
            "content",
            tags.iter().map(|t| (*t).to_string()).collect(),
        )
    }

    #[tokio::test]
    async fn kb_search_reports_scope_size_none_for_an_empty_scope() {
        // An empty page alone cannot tell the model whether its query matched
        // nothing or its tag filter selected nothing. NONE says the scope
        // itself is empty, so a different query cannot help - only dropping
        // the filters can, because the store as a whole may hold plenty.
        let (service, _probe) = kb_service_reporting(KnowledgeSearchPage {
            entries: Vec::new(),
            scope_size: ScopeSize::None,
            available_tags: Vec::new(),
        });

        let json = kb_search_response(&service, serde_json::json!({"query": "anything"})).await;

        assert_eq!(json["ok"], true);
        assert_eq!(json["scope_size"], "NONE");
        assert_eq!(json["returned"], 0);
        assert_eq!(json["available_tags"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn kb_search_reports_scope_size_few_when_every_entry_fits_the_page() {
        // FEW means the scope is no larger than the page, so a narrower tag
        // filter can only remove entries, never reveal one. It does not mean
        // the model has seen everything: the page holds what matched the
        // query, and an unmatched entry can still be sitting in that scope.
        let (service, _probe) = kb_service_reporting(KnowledgeSearchPage {
            entries: vec![kb_entry("kb-1", &["preference"])],
            scope_size: ScopeSize::Few,
            available_tags: vec!["preference".to_string()],
        });

        let json = kb_search_response(&service, serde_json::json!({"query": "dark mode"})).await;

        assert_eq!(json["scope_size"], "FEW");
        assert_eq!(json["returned"], 1);
    }

    #[tokio::test]
    async fn kb_search_reports_scope_size_many_when_the_scope_exceeds_the_page() {
        // MANY is the only value that tells the model a narrower filter can
        // still pay off, so it must survive the trip to the wire. The page also
        // filled up here, which is a separate claim: `truncated`.
        let (service, probe) = kb_service_reporting(KnowledgeSearchPage {
            entries: vec![kb_entry("kb-1", &["preference"]), kb_entry("kb-2", &[])],
            scope_size: ScopeSize::Many,
            available_tags: vec!["preference".to_string()],
        });

        let json =
            kb_search_response(&service, serde_json::json!({"query": "notes", "limit": 2})).await;

        assert_eq!(
            probe.lock().unwrap().limit,
            2,
            "premise: the caller's page size reached the store"
        );
        assert_eq!(json["scope_size"], "MANY");
        assert_eq!(json["returned"], 2);
        assert_eq!(json["truncated"], true);
        assert!(
            json["message"].as_str().is_some_and(|m| !m.is_empty()),
            "a truncated page must say how to get the rest"
        );
    }

    #[tokio::test]
    async fn kb_search_available_tags_are_ordered_by_frequency_then_name() {
        // The order carries the whole signal, because no counts travel with the
        // tags. The tool must report the store's order unchanged - re-sorting
        // it (alphabetically, say) would destroy the signal silently.
        let store_order = vec![
            "project:adelie-ai".to_string(),
            "preference".to_string(),
            "topic:weather".to_string(),
        ];
        let (service, _probe) = kb_service_reporting(KnowledgeSearchPage {
            entries: Vec::new(),
            scope_size: ScopeSize::Many,
            available_tags: store_order.clone(),
        });

        let json = kb_search_response(&service, serde_json::json!({"query": "anything"})).await;

        let reported: Vec<String> = serde_json::from_value(json["available_tags"].clone())
            .expect("available_tags is an array of strings");
        assert_eq!(reported, store_order);
    }

    #[tokio::test]
    async fn kb_search_available_tags_honour_include_and_exclude_filters() {
        // The filters define the scope, so the census must see the same ones
        // the search did. A tool that dropped `exclude_tags` on the way down
        // would report a tag vocabulary for a scope nobody searched.
        let (service, probe) = kb_service_reporting(KnowledgeSearchPage {
            entries: Vec::new(),
            scope_size: ScopeSize::Few,
            available_tags: vec!["project:adelie-ai".to_string()],
        });

        let json = kb_search_response(
            &service,
            serde_json::json!({
                "query": "deploy",
                "tags": ["project:adelie-ai"],
                "exclude_tags": ["archived"],
            }),
        )
        .await;

        let seen = probe.lock().unwrap();
        assert_eq!(
            seen.tags.as_deref(),
            Some(["project:adelie-ai".to_string()].as_slice())
        );
        assert_eq!(
            seen.exclude_tags.as_deref(),
            Some(["archived".to_string()].as_slice())
        );
        assert_eq!(
            json["available_tags"],
            serde_json::json!(["project:adelie-ai"])
        );
    }

    #[tokio::test]
    async fn kb_search_available_tags_are_capped_at_fifty() {
        // The list travels to the model inside a tool result, so the cap is a
        // context budget, not a storage detail. The tool enforces it on what it
        // reports, whatever the store hands it.
        let over_cap: Vec<String> = (0..60).map(|i| format!("topic:t{i:02}")).collect();
        let (service, _probe) = kb_service_reporting(KnowledgeSearchPage {
            entries: Vec::new(),
            scope_size: ScopeSize::Many,
            available_tags: over_cap.clone(),
        });

        let json = kb_search_response(&service, serde_json::json!({"query": "anything"})).await;

        let reported: Vec<String> = serde_json::from_value(json["available_tags"].clone())
            .expect("available_tags is an array of strings");
        assert_eq!(reported.len(), 50);
        assert_eq!(reported, over_cap[..50]);
    }

    #[tokio::test]
    async fn kb_search_omits_truncated_when_the_results_fit() {
        // `truncated` is a claim that entries were left behind. Sending it on
        // every response would train the model to ignore it.
        //
        // The scope is `Many` on purpose. Under `Few` the suppression arm hides
        // whether the page-not-full half of the rule works at all, so this
        // would pass against a rule that ignored `limit` entirely.
        let (service, _probe) = kb_service_reporting(KnowledgeSearchPage {
            entries: vec![kb_entry("kb-1", &["preference"])],
            scope_size: ScopeSize::Many,
            available_tags: vec!["preference".to_string()],
        });

        let json =
            kb_search_response(&service, serde_json::json!({"query": "notes", "limit": 5})).await;

        assert_eq!(json["returned"], 1);
        assert!(
            json.get("truncated").is_none(),
            "truncated must be absent when the page did not fill up"
        );
        assert!(
            json.get("message").is_none(),
            "the truncation message must not travel without `truncated`"
        );
    }

    #[tokio::test]
    async fn kb_search_reports_unknown_scope_when_the_census_fails() {
        // The census runs after the search has already returned its entries, so
        // a census that fails must cost the measurement and nothing else. The
        // store therefore hands back UNKNOWN rather than raising, and the tool
        // must pass that through: reporting NONE here would tell the model the
        // store is empty when it is not.
        let (service, _probe) = kb_service_reporting(KnowledgeSearchPage {
            entries: vec![kb_entry("kb-1", &["preference"])],
            scope_size: ScopeSize::Unknown,
            available_tags: Vec::new(),
        });

        let json = kb_search_response(&service, serde_json::json!({"query": "dark mode"})).await;

        assert_eq!(json["ok"], true);
        assert_eq!(json["scope_size"], "UNKNOWN");
        assert_eq!(json["returned"], 1);
        assert_eq!(json["results"][0]["id"], "kb-1");
        assert_eq!(json["available_tags"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn kb_search_still_reports_truncated_under_unknown_scope() {
        // FEW suppresses `truncated` because it proves the page holds the whole
        // scope. UNKNOWN proves nothing, so a full page under it is still
        // evidence that entries were left behind.
        let (service, _probe) = kb_service_reporting(KnowledgeSearchPage {
            entries: vec![kb_entry("kb-1", &["preference"]), kb_entry("kb-2", &[])],
            scope_size: ScopeSize::Unknown,
            available_tags: Vec::new(),
        });

        let json =
            kb_search_response(&service, serde_json::json!({"query": "notes", "limit": 2})).await;

        assert_eq!(json["scope_size"], "UNKNOWN");
        assert_eq!(json["truncated"], true);
        assert!(
            json["message"].as_str().is_some_and(|m| !m.is_empty()),
            "a truncated page must say how to get the rest"
        );
    }

    #[tokio::test]
    async fn kb_search_omits_truncated_when_the_scope_is_few() {
        // A full page is normally evidence that entries were left behind. Under
        // FEW it is not: the scope is no larger than the page, and what matched
        // is a subset of the scope, so the page holds everything there was.
        // Claiming truncation here sends the model narrowing a search that
        // already returned the whole scope.
        let (service, _probe) = kb_service_reporting(KnowledgeSearchPage {
            entries: vec![kb_entry("kb-1", &["preference"]), kb_entry("kb-2", &[])],
            scope_size: ScopeSize::Few,
            available_tags: vec!["preference".to_string()],
        });

        let json =
            kb_search_response(&service, serde_json::json!({"query": "notes", "limit": 2})).await;

        assert_eq!(json["returned"], 2, "premise: the page filled up");
        assert!(
            json.get("truncated").is_none(),
            "FEW proves the page holds the whole scope, so nothing was left behind"
        );
        assert!(
            json.get("message").is_none(),
            "the truncation message must not travel without `truncated`"
        );
    }

    #[tokio::test]
    async fn kb_search_clamps_a_zero_limit() {
        // `results.len() >= limit` holds vacuously at a limit of zero, so an
        // unclamped zero would return nothing and then claim the page filled
        // up. The schema's minimum is honoured here, not merely advertised.
        let (service, probe) = kb_service_reporting(KnowledgeSearchPage {
            entries: Vec::new(),
            scope_size: ScopeSize::None,
            available_tags: Vec::new(),
        });

        let json =
            kb_search_response(&service, serde_json::json!({"query": "x", "limit": 0})).await;

        assert_eq!(
            probe.lock().unwrap().limit,
            1,
            "a zero limit is clamped to the schema's minimum before it reaches the store"
        );
        assert_eq!(json["ok"], true);
        assert_eq!(json["returned"], 0);
        assert!(
            json.get("truncated").is_none(),
            "an empty page must never claim it was truncated"
        );
    }

    #[tokio::test]
    async fn kb_search_clamps_a_limit_above_the_maximum() {
        // The store multiplies the limit by two to seed the RRF fetch, so an
        // unbounded limit overflows there. The clamp must land before the call,
        // which is what the probe reads - the response cannot show it.
        let (service, probe) = kb_service_reporting(KnowledgeSearchPage {
            entries: Vec::new(),
            scope_size: ScopeSize::None,
            available_tags: Vec::new(),
        });

        let _ = kb_search_response(
            &service,
            serde_json::json!({"query": "x", "limit": u64::MAX}),
        )
        .await;

        assert_eq!(
            probe.lock().unwrap().limit,
            KB_SEARCH_MAX_LIMIT as usize,
            "a limit past the cap is clamped to the cap before it reaches the store"
        );
    }

    #[tokio::test]
    async fn kb_search_reports_scope_and_tags_on_the_text_only_fallback_path() {
        // With no embedding backend wired the query embedding is empty and the
        // store falls back to full-text search. That is a recall degradation,
        // not a contract change: the response keeps every field.
        let (service, probe) = kb_service_reporting(KnowledgeSearchPage {
            entries: vec![kb_entry("kb-1", &["preference"])],
            scope_size: ScopeSize::Many,
            available_tags: vec!["preference".to_string(), "topic:weather".to_string()],
        });

        let json = kb_search_response(&service, serde_json::json!({"query": "weather"})).await;

        assert!(
            probe.lock().unwrap().embedding_was_empty,
            "premise: with no embedding backend the store gets an empty embedding, \
             which is what makes it take the full-text-only path"
        );
        assert_eq!(json["scope_size"], "MANY");
        assert_eq!(
            json["available_tags"],
            serde_json::json!(["preference", "topic:weather"])
        );
        assert_eq!(json["returned"], 1);
    }

    // --- The tag-registry gate on tool-path writes (#1070) -----------------

    /// What one call of the tag-registry gate saw and answered with.
    #[derive(Debug, Clone)]
    struct TagProbe {
        /// Every tag the write path proposed, in the order it proposed them.
        proposals: Vec<ProposedTag>,
    }

    /// The in-memory knowledge store the write-path tests share: the rows a
    /// write landed, in the order it landed them.
    type KbStore =
        std::sync::Arc<std::sync::Mutex<Vec<desktop_assistant_core::domain::KnowledgeEntry>>>;

    /// A knowledge-base service whose store keeps its entries in memory, with
    /// an optional tag-registry gate in front of the write path.
    ///
    /// The probe records the proposals the gate received, so a test can tell
    /// "the gate was consulted and answered" from "the tag was stored as the
    /// model wrote it".
    fn kb_service_with_tag_gate(
        resolve: Option<KnowledgeTagResolveFn>,
    ) -> (BuiltinToolService, KbStore) {
        kb_service_with_slow_store(resolve, std::time::Duration::ZERO)
    }

    /// The same service, with each store write taking `store_delay`.
    ///
    /// Storing an entry is not a consultation of the tag vocabulary, so a slow
    /// store must not spend the vocabulary's share of the write. A test drives
    /// that with a paused clock rather than a real wait.
    fn kb_service_with_slow_store(
        resolve: Option<KnowledgeTagResolveFn>,
        store_delay: std::time::Duration,
    ) -> (BuiltinToolService, KbStore) {
        use desktop_assistant_core::domain::KnowledgeEntry;
        use std::sync::{Arc, Mutex};

        let store: Arc<Mutex<Vec<KnowledgeEntry>>> = Arc::new(Mutex::new(Vec::new()));

        let write_store = Arc::clone(&store);
        let write_fn: KnowledgeWriteFn = Arc::new(move |mut entry| {
            let s = Arc::clone(&write_store);
            Box::pin(async move {
                if !store_delay.is_zero() {
                    tokio::time::sleep(store_delay).await;
                }
                entry.created_at = "2026-01-01".to_string();
                entry.updated_at = "2026-01-01".to_string();
                let mut g = s.lock().expect("write store lock");
                g.retain(|e| e.id != entry.id);
                g.push(entry.clone());
                Ok(entry)
            })
        });
        let search_fn: KnowledgeSearchFn = Arc::new(|_q, _emb, _model, _tags, _exclude, _limit| {
            Box::pin(async {
                Ok(KnowledgeSearchPage {
                    entries: Vec::new(),
                    scope_size: ScopeSize::None,
                    available_tags: Vec::new(),
                })
            })
        });
        let delete_fn: KnowledgeDeleteFn = Arc::new(|_ids| Box::pin(async { Ok(0) }));
        let list_fn: KnowledgeListFn = Arc::new(|_q| {
            Box::pin(async {
                Ok(
                    desktop_assistant_core::ports::knowledge::KnowledgeListPage {
                        entries: Vec::new(),
                        next_cursor: None,
                    },
                )
            })
        });
        let get_store = Arc::clone(&store);
        let get_fn: KnowledgeGetFn = Arc::new(move |id| {
            let s = Arc::clone(&get_store);
            Box::pin(async move {
                Ok(s.lock()
                    .expect("get store lock")
                    .iter()
                    .find(|e| e.id == id)
                    .cloned())
            })
        });

        let mut service = BuiltinToolService::new()
            .with_knowledge_base(write_fn, search_fn, delete_fn, list_fn, get_fn);
        if let Some(resolve) = resolve {
            service = service.with_tag_registry(resolve);
        }
        (service, store)
    }

    /// A tag-registry gate that answers from a fixed proposed-name -> stored-name
    /// table, and records every proposal it saw. A name absent from the table is
    /// returned unchanged, which is what the registry does for a genuinely new
    /// tag it just created.
    fn recording_tag_gate(
        redirects: &[(&str, &str)],
    ) -> (
        KnowledgeTagResolveFn,
        std::sync::Arc<std::sync::Mutex<TagProbe>>,
    ) {
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex};

        let table: HashMap<String, String> = redirects
            .iter()
            .map(|(from, to)| ((*from).to_string(), (*to).to_string()))
            .collect();
        let probe = Arc::new(Mutex::new(TagProbe {
            proposals: Vec::new(),
        }));
        let probe_for_fn = Arc::clone(&probe);
        let resolve: KnowledgeTagResolveFn = Arc::new(move |proposed: ProposedTag| {
            let table = table.clone();
            let probe = Arc::clone(&probe_for_fn);
            Box::pin(async move {
                let resolved = table
                    .get(&proposed.name)
                    .cloned()
                    .unwrap_or_else(|| proposed.name.clone());
                probe.lock().expect("probe lock").proposals.push(proposed);
                Ok(resolved)
            })
        });
        (resolve, probe)
    }

    /// Run `builtin_knowledge_base_write` and parse its response.
    async fn kb_write_response(
        service: &BuiltinToolService,
        arguments: serde_json::Value,
    ) -> serde_json::Value {
        let raw = service
            .execute_tool(TOOL_KB_WRITE, arguments)
            .await
            .expect("knowledge base write succeeds");
        serde_json::from_str(&raw).expect("write response is JSON")
    }

    /// The fake store's rows, for a test that asserts on what a write stored.
    fn kb_stored(store: &KbStore) -> Vec<desktop_assistant_core::domain::KnowledgeEntry> {
        store.lock().expect("store lock").clone()
    }

    /// Put a row straight into the fake store, around the write path.
    ///
    /// The write path is what these tests measure, so the row a test starts
    /// from has to arrive another way. `metadata` has no tool argument at all,
    /// so seeding is the only way to give a row any.
    fn seed_kb_entry(
        store: &KbStore,
        id: &str,
        content: &str,
        tags: &[&str],
        metadata: serde_json::Value,
    ) {
        use desktop_assistant_core::domain::KnowledgeEntry;
        store.lock().expect("seed store lock").push(KnowledgeEntry {
            id: id.to_string(),
            content: content.to_string(),
            tags: tags.iter().map(|t| (*t).to_string()).collect(),
            metadata,
            created_at: "2026-01-01".to_string(),
            updated_at: "2026-01-01".to_string(),
            source: Some("explicit".to_string()),
            summary: None,
        });
    }

    #[tokio::test]
    async fn kb_write_content_update_by_id_preserves_tags() {
        // The prompt tells the model to update an existing entry instead of
        // creating a near duplicate, and a content update names only `id` and
        // `content`. Reads filter by tag overlap, so an entry that loses its
        // tags is still in the store and no tag-scoped search finds it.
        let (service, store) = kb_service_with_tag_gate(None);
        seed_kb_entry(
            &store,
            "entry-1",
            "Rain is expected on Tuesday.",
            &["memory", "topic:weather"],
            serde_json::json!({}),
        );

        kb_write_response(
            &service,
            serde_json::json!({
                "id": "entry-1",
                "content": "Rain is expected on Wednesday.",
            }),
        )
        .await;

        let stored = kb_stored(&store);
        assert_eq!(stored.len(), 1, "the update replaced the entry in place");
        assert_eq!(stored[0].content, "Rain is expected on Wednesday.");
        assert_eq!(
            stored[0].tags,
            vec!["memory".to_string(), "topic:weather".to_string()],
            "a write that does not mention tags keeps the ones the entry carries"
        );
    }

    #[tokio::test]
    async fn kb_write_content_update_by_id_preserves_metadata() {
        // `metadata` has no tool argument, so a caller cannot send it back.
        // Whatever a content update drops from it is gone for good.
        let (service, store) = kb_service_with_tag_gate(None);
        seed_kb_entry(
            &store,
            "entry-1",
            "Rain is expected on Tuesday.",
            &["memory"],
            serde_json::json!({"confidence": "high", "source_url": "https://example.com/forecast"}),
        );

        kb_write_response(
            &service,
            serde_json::json!({
                "id": "entry-1",
                "content": "Rain is expected on Wednesday.",
            }),
        )
        .await;

        let stored = kb_stored(&store);
        assert_eq!(
            stored[0].metadata,
            serde_json::json!({"confidence": "high", "source_url": "https://example.com/forecast"}),
            "a write that does not mention a metadata key keeps it"
        );
    }

    #[tokio::test]
    async fn kb_write_content_update_by_id_keeps_the_conversation_provenance() {
        // The #240 stamp names the conversation an entry was learned in. A
        // content update happens in a later conversation, so losing the stamp
        // does not leave it blank - it replaces it with the wrong
        // conversation, which reads as true.
        use desktop_assistant_core::domain::ConversationId;
        use desktop_assistant_core::ports::conversation_ctx::with_conversation_id;

        let (service, store) = kb_service_with_tag_gate(None);
        seed_kb_entry(
            &store,
            "entry-1",
            "Rain is expected on Tuesday.",
            &["memory"],
            serde_json::json!({"source_conversation_id": "conversation-where-it-was-learned"}),
        );

        with_conversation_id(
            ConversationId::from("a-later-conversation"),
            kb_write_response(
                &service,
                serde_json::json!({
                    "id": "entry-1",
                    "content": "Rain is expected on Wednesday.",
                }),
            ),
        )
        .await;

        let stored = kb_stored(&store);
        assert_eq!(
            stored[0].metadata["source_conversation_id"], "conversation-where-it-was-learned",
            "the stamp names where the fact was learned, not where it was last edited"
        );
    }

    #[tokio::test]
    async fn kb_write_explicit_empty_tags_still_clears_them() {
        // Absent means "leave the stored tags alone"; present-and-empty means
        // "clear them". Preserving the stored tags whenever the supplied list
        // came out empty would take the second away.
        let (service, store) = kb_service_with_tag_gate(None);
        seed_kb_entry(
            &store,
            "entry-1",
            "Rain is expected on Tuesday.",
            &["memory", "topic:weather"],
            serde_json::json!({}),
        );
        seed_kb_entry(
            &store,
            "entry-2",
            "Snow is expected on Friday.",
            &["memory", "topic:weather"],
            serde_json::json!({}),
        );

        kb_write_response(
            &service,
            serde_json::json!({
                "id": "entry-1",
                "content": "Rain is expected on Wednesday.",
                "tags": [],
            }),
        )
        .await;
        // The same rule with no content: a tags-only write that sends an empty
        // list clears the tags and keeps the content.
        kb_write_response(&service, serde_json::json!({"id": "entry-2", "tags": []})).await;

        let stored = kb_stored(&store);
        let first = stored
            .iter()
            .find(|e| e.id == "entry-1")
            .expect("entry-1 is in the store");
        assert!(
            first.tags.is_empty(),
            "an empty tag list on a content update clears the tags: {:?}",
            first.tags
        );
        let second = stored
            .iter()
            .find(|e| e.id == "entry-2")
            .expect("entry-2 is in the store");
        assert!(
            second.tags.is_empty(),
            "an empty tag list on a tags-only write clears the tags: {:?}",
            second.tags
        );
        assert_eq!(
            second.content, "Snow is expected on Friday.",
            "clearing the tags does not touch the content"
        );
    }

    #[tokio::test]
    async fn kb_write_reads_a_null_tags_field_as_absent() {
        // Several providers encode "I am not setting this field" as an
        // explicit null. Reading that as an empty list clears the tags of
        // every entry those writes touch, which is the loss this whole path
        // exists to stop.
        let (service, store) = kb_service_with_tag_gate(None);
        seed_kb_entry(
            &store,
            "entry-1",
            "Rain is expected on Tuesday.",
            &["memory", "topic:weather"],
            serde_json::json!({}),
        );

        kb_write_response(
            &service,
            serde_json::json!({
                "id": "entry-1",
                "content": "Rain is expected on Wednesday.",
                "tags": null,
            }),
        )
        .await;

        assert_eq!(
            kb_stored(&store)[0].tags,
            vec!["memory".to_string(), "topic:weather".to_string()],
            "a null tag field means the write said nothing about tags"
        );
    }

    #[tokio::test]
    async fn kb_write_refuses_a_tags_field_that_is_not_a_list() {
        // Nothing validates the tool schema between the model and this code,
        // so a `tags` of the wrong shape arrives as written. Reading it as an
        // empty list answers "set this tag" with a wipe, and reports `ok`.
        let (service, store) = kb_service_with_tag_gate(None);
        seed_kb_entry(
            &store,
            "entry-1",
            "Rain is expected on Tuesday.",
            &["memory", "topic:weather"],
            serde_json::json!({}),
        );

        let err = service
            .execute_tool(
                TOOL_KB_WRITE,
                serde_json::json!({
                    "id": "entry-1",
                    "content": "Rain is expected on Wednesday.",
                    "tags": "topic:weather",
                }),
            )
            .await
            .expect_err("a `tags` that is not a list is refused");
        assert!(
            err.to_string().contains("`tags` must be a list of strings"),
            "the error says what shape `tags` takes: {err}"
        );
        assert_eq!(
            kb_stored(&store)[0].tags,
            vec!["memory".to_string(), "topic:weather".to_string()],
            "the refused write stored nothing, so the entry is untouched"
        );
    }

    #[tokio::test]
    async fn kb_write_refuses_a_tag_that_is_not_a_string() {
        // A tag that arrives as an object is a tag the caller asked to store.
        // Dropping it silently loses the caller's intent, and dropping the
        // only one wipes the entry.
        let (service, store) = kb_service_with_tag_gate(None);
        seed_kb_entry(
            &store,
            "entry-1",
            "Rain is expected on Tuesday.",
            &["memory", "topic:weather"],
            serde_json::json!({}),
        );

        let err = service
            .execute_tool(
                TOOL_KB_WRITE,
                serde_json::json!({
                    "id": "entry-1",
                    "content": "Rain is expected on Wednesday.",
                    "tags": [{"name": "memory"}],
                }),
            )
            .await
            .expect_err("a tag that is not a string is refused");
        assert!(
            err.to_string()
                .contains("every tag in `tags` must be a string"),
            "the error says what a tag is: {err}"
        );
        assert_eq!(
            kb_stored(&store)[0].tags,
            vec!["memory".to_string(), "topic:weather".to_string()],
            "the refused write stored nothing, so the entry is untouched"
        );
    }

    #[tokio::test]
    async fn kb_write_still_drops_a_blank_tag() {
        // A blank string is not a tag, and trimming it away loses no intent,
        // so it stays a normalisation rather than joining the refusals above.
        let (service, store) = kb_service_with_tag_gate(None);

        kb_write_response(
            &service,
            serde_json::json!({
                "content": "Rain is expected on Tuesday.",
                "tags": ["memory", "   ", ""],
            }),
        )
        .await;

        assert_eq!(kb_stored(&store)[0].tags, vec!["memory".to_string()]);
    }

    #[tokio::test]
    async fn kb_write_without_an_id_is_unaffected() {
        // A create starts from nothing. No row already in the store may reach
        // it, whatever the store holds.
        let (service, store) = kb_service_with_tag_gate(None);
        seed_kb_entry(
            &store,
            "entry-1",
            "Rain is expected on Tuesday.",
            &["memory", "topic:weather"],
            serde_json::json!({"confidence": "high"}),
        );

        let created =
            kb_write_response(&service, serde_json::json!({"content": "A separate fact."})).await;
        let new_id = created["entries"][0]["id"]
            .as_str()
            .expect("the write reports the entry id")
            .to_string();
        assert_ne!(new_id, "entry-1", "a write with no id creates a new entry");

        let stored = kb_stored(&store);
        assert_eq!(stored.len(), 2, "the create left the existing entry alone");
        let fresh = stored
            .iter()
            .find(|e| e.id == new_id)
            .expect("the create landed");
        assert!(
            fresh.tags.is_empty(),
            "a create that names no tags carries none: {:?}",
            fresh.tags
        );
        assert_eq!(
            fresh.metadata,
            serde_json::json!({}),
            "a create starts with empty metadata"
        );
    }

    #[tokio::test]
    async fn kb_write_with_content_creates_at_a_caller_supplied_id() {
        // `id` is optional, and supplying one is how a caller makes a write
        // idempotent under retry: the retry lands on the row the first attempt
        // created instead of a duplicate. A write that carries content holds
        // everything a create needs, so an id no row holds is a create at that
        // id, not an error.
        let (service, store) = kb_service_with_tag_gate(None);

        let spec = serde_json::json!({
            "id": "caller-chosen-id",
            "content": "A fact worth keeping.",
            "tags": ["memory"],
        });
        let first = kb_write_response(&service, spec.clone()).await;
        assert_eq!(
            first["entries"][0]["id"], "caller-chosen-id",
            "the create used the id the caller chose"
        );

        // The same call again, as a retry would send it.
        kb_write_response(&service, spec).await;

        let stored = kb_stored(&store);
        assert_eq!(
            stored.len(),
            1,
            "the retry landed on the row the first write created"
        );
        assert_eq!(stored[0].tags, vec!["memory".to_string()]);
        assert_eq!(stored[0].content, "A fact worth keeping.");
    }

    #[tokio::test]
    async fn kb_write_by_id_without_content_needs_an_entry_to_update() {
        // With no content and no stored row there is nothing to write, so this
        // stays an error even though a create at a caller-chosen id does not.
        let (service, _store) = kb_service_with_tag_gate(None);

        let err = service
            .execute_tool(
                TOOL_KB_WRITE,
                serde_json::json!({"id": "no-such-entry", "tags": ["memory"]}),
            )
            .await
            .expect_err("a tags-only write needs an entry to re-tag");
        assert!(
            err.to_string().contains("no knowledge entry with id"),
            "the error names the missing entry: {err}"
        );
    }

    #[tokio::test]
    async fn kb_write_content_update_by_id_does_not_re_register_the_carried_over_tags() {
        // A tag the entry already carries is already in the vocabulary.
        // Offering it again on every content update pays an embedding per tag
        // per write, and lets the registry rename a stored tag behind the
        // caller's back on a write that never mentioned it.
        let (resolve, probe) = recording_tag_gate(&[]);
        let (service, store) = kb_service_with_tag_gate(Some(resolve));
        seed_kb_entry(
            &store,
            "entry-1",
            "Rain is expected on Tuesday.",
            &["memory", "topic:weather"],
            serde_json::json!({}),
        );

        kb_write_response(
            &service,
            serde_json::json!({
                "id": "entry-1",
                "content": "Rain is expected on Wednesday.",
            }),
        )
        .await;

        assert!(
            probe.lock().expect("probe lock").proposals.is_empty(),
            "a write that does not mention tags consults the vocabulary for none"
        );
        assert_eq!(
            kb_stored(&store)[0].tags,
            vec!["memory".to_string(), "topic:weather".to_string()],
        );
    }

    #[tokio::test]
    async fn kb_write_redirects_a_near_duplicate_tag_to_the_existing_tag() {
        // The vocabulary fragments when `topic:forecast` lands beside
        // `topic:weather`: reads filter by exact array overlap, so the two
        // never match each other. The registry already knows they are the same
        // concept, so the entry must carry the tag the registry chose.
        let (resolve, probe) = recording_tag_gate(&[("topic:forecast", "topic:weather")]);
        let (service, store) = kb_service_with_tag_gate(Some(resolve));

        let json = kb_write_response(
            &service,
            serde_json::json!({
                "content": "Rain is expected on Tuesday.",
                "tags": ["memory", "topic:forecast"],
            }),
        )
        .await;

        assert_eq!(json["ok"], true);
        let stored = store.lock().expect("store lock");
        assert_eq!(stored.len(), 1, "one entry was written");
        assert_eq!(
            stored[0].tags,
            vec!["memory".to_string(), "topic:weather".to_string()],
            "the near-duplicate must be stored under the existing tag"
        );
        assert_eq!(
            probe
                .lock()
                .expect("probe lock")
                .proposals
                .iter()
                .map(|p| p.name.clone())
                .collect::<Vec<_>>(),
            vec!["memory".to_string(), "topic:forecast".to_string()],
            "every tag on the write goes through the registry, not just the new one"
        );
        assert_eq!(
            json["entries"][0]["tags"],
            serde_json::json!(["memory", "topic:weather"]),
            "the response reports the tags actually stored, so the model does not \
             believe the entry carries a tag it does not"
        );
    }

    #[tokio::test]
    async fn kb_write_stores_a_genuinely_new_tag_and_registers_it_with_its_description() {
        // A short facet tag carries almost no signal on its own, so the dedup
        // needs the model's one-line description of what the tag means. It
        // arrives in `new_tag_descriptions`, keyed by tag name.
        let (resolve, probe) = recording_tag_gate(&[]);
        let (service, store) = kb_service_with_tag_gate(Some(resolve));

        kb_write_response(
            &service,
            serde_json::json!({
                "content": "Embeddings are backfilled by the maintenance task.",
                "tags": ["topic:embeddings"],
                "new_tag_descriptions": {
                    "topic:embeddings": "Vector embedding generation, models, and backfill",
                },
            }),
        )
        .await;

        let proposals = probe.lock().expect("probe lock").proposals.clone();
        assert_eq!(proposals.len(), 1, "one tag was proposed");
        assert_eq!(proposals[0].name, "topic:embeddings");
        assert_eq!(
            proposals[0].description.as_deref(),
            Some("Vector embedding generation, models, and backfill"),
            "the description must reach the registry, or the dedup has nothing to compare"
        );
        assert_eq!(
            store.lock().expect("store lock")[0].tags,
            vec!["topic:embeddings".to_string()],
            "a tag with no near duplicate is stored as proposed"
        );
    }

    #[tokio::test]
    async fn kb_write_matches_a_description_to_its_tag_on_the_normalised_name() {
        // The model writes each field in whatever shape reads well, and the two
        // fields drift apart in both directions: a pretty description key
        // beside a normalised tag, and a normalised key beside a pretty tag.
        // The vocabulary keys on the normalised name, so a raw string match
        // drops the description and the tag embeds as its bare name - the weak
        // option #1070 rejected.
        let (resolve, probe) = recording_tag_gate(&[]);
        let (service, _store) = kb_service_with_tag_gate(Some(resolve));

        kb_write_response(
            &service,
            serde_json::json!({
                "content": "Embeddings are backfilled by the maintenance task.",
                "tags": ["topic:embeddings", "Topic: Deploy"],
                "new_tag_descriptions": {
                    // Pretty key, normalised tag.
                    "Topic: Embeddings": "Vector embedding generation, models, and backfill",
                    // Normalised key, pretty tag.
                    "topic:deploy": "Releases, rollouts, and the justfile recipes",
                },
            }),
        )
        .await;

        let proposals = probe.lock().expect("probe lock").proposals.clone();
        let described: Vec<(String, Option<String>)> = proposals
            .iter()
            .map(|p| (p.name.clone(), p.description.clone()))
            .collect();
        assert_eq!(
            described,
            vec![
                (
                    "topic:embeddings".to_string(),
                    Some("Vector embedding generation, models, and backfill".to_string())
                ),
                (
                    "Topic: Deploy".to_string(),
                    Some("Releases, rollouts, and the justfile recipes".to_string())
                ),
            ],
            "a description reaches its tag whichever side arrived normalised"
        );
    }

    #[tokio::test]
    async fn kb_write_caps_a_tag_description_at_the_tool_boundary() {
        // A description is written once and read forever: it lands on a registry
        // row that nothing deletes, and the dreaming extraction prompt renders
        // every active tag's description in full. Every knowledge write is now a
        // writer of that surface, so an uncapped description is a permanent
        // charge on every later extraction.
        //
        // It is truncated, never refused. Refusing it would cost the tag its
        // description and drop the check back to matching bare names.
        let (resolve, probe) = recording_tag_gate(&[]);
        let (service, _store) = kb_service_with_tag_gate(Some(resolve));

        let long = "x".repeat(TAG_DESCRIPTION_MAX_CHARS * 3);
        let json = kb_write_response(
            &service,
            serde_json::json!({
                "content": "Embeddings are backfilled by the maintenance task.",
                "tags": ["topic:embeddings"],
                "new_tag_descriptions": {"topic:embeddings": long},
            }),
        )
        .await;

        assert_eq!(
            json["ok"], true,
            "an over-long description never fails a write"
        );
        let proposals = probe.lock().expect("probe lock").proposals.clone();
        let description = proposals[0]
            .description
            .as_deref()
            .expect("the tag keeps a description rather than losing it");
        assert_eq!(
            description.chars().count(),
            TAG_DESCRIPTION_MAX_CHARS,
            "the description reaching the vocabulary is capped"
        );
    }

    #[test]
    fn tag_description_cap_cuts_on_a_character_boundary() {
        // Counted in characters, not bytes, so a description in any script gets
        // the same allowance and the cut can never split one character.
        let multibyte = "\u{e6f8}".repeat(TAG_DESCRIPTION_MAX_CHARS + 10);
        let capped = cap_tag_description(&multibyte);
        assert_eq!(capped.chars().count(), TAG_DESCRIPTION_MAX_CHARS);

        // A description inside the cap is untouched, trailing space and all -
        // the caller already trimmed it.
        assert_eq!(cap_tag_description("short one"), "short one");

        // Whitespace exposed by the cut goes, so the text never ends mid-space.
        let padded = format!("{}   tail", "y".repeat(TAG_DESCRIPTION_MAX_CHARS - 2));
        assert_eq!(
            cap_tag_description(&padded),
            "y".repeat(TAG_DESCRIPTION_MAX_CHARS - 2)
        );
    }

    #[tokio::test]
    async fn kb_write_registers_a_new_tag_that_arrived_without_a_description() {
        // A missing description is not an error. The write must never fail
        // because the model omitted one, and the tag still goes through the
        // registry - the registry falls back to the name alone as its embed
        // text.
        let (resolve, probe) = recording_tag_gate(&[]);
        let (service, store) = kb_service_with_tag_gate(Some(resolve));

        let json = kb_write_response(
            &service,
            serde_json::json!({
                "content": "The deploy runs from the justfile.",
                "tags": ["topic:deploy"],
            }),
        )
        .await;

        assert_eq!(json["ok"], true, "the write succeeds without a description");
        let proposals = probe.lock().expect("probe lock").proposals.clone();
        assert_eq!(proposals.len(), 1, "the tag still reached the registry");
        assert_eq!(proposals[0].name, "topic:deploy");
        assert_eq!(
            proposals[0].description, None,
            "no description was supplied, and the gate is told so rather than \
             being handed an empty string"
        );
        assert_eq!(
            store.lock().expect("store lock")[0].tags,
            vec!["topic:deploy".to_string()]
        );
    }

    #[tokio::test]
    async fn kb_write_ignores_a_description_for_a_tag_the_registry_already_holds() {
        // A description for a tag the registry already holds changes nothing:
        // the registry matches on the name and answers with the stored tag. The
        // write path must not treat the description as an instruction to
        // re-describe or to create a second tag.
        let (resolve, probe) = recording_tag_gate(&[("topic:weather", "topic:weather")]);
        let (service, store) = kb_service_with_tag_gate(Some(resolve));

        kb_write_response(
            &service,
            serde_json::json!({
                "content": "It rained on Tuesday.",
                "tags": ["topic:weather"],
                "new_tag_descriptions": {
                    "topic:weather": "Something else entirely",
                },
            }),
        )
        .await;

        assert_eq!(
            store.lock().expect("store lock")[0].tags,
            vec!["topic:weather".to_string()],
            "the stored tag is the one the registry answered with"
        );
        assert_eq!(
            probe.lock().expect("probe lock").proposals.len(),
            1,
            "one proposal, not one per description"
        );
    }

    #[tokio::test]
    async fn kb_write_keeps_the_normalised_tag_when_the_embedding_backend_is_unavailable() {
        // The registry needs an embedding to find a near duplicate. When the
        // embedding backend is down the gate fails, and the prior behaviour -
        // store the tag as written - is the fallback. A write must never fail
        // because an optional backend is absent.
        use std::sync::Arc;

        let resolve: KnowledgeTagResolveFn = Arc::new(|_proposed: ProposedTag| {
            Box::pin(async {
                Err(CoreError::Storage(
                    "tag_registry: embed returned no vectors".to_string(),
                ))
            })
        });
        let (service, store) = kb_service_with_tag_gate(Some(resolve));

        let json = kb_write_response(
            &service,
            serde_json::json!({
                "content": "Rain is expected on Tuesday.",
                "tags": ["memory", "topic:forecast"],
            }),
        )
        .await;

        assert_eq!(json["ok"], true, "the write succeeds");
        let stored = store.lock().expect("store lock");
        assert_eq!(stored.len(), 1, "the entry landed");
        assert_eq!(
            stored[0].tags,
            vec!["memory".to_string(), "topic:forecast".to_string()],
            "the entry keeps the tags the model wrote"
        );
    }

    #[tokio::test]
    async fn kb_retag_by_id_goes_through_the_registry() {
        // Re-tagging an entry the model found is the path that adds tags most
        // often, so an ungated one defeats the whole gate. It carries an `id`
        // and `tags` with no `content`.
        let (resolve, probe) = recording_tag_gate(&[("topic:forecast", "topic:weather")]);
        let (service, store) = kb_service_with_tag_gate(Some(resolve));

        // Seed an entry through the write path, then re-tag it by id.
        let created = kb_write_response(
            &service,
            serde_json::json!({"content": "Rain is expected on Tuesday.", "tags": ["memory"]}),
        )
        .await;
        let id = created["entries"][0]["id"]
            .as_str()
            .expect("the write reports the entry id")
            .to_string();

        kb_write_response(
            &service,
            serde_json::json!({"id": id, "tags": ["memory", "topic:forecast"]}),
        )
        .await;

        let stored = store.lock().expect("store lock");
        assert_eq!(stored.len(), 1, "the re-tag updated the entry in place");
        assert_eq!(
            stored[0].content, "Rain is expected on Tuesday.",
            "premise: the content is preserved, so this is the tags-only path"
        );
        assert_eq!(
            stored[0].tags,
            vec!["memory".to_string(), "topic:weather".to_string()],
            "the re-tag path is gated too"
        );
        assert!(
            probe
                .lock()
                .expect("probe lock")
                .proposals
                .iter()
                .any(|p| p.name == "topic:forecast"),
            "the re-tagged tag reached the registry"
        );
    }

    #[tokio::test]
    async fn kb_write_stops_consulting_the_tag_vocabulary_after_it_fails_once() {
        // A failure means the vocabulary cannot answer. Asking it again for
        // every remaining tag buys nothing and pays the embedding timeout each
        // time, so one write with many tags would add minutes to a live turn.
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_fn = Arc::clone(&calls);
        let resolve: KnowledgeTagResolveFn = Arc::new(move |_proposed: ProposedTag| {
            let calls = Arc::clone(&calls_for_fn);
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Err(CoreError::Storage(
                    "embedding backend unreachable".to_string(),
                ))
            })
        });
        let (service, store) = kb_service_with_tag_gate(Some(resolve));

        kb_write_response(
            &service,
            serde_json::json!({
                "entries": [
                    {"content": "one", "tags": ["memory", "topic:a", "topic:b"]},
                    {"content": "two", "tags": ["topic:c", "topic:d"]},
                ],
            }),
        )
        .await;

        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "the vocabulary is asked once, then left alone for the rest of the call - \
             across entries, not just within one"
        );
        let stored = store.lock().expect("store lock");
        assert_eq!(stored.len(), 2, "both entries landed");
        assert_eq!(
            stored[0].tags,
            vec![
                "memory".to_string(),
                "topic:a".to_string(),
                "topic:b".to_string()
            ],
            "every tag is stored as written"
        );
        assert_eq!(
            stored[1].tags,
            vec!["topic:c".to_string(), "topic:d".to_string()]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn kb_write_stops_consulting_the_tag_vocabulary_when_the_write_budget_is_spent() {
        // A vocabulary that answers, but slowly, is not an error, so nothing
        // else stops it. One write carries as many tags as the model chose to
        // send, so without a budget the added wait grows with the tag count
        // inside a live turn.
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Each answer costs well over half the budget, so the third tag finds
        // the budget spent.
        let per_call = TAG_RESOLVE_BUDGET.mul_f32(0.6);
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_fn = Arc::clone(&calls);
        let resolve: KnowledgeTagResolveFn = Arc::new(move |proposed: ProposedTag| {
            let calls = Arc::clone(&calls_for_fn);
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(per_call).await;
                Ok(format!("resolved:{}", proposed.name))
            })
        });
        let (service, store) = kb_service_with_tag_gate(Some(resolve));

        kb_write_response(
            &service,
            serde_json::json!({
                "content": "slow",
                "tags": ["topic:a", "topic:b", "topic:c"],
            }),
        )
        .await;

        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "the budget stops the third consultation"
        );
        assert_eq!(
            store.lock().expect("store lock")[0].tags,
            vec![
                "resolved:topic:a".to_string(),
                "resolved:topic:b".to_string(),
                "topic:c".to_string(),
            ],
            "tags resolved before the budget ran out keep their answers; the rest \
             are stored as written"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn kb_write_bounds_one_tag_consultation_and_stores_that_tag_as_written() {
        // One consultation reads the vocabulary, embeds, searches for a near
        // neighbour, and registers. Bounding only the embedding leaves the
        // database round trips around it bounded by the connection pool's
        // acquire timeout, which is tens of seconds each. The whole
        // consultation is bounded instead, so a live turn cannot be held by
        // whatever inside it is slow.
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_fn = Arc::clone(&calls);
        let resolve: KnowledgeTagResolveFn = Arc::new(move |_proposed: ProposedTag| {
            let calls = Arc::clone(&calls_for_fn);
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                // Far longer than any ceiling: a vocabulary that never answers.
                tokio::time::sleep(TAG_RESOLVE_CALL_CEILING * 100).await;
                Ok("never reached".to_string())
            })
        });
        let (service, store) = kb_service_with_tag_gate(Some(resolve));

        let started = tokio::time::Instant::now();
        let json = kb_write_response(
            &service,
            serde_json::json!({
                "content": "hung",
                "tags": ["topic:a", "topic:b"],
            }),
        )
        .await;
        let elapsed = started.elapsed();

        assert_eq!(json["ok"], true, "the write still succeeds");
        assert_eq!(
            elapsed, TAG_RESOLVE_CALL_CEILING,
            "the write waits exactly one ceiling, not the vocabulary's own pace"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "a vocabulary that did not answer is not asked again for this write"
        );
        assert_eq!(
            store.lock().expect("store lock")[0].tags,
            vec!["topic:a".to_string(), "topic:b".to_string()],
            "both tags are stored as the model wrote them"
        );
        assert_eq!(
            json["tag_check"], TAG_CHECK_UNKNOWN,
            "a write whose tags went unchecked says so"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn kb_write_charges_the_tag_budget_only_for_time_spent_in_the_vocabulary() {
        // Storing an entry is not a consultation. Against a wall-clock deadline
        // a batch with slow stores loses the vocabulary on its last entries
        // with nothing slow about the vocabulary at all - the model's later
        // tags then go unchecked for a reason the vocabulary did not cause.
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Three stores at 60% of the budget each: 180% of the budget spent
        // outside the vocabulary, which answers instantly.
        let store_delay = TAG_RESOLVE_BUDGET.mul_f32(0.6);
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_fn = Arc::clone(&calls);
        let resolve: KnowledgeTagResolveFn = Arc::new(move |proposed: ProposedTag| {
            let calls = Arc::clone(&calls_for_fn);
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(format!("resolved:{}", proposed.name))
            })
        });
        let (service, store) = kb_service_with_slow_store(Some(resolve), store_delay);

        let json = kb_write_response(
            &service,
            serde_json::json!({
                "entries": [
                    {"content": "one", "tags": ["topic:a"]},
                    {"content": "two", "tags": ["topic:b"]},
                    {"content": "three", "tags": ["topic:c"]},
                ],
            }),
        )
        .await;

        assert_eq!(
            calls.load(Ordering::SeqCst),
            3,
            "every tag is still checked: the slow stores are not the vocabulary's spend"
        );
        let stored = store.lock().expect("store lock");
        assert_eq!(
            stored
                .iter()
                .flat_map(|e| e.tags.clone())
                .collect::<Vec<_>>(),
            vec![
                "resolved:topic:a".to_string(),
                "resolved:topic:b".to_string(),
                "resolved:topic:c".to_string(),
            ],
            "every entry carries the vocabulary's answer"
        );
        assert!(
            json.get("tag_check").is_none(),
            "nothing was left unchecked, so the write reports no degradation: {json}"
        );
    }

    #[tokio::test]
    async fn kb_write_reports_that_a_tag_was_stored_without_being_checked() {
        // A degraded write stores whatever the model wrote. Answering with a
        // response byte-identical to a checked write lets the model read its
        // own drift back as established vocabulary, which is the failure the
        // vocabulary exists to prevent.
        use std::sync::Arc;

        let resolve: KnowledgeTagResolveFn = Arc::new(|_proposed: ProposedTag| {
            Box::pin(async {
                Err(CoreError::Storage(
                    "tag_registry: embed returned no vectors".to_string(),
                ))
            })
        });
        let (service, _store) = kb_service_with_tag_gate(Some(resolve));

        let json = kb_write_response(
            &service,
            serde_json::json!({"content": "Rain on Tuesday.", "tags": ["topic:forecast"]}),
        )
        .await;

        assert_eq!(json["ok"], true, "the write still succeeds");
        assert_eq!(
            json["tag_check"], TAG_CHECK_UNKNOWN,
            "not measured must never read as measured: {json}"
        );
    }

    #[tokio::test]
    async fn kb_write_reports_unchecked_tags_when_no_vocabulary_is_wired() {
        // With no database or no embedding backend the gate is not wired at
        // all, so no tag on the write was checked. That is the same claim as a
        // vocabulary that failed, and it reads the same way.
        let (service, _store) = kb_service_with_tag_gate(None);

        let json = kb_write_response(
            &service,
            serde_json::json!({"content": "Rain on Tuesday.", "tags": ["topic:forecast"]}),
        )
        .await;

        assert_eq!(
            json["tag_check"], TAG_CHECK_UNKNOWN,
            "an unwired vocabulary checked nothing, and says so: {json}"
        );
    }

    #[tokio::test]
    async fn kb_write_omits_the_tag_check_when_the_vocabulary_answered_for_every_tag() {
        // The field is absent on the ordinary path, so its presence carries the
        // whole signal and a checked write costs no extra bytes.
        let (resolve, _probe) = recording_tag_gate(&[]);
        let (service, _store) = kb_service_with_tag_gate(Some(resolve));

        let json = kb_write_response(
            &service,
            serde_json::json!({"content": "Rain on Tuesday.", "tags": ["topic:forecast"]}),
        )
        .await;

        assert!(
            json.get("tag_check").is_none(),
            "every tag was checked, so nothing is reported: {json}"
        );
    }

    #[tokio::test]
    async fn kb_write_without_tags_reports_nothing_about_the_vocabulary() {
        // A write that carries no tags has nothing to check, so an unwired
        // vocabulary is not a degradation of it.
        let (service, _store) = kb_service_with_tag_gate(None);

        let json =
            kb_write_response(&service, serde_json::json!({"content": "Rain on Tuesday."})).await;

        assert!(
            json.get("tag_check").is_none(),
            "no tags, nothing unchecked: {json}"
        );
    }

    #[test]
    fn kb_write_schema_advertises_new_tag_descriptions() {
        // A schema that promises what the code does not honour is a false
        // contract, and so is code that honours what the schema never
        // advertised: the model cannot supply a field it was never told about.
        let service = BuiltinToolService::new();
        let def = service
            .tool_definitions()
            .into_iter()
            .find(|t| t.name == TOOL_KB_WRITE)
            .expect("kb_write tool is advertised");
        let props = &def.parameters["properties"];

        let single = &props["new_tag_descriptions"];
        assert_eq!(
            single["type"], "object",
            "new_tag_descriptions is a map from tag name to description: {single}"
        );
        assert_eq!(
            single["additionalProperties"]["type"], "string",
            "each value is a one-line description: {single}"
        );
        assert!(
            single["description"]
                .as_str()
                .is_some_and(|d| !d.is_empty()),
            "the field carries a description of its own: {single}"
        );
        // A schema that promises what the code does not honour is a false
        // contract, and so is one that stays silent about a constraint the code
        // applies. The cap is enforced, so it is advertised.
        assert!(
            single["description"]
                .as_str()
                .is_some_and(|d| d.contains(&TAG_DESCRIPTION_MAX_CHARS.to_string())
                    && d.contains("truncated")),
            "the field must advertise the length cap it enforces: {single}"
        );

        // The batch form is held to the same shape. A model that batches its
        // writes reads only this half of the schema.
        let batch = &props["entries"]["items"]["properties"]["new_tag_descriptions"];
        assert_eq!(
            batch["type"], "object",
            "the batch form advertises the field too, or a batched write cannot \
             describe its new tags: {batch}"
        );
        assert_eq!(
            batch["additionalProperties"]["type"], "string",
            "each batch value is a one-line description: {batch}"
        );
        assert!(
            batch["description"].as_str().is_some_and(|d| !d.is_empty()),
            "the batch field carries a description of its own: {batch}"
        );

        // The batch form ignores every top-level field, this one included. A
        // description sent at the top level beside `entries` is dropped in
        // silence, so the list of what `entries` ignores has to name it.
        let entries = props["entries"]["description"]
            .as_str()
            .expect("entries carries a description");
        assert!(
            entries.contains("new_tag_descriptions"),
            "the entries description must say that a top-level \
             new_tag_descriptions is ignored: {entries}"
        );
    }

    #[test]
    fn kb_write_schema_says_a_write_by_id_keeps_what_it_leaves_out() {
        // The model decides what to send from this text alone. Told only that
        // `id` updates an entry, it re-sends the tags on every content update
        // to be safe, and each of those round trips is an unnecessary
        // vocabulary consultation that can rename a stored tag. Told the rule,
        // it sends only what changed.
        let service = BuiltinToolService::new();
        let def = service
            .tool_definitions()
            .into_iter()
            .find(|t| t.name == TOOL_KB_WRITE)
            .expect("kb_write tool is advertised");

        let lower = def.description.to_lowercase();
        // Named, not summarised as "every field". A write does not keep the
        // entry's provenance, so a description that says "every" is a promise
        // the code breaks - see
        // `kb_write_schema_says_a_write_records_the_entry_as_explicit`.
        assert!(
            lower.contains("keeps the content, the tags and the stored metadata"),
            "the description must name what a write by id keeps, field by \
             field: {}",
            def.description
        );
        assert!(
            !lower.contains("keeps every field"),
            "a write does not keep every field - it overwrites the entry's \
             provenance - so the description must not claim it does: {}",
            def.description
        );
        assert!(
            lower.contains("empty `tags` list"),
            "the description must say how to clear the tags, or the model has \
             no way to: {}",
            def.description
        );

        // An id no entry holds is a create at that id, which is what makes a
        // write repeatable. A model told only "ID for updates" will not retry
        // with the same id.
        let id = def.parameters["properties"]["id"]["description"]
            .as_str()
            .expect("id carries a description");
        assert!(
            id.contains("no entry holds") && id.contains("repeated"),
            "the id description must say that an unheld id creates, so a write \
             can be repeated: {id}"
        );
    }

    #[test]
    fn kb_write_schema_says_a_write_records_the_entry_as_explicit() {
        // `build_write_entry` always sends `source = "explicit"`, and the
        // upsert only preserves the stored value when the write sends none. So
        // a write flips a consolidated entry's provenance, and
        // `builtin_knowledge_base_list` with `source: "consolidation"` stops
        // returning it. A model that reads "a write keeps what it leaves out"
        // and then cannot find that entry again has been told something false.
        let service = BuiltinToolService::new();
        let def = service
            .tool_definitions()
            .into_iter()
            .find(|t| t.name == TOOL_KB_WRITE)
            .expect("kb_write tool is advertised");

        let lower = def.description.to_lowercase();
        assert!(
            lower.contains("provenance") && lower.contains("'explicit'"),
            "the description must say that a write records the entry as \
             explicit: {}",
            def.description
        );
        assert!(
            lower.contains("builtin_knowledge_base_list"),
            "the description must say which read that provenance change \
             affects, or the model cannot tell what it costs: {}",
            def.description
        );
    }

    #[test]
    fn kb_write_schema_says_a_retired_id_is_refused() {
        // Consolidation retires an entry overnight while the model still holds
        // its id, so this is a live outcome, not a corner. Told that an id no
        // entry holds simply creates, a model reads the refusal as a broken
        // tool and retries the same call. Told the rule, it drops the id and
        // the fact is saved.
        let service = BuiltinToolService::new();
        let def = service
            .tool_definitions()
            .into_iter()
            .find(|t| t.name == TOOL_KB_WRITE)
            .expect("kb_write tool is advertised");

        let id = def.parameters["properties"]["id"]["description"]
            .as_str()
            .expect("id carries a description")
            .to_lowercase();
        assert!(
            id.contains("retired") && id.contains("refused"),
            "the id description must say a retired id is refused: {id}"
        );
        assert!(
            id.contains("new entry with no id"),
            "the id description must say what to do about a refusal, or the \
             model retries the same call: {id}"
        );
    }

    #[test]
    fn kb_write_schema_warns_against_a_readable_caller_chosen_id() {
        // An id is a bare key, not scoped to this conversation or this topic.
        // A model told it may choose one picks a readable name, that name can
        // already be taken, and the write then fails with nothing stored -
        // which reads to the model as a broken tool rather than a naming rule
        // it broke.
        let service = BuiltinToolService::new();
        let def = service
            .tool_definitions()
            .into_iter()
            .find(|t| t.name == TOOL_KB_WRITE)
            .expect("kb_write tool is advertised");

        let id = def.parameters["properties"]["id"]["description"]
            .as_str()
            .expect("id carries a description")
            .to_lowercase();
        assert!(
            id.contains("uuid") && id.contains("readable name"),
            "the id description must say a made-up id is a fresh random one, \
             never a readable name: {id}"
        );
        assert!(
            id.contains("already be taken") && id.contains("fails"),
            "the id description must say what a readable name costs, or the \
             rule reads as arbitrary: {id}"
        );
    }

    #[test]
    fn kb_write_schema_describes_the_degraded_write_report() {
        // The model can only act on `tag_check` if it was told the field
        // exists and what it means. A response field the schema never mentions
        // is one the model reads as noise.
        let service = BuiltinToolService::new();
        let def = service
            .tool_definitions()
            .into_iter()
            .find(|t| t.name == TOOL_KB_WRITE)
            .expect("kb_write tool is advertised");

        assert!(
            def.description.contains("tag_check")
                && def.description.contains(TAG_CHECK_UNKNOWN)
                && def.description.contains("without"),
            "the tool description must name the field, its value, and what it means: {}",
            def.description
        );
    }

    #[test]
    fn kb_write_schema_states_what_the_tag_check_compares_against() {
        // The vocabulary is the registry's own list of registered tags, which
        // starts empty and grows as tags are written; nothing seeds it from
        // the tags entries already carry (#1094). Advertising it as a check
        // "against the tags that already exist" promises a check that a
        // default install cannot perform.
        let service = BuiltinToolService::new();
        let def = service
            .tool_definitions()
            .into_iter()
            .find(|t| t.name == TOOL_KB_WRITE)
            .expect("kb_write tool is advertised");

        assert!(
            def.description.contains("vocabulary") && def.description.contains("starts empty"),
            "the description must say what the check compares against, and that it \
             starts empty: {}",
            def.description
        );
        assert!(
            !def.description
                .contains("checked against the tags that already exist"),
            "the tags an entry carries are not what a tag is checked against: {}",
            def.description
        );
    }

    // -- knowledge-base reads: the one-line summary (#1097) ------------------

    /// A knowledge-base service whose store answers every list with `entries`.
    /// The list tool is the second read surface the model sees, so it needs a
    /// store that returns rows where `kb_service_reporting`'s returns none.
    fn kb_service_listing(
        entries: Vec<desktop_assistant_core::domain::KnowledgeEntry>,
    ) -> BuiltinToolService {
        use std::sync::Arc;

        let write_fn: KnowledgeWriteFn = Arc::new(|entry| Box::pin(async move { Ok(entry) }));
        let search_fn: KnowledgeSearchFn = Arc::new(|_q, _e, _m, _t, _x, _l| {
            Box::pin(async {
                Ok(KnowledgeSearchPage {
                    entries: Vec::new(),
                    scope_size: ScopeSize::None,
                    available_tags: Vec::new(),
                })
            })
        });
        let delete_fn: KnowledgeDeleteFn = Arc::new(|_ids| Box::pin(async { Ok(0) }));
        let list_fn: KnowledgeListFn = Arc::new(move |_q| {
            let entries = entries.clone();
            Box::pin(async move {
                Ok(
                    desktop_assistant_core::ports::knowledge::KnowledgeListPage {
                        entries,
                        next_cursor: None,
                    },
                )
            })
        });
        let get_fn: KnowledgeGetFn = Arc::new(|_id| Box::pin(async { Ok(None) }));
        BuiltinToolService::new()
            .with_knowledge_base(write_fn, search_fn, delete_fn, list_fn, get_fn)
    }

    #[tokio::test]
    async fn kb_search_results_carry_the_summary() {
        // A search result is what the model reads before it decides whether to
        // pull the body. Without the summary on that row it can only judge the
        // entry from its whole content.
        let mut entry = kb_entry("kb-1", &["preference"]);
        entry.summary = Some("Prefers dark mode in every editor".to_string());
        let (service, _probe) = kb_service_reporting(KnowledgeSearchPage {
            entries: vec![entry],
            scope_size: ScopeSize::Few,
            available_tags: Vec::new(),
        });

        let json = kb_search_response(&service, serde_json::json!({"query": "dark mode"})).await;

        assert_eq!(
            json["results"][0]["summary"], "Prefers dark mode in every editor",
            "the search result carries the entry's one-line summary"
        );
    }

    #[tokio::test]
    async fn kb_search_reports_a_null_summary_for_an_entry_that_has_none() {
        // Every entry written before the field existed has no summary. The row
        // must still report the field, as null, so a caller can tell "no
        // summary yet" from "this reader dropped it".
        let (service, _probe) = kb_service_reporting(KnowledgeSearchPage {
            entries: vec![kb_entry("kb-1", &["preference"])],
            scope_size: ScopeSize::Few,
            available_tags: Vec::new(),
        });

        let json = kb_search_response(&service, serde_json::json!({"query": "dark mode"})).await;

        assert_eq!(json["results"][0]["summary"], serde_json::Value::Null);
    }

    #[tokio::test]
    async fn kb_list_carries_the_summary() {
        let mut entry = kb_entry("kb-1", &["preference"]);
        entry.summary = Some("Prefers dark mode in every editor".to_string());
        let service = kb_service_listing(vec![entry]);

        let raw = service
            .execute_tool(TOOL_KB_LIST, serde_json::json!({"limit": 10}))
            .await
            .expect("knowledge base list succeeds");
        let json: serde_json::Value = serde_json::from_str(&raw).expect("list response is JSON");

        assert_eq!(json["count"], 1);
        assert_eq!(
            json["entries"][0]["summary"], "Prefers dark mode in every editor",
            "the list row carries the entry's one-line summary"
        );
    }

    #[tokio::test]
    async fn tool_search_with_closure() {
        use desktop_assistant_core::domain::ToolDefinition;
        use std::sync::Arc;

        let search_fn: ToolSearchFn = Arc::new(|_query, _emb, _limit| {
            Box::pin(async {
                Ok(vec![ToolDefinition::new(
                    "jira__create_issue",
                    "Create a Jira issue",
                    serde_json::json!({}),
                )])
            })
        });

        let def_fn: ToolDefinitionFn = Arc::new(|_name| Box::pin(async { Ok(None) }));

        let service = BuiltinToolService::new().with_tool_registry(search_fn, def_fn);

        let result = service
            .execute_tool(TOOL_SEARCH, serde_json::json!({"query": "create ticket"}))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["ok"], true);
        let tools = json["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "jira__create_issue");
    }

    // --- Runner on search results (#1082) ---------------------------------

    /// A registry search that always returns the given tools.
    fn fixed_search(tools: Vec<ToolDefinition>) -> ToolSearchFn {
        std::sync::Arc::new(move |_query, _emb, _limit| {
            let tools = tools.clone();
            Box::pin(async move { Ok(tools) })
        })
    }

    fn noop_definition_fn() -> ToolDefinitionFn {
        std::sync::Arc::new(|_name| Box::pin(async { Ok(None) }))
    }

    /// A client-tool port registering the given definitions, so a search can
    /// see what the connected client offers.
    struct FakeClientTools(Vec<ToolDefinition>);

    #[async_trait::async_trait]
    impl desktop_assistant_core::ports::client_tools::ClientToolPort for FakeClientTools {
        async fn tool_definitions(&self) -> Vec<ToolDefinition> {
            self.0.clone()
        }
        async fn is_registered(&self, name: &str) -> bool {
            self.0.iter().any(|t| t.name == name)
        }
        async fn execute(
            &self,
            _id: &str,
            _name: &str,
            _args: serde_json::Value,
        ) -> Result<String, CoreError> {
            Ok(String::new())
        }
    }

    /// Run a tool search with the given client-registered tools in scope.
    async fn search_with_client_tools(
        service: &BuiltinToolService,
        query: &str,
        client_tools: Vec<ToolDefinition>,
    ) -> serde_json::Value {
        use desktop_assistant_core::ports::client_tools::with_client_tools;
        let port: std::sync::Arc<dyn desktop_assistant_core::ports::client_tools::ClientToolPort> =
            std::sync::Arc::new(FakeClientTools(client_tools));
        let result = with_client_tools(
            port,
            service.execute_tool(TOOL_SEARCH, serde_json::json!({ "query": query })),
        )
        .await
        .expect("tool search must succeed");
        serde_json::from_str(&result).expect("tool search must return JSON")
    }

    #[tokio::test]
    async fn search_result_carries_daemon_runner_for_mcp_tool() {
        // No MCP executor is wired, so nothing is routed to an HTTP server and
        // every registry row runs inside the daemon.
        let service = BuiltinToolService::new()
            .with_tool_registry(
                fixed_search(vec![ToolDefinition::new(
                    "fileio__read_file",
                    "Read a file from disk",
                    serde_json::json!({}),
                )]),
                noop_definition_fn(),
            )
            .with_topology("daemon-host", false);

        let result = service
            .execute_tool(TOOL_SEARCH, serde_json::json!({"query": "read a file"}))
            .await
            .expect("tool search must succeed");
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["tools"][0]["runs_on"], "daemon");
        let legend = json["runs_on"]["daemon"].as_str().expect("daemon legend");
        assert!(
            legend.contains("daemon-host"),
            "the legend must name the daemon's machine: {legend}"
        );
        assert!(
            legend.contains("not the user's own computer"),
            "and say it is not the user's, since it is not on a workstation: {legend}"
        );
    }

    #[tokio::test]
    async fn search_result_carries_remote_service_runner_for_http_mcp_server() {
        // A server reached over HTTP acts on a third-party service. Reporting
        // it as "the daemon's machine" is what makes a model believe a remote
        // calendar tool can read local files.
        use crate::executor::McpServerConfig;
        // Built through deserialization, the same path a real config takes, so
        // a new field on either struct does not break this fixture.
        let http_server: McpServerConfig = serde_json::from_value(serde_json::json!({
            "name": "calendar",
            "http": { "url": "https://mcp.example.com/calendar" },
        }))
        .expect("http server fixture must deserialize");
        let stdio_server: McpServerConfig = serde_json::from_value(serde_json::json!({
            "name": "fileio",
            "command": "fileio-mcp",
        }))
        .expect("stdio server fixture must deserialize");
        let routing = std::collections::HashMap::from([
            (
                "calendar__list_events".to_string(),
                (0usize, "list_events".to_string()),
            ),
            (
                "fileio__read_file".to_string(),
                (1usize, "read_file".to_string()),
            ),
        ]);
        let handle = McpControlHandle::seeded_for_test(vec![http_server, stdio_server], routing);

        let mut service = BuiltinToolService::new().with_tool_registry(
            fixed_search(vec![
                ToolDefinition::new(
                    "calendar__list_events",
                    "List calendar events",
                    serde_json::json!({}),
                ),
                ToolDefinition::new(
                    "fileio__read_file",
                    "Read a file from disk",
                    serde_json::json!({}),
                ),
            ]),
            noop_definition_fn(),
        );
        service.set_mcp_control(handle);

        let result = service
            .execute_tool(
                TOOL_SEARCH,
                serde_json::json!({"query": "events and files"}),
            )
            .await
            .expect("tool search must succeed");
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let by_name: std::collections::HashMap<&str, &str> = json["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| (t["name"].as_str().unwrap(), t["runs_on"].as_str().unwrap()))
            .collect();
        assert_eq!(by_name["calendar__list_events"], "remote-service");
        assert_eq!(by_name["fileio__read_file"], "daemon");
        assert!(
            json["runs_on"]["remote-service"]
                .as_str()
                .is_some_and(|l| l.contains("no local files")),
            "the legend must say a remote service touches no local files: {json}"
        );
    }

    #[tokio::test]
    async fn search_result_carries_device_runner_for_client_registered_tool() {
        let service = BuiltinToolService::new()
            .with_tool_registry(fixed_search(Vec::new()), noop_definition_fn());
        let json = search_with_client_tools(
            &service,
            "read a file",
            vec![ToolDefinition::new(
                "device__read_file",
                "Read a file on the user's own computer",
                serde_json::json!({}),
            )],
        )
        .await;
        assert_eq!(json["tools"][0]["name"], "device__read_file");
        assert_eq!(json["tools"][0]["runs_on"], "device");
        assert!(
            json["runs_on"]["device"]
                .as_str()
                .is_some_and(|l| l.contains("the user's own")),
            "the device legend must say whose machine it is: {json}"
        );
    }

    #[tokio::test]
    async fn client_tools_match_the_query_and_join_the_results() {
        // The registry hit and the client tool answer the same need on two
        // different machines. Both must be offered, so the model can choose.
        let service = BuiltinToolService::new().with_tool_registry(
            fixed_search(vec![ToolDefinition::new(
                "fileio__read_file",
                "Read a file from disk",
                serde_json::json!({}),
            )]),
            noop_definition_fn(),
        );
        let json = search_with_client_tools(
            &service,
            "read a file",
            vec![
                ToolDefinition::new(
                    "device__read_file",
                    "Read a file on the user's own computer",
                    serde_json::json!({}),
                ),
                ToolDefinition::new(
                    "device__play_music",
                    "Start playback on the speakers",
                    serde_json::json!({}),
                ),
            ],
        )
        .await;
        let names: Vec<&str> = json["tools"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| t["name"].as_str())
            .collect();
        assert!(
            names.contains(&"fileio__read_file") && names.contains(&"device__read_file"),
            "both machines' answers must be offered: {names:?}"
        );
        assert!(
            !names.contains(&"device__play_music"),
            "an unrelated client tool must not be returned: {names:?}"
        );
    }

    #[tokio::test]
    async fn co_located_turn_collapses_a_duplicated_capability_to_the_daemon_entry() {
        // On one machine a client tool and a daemon tool of the same name are
        // the same capability, so offering both is a choice with no difference.
        // The default transport is UDS, which means co-located.
        let service = BuiltinToolService::new().with_tool_registry(
            fixed_search(vec![ToolDefinition::new(
                "read_file",
                "Read a file from disk",
                serde_json::json!({}),
            )]),
            noop_definition_fn(),
        );
        let json = search_with_client_tools(
            &service,
            "read a file",
            vec![ToolDefinition::new(
                "read_file",
                "Read a file from disk",
                serde_json::json!({}),
            )],
        )
        .await;
        assert_eq!(json["same_machine"], true);
        let tools = json["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1, "the duplicate must collapse: {json}");
        assert_eq!(tools[0]["runs_on"], "daemon");
    }

    #[tokio::test]
    async fn search_response_carries_one_locations_block() {
        // The legend is emitted once per response, and only for the runners
        // that actually appear, so it never describes a choice the model does
        // not have.
        let service = BuiltinToolService::new().with_tool_registry(
            fixed_search(vec![
                ToolDefinition::new("a__one", "First tool", serde_json::json!({})),
                ToolDefinition::new("a__two", "Second tool", serde_json::json!({})),
            ]),
            noop_definition_fn(),
        );
        let result = service
            .execute_tool(TOOL_SEARCH, serde_json::json!({"query": "tool"}))
            .await
            .expect("tool search must succeed");
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let legend = json["runs_on"].as_object().expect("a legend object");
        assert_eq!(legend.len(), 1, "only the present runner: {json}");
        assert!(legend.contains_key("daemon"));
        assert!(json["tools"].as_array().unwrap().len() == 2);
    }

    #[tokio::test]
    async fn unknown_source_falls_back_to_daemon_runner() {
        // A built-in has no MCP route at all. It still runs inside the daemon
        // process, so reporting anything else would be wrong.
        let service = BuiltinToolService::new().with_tool_registry(
            fixed_search(vec![ToolDefinition::new(
                "builtin_knowledge_base_search",
                "Search the knowledge base",
                serde_json::json!({}),
            )]),
            noop_definition_fn(),
        );
        let result = service
            .execute_tool(TOOL_SEARCH, serde_json::json!({"query": "knowledge"}))
            .await
            .expect("tool search must succeed");
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["tools"][0]["runs_on"], "daemon");
    }

    #[test]
    fn client_tool_matching_ranks_and_reports_what_it_dropped() {
        let tools: Vec<ToolDefinition> = (0..12)
            .map(|i| {
                ToolDefinition::new(
                    format!("device__read_{i:02}"),
                    "Read a file",
                    serde_json::json!({}),
                )
            })
            .collect();
        let (kept, dropped) = match_client_tools("read files", &tools, 10);
        assert_eq!(kept.len(), 10);
        assert_eq!(dropped, 2, "a truncated set must report what it dropped");
        // Deterministic order: equal scores fall back to the name.
        assert_eq!(kept[0].name, "device__read_00");

        // "files" must find "file": the matcher agrees on a prefix either way.
        let (kept, _) = match_client_tools("files", &tools[..1], 10);
        assert_eq!(kept.len(), 1);

        // A query of only short words separates nothing, so it matches nothing.
        let (kept, _) = match_client_tools("a of", &tools, 10);
        assert!(kept.is_empty());
    }

    #[tokio::test]
    async fn skill_search_and_get_with_closures() {
        use desktop_assistant_core::domain::{IndexedSkill, Locality, SkillKind, TrustTier};
        use desktop_assistant_core::ports::skill_index::{SkillGetFn, SkillSearchFn};
        use std::sync::Arc;

        fn sample(name: &str, kind: SkillKind) -> IndexedSkill {
            IndexedSkill {
                name: name.to_string(),
                description: format!("does {name}"),
                kind,
                disk_path: format!("/skills/{name}/SKILL.md"),
                owner_user_id: None,
                locality: Locality::Daemon,
                content_hash: "h".to_string(),
                trust_tier: TrustTier::Local,
                source: None,
                tags: vec!["ops".to_string()],
                attachments: vec!["scripts/run.sh".to_string()],
                body: "# body\n\n## Steps\n1. go".to_string(),
                metadata: serde_json::Value::Null,
                present_on_disk: true,
                last_seen_at: None,
            }
        }

        let search_fn: SkillSearchFn = Arc::new(|_q, _emb, _model, _limit| {
            Box::pin(async {
                Ok(vec![
                    sample("invoicing", SkillKind::Workflow),
                    sample("notes", SkillKind::Skill),
                ])
            })
        });
        let get_fn: SkillGetFn = Arc::new(|name, _owner| {
            Box::pin(async move {
                Ok((name == "invoicing").then(|| sample("invoicing", SkillKind::Workflow)))
            })
        });
        let service = BuiltinToolService::new().with_skills(search_fn, get_fn);

        // The `kind` filter keeps only workflows.
        let out = service
            .execute_tool(
                TOOL_SKILL_SEARCH,
                serde_json::json!({"query": "invoice", "kind": "workflow"}),
            )
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(json["ok"], true);
        let results = json["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["name"], "invoicing");
        assert_eq!(results[0]["kind"], "workflow");

        // `get` returns the full body for a hit and `ok:false` for a miss.
        let hit = service
            .execute_tool(TOOL_SKILL_GET, serde_json::json!({"name": "invoicing"}))
            .await
            .unwrap();
        let hit_json: serde_json::Value = serde_json::from_str(&hit).unwrap();
        assert_eq!(hit_json["ok"], true);
        assert!(hit_json["body"].as_str().unwrap().contains("## Steps"));
        assert_eq!(hit_json["attachments"][0], "scripts/run.sh");

        let miss = service
            .execute_tool(TOOL_SKILL_GET, serde_json::json!({"name": "nope"}))
            .await
            .unwrap();
        let miss_json: serde_json::Value = serde_json::from_str(&miss).unwrap();
        assert_eq!(miss_json["ok"], false);
    }

    /// Build a same-named "deploy" `IndexedSkill` for the `skill_get`
    /// fallback-chain tests below, so each test states only what varies: a
    /// `description` (to prove which row won) and whether the row is live or
    /// a tombstone (`present_on_disk`).
    fn deploy_skill(
        description: &str,
        present_on_disk: bool,
    ) -> desktop_assistant_core::domain::IndexedSkill {
        use desktop_assistant_core::domain::{IndexedSkill, Locality, SkillKind, TrustTier};
        IndexedSkill {
            name: "deploy".to_string(),
            description: description.to_string(),
            kind: SkillKind::Skill,
            disk_path: "/skills/deploy/SKILL.md".to_string(),
            owner_user_id: None,
            locality: Locality::Daemon,
            content_hash: "h".to_string(),
            trust_tier: TrustTier::Local,
            source: None,
            tags: vec![],
            attachments: vec![],
            body: "body".to_string(),
            metadata: serde_json::Value::Null,
            present_on_disk,
            last_seen_at: None,
        }
    }

    /// #911: the `owner` argument is gone from the wire, so `skill_get` has
    /// to resolve scope itself. When a row exists in both the caller's own
    /// scope and the global one, the caller's own wins -- the override-layer
    /// precedence the schema's description now promises.
    #[tokio::test]
    async fn skill_get_prefers_the_callers_own_skill_over_the_global_one() {
        use desktop_assistant_core::ports::skill_index::{SkillGetFn, SkillSearchFn};
        use std::sync::Arc;

        // The closure stands in for the store: `owner.is_some()` is exactly
        // what `PgSkillIndexStore::get` treats as "the caller's own scope"
        // (its string content is never inspected -- see `get`'s doc).
        let get_fn: SkillGetFn = Arc::new(|name, owner| {
            Box::pin(async move {
                if name != "deploy" {
                    return Ok(None);
                }
                Ok(Some(match owner {
                    Some(_) => deploy_skill("caller's own", true),
                    None => deploy_skill("global", true),
                }))
            })
        });
        let search_fn: SkillSearchFn =
            Arc::new(|_q, _emb, _model, _limit| Box::pin(async { Ok(Vec::new()) }));
        let service = BuiltinToolService::new().with_skills(search_fn, get_fn);

        let out = service
            .execute_tool(TOOL_SKILL_GET, serde_json::json!({"name": "deploy"}))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(
            json["description"], "caller's own",
            "the caller's own row wins over the global one of the same name"
        );
    }

    /// #911: with no own-scoped row, `skill_get` falls back to the global
    /// skill of the same name rather than reporting a miss.
    ///
    /// Regression guard, not a discriminator: with no `owner` key in the
    /// payload at all, the pre-#911 handler also computed `owner = None` and
    /// made this same single `get_fn(name, None)` call, so this scenario
    /// passes unchanged against the old handler. It still pins down real
    /// behavior worth keeping -- `skill_get_prefers_the_callers_own_skill_over_the_global_one`
    /// and the tombstone-shadowing tests below are the ones that actually
    /// exercise the new two-step (own-then-global) lookup and would fail
    /// against the old single-call handler.
    #[tokio::test]
    async fn skill_get_falls_back_to_the_global_skill_when_the_caller_has_none() {
        use desktop_assistant_core::ports::skill_index::{SkillGetFn, SkillSearchFn};
        use std::sync::Arc;

        // The caller has no own-scoped row: `owner.is_some()` misses, `None`
        // (the global lookup) hits.
        let get_fn: SkillGetFn = Arc::new(|name, owner| {
            Box::pin(async move {
                if name != "deploy" || owner.is_some() {
                    return Ok(None);
                }
                Ok(Some(deploy_skill("global", true)))
            })
        });
        let search_fn: SkillSearchFn =
            Arc::new(|_q, _emb, _model, _limit| Box::pin(async { Ok(Vec::new()) }));
        let service = BuiltinToolService::new().with_skills(search_fn, get_fn);

        let out = service
            .execute_tool(TOOL_SKILL_GET, serde_json::json!({"name": "deploy"}))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(json["description"], "global");
    }

    /// A personal skill removed from disk survives in the append-only
    /// catalog as a `present_on_disk: false` tombstone. Before #911, a
    /// caller who hit that tombstone could still reach the global skill of
    /// the same name by omitting `owner`. That escape hatch is gone now that
    /// `owner` is off the wire, so `skill_get` itself must not let a dead
    /// personal row permanently shadow a live global one.
    #[tokio::test]
    async fn skill_get_a_removed_personal_skill_does_not_shadow_a_live_global_skill() {
        use desktop_assistant_core::ports::skill_index::{SkillGetFn, SkillSearchFn};
        use std::sync::Arc;

        let get_fn: SkillGetFn = Arc::new(|name, owner| {
            Box::pin(async move {
                if name != "deploy" {
                    return Ok(None);
                }
                Ok(Some(match owner {
                    Some(_) => deploy_skill("caller's own (removed)", false),
                    None => deploy_skill("global", true),
                }))
            })
        });
        let search_fn: SkillSearchFn =
            Arc::new(|_q, _emb, _model, _limit| Box::pin(async { Ok(Vec::new()) }));
        let service = BuiltinToolService::new().with_skills(search_fn, get_fn);

        let out = service
            .execute_tool(TOOL_SKILL_GET, serde_json::json!({"name": "deploy"}))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(
            json["description"], "global",
            "a personal tombstone must not shadow a live global skill of the same name"
        );
        assert_eq!(json["present_on_disk"], true);
    }

    /// The mirror case: when the caller's personal row is a tombstone and
    /// there is no global skill of the same name at all, the tombstone is
    /// the only record that ever existed for that name, so it is still
    /// returned (flagged, not hidden) rather than reporting a plain miss.
    #[tokio::test]
    async fn skill_get_returns_the_personal_tombstone_when_no_global_skill_exists() {
        use desktop_assistant_core::ports::skill_index::{SkillGetFn, SkillSearchFn};
        use std::sync::Arc;

        let get_fn: SkillGetFn = Arc::new(|name, owner| {
            Box::pin(async move {
                if name != "deploy" {
                    return Ok(None);
                }
                Ok(owner.map(|_| deploy_skill("caller's own (removed)", false)))
            })
        });
        let search_fn: SkillSearchFn =
            Arc::new(|_q, _emb, _model, _limit| Box::pin(async { Ok(Vec::new()) }));
        let service = BuiltinToolService::new().with_skills(search_fn, get_fn);

        let out = service
            .execute_tool(TOOL_SKILL_GET, serde_json::json!({"name": "deploy"}))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(json["description"], "caller's own (removed)");
        assert_eq!(json["present_on_disk"], false);
    }

    /// #911: `owner` used to be an LLM-supplied scope selector forwarded
    /// straight to the store -- a false contract once the store started
    /// ignoring its value. It must not still appear on the wire, or a caller
    /// keeps passing it and silently gets different behavior than any
    /// lingering documentation would promise. This guard and the schema stay
    /// in the same file, so the two cannot drift apart again.
    #[test]
    fn skill_get_schema_does_not_advertise_an_owner_argument() {
        let service = fully_wired_service();
        let def = service
            .tool_definitions()
            .into_iter()
            .find(|d| d.name == TOOL_SKILL_GET)
            .expect("builtin_skill_get is advertised");
        let properties = def.parameters["properties"]
            .as_object()
            .expect("object schema with a properties map");
        assert!(
            !properties.contains_key("owner"),
            "builtin_skill_get's schema must not advertise an owner argument: the handler \
             resolves scope itself and never reads one, so an advertised property here would \
             be a contract callers cannot rely on"
        );
    }

    /// A service with *every* capability closure wired, so `tool_definitions()`
    /// emits the maximal builtin set.
    ///
    /// Why: the capability-gated tools (`builtin_notify`, the skill tools,
    /// `builtin_scratchpad_pin`) are invisible to a bare `new()` service, and a
    /// drift guard that walks a bare service therefore cannot see the very
    /// tools most likely to drift. Wire a new capability here the moment you
    /// add one, or the guards below stop covering it.
    fn fully_wired_service() -> BuiltinToolService {
        use desktop_assistant_core::ports::knowledge::KnowledgeListPage;

        let embed_fn: EmbedFn = Arc::new(|inputs: Vec<String>| {
            Box::pin(async move { Ok(inputs.iter().map(|_| vec![0.0f32; 3]).collect()) })
        });
        let kb_write: KnowledgeWriteFn = Arc::new(|entry| Box::pin(async move { Ok(entry) }));
        let kb_search: KnowledgeSearchFn =
            Arc::new(|_query, _emb, _model, _tags, _exclude_tags, _limit| {
                Box::pin(async {
                    Ok(KnowledgeSearchPage {
                        entries: Vec::new(),
                        scope_size: ScopeSize::None,
                        available_tags: Vec::new(),
                    })
                })
            });
        let kb_delete: KnowledgeDeleteFn = Arc::new(|_ids| Box::pin(async { Ok(0) }));
        let kb_list: KnowledgeListFn = Arc::new(|_query| {
            Box::pin(async {
                Ok(KnowledgeListPage {
                    entries: Vec::new(),
                    next_cursor: None,
                })
            })
        });
        let kb_get: KnowledgeGetFn = Arc::new(|_id| Box::pin(async { Ok(None) }));
        let tool_search: ToolSearchFn =
            Arc::new(|_query, _emb, _limit| Box::pin(async { Ok(Vec::new()) }));
        let tool_def: ToolDefinitionFn = Arc::new(|_name| Box::pin(async { Ok(None) }));
        let db_query: DbQueryFn =
            Arc::new(|_sql, _limit| Box::pin(async { Ok(serde_json::json!({"rows": []})) }));
        let conv_search: ConversationSearchFn =
            Arc::new(|_query, _limit, _role| Box::pin(async { Ok(Vec::new()) }));
        let notify: NotifyFn =
            Arc::new(|_summary, _body, _urgency| Box::pin(async { Ok(Some(1u32)) }));
        let skill_search: SkillSearchFn =
            Arc::new(|_query, _emb, _model, _limit| Box::pin(async { Ok(Vec::new()) }));
        let skill_get: SkillGetFn = Arc::new(|_name, _owner| Box::pin(async { Ok(None) }));

        // The scratchpad closures (including the pin write) come from the
        // shared in-memory pad so the pin tool is wired the same way the daemon
        // wires it.
        let mut service = scratchpad_service()
            .0
            .with_embedding(embed_fn, "test-embed-model".to_string())
            .with_knowledge_base(kb_write, kb_search, kb_delete, kb_list, kb_get)
            .with_tool_registry(tool_search, tool_def)
            .with_database(db_query)
            .with_conversation_search(conv_search)
            .with_notify(notify)
            .with_skills(skill_search, skill_get);
        service.set_mcp_control(crate::executor::McpToolExecutor::new(Vec::new()).control_handle());
        service
    }

    #[test]
    fn scratchpad_pin_is_routable_not_an_unknown_tool() {
        // #725: the pin tool is advertised whenever the pin write is wired, and
        // the always-present system prompt tells the model to call it, so the
        // gate the executor consults before falling through to MCP routing
        // (`supports_tool`) has to claim it, or every call answers
        // "unknown tool" and pinning is dead.
        let service = fully_wired_service();
        assert!(
            service
                .tool_definitions()
                .iter()
                .any(|def| def.name == TOOL_SCRATCHPAD_PIN),
            "premise: wiring the pin write advertises {TOOL_SCRATCHPAD_PIN}"
        );
        assert!(
            BuiltinToolService::supports_tool(TOOL_SCRATCHPAD_PIN),
            "{TOOL_SCRATCHPAD_PIN} is advertised to the model but supports_tool() rejects it - \
             the executor would route it to MCP and answer \"unknown tool\""
        );
    }

    #[tokio::test]
    async fn every_advertised_builtin_is_routable() {
        // Regression guard: a tool that `tool_definitions()` advertises but
        // `supports_tool()` doesn't recognize gets routed to MCP at execution
        // and fails with "unknown tool" (this bit builtin_knowledge_base_list,
        // builtin_notify, and builtin_scratchpad_pin). Every capability is
        // wired so the guard sees the capability-gated tools too.
        let service = fully_wired_service();
        for def in service.tool_definitions() {
            assert!(
                BuiltinToolService::supports_tool(&def.name),
                "tool '{}' is advertised by tool_definitions() but supports_tool() rejects it — \
                 it would fail to route at execution time",
                def.name
            );

            // Routing is only half of it: dispatch must also have an arm for
            // the name. Arguments are deliberately empty: what is asserted is
            // that the call lands on the tool's own implementation (which then
            // validates them) and not on `execute_tool`'s catch-all.
            let outcome = service.execute_tool(&def.name, serde_json::json!({})).await;
            if let Err(CoreError::ToolExecution(message)) = &outcome {
                assert!(
                    !message.contains("unknown built-in tool"),
                    "tool '{}' is advertised but execute_tool() has no arm for it: {message}",
                    def.name
                );
            }
        }
    }

    #[test]
    fn all_tool_names_matches_the_advertised_builtin_set() {
        // ALL_TOOL_NAMES is the one list routing is derived from, so it must be
        // exactly what a fully-wired service advertises. A name only in the
        // list routes a tool that does not exist; a name only in
        // `tool_definitions()` is advertised without being routable.
        use std::collections::BTreeSet;

        let advertised: BTreeSet<String> = fully_wired_service()
            .tool_definitions()
            .into_iter()
            .map(|def| def.name)
            .collect();
        let listed: BTreeSet<String> = BuiltinToolService::ALL_TOOL_NAMES
            .iter()
            .map(|name| (*name).to_string())
            .collect();
        assert_eq!(
            advertised, listed,
            "ALL_TOOL_NAMES and the advertised builtin set have drifted"
        );
    }

    #[test]
    fn supports_tool_rejects_names_that_are_not_builtins() {
        // The gate is exact-match: a name it wrongly claims is swallowed by the
        // builtin service instead of reaching the MCP server that owns it.
        for name in [
            "",
            "   ",
            "builtin_",
            "builtin_scratchpad",
            "builtin_scratchpad_pinned",
            "BUILTIN_SCRATCHPAD_PIN",
            "scratchpad_pin",
            "fileio__read_file",
        ] {
            assert!(
                !BuiltinToolService::supports_tool(name),
                "'{name}' is not a builtin, but supports_tool() claimed it"
            );
        }
    }

    #[tokio::test]
    async fn notify_absent_and_errors_without_capability() {
        let service = BuiltinToolService::new();
        // Not advertised when no notification capability is wired.
        assert!(
            !service
                .tool_definitions()
                .iter()
                .any(|t| t.name == TOOL_NOTIFY)
        );
        // Calling it anyway is a clean error, not a panic.
        let err = service
            .execute_tool(TOOL_NOTIFY, serde_json::json!({"summary": "hi"}))
            .await;
        assert!(matches!(err, Err(CoreError::ToolExecution(_))));
    }

    #[tokio::test]
    async fn notify_with_closure_reports_shown_and_suppressed() {
        use std::sync::Arc;

        // Returns an id for "show me", None for "duplicate" — keyed off summary.
        let notify_fn: NotifyFn = Arc::new(|summary, _body, _urgency| {
            Box::pin(async move {
                if summary == "dup" {
                    Ok(None)
                } else {
                    Ok(Some(42u32))
                }
            })
        });
        let service = BuiltinToolService::new().with_notify(notify_fn);

        // Advertised once wired.
        assert!(
            service
                .tool_definitions()
                .iter()
                .any(|t| t.name == TOOL_NOTIFY)
        );

        // summary is required.
        assert!(
            service
                .execute_tool(TOOL_NOTIFY, serde_json::json!({"body": "no summary"}))
                .await
                .is_err()
        );

        let shown = service
            .execute_tool(
                TOOL_NOTIFY,
                serde_json::json!({"summary": "Build done", "urgency": "low"}),
            )
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&shown).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(json["shown"], true);
        assert_eq!(json["id"], 42);

        let suppressed = service
            .execute_tool(TOOL_NOTIFY, serde_json::json!({"summary": "dup"}))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&suppressed).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(json["shown"], false);
    }

    #[tokio::test(start_paused = true)]
    async fn embedding_timeout_falls_back_to_empty_embedding() {
        // A wedged embedding backend (a never-completing future, like a stuck
        // Ollama) must not hang the search: `embed_text` times out after
        // EMBED_TIMEOUT and the search runs with an empty embedding, which the
        // store turns into an FTS-only query. With the clock paused, the 5s
        // timeout elapses immediately so the test is instant.
        use desktop_assistant_core::domain::ToolDefinition;
        use desktop_assistant_core::ports::embedding::EmbedFn;
        use std::sync::{Arc, Mutex};

        let embed_fn: EmbedFn = Arc::new(|_texts| Box::pin(std::future::pending()));

        // Capture the embedding the search closure is handed.
        let seen: Arc<Mutex<Option<Vec<f32>>>> = Arc::new(Mutex::new(None));
        let seen_w = Arc::clone(&seen);
        let search_fn: ToolSearchFn = Arc::new(move |_query, emb, _limit| {
            *seen_w.lock().unwrap() = Some(emb);
            Box::pin(async {
                Ok(vec![ToolDefinition::new(
                    "weather__forecast",
                    "Get the forecast",
                    serde_json::json!({}),
                )])
            })
        });
        let def_fn: ToolDefinitionFn = Arc::new(|_name| Box::pin(async { Ok(None) }));

        let service = BuiltinToolService::new()
            .with_embedding(embed_fn, "test-model".to_string())
            .with_tool_registry(search_fn, def_fn);

        let result = service
            .execute_tool(
                TOOL_SEARCH,
                serde_json::json!({"query": "weather forecast"}),
            )
            .await
            .expect("tool search must return, not hang, when embedding times out");
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(json["tools"].as_array().unwrap().len(), 1);
        assert_eq!(
            seen.lock().unwrap().as_ref().expect("search ran").len(),
            0,
            "a timed-out embedding must yield an empty vector so the store falls back to FTS"
        );
    }
}

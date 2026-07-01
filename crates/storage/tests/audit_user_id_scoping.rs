//! Static audit (#105): every SQL query that touches a personal-data
//! table must scope by `user_id`.
//!
//! The acceptance criterion for #105 reads:
//!
//! > Audit pass: no SQL query reaches the DB without `user_id` in its
//! > `WHERE`.
//!
//! A live `EXPLAIN`-based check would be more rigorous, but it requires
//! a Postgres instance and would only catch queries we actually
//! exercised in CI. This static scan reads every `crates/storage/src/`
//! file, finds every `sqlx::query*` call, extracts the SQL string,
//! identifies which personal-data tables it touches, and asserts that
//! the SQL mentions `user_id`. If a query targets a personal-data
//! table without saying `user_id`, the test names the file/line and
//! the offending SQL fragment.
//!
//! ## What counts as a personal-data table
//!
//! The list mirrors the columns added by migration
//! `016_multi_tenant_user_id.sql`:
//!
//! - `conversations`
//! - `messages`
//! - `knowledge_base`
//! - `message_summaries`
//! - `dreaming_watermarks`
//! - `tag_registry`
//!
//! Tables that are deliberately cross-user (system-wide) are exempt:
//!
//! - `tool_definitions` — registered tools live at the daemon scope,
//!   not the user scope. The architecture-evolution doc (#7) notes
//!   that stdio MCPs are single-tenant-only; multi-tenant tool
//!   registration is a separate follow-up.
//! - `dreaming_watermarks` *queries from the dreaming worker* are
//!   per-conversation: the worker iterates over conversations and
//!   inherits each conversation's `user_id` via the JOIN. The audit
//!   still requires the WHERE to mention `user_id` because the worker
//!   loops should include it as defense-in-depth.
//!
//! ## What the scan ignores
//!
//! - Migration files (`migrations/*.sql`) — those are the DDL that
//!   creates the columns; they can't reference them in a WHERE.
//! - `information_schema.*` and `pg_*` queries — Postgres catalogs are
//!   shared.
//! - DDL-style statements (`CREATE`, `ALTER`, `SET`) — they don't
//!   address user data.
//! - The scratch-schema execution path in `database.rs` — it runs
//!   user-authored read-only SQL inside a transaction that's already
//!   per-user-scoped (see `execute_database_query`'s contract).
//!
//! ## When the test fails
//!
//! The error message names the table, the file, the line number, and
//! the SQL fragment that doesn't mention `user_id`. Either add
//! `user_id = $N AND …` (or include `user_id` in the INSERT column
//! list), or — if the query genuinely needs to be cross-user — add an
//! entry to the documented allowlist below with a one-line rationale.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Personal-data tables — every SQL targeting these must scope by
/// `user_id`. Consumes the single source of truth exported by the storage
/// crate so this audit's scan set can never drift from the db_query tool's
/// grafting set. That drift is exactly what left `turns` and
/// `idempotency_keys` reachable cross-tenant (#431): this list used to
/// keep its own copy that omitted them.
fn personal_data_tables() -> &'static [&'static str] {
    desktop_assistant_storage::personal_data_tables()
}

/// Allowlist for queries that legitimately span multiple users.
///
/// Each entry is `(file, line, reason)` — the line is the line where
/// the `sqlx::query…(` call begins, matching the diagnostic the audit
/// emits. Keep this small; prefer fixing the query.
struct AllowedCrossUserQuery {
    file: &'static str,
    line_hint: usize,
    /// Reason this query is exempt — read by humans reviewing the
    /// allowlist, not consumed by the test runtime. `#[allow(dead_code)]`
    /// keeps the field as code-as-documentation without warnings.
    #[allow(dead_code)]
    rationale: &'static str,
}

const ALLOWED_CROSS_USER_QUERIES: &[AllowedCrossUserQuery] = &[
    // Background worker startup probes — count rows across the whole
    // table to decide whether to run the legacy JSON migration. Pre-
    // multi-tenant data has already been backfilled to the sentinel
    // user_id; this is a one-shot bootstrap helper, not a per-request
    // path.
    AllowedCrossUserQuery {
        file: "src/migrate_json.rs",
        line_hint: 0,
        rationale: "is_*_table_empty: bootstrap-only COUNT(*) probe \
                    used to decide whether to run JSON->Postgres \
                    migration; runs once at daemon startup, not in any \
                    request path.",
    },
    // Embedding backfill is a daemon-wide background worker (#74) that
    // iterates over every row regardless of user_id; the model-stamp
    // invariant it enforces is system-wide. Each per-row write inherits
    // the row's existing user_id (the column is not modified).
    AllowedCrossUserQuery {
        file: "src/embedding_backfill.rs",
        line_hint: 0,
        rationale: "embedding backfill is a daemon-wide background \
                    worker; it iterates all rows by design and \
                    preserves the existing user_id on each row.",
    },
    // Dreaming archival, when invoked without a per-user task-local
    // scope, archives across all users (single-tenant degenerate
    // case). When called from within a per-user consolidation cycle
    // it takes the scoped branch which DOES filter by user_id; the
    // unscoped branch lives next to it in the same file.
    AllowedCrossUserQuery {
        file: "src/dreaming/archival.rs",
        line_hint: 0,
        rationale: "archival sweep with no per-user scope installed; \
                    single-tenant fallback that archives across the \
                    sentinel user_id partition. A scoped branch \
                    immediately follows.",
    },
    // Dreaming background-worker cross-user scans. These are the only
    // queries in the worker that intentionally cross tenancy — the
    // worker groups results by user_id and installs a per-user
    // task-local scope before any subsequent SQL.
    AllowedCrossUserQuery {
        file: "src/dreaming/common.rs",
        line_hint: 0,
        rationale: "find_conversations_with_new_messages: dreaming \
                    worker entry point. Returns rows including user_id \
                    so the worker can install a per-user scope.",
    },
    AllowedCrossUserQuery {
        file: "src/dreaming/consolidation.rs",
        line_hint: 0,
        rationale: "load_entries_needing_review_by_user: dreaming \
                    consolidation entry point. Returns rows grouped \
                    by user_id; the worker installs a per-user \
                    scope before processing each group.",
    },
];

/// Result of scanning a single SQL fragment.
#[derive(Debug)]
struct Finding {
    file: PathBuf,
    line: usize,
    tables: Vec<String>,
    sql_excerpt: String,
}

#[test]
fn every_storage_query_targeting_a_personal_data_table_includes_user_id() {
    let storage_src = storage_src_root();
    let mut files: Vec<PathBuf> = Vec::new();
    walk_rust_files(&storage_src, &mut files);
    assert!(
        !files.is_empty(),
        "audit: expected to find Rust files under {}",
        storage_src.display()
    );

    let mut findings: Vec<Finding> = Vec::new();
    for path in &files {
        let content = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        for site in extract_query_sites(&content) {
            // Skip queries whose SQL doesn't touch any personal-data
            // table (system tables, catalogs, settings, etc.).
            let tables = personal_tables_referenced(&site.sql);
            if tables.is_empty() {
                continue;
            }
            // Pass: query mentions user_id, however it's spelled. We're
            // lenient on the exact form so a `WHERE user_id = $1` or an
            // INSERT column list of `(id, user_id, …)` or a JOIN's
            // `USING (user_id, …)` all count.
            if sql_mentions_user_id(&site.sql) {
                continue;
            }
            // Pass: the call site is in the documented allowlist.
            let rel = path.strip_prefix(crate_root()).unwrap_or(path);
            if is_allowed(rel, site.line) {
                continue;
            }
            findings.push(Finding {
                file: rel.to_path_buf(),
                line: site.line,
                tables,
                sql_excerpt: shrink_excerpt(&site.sql),
            });
        }
    }

    if !findings.is_empty() {
        let mut msg = String::from(
            "audit: SQL queries against personal-data tables must scope by `user_id` (#105).\n\n",
        );
        for f in &findings {
            msg.push_str(&format!(
                "  {}:{}\n    tables: {}\n    sql: {}\n\n",
                f.file.display(),
                f.line,
                f.tables.join(", "),
                f.sql_excerpt,
            ));
        }
        msg.push_str(
            "If a query is legitimately cross-user (background worker, system table, \
             startup probe), add it to ALLOWED_CROSS_USER_QUERIES in this test with a \
             one-line rationale. Otherwise add `user_id` to the WHERE clause or \
             INSERT column list and bind `current_user_id()`.",
        );
        panic!("{msg}");
    }
}

/// #431 drift-guard. The audit now consumes the storage crate's canonical
/// personal-data list, so the db_query tool's grafting set and this scan's
/// set are the same list by construction. This pins that the canonical
/// list still contains the tables whose omission caused the cross-tenant
/// hole — a regression that dropped them (or re-introduced the
/// `turn_state` filename typo) fails here loudly.
#[test]
fn assert_personal_tables_match_audit() {
    let canonical = personal_data_tables();
    for required in ["turns", "idempotency_keys", "background_tasks", "scratchpads"] {
        assert!(
            canonical.contains(&required),
            "canonical personal-data list must include `{required}`; dropping it \
             unscopes that table in the db_query tool"
        );
    }
    assert!(
        !canonical.contains(&"turn_state"),
        "`turn_state` is a migration filename, not a table — the real table is `turns`"
    );
}

// ---------- helpers ---------------------------------------------------------

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn storage_src_root() -> PathBuf {
    crate_root().join("src")
}

fn walk_rust_files(dir: &Path, acc: &mut Vec<PathBuf>) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in read.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_rust_files(&path, acc);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            acc.push(path);
        }
    }
}

/// A single `sqlx::query…(` call site discovered by the scan.
#[derive(Debug)]
struct QuerySite {
    line: usize,
    sql: String,
}

/// Walk `content` and pull out every SQL string passed to a
/// `sqlx::query…(` call. The extractor is line-aware (so we can report
/// the line of the call) and handles multi-line raw strings.
///
/// Implementation note: `content` may contain non-ASCII (em-dashes in
/// doc comments, Unicode quote marks, ...). We use `match_indices`
/// rather than byte-stepping with `i += 1` so we never land inside a
/// multi-byte UTF-8 code-point.
fn extract_query_sites(content: &str) -> Vec<QuerySite> {
    let mut sites = Vec::new();
    let bytes = content.as_bytes();
    for (i, _) in content.match_indices("sqlx::query") {
        // Advance past the method name and any `_as` / `_scalar` /
        // generic args, up to the `(` that opens the call. ASCII-only
        // characters in this range, so byte stepping is safe.
        let mut j = i;
        while j < bytes.len() && bytes[j] != b'(' {
            // If we hit a non-paren that means this isn't a call
            // expression (e.g. `use sqlx::query` import line) — bail
            // on excessive lookahead.
            if j - i > 64 {
                break;
            }
            j += 1;
        }
        if j >= bytes.len() || bytes[j] != b'(' {
            continue;
        }
        // After `(`, the first non-whitespace argument should be a
        // string literal — `"..."`, `r#"..."#`, or some other expression
        // we don't recognise (in which case we skip this call site).
        let arg_start = skip_whitespace(content, j + 1);
        if let Some(sql) = extract_string_literal(content, arg_start) {
            sites.push(QuerySite {
                line: line_of(content, i),
                sql,
            });
        }
    }
    sites
}

fn skip_whitespace(s: &str, mut i: usize) -> usize {
    let bytes = s.as_bytes();
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\n' || bytes[i] == b'\t') {
        i += 1;
    }
    i
}

fn extract_string_literal(content: &str, start: usize) -> Option<String> {
    let bytes = content.as_bytes();
    if start >= bytes.len() {
        return None;
    }
    // Handle `r#"..."#` and `r##"..."##`. The opening sequence is
    // pure-ASCII so byte indexing is safe up to the body.
    if bytes[start] == b'r' {
        let mut hash_count = 0;
        let mut k = start + 1;
        while k < bytes.len() && bytes[k] == b'#' {
            hash_count += 1;
            k += 1;
        }
        if k < bytes.len() && bytes[k] == b'"' {
            // Body starts at k+1; ends at the first `"<hashes>` of the
            // matching hash count. `content[..]` slice is UTF-8 safe.
            let body_start = k + 1;
            let mut needle = String::from("\"");
            for _ in 0..hash_count {
                needle.push('#');
            }
            if let Some(rel_end) = content[body_start..].find(&needle) {
                return Some(content[body_start..body_start + rel_end].to_string());
            }
            return None;
        }
        return None;
    }
    if bytes[start] != b'"' {
        return None;
    }
    // Plain `"..."` literal. The literal body may contain multi-byte
    // characters, so we step by char-indices on the slice after the
    // opening quote, watching for unescaped `"` to terminate.
    let body_start = start + 1;
    let tail = &content[body_start..];
    let mut chars = tail.char_indices().peekable();
    while let Some((idx, c)) = chars.next() {
        if c == '\\' {
            // Skip the next escaped char regardless of width.
            chars.next();
            continue;
        }
        if c == '"' {
            return Some(tail[..idx].to_string());
        }
    }
    None
}

fn line_of(content: &str, byte_offset: usize) -> usize {
    1 + content[..byte_offset.min(content.len())]
        .matches('\n')
        .count()
}

/// Return the personal-data tables that the SQL fragment references.
/// Matches are case-insensitive whole-word — `messages` matches but
/// `message_summaries` does not when looking for `messages` alone.
fn personal_tables_referenced(sql: &str) -> Vec<String> {
    let mut found: BTreeSet<&'static str> = BTreeSet::new();
    let lower = sql.to_ascii_lowercase();
    for tbl in personal_data_tables() {
        if contains_whole_word(&lower, tbl) {
            found.insert(*tbl);
        }
    }
    found.into_iter().map(String::from).collect()
}

fn contains_whole_word(haystack: &str, needle: &str) -> bool {
    let bytes = haystack.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() {
        return false;
    }
    let mut i = 0;
    while i + n.len() <= bytes.len() {
        if &bytes[i..i + n.len()] == n {
            let before_ok = i == 0 || !is_ident_char(bytes[i - 1]);
            let after_ok = i + n.len() == bytes.len() || !is_ident_char(bytes[i + n.len()]);
            if before_ok && after_ok {
                return true;
            }
        }
        i += 1;
    }
    false
}

fn is_ident_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

fn sql_mentions_user_id(sql: &str) -> bool {
    sql.to_ascii_lowercase().contains("user_id")
}

fn is_allowed(file: &Path, line: usize) -> bool {
    let file_str = file.to_string_lossy();
    ALLOWED_CROSS_USER_QUERIES.iter().any(|allowed| {
        file_str.replace('\\', "/").ends_with(allowed.file)
            && (allowed.line_hint == 0 || allowed.line_hint == line)
    })
}

fn shrink_excerpt(sql: &str) -> String {
    let one_line = sql
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    // Avoid slicing inside multi-byte chars when the excerpt contains
    // non-ASCII (e.g. an em-dash in a code comment). `char_indices`
    // gives us a safe truncation point.
    if one_line.chars().count() > 160 {
        let cutoff = one_line
            .char_indices()
            .nth(160)
            .map(|(i, _)| i)
            .unwrap_or(one_line.len());
        format!("{}...", &one_line[..cutoff])
    } else {
        one_line
    }
}

// ---------- self-tests of the scanner --------------------------------------

#[test]
fn scanner_finds_query_sites() {
    let src = "\
        let row = sqlx::query(\"SELECT 1 FROM conversations WHERE id = $1\")\n\
            .bind(&id)\n\
            .fetch_one(&pool);\n\
        let row = sqlx::query_as::<_, MyRow>(\n\
            r#\"SELECT id FROM messages WHERE conversation_id = $1\"#\n\
        )\n\
            .bind(&cid)\n\
            .fetch_one(&pool);\n";
    let sites = extract_query_sites(src);
    // The Rust source above uses both the plain "..." and the raw
    // form, so we expect both to be detected.
    assert_eq!(sites.len(), 2, "{sites:?}");
    assert!(sites[0].sql.contains("conversations"));
    assert!(sites[1].sql.contains("messages"));
}

#[test]
fn personal_tables_distinguishes_messages_from_message_summaries() {
    let q = "SELECT id FROM message_summaries WHERE id = $1";
    let found = personal_tables_referenced(q);
    assert!(found.contains(&"message_summaries".to_string()));
    assert!(!found.contains(&"messages".to_string()));
}

#[test]
fn sql_with_user_id_in_where_passes_the_mention_check() {
    assert!(sql_mentions_user_id(
        "SELECT * FROM conversations WHERE user_id = $1 AND id = $2"
    ));
    assert!(sql_mentions_user_id(
        "INSERT INTO messages (id, user_id, content) VALUES ($1, $2, $3)"
    ));
    assert!(!sql_mentions_user_id("SELECT * FROM conversations c"));
}

#[test]
fn scanner_extracts_multiline_raw_string_correctly() {
    let src = "\
        sqlx::query(\n\
            r#\"SELECT id, content FROM knowledge_base\n\
               WHERE user_id = $1 AND tags && $2\"#,\n\
        )\n";
    let sites = extract_query_sites(src);
    assert_eq!(sites.len(), 1);
    assert!(sites[0].sql.contains("knowledge_base"));
    assert!(sites[0].sql.contains("user_id"));
}

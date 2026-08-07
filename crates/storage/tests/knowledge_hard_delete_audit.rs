//! Static audit (#1122): the knowledge table is destroyed in exactly one
//! place.
//!
//! The deletion safety flag is only worth having if a path added later
//! inherits it. A guard written into each call site does not have that
//! property, because the next call site is written by someone who never read
//! the guard. So the rule is structural: every SQL statement that removes a
//! `knowledge_base` row lives in one module, and this scan fails when a second
//! one appears.
//!
//! The scan reads every `.rs` file under each workspace crate's `src`
//! directory and every `.sql` file under its `migrations` directory, folds
//! away Rust string continuations and repeated whitespace, and looks for a
//! destructive verb whose operand is the knowledge table. Anything found
//! outside the guard module is reported with its file and line.
//!
//! Migrations are in scope because they run automatically on the next boot of
//! every daemon. A one-time repair that reclaims disk space by deleting old
//! tombstones would reach past the policy exactly the way a new Rust call site
//! would.
//!
//! A verb whose operand is a Rust interpolation - `DELETE FROM {table}` - is
//! reported too, whatever the table turns out to be at run time. The scanner
//! cannot tell, and a generic per-table sweep is the natural refactor that
//! would otherwise carry the knowledge table out of reach of this test.
//!
//! Test code is out of scope. A test may write any SQL it likes, including the
//! statements the `db_query` tool must refuse, and no test is a production
//! deletion path.
//!
//! ## What it does not reach
//!
//! SQL assembled from pieces that never sit next to each other in one file.
//! The workspace has no query builder and forbids string-built SQL, so the
//! remaining route is a deliberate one, not a slip.
//!
//! This runs without a database, so `just check` covers it.
//!
//! ## When this test fails
//!
//! Route the new statement through
//! `desktop_assistant_storage::knowledge_delete::hard_delete_knowledge`, which
//! reads who asked for the delete and applies the configured policy. Add a
//! target variant there when the predicate you need is not yet expressed.

use std::path::{Path, PathBuf};

/// The one module allowed to destroy a knowledge row, as a path suffix.
const GUARD_MODULE: &str = "storage/src/knowledge_delete.rs";

/// SQL verbs that can remove rows for good. A soft delete is an `UPDATE`, so it
/// is deliberately absent.
///
/// `merge into` is here because Postgres 15 added a bare `DELETE` action to
/// `MERGE`, which carries no `FROM` and so never contains the first entry. A
/// merge that only inserts and updates is caught as well; the scanner cannot
/// read the action list, and a `MERGE` against this table is close enough to a
/// delete to belong beside one.
const DESTRUCTIVE_VERBS: &[&str] = &["delete from", "truncate", "drop table", "merge into"];

/// The table this audit protects. A schema-qualified `public.knowledge_base`
/// matches too, because the qualifier is not an identifier character.
const TABLE: &str = "knowledge_base";

/// How many following lines are joined to the line under test, so a statement
/// broken over a continued string literal is still read as one statement.
///
/// Five is generous: in SQL the operand follows its verb immediately, so the
/// two are never far apart however the literal is wrapped.
const LOOKAHEAD_LINES: usize = 5;

#[test]
fn unguarded_delete_against_the_knowledge_table_fails_the_audit() {
    let files = scanned_files();
    assert!(
        files.len() > 100,
        "the scan found only {} files; it is not reaching the workspace",
        files.len()
    );
    assert!(
        files
            .iter()
            .any(|f| f.extension().is_some_and(|e| e == "sql")),
        "the scan found no migrations; a repair migration would slip past it"
    );

    let mut offenders: Vec<String> = Vec::new();
    for file in &files {
        if is_guard_module(file) {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(file) else {
            continue;
        };
        for hit in destructive_statements(&content) {
            offenders.push(format!("{}:{}: {}", file.display(), hit.line, hit.excerpt));
        }
    }

    assert!(
        offenders.is_empty(),
        "a knowledge row may only be destroyed by \
         `knowledge_delete::hard_delete_knowledge`, which applies the configured \
         deletion policy. These statements bypass it:\n  {}",
        offenders.join("\n  ")
    );
}

/// The guard module must still hold the statements, or the audit above passes
/// because the capability moved rather than because it is contained.
#[test]
fn the_guard_module_holds_the_only_knowledge_delete() {
    let guard = crates_root().join(GUARD_MODULE);
    let content =
        std::fs::read_to_string(&guard).unwrap_or_else(|e| panic!("read {}: {e}", guard.display()));
    assert!(
        !destructive_statements(&content).is_empty(),
        "{} no longer holds a knowledge-base delete, so the audit would pass \
         over a codebase that had moved the statement elsewhere",
        guard.display()
    );
}

// ---------- scanner ---------------------------------------------------------

#[derive(Debug)]
struct Hit {
    line: usize,
    excerpt: String,
}

/// Every destructive statement against the knowledge table in `content`.
///
/// SQL here is written as a Rust literal broken over several lines with a
/// trailing backslash, so each line is tested together with the few that
/// follow it, normalized into one line. A hit is reported against the line the
/// verb starts on, which keeps one statement from being reported twice.
fn destructive_statements(content: &str) -> Vec<Hit> {
    let lines: Vec<&str> = content.lines().collect();
    let mut hits = Vec::new();

    for (index, line) in lines.iter().enumerate() {
        let head_len = normalize(line).len();
        if head_len == 0 {
            continue;
        }
        let end = (index + LOOKAHEAD_LINES).min(lines.len());
        let window = normalize(&lines[index..end].join("\n"));
        let lower = window.to_ascii_lowercase();

        for verb in DESTRUCTIVE_VERBS {
            for (at, _) in lower.match_indices(verb) {
                // Only count a statement once, against the line it starts on.
                if at >= head_len || !is_whole_word_at(&lower, at, verb) {
                    continue;
                }
                let reportable = match operand(&lower[at + verb.len()..]) {
                    Operand::KnowledgeTable => true,
                    // A Rust match arm reads as an interpolated operand -
                    // `Statement::Truncate { .. }` - so require the mark of a
                    // string literal before the verb. Every SQL statement has
                    // one; a match arm does not.
                    Operand::Interpolated => lower[..at].contains('"'),
                    Operand::Other => false,
                };
                if reportable {
                    hits.push(Hit {
                        line: index + 1,
                        excerpt: excerpt(&window, at),
                    });
                }
            }
        }
    }
    hits
}

/// Fold Rust string continuations away and collapse runs of whitespace to a
/// single space, so a statement written over several lines reads as one.
fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last_was_space = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        // A backslash at the end of a line continues a Rust string literal.
        // Drop it with the newline and the indentation that follows.
        if c == '\\' && matches!(chars.peek(), Some('\n' | '\r')) {
            while chars.peek().is_some_and(|c| c.is_whitespace()) {
                chars.next();
            }
            if !last_was_space && !out.is_empty() {
                out.push(' ');
                last_was_space = true;
            }
            continue;
        }
        if c.is_whitespace() {
            if !last_was_space && !out.is_empty() {
                out.push(' ');
                last_was_space = true;
            }
            continue;
        }
        out.push(c);
        last_was_space = false;
    }
    out.trim_end().to_string()
}

fn excerpt(normalized: &str, at: usize) -> String {
    normalized[at..].chars().take(120).collect()
}

/// Is the verb at `at` a word of its own, rather than part of a longer
/// identifier? Without this, `truncated` reads as `TRUNCATE`.
fn is_whole_word_at(text: &str, at: usize, verb: &str) -> bool {
    let before_ok = at == 0 || !is_ident_byte(text.as_bytes()[at - 1]);
    let end = at + verb.len();
    let after_ok = end == text.len() || !is_ident_byte(text.as_bytes()[end]);
    before_ok && after_ok
}

fn is_ident_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// What a destructive verb is aimed at.
#[derive(Debug, PartialEq)]
enum Operand {
    /// The knowledge table, named outright.
    KnowledgeTable,
    /// A Rust interpolation, so the table is decided at run time and could be
    /// the knowledge table.
    Interpolated,
    /// Some other table.
    Other,
}

/// Read the operand out of the text right after a destructive verb.
///
/// Only the next token counts, because the operand is the verb's own. A window
/// of a fixed size instead reads across the end of the SQL literal and reports
/// a delete against one table because a neighbouring literal mentions another.
fn operand(after_verb: &str) -> Operand {
    let mut tokens = after_verb.split_whitespace();
    let mut token = tokens.next();
    // Optional noise between the verb and its operand.
    while matches!(token, Some("only" | "if" | "exists" | "table")) {
        token = tokens.next();
    }
    let Some(token) = token else {
        return Operand::Other;
    };
    if token.contains('{') {
        return Operand::Interpolated;
    }
    let bare = token
        .trim_matches(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '.'))
        .rsplit('.')
        .next()
        .unwrap_or_default();
    if bare == TABLE {
        Operand::KnowledgeTable
    } else {
        Operand::Other
    }
}

fn is_guard_module(file: &Path) -> bool {
    file.to_string_lossy()
        .replace('\\', "/")
        .ends_with(GUARD_MODULE)
}

/// `crates/`, one level above this crate's manifest directory.
fn crates_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the storage crate sits under crates/")
        .to_path_buf()
}

/// Every `.rs` file under `crates/*/src` and every `.sql` file under
/// `crates/*/migrations`.
fn scanned_files() -> Vec<PathBuf> {
    let mut acc = Vec::new();
    let Ok(read) = std::fs::read_dir(crates_root()) else {
        return acc;
    };
    for entry in read.flatten() {
        walk_files(&entry.path().join("src"), "rs", &mut acc);
        walk_files(&entry.path().join("migrations"), "sql", &mut acc);
    }
    acc.sort();
    acc
}

fn walk_files(dir: &Path, extension: &str, acc: &mut Vec<PathBuf>) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in read.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_files(&path, extension, acc);
        } else if path.extension().and_then(|e| e.to_str()) == Some(extension) {
            acc.push(path);
        }
    }
}

// ---------- self-tests of the scanner --------------------------------------

#[test]
fn scanner_finds_a_single_line_delete() {
    let src = r#"sqlx::query("DELETE FROM knowledge_base WHERE id = $1")"#;
    assert_eq!(destructive_statements(src).len(), 1);
}

#[test]
fn scanner_finds_a_delete_split_over_continued_lines() {
    let src = "sqlx::query(\n    \"DELETE FROM knowledge_base \\\n     WHERE user_id = $1\",\n)";
    assert_eq!(
        destructive_statements(src).len(),
        1,
        "a continued string literal must not hide the statement"
    );
}

#[test]
fn scanner_reports_one_hit_per_statement() {
    let src = "\"DELETE FROM knowledge_base WHERE id = $1\"\n// a following line\n";
    assert_eq!(destructive_statements(src).len(), 1);
}

#[test]
fn scanner_finds_a_merge_that_may_carry_a_delete_action() {
    let src = "\"MERGE INTO knowledge_base USING plan ON plan.id = knowledge_base.id\"";
    assert_eq!(destructive_statements(src).len(), 1);
}

#[test]
fn scanner_finds_a_delete_against_a_table_chosen_at_run_time() {
    let src = r#"sqlx::query(&format!("DELETE FROM {table} WHERE id = $1"))"#;
    assert_eq!(
        destructive_statements(src).len(),
        1,
        "a table name decided at run time could be the knowledge table"
    );
}

#[test]
fn scanner_ignores_a_delete_against_another_table() {
    let src = r#"sqlx::query("DELETE FROM scratchpads WHERE id = $1")"#;
    assert!(destructive_statements(src).is_empty());
}

#[test]
fn scanner_ignores_a_verb_that_is_part_of_a_longer_word() {
    let src = r#"format!("{truncated}-{suffix}")"#;
    assert!(destructive_statements(src).is_empty());
}

#[test]
fn scanner_ignores_a_rust_match_arm_that_names_a_statement_kind() {
    let src = r#"Statement::Truncate { .. } => "TRUNCATE","#;
    assert!(
        destructive_statements(src).is_empty(),
        "a match arm is not a statement against a table"
    );
}

#[test]
fn scanner_ignores_a_soft_delete() {
    let src = r#"sqlx::query("UPDATE knowledge_base SET deleted_at = NOW() WHERE id = $1")"#;
    assert!(destructive_statements(src).is_empty());
}

#[test]
fn scanner_ignores_the_knowledge_table_named_by_a_neighbouring_statement() {
    let src = "\"DELETE FROM messages WHERE 1=1\",\n\"INSERT INTO knowledge_base VALUES (1)\",";
    assert!(
        destructive_statements(src).is_empty(),
        "the table must be the verb's own operand"
    );
}

#[test]
fn scanner_finds_a_truncate_and_a_schema_qualified_delete() {
    assert_eq!(
        destructive_statements("\"TRUNCATE knowledge_base\"").len(),
        1
    );
    assert_eq!(
        destructive_statements("\"DELETE FROM public.knowledge_base WHERE id = $1\"").len(),
        1
    );
}

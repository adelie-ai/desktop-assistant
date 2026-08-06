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
//! directory, folds away Rust string continuations and repeated whitespace,
//! and looks for a `DELETE`, `TRUNCATE` or `DROP TABLE` that names the
//! knowledge table. Anything found outside the guard module is reported with
//! its file and line.
//!
//! Test code is out of scope. A test may write any SQL it likes, including the
//! statements the `db_query` tool must refuse, and no test is a production
//! deletion path.
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

/// SQL verbs that remove rows for good. A soft delete is an `UPDATE`, so it is
/// deliberately absent.
const DESTRUCTIVE_VERBS: &[&str] = &["delete from", "truncate", "drop table"];

/// The table this audit protects. A schema-qualified `public.knowledge_base`
/// matches too, because the qualifier is not an identifier character.
const TABLE: &str = "knowledge_base";

/// How many following lines are joined to the line under test, so a statement
/// broken over a continued string literal is still read as one statement.
const LOOKAHEAD_LINES: usize = 5;

#[test]
fn unguarded_delete_against_the_knowledge_table_fails_the_audit() {
    let files = crate_source_files();
    assert!(
        files.len() > 100,
        "the scan found only {} source files; it is not reaching the workspace",
        files.len()
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
                if at >= head_len {
                    continue;
                }
                if !names_the_table(&lower[at + verb.len()..]) {
                    continue;
                }
                hits.push(Hit {
                    line: index + 1,
                    excerpt: excerpt(&window, at),
                });
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

/// Does the text right after a destructive verb name the knowledge table?
///
/// The table must be the verb's own operand, so only the next token counts. A
/// window of a fixed size instead reads across the end of the SQL literal and
/// reports a delete against one table because a neighbouring literal mentions
/// another.
fn names_the_table(after_verb: &str) -> bool {
    let mut tokens = after_verb.split_whitespace();
    let mut token = tokens.next();
    // Optional noise between the verb and its operand.
    while matches!(token, Some("only" | "if" | "exists" | "table")) {
        token = tokens.next();
    }
    let Some(token) = token else {
        return false;
    };
    let bare = token
        .trim_matches(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '.'))
        .rsplit('.')
        .next()
        .unwrap_or_default();
    bare == TABLE
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

/// Every `.rs` file under `crates/*/src`.
fn crate_source_files() -> Vec<PathBuf> {
    let mut acc = Vec::new();
    let Ok(read) = std::fs::read_dir(crates_root()) else {
        return acc;
    };
    for entry in read.flatten() {
        let src = entry.path().join("src");
        if src.is_dir() {
            walk_rust_files(&src, &mut acc);
        }
    }
    acc.sort();
    acc
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
fn scanner_ignores_a_delete_against_another_table() {
    let src = r#"sqlx::query("DELETE FROM scratchpads WHERE id = $1")"#;
    assert!(destructive_statements(src).is_empty());
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

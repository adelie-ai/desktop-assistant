use desktop_assistant_core::CoreError;
use sqlx::postgres::PgRow;
use sqlx::{Column, PgPool, Row, TypeInfo};

use sqlparser::ast::{
    BinaryOperator, Expr, Ident, ObjectName, Query, Select, SetExpr, Statement, TableFactor,
    TableWithJoins, Value, ValueWithSpan,
};
use sqlparser::dialect::PostgreSqlDialect;
use sqlparser::parser::Parser;

use crate::current_user_id;

// ---------------------------------------------------------------------------
// #141: security model for the LLM-facing `execute_database_query` tool.
//
// **Threat model.** A hostile LLM that knows the tool exists tries to
// (a) read another user's rows from personal-data tables, or (b)
// modify / drop personal-data tables via a qualified write that
// bypasses the `search_path TO scratch, public` redirect, or (c)
// stuff a second statement past the first-keyword classifier.
//
// **Defenses, applied in order before any text reaches the pool:**
//
// 1. Parse with `sqlparser` (PostgreSQL dialect). Any input that
//    isn't recognised as Postgres SQL is rejected at parse time.
// 2. Exactly one top-level statement. `SELECT 1; DROP TABLE …` is
//    refused.
// 3. Route by statement type:
//    - `Statement::Query` → SELECT path. Walk the AST; for each
//      `Select` node whose `FROM`/`JOIN` references a personal-data
//      table, graft a `<alias>.user_id = $N` predicate AND'd into the
//      `selection` (or installed as the new `selection` if none
//      existed). Bind `$N` to the caller's task-local `UserId`.
//      Tables without a `user_id` column (system catalogs,
//      `tool_definitions`, scratch tables) are passed through.
//    - Everything else → write path. Reject any reference (qualified
//      or otherwise) to a personal-data table by name. Otherwise
//      execute under `search_path TO scratch, public` so unqualified
//      DDL lands in `scratch`.
// 4. The read path additionally runs inside `SET TRANSACTION READ
//    ONLY`, as it always has — defense-in-depth in case a future
//    rewrite mistake re-introduces a write under a SELECT-shaped
//    statement (e.g. `WITH … DELETE … SELECT *`).
//
// The list of personal-data tables mirrors the one the static audit
// in `tests/audit_user_id_scoping.rs` enforces — they're the same
// tables migration `016_multi_tenant_user_id.sql` adds `user_id`
// columns to, plus `background_tasks` and `turn_state` which were
// added in 017 and 018 with their own `user_id` columns. Keep this
// list in sync with the audit's `PERSONAL_DATA_TABLES`; the
// `assert_personal_tables_match_audit` test below makes drift loud.

/// Personal-data tables — every reference to these in user-supplied
/// SQL must either be grafted with a `user_id = $N` predicate (read
/// path) or refused outright (write path). Names are lowercase; the
/// matcher is case-insensitive.
const PERSONAL_DATA_TABLES: &[&str] = &[
    "conversations",
    "messages",
    "knowledge_base",
    "message_summaries",
    "dreaming_watermarks",
    "tag_registry",
    // 017 + 018 — these also carry `user_id` columns and are written
    // by per-user code paths.
    "background_tasks",
    "turn_state",
];

/// Output of `prepare_select_for_user` — the rewritten SELECT plus the
/// caller's user_id ready to bind as `$1` when grafting added a
/// `user_id = $1` predicate. When no personal-data table was
/// referenced (e.g. `SELECT now()` or a query over
/// `information_schema`), `bound_user_id` is `None` and the SQL is
/// returned essentially unchanged.
pub(crate) struct PreparedSelect {
    pub sql: String,
    /// `Some(user_id)` when the rewriter grafted at least one
    /// `user_id = '<user_id>'` predicate; `None` when the query did
    /// not reference any personal-data table. Currently informational
    /// (the value is inlined as a SQL literal so there's no bind to
    /// perform), but exposed so a future migration to parameter-
    /// rebinding can use it.
    #[allow(dead_code)]
    pub bound_user_id: Option<String>,
}

/// Parse `sql` as a single SELECT and graft `user_id = $1` predicates
/// onto any references to personal-data tables, scoped to `user_id`.
///
/// Returns an error if the input is not a single statement, is not a
/// SELECT-shaped query, or fails to parse against the PostgreSQL
/// dialect. See the module-level threat model for the full contract.
pub(crate) fn prepare_select_for_user(
    sql: &str,
    user_id: &str,
) -> Result<PreparedSelect, CoreError> {
    let mut stmts = parse_one_or_more(sql)?;
    require_single_statement(&stmts)?;
    let stmt = stmts.pop().expect("require_single_statement guards");

    let mut query = match stmt {
        Statement::Query(q) => q,
        other => {
            return Err(reject(format!(
                "only SELECT statements are allowed on the read path; \
                 got `{}` — use a different builtin tool for writes",
                statement_kind(&other),
            )));
        }
    };

    let mut grafter = UserIdGrafter::new(user_id);
    grafter.visit_query(&mut query);

    // sqlparser's `Display` for `Query` round-trips the AST back to
    // canonical SQL. We don't reformat — Postgres parses it again.
    let sql_out = query.to_string();

    Ok(PreparedSelect {
        sql: sql_out,
        bound_user_id: grafter.bound.then(|| user_id.to_string()),
    })
}

/// Parse `sql` as a single non-SELECT statement and verify it does
/// not reference any personal-data table (qualified or otherwise).
///
/// Returns `Ok(())` for statements that are safe to dispatch through
/// the scratch-namespace write path. Returns an error for compound
/// inputs, parse failures, and any reference to a personal-data
/// table — at any depth — in the AST.
pub(crate) fn validate_write_statement(sql: &str) -> Result<(), CoreError> {
    let stmts = parse_one_or_more(sql)?;
    require_single_statement(&stmts)?;
    let stmt = &stmts[0];

    // If the parser produced a Query here, the caller should have
    // routed via `prepare_select_for_user` — but reject it loudly so
    // we don't silently lose the user_id scoping.
    if let Statement::Query(_) = stmt {
        return Err(reject(
            "SELECT-shaped statement reached the write path; this is a \
             routing bug — reads must go through `prepare_select_for_user`"
                .to_string(),
        ));
    }

    let mut finder = PersonalDataTableFinder::default();
    finder.visit_statement(stmt);
    if let Some(hit) = finder.first_hit {
        return Err(reject(format!(
            "write statement targets the personal-data table `{hit}`; the \
             write path is restricted to the `scratch` schema and other \
             user-defined namespaces. Use the read path (SELECT) to inspect \
             personal data."
        )));
    }
    Ok(())
}

/// Build the `ToolExecution` error all rejection paths share.
fn reject(msg: String) -> CoreError {
    CoreError::ToolExecution(msg)
}

/// Parse `sql` with the PostgreSQL dialect, mapping syntax errors to
/// `CoreError::ToolExecution` so the LLM gets a single consistent error
/// shape regardless of which leg of the pipeline rejected.
fn parse_one_or_more(sql: &str) -> Result<Vec<Statement>, CoreError> {
    Parser::parse_sql(&PostgreSqlDialect {}, sql)
        .map_err(|e| reject(format!("SQL parse error: {e}")))
}

fn require_single_statement(stmts: &[Statement]) -> Result<(), CoreError> {
    match stmts.len() {
        0 => Err(reject(
            "no statement found — expected exactly one SQL statement".to_string(),
        )),
        1 => Ok(()),
        n => Err(reject(format!(
            "compound input rejected: got {n} statements; this tool requires \
             a single SQL statement"
        ))),
    }
}

/// Friendly statement-type label for rejection messages. We don't
/// enumerate every variant — the common ones are enough to make the
/// LLM-facing error actionable.
fn statement_kind(stmt: &Statement) -> &'static str {
    match stmt {
        Statement::Insert(_) => "INSERT",
        Statement::Update(_) => "UPDATE",
        Statement::Delete(_) => "DELETE",
        Statement::Truncate { .. } => "TRUNCATE",
        Statement::Drop { .. } => "DROP",
        Statement::CreateTable(_) => "CREATE TABLE",
        Statement::CreateView(_) => "CREATE VIEW",
        Statement::CreateIndex(_) => "CREATE INDEX",
        Statement::CreateSchema { .. } => "CREATE SCHEMA",
        Statement::AlterTable { .. } => "ALTER TABLE",
        Statement::Copy { .. } => "COPY",
        Statement::Grant { .. } => "GRANT",
        Statement::Revoke { .. } => "REVOKE",
        _ => "non-SELECT",
    }
}

// ---------------------------------------------------------------------------
// AST walkers — manual because `sqlparser`'s derive-based Visit is
// behind the `visitor` feature, which is off by default. The walks are
// narrow: we only descend into the shapes that can carry a Select or
// a TableFactor. For everything else, the AST is opaque.
// ---------------------------------------------------------------------------

/// Returns `Some(simple_name)` if `name` is a 1- or 2-part object name
/// whose final part is a personal-data table. The 2-part case allows
/// `public.conversations` (or any other schema-qualified reference)
/// to match. The match is case-insensitive — Postgres folds
/// unquoted identifiers to lowercase.
fn personal_table_match(name: &ObjectName) -> Option<&'static str> {
    let parts = &name.0;
    if parts.is_empty() || parts.len() > 3 {
        return None;
    }
    let last = parts.last()?.as_ident()?;
    let lower = last.value.to_ascii_lowercase();
    PERSONAL_DATA_TABLES
        .iter()
        .copied()
        .find(|t| *t == lower)
}

/// Walks a `Statement` looking for the *first* personal-data table
/// reference. Used by the write path to refuse anything that touches
/// personal data.
#[derive(Default)]
struct PersonalDataTableFinder {
    first_hit: Option<String>,
}

impl PersonalDataTableFinder {
    fn record(&mut self, name: &ObjectName) {
        if self.first_hit.is_some() {
            return;
        }
        if let Some(matched) = personal_table_match(name) {
            self.first_hit = Some(matched.to_string());
        }
    }

    fn visit_statement(&mut self, stmt: &Statement) {
        match stmt {
            Statement::Insert(ins) => self.visit_table_object(&ins.table),
            Statement::Update(upd) => {
                self.visit_table_with_joins(&upd.table);
                if let Some(from) = &upd.from {
                    self.visit_update_from(from);
                }
            }
            Statement::Delete(del) => {
                for t in &del.tables {
                    self.record(t);
                }
                self.visit_from_table(&del.from);
            }
            Statement::Truncate(t) => {
                for tgt in &t.table_names {
                    self.record(&tgt.name);
                }
            }
            Statement::Drop { names, .. } => {
                for n in names {
                    self.record(n);
                }
            }
            Statement::AlterTable(at) => self.record(&at.name),
            Statement::CreateTable(ct) => self.record(&ct.name),
            Statement::CreateView(cv) => self.record(&cv.name),
            Statement::CreateIndex(ci) => {
                if let Some(name) = &ci.name {
                    self.record(name);
                }
                self.record(&ci.table_name);
            }
            Statement::Grant(g) => {
                if let Some(objs) = &g.objects {
                    self.visit_grant_objects(objs);
                }
            }
            Statement::Revoke(r) => {
                if let Some(objs) = &r.objects {
                    self.visit_grant_objects(objs);
                }
            }
            Statement::Copy {
                source: sqlparser::ast::CopySource::Table { table_name, .. },
                ..
            } => self.record(table_name),
            _ => {}
        }
    }

    fn visit_table_object(&mut self, table: &sqlparser::ast::TableObject) {
        match table {
            sqlparser::ast::TableObject::TableName(name) => self.record(name),
            sqlparser::ast::TableObject::TableFunction(_) => {}
            // INSERT INTO (subquery) target — walk the query body so
            // a nested personal-data reference is still caught.
            sqlparser::ast::TableObject::TableQuery(q) => self.visit_query(q),
        }
    }

    fn visit_table_with_joins(&mut self, twj: &TableWithJoins) {
        self.visit_table_factor(&twj.relation);
        for join in &twj.joins {
            self.visit_table_factor(&join.relation);
        }
    }

    fn visit_update_from(&mut self, from: &sqlparser::ast::UpdateTableFromKind) {
        use sqlparser::ast::UpdateTableFromKind::*;
        match from {
            BeforeSet(items) | AfterSet(items) => {
                for twj in items {
                    self.visit_table_with_joins(twj);
                }
            }
        }
    }

    fn visit_from_table(&mut self, from: &sqlparser::ast::FromTable) {
        use sqlparser::ast::FromTable::*;
        match from {
            WithFromKeyword(items) | WithoutKeyword(items) => {
                for twj in items {
                    self.visit_table_with_joins(twj);
                }
            }
        }
    }

    fn visit_table_factor(&mut self, tf: &TableFactor) {
        match tf {
            TableFactor::Table { name, .. } => self.record(name),
            TableFactor::Derived { subquery, .. } => {
                // A DELETE / UPDATE / etc. with a derived table in
                // its from list — keep walking; the subquery may
                // reference a personal-data table indirectly.
                self.visit_query(subquery);
            }
            _ => {}
        }
    }

    fn visit_query(&mut self, q: &Query) {
        // For the write-path finder, any personal-data table named
        // inside a sub-SELECT is also a refusal — a hostile
        // construction like `DELETE FROM scratch.x WHERE id IN
        // (SELECT id FROM public.messages)` shouldn't slip past the
        // checker even though the deletion target is scratch.
        self.visit_set_expr(&q.body);
    }

    fn visit_set_expr(&mut self, expr: &SetExpr) {
        match expr {
            SetExpr::Select(sel) => self.visit_select(sel),
            SetExpr::Query(q) => self.visit_query(q),
            SetExpr::SetOperation { left, right, .. } => {
                self.visit_set_expr(left);
                self.visit_set_expr(right);
            }
            _ => {}
        }
    }

    fn visit_select(&mut self, sel: &Select) {
        for twj in &sel.from {
            self.visit_table_with_joins(twj);
        }
    }

    fn visit_grant_objects(&mut self, objs: &sqlparser::ast::GrantObjects) {
        use sqlparser::ast::GrantObjects::*;
        if let Tables(names) | Sequences(names) | Schemas(names) = objs {
            for n in names {
                self.record(n);
            }
        }
    }
}

/// Mutating walker that grafts `WHERE <alias>.user_id = $1` onto every
/// `Select` that has a personal-data table in its FROM list (directly
/// or via a JOIN). The grafter intentionally does NOT walk into
/// `Statement::Insert/Update/Delete` etc. — those are write-path
/// statements and never reach this code (the read path is
/// SELECT-only). It DOES walk into derived tables (subqueries) and
/// CTEs because those can name personal-data tables and still need
/// scoping.
struct UserIdGrafter<'a> {
    user_id: &'a str,
    /// Tracks whether at least one graft happened — when true, the
    /// resulting `PreparedSelect.bound_user_id` is `Some(user_id)`.
    bound: bool,
}

impl<'a> UserIdGrafter<'a> {
    fn new(user_id: &'a str) -> Self {
        Self {
            user_id,
            bound: false,
        }
    }

    fn visit_query(&mut self, q: &mut Query) {
        if let Some(with) = &mut q.with {
            for cte in &mut with.cte_tables {
                self.visit_query(&mut cte.query);
            }
        }
        self.visit_set_expr(&mut q.body);
    }

    fn visit_set_expr(&mut self, expr: &mut SetExpr) {
        match expr {
            SetExpr::Select(sel) => self.visit_select(sel),
            SetExpr::Query(q) => self.visit_query(q),
            SetExpr::SetOperation { left, right, .. } => {
                self.visit_set_expr(left);
                self.visit_set_expr(right);
            }
            _ => {}
        }
    }

    fn visit_select(&mut self, sel: &mut Select) {
        // First, descend into derived tables in the FROM list so
        // *their* personal-data references get grafted too. Doing
        // the inner walk first means a nested SELECT against
        // `messages` gets its own predicate even if the outer
        // SELECT also references a personal-data table.
        for twj in &mut sel.from {
            self.visit_table_with_joins_inner(twj);
        }

        // Then, collect the personal-data table refs *at this Select
        // level* and graft a predicate referencing each one's alias
        // (or table name, if no alias).
        let mut refs: Vec<Ident> = Vec::new();
        for twj in &mut sel.from {
            self.collect_personal_refs(twj, &mut refs);
        }

        for ident in refs {
            let predicate = make_user_id_predicate(ident, self.user_id);
            sel.selection = Some(match sel.selection.take() {
                Some(existing) => Expr::BinaryOp {
                    left: Box::new(existing),
                    op: BinaryOperator::And,
                    right: Box::new(predicate),
                },
                None => predicate,
            });
            self.bound = true;
        }
    }

    fn visit_table_with_joins_inner(&mut self, twj: &mut TableWithJoins) {
        self.visit_table_factor_inner(&mut twj.relation);
        for join in &mut twj.joins {
            self.visit_table_factor_inner(&mut join.relation);
        }
    }

    fn visit_table_factor_inner(&mut self, tf: &mut TableFactor) {
        if let TableFactor::Derived { subquery, .. } = tf {
            self.visit_query(subquery);
        }
    }

    /// For each personal-data table reference in `twj`, ensure the
    /// table has a usable alias (assigning a synthetic one if not),
    /// and push the alias's identifier so the caller can build a
    /// `<alias>.user_id = $1` predicate.
    fn collect_personal_refs(&self, twj: &mut TableWithJoins, out: &mut Vec<Ident>) {
        Self::collect_in_factor(&mut twj.relation, out);
        for join in &mut twj.joins {
            Self::collect_in_factor(&mut join.relation, out);
        }
    }

    fn collect_in_factor(tf: &mut TableFactor, out: &mut Vec<Ident>) {
        if let TableFactor::Table { name, alias, .. } = tf
            && let Some(matched) = personal_table_match(name)
        {
            let ident = match alias {
                Some(a) => a.name.clone(),
                None => {
                    // Use the table's final-part identifier as the
                    // qualifier. This matches Postgres's own default —
                    // `SELECT conversations.id FROM conversations`
                    // uses the implicit alias. We don't need to
                    // mutate the AST to attach an alias; the
                    // column-reference form `<table>.user_id` works
                    // against the implicit name.
                    Ident::new(matched)
                }
            };
            out.push(ident);
        }
    }
}

/// Build a `<qualifier>.user_id = '<user_id>'` predicate as an `Expr`.
///
/// We inline the user_id as a quoted string literal rather than a
/// bind parameter because the SQL we hand to Postgres comes back
/// through `query.to_string()` — embedding `$1` would conflict with
/// any `$N` markers the user's SQL already uses, and we'd have to
/// rewrite all of them. A safely-escaped string literal sidesteps the
/// numbering problem entirely.
///
/// The string is escaped by sqlparser's `Value::SingleQuotedString`
/// formatter, which doubles embedded single quotes (Postgres's
/// standard SQL escape). The `user_id` value originates in the
/// trusted JWT extraction path (`auth-jwt`); a malicious value would
/// be a defense-in-depth concern, not the primary trust boundary —
/// but the standard-conforming escape closes it anyway.
fn make_user_id_predicate(qualifier: Ident, user_id: &str) -> Expr {
    let column = Expr::CompoundIdentifier(vec![qualifier, Ident::new("user_id")]);
    let literal = Expr::Value(ValueWithSpan {
        value: Value::SingleQuotedString(user_id.to_string()),
        span: sqlparser::tokenizer::Span::empty(),
    });
    Expr::BinaryOp {
        left: Box::new(column),
        op: BinaryOperator::Eq,
        right: Box::new(literal),
    }
}


/// Execute an LLM-supplied SQL query and return results as JSON.
///
/// See the module-level threat model (around `PERSONAL_DATA_TABLES`)
/// for the full security contract added in #141. The summary:
///
/// **Read queries** (`SELECT` / `WITH` / `TABLE` / `VALUES` / `EXPLAIN`)
/// run inside a READ ONLY transaction. Every reference to a
/// personal-data table has a `<table>.user_id = '<caller>'` predicate
/// AND'd into its `WHERE` clause via an AST rewrite. Tables without a
/// `user_id` column (system catalogs, `tool_definitions`, scratch
/// tables) are passed through unchanged. An automatic `LIMIT` is
/// appended when none is present.
///
/// **Write queries** (`CREATE` / `INSERT` / `UPDATE` / `DELETE` /
/// `DROP` / `ALTER` / …) are AST-validated against the personal-data
/// table list before execution: any reference (qualified or
/// otherwise) is refused. Surviving writes run in a normal
/// transaction with `search_path TO scratch, public`, so unqualified
/// DDL lands in `scratch` while custom user-named schemas remain
/// usable.
///
/// **Compound statements** (`SELECT 1; DROP TABLE …`) and
/// non-Postgres SQL are rejected at parse time.
///
/// Returns:
/// - Row-returning queries: `{ "columns": [...], "rows": [[...], ...], "row_count": N }`
/// - Non-row-returning writes: `{ "rows_affected": N }`
///
/// Errors are wrapped in `CoreError::ToolExecution` with a
/// human-readable explanation suitable for surfacing back to the LLM.
pub async fn execute_database_query(
    pool: &PgPool,
    sql: &str,
    limit: usize,
) -> Result<serde_json::Value, CoreError> {
    let sql_trimmed = sql.trim().trim_end_matches(';');

    // Cheap classifier on the *first non-comment* keyword (#40) just
    // to decide which validator to call. The validators each parse
    // again with sqlparser — the cheap pre-check lets us produce a
    // better-targeted error message ("SELECT-only on the read path"
    // vs. "personal-data target on the write path") without
    // double-parsing on every request.
    let upper = sql_trimmed.to_uppercase();
    let stripped = strip_leading_sql_comments(&upper);
    let first_keyword = stripped.split_whitespace().next().unwrap_or("");
    let is_read = matches!(
        first_keyword,
        "SELECT" | "WITH" | "TABLE" | "VALUES" | "EXPLAIN"
    );

    if is_read {
        let user_id = current_user_id();
        let prepared = prepare_select_for_user(sql_trimmed, user_id.as_str())?;
        execute_read(pool, &prepared.sql, limit).await
    } else {
        validate_write_statement(sql_trimmed)?;
        let upper = sql_trimmed.to_uppercase();
        execute_write(pool, sql_trimmed, &upper).await
    }
}

/// Strip leading SQL comments (`--` line comments and `/* … */` block
/// comments, including nested blocks per Postgres) plus the
/// whitespace between them. Returns a substring of `sql` starting at
/// the first character that is neither a comment nor whitespace.
///
/// On a malformed leading block comment (no closing `*/`), returns an
/// empty string — the caller treats that as "no recognisable
/// keyword", which routes to the write path where Postgres rejects
/// the malformed statement at parse time. Same outcome as a
/// nonsensical query without the comment.
fn strip_leading_sql_comments(sql: &str) -> &str {
    let bytes = sql.as_bytes();
    let mut i = 0;
    loop {
        // Skip whitespace.
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i + 1 >= bytes.len() {
            break;
        }
        if bytes[i] == b'-' && bytes[i + 1] == b'-' {
            // Line comment runs to end of line (LF or CR/LF) or end of input.
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            // Skip the newline so the next iteration sees the post-comment text.
            if i < bytes.len() {
                i += 1;
            }
            continue;
        }
        if bytes[i] == b'/' && bytes[i + 1] == b'*' {
            // Block comment, with nesting (Postgres extension to ANSI SQL).
            let mut depth: usize = 1;
            i += 2;
            while i + 1 < bytes.len() && depth > 0 {
                if bytes[i] == b'/' && bytes[i + 1] == b'*' {
                    depth += 1;
                    i += 2;
                } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if depth > 0 {
                // Unterminated block comment — treat as if the whole
                // remainder is still inside a comment so the caller
                // sees no keyword and routes to the write path, where
                // Postgres will reject the malformed statement.
                return "";
            }
            continue;
        }
        break;
    }
    &sql[i..]
}

/// Read path — READ ONLY transaction, auto-LIMIT, always rolled back.
///
/// `sql` is the post-rewrite SQL produced by `prepare_select_for_user`.
/// We re-derive the uppercase view for the `LIMIT` heuristic so the
/// caller doesn't have to recompute it after the rewrite.
async fn execute_read(
    pool: &PgPool,
    sql: &str,
    limit: usize,
) -> Result<serde_json::Value, CoreError> {
    let has_limit = sql.to_uppercase().contains(" LIMIT ");

    let mut tx = pool
        .begin()
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

    sqlx::query("SET TRANSACTION READ ONLY")
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

    // When the user query lacks a LIMIT clause, wrap it in a subquery with a
    // parameterised limit to avoid string-formatting user SQL.
    let rows: Vec<PgRow> = if has_limit {
        sqlx::query(sql)
            .fetch_all(&mut *tx)
            .await
            .map_err(|e| CoreError::ToolExecution(format!("query error: {e}")))?
    } else {
        let wrapped = format!("SELECT * FROM ({sql}) AS _limited LIMIT $1");
        sqlx::query(&wrapped)
            .bind(limit as i64)
            .fetch_all(&mut *tx)
            .await
            .map_err(|e| CoreError::ToolExecution(format!("query error: {e}")))?
    };

    tx.rollback()
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

    rows_to_json(&rows)
}

/// Write path — ensures `scratch` schema exists, sets search_path to
/// `scratch, public`, executes the statement, and commits.
async fn execute_write(
    pool: &PgPool,
    sql: &str,
    upper: &str,
) -> Result<serde_json::Value, CoreError> {
    // Ensure the scratch schema exists (idempotent).
    sqlx::query("CREATE SCHEMA IF NOT EXISTS scratch")
        .execute(pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

    let mut tx = pool
        .begin()
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

    // Unqualified writes go to scratch; public tables are still readable.
    sqlx::query("SET LOCAL search_path TO scratch, public")
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

    // If the statement contains RETURNING it will produce rows.
    let has_returning = upper.contains("RETURNING");

    if has_returning {
        let rows: Vec<PgRow> = sqlx::query(sql)
            .fetch_all(&mut *tx)
            .await
            .map_err(|e| CoreError::ToolExecution(format!("query error: {e}")))?;

        tx.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        rows_to_json(&rows)
    } else {
        let result = sqlx::query(sql)
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::ToolExecution(format!("query error: {e}")))?;

        tx.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(serde_json::json!({
            "rows_affected": result.rows_affected()
        }))
    }
}

/// Convert a slice of `PgRow` into the standard JSON result envelope.
fn rows_to_json(rows: &[PgRow]) -> Result<serde_json::Value, CoreError> {
    let columns: Vec<String> = if let Some(first) = rows.first() {
        first
            .columns()
            .iter()
            .map(|c| c.name().to_string())
            .collect()
    } else {
        return Ok(serde_json::json!({
            "columns": [],
            "rows": [],
            "row_count": 0
        }));
    };

    let mut json_rows: Vec<Vec<serde_json::Value>> = Vec::with_capacity(rows.len());

    for row in rows {
        let mut json_row = Vec::with_capacity(columns.len());
        for (i, col) in row.columns().iter().enumerate() {
            let type_name = col.type_info().name();
            json_row.push(pg_value_to_json(row, i, type_name));
        }
        json_rows.push(json_row);
    }

    let row_count = json_rows.len();
    Ok(serde_json::json!({
        "columns": columns,
        "rows": json_rows,
        "row_count": row_count
    }))
}

/// Convert a single column value from a PgRow into a serde_json::Value.
fn pg_value_to_json(row: &PgRow, index: usize, type_name: &str) -> serde_json::Value {
    match type_name {
        "TEXT" | "VARCHAR" | "CHAR" | "BPCHAR" | "NAME" => {
            match row.try_get::<Option<String>, _>(index) {
                Ok(Some(v)) => serde_json::Value::String(v),
                Ok(None) => serde_json::Value::Null,
                Err(_) => serde_json::Value::Null,
            }
        }
        "UUID" => match row.try_get::<Option<uuid::Uuid>, _>(index) {
            Ok(Some(v)) => serde_json::Value::String(v.to_string()),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        },
        "INT2" => match row.try_get::<Option<i16>, _>(index) {
            Ok(Some(v)) => serde_json::json!(v),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        },
        "INT4" => match row.try_get::<Option<i32>, _>(index) {
            Ok(Some(v)) => serde_json::json!(v),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        },
        "INT8" => match row.try_get::<Option<i64>, _>(index) {
            Ok(Some(v)) => serde_json::json!(v),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        },
        "FLOAT4" => match row.try_get::<Option<f32>, _>(index) {
            Ok(Some(v)) => serde_json::json!(v),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        },
        "FLOAT8" | "NUMERIC" => match row.try_get::<Option<f64>, _>(index) {
            Ok(Some(v)) => serde_json::json!(v),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        },
        "BOOL" => match row.try_get::<Option<bool>, _>(index) {
            Ok(Some(v)) => serde_json::json!(v),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        },
        "TIMESTAMPTZ" | "TIMESTAMP" => {
            match row.try_get::<Option<chrono::DateTime<chrono::Utc>>, _>(index) {
                Ok(Some(v)) => serde_json::Value::String(v.to_rfc3339()),
                Ok(None) => serde_json::Value::Null,
                Err(_) => match row.try_get::<Option<chrono::NaiveDateTime>, _>(index) {
                    Ok(Some(v)) => serde_json::Value::String(v.to_string()),
                    _ => serde_json::Value::Null,
                },
            }
        }
        "DATE" => match row.try_get::<Option<chrono::NaiveDate>, _>(index) {
            Ok(Some(v)) => serde_json::Value::String(v.to_string()),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        },
        "JSON" | "JSONB" => match row.try_get::<Option<serde_json::Value>, _>(index) {
            Ok(Some(v)) => v,
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        },
        "TEXT[]" | "_TEXT" | "VARCHAR[]" | "_VARCHAR" => {
            match row.try_get::<Option<Vec<String>>, _>(index) {
                Ok(Some(v)) => serde_json::json!(v),
                Ok(None) => serde_json::Value::Null,
                Err(_) => serde_json::Value::Null,
            }
        }
        _ => match row.try_get::<Option<String>, _>(index) {
            Ok(Some(v)) => serde_json::Value::String(v),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::String(format!("<unsupported type: {type_name}>")),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classify(sql: &str) -> bool {
        // Mirror what `execute_database_query` does to pick the path,
        // without needing a live Postgres pool. Returns `true` for
        // reads, `false` for writes.
        let trimmed = sql.trim().trim_end_matches(';');
        let upper = trimmed.to_uppercase();
        let stripped = strip_leading_sql_comments(&upper);
        let first_keyword = stripped.split_whitespace().next().unwrap_or("");
        matches!(
            first_keyword,
            "SELECT" | "WITH" | "TABLE" | "VALUES" | "EXPLAIN"
        )
    }

    #[test]
    fn plain_select_routes_to_read() {
        assert!(classify("SELECT * FROM conversations"));
        assert!(classify("WITH x AS (SELECT 1) SELECT * FROM x"));
        assert!(classify("EXPLAIN SELECT 1"));
    }

    #[test]
    fn plain_write_routes_to_write() {
        assert!(!classify("DELETE FROM scratch.foo"));
        assert!(!classify("INSERT INTO scratch.foo VALUES (1)"));
        assert!(!classify("UPDATE scratch.foo SET bar = 1"));
        assert!(!classify("CREATE TABLE scratch.foo (id INT)"));
    }

    #[test]
    fn leading_block_comment_does_not_promote_write_to_read() {
        // The original bypass: `/* */ DELETE` previously had
        // `first_keyword = "/*"` which doesn't match read keywords,
        // so it routed to the *write* path — but as an unwanted side
        // effect a leading comment in front of a SELECT also routed
        // to write (commits). After #40, comment-prefixed reads are
        // recognised as reads, and comment-prefixed writes still
        // route to write (so legitimate writes keep working).
        assert!(classify("/* comment */ SELECT * FROM conversations"));
        assert!(!classify("/* comment */ DELETE FROM public.foo"));
    }

    #[test]
    fn line_comment_is_stripped() {
        assert!(classify("-- hi\nSELECT 1"));
        assert!(classify("--  multiple    spaces \nSELECT 1"));
        assert!(!classify("-- hi\nDELETE FROM scratch.foo"));
    }

    #[test]
    fn nested_block_comments_are_handled() {
        // Postgres allows `/* outer /* inner */ still outer */`. A
        // naive `find("*/")` strip would terminate after the inner
        // close and mis-classify the outer text.
        assert!(classify("/* outer /* nested */ still outer */ SELECT 1"));
        assert!(classify(
            "/* /* /* deep */ */ */ WITH x AS (SELECT 1) SELECT * FROM x"
        ));
    }

    #[test]
    fn mixed_comment_kinds_strip_correctly() {
        assert!(classify("-- first\n/* block */\n-- another\nSELECT 1"));
        assert!(!classify("/* */ -- line\n /* */ DELETE FROM scratch.foo"));
    }

    #[test]
    fn unterminated_block_comment_routes_to_write() {
        // No `*/` — every char is consumed as comment, no keyword,
        // routes to the write path where Postgres will reject the
        // malformed statement at parse time.
        assert!(!classify("/* never closes SELECT 1"));
    }

    #[test]
    fn empty_or_whitespace_only_routes_to_write() {
        assert!(!classify(""));
        assert!(!classify("   "));
        assert!(!classify("\n\t\n"));
        assert!(!classify("-- only a comment"));
        assert!(!classify("/* only */"));
    }

    #[test]
    fn strip_does_not_modify_keyword_after_skipping() {
        // The strip should land *exactly* on the first non-comment
        // character so the upstream `to_uppercase()` + keyword match
        // still sees the canonical keyword.
        let stripped = strip_leading_sql_comments("/* x */SELECT 1");
        assert_eq!(stripped, "SELECT 1");
        let stripped = strip_leading_sql_comments("--c\n--d\nSELECT 1");
        assert_eq!(stripped, "SELECT 1");
    }

    // -------------------------------------------------------------------
    // #141: parser-level validation/rewriting. These tests don't need a
    // live DB — they exercise the AST-based rules that gate the
    // statement before it ever reaches the pool.
    // -------------------------------------------------------------------

    /// Helper that mirrors what `execute_database_query` does internally
    /// for the read path: parse, validate, rewrite for user_id. We test
    /// just the rewriter so test failures point straight at the rule
    /// that broke, not at the DB round-trip.
    fn rewrite_select(sql: &str, user_id: &str) -> Result<String, CoreError> {
        super::prepare_select_for_user(sql, user_id).map(|p| p.sql)
    }

    /// Helper for write-path validation — no DB required.
    fn validate_write(sql: &str) -> Result<(), CoreError> {
        super::validate_write_statement(sql).map(|_| ())
    }

    #[test]
    fn rewrite_grafts_user_id_into_bare_select() {
        let rewritten =
            rewrite_select("SELECT id FROM conversations", "alice").expect("rewrite");
        // The rewriter must inject a parameterised user_id filter
        // qualified by the `conversations` alias so it survives joins
        // against tables that also happen to have a `user_id` column.
        let lower = rewritten.to_ascii_lowercase();
        assert!(
            lower.contains("user_id ="),
            "rewritten SQL must include user_id filter, got: {rewritten}"
        );
        assert!(
            lower.contains("$1") || lower.contains("'alice'"),
            "rewritten SQL must bind/quote the caller user_id, got: {rewritten}"
        );
    }

    #[test]
    fn rewrite_ands_into_existing_where() {
        let rewritten =
            rewrite_select("SELECT id FROM conversations WHERE id = 'x'", "alice")
                .expect("rewrite");
        let lower = rewritten.to_ascii_lowercase();
        // Both predicates must survive — the original (id = 'x') and
        // the grafted (user_id = …).
        assert!(lower.contains("id = 'x'"), "original predicate dropped: {rewritten}");
        assert!(lower.contains("user_id ="), "user_id predicate missing: {rewritten}");
        // And there must be an explicit AND joining them, not an OR
        // or a comma — OR would weaken the guard, comma would mean
        // "SELECT a, b FROM …" which makes no sense in WHERE.
        assert!(lower.contains(" and "), "predicates must be AND'd, got: {rewritten}");
    }

    #[test]
    fn rewrite_skips_tables_without_user_id_column() {
        // System catalogs and `tool_definitions` (the system-wide tool
        // registry from #105's allowlist) have no user_id column, so
        // the rewriter must NOT graft anything onto them.
        let rewritten = rewrite_select(
            "SELECT table_name FROM information_schema.tables",
            "alice",
        )
        .expect("rewrite");
        assert!(
            !rewritten.to_ascii_lowercase().contains("user_id"),
            "must not graft user_id onto information_schema, got: {rewritten}"
        );

        let rewritten = rewrite_select("SELECT name FROM tool_definitions", "alice")
            .expect("rewrite");
        assert!(
            !rewritten.to_ascii_lowercase().contains("user_id"),
            "must not graft user_id onto tool_definitions, got: {rewritten}"
        );
    }

    #[test]
    fn rewrite_rejects_compound_select() {
        // Two statements is always wrong — we don't want statement-
        // stuffing slipping past a too-permissive first-keyword check.
        let err = rewrite_select("SELECT 1; SELECT 2", "alice").unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.to_ascii_lowercase().contains("single") || msg.contains("compound"),
            "rejection message must explain the compound-statement rule, got: {msg}"
        );
    }

    #[test]
    fn rewrite_rejects_non_select_statement() {
        // The read path is reserved for SELECT/WITH only.
        let err = rewrite_select("DELETE FROM conversations", "alice").unwrap_err();
        let msg = format!("{err:?}").to_ascii_lowercase();
        // The rejection must name DELETE specifically OR explain the
        // "SELECT-only" rule — a generic "not implemented" doesn't
        // count.
        assert!(
            msg.contains("delete") || msg.contains("only select") || msg.contains("not allowed"),
            "rejection message must name the offending statement type or the SELECT-only \
             rule, got: {msg}"
        );
    }

    #[test]
    fn validate_write_rejects_personal_data_targets() {
        // The write path runs in the scratch namespace; touching a
        // personal-data table from there — qualified or otherwise —
        // is a hostile move and must be refused.
        for sql in [
            "DROP TABLE public.conversations",
            "DROP TABLE conversations",
            "UPDATE public.conversations SET title = 'x'",
            "DELETE FROM messages WHERE 1=1",
            "INSERT INTO knowledge_base (id, content) VALUES ('x', 'y')",
            "TRUNCATE public.messages",
            "ALTER TABLE conversations DROP COLUMN title",
        ] {
            let err = validate_write(sql).unwrap_err_or_else(|_| {
                panic!("validate_write must reject {sql:?}");
            });
            let msg = format!("{err:?}").to_ascii_lowercase();
            assert!(
                msg.contains("personal-data") || msg.contains("not allowed"),
                "rejection message must explain the personal-data rule for {sql:?}, got: {msg}"
            );
        }
    }

    #[test]
    fn validate_write_accepts_scratch_namespace_ddl() {
        // Unqualified DDL — what the LLM uses for staging tables. Must
        // pass through to the existing scratch search_path machinery.
        for sql in [
            "CREATE TABLE staging_foo (id INT)",
            "DROP TABLE staging_foo",
            "CREATE SCHEMA my_scratch",
            "CREATE TABLE scratch.intermediate (x INT)",
        ] {
            validate_write(sql).unwrap_or_else(|e| {
                panic!("validate_write must accept {sql:?}, got: {e:?}");
            });
        }
    }

    #[test]
    fn validate_write_rejects_compound_statement() {
        // `CREATE TABLE foo (); DROP TABLE public.conversations` must
        // not slip in via the write path either.
        let err = validate_write("CREATE TABLE foo (); DROP TABLE public.conversations")
            .unwrap_err_or_else(|_| panic!("compound write must be rejected"));
        let msg = format!("{err:?}").to_ascii_lowercase();
        assert!(
            msg.contains("single") || msg.contains("compound"),
            "rejection must explain the compound-statement rule, got: {msg}"
        );
    }

    /// Small `Result::unwrap_err`-style helper that produces a clearer
    /// failure message when the result is unexpectedly `Ok`. The
    /// closure runs only on the `Ok` path.
    trait UnwrapErrOrElse<T, E> {
        fn unwrap_err_or_else<F: FnOnce(&T)>(self, f: F) -> E;
    }
    impl<T, E> UnwrapErrOrElse<T, E> for Result<T, E> {
        fn unwrap_err_or_else<F: FnOnce(&T)>(self, f: F) -> E {
            match self {
                Ok(v) => {
                    f(&v);
                    panic!("expected Err, got Ok");
                }
                Err(e) => e,
            }
        }
    }
}

#![allow(dead_code)]
//! Shared test support for the DB-gated storage suites.
//!
//! Every isolation suite is gated on `TEST_DATABASE_URL` and pass-skips when
//! it is unset — but the skip used to be a single easy-to-miss `eprintln!`
//! line buried among passing tests, so a green `cargo test` run read as
//! "multi-tenant isolation: covered" when in fact it had verified nothing
//! (this is how the #431 cross-tenant db_query bug went unnoticed). This
//! module centralizes the gate so the skip is *loud* and actionable, and
//! points at the one-command self-provisioning harness (`just test-db`).
//!
//! Included by each integration test via `mod support;` (it lives in a
//! subdirectory so cargo does not compile it as its own test binary).

use std::sync::Once;

static SKIP_BANNER: Once = Once::new();

/// The connection URL for the DB-gated suites, or `None` when no database is
/// available (in which case the caller should pass-skip). On the first `None`
/// in a test binary, prints a prominent, actionable banner so the skip is
/// impossible to mistake for "isolation is covered".
///
/// Set `TEST_DATABASE_URL` yourself, or run `just test-db` which boots an
/// ephemeral pgvector container (with the `vector` extension pre-created via
/// an auto-loaded init fixture), points this at it, runs the suites, and
/// tears the container down.
pub fn test_database_url() -> Option<String> {
    match std::env::var("TEST_DATABASE_URL") {
        Ok(url) if !url.trim().is_empty() => Some(url),
        _ => {
            SKIP_BANNER.call_once(print_skip_banner);
            None
        }
    }
}

fn print_skip_banner() {
    let banner = "\n\
         ┌──────────────────────────────────────────────────────────────────────┐\n\
         │  ⚠  storage DB-gated tests SKIPPED — TEST_DATABASE_URL is not set.     │\n\
         │                                                                        │\n\
         │  These verify multi-tenant user_id isolation. A green run WITHOUT a    │\n\
         │  database proves nothing about cross-tenant safety — it only means     │\n\
         │  the suites were skipped.                                              │\n\
         │                                                                        │\n\
         │  Run them against an ephemeral Postgres with:   just test-db           │\n\
         └──────────────────────────────────────────────────────────────────────┘\n";
    // libtest captures stdout/stderr for *passing* tests, so a plain
    // `eprintln!` here is hidden under a normal `cargo test` run (which is why
    // the old one-line skip was effectively invisible). Writing to the
    // controlling terminal bypasses that capture so the warning is actually
    // seen; fall back to stderr when there is no tty (CI, piped output — where
    // the `just test`/`just check` recipe-level warning covers it instead).
    use std::io::Write;
    match std::fs::OpenOptions::new().write(true).open("/dev/tty") {
        Ok(mut tty) => {
            let _ = tty.write_all(banner.as_bytes());
        }
        Err(_) => eprintln!("{banner}"),
    }
}

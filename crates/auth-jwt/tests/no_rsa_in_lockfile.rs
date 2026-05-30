//! Issue #143: assert that `jsonwebtoken` is no longer pulling in the
//! `rsa` crate (RUSTSEC-2023-0071, Marvin timing attack, no upstream
//! fix).
//!
//! Background: `jsonwebtoken`'s `rust_crypto` feature pulled in the
//! pure-Rust `rsa` crate. The codebase uses HS256 (in this crate) and
//! RS256 (in the daemon's OIDC validator). Switching the `jsonwebtoken`
//! backend from `rust_crypto` to `aws_lc_rs` preserves both algorithms
//! while routing all crypto through `aws-lc-rs` (already in the tree via
//! `rustls`) and drops the `rsa` build-graph contribution.
//!
//! Scope note (intentionally narrow): we check that `jsonwebtoken` no
//! longer declares `rsa` as a dependency in `Cargo.lock`, AND that `rsa`
//! is no longer in our compiled dependency graph. We deliberately do
//! NOT assert "`rsa` is fully absent from `Cargo.lock`" because a
//! separate, non-active path (`sqlx-mysql` declared as an optional dep
//! of `sqlx-macros-core`) still lists `rsa` in the lockfile even though
//! the `mysql` Cargo feature is never enabled and `sqlx-mysql` is never
//! compiled into our binary. Eliminating that record-only-in-lockfile
//! path is a separate cargo/sqlx-resolver concern outside #143's scope.
//!
//! Implementation note: we parse the workspace `Cargo.lock` from disk
//! relative to `CARGO_MANIFEST_DIR` rather than using `include_str!`. The
//! latter would freeze the lockfile contents at build time and produce
//! stale results after `cargo update`; reading at test runtime always
//! sees the live file. We also shell out to `cargo tree` to assert
//! `rsa` is absent from the *active* build graph — that's the
//! security-relevant property.

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("workspace root must exist")
}

#[test]
fn jsonwebtoken_does_not_depend_on_rsa_in_lockfile() {
    let lock_path = workspace_root().join("Cargo.lock");
    let lock = std::fs::read_to_string(&lock_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", lock_path.display()));

    // Find the `[[package]]` block whose `name = "jsonwebtoken"` and
    // inspect its `dependencies = [ ... ]` list for `rsa`. The block
    // ends at the next `[[package]]` header or EOF.
    let header = "name = \"jsonwebtoken\"";
    let start = lock
        .find(header)
        .expect("Cargo.lock must contain a jsonwebtoken package entry");
    let rest = &lock[start..];
    let block_end = rest[header.len()..]
        .find("[[package]]")
        .map(|i| header.len() + i)
        .unwrap_or(rest.len());
    let block = &rest[..block_end];

    let has_rsa_dep = block
        .lines()
        .any(|line| line.trim() == "\"rsa\",");

    assert!(
        !has_rsa_dep,
        "Cargo.lock shows `jsonwebtoken` still depends on `rsa` \
         (RUSTSEC-2023-0071). The `rust_crypto` feature must be off; \
         use `default-features = false` plus `aws_lc_rs`. Offending \
         block:\n{block}"
    );
}

#[test]
fn rsa_crate_absent_from_compiled_workspace_tree() {
    // `cargo tree -p rsa -i` prints the reverse dependency tree for
    // `rsa` if it's in the compiled graph, or warns "nothing to print"
    // if it's not. We use --quiet to suppress the warning so a clean
    // tree yields an empty stdout/stderr (modulo the warning).
    //
    // This guards against a future change re-adding any consumer of
    // `rsa` to the active build graph — whether through `jsonwebtoken`
    // regaining `rust_crypto` or another crate adopting `rsa`.
    let output = Command::new(env!("CARGO"))
        .args(["tree", "--workspace", "-i", "rsa"])
        .current_dir(workspace_root())
        .output()
        .expect("failed to invoke cargo tree");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Cargo signals "this package isn't in the graph" two different ways
    // depending on version: older cargo prints "warning: nothing to print."
    // to stderr (exit 0), while cargo >= 1.x hard-errors with "package ID
    // specification `rsa` did not match any packages" (exit 101). Either one
    // means `rsa` is absent — the security-relevant property. A present `rsa`
    // would instead print its reverse-dep tree to stdout.
    let absent = stdout.trim().is_empty()
        && (stderr.contains("nothing to print")
            || stderr.contains("did not match any packages"));

    assert!(
        absent,
        "`rsa` is present in the compiled workspace dependency graph \
         (RUSTSEC-2023-0071). cargo tree stdout:\n{stdout}\nstderr:\n{stderr}"
    );
}

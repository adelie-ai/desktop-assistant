# Agent Instructions

## Dependency Security Policy

After adding any new packages, **scan for CVEs before building**.

Build scripts (e.g. `build.rs` in Rust, install scripts in npm) execute at build time and are a potential attack vector. Scanning the updated lockfile *before* running a build catches malicious or vulnerable transitive dependencies before any build-time code can execute.

### Workflow

1. Add the dependency (e.g. `cargo add <crate>`) — this updates the lockfile but does not build.
2. Scan immediately using `cve-mcp scan_packages` — parse the updated lockfile and pass all (name, version, ecosystem) tuples to the tool.
3. Review findings. Investigate any Critical or High severity issues before proceeding.
4. Only build once the scan is clean (or findings are understood and accepted).

This applies regardless of ecosystem (Cargo, npm, PyPI, etc.).

//! Regenerates the C header (`include/adele_client.h`) from the `extern "C"`
//! surface via cbindgen, so the committed header stays in lockstep with the Rust
//! signatures. Build-time only — cbindgen does not ship to the target machine.

use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by cargo"),
    );

    // Regenerate only when the surface or its config changes — not every build.
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=build.rs");

    let include_dir = crate_dir.join("include");
    if let Err(e) = std::fs::create_dir_all(&include_dir) {
        println!(
            "cargo:warning=could not create {}: {e}",
            include_dir.display()
        );
        return;
    }
    let header = include_dir.join("adele_client.h");

    let config = cbindgen::Config::from_root_or_default(&crate_dir);
    match cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
    {
        // write_to_file rewrites only when the contents change, so a clean tree
        // stays clean across rebuilds.
        Ok(bindings) => {
            bindings.write_to_file(&header);
        }
        // A cbindgen parse failure in some toolchain must not break the workspace
        // build; the committed header stays authoritative. Surface it as a warning.
        Err(e) => println!("cargo:warning=cbindgen header generation failed: {e}"),
    }
}

# Changelog

This file records changes that a person who runs or integrates with the daemon
must know about: features that go away, configuration that changes meaning, and
wire contracts that move.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Removed

- The `[persistence]` git-backed history mirror. The settings form had no
  implementation behind it: no code read the configured remote, and no history
  was ever mirrored to git. The whole surface is gone - the `[persistence]`
  section of `daemon.toml`, the `GetPersistenceSettings` and
  `SetPersistenceSettings` commands, the `persistence` block of the aggregate
  `GetConfig` payload, and the `persistence` restart-required area.

  **For operators.** A `daemon.toml` that still carries a `[persistence]`
  section keeps loading. The daemon ignores the section and does not report it,
  so no action is needed. The section disappears from the file the next time a
  settings command rewrites it.

  **For client authors.** The D-Bus `GetConfig` and `SetConfig` methods and the
  `ConfigChanged` signal carry a positional struct, and the four
  `persistence_*` fields are removed from the middle of it. A client that reads
  those structs by position must move to the new signature. `adele-kde` moves
  in the same release.

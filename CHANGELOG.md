# Changelog

This file records changes that a person who runs or integrates with the daemon
must know about: features that go away, configuration that changes meaning, and
wire contracts that move.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Changed

- A tool result too large for the model to read inline is now stored whole
  instead of being cut down to 256 KiB before the row is written. The model
  still reads only the first 256 KiB, followed by a notice naming the message
  the whole output is stored under; `builtin_transcript_get` pages the rest
  back. A result that has no narrowing parameter - a page fetch takes a URL
  and nothing else - therefore has a way to reach its own output for the first
  time.

  **For operators.** Tool-result rows in the `messages` table can now reach
  1 MiB each, where they were previously bounded at 256 KiB. An absolute
  storage cap still applies at 1 MiB, above which the tail is dropped and the
  notice says so; that cap is what stops a runaway tool from stalling the
  `messages` INSERT.

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

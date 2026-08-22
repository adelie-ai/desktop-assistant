# Changelog

This file records changes that a person who runs or integrates with the daemon
must know about: features that go away, configuration that changes meaning, and
wire contracts that move.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- Two commands report what filled a turn's prompt:
  `list_context_breakdowns { conversation_id, limit?, offset? }` returns one
  entry per turn of a conversation, oldest first, and
  `get_context_breakdown { request_id }` returns one turn by the correlation id
  its own events carried. Each entry gives the estimated tokens for every part
  of the prompt - system instruction, summary, plan, pinned notes, scratchpad
  index, recall, transcript, tool schemas - beside the count the provider
  reported, the input-token budget the turn ran under, which tier resolved that
  budget, whether the turn compacted itself, and how many of its messages it
  read as a pointer, a head or a notice instead of as their stored content. The last of those is what tells a curated limit for the model
  apart from the conservative fallback the daemon uses when nothing supplied
  one; the two are the same number and a different situation, and until now
  nothing outside the daemon could tell them apart.

  **For integrators.** The per-part estimate and the provider's reported count
  are two measurements of one prompt, taken by different counters. They do not
  agree, and the difference is worth showing. Do not sum them, and do not
  present either as a component of the other. An absent `provider_used_tokens`
  means the provider reported no count, which is not zero.

  **For operators.** A new table, `context_breakdowns`, holds one row per turn.
  It carries `user_id`, is scoped by it on every read, and has its own
  row-level-security policy. A deployment with no database keeps no rows and
  answers both commands with an error rather than an empty list, so "this
  conversation has no entries" and "this daemon keeps none" stay different
  answers.
- The daemon can keep the full text of every turn: the request exactly as sent
  to the connector (system prompt and every injected block included), the
  reply, the tool calls with their arguments, and each tool result. One record
  per turn and one per round within it, keyed by the turn's correlation id -
  the same value the client's event stream and a trace backend already carry.

  **For operators.** The new `[inspector]` section holds `enabled` and
  `retention_days`. An absent `enabled` follows the deployment: on for a daemon
  reachable only over its Unix socket, off once `[transports] ws_enabled` opens
  the remote door, because there the same record is one principal's
  conversations in a store a second principal operates. A stated value wins in
  both directions, and the boot log states which it resolved to and why. The
  window defaults to 7 days and is held to at least 1; there is no
  keep-forever value, and an hourly sweep removes what is past it. Capture also
  needs a database - configured without one, the daemon warns rather than
  writing nothing quietly. Records live in the new `turn_records` and
  `turn_round_records` tables, both `user_id`-scoped with their own row-level
  security.

  Two operational notes. The sweep runs wherever there is a database, whether
  or not capture is on now, so turning capture off still ages out what was
  already written. And each round records the whole prompt it sent, which
  contains the rounds before it, so a long tool-using turn writes several
  megabytes; the retention window is the only bound, so size it against the
  disk. `docs/features/turn-records.md` has the whole contract.

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
  storage cap still applies at 1 MiB, above which the tail is dropped; that cap
  is what stops a runaway tool from stalling the `messages` INSERT. A row cut
  that way opens with a note saying how many bytes the tool produced and how
  many were dropped, so both the model and a person reading the transcript meet
  the loss first rather than after paging to the end.

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

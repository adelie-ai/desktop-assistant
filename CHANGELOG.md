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
### Fixed

- A conversation larger than one transport frame can be opened again. The
  daemon used to answer `GetConversation` with every message, content whole.
  Every transport caps one message at 4 MiB, nothing bounded the answer
  against that cap, and the client's reader broke the connection on the
  rejected frame rather than failing the one request - so every outstanding
  request failed with it and the conversation stayed unopenable, attempt after
  attempt.

  `GetConversation` and `GetMessages` now bound their answer to 3 MiB of
  serialized bytes, measured as the encoder emits them, and report what they
  left out in the payload. `ConversationView` carries
  `omitted_leading_messages`, which is both the count of dropped older
  messages and the cursor to read them back with `GetMessages { after_count }`.
  `MessagesView` carries `size_capped`. A single message past the whole budget
  arrives headed rather than dropped, opening with a line that states its true
  size, so a conversation is never empty because one row was too large. Nothing
  is deleted: the message stays stored whole.

  **For integrators.** All three fields are additive with serde defaults, so an
  older payload parses unchanged and an older client ignores them. Two
  behaviour changes are worth reading. A conversation between 3 MiB and 4 MiB
  of serialized messages now returns partial where it previously returned whole
  - it was already at the edge of unreadable. And `GetConversation` over D-Bus
  returns the newest messages only, with no field in that signature to say so;
  a D-Bus caller that needs the whole history must page it with `GetMessages`.

  `ListConversations` is bounded as a whole answer, not field by field. A
  conversation **title** and a summary's **tags** are both written by the client
  and were validated nowhere, so either one could take the whole response budget
  on its own and make the conversation list unreadable - three conversations
  carrying a 2 MiB tag each produced an answer of 6,291,823 bytes, past the
  3,145,728-byte response budget and past the 4,194,304-byte transport cap.
  The answer is now cut as a whole, keeping the front of the list, and every
  returned row carries `omitted_trailing_conversations`. A row that does not
  fit loses tags from the end of its tag list before it loses its place in the
  list, and says how many with `omitted_tags`, because a conversation the
  caller cannot see at all is worse than one shown with fewer tags.

  **A conversation title is bounded where it is written.** The cap is 4 KiB of
  serialized bytes, and it is applied two ways because a title has two sources.
  A title a client supplies is **refused** past the cap: `CreateConversation`
  and `RenameConversation` return a classified decline carrying the business
  code `conversation_title_too_long`, a description that names the size only, a
  message fit to show the person who typed the title, and `retryable: false`.
  Nothing is stored and nothing already stored is changed. It is a refusal
  rather than a truncation because silently rewriting what a person typed is
  the loss this work removes. A title the daemon composes for itself is **cut
  at generation** instead - `Standalone: <name>`, `Subagent: <name>`, and the
  name an LLM writes after the first message - because refusing there would
  fail an operation the user did not ask for. A cut generated title ends with
  `...`, and the label leads, so a cut removes the supplied name rather than
  the label.

  A response still cuts an over-cap title, as a **backstop for rows written
  before that rule**. A write rule cannot repair a title that is already
  stored. Such a title arrives cut, states the loss in words at the end of what
  is kept, and carries `title_total_bytes` with the stored size. Nothing is
  deleted: the title stays stored whole. A title inside the bound is unchanged
  and unmarked, so an ordinary answer keeps the bytes it always had. The write
  cap and the response cap are one number, held equal by a compile-time
  assertion, so a title accepted on the way in is never cut on the way out.

  **A cut title must not be written back.** A client that pre-fills a rename
  field from a legacy cut title would offer the cut value, and
  `RenameConversation` now refuses it. `title_total_bytes` is the check: it is
  set only when the title is a head, so a client refuses to pre-fill from a
  title that carries it and the person never meets the refusal. The notice
  inside the value cannot serve for this - it is prose, and prose cannot be
  told from a title the user typed. The `ConversationTitleChanged` event
  carries a bounded title and no marker, so a rename field is pre-filled from
  `GetConversation` or `ListConversations`, never from that event.

  **For integrators.** A title between 1 KiB and 4 KiB was cut in a response
  before this change and is returned whole now. A `CreateConversation` or
  `RenameConversation` past 4 KiB failed nothing before and is refused now.

  `MessagesView` carries `next_after_count`, the raw message index a window
  ended at, and that is the value to send as the next request's `after_count`.
  A caller must not compute the cursor as `after_count + messages.len()`:
  `after_count` counts raw messages while `messages` counts what `include_roles`
  let through, so a computed cursor names an earlier row than the window
  reached and returns rows twice whenever a role filter is in force.
  `size_capped` is the stop condition of the walk in **cursor mode**
  (`after_count >= 0`). In **tail mode** the cursor names the end of the
  conversation, because a tail window keeps the newest messages, so following it
  returns an empty window and the walk would stop without reading the older
  messages the tail dropped. A tail caller reads `truncated`, which says older
  messages were dropped, and reads them by switching to cursor mode from
  `after_count: 0`.

  `omitted_leading_messages` at `0` means no message was left out, and
  `size_capped` at `false` means no message was removed from a window. Neither
  means the answer is whole: a conversation of one over-budget message comes
  back with the count at `0`, `size_capped` at `false`, and that one row headed.
  A reader that wants to know it has everything checks `content_total_bytes` on
  the messages and `title_total_bytes` on the title as well.

  The transports no longer write a frame past their own cap at all.
  `write_frame` refuses an oversize body before writing any bytes, and each
  transport turns that refusal into a failure of the one request, carrying that
  request's id, rather than a dropped connection. Both clients hold themselves to the
  same rule on the way out: a request past the transport's own cap -
  `MAX_FRAME_LEN` over UDS, `MAX_WS_MESSAGE_BYTES` over WebSocket - fails on the
  spot, with an error naming the size, and the connection keeps serving. It used
  to be enqueued and cost the socket, and every request in flight with it.

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

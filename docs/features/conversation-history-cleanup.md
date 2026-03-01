# Conversation history cleanup (archive/stub/delete)

This doc describes how to shrink `conversations.json` while preserving useful history.

It covers two approaches:

1) **Offline file-edit workflow** (what we can do today)
2) **Daemon-native workflow** (what we should implement via built-in MCP tools so the daemon can do it safely without manual file edits)

---

## Goals

- Reduce size of conversation history by removing low-value threads.
- Preserve potentially useful context by **archiving** full conversations, and keeping a **stub summary** in the main history.
- Ensure substantive conversations have:
  - a good title
  - durable extractions saved as **preferences** (key/value) and/or **memory** (facts/rules-of-thumb)
- Process order: **oldest → newest**, with **newer information superseding older**.
- Prioritize certain naming patterns (historically: threads titled `Desktop Chat …`).

---

## Data layout (current)

The daemon persists conversations under the XDG data dir:

- Data dir: `${XDG_DATA_HOME:-~/.local/share}/desktop-assistant`
- Main store: `conversations.json`
- Recommended archive root: `conversation_archive/YYYY/MM/`

We treat the archive as write-once append-only storage.

---

## Classification rules (aggressive)

When reviewing a conversation, classify it into one of:

### A) Hard delete ("dumb")
Delete if the thread is:

- Pure testing / demo prompts ("leading questions")
- Repeated retries after backend errors
- No follow-through, no durable facts, no work product
- Redundant with newer threads or already-captured notes

### B) Archive + stub ("middling" / "potential value")
Archive and replace with a stub if the thread has *some* value but doesn’t need to stay in the main file.

Examples:
- Work/client context that might be referenced later
- One-off doc generation where the final artifact lives elsewhere
- Decisions or status updates where the details are no longer needed, but an audit trail is nice

### C) Keep as-is (rare)
Keep only if:
- it’s currently active/ongoing, or
- it’s a canonical reference thread used frequently

---

## Stub format (in `conversations.json`)

A stubbed conversation should include:

- Title: descriptive + suffix `(archived)`
- Messages: replace the full message list with a single assistant message:
  - 1 short paragraph summary
  - `Archive: <absolute path>` reference

Example stub content:

```
Summary: Created a timeclock project for metersys and recorded that it is time-billed and requires time entries.

Archive: /home/dave/.local/share/desktop-assistant/conversation_archive/2026/02/2026-02-20_Timeclock-tools_and_metersys-project.json
```

---

## Durable extraction rules

### Preferences
Store concrete key/value settings as preferences (paths, URLs, commands, IDs).

- Examples:
  - `project.adelie_platform.path = /home/dave/projects/adelie-platform`
  - `project.metersys.path = /home/dave/projects/clients/metersys`

### Memory
Store durable factual context and rules-of-thumb as memory.

- Example:
  - “MCP repos are often (but not always) part of the adelie-platform project.”

### Supersession
When an older conversation conflicts with a newer one:

- Update/overwrite the preference
- Update memory (or add a new memory note that supersedes the old one)

---

## Critical note: don’t edit while daemon/TUI is running

The daemon/TUI may hold conversations in memory and periodically write them back to disk.

**If you edit `conversations.json` while the daemon is running, your edits may be overwritten.**

Safe offline workflow:

1) Stop any writers
   - `desktop-assistant-daemon`
   - `desktop-assistant-tui`
2) Apply cleanup edits (delete/archive/stub) to `conversations.json`
3) Restart the daemon/TUI

We used a helper script (`~/da_convo_cleanup_apply.sh`) to restore a cleaned `conversations.json` from a backup after stopping writers.

---

## Offline workflow (today): step-by-step

1) Create a timestamped backup of `conversations.json`.
2) Enumerate conversations, oldest → newest.
3) For each conversation:
   - Classify (delete vs archive+stub vs keep)
   - If archive+stub:
     - write the full conversation JSON to `conversation_archive/YYYY/MM/<date>_<slug>.json`
     - replace messages with the stub summary in the main store
   - Extract durable preferences and memory
4) After edits, validate:
   - count of `Desktop Chat …` titles is 0 (if that’s a goal)
   - archived stubs have `(archived)` in the title
   - JSON parses

---

## Daemon-native workflow (recommended)

Instead of editing JSON files, the daemon should provide built-in tools to:

- list conversations
- archive conversations
- stub/retitle conversations
- delete conversations
- ensure atomic persistence (no overwrite races)

This avoids the "daemon rewrote the file" problem.

See the next section for proposed tools.

---

## Proposed built-in MCP tools

These should be **builtin_*** tools implemented by the daemon and exposed to the assistant.

### Conversation store

#### `builtin_conversations_list`
List conversations with lightweight metadata.

- Inputs:
  - `limit`, `offset`
  - `order`: `oldest|newest`
  - `title_prefix` / `title_regex`
  - `created_before` / `created_after`
  - `min_messages` / `max_messages`
- Output:
  - `id`, `created_at`, `updated_at`, `title`, `message_count`, `byte_size` (optional)

#### `builtin_conversations_get`
Fetch a single conversation.

- Inputs: `id`
- Output: full conversation (or optionally messages only)

#### `builtin_conversations_update_metadata`
Update title and other metadata without touching messages.

- Inputs: `id`, `title` (and/or tags)

#### `builtin_conversations_replace_with_stub`
Atomically replace a conversation’s messages with a stub message.

- Inputs:
  - `id`
  - `title`
  - `summary`
  - `archive_ref` (path or opaque archive id)

#### `builtin_conversations_delete`
Delete a conversation.

- Inputs:
  - `id`
  - `hard`: true/false
  - `reason`

### Archiving

#### `builtin_conversations_archive`
Write a conversation to the archive store and return a reference.

- Inputs:
  - `id`
  - `archive_policy`: `verbatim|redacted|metadata_only`
  - `slug` (optional)
- Output:
  - `archive_ref` (opaque id or absolute path)

#### `builtin_conversations_archive_delete`
Delete an archive record (rare; for cleanup).

- Inputs: `archive_ref`

### Transactions / concurrency

#### `builtin_conversations_begin_transaction`
Start a transaction/lock so edits are consistent.

#### `builtin_conversations_commit_transaction`
Commit and flush to disk.

#### `builtin_conversations_abort_transaction`
Abort.

(Alternate design: each mutation tool is internally transactional; still needs locking to prevent daemon/TUI rewrite races.)

### Search / analysis helpers

#### `builtin_conversations_classify`
Let the daemon (or assistant) classify a conversation:

- Inputs: `id`, `policy_profile` (e.g. aggressive)
- Output: `delete|archive_stub|keep` + rationale

#### `builtin_conversations_extract_durable_items`
Extract candidate preferences/memories from a conversation.

- Inputs: `id`
- Output:
  - `preferences`: [{key,value,confidence}]
  - `memories`: [{fact,confidence,tags}]

### Integration with existing memory/preferences tools

Once extracted, the assistant would still call:
- `builtin_preferences_remember`
- `builtin_memory_remember`

…but having a conversation extraction tool dramatically reduces manual parsing.

---

## Implementation notes / acceptance criteria

- All conversation mutations must be **atomic** and **durable**.
- Mutations must not be lost if a TUI client is running.
- Archive references should be stable:
  - either absolute file paths under the archive root, or
  - opaque IDs resolvable via `builtin_conversations_archive_get`.
- Tooling should support a "dry run" mode for audits.

---

## Suggested future: server-side archive store

File-based archives are fine to start, but consider a simple SQLite archive index to support:

- listing archives
- full-text search
- retention policies
- content redaction


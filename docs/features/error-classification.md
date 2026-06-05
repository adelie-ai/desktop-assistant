# Backend error classification (self-learning)

LLM backends return errors as opaque strings, and the wording differs by
provider and model. The daemon needs to *act* on them — shrink the prompt and
retry on a context overflow, back off on a rate limit, stop (not retry) on a
billing failure. The classifier turns those opaque strings into a small,
**closed** set of structured causes the rest of the system already handles, and
it **learns** new error shapes over time so it gets better without code changes.

Epic: #178. Implemented as a decorator (`ClassifyingLlmClient`) that wraps every
connector innermost, so it sees a connector's raw error before the
retry/recovery layers above it.

## Normalized causes

Every opaque error is mapped to one of (or left unchanged):

| Cause | Maps to `CoreError` | Behavior |
| --- | --- | --- |
| `context_overflow` | `ContextOverflow` | recovery ladder truncates/compacts and retries (bounded) |
| `rate_limited` | `RateLimited` | retried with backoff (bounded) |
| `billing_fatal` | `QuotaExceeded` | **terminal** — surfaced, never retried |
| `auth` | *(unchanged)* | **terminal** — surfaced |
| `model_loading` | `ModelLoading` | surfaced |
| `tools_unsupported` | `ToolsUnsupported` | surfaced |
| `transient` | *(unchanged)* | surfaced |
| `unknown` | *(unchanged)* | original error surfaced — **no behavior change** |

`unknown` is the safe default: anything we can't confidently classify behaves
exactly as it did before the classifier existed.

## How it works — three tiers

On an opaque `CoreError::Llm(...)`, the decorator tries, in order:

1. **Built-in matchers** — deterministic substring/HTTP-status rules in
   `core/src/error_classify.rs`. Pure, instant, always run. This is where the
   well-known phrasings live (e.g. `prompt is too long`, `Input is too long`,
   `exceeded your current quota`).
2. **Learned cache** — a lookup in the `error_classifications` table for a
   stored *signature* that occurs in this message for this connector. Local, no
   LLM call.
3. **Cheap LLM** — only when 1 and 2 miss. The titling-purpose LLM is asked to
   classify the error into the closed set above plus a short *signature*
   substring. The result is **persisted** to the learned cache, so the next
   occurrence is resolved at tier 2.

A signature is a distinctive, case-insensitive substring of the error message
(for example `exceeded your current quota`). Lookups match a signature as a
literal substring; when several match, the longest (most specific) wins.

## Requirements

- **Tier 1** always works — no configuration needed.
- **Tiers 2 and 3** require a configured database (`[database]` in
  `daemon.toml`). The learned cache lives there; tier 3 writes to it.
- **Tier 3** uses the titling-purpose LLM as the classifier (cheap by design —
  configure `[purposes.titling]`). With no DB, tiers 2–3 are simply skipped.

## Loop-safety

The classifier is a **best-effort, non-reentrant, single-shot, time-bounded**
side path. It can never spin or recurse:

- **Reentrancy guard.** Tier 3 calls an LLM that is itself wrapped by the
  classifier. The decorator sets a task-local guard around its classification
  call; any decorator that sees the guard set skips tiers 2–3 (tier 1 still
  runs). So a classification call's own errors are never re-classified.
- **Single-shot.** Each error is classified at most once; the decorator never
  retries the classification.
- **Time-bounded.** A tier-3 call has a 5s timeout; on timeout (or any failure,
  or an unusable answer) the classifier falls back to the original error.
- **Terminal causes never auto-retry.** `RetryingLlmClient` retries only
  `RateLimited`, so a `billing_fatal` → `QuotaExceeded` classification cannot
  start a retry/cost loop. The classifier only *labels*; it does not retry.
- **No hallucinated patterns.** A learned signature must be at least 8
  characters **and** actually occur in the original message, or it's rejected.
- **No secrets exfiltrated.** Before an error is sent to the (possibly remote)
  classifier LLM, credential-looking tokens are redacted.

## Inspecting and curating the learned table

The cache is a plain, global (not per-user), human-auditable table. Connect to
the configured database (`[database].url` in `daemon.toml`).

```sql
-- What has the classifier learned, most-used first?
SELECT connector, cause, signature, hit_count, last_matched_at
FROM   error_classifications
ORDER  BY hit_count DESC;
```

```sql
-- Fix a misclassification (the next lookup uses the new cause immediately):
UPDATE error_classifications
SET    cause = 'rate_limited'
WHERE  connector = 'bedrock' AND signature = 'temporarily throttled';
```

```sql
-- Forget a learned entry (it will be re-learned if seen again):
DELETE FROM error_classifications
WHERE  connector = 'openai' AND signature = 'some bad signature';
```

```sql
-- Pre-seed a mapping by hand (e.g. for an error you know but haven't hit yet):
INSERT INTO error_classifications (connector, signature, cause, source)
VALUES ('bedrock', 'model is currently loading', 'model_loading', 'manual')
ON CONFLICT (connector, signature) DO UPDATE SET cause = EXCLUDED.cause;
```

`cause` must be one of the keys in the table above. `signature` should be a
distinctive substring — specific enough not to match unrelated errors, but not
so specific (e.g. containing a request id or token count) that it never matches
again.

## Extending the built-in matchers

If a new error phrasing is common enough to hard-code, add it to
`classify_builtin` in `crates/core/src/error_classify.rs` (tier 1) rather than
relying on the learned cache. Keep `billing`/`quota` checks ahead of the
rate-limit check — some providers return quota exhaustion with HTTP 429, and it
must be treated as terminal, not retried.

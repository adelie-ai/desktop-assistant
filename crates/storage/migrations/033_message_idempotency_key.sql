-- #570 Phase 1b: persist the client-supplied idempotency key on the message
-- row so a transcript reload / reconnect returns the key and clients can dedup
-- an echoed UserMessageAdded by exact match rather than a content compare.
--
-- Only USER rows ever carry a key (the message that initiated a client-retryable
-- send); assistant/tool rows and keyless sends leave it NULL. Nullable and
-- forward-only.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS idempotency_key TEXT;

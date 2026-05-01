-- Issue #57: anchor the user's active task on the conversation.
--
-- Captured at the start of each `send_prompt` and re-injected into the
-- message stream as a `[Current task]` system message when the original
-- user message has drifted out of the context window or been absorbed
-- by the rolling summariser. Nullable because legacy rows have no anchor
-- and conversations without messages have no task.
ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS active_task TEXT;

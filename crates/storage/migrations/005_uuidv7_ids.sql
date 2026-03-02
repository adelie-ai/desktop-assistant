-- Convert messages.id from BIGSERIAL to TEXT (UUIDv7 populated by application code).
ALTER TABLE messages ALTER COLUMN id DROP DEFAULT;
ALTER TABLE messages ALTER COLUMN id TYPE TEXT USING id::TEXT;
DROP SEQUENCE IF EXISTS messages_id_seq;

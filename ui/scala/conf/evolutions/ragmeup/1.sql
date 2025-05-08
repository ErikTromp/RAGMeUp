-- conf/evolutions/ragmeup/1.sql

# --- !Ups
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at NUMERIC
);

CREATE TABLE IF NOT EXISTS chat_messages (
    chat_id TEXT,
    message_offset INTEGER,
    created_at NUMERIC,
    text TEXT,
    role TEXT,
    documents TEXT,
    rewritten TEXT,
    fetched_new_documents BOOLEAN,
    PRIMARY KEY (chat_id, message_offset),
    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS chat_messages_chat_id_message_offset_idx ON chat_messages (chat_id, message_offset);

CREATE TABLE IF NOT EXISTS feedback (
    chat_id TEXT,
    message_offset INTEGER,
    feedback BOOLEAN,
    feedback_text TEXT,
    FOREIGN KEY (chat_id, message_offset) REFERENCES chat_messages(chat_id, message_offset) ON DELETE CASCADE
);

# --- !Downs
DROP TABLE IF EXISTS chats;
DROP TABLE IF EXISTS chat_messages;
DROP TABLE IF EXISTS feedback;

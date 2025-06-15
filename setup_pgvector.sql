-- This script sets up pgvector extension and tables for semantic conversation retrieval

-- Enable the pgvector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table for storing conversation embeddings
CREATE TABLE IF NOT EXISTS conversation_embeddings (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL,
    message TEXT NOT NULL,
    response TEXT NOT NULL,
    combined_text TEXT NOT NULL,
    embedding VECTOR(1536), -- For Titan embeddings (1536 dimensions)
    metadata JSONB,
    audio_url TEXT,
    response_audio_url TEXT,
    language TEXT,
    confidence_score FLOAT,
    format TEXT,
    source TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    access_count INTEGER DEFAULT 0
);

-- Create an index on the user_id for faster lookups
CREATE INDEX IF NOT EXISTS conversation_embeddings_user_id_idx ON conversation_embeddings(user_id);

-- Create a vector index on the embedding column
CREATE INDEX IF NOT EXISTS conversation_embeddings_embedding_idx ON conversation_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create a function for similarity search
CREATE OR REPLACE FUNCTION match_conversations(
    query_embedding VECTOR(1536),
    match_threshold FLOAT,
    match_count INT,
    p_user_id UUID
)
RETURNS TABLE (
    id BIGINT,
    user_id UUID,
    message TEXT,
    response TEXT, 
    combined_text TEXT,
    similarity FLOAT,
    metadata JSONB,
    created_at TIMESTAMPTZ
)
LANGUAGE plpgsql
AS $$
DECLARE
    matched_id BIGINT;
    matched_records RECORD;
BEGIN
    FOR matched_records IN 
        SELECT
            c.id,
            c.user_id,
            c.message,
            c.response,
            c.combined_text,
            1 - (c.embedding <=> query_embedding) AS similarity,
            c.metadata,
            c.created_at
        FROM conversation_embeddings c
        WHERE c.user_id = p_user_id
        AND 1 - (c.embedding <=> query_embedding) > match_threshold
        ORDER BY c.embedding <=> query_embedding
        LIMIT match_count
    LOOP
        -- Update access statistics for each matched record
        PERFORM record_conversation_access(matched_records.id);
        
        -- Return the matched record
        id := matched_records.id;
        user_id := matched_records.user_id;
        message := matched_records.message;
        response := matched_records.response;
        combined_text := matched_records.combined_text;
        similarity := matched_records.similarity;
        metadata := matched_records.metadata;
        created_at := matched_records.created_at;
        
        RETURN NEXT;
    END LOOP;
END;
$$;

-- Create a trigger function to update last_accessed and access_count
CREATE OR REPLACE FUNCTION update_conversation_access()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE conversation_embeddings
    SET last_accessed = NOW(),
        access_count = access_count + 1
    WHERE id = NEW.id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create a function to manually update access statistics
-- (PostgreSQL doesn't support AFTER SELECT triggers directly)
CREATE OR REPLACE FUNCTION record_conversation_access(conversation_id BIGINT) 
RETURNS VOID AS $$
BEGIN
    UPDATE conversation_embeddings
    SET last_accessed = NOW(),
        access_count = access_count + 1
    WHERE id = conversation_id;
END;
$$ LANGUAGE plpgsql;

-- Add permissions for authenticated users
ALTER TABLE conversation_embeddings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view their own conversations"
    ON conversation_embeddings FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own conversations"
    ON conversation_embeddings FOR INSERT
    WITH CHECK (auth.uid() = user_id);

COMMENT ON TABLE conversation_embeddings IS 'Stores conversation history with vector embeddings for semantic search';

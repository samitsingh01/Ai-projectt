-- database/enhanced_init.sql
-- Enhanced initialization script with Q&A and RAG capabilities

CREATE DATABASE IF NOT EXISTS bedrock_chat;
USE bedrock_chat;

-- Enhanced users table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    user_preferences JSON,
    total_questions INT DEFAULT 0,
    INDEX idx_session_id (session_id)
);

-- Enhanced conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    response TEXT NOT NULL,
    model_used VARCHAR(100) DEFAULT 'unknown',
    conversation_type ENUM('chat', 'qa', 'analysis') DEFAULT 'chat',
    has_sources BOOLEAN DEFAULT FALSE,
    response_time_ms INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session_id (session_id),
    INDEX idx_created_at (created_at),
    INDEX idx_session_created (session_id, created_at),
    INDEX idx_conversation_type (conversation_type)
);

-- Enhanced uploaded_files table
CREATE TABLE IF NOT EXISTS uploaded_files (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    original_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(100) NOT NULL,
    file_size BIGINT NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    extracted_text LONGTEXT,
    is_processed BOOLEAN DEFAULT FALSE,
    processing_status ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending',
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_date TIMESTAMP NULL,
    INDEX idx_session_id (session_id),
    INDEX idx_upload_date (upload_date),
    INDEX idx_session_upload (session_id, upload_date),
    INDEX idx_processing_status (processing_status)
);

-- NEW: Document chunks for RAG
CREATE TABLE IF NOT EXISTS document_chunks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    file_id INT NOT NULL,
    chunk_index INT NOT NULL,
    chunk_text LONGTEXT NOT NULL,
    chunk_embedding JSON,  -- Vector embedding stored as JSON
    metadata JSON,         -- Additional metadata (page, section, etc.)
    word_count INT DEFAULT 0,
    char_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_id) REFERENCES uploaded_files(id) ON DELETE CASCADE,
    INDEX idx_session_id (session_id),
    INDEX idx_file_id (file_id),
    INDEX idx_session_file (session_id, file_id),
    INDEX idx_chunk_index (chunk_index),
    UNIQUE KEY unique_file_chunk (file_id, chunk_index)
);

-- NEW: Q&A interactions
CREATE TABLE IF NOT EXISTS qa_interactions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    question TEXT NOT NULL,
    answer LONGTEXT NOT NULL,
    source_chunks JSON,           -- Array of chunk IDs used for answer
    confidence_score FLOAT DEFAULT 0.0,
    processing_time_ms INT DEFAULT 0,
    model_used VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session_id (session_id),
    INDEX idx_created_at (created_at),
    INDEX idx_confidence (confidence_score),
    INDEX idx_session_confidence (session_id, confidence_score)
);

-- NEW: Q&A feedback for improvement
CREATE TABLE IF NOT EXISTS qa_feedback (
    id INT AUTO_INCREMENT PRIMARY KEY,
    interaction_id INT NOT NULL,
    rating INT NOT NULL CHECK (rating BETWEEN 1 AND 5),
    feedback_text TEXT,
    is_helpful BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (interaction_id) REFERENCES qa_interactions(id) ON DELETE CASCADE,
    INDEX idx_interaction_id (interaction_id),
    INDEX idx_rating (rating),
    INDEX idx_created_at (created_at)
);

-- NEW: Document analysis results
CREATE TABLE IF NOT EXISTS document_analysis (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_id INT NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    analysis_type ENUM('summary', 'classification', 'entities', 'sentiment', 'topics') NOT NULL,
    analysis_results JSON NOT NULL,
    confidence_score FLOAT DEFAULT 0.0,
    model_used VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_id) REFERENCES uploaded_files(id) ON DELETE CASCADE,
    INDEX idx_file_id (file_id),
    INDEX idx_session_id (session_id),
    INDEX idx_analysis_type (analysis_type),
    INDEX idx_confidence (confidence_score)
);

-- NEW: Session analytics
CREATE TABLE IF NOT EXISTS session_analytics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    total_documents INT DEFAULT 0,
    total_chunks INT DEFAULT 0,
    total_questions INT DEFAULT 0,
    avg_confidence FLOAT DEFAULT 0.0,
    total_chat_messages INT DEFAULT 0,
    session_duration_minutes INT DEFAULT 0,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_session_id (session_id),
    INDEX idx_last_activity (last_activity)
);

-- Enhanced views for analytics
CREATE OR REPLACE VIEW conversation_summary AS
SELECT 
    c.session_id,
    COUNT(*) as total_messages,
    COUNT(CASE WHEN c.conversation_type = 'qa' THEN 1 END) as qa_messages,
    COUNT(CASE WHEN c.conversation_type = 'chat' THEN 1 END) as chat_messages,
    AVG(c.response_time_ms) as avg_response_time_ms,
    MIN(c.created_at) as first_message,
    MAX(c.created_at) as last_message,
    COUNT(CASE WHEN c.has_sources = TRUE THEN 1 END) as messages_with_sources
FROM conversations c
GROUP BY c.session_id;

-- Enhanced file summary view
CREATE OR REPLACE VIEW file_summary AS
SELECT 
    uf.session_id,
    COUNT(*) as file_count,
    SUM(uf.file_size) as total_size,
    COUNT(CASE WHEN uf.extracted_text IS NOT NULL AND uf.extracted_text != '' THEN 1 END) as files_with_text,
    COUNT(CASE WHEN uf.is_processed = TRUE THEN 1 END) as processed_files,
    AVG(CASE WHEN uf.processed_date IS NOT NULL 
        THEN TIMESTAMPDIFF(SECOND, uf.upload_date, uf.processed_date) 
        ELSE NULL END) as avg_processing_time_seconds
FROM uploaded_files uf
GROUP BY uf.session_id;

-- Q&A performance view
CREATE OR REPLACE VIEW qa_performance AS
SELECT 
    qi.session_id,
    COUNT(*) as total_qa_interactions,
    AVG(qi.confidence_score) as avg_confidence,
    AVG(qi.processing_time_ms) as avg_processing_time_ms,
    COUNT(CASE WHEN qi.confidence_score >= 0.8 THEN 1 END) as high_confidence_answers,
    MAX(qi.created_at) as last_qa_interaction
FROM qa_interactions qi
GROUP BY qi.session_id;

-- Document chunks summary view
CREATE OR REPLACE VIEW chunks_summary AS
SELECT 
    dc.session_id,
    dc.file_id,
    uf.original_name as filename,
    COUNT(*) as chunk_count,
    AVG(dc.word_count) as avg_words_per_chunk,
    AVG(dc.char_count) as avg_chars_per_chunk,
    MAX(dc.created_at) as last_chunk_created
FROM document_chunks dc
LEFT JOIN uploaded_files uf ON dc.file_id = uf.id
GROUP BY dc.session_id, dc.file_id, uf.original_name;

-- Comprehensive session overview
CREATE OR REPLACE VIEW session_overview AS
SELECT 
    cs.session_id,
    cs.total_messages,
    cs.qa_messages,
    cs.chat_messages,
    cs.avg_response_time_ms,
    cs.first_message,
    cs.last_message,
    COALESCE(fs.file_count, 0) as uploaded_files,
    COALESCE(fs.processed_files, 0) as processed_files,
    COALESCE(qp.total_qa_interactions, 0) as qa_interactions,
    COALESCE(qp.avg_confidence, 0) as avg_qa_confidence,
    COALESCE(chs.total_chunks, 0) as total_document_chunks
FROM conversation_summary cs
LEFT JOIN file_summary fs ON cs.session_id = fs.session_id
LEFT JOIN qa_performance qp ON cs.session_id = qp.session_id
LEFT JOIN (
    SELECT session_id, COUNT(*) as total_chunks 
    FROM document_chunks 
    GROUP BY session_id
) chs ON cs.session_id = chs.session_id;

-- Insert sample data for testing (optional)
INSERT INTO users (session_id, user_preferences) VALUES 
('demo_session_001', '{"theme": "dark", "language": "en"}')
ON DUPLICATE KEY UPDATE updated_at = CURRENT_TIMESTAMP;

INSERT INTO conversations (session_id, message, response, model_used, conversation_type) VALUES 
('demo_session_001', 'Hello!', 'Hi there! How can I help you today? You can upload documents for intelligent Q&A or just chat with me.', 'Claude 3.5 Sonnet', 'chat')
ON DUPLICATE KEY UPDATE created_at = created_at;

-- Create indexes for better performance
CREATE INDEX idx_chunks_session_embedding ON document_chunks(session_id, id);
CREATE INDEX idx_qa_session_confidence ON qa_interactions(session_id, confidence_score DESC);
CREATE INDEX idx_files_session_processed ON uploaded_files(session_id, is_processed);

-- Create full-text search indexes for better text search
ALTER TABLE document_chunks ADD FULLTEXT(chunk_text);
ALTER TABLE qa_interactions ADD FULLTEXT(question, answer);

-- Procedures for analytics
DELIMITER //

CREATE PROCEDURE GetSessionStatistics(IN session_id VARCHAR(255))
BEGIN
    SELECT 
        so.*,
        TIMESTAMPDIFF(MINUTE, so.first_message, so.last_message) as session_duration_minutes
    FROM session_overview so
    WHERE so.session_id = session_id;
END //

CREATE PROCEDURE GetTopQAQuestions(IN session_id VARCHAR(255), IN limit_count INT)
BEGIN
    SELECT 
        qi.question,
        qi.confidence_score,
        qi.processing_time_ms,
        qi.created_at,
        COALESCE(qf.avg_rating, 0) as avg_rating
    FROM qa_interactions qi
    LEFT JOIN (
        SELECT interaction_id, AVG(rating) as avg_rating
        FROM qa_feedback
        GROUP BY interaction_id
    ) qf ON qi.id = qf.interaction_id
    WHERE qi.session_id = session_id
    ORDER BY qi.confidence_score DESC, qi.created_at DESC
    LIMIT limit_count;
END //

CREATE PROCEDURE CleanupOldSessions(IN days_old INT)
BEGIN
    -- Clean up sessions older than specified days
    DELETE FROM conversations 
    WHERE created_at < DATE_SUB(NOW(), INTERVAL days_old DAY);
    
    DELETE FROM qa_interactions 
    WHERE created_at < DATE_SUB(NOW(), INTERVAL days_old DAY);
    
    DELETE FROM uploaded_files 
    WHERE upload_date < DATE_SUB(NOW(), INTERVAL days_old DAY);
END //

DELIMITER ;

-- Final optimizations
OPTIMIZE TABLE conversations;
OPTIMIZE TABLE uploaded_files;
OPTIMIZE TABLE document_chunks;
OPTIMIZE TABLE qa_interactions;

-- Grant permissions
GRANT ALL PRIVILEGES ON bedrock_chat.* TO 'bedrock_user'@'%';
FLUSH PRIVILEGES;

-- database/init.sql - Fixed initialization script
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

-- Document chunks for RAG (fixed table structure)
CREATE TABLE IF NOT EXISTS document_chunks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    file_id INT NOT NULL,
    chunk_index INT NOT NULL,
    chunk_text LONGTEXT NOT NULL,
    chunk_embedding LONGTEXT,  -- Store as JSON string
    metadata JSON,
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

-- Q&A interactions
CREATE TABLE IF NOT EXISTS qa_interactions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    question TEXT NOT NULL,
    answer LONGTEXT NOT NULL,
    source_chunks JSON,
    confidence_score FLOAT DEFAULT 0.0,
    processing_time_ms INT DEFAULT 0,
    model_used VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session_id (session_id),
    INDEX idx_created_at (created_at),
    INDEX idx_confidence (confidence_score),
    INDEX idx_session_confidence (session_id, confidence_score)
);

-- Q&A feedback
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

-- Document analysis results
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

-- Session analytics
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

-- Grant permissions
GRANT ALL PRIVILEGES ON bedrock_chat.* TO 'bedrock_user'@'%';
FLUSH PRIVILEGES;

-- Insert sample data
INSERT INTO users (session_id, user_preferences) VALUES 
('demo_session_001', '{"theme": "dark", "language": "en"}')
ON DUPLICATE KEY UPDATE updated_at = CURRENT_TIMESTAMP;

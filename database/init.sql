-- Initialize bedrock_chat database
USE bedrock_chat;

-- Create users table for session management
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_session_id (session_id)
);

-- Create conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    response TEXT NOT NULL,
    model_used VARCHAR(100) DEFAULT 'unknown',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session_id (session_id),
    INDEX idx_created_at (created_at),
    INDEX idx_session_created (session_id, created_at)
);

-- Create files table for uploaded files
CREATE TABLE IF NOT EXISTS uploaded_files (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    original_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(100) NOT NULL,
    file_size BIGINT NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    extracted_text LONGTEXT,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session_id (session_id),
    INDEX idx_upload_date (upload_date),
    INDEX idx_session_upload (session_id, upload_date)
);

-- Insert sample data (optional)
-- INSERT INTO users (session_id) VALUES ('sample_session_123');
-- INSERT INTO conversations (session_id, message, response, model_used) VALUES 
-- ('sample_session_123', 'Hello!', 'Hi there! How can I help you today?', 'Claude 3.5 Sonnet');

-- Create a view for conversation summaries
CREATE OR REPLACE VIEW conversation_summary AS
SELECT 
    session_id,
    COUNT(*) as message_count,
    MIN(created_at) as first_message,
    MAX(created_at) as last_message
FROM conversations
GROUP BY session_id;

-- Create a view for file summaries
CREATE OR REPLACE VIEW file_summary AS
SELECT 
    session_id,
    COUNT(*) as file_count,
    SUM(file_size) as total_size,
    COUNT(CASE WHEN extracted_text IS NOT NULL AND extracted_text != '' THEN 1 END) as files_with_text
FROM uploaded_files
GROUP BY session_id;

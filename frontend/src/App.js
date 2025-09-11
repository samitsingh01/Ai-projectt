// frontend/src/App.js - Enhanced with Q&A Features
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState('');
  const [files, setFiles] = useState([]);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [dragOver, setDragOver] = useState(false);
  const [qaMode, setQaMode] = useState(true); // New: Q&A mode toggle
  const [qaStatus, setQaStatus] = useState(null); // New: Q&A processing status
  const [relatedQuestions, setRelatedQuestions] = useState([]); // New: Related questions
  const [showAnalytics, setShowAnalytics] = useState(false); // New: Analytics panel
  const [analytics, setAnalytics] = useState(null); // New: Session analytics
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Generate session ID on component mount
    const newSessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    setSessionId(newSessionId);
    loadConversationHistory(newSessionId);
    loadUploadedFiles(newSessionId);
    loadQAStatus(newSessionId);
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Check Q&A status periodically when files are being processed
    if (qaStatus && !qaStatus.processing_complete && sessionId) {
      const interval = setInterval(() => {
        loadQAStatus(sessionId);
      }, 5000); // Check every 5 seconds
      
      return () => clearInterval(interval);
    }
  }, [qaStatus, sessionId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const loadConversationHistory = async (sessionId) => {
    try {
      const response = await axios.get(`/api/conversation/${sessionId}`);
      if (response.data.history) {
        const formattedMessages = [];
        response.data.history.forEach(item => {
          formattedMessages.push({
            type: 'user',
            content: item.message,
            timestamp: item.created_at
          });
          formattedMessages.push({
            type: 'assistant',
            content: item.response,
            timestamp: item.created_at,
            responseType: item.response_type || 'chat',
            hasSources: item.has_sources || false
          });
        });
        setMessages(formattedMessages);
      }
    } catch (error) {
      console.error('Error loading conversation history:', error);
    }
  };

  const loadUploadedFiles = async (sessionId) => {
    try {
      const response = await axios.get(`/api/files/${sessionId}`);
      setUploadedFiles(response.data.files || []);
    } catch (error) {
      console.error('Error loading uploaded files:', error);
    }
  };

  const loadQAStatus = async (sessionId) => {
    try {
      const response = await axios.get(`/api/qa/status/${sessionId}`);
      setQaStatus(response.data);
      
      // Auto-enable Q&A mode when documents are processed
      if (response.data.processing_complete && response.data.processed_documents > 0) {
        setQaMode(true);
      }
    } catch (error) {
      console.error('Error loading Q&A status:', error);
    }
  };

  const loadAnalytics = async () => {
    try {
      const response = await axios.get(`/api/analytics/${sessionId}`);
      setAnalytics(response.data);
    } catch (error) {
      console.error('Error loading analytics:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!message.trim() && files.length === 0) return;

    const userMessage = message || 'Uploaded files for analysis';
    
    // Add user message to chat
    setMessages(prev => [...prev, {
      type: 'user',
      content: userMessage,
      timestamp: new Date().toISOString()
    }]);

    setLoading(true);

    try {
      // Upload files first if any
      if (files.length > 0) {
        await uploadFiles();
      }

      // Send chat message with Q&A mode
      const result = await axios.post('/api/chat', {
        message: message || 'Please analyze the uploaded files and tell me about their content.',
        session_id: sessionId,
        use_qa: qaMode && uploadedFiles.length > 0
      });

      // Add assistant response to chat
      const newMessage = {
        type: 'assistant',
        content: result.data.response,
        timestamp: new Date().toISOString(),
        modelUsed: result.data.model_used,
        responseType: result.data.response_type,
        sources: result.data.sources,
        confidence: result.data.confidence,
        processingTime: result.data.processing_time
      };

      setMessages(prev => [...prev, newMessage]);

      // Set related questions if provided
      if (result.data.related_questions) {
        setRelatedQuestions(result.data.related_questions);
      }

    } catch (error) {
      setMessages(prev => [...prev, {
        type: 'error',
        content: 'Error: Unable to get response. Please try again.',
        timestamp: new Date().toISOString()
      }]);
      console.error('Error:', error);
    } finally {
      setMessage('');
      setFiles([]);
      setLoading(false);
    }
  };

  const handleQuestionClick = (question) => {
    setMessage(question);
    setRelatedQuestions([]);
  };

  const uploadFiles = async () => {
    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });
      formData.append('session_id', sessionId);

      const response = await axios.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Refresh uploaded files list and Q&A status
      loadUploadedFiles(sessionId);
      loadQAStatus(sessionId);
      
      // Add upload confirmation to chat
      setMessages(prev => [...prev, {
        type: 'system',
        content: `✅ Successfully uploaded ${files.length} file(s): ${files.map(f => f.name).join(', ')}. Q&A processing started.`,
        timestamp: new Date().toISOString()
      }]);

    } catch (error) {
      setMessages(prev => [...prev, {
        type: 'error',
        content: `❌ File upload failed: ${error.response?.data?.detail || error.message}`,
        timestamp: new Date().toISOString()
      }]);
      console.error('Upload error:', error);
    }
  };

  const handleFileSelect = (selectedFiles) => {
    const fileArray = Array.from(selectedFiles);
    const validFiles = fileArray.filter(file => {
      const validTypes = ['.pdf', '.txt', '.docx', '.csv', '.json', '.md'];
      const fileExt = '.' + file.name.split('.').pop().toLowerCase();
      const isValidSize = file.size <= 10 * 1024 * 1024; // 10MB
      const isValidType = validTypes.includes(fileExt);
      
      if (!isValidSize) {
        alert(`File ${file.name} is too large. Maximum size is 10MB.`);
        return false;
      }
      if (!isValidType) {
        alert(`File ${file.name} has unsupported format. Supported: ${validTypes.join(', ')}`);
        return false;
      }
      return true;
    });
    
    setFiles(prev => [...prev, ...validFiles]);
  };

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragOver(true);
    } else if (e.type === "dragleave") {
      setDragOver(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    
    if (e.dataTransfer.files) {
      handleFileSelect(e.dataTransfer.files);
    }
  };

  const clearConversation = async () => {
    if (window.confirm('Are you sure you want to clear the conversation history?')) {
      try {
        await axios.delete(`/api/conversation/${sessionId}`);
        setMessages([]);
        setRelatedQuestions([]);
      } catch (error) {
        console.error('Error clearing conversation:', error);
      }
    }
  };

  const toggleAnalytics = () => {
    if (!showAnalytics) {
      loadAnalytics();
    }
    setShowAnalytics(!showAnalytics);
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getResponseTypeIcon = (responseType) => {
    switch (responseType) {
      case 'qa': return '🧠';
      case 'analysis': return '📊';
      default: return '💬';
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <div className="chat-header">
          <h1>🤖 AI Chat with Intelligent Q&A</h1>
          <div className="header-controls">
            <div className="qa-toggle">
              <label>
                <input
                  type="checkbox"
                  checked={qaMode}
                  onChange={(e) => setQaMode(e.target.checked)}
                  disabled={uploadedFiles.length === 0}
                />
                Q&A Mode {uploadedFiles.length === 0 && '(Upload documents first)'}
              </label>
            </div>
            <div className="session-info">
              <span>Session: {sessionId.slice(-8)}</span>
              <button onClick={toggleAnalytics} className="analytics-btn">
                📊 Analytics
              </button>
              <button onClick={clearConversation} className="clear-btn">Clear Chat</button>
            </div>
          </div>
        </div>

        {/* Q&A Status Bar */}
        {qaStatus && (
          <div className="qa-status-bar">
            <div className="status-info">
              📄 Documents: {qaStatus.processed_documents}/{qaStatus.total_documents}
              {qaStatus.processing_complete ? 
                <span className="status-ready"> ✅ Ready for Q&A</span> : 
                <span className="status-processing"> ⏳ Processing...</span>
              }
              {qaStatus.readiness_percentage > 0 && (
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{width: `${qaStatus.readiness_percentage}%`}}
                  ></div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Analytics Panel */}
        {showAnalytics && analytics && (
          <div className="analytics-panel">
            <h3>📊 Session Analytics</h3>
            <div className="analytics-grid">
              <div className="stat-card">
                <h4>Conversations</h4>
                <p>Total: {analytics.conversation_analytics?.total_messages || 0}</p>
                <p>Q&A: {analytics.conversation_analytics?.qa_messages || 0}</p>
                <p>Chat: {analytics.conversation_analytics?.chat_messages || 0}</p>
              </div>
              <div className="stat-card">
                <h4>Documents</h4>
                <p>Processed: {analytics.document_analytics?.documents_processed || 0}</p>
                <p>Chunks: {analytics.document_analytics?.total_chunks || 0}</p>
              </div>
              <div className="stat-card">
                <h4>Performance</h4>
                <p>Avg Response: {Math.round(analytics.conversation_analytics?.avg_response_time_ms || 0)}ms</p>
                <p>Avg Confidence: {Math.round((analytics.qa_analytics?.avg_confidence || 0) * 100)}%</p>
              </div>
            </div>
          </div>
        )}

        <div className="chat-container">
          <div className="chat-messages">
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.type}`}>
                <div className="message-content">
                  <div className="message-text">{msg.content}</div>
                  
                  {/* Sources for Q&A responses */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="sources-section">
                      <h4>📚 Sources:</h4>
                      {msg.sources.map((source, idx) => (
                        <div key={idx} className="source-item">
                          <strong>{source.document}</strong> (p. {source.page})
                          <div className="source-snippet">"{source.snippet}"</div>
                          <div className="source-meta">
                            Relevance: {Math.round(source.relevance_score * 100)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  
                  <div className="message-meta">
                    <span className="timestamp">{formatTimestamp(msg.timestamp)}</span>
                    {msg.responseType && (
                      <span className="response-type">
                        {getResponseTypeIcon(msg.responseType)} {msg.responseType}
                      </span>
                    )}
                    {msg.confidence && (
                      <span className="confidence">
                        Confidence: {Math.round(msg.confidence * 100)}%
                      </span>
                    )}
                    {msg.modelUsed && <span className="model">via {msg.modelUsed}</span>}
                    {msg.processingTime && (
                      <span className="processing-time">
                        {msg.processingTime.toFixed(2)}s
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
            
            {/* Related Questions */}
            {relatedQuestions.length > 0 && (
              <div className="related-questions">
                <h4>💡 Related Questions:</h4>
                {relatedQuestions.map((question, idx) => (
                  <button
                    key={idx}
                    className="related-question-btn"
                    onClick={() => handleQuestionClick(question)}
                  >
                    {question}
                  </button>
                ))}
              </div>
            )}
            
            {loading && (
              <div className="message assistant">
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="upload-section">
            {uploadedFiles.length > 0 && (
              <div className="uploaded-files">
                <h4>📁 Uploaded Files ({uploadedFiles.length})</h4>
                <div className="file-list">
                  {uploadedFiles.map((file, index) => (
                    <span key={index} className="uploaded-file-tag">
                      {file.filename} ({(file.file_size / 1024).toFixed(1)}KB)
                      {file.has_text && <span className="text-indicator">📄</span>}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {files.length > 0 && (
              <div className="selected-files">
                <h4>📎 Selected Files</h4>
                <div className="file-list">
                  {files.map((file, index) => (
                    <div key={index} className="selected-file">
                      <span>{file.name} ({(file.size / 1024).toFixed(1)}KB)</span>
                      <button onClick={() => removeFile(index)} className="remove-file">×</button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div 
              className={`file-drop-zone ${dragOver ? 'drag-over' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="drop-zone-content">
                <span className="drop-icon">📁</span>
                <p>Drop files here or click to select</p>
                <p className="file-info">Supported: PDF, TXT, DOCX, CSV, JSON, MD (max 10MB each)</p>
                {qaMode && <p className="qa-info">Q&A Mode: Files will be processed for intelligent questioning</p>}
              </div>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf,.txt,.docx,.csv,.json,.md"
                onChange={(e) => handleFileSelect(e.target.files)}
                style={{ display: 'none' }}
              />
            </div>
          </div>

          <form onSubmit={handleSubmit} className="chat-form">
            <div className="input-container">
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder={
                  qaMode && uploadedFiles.length > 0 
                    ? "Ask me anything about your uploaded documents..." 
                    : "Ask me anything or upload documents for intelligent Q&A..."
                }
                rows="3"
                disabled={loading}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
              />
              <button type="submit" disabled={loading} className="send-btn">
                {loading ? '⏳' : qaMode && uploadedFiles.length > 0 ? '🧠' : '🚀'}
              </button>
            </div>
            <div className="form-help">
              <span>
                Press Enter to send, Shift+Enter for new line | 
                {qaMode && uploadedFiles.length > 0 ? ' Q&A Mode Active' : ' Chat Mode'}
              </span>
            </div>
          </form>
        </div>
      </header>
    </div>
  );
}

export default App;

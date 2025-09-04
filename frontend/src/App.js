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
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Generate session ID on component mount
    const newSessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    setSessionId(newSessionId);
    loadConversationHistory(newSessionId);
    loadUploadedFiles(newSessionId);
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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
            timestamp: item.created_at
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

      // Send chat message
      const result = await axios.post('/api/chat', {
        message: message || 'Please analyze the uploaded files and tell me about their content.',
        session_id: sessionId
      });

      // Add assistant response to chat
      setMessages(prev => [...prev, {
        type: 'assistant',
        content: result.data.response,
        timestamp: new Date().toISOString(),
        modelUsed: result.data.model_used
      }]);

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

      // Refresh uploaded files list
      loadUploadedFiles(sessionId);
      
      // Add upload confirmation to chat
      setMessages(prev => [...prev, {
        type: 'system',
        content: `✅ Successfully uploaded ${files.length} file(s): ${files.map(f => f.name).join(', ')}`,
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
      } catch (error) {
        console.error('Error clearing conversation:', error);
      }
    }
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  return (
    <div className="App">
      <header className="App-header">
        <div className="chat-header">
          <h1>🤖 AI Chat with File Analysis</h1>
          <div className="session-info">
            <span>Session: {sessionId.slice(-8)}</span>
            <button onClick={clearConversation} className="clear-btn">Clear Chat</button>
          </div>
        </div>

        <div className="chat-container">
          <div className="chat-messages">
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.type}`}>
                <div className="message-content">
                  <div className="message-text">{msg.content}</div>
                  <div className="message-meta">
                    <span className="timestamp">{formatTimestamp(msg.timestamp)}</span>
                    {msg.modelUsed && <span className="model">via {msg.modelUsed}</span>}
                  </div>
                </div>
              </div>
            ))}
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
                      {file.original_name} ({(file.file_size / 1024).toFixed(1)}KB)
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
                placeholder="Ask me anything about your files or general questions..."
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
                {loading ? '⏳' : '🚀'}
              </button>
            </div>
            <div className="form-help">
              <span>Press Enter to send, Shift+Enter for new line</span>
            </div>
          </form>
        </div>
      </header>
    </div>
  );
}

export default App;

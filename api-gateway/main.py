# api-gateway/main.py - Enhanced with Q&A Service Integration
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
import logging
import mysql.connector
from mysql.connector import Error
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Enhanced API Gateway", version="4.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs
BEDROCK_SERVICE_URL = os.getenv("BEDROCK_SERVICE_URL", "http://bedrock-service:9000")
FILE_SERVICE_URL = os.getenv("FILE_SERVICE_URL", "http://file-service:7000")
QA_SERVICE_URL = os.getenv("QA_SERVICE_URL", "http://intelligent-qa-service:6000")
DATABASE_URL = os.getenv("DATABASE_URL", "mysql://bedrock_user:bedrock_password@mysql:3306/bedrock_chat")

# Database connection function
def get_db_connection():
    try:
        url_parts = DATABASE_URL.replace("mysql://", "").split("/")
        auth_host = url_parts[0].split("@")
        auth = auth_host[0].split(":")
        host_port = auth_host[1].split(":")
        
        connection = mysql.connector.connect(
            host=host_port[0],
            port=int(host_port[1]) if len(host_port) > 1 else 3306,
            user=auth[0],
            password=auth[1],
            database=url_parts[1]
        )
        return connection
    except Error as e:
        logger.error(f"Database connection error: {e}")
        return None

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_qa: Optional[bool] = True  # New: Enable Q&A mode

class EnhancedChatResponse(BaseModel):
    response: str
    session_id: str
    model_used: str
    response_type: str  # 'chat', 'qa', 'analysis'
    sources: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None
    related_questions: Optional[List[str]] = None
    processing_time: Optional[float] = None

class QARequest(BaseModel):
    question: str
    session_id: str
    max_sources: Optional[int] = 5

class ConversationHistory(BaseModel):
    message: str
    response: str
    created_at: str
    response_type: str
    has_sources: bool

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Enhanced API Gateway with Intelligent Q&A", 
        "version": "4.0.0",
        "features": ["conversation-memory", "file-upload", "bedrock-integration", 
                    "intelligent-qa", "document-rag", "citation-tracking"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "enhanced-api-gateway"}

@app.post("/chat", response_model=EnhancedChatResponse)
async def enhanced_chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Enhanced chat endpoint with intelligent Q&A capabilities
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"Enhanced chat request for session: {session_id}")
        logger.info(f"Message: {request.message[:50]}... | Q&A Mode: {request.use_qa}")
        
        # Check if user has uploaded documents and wants Q&A
        has_documents = await check_session_documents(session_id)
        
        if request.use_qa and has_documents:
            # Use Q&A service for document-based responses
            logger.info("Using Q&A service for document-based response")
            return await handle_qa_request(session_id, request.message, start_time)
        else:
            # Use traditional chat
            logger.info("Using traditional chat service")
            return await handle_traditional_chat(session_id, request.message, start_time)
            
    except Exception as e:
        logger.error(f"Error in enhanced chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/qa/ask", response_model=EnhancedChatResponse)
async def ask_question_about_documents(request: QARequest):
    """
    Direct Q&A endpoint for asking questions about uploaded documents
    """
    try:
        start_time = datetime.now()
        
        # Forward request to Q&A service
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{QA_SERVICE_URL}/qa/ask",
                json={
                    "question": request.question,
                    "session_id": request.session_id,
                    "max_sources": request.max_sources
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Q&A service error")
            
            qa_result = response.json()
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Store in conversations table
            store_enhanced_conversation(
                session_id=request.session_id,
                message=request.question,
                response=qa_result["answer"],
                model_used=qa_result.get("model_used", "qa-system"),
                response_type="qa",
                has_sources=len(qa_result.get("sources", [])) > 0,
                response_time_ms=int(processing_time * 1000)
            )
            
            return EnhancedChatResponse(
                response=qa_result["answer"],
                session_id=request.session_id,
                model_used="intelligent-qa-system",
                response_type="qa",
                sources=qa_result.get("sources", []),
                confidence=qa_result.get("confidence"),
                related_questions=qa_result.get("related_questions", []),
                processing_time=processing_time
            )
            
    except Exception as e:
        logger.error(f"Error in Q&A request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(
    files: List[UploadFile] = File(...),
    session_id: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """
    Enhanced file upload with automatic Q&A processing
    """
    try:
        logger.info(f"Uploading {len(files)} files for session: {session_id}")
        
        # Upload files via file service
        async with httpx.AsyncClient(timeout=120.0) as client:
            files_data = []
            for file in files:
                files_data.append(
                    ("files", (file.filename, await file.read(), file.content_type))
                )
            
            response = await client.post(
                f"{FILE_SERVICE_URL}/upload",
                files=files_data,
                data={"session_id": session_id}
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="File upload error")
            
            upload_result = response.json()
            
            # Process files for Q&A in background
            if background_tasks:
                for file_info in upload_result.get("files", []):
                    background_tasks.add_task(
                        process_file_for_qa,
                        session_id,
                        file_info["id"]
                    )
            
            return {
                **upload_result,
                "qa_processing": "started",
                "message": f"Uploaded {len(files)} files. Q&A processing started in background."
            }
            
    except Exception as e:
        logger.error(f"Enhanced file upload error: {str(e)}")
        raise HTTPException(status_code=500, detail="File upload failed")

@app.get("/qa/status/{session_id}")
async def get_qa_processing_status(session_id: str):
    """
    Get Q&A processing status for session documents
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{QA_SERVICE_URL}/documents/{session_id}/status")
            return response.json()
    except Exception as e:
        logger.error(f"Error getting Q&A status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get Q&A status")

@app.get("/conversation/{session_id}")
async def get_enhanced_conversation(session_id: str):
    """
    Get enhanced conversation history with Q&A metadata
    """
    try:
        history = get_enhanced_conversation_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"Error retrieving conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation")

@app.get("/analytics/{session_id}")
async def get_session_analytics(session_id: str):
    """
    Get comprehensive session analytics
    """
    try:
        # Get analytics from Q&A service
        async with httpx.AsyncClient() as client:
            qa_response = await client.get(f"{QA_SERVICE_URL}/analytics/{session_id}")
            qa_analytics = qa_response.json() if qa_response.status_code == 200 else {}
        
        # Get basic conversation stats
        conversation_stats = get_conversation_stats(session_id)
        
        return {
            "session_id": session_id,
            "conversation_analytics": conversation_stats,
            **qa_analytics
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get analytics")

@app.get("/services/status")
async def enhanced_services_status():
    """Check the status of all services including Q&A"""
    services = {
        "api-gateway": "healthy",
        "bedrock-service": "unknown",
        "file-service": "unknown",
        "intelligent-qa-service": "unknown",
        "database": "unknown"
    }
    
    # Check all services
    service_checks = [
        (BEDROCK_SERVICE_URL, "bedrock-service"),
        (FILE_SERVICE_URL, "file-service"),
        (QA_SERVICE_URL, "intelligent-qa-service")
    ]
    
    for service_url, service_name in service_checks:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{service_url}/health")
                if response.status_code == 200:
                    services[service_name] = "healthy"
        except:
            services[service_name] = "unreachable"
    
    # Check database
    try:
        connection = get_db_connection()
        if connection:
            connection.close()
            services["database"] = "healthy"
    except:
        services["database"] = "unreachable"
    
    return {"services": services}

# Helper functions
async def check_session_documents(session_id: str) -> bool:
    """Check if session has uploaded documents"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{FILE_SERVICE_URL}/files/{session_id}")
            if response.status_code == 200:
                files_data = response.json()
                return len(files_data.get("files", [])) > 0
    except:
        pass
    return False

async def handle_qa_request(session_id: str, message: str, start_time: datetime) -> EnhancedChatResponse:
    """Handle request using Q&A service"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{QA_SERVICE_URL}/qa/ask",
                json={
                    "question": message,
                    "session_id": session_id,
                    "max_sources": 5
                }
            )
            
            if response.status_code == 200:
                qa_result = response.json()
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Store conversation
                store_enhanced_conversation(
                    session_id=session_id,
                    message=message,
                    response=qa_result["answer"],
                    model_used="intelligent-qa-system",
                    response_type="qa",
                    has_sources=len(qa_result.get("sources", [])) > 0,
                    response_time_ms=int(processing_time * 1000)
                )
                
                return EnhancedChatResponse(
                    response=qa_result["answer"],
                    session_id=session_id,
                    model_used="intelligent-qa-system",
                    response_type="qa",
                    sources=qa_result.get("sources", []),
                    confidence=qa_result.get("confidence"),
                    related_questions=qa_result.get("related_questions", []),
                    processing_time=processing_time
                )
        
        # Fallback to traditional chat if Q&A fails
        logger.warning("Q&A service failed, falling back to traditional chat")
        return await handle_traditional_chat(session_id, message, start_time)
        
    except Exception as e:
        logger.error(f"Q&A request failed: {e}")
        return await handle_traditional_chat(session_id, message, start_time)

async def handle_traditional_chat(session_id: str, message: str, start_time: datetime) -> EnhancedChatResponse:
    """Handle request using traditional chat"""
    try:
        # Get conversation history
        conversation_history = get_enhanced_conversation_history(session_id)
        
        # Get uploaded files for context
        uploaded_files = []
        try:
            async with httpx.AsyncClient() as client:
                files_response = await client.get(f"{FILE_SERVICE_URL}/files/{session_id}")
                if files_response.status_code == 200:
                    files_data = files_response.json()
                    uploaded_files = files_data.get("files", [])
        except Exception as e:
            logger.error(f"Error retrieving files: {e}")
        
        # Build context
        context_message = build_enhanced_context_message(message, conversation_history, uploaded_files)
        
        # Call Bedrock service
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{BEDROCK_SERVICE_URL}/generate",
                json={
                    "prompt": context_message,
                    "max_tokens": 1000,
                    "temperature": 0.7
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Bedrock service error")
            
            result = response.json()
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Store conversation
            store_enhanced_conversation(
                session_id=session_id,
                message=message,
                response=result["response"],
                model_used=result.get("model_used", "unknown"),
                response_type="chat",
                has_sources=len(uploaded_files) > 0,
                response_time_ms=int(processing_time * 1000)
            )
            
            return EnhancedChatResponse(
                response=result["response"],
                session_id=session_id,
                model_used=result.get("model_used", "unknown"),
                response_type="chat",
                processing_time=processing_time
            )
            
    except Exception as e:
        logger.error(f"Traditional chat failed: {e}")
        raise HTTPException(status_code=500, detail="Chat service failed")

async def process_file_for_qa(session_id: str, file_id: int):
    """Background task to process file for Q&A"""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            await client.post(
                f"{QA_SERVICE_URL}/documents/process",
                json={
                    "session_id": session_id,
                    "file_id": file_id,
                    "force_reprocess": False
                }
            )
        logger.info(f"Started Q&A processing for file {file_id}")
    except Exception as e:
        logger.error(f"Error processing file {file_id} for Q&A: {e}")

def store_enhanced_conversation(session_id: str, message: str, response: str, 
                              model_used: str, response_type: str, has_sources: bool, 
                              response_time_ms: int):
    """Store enhanced conversation with metadata"""
    try:
        connection = get_db_connection()
        if not connection:
            return
        
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO conversations 
            (session_id, message, response, model_used, conversation_type, has_sources, response_time_ms) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (session_id, message, response, model_used, response_type, has_sources, response_time_ms))
        
        connection.commit()
        cursor.close()
        connection.close()
        
    except Exception as e:
        logger.error(f"Error storing enhanced conversation: {str(e)}")

def get_enhanced_conversation_history(session_id: str) -> List[ConversationHistory]:
    """Get enhanced conversation history"""
    try:
        connection = get_db_connection()
        if not connection:
            return []
        
        cursor = connection.cursor()
        cursor.execute("""
            SELECT message, response, created_at, conversation_type, has_sources
            FROM conversations 
            WHERE session_id = %s 
            ORDER BY created_at ASC 
            LIMIT 20
        """, (session_id,))
        
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        
        return [
            ConversationHistory(
                message=row[0],
                response=row[1],
                created_at=row[2].isoformat(),
                response_type=row[3] or "chat",
                has_sources=bool(row[4])
            ) for row in results
        ]
        
    except Exception as e:
        logger.error(f"Error getting enhanced conversation history: {str(e)}")
        return []

def get_conversation_stats(session_id: str) -> Dict:
    """Get conversation statistics"""
    try:
        connection = get_db_connection()
        if not connection:
            return {}
        
        cursor = connection.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total_messages,
                COUNT(CASE WHEN conversation_type = 'qa' THEN 1 END) as qa_messages,
                COUNT(CASE WHEN conversation_type = 'chat' THEN 1 END) as chat_messages,
                AVG(response_time_ms) as avg_response_time,
                MAX(created_at) as last_activity
            FROM conversations 
            WHERE session_id = %s
        """, (session_id,))
        
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if result:
            return {
                "total_messages": result[0] or 0,
                "qa_messages": result[1] or 0,
                "chat_messages": result[2] or 0,
                "avg_response_time_ms": float(result[3]) if result[3] else 0.0,
                "last_activity": result[4].isoformat() if result[4] else None
            }
        return {}
        
    except Exception as e:
        logger.error(f"Error getting conversation stats: {e}")
        return {}

def build_enhanced_context_message(current_message: str, history: List[ConversationHistory], files: list) -> str:
    """Build enhanced context message"""
    context_parts = []
    
    # Add conversation history
    if history:
        context_parts.append("Previous conversation context:")
        for conv in history[-5:]:
            context_parts.append(f"User: {conv.message}")
            context_parts.append(f"Assistant ({conv.response_type}): {conv.response}")
        context_parts.append("")
    
    # Add file information
    if files:
        context_parts.append("Available documents for reference:")
        for file_info in files:
            filename = file_info.get('filename', 'Unknown file')
            has_text = file_info.get('has_text', False)
            context_parts.append(f"- {filename} {'(processed)' if has_text else '(raw)'}")
        context_parts.append("Note: For detailed document analysis, use Q&A mode.\n")
    
    # Add current message
    context_parts.append(f"Current question: {current_message}")
    
    return "\n".join(context_parts)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

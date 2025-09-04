from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
import logging
import mysql.connector
from mysql.connector import Error
import uuid
from datetime import datetime
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="API Gateway", version="3.0.0")

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
DATABASE_URL = os.getenv("DATABASE_URL", "mysql://bedrock_user:bedrock_password@mysql:3306/bedrock_chat")

# Database connection function
def get_db_connection():
    try:
        # Parse DATABASE_URL
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

class ChatResponse(BaseModel):
    response: str
    session_id: str
    model_used: str

class ConversationHistory(BaseModel):
    message: str
    response: str
    created_at: str

class HealthResponse(BaseModel):
    status: str
    service: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "API Gateway is running", 
        "version": "3.0.0",
        "features": ["conversation-memory", "file-upload", "bedrock-integration", "file-analysis"]
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "api-gateway"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with conversation memory and file analysis
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        logger.info(f"Received chat request for session: {session_id}")
        logger.info(f"Message: {request.message[:50]}...")
        
        # Get conversation history
        conversation_history = get_conversation_history(session_id)
        
        # Get uploaded files for this session
        uploaded_files = []
        try:
            async with httpx.AsyncClient() as client:
                files_response = await client.get(f"{FILE_SERVICE_URL}/files/{session_id}")
                if files_response.status_code == 200:
                    files_data = files_response.json()
                    uploaded_files = files_data.get("files", [])
                    logger.info(f"Found {len(uploaded_files)} files for session {session_id}")
                else:
                    logger.warning(f"Could not retrieve files for session {session_id}: {files_response.status_code}")
        except Exception as e:
            logger.error(f"Error retrieving files: {e}")
        
        # Prepare context with history and file content
        context_message = build_context_message_with_files(
            request.message, 
            conversation_history, 
            uploaded_files
        )
        
        # Log the enhanced context being sent (truncated for readability)
        logger.info(f"Enhanced prompt length: {len(context_message)} characters")
        if uploaded_files:
            logger.info(f"Including {len(uploaded_files)} files in context")
        
        # Forward to bedrock service
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
                logger.error(f"Bedrock service returned status {response.status_code}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Error from bedrock service"
                )
            
            result = response.json()
            ai_response = result.get("response", "No response generated")
            model_used = result.get("model_used", "unknown")
            
            # Store conversation in database
            store_conversation(session_id, request.message, ai_response, model_used)
            
            logger.info(f"Response generated using model: {model_used}")
            
            return ChatResponse(
                response=ai_response,
                session_id=session_id,
                model_used=model_used
            )
            
    except httpx.RequestError as e:
        logger.error(f"Error connecting to bedrock service: {str(e)}")
        raise HTTPException(status_code=503, detail="Bedrock service unavailable")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/upload")
async def upload_file(
    files: List[UploadFile] = File(...),
    session_id: str = Form(...)
):
    """
    Upload files and process them
    """
    try:
        logger.info(f"Uploading {len(files)} files for session: {session_id}")
        
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
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Error from file service"
                )
            
            return response.json()
            
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        raise HTTPException(status_code=500, detail="File upload failed")

@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str):
    """
    Get conversation history for a session
    """
    try:
        history = get_conversation_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"Error retrieving conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation")

@app.get("/files/{session_id}")
async def get_uploaded_files(session_id: str):
    """
    Get uploaded files for a session
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{FILE_SERVICE_URL}/files/{session_id}")
            return response.json()
    except Exception as e:
        logger.error(f"Error retrieving files: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve files")

@app.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    """
    Clear conversation history for a session
    """
    try:
        connection = get_db_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor()
        cursor.execute("DELETE FROM conversations WHERE session_id = %s", (session_id,))
        connection.commit()
        
        cursor.close()
        connection.close()
        
        return {"message": "Conversation cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear conversation")

def get_conversation_history(session_id: str) -> List[ConversationHistory]:
    """
    Get conversation history from database
    """
    try:
        connection = get_db_connection()
        if not connection:
            return []
        
        cursor = connection.cursor()
        cursor.execute("""
            SELECT message, response, created_at 
            FROM conversations 
            WHERE session_id = %s 
            ORDER BY created_at ASC 
            LIMIT 10
        """, (session_id,))
        
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        
        return [
            ConversationHistory(
                message=row[0],
                response=row[1],
                created_at=row[2].isoformat()
            ) for row in results
        ]
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        return []

def store_conversation(session_id: str, message: str, response: str, model_used: str):
    """
    Store conversation in database
    """
    try:
        connection = get_db_connection()
        if not connection:
            return
        
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO conversations (session_id, message, response, model_used) 
            VALUES (%s, %s, %s, %s)
        """, (session_id, message, response, model_used))
        
        connection.commit()
        cursor.close()
        connection.close()
        
    except Exception as e:
        logger.error(f"Error storing conversation: {str(e)}")

def build_context_message_with_files(current_message: str, history: List[ConversationHistory], files: list) -> str:
    """
    Build context message with conversation history and file content
    """
    context_parts = []
    
    # Add conversation history
    if history:
        context_parts.append("Previous conversation context:")
        for conv in history[-5:]:  # Last 5 conversations for context
            context_parts.append(f"User: {conv.message}")
            context_parts.append(f"Assistant: {conv.response}")
        context_parts.append("")  # Empty line for separation
    
    # Add uploaded files content
    if files:
        context_parts.append("Uploaded files for analysis:")
        for file_info in files:
            filename = file_info.get('filename', 'Unknown file')
            content = file_info.get('content', 'No content available')
            file_type = file_info.get('content_type', 'Unknown type')
            file_size = len(content) if content else 0
            
            context_parts.append(f"\n=== FILE: {filename} (Type: {file_type}, Size: {file_size} characters) ===")
            
            # Truncate very large files to prevent token limit issues
            if len(content) > 10000:
                context_parts.append(content[:10000] + "\n...[FILE TRUNCATED DUE TO LENGTH]...")
            else:
                context_parts.append(content)
            
            context_parts.append("=== END OF FILE ===\n")
        
        context_parts.append("")  # Empty line for separation
    
    # Add current message
    context_parts.append(f"Current question: {current_message}")
    
    final_context = "\n".join(context_parts)
    
    # Log context details for debugging
    logger.info(f"Context includes: {len(history)} history items, {len(files)} files")
    
    return final_context

# Legacy function kept for backwards compatibility - remove if not needed elsewhere
def build_context_message(current_message: str, history: List[ConversationHistory]) -> str:
    """
    Legacy function - use build_context_message_with_files instead
    """
    return build_context_message_with_files(current_message, history, [])

@app.get("/services/status")
async def services_status():
    """Check the status of all backend services"""
    services = {
        "api-gateway": "healthy",
        "bedrock-service": "unknown",
        "file-service": "unknown",
        "database": "unknown"
    }
    
    # Check bedrock service
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{BEDROCK_SERVICE_URL}/health")
            if response.status_code == 200:
                services["bedrock-service"] = "healthy"
    except:
        services["bedrock-service"] = "unreachable"
    
    # Check file service
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{FILE_SERVICE_URL}/health")
            if response.status_code == 200:
                services["file-service"] = "healthy"
    except:
        services["file-service"] = "unreachable"
    
    # Check database
    try:
        connection = get_db_connection()
        if connection:
            connection.close()
            services["database"] = "healthy"
    except:
        services["database"] = "unreachable"
    
    return {"services": services}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

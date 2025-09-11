# intelligent-qa-service/main.py - Fixed version with proper service integration
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging
import mysql.connector
from mysql.connector import Error
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import httpx

# Import service classes
from services.document_chunker import DocumentChunker
from services.vector_store import VectorStore
from services.rag_engine import RAGEngine
from services.citation_tracker import CitationTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Intelligent Q&A Service", version="2.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "mysql://bedrock_user:bedrock_password@mysql:3306/bedrock_chat")
BEDROCK_SERVICE_URL = os.getenv("BEDROCK_SERVICE_URL", "http://bedrock-service:9000")
FILE_SERVICE_URL = os.getenv("FILE_SERVICE_URL", "http://file-service:7000")

# Initialize services
document_chunker = DocumentChunker(max_chunk_size=1000)
vector_store = VectorStore(DATABASE_URL)
rag_engine = RAGEngine(BEDROCK_SERVICE_URL)
citation_tracker = CitationTracker()

# Request/Response models
class QARequest(BaseModel):
    question: str
    session_id: str
    max_sources: Optional[int] = 5
    confidence_threshold: Optional[float] = 0.3

class QAResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    related_questions: List[str]
    processing_time: float
    model_used: Optional[str] = "intelligent-qa-system"

class ProcessDocumentRequest(BaseModel):
    session_id: str
    file_id: int
    force_reprocess: Optional[bool] = False

class FeedbackRequest(BaseModel):
    interaction_id: int
    rating: int
    feedback_text: Optional[str] = None

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

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Intelligent Q&A Service is running",
        "version": "2.0.0",
        "features": [
            "document-rag", 
            "cross-document-search", 
            "citation-tracking", 
            "smart-qa",
            "vector-similarity",
            "intelligent-chunking"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with service status"""
    try:
        # Test database connection
        connection = get_db_connection()
        db_status = "healthy" if connection else "unhealthy"
        if connection:
            connection.close()
        
        # Test Bedrock service
        bedrock_status = "unknown"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{BEDROCK_SERVICE_URL}/health")
                bedrock_status = "healthy" if response.status_code == 200 else "unhealthy"
        except:
            bedrock_status = "unreachable"
        
        return {
            "status": "healthy" if db_status == "healthy" else "degraded",
            "service": "intelligent-qa-service",
            "version": "2.0.0",
            "dependencies": {
                "database": db_status,
                "bedrock_service": bedrock_status
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"status": "unhealthy", "error": str(e)}

@app.post("/qa/ask", response_model=QAResponse)
async def ask_question(request: QARequest):
    """Ask a question about uploaded documents using advanced RAG"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing question for session {request.session_id}: {request.question[:50]}...")
        
        # Search for relevant document chunks using vector similarity
        relevant_chunks = await vector_store.search_similar_chunks(
            session_id=request.session_id,
            query=request.question,
            top_k=request.max_sources,
            confidence_threshold=request.confidence_threshold
        )
        
        if not relevant_chunks:
            return QAResponse(
                answer="I couldn't find relevant information in your uploaded documents to answer this question. Please make sure you have uploaded documents and they contain information related to your query.",
                sources=[],
                confidence=0.0,
                related_questions=[],
                processing_time=(datetime.now() - start_time).total_seconds()
            )
        
        # Generate answer using RAG engine
        answer_result = await rag_engine.generate_answer(
            question=request.question,
            context_chunks=relevant_chunks
        )
        
        # Generate citations using citation tracker
        citations = citation_tracker.generate_citations(relevant_chunks, answer_result["answer"])
        
        # Generate related questions
        related_questions = await rag_engine.generate_related_questions(
            question=request.question,
            context_chunks=relevant_chunks[:3]
        )
        
        # Store interaction
        interaction_id = store_qa_interaction(
            session_id=request.session_id,
            question=request.question,
            answer=answer_result["answer"],
            sources=citations,
            confidence=answer_result["confidence"],
            model_used=answer_result.get("model_used", "intelligent-qa-system")
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Generated answer with {len(citations)} sources, confidence: {answer_result['confidence']:.2f}, interaction_id: {interaction_id}")
        
        return QAResponse(
            answer=answer_result["answer"],
            sources=citations,
            confidence=answer_result["confidence"],
            related_questions=related_questions,
            processing_time=processing_time,
            model_used=answer_result.get("model_used", "intelligent-qa-system")
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        processing_time = (datetime.now() - start_time).total_seconds()
        return QAResponse(
            answer=f"I'm sorry, I encountered an error while processing your question: {str(e)}. Please try again.",
            sources=[],
            confidence=0.0,
            related_questions=[],
            processing_time=processing_time
        )

@app.post("/documents/process")
async def process_document(request: ProcessDocumentRequest, background_tasks: BackgroundTasks):
    """Process a document for RAG with improved chunking"""
    try:
        logger.info(f"Processing document {request.file_id} for session {request.session_id}")
        
        # Check if document is already processed
        if not request.force_reprocess:
            existing_chunks = get_document_chunks(request.file_id)
            if existing_chunks:
                return {
                    "message": "Document already processed",
                    "chunks_count": len(existing_chunks),
                    "file_id": request.file_id,
                    "status": "already_processed"
                }
        
        # Get file content from file service
        file_content = await get_file_content(request.file_id)
        if not file_content:
            raise HTTPException(status_code=404, detail="File not found or has no content")
        
        # Process in background with improved services
        background_tasks.add_task(
            process_document_background,
            request.session_id,
            request.file_id,
            file_content
        )
        
        return {
            "message": "Document processing started",
            "file_id": request.file_id,
            "status": "processing",
            "estimated_time": "30-60 seconds"
        }
        
    except Exception as e:
        logger.error(f"Error initiating document processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{session_id}/status")
async def get_processing_status(session_id: str):
    """Get document processing status with detailed information"""
    try:
        connection = get_db_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(dictionary=True)
        
        # Get processed documents count and details
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT dc.file_id) as processed_count,
                SUM(dc.word_count) as total_words,
                COUNT(dc.id) as total_chunks
            FROM document_chunks dc
            WHERE dc.session_id = %s
        """, (session_id,))
        processed_result = cursor.fetchone()
        
        # Get total documents count
        cursor.execute("""
            SELECT COUNT(*) as total_count
            FROM uploaded_files 
            WHERE session_id = %s
        """, (session_id,))
        total_result = cursor.fetchone()
        
        # Get processing details
        cursor.execute("""
            SELECT 
                uf.original_name,
                uf.file_size,
                COUNT(dc.id) as chunk_count,
                SUM(dc.word_count) as word_count
            FROM uploaded_files uf
            LEFT JOIN document_chunks dc ON uf.id = dc.file_id
            WHERE uf.session_id = %s
            GROUP BY uf.id, uf.original_name, uf.file_size
        """, (session_id,))
        file_details = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        processed_count = processed_result['processed_count'] if processed_result else 0
        total_count = total_result['total_count'] if total_result else 0
        
        return {
            "session_id": session_id,
            "processed_documents": processed_count,
            "total_documents": total_count,
            "processing_complete": processed_count >= total_count and total_count > 0,
            "readiness_percentage": (processed_count / total_count * 100) if total_count > 0 else 0,
            "total_chunks": processed_result['total_chunks'] if processed_result else 0,
            "total_words": processed_result['total_words'] if processed_result else 0,
            "file_details": file_details or []
        }
        
    except Exception as e:
        logger.error(f"Error getting processing status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/{session_id}")
async def get_session_analytics(session_id: str):
    """Get comprehensive analytics for a session"""
    try:
        connection = get_db_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(dictionary=True)
        
        # Get Q&A interaction stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_questions,
                AVG(confidence_score) as avg_confidence,
                MAX(confidence_score) as max_confidence,
                MIN(confidence_score) as min_confidence,
                AVG(processing_time_ms) as avg_processing_time,
                MAX(created_at) as last_question
            FROM qa_interactions 
            WHERE session_id = %s
        """, (session_id,))
        qa_result = cursor.fetchone()
        
        # Get document processing stats
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT file_id) as documents_processed,
                COUNT(*) as total_chunks,
                AVG(word_count) as avg_chunk_words,
                AVG(char_count) as avg_chunk_chars,
                SUM(word_count) as total_words
            FROM document_chunks 
            WHERE session_id = %s
        """, (session_id,))
        doc_result = cursor.fetchone()
        
        # Get recent questions
        cursor.execute("""
            SELECT question, confidence_score, created_at
            FROM qa_interactions 
            WHERE session_id = %s 
            ORDER BY created_at DESC 
            LIMIT 5
        """, (session_id,))
        recent_questions = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return {
            "session_id": session_id,
            "qa_analytics": {
                "total_questions": qa_result['total_questions'] if qa_result else 0,
                "avg_confidence": float(qa_result['avg_confidence']) if qa_result and qa_result['avg_confidence'] else 0.0,
                "max_confidence": float(qa_result['max_confidence']) if qa_result and qa_result['max_confidence'] else 0.0,
                "min_confidence": float(qa_result['min_confidence']) if qa_result and qa_result['min_confidence'] else 0.0,
                "avg_processing_time_ms": float(qa_result['avg_processing_time']) if qa_result and qa_result['avg_processing_time'] else 0.0,
                "last_question": qa_result['last_question'].isoformat() if qa_result and qa_result['last_question'] else None
            },
            "document_analytics": {
                "documents_processed": doc_result['documents_processed'] if doc_result else 0,
                "total_chunks": doc_result['total_chunks'] if doc_result else 0,
                "total_words": doc_result['total_words'] if doc_result else 0,
                "avg_chunk_words": float(doc_result['avg_chunk_words']) if doc_result and doc_result['avg_chunk_words'] else 0.0,
                "avg_chunk_chars": float(doc_result['avg_chunk_chars']) if doc_result and doc_result['avg_chunk_chars'] else 0.0
            },
            "recent_questions": [
                {
                    "question": q['question'][:100] + "..." if len(q['question']) > 100 else q['question'],
                    "confidence": float(q['confidence_score']),
                    "timestamp": q['created_at'].isoformat()
                }
                for q in recent_questions
            ] if recent_questions else []
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/qa/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for Q&A interaction"""
    try:
        connection = get_db_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO qa_feedback 
            (interaction_id, rating, feedback_text, is_helpful, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            request.interaction_id, 
            request.rating, 
            request.feedback_text,
            request.rating >= 3,  # Consider 3+ as helpful
            datetime.now()
        ))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info(f"Feedback submitted for interaction {request.interaction_id}: {request.rating}/5")
        
        return {
            "message": "Feedback submitted successfully",
            "interaction_id": request.interaction_id,
            "rating": request.rating
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
async def get_file_content(file_id: int) -> Optional[Dict]:
    """Get file content from file service"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{FILE_SERVICE_URL}/file/content/{file_id}")
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"File service error {response.status_code} for file {file_id}")
    except Exception as e:
        logger.error(f"Error getting file content for file {file_id}: {e}")
    return None

def get_document_chunks(file_id: int) -> List[Dict]:
    """Check if document chunks exist for a file"""
    try:
        connection = get_db_connection()
        if not connection:
            return []
        
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, chunk_index, word_count 
            FROM document_chunks 
            WHERE file_id = %s
        """, (file_id,))
        results = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return results
    except Exception as e:
        logger.error(f"Error checking document chunks for file {file_id}: {e}")
        return []

async def process_document_background(session_id: str, file_id: int, file_content: Dict):
    """Background task to process document with improved services"""
    try:
        logger.info(f"Processing file {file_id} for session {session_id}")
        
        text = file_content.get("content", "")
        filename = file_content.get("filename", "Unknown")
        
        if not text or len(text.strip()) < 50:
            logger.warning(f"File {file_id} has insufficient text content")
            return
        
        # Use document chunker service for intelligent chunking
        chunks = document_chunker.chunk_document(text, filename)
        
        if not chunks:
            logger.warning(f"No chunks created for file {file_id}")
            return
        
        # Process each chunk
        for chunk in chunks:
            # Generate embedding for the chunk
            embedding = await vector_store.generate_embedding(chunk["text"])
            
            # Store chunk with embedding
            store_document_chunk(
                session_id=session_id,
                file_id=file_id,
                chunk_index=chunk["metadata"]["chunk_index"],
                chunk_text=chunk["text"],
                metadata=chunk["metadata"],
                embedding=embedding
            )
        
        logger.info(f"Successfully processed {len(chunks)} chunks for file {file_id}")
        
        # Update file processing status
        update_file_processing_status(file_id, "completed")
        
    except Exception as e:
        logger.error(f"Error processing document {file_id}: {e}")
        update_file_processing_status(file_id, "failed")

def store_document_chunk(session_id: str, file_id: int, chunk_index: int, 
                        chunk_text: str, metadata: Dict, embedding: List[float]):
    """Store document chunk with embedding"""
    try:
        connection = get_db_connection()
        if not connection:
            return
        
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO document_chunks 
            (session_id, file_id, chunk_index, chunk_text, chunk_embedding, metadata, word_count, char_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            chunk_text = VALUES(chunk_text),
            chunk_embedding = VALUES(chunk_embedding),
            metadata = VALUES(metadata),
            word_count = VALUES(word_count),
            char_count = VALUES(char_count)
        """, (
            session_id, 
            file_id, 
            chunk_index, 
            chunk_text, 
            json.dumps(embedding),
            json.dumps(metadata),
            metadata.get("word_count", len(chunk_text.split())),
            metadata.get("char_count", len(chunk_text))
        ))
        
        connection.commit()
        cursor.close()
        connection.close()
        
    except Exception as e:
        logger.error(f"Error storing document chunk: {e}")

def update_file_processing_status(file_id: int, status: str):
    """Update file processing status"""
    try:
        connection = get_db_connection()
        if not connection:
            return
        
        cursor = connection.cursor()
        cursor.execute("""
            UPDATE uploaded_files 
            SET processing_status = %s, processed_date = %s
            WHERE id = %s
        """, (status, datetime.now() if status == "completed" else None, file_id))
        
        connection.commit()
        cursor.close()
        connection.close()
        
    except Exception as e:
        logger.error(f"Error updating file processing status: {e}")

def store_qa_interaction(session_id: str, question: str, answer: str, 
                        sources: List[Dict], confidence: float, model_used: str) -> int:
    """Store Q&A interaction with processing time"""
    try:
        connection = get_db_connection()
        if not connection:
            return 0
        
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO qa_interactions 
            (session_id, question, answer, source_chunks, confidence_score, model_used, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            session_id, 
            question, 
            answer, 
            json.dumps(sources), 
            confidence, 
            model_used,
            datetime.now()
        ))
        
        interaction_id = cursor.lastrowid
        connection.commit()
        cursor.close()
        connection.close()
        
        return interaction_id
        
    except Exception as e:
        logger.error(f"Error storing Q&A interaction: {e}")
        return 0

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6000)

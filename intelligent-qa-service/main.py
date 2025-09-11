# intelligent-qa-service/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging
import mysql.connector
from mysql.connector import Error
import json
from typing import List, Optional, Dict, Any
import numpy as np
from datetime import datetime
import asyncio
import httpx
from services.rag_engine import RAGEngine
from services.vector_store import VectorStore
from services.document_chunker import DocumentChunker
from services.citation_tracker import CitationTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Intelligent Q&A Service", version="1.0.0")

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
rag_engine = RAGEngine()
vector_store = VectorStore()
document_chunker = DocumentChunker()
citation_tracker = CitationTracker()

# Request/Response models
class QARequest(BaseModel):
    question: str
    session_id: str
    max_sources: Optional[int] = 5
    confidence_threshold: Optional[float] = 0.7

class QAResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    related_questions: List[str]
    processing_time: float

class ProcessDocumentRequest(BaseModel):
    session_id: str
    file_id: int
    force_reprocess: Optional[bool] = False

class FeedbackRequest(BaseModel):
    interaction_id: int
    rating: int  # 1-5 stars
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
        "version": "1.0.0",
        "features": ["document-rag", "cross-document-search", "citation-tracking", "smart-qa"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "intelligent-qa-service"}

@app.post("/qa/ask", response_model=QAResponse)
async def ask_question(request: QARequest):
    """
    Ask a question about uploaded documents using RAG
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing question for session {request.session_id}: {request.question[:50]}...")
        
        # Get relevant document chunks using vector search
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
                processing_time=0.0
            )
        
        # Generate answer using RAG
        answer_result = await rag_engine.generate_answer(
            question=request.question,
            context_chunks=relevant_chunks
        )
        
        # Track citations
        citations = citation_tracker.generate_citations(relevant_chunks, answer_result["answer"])
        
        # Generate related questions
        related_questions = await rag_engine.generate_related_questions(
            question=request.question,
            context_chunks=relevant_chunks[:3]  # Use top 3 chunks
        )
        
        # Store interaction in database
        interaction_id = store_qa_interaction(
            session_id=request.session_id,
            question=request.question,
            answer=answer_result["answer"],
            sources=citations,
            confidence=answer_result["confidence"]
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Generated answer with {len(citations)} sources, confidence: {answer_result['confidence']:.2f}")
        
        return QAResponse(
            answer=answer_result["answer"],
            sources=citations,
            confidence=answer_result["confidence"],
            related_questions=related_questions,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")

@app.post("/documents/process")
async def process_document(request: ProcessDocumentRequest, background_tasks: BackgroundTasks):
    """
    Process a document for RAG (chunking and embedding)
    """
    try:
        # Check if document is already processed
        if not request.force_reprocess:
            existing_chunks = get_document_chunks(request.file_id)
            if existing_chunks:
                return {
                    "message": "Document already processed",
                    "chunks_count": len(existing_chunks),
                    "file_id": request.file_id
                }
        
        # Get file content from file service
        file_content = await get_file_content(request.file_id)
        if not file_content:
            raise HTTPException(status_code=404, detail="File not found or has no content")
        
        # Process document in background
        background_tasks.add_task(
            process_document_background,
            request.session_id,
            request.file_id,
            file_content
        )
        
        return {
            "message": "Document processing started",
            "file_id": request.file_id,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error initiating document processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{session_id}/status")
async def get_processing_status(session_id: str):
    """
    Get document processing status for a session
    """
    try:
        connection = get_db_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor()
        
        # Get processed documents count
        cursor.execute("""
            SELECT COUNT(DISTINCT file_id) as processed_count
            FROM document_chunks 
            WHERE session_id = %s
        """, (session_id,))
        processed_count = cursor.fetchone()[0]
        
        # Get total documents count
        cursor.execute("""
            SELECT COUNT(*) as total_count
            FROM uploaded_files 
            WHERE session_id = %s
        """, (session_id,))
        total_count = cursor.fetchone()[0]
        
        cursor.close()
        connection.close()
        
        return {
            "session_id": session_id,
            "processed_documents": processed_count,
            "total_documents": total_count,
            "processing_complete": processed_count == total_count,
            "readiness_percentage": (processed_count / total_count * 100) if total_count > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting processing status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/qa/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback for a Q&A interaction
    """
    try:
        connection = get_db_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor()
        
        # Store feedback
        cursor.execute("""
            INSERT INTO qa_feedback (interaction_id, rating, feedback_text, created_at)
            VALUES (%s, %s, %s, %s)
        """, (request.interaction_id, request.rating, request.feedback_text, datetime.now()))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info(f"Feedback received for interaction {request.interaction_id}: {request.rating} stars")
        
        return {"message": "Feedback received successfully"}
        
    except Exception as e:
        logger.error(f"Error storing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/qa/popular-questions/{session_id}")
async def get_popular_questions(session_id: str, limit: int = 10):
    """
    Get popular questions for a session
    """
    try:
        connection = get_db_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor()
        
        cursor.execute("""
            SELECT question, COUNT(*) as frequency, AVG(confidence_score) as avg_confidence
            FROM qa_interactions 
            WHERE session_id = %s 
            GROUP BY question 
            ORDER BY frequency DESC, avg_confidence DESC 
            LIMIT %s
        """, (session_id, limit))
        
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        
        popular_questions = [
            {
                "question": row[0],
                "frequency": row[1],
                "avg_confidence": float(row[2]) if row[2] else 0.0
            }
            for row in results
        ]
        
        return {"popular_questions": popular_questions}
        
    except Exception as e:
        logger.error(f"Error getting popular questions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/{session_id}")
async def get_session_analytics(session_id: str):
    """
    Get analytics for a session
    """
    try:
        connection = get_db_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor()
        
        # Get Q&A stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_questions,
                AVG(confidence_score) as avg_confidence,
                MAX(created_at) as last_question
            FROM qa_interactions 
            WHERE session_id = %s
        """, (session_id,))
        qa_stats = cursor.fetchone()
        
        # Get document stats
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT file_id) as documents_processed,
                COUNT(*) as total_chunks,
                AVG(CHAR_LENGTH(chunk_text)) as avg_chunk_size
            FROM document_chunks 
            WHERE session_id = %s
        """, (session_id,))
        doc_stats = cursor.fetchone()
        
        cursor.close()
        connection.close()
        
        return {
            "session_id": session_id,
            "qa_analytics": {
                "total_questions": qa_stats[0] or 0,
                "avg_confidence": float(qa_stats[1]) if qa_stats[1] else 0.0,
                "last_question": qa_stats[2].isoformat() if qa_stats[2] else None
            },
            "document_analytics": {
                "documents_processed": doc_stats[0] or 0,
                "total_chunks": doc_stats[1] or 0,
                "avg_chunk_size": float(doc_stats[2]) if doc_stats[2] else 0.0
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
async def get_file_content(file_id: int) -> Optional[Dict]:
    """Get file content from file service"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{FILE_SERVICE_URL}/file/content/{file_id}")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        logger.error(f"Error getting file content: {e}")
    return None

def get_document_chunks(file_id: int) -> List[Dict]:
    """Check if document chunks already exist"""
    try:
        connection = get_db_connection()
        if not connection:
            return []
        
        cursor = connection.cursor()
        cursor.execute("SELECT id FROM document_chunks WHERE file_id = %s LIMIT 1", (file_id,))
        result = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return result
    except Exception as e:
        logger.error(f"Error checking document chunks: {e}")
        return []

async def process_document_background(session_id: str, file_id: int, file_content: Dict):
    """Background task to process document"""
    try:
        logger.info(f"Starting background processing for file {file_id}")
        
        # Chunk the document
        chunks = document_chunker.chunk_document(
            text=file_content["content"],
            filename=file_content["filename"]
        )
        
        # Generate embeddings and store chunks
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = await vector_store.generate_embedding(chunk["text"])
            
            # Store chunk in database
            store_document_chunk(
                session_id=session_id,
                file_id=file_id,
                chunk_index=i,
                chunk_text=chunk["text"],
                embedding=embedding,
                metadata=chunk["metadata"]
            )
        
        logger.info(f"Processed {len(chunks)} chunks for file {file_id}")
        
    except Exception as e:
        logger.error(f"Error in background processing: {e}")

def store_document_chunk(session_id: str, file_id: int, chunk_index: int, 
                        chunk_text: str, embedding: List[float], metadata: Dict):
    """Store document chunk in database"""
    try:
        connection = get_db_connection()
        if not connection:
            return
        
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO document_chunks 
            (session_id, file_id, chunk_index, chunk_text, chunk_embedding, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (session_id, file_id, chunk_index, chunk_text, json.dumps(embedding), json.dumps(metadata)))
        
        connection.commit()
        cursor.close()
        connection.close()
        
    except Exception as e:
        logger.error(f"Error storing document chunk: {e}")

def store_qa_interaction(session_id: str, question: str, answer: str, 
                        sources: List[Dict], confidence: float) -> int:
    """Store Q&A interaction in database"""
    try:
        connection = get_db_connection()
        if not connection:
            return 0
        
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO qa_interactions 
            (session_id, question, answer, source_chunks, confidence_score)
            VALUES (%s, %s, %s, %s, %s)
        """, (session_id, question, answer, json.dumps(sources), confidence))
        
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

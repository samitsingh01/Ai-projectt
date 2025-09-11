# intelligent-qa-service/main.py - Fixed version
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
import re
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# Global TF-IDF vectorizer for text similarity
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))

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
        "version": "1.0.0",
        "features": ["document-rag", "cross-document-search", "citation-tracking", "smart-qa"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "intelligent-qa-service"}

@app.post("/qa/ask", response_model=QAResponse)
async def ask_question(request: QARequest):
    """Ask a question about uploaded documents using RAG"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing question for session {request.session_id}: {request.question[:50]}...")
        
        # Get relevant document chunks
        relevant_chunks = await search_similar_chunks(
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
        answer_result = await generate_answer_with_rag(
            question=request.question,
            context_chunks=relevant_chunks
        )
        
        # Generate citations
        citations = generate_citations(relevant_chunks, answer_result["answer"])
        
        # Generate related questions
        related_questions = await generate_related_questions(
            question=request.question,
            context_chunks=relevant_chunks[:3]
        )
        
        # Store interaction
        store_qa_interaction(
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
    """Process a document for RAG"""
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
        
        # Get file content
        file_content = await get_file_content(request.file_id)
        if not file_content:
            raise HTTPException(status_code=404, detail="File not found or has no content")
        
        # Process in background
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
    """Get document processing status"""
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
        result = cursor.fetchone()
        processed_count = result[0] if result else 0
        
        # Get total documents count
        cursor.execute("""
            SELECT COUNT(*) as total_count
            FROM uploaded_files 
            WHERE session_id = %s
        """, (session_id,))
        result = cursor.fetchone()
        total_count = result[0] if result else 0
        
        cursor.close()
        connection.close()
        
        return {
            "session_id": session_id,
            "processed_documents": processed_count,
            "total_documents": total_count,
            "processing_complete": processed_count >= total_count and total_count > 0,
            "readiness_percentage": (processed_count / total_count * 100) if total_count > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting processing status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/{session_id}")
async def get_session_analytics(session_id: str):
    """Get analytics for a session"""
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
        qa_result = cursor.fetchone()
        
        # Get document stats
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT file_id) as documents_processed,
                COUNT(*) as total_chunks,
                AVG(CHAR_LENGTH(chunk_text)) as avg_chunk_size
            FROM document_chunks 
            WHERE session_id = %s
        """, (session_id,))
        doc_result = cursor.fetchone()
        
        cursor.close()
        connection.close()
        
        return {
            "session_id": session_id,
            "qa_analytics": {
                "total_questions": qa_result[0] if qa_result else 0,
                "avg_confidence": float(qa_result[1]) if qa_result and qa_result[1] else 0.0,
                "last_question": qa_result[2].isoformat() if qa_result and qa_result[2] else None
            },
            "document_analytics": {
                "documents_processed": doc_result[0] if doc_result else 0,
                "total_chunks": doc_result[1] if doc_result else 0,
                "avg_chunk_size": float(doc_result[2]) if doc_result and doc_result[2] else 0.0
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
async def search_similar_chunks(session_id: str, query: str, top_k: int = 5, confidence_threshold: float = 0.3) -> List[Dict]:
    """Search for similar document chunks"""
    try:
        # Get all chunks for the session
        connection = get_db_connection()
        if not connection:
            return []
        
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT 
                dc.id,
                dc.file_id,
                dc.chunk_text,
                dc.metadata,
                uf.original_name as filename
            FROM document_chunks dc
            LEFT JOIN uploaded_files uf ON dc.file_id = uf.id
            WHERE dc.session_id = %s
            ORDER BY dc.file_id, dc.chunk_index
        """, (session_id,))
        
        chunks = cursor.fetchall()
        cursor.close()
        connection.close()
        
        if not chunks:
            return []
        
        # Calculate text similarities using TF-IDF
        chunk_texts = [chunk['chunk_text'] for chunk in chunks]
        try:
            # Fit TF-IDF on chunk texts and query
            all_texts = chunk_texts + [query]
            tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
            
            # Calculate similarities
            query_vector = tfidf_matrix[-1]
            chunk_vectors = tfidf_matrix[:-1]
            similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
            
            # Create results with similarities
            results = []
            for i, chunk in enumerate(chunks):
                similarity = float(similarities[i])
                if similarity >= confidence_threshold:
                    chunk_data = {
                        'id': chunk['id'],
                        'file_id': chunk['file_id'],
                        'text': chunk['chunk_text'],
                        'similarity': similarity,
                        'filename': chunk['filename'] or 'Unknown',
                        'page_number': 1,
                        'metadata': json.loads(chunk['metadata']) if chunk['metadata'] else {}
                    }
                    results.append(chunk_data)
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
            
        except Exception as tfidf_error:
            logger.error(f"TF-IDF error: {tfidf_error}")
            # Fallback: simple keyword matching
            results = []
            query_words = set(query.lower().split())
            
            for chunk in chunks:
                chunk_words = set(chunk['chunk_text'].lower().split())
                overlap = len(query_words & chunk_words)
                similarity = overlap / max(len(query_words), 1)
                
                if similarity >= confidence_threshold:
                    chunk_data = {
                        'id': chunk['id'],
                        'file_id': chunk['file_id'],
                        'text': chunk['chunk_text'],
                        'similarity': similarity,
                        'filename': chunk['filename'] or 'Unknown',
                        'page_number': 1,
                        'metadata': json.loads(chunk['metadata']) if chunk['metadata'] else {}
                    }
                    results.append(chunk_data)
            
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
            
    except Exception as e:
        logger.error(f"Error searching similar chunks: {e}")
        return []

async def generate_answer_with_rag(question: str, context_chunks: List[Dict]) -> Dict[str, Any]:
    """Generate answer using RAG"""
    try:
        # Build context
        context = build_context(context_chunks)
        
        # Create prompt
        prompt = f"""You are an intelligent document analysis assistant. Answer questions based ONLY on the provided context from uploaded documents.

INSTRUCTIONS:
1. Answer based solely on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Include specific references to documents when possible
4. Be precise and factual
5. Keep your answer concise but complete

CONTEXT FROM UPLOADED DOCUMENTS:
{context}

QUESTION: {question}

ANSWER (be specific and include document references):"""

        # Call Bedrock service
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{BEDROCK_SERVICE_URL}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": 1500,
                    "temperature": 0.3
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Calculate confidence
                confidence = calculate_confidence(question, result["response"], context_chunks)
                
                return {
                    "answer": result["response"],
                    "confidence": confidence
                }
            else:
                logger.error(f"Bedrock service error: {response.status_code}")
                return {
                    "answer": "I'm sorry, I'm having trouble accessing the AI service to generate an answer. Please try again later.",
                    "confidence": 0.0
                }
                
    except Exception as e:
        logger.error(f"Error generating RAG answer: {e}")
        return {
            "answer": f"I encountered an error while generating an answer: {str(e)}. Please try again.",
            "confidence": 0.0
        }

def build_context(chunks: List[Dict]) -> str:
    """Build context string from chunks"""
    context_parts = []
    max_context_length = 6000
    current_length = 0
    
    for chunk in chunks:
        chunk_text = f"""Document: {chunk['filename']}
Content: {chunk['text']}

---

"""
        if current_length + len(chunk_text) > max_context_length:
            break
        
        context_parts.append(chunk_text)
        current_length += len(chunk_text)
    
    return "".join(context_parts)

def generate_citations(chunks: List[Dict], answer: str) -> List[Dict[str, Any]]:
    """Generate citations from chunks"""
    citations = []
    
    for chunk in chunks:
        # Extract snippet
        snippet = extract_snippet(chunk['text'], answer)
        
        citation = {
            "source_id": chunk['id'],
            "document": chunk['filename'],
            "page": chunk.get('page_number', 1),
            "snippet": snippet,
            "relevance_score": chunk['similarity'],
            "similarity": chunk['similarity']
        }
        citations.append(citation)
    
    return citations

def extract_snippet(text: str, answer: str, max_length: int = 150) -> str:
    """Extract relevant snippet from text"""
    sentences = re.split(r'[.!?]+', text)
    if not sentences:
        return text[:max_length] + "..."
    
    # Find sentence with most overlap with answer
    answer_words = set(answer.lower().split())
    best_sentence = ""
    best_overlap = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:
            continue
        
        sentence_words = set(sentence.lower().split())
        overlap = len(answer_words & sentence_words)
        
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sentence
    
    if len(best_sentence) > max_length:
        return best_sentence[:max_length] + "..."
    
    return best_sentence or text[:max_length] + "..."

def calculate_confidence(question: str, answer: str, context_chunks: List[Dict]) -> float:
    """Calculate confidence score"""
    try:
        # Simple confidence calculation
        answer_length_score = min(len(answer) / 500, 1.0) * 0.2
        source_score = min(len(context_chunks) / 3, 1.0) * 0.4
        
        # Keyword overlap
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        overlap_score = len(question_words & answer_words) / max(len(question_words), 1) * 0.4
        
        confidence = answer_length_score + source_score + overlap_score
        return max(0.1, min(0.95, confidence))
        
    except Exception as e:
        logger.error(f"Error calculating confidence: {e}")
        return 0.5

async def generate_related_questions(question: str, context_chunks: List[Dict]) -> List[str]:
    """Generate related questions"""
    try:
        if not context_chunks:
            return []
        
        context = build_context(context_chunks[:2])
        
        prompt = f"""Based on the following context and original question, suggest 3 related questions that would be helpful.

Context:
{context[:1000]}

Original Question: {question}

Generate 3 related questions (one per line):"""

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{BEDROCK_SERVICE_URL}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": 200,
                    "temperature": 0.7
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                questions = parse_questions(result["response"])
                return questions[:3]
                
    except Exception as e:
        logger.error(f"Error generating related questions: {e}")
    
    return []

def parse_questions(text: str) -> List[str]:
    """Parse questions from text"""
    lines = text.strip().split('\n')
    questions = []
    
    for line in lines:
        line = line.strip()
        line = re.sub(r'^[\d\.\-\*\s]+', '', line)
        
        if line and len(line) > 10 and '?' in line:
            questions.append(line)
    
    return questions

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
    """Check if document chunks exist"""
    try:
        connection = get_db_connection()
        if not connection:
            return []
        
        cursor = connection.cursor()
        cursor.execute("SELECT id FROM document_chunks WHERE file_id = %s", (file_id,))
        results = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return results
    except Exception as e:
        logger.error(f"Error checking document chunks: {e}")
        return []

async def process_document_background(session_id: str, file_id: int, file_content: Dict):
    """Background task to process document"""
    try:
        logger.info(f"Processing file {file_id} for session {session_id}")
        
        text = file_content.get("content", "")
        filename = file_content.get("filename", "Unknown")
        
        # Simple chunking
        chunks = chunk_text(text, filename)
        
        # Store chunks
        for i, chunk in enumerate(chunks):
            store_document_chunk(
                session_id=session_id,
                file_id=file_id,
                chunk_index=i,
                chunk_text=chunk["text"],
                metadata=chunk["metadata"]
            )
        
        logger.info(f"Processed {len(chunks)} chunks for file {file_id}")
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")

def chunk_text(text: str, filename: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Simple text chunking"""
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        
        # Find sentence boundary if possible
        if end < len(text):
            last_period = chunk_text.rfind('.')
            if last_period > chunk_size * 0.5:
                chunk_text = chunk_text[:last_period + 1]
                end = start + last_period + 1
        
        if len(chunk_text.strip()) > 50:  # Only store substantial chunks
            chunks.append({
                "text": chunk_text.strip(),
                "metadata": {
                    "filename": filename,
                    "chunk_index": chunk_index,
                    "start_pos": start,
                    "end_pos": end
                }
            })
            chunk_index += 1
        
        start = max(start + chunk_size - overlap, end)
    
    return chunks

def store_document_chunk(session_id: str, file_id: int, chunk_index: int, chunk_text: str, metadata: Dict):
    """Store document chunk"""
    try:
        connection = get_db_connection()
        if not connection:
            return
        
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO document_chunks 
            (session_id, file_id, chunk_index, chunk_text, metadata, word_count, char_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            chunk_text = VALUES(chunk_text),
            metadata = VALUES(metadata),
            word_count = VALUES(word_count),
            char_count = VALUES(char_count)
        """, (
            session_id, 
            file_id, 
            chunk_index, 
            chunk_text, 
            json.dumps(metadata),
            len(chunk_text.split()),
            len(chunk_text)
        ))
        
        connection.commit()
        cursor.close()
        connection.close()
        
    except Exception as e:
        logger.error(f"Error storing document chunk: {e}")

def store_qa_interaction(session_id: str, question: str, answer: str, sources: List[Dict], confidence: float) -> int:
    """Store Q&A interaction"""
    try:
        connection = get_db_connection()
        if not connection:
            return 0
        
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO qa_interactions 
            (session_id, question, answer, source_chunks, confidence_score, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (session_id, question, answer, json.dumps(sources), confidence, datetime.now()))
        
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

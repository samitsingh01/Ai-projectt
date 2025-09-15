# intelligent-qa-service/main.py - FIXED for t2.large instance
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
import time

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

# Simple text processing utilities
def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract important keywords from text"""
    import re
    # Remove common stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'will', 'would', 'could', 'this', 'that', 'these', 'those'}
    
    # Extract words, filter out stop words, and get unique words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    keywords = [word for word in words if word not in stop_words]
    
    # Count frequency and return most common
    from collections import Counter
    word_counts = Counter(keywords)
    return [word for word, count in word_counts.most_common(max_keywords)]

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity based on word overlap"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union) if union else 0.0

# Request/Response models
class QARequest(BaseModel):
    question: str
    session_id: str
    max_sources: Optional[int] = 5
    confidence_threshold: Optional[float] = 0.2

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
            database=url_parts[1],
            connect_timeout=10,
            autocommit=True
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
            "keyword-search", 
            "simple-qa",
            "text-chunking"
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
        
        # Test Bedrock service with timeout
        bedrock_status = "unknown"
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{BEDROCK_SERVICE_URL}/health")
                bedrock_status = "healthy" if response.status_code == 200 else "unhealthy"
        except:
            bedrock_status = "unreachable"
        
        return {
            "status": "healthy",
            "service": "intelligent-qa-service",
            "version": "2.0.0",
            "dependencies": {
                "database": db_status,
                "bedrock_service": bedrock_status
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"status": "degraded", "error": str(e)}

@app.post("/qa/ask", response_model=QAResponse)
async def ask_question(request: QARequest):
    """Ask a question about uploaded documents using simple RAG"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing question for session {request.session_id}: {request.question[:50]}...")
        
        # Get relevant document chunks using simple keyword matching
        relevant_chunks = get_relevant_chunks(
            session_id=request.session_id,
            query=request.question,
            max_chunks=request.max_sources,
            confidence_threshold=request.confidence_threshold
        )
        
        if not relevant_chunks:
            return QAResponse(
                answer="I couldn't find relevant information in your uploaded documents to answer this question. Please make sure you have uploaded documents that contain information related to your query.",
                sources=[],
                confidence=0.0,
                related_questions=[],
                processing_time=time.time() - start_time
            )
        
        # Generate answer using available chunks
        answer_result = await generate_answer_from_chunks(
            question=request.question,
            chunks=relevant_chunks
        )
        
        # Generate citations
        citations = generate_simple_citations(relevant_chunks, answer_result["answer"])
        
        # Generate related questions
        related_questions = generate_related_questions_simple(
            question=request.question,
            chunks=relevant_chunks[:2]
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
        
        processing_time = time.time() - start_time
        
        logger.info(f"Generated answer with {len(citations)} sources, confidence: {answer_result['confidence']:.2f}")
        
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
        processing_time = time.time() - start_time
        return QAResponse(
            answer=f"I'm sorry, I encountered an error while processing your question. Please try again.",
            sources=[],
            confidence=0.0,
            related_questions=[],
            processing_time=processing_time
        )

@app.post("/documents/process")
async def process_document(request: ProcessDocumentRequest, background_tasks: BackgroundTasks):
    """Process a document for Q&A with simple chunking"""
    try:
        logger.info(f"Processing document {request.file_id} for session {request.session_id}")
        
        # Check if document is already processed
        if not request.force_reprocess:
            existing_chunks = get_document_chunks_count(request.file_id)
            if existing_chunks > 0:
                return {
                    "message": "Document already processed",
                    "chunks_count": existing_chunks,
                    "file_id": request.file_id,
                    "status": "already_processed"
                }
        
        # Get file content from file service
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
        
        cursor = connection.cursor(dictionary=True)
        
        # Get processed documents count
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT dc.file_id) as processed_count,
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
            "total_chunks": processed_result['total_chunks'] if processed_result else 0
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
        
        cursor = connection.cursor(dictionary=True)
        
        # Get Q&A interaction stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_questions,
                AVG(confidence_score) as avg_confidence
            FROM qa_interactions 
            WHERE session_id = %s
        """, (session_id,))
        qa_result = cursor.fetchone()
        
        # Get document processing stats
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT file_id) as documents_processed,
                COUNT(*) as total_chunks
            FROM document_chunks 
            WHERE session_id = %s
        """, (session_id,))
        doc_result = cursor.fetchone()
        
        cursor.close()
        connection.close()
        
        return {
            "session_id": session_id,
            "qa_analytics": {
                "total_questions": qa_result['total_questions'] if qa_result else 0,
                "avg_confidence": float(qa_result['avg_confidence']) if qa_result and qa_result['avg_confidence'] else 0.0
            },
            "document_analytics": {
                "documents_processed": doc_result['documents_processed'] if doc_result else 0,
                "total_chunks": doc_result['total_chunks'] if doc_result else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
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

def get_document_chunks_count(file_id: int) -> int:
    """Get count of document chunks for a file"""
    try:
        connection = get_db_connection()
        if not connection:
            return 0
        
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM document_chunks WHERE file_id = %s", (file_id,))
        result = cursor.fetchone()
        
        cursor.close()
        connection.close()
        
        return result[0] if result else 0
    except Exception as e:
        logger.error(f"Error checking document chunks for file {file_id}: {e}")
        return 0

def get_relevant_chunks(session_id: str, query: str, max_chunks: int = 5, confidence_threshold: float = 0.2) -> List[Dict]:
    """Get relevant chunks using simple keyword matching"""
    try:
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
        
        all_chunks = cursor.fetchall()
        cursor.close()
        connection.close()
        
        if not all_chunks:
            logger.warning(f"No chunks found for session {session_id}")
            return []
        
        # Calculate similarity scores
        query_keywords = extract_keywords(query)
        scored_chunks = []
        
        for chunk in all_chunks:
            chunk_text = chunk['chunk_text']
            
            # Simple keyword matching
            similarity = calculate_text_similarity(query, chunk_text)
            
            # Keyword boost
            chunk_keywords = extract_keywords(chunk_text)
            keyword_overlap = len(set(query_keywords) & set(chunk_keywords))
            keyword_boost = keyword_overlap / max(len(query_keywords), 1) * 0.3
            
            final_score = similarity + keyword_boost
            
            if final_score >= confidence_threshold:
                chunk_data = {
                    'id': chunk['id'],
                    'file_id': chunk['file_id'],
                    'text': chunk_text,
                    'similarity': final_score,
                    'metadata': json.loads(chunk['metadata']) if chunk['metadata'] else {},
                    'filename': chunk.get('filename', 'Unknown'),
                    'page_number': 1
                }
                scored_chunks.append(chunk_data)
        
        # Sort by similarity and return top chunks
        scored_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        
        # If no chunks meet threshold, return top 2 chunks with minimum similarity
        if not scored_chunks and all_chunks:
            for chunk in all_chunks[:2]:
                chunk_data = {
                    'id': chunk['id'],
                    'file_id': chunk['file_id'],
                    'text': chunk['chunk_text'],
                    'similarity': 0.3,
                    'metadata': json.loads(chunk['metadata']) if chunk['metadata'] else {},
                    'filename': chunk.get('filename', 'Unknown'),
                    'page_number': 1
                }
                scored_chunks.append(chunk_data)
        
        logger.info(f"Found {len(scored_chunks)} relevant chunks for query")
        return scored_chunks[:max_chunks]
        
    except Exception as e:
        logger.error(f"Error getting relevant chunks: {e}")
        return []

async def generate_answer_from_chunks(question: str, chunks: List[Dict]) -> Dict[str, Any]:
    """Generate answer from chunks using Bedrock or fallback"""
    try:
        # Build context
        context_parts = []
        for i, chunk in enumerate(chunks[:3]):  # Limit to 3 chunks
            filename = chunk.get('filename', 'Unknown')
            text = chunk['text'][:500]  # Limit chunk size
            context_parts.append(f"[Document: {filename}]\n{text}\n")
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following document excerpts, answer the user's question. Be specific and cite the documents when possible.

Document excerpts:
{context}

Question: {question}

Answer:"""
        
        # Try Bedrock service
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{BEDROCK_SERVICE_URL}/generate",
                    json={
                        "prompt": prompt,
                        "max_tokens": 500,
                        "temperature": 0.3
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    confidence = calculate_answer_confidence(question, result.get("response", ""), chunks)
                    
                    return {
                        "answer": result.get("response", ""),
                        "confidence": confidence,
                        "model_used": result.get("model_used", "bedrock-service")
                    }
        except Exception as e:
            logger.warning(f"Bedrock service unavailable: {e}")
        
        # Fallback to extractive answer
        return generate_extractive_answer(question, chunks)
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return {
            "answer": "I encountered an error generating an answer. Please try again.",
            "confidence": 0.0,
            "model_used": "error-fallback"
        }

def generate_extractive_answer(question: str, chunks: List[Dict]) -> Dict[str, Any]:
    """Generate answer by extracting relevant sentences"""
    try:
        question_keywords = extract_keywords(question)
        best_sentences = []
        
        for chunk in chunks[:2]:  # Use top 2 chunks
            text = chunk['text']
            sentences = text.split('.')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # Minimum sentence length
                    sentence_keywords = extract_keywords(sentence)
                    overlap = len(set(question_keywords) & set(sentence_keywords))
                    
                    if overlap > 0:
                        score = overlap / max(len(question_keywords), 1)
                        best_sentences.append((sentence, score, chunk.get('filename', 'Unknown')))
        
        if best_sentences:
            # Sort by score and take top sentences
            best_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Combine top sentences
            answer_parts = []
            used_files = set()
            
            for sentence, score, filename in best_sentences[:3]:
                if filename not in used_files:
                    answer_parts.append(f"According to {filename}: {sentence}.")
                    used_files.add(filename)
                else:
                    answer_parts.append(sentence + ".")
            
            answer = " ".join(answer_parts)
            confidence = min(0.8, best_sentences[0][1] + 0.2)
            
            return {
                "answer": answer,
                "confidence": confidence,
                "model_used": "extractive-fallback"
            }
        else:
            # Return first chunk snippet as last resort
            first_chunk = chunks[0] if chunks else None
            if first_chunk:
                snippet = first_chunk['text'][:200] + "..."
                filename = first_chunk.get('filename', 'Unknown')
                answer = f"Based on {filename}: {snippet}"
                
                return {
                    "answer": answer,
                    "confidence": 0.4,
                    "model_used": "snippet-fallback"
                }
            else:
                return {
                    "answer": "I couldn't find a specific answer in the documents.",
                    "confidence": 0.0,
                    "model_used": "no-content-fallback"
                }
                
    except Exception as e:
        logger.error(f"Error in extractive answer generation: {e}")
        return {
            "answer": "I encountered an error while analyzing the documents.",
            "confidence": 0.0,
            "model_used": "error-fallback"
        }

def calculate_answer_confidence(question: str, answer: str, chunks: List[Dict]) -> float:
    """Calculate confidence score for answer"""
    try:
        # Factor 1: Answer length and completeness
        length_score = min(len(answer) / 200, 1.0) * 0.3
        
        # Factor 2: Question-answer similarity
        similarity_score = calculate_text_similarity(question, answer) * 0.4
        
        # Factor 3: Source quality
        source_score = min(len(chunks) / 3, 1.0) * 0.3
        
        confidence = length_score + similarity_score + source_score
        return max(0.1, min(0.9, confidence))
        
    except Exception as e:
        logger.error(f"Error calculating confidence: {e}")
        return 0.5

def generate_simple_citations(chunks: List[Dict], answer: str) -> List[Dict[str, Any]]:
    """Generate simple citations"""
    try:
        citations = []
        
        for chunk in chunks[:3]:  # Limit to 3 citations
            text = chunk['text']
            filename = chunk.get('filename', 'Unknown')
            
            # Extract relevant snippet
            snippet = text[:150]
            if len(text) > 150:
                snippet += "..."
            
            # Calculate relevance
            relevance = calculate_text_similarity(answer, text)
            
            citation = {
                "source_id": chunk['id'],
                "document": filename,
                "page": chunk.get('page_number', 1),
                "snippet": snippet,
                "relevance_score": relevance,
                "similarity": chunk.get('similarity', 0.0)
            }
            
            citations.append(citation)
        
        return citations
        
    except Exception as e:
        logger.error(f"Error generating citations: {e}")
        return []

def generate_related_questions_simple(question: str, chunks: List[Dict]) -> List[str]:
    """Generate related questions using simple templates"""
    try:
        if not chunks:
            return []
        
        # Extract key terms from chunks
        all_text = " ".join(chunk['text'] for chunk in chunks)
        keywords = extract_keywords(all_text, max_keywords=5)
        
        # Question templates
        templates = [
            "What is {}?",
            "How does {} work?",
            "Why is {} important?",
            "Where is {} used?",
            "When was {} developed?"
        ]
        
        related_questions = []
        for i, keyword in enumerate(keywords[:3]):
            if keyword.lower() not in question.lower():
                template = templates[i % len(templates)]
                related_questions.append(template.format(keyword))
        
        return related_questions
        
    except Exception as e:
        logger.error(f"Error generating related questions: {e}")
        return []

async def process_document_background(session_id: str, file_id: int, file_content: Dict):
    """Background task to process document with simple chunking"""
    try:
        logger.info(f"Processing file {file_id} for session {session_id}")
        
        text = file_content.get("content", "")
        filename = file_content.get("filename", "Unknown")
        
        if not text or len(text.strip()) < 50:
            logger.warning(f"File {file_id} has insufficient text content")
            return
        
        # Simple chunking by paragraph or size
        chunks = simple_chunk_text(text, filename)
        
        if not chunks:
            logger.warning(f"No chunks created for file {file_id}")
            return
        
        # Store chunks
        for i, chunk in enumerate(chunks):
            store_document_chunk(
                session_id=session_id,
                file_id=file_id,
                chunk_index=i,
                chunk_text=chunk,
                metadata={"chunk_index": i, "filename": filename}
            )
        
        logger.info(f"Successfully processed {len(chunks)} chunks for file {file_id}")
        
    except Exception as e:
        logger.error(f"Error processing document {file_id}: {e}")

def simple_chunk_text(text: str, filename: str, max_chunk_size: int = 800) -> List[str]:
    """Simple text chunking"""
    try:
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed max size, save current chunk
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                if len(current_chunk.strip()) > 50:  # Minimum chunk size
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add final chunk
        if len(current_chunk.strip()) > 50:
            chunks.append(current_chunk.strip())
        
        # If no paragraph-based chunks, split by sentences
        if not chunks:
            sentences = text.split('.')
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                    if len(current_chunk.strip()) > 50:
                        chunks.append(current_chunk.strip() + ".")
                    current_chunk = sentence
                else:
                    current_chunk += ". " + sentence if current_chunk else sentence
            
            if len(current_chunk.strip()) > 50:
                chunks.append(current_chunk.strip() + ".")
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        return [text[:max_chunk_size]] if text else []

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
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        logger.error(f"Error storing document chunk: {e}")

def store_qa_interaction(session_id: str, question: str, answer: str, sources: List[Dict], confidence: float, model_used: str) -> int:
    """Store Q&A interaction"""
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
        cursor.close()
        connection.close()
        
        return interaction_id
        
    except Exception as e:
        logger.error(f"Error storing Q&A interaction: {e}")
        return 0

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6000)

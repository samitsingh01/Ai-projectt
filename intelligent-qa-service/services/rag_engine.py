# intelligent-qa-service/services/rag_engine.py - Improved version
import httpx
import json
import logging
from typing import List, Dict, Any
import re
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, bedrock_url: str):
        self.bedrock_url = bedrock_url
        self.max_context_length = 8000
        self.timeout = 60.0
        
    async def generate_answer(self, question: str, context_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate an answer using RAG with document context"""
        try:
            if not context_chunks:
                return {
                    "answer": "I don't have any relevant document context to answer this question.",
                    "confidence": 0.0,
                    "model_used": "rag-engine"
                }
            
            # Build context from chunks
            context = self._build_context(context_chunks)
            
            # Create enhanced prompt
            prompt = self._create_rag_prompt(question, context, context_chunks)
            
            # Try to get answer from Bedrock
            bedrock_result = await self._call_bedrock_service(prompt)
            
            if bedrock_result:
                # Calculate confidence score
                confidence = self._calculate_confidence(
                    question=question,
                    answer=bedrock_result["answer"],
                    context_chunks=context_chunks
                )
                
                return {
                    "answer": bedrock_result["answer"],
                    "confidence": confidence,
                    "model_used": bedrock_result.get("model_used", "bedrock-service")
                }
            else:
                # Fallback to rule-based answer
                logger.warning("Bedrock service unavailable, using fallback answer generation")
                return await self._generate_fallback_answer(question, context_chunks)
                
        except Exception as e:
            logger.error(f"Error generating RAG answer: {e}")
            return {
                "answer": f"I encountered an error while generating an answer: {str(e)}. Please try again.",
                "confidence": 0.0,
                "model_used": "rag-engine-error"
            }
    
    async def generate_related_questions(self, question: str, context_chunks: List[Dict]) -> List[str]:
        """Generate related questions based on the context"""
        try:
            if not context_chunks:
                return []
            
            context = self._build_context(context_chunks[:2])  # Use top 2 chunks
            
            prompt = self._create_related_questions_prompt(question, context)
            
            # Try Bedrock first
            bedrock_result = await self._call_bedrock_service(prompt, max_tokens=300, temperature=0.7)
            
            if bedrock_result:
                questions = self._parse_related_questions(bedrock_result["answer"])
                return questions[:3]
            else:
                # Fallback to rule-based question generation
                return self._generate_fallback_questions(question, context_chunks)
                
        except Exception as e:
            logger.error(f"Error generating related questions: {e}")
            return []
    
    async def _call_bedrock_service(self, prompt: str, max_tokens: int = 1500, 
                                  temperature: float = 0.3) -> Dict[str, Any]:
        """Call Bedrock service with proper error handling"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.bedrock_url}/generate",
                    json={
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "model": "claude-3-5-sonnet"  # Prefer Claude for RAG
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "answer": result.get("response", ""),
                        "model_used": result.get("model_used", "bedrock-service")
                    }
                else:
                    logger.error(f"Bedrock service HTTP error: {response.status_code}")
                    return None
                    
        except httpx.TimeoutException:
            logger.error("Bedrock service timeout")
            return None
        except httpx.ConnectError:
            logger.error("Cannot connect to Bedrock service")
            return None
        except Exception as e:
            logger.error(f"Bedrock service error: {e}")
            return None
    
    async def _generate_fallback_answer(self, question: str, context_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate a fallback answer using rule-based approach"""
        try:
            # Simple extractive approach
            question_words = set(question.lower().split())
            
            # Find chunks with highest word overlap
            best_chunk = None
            best_overlap = 0
            
            for chunk in context_chunks:
                chunk_words = set(chunk['text'].lower().split())
                overlap = len(question_words & chunk_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_chunk = chunk
            
            if best_chunk and best_overlap > 0:
                # Extract relevant sentences
                sentences = re.split(r'[.!?]+', best_chunk['text'])
                relevant_sentences = []
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 20:  # Minimum sentence length
                        sentence_words = set(sentence.lower().split())
                        if len(question_words & sentence_words) > 0:
                            relevant_sentences.append(sentence)
                
                if relevant_sentences:
                    answer = '. '.join(relevant_sentences[:3]) + '.'
                    confidence = min(0.7, best_overlap / len(question_words))
                else:
                    answer = f"Based on the document '{best_chunk['filename']}', {best_chunk['text'][:200]}..."
                    confidence = 0.4
                    
                return {
                    "answer": answer,
                    "confidence": confidence,
                    "model_used": "fallback-extraction"
                }
            else:
                return {
                    "answer": "I found relevant documents but couldn't extract a specific answer to your question. Please try rephrasing your question.",
                    "confidence": 0.2,
                    "model_used": "fallback-extraction"
                }
                
        except Exception as e:
            logger.error(f"Error in fallback answer generation: {e}")
            return {
                "answer": "I encountered an error while processing the documents. Please try again.",
                "confidence": 0.0,
                "model_used": "fallback-error"
            }
    
    def _generate_fallback_questions(self, question: str, context_chunks: List[Dict]) -> List[str]:
        """Generate related questions using rule-based approach"""
        try:
            related_questions = []
            
            # Extract key topics from context
            all_text = " ".join(chunk['text'] for chunk in context_chunks[:3])
            words = re.findall(r'\b[A-Z][a-z]+\b', all_text)  # Proper nouns
            
            # Common question patterns
            patterns = [
                "What is {}?",
                "How does {} work?",
                "Why is {} important?",
                "When was {} developed?",
                "Where is {} used?"
            ]
            
            # Generate questions from key terms
            key_terms = list(set(words))[:10]  # Top 10 unique proper nouns
            
            for term in key_terms[:3]:  # Limit to 3 terms
                if len(term) > 3 and term.lower() not in question.lower():
                    pattern = patterns[len(related_questions) % len(patterns)]
                    related_questions.append(pattern.format(term))
            
            return related_questions[:3]
            
        except Exception as e:
            logger.error(f"Error generating fallback questions: {e}")
            return []
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string from document chunks with length control"""
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks):
            # Create chunk header
            chunk_header = f"[Document: {chunk.get('filename', 'Unknown')} | Page: {chunk.get('page_number', 'Unknown')}]"
            chunk_content = f"{chunk_header}\n{chunk['text']}\n\n"
            
            # Check if adding this chunk would exceed context limit
            if current_length + len(chunk_content) > self.max_context_length:
                # Try to fit a truncated version
                remaining_space = self.max_context_length - current_length - len(chunk_header) - 10
                if remaining_space > 100:  # Only if we have reasonable space
                    truncated_text = chunk['text'][:remaining_space] + "..."
                    chunk_content = f"{chunk_header}\n{truncated_text}\n\n"
                    context_parts.append(chunk_content)
                break
            
            context_parts.append(chunk_content)
            current_length += len(chunk_content)
        
        return "".join(context_parts)
    
    def _create_rag_prompt(self, question: str, context: str, chunks: List[Dict]) -> str:
        """Create an enhanced RAG prompt with better instructions"""
        chunk_count = len(chunks)
        doc_names = list(set(chunk.get('filename', 'Unknown') for chunk in chunks))
        
        return f"""You are an intelligent document analysis assistant. Answer the user's question based ONLY on the provided context from uploaded documents.

INSTRUCTIONS:
1. Answer based solely on the provided context - do not use external knowledge
2. If the context doesn't contain enough information, say so clearly
3. Include specific references to documents and page numbers when possible
4. Be precise, factual, and concise
5. If you're uncertain about any aspect, indicate your level of confidence
6. Use direct quotes from the documents when relevant (in quotation marks)

CONTEXT FROM UPLOADED DOCUMENTS ({chunk_count} sections from {len(doc_names)} document(s): {', '.join(doc_names)}):
{context}

USER QUESTION: {question}

Please provide a comprehensive answer based on the document context above:"""
    
    def _create_related_questions_prompt(self, question: str, context: str) -> str:
        """Create prompt for generating related questions"""
        return f"""Based on the following document context and the user's original question, suggest 3 related questions that would be helpful and relevant.

DOCUMENT CONTEXT:
{context[:1500]}

ORIGINAL QUESTION: {question}

Please generate exactly 3 related questions that:
1. Are directly related to the document content
2. Would provide additional useful information
3. Are different from the original question
4. Can potentially be answered using the document context

Format: Return only the questions, one per line, without numbering or bullet points."""
    
    def _calculate_confidence(self, question: str, answer: str, context_chunks: List[Dict]) -> float:
        """Calculate confidence score for the RAG answer"""
        try:
            confidence_factors = []
            
            # Factor 1: Answer length and completeness (20%)
            answer_length_score = min(len(answer) / 300, 1.0)  # Normalize to 300 chars
            confidence_factors.append(answer_length_score * 0.2)
            
            # Factor 2: Number and quality of source chunks (30%)
            source_score = min(len(context_chunks) / 3, 1.0)
            avg_similarity = sum(chunk.get('similarity', 0.5) for chunk in context_chunks) / len(context_chunks)
            source_quality_score = source_score * avg_similarity
            confidence_factors.append(source_quality_score * 0.3)
            
            # Factor 3: Question-answer semantic overlap (25%)
            question_words = set(re.findall(r'\b\w+\b', question.lower()))
            answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
            overlap_score = len(question_words & answer_words) / max(len(question_words), 1)
            confidence_factors.append(overlap_score * 0.25)
            
            # Factor 4: Presence of document references and quotes (15%)
            reference_indicators = ['document', 'page', 'according to', 'states that', '"']
            reference_count = sum(1 for indicator in reference_indicators if indicator in answer.lower())
            reference_score = min(reference_count / len(reference_indicators), 1.0)
            confidence_factors.append(reference_score * 0.15)
            
            # Factor 5: Answer specificity (10%)
            specific_terms = len(re.findall(r'\b[A-Z][a-z]+\b', answer))  # Proper nouns
            specificity_score = min(specific_terms / 5, 1.0)
            confidence_factors.append(specificity_score * 0.1)
            
            # Calculate final confidence
            final_confidence = sum(confidence_factors)
            
            # Apply bounds and add small random variation
            import random
            final_confidence += random.uniform(-0.03, 0.03)
            final_confidence = max(0.15, min(0.95, final_confidence))
            
            return round(final_confidence, 3)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _parse_related_questions(self, response: str) -> List[str]:
        """Parse related questions from AI response"""
        try:
            lines = response.strip().split('\n')
            questions = []
            
            for line in lines:
                line = line.strip()
                # Remove numbering, bullets, etc.
                line = re.sub(r'^[\d\.\-\*\s]+', '', line)
                
                # Basic question validation
                if line and len(line) > 10 and '?' in line and len(line) < 200:
                    questions.append(line)
            
            return questions
            
        except Exception as e:
            logger.error(f"Error parsing related questions: {e}")
            return []

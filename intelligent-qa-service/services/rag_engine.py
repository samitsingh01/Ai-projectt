# intelligent-qa-service/services/rag_engine.py
import httpx
import json
import logging
from typing import List, Dict, Any
import asyncio
import re

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, bedrock_url: str):
        self.bedrock_url = bedrock_url
        self.max_context_length = 8000  # Maximum context length for Claude
        
    async def generate_answer(self, question: str, context_chunks: List[Dict]) -> Dict[str, Any]:
        """
        Generate an answer using RAG with document context
        """
        try:
            # Build context from chunks
            context = self._build_context(context_chunks)
            
            # Create enhanced prompt
            prompt = self._create_rag_prompt(question, context)
            
            # Call Bedrock service
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.bedrock_url}/generate",
                    json={
                        "prompt": prompt,
                        "max_tokens": 1500,
                        "temperature": 0.3,  # Lower temperature for factual responses
                        "model": "claude-3-5-sonnet"  # Prefer Claude for RAG
                    }
                )
                
                if response.status_code != 200:
                    raise Exception(f"Bedrock service error: {response.status_code}")
                
                result = response.json()
                
                # Calculate confidence score
                confidence = self._calculate_confidence(
                    question=question,
                    answer=result["response"],
                    context_chunks=context_chunks
                )
                
                return {
                    "answer": result["response"],
                    "confidence": confidence,
                    "model_used": result.get("model_used", "unknown")
                }
                
        except Exception as e:
            logger.error(f"Error generating RAG answer: {e}")
            raise Exception(f"Failed to generate answer: {str(e)}")
    
    async def generate_related_questions(self, question: str, context_chunks: List[Dict]) -> List[str]:
        """
        Generate related questions based on the context
        """
        try:
            context = self._build_context(context_chunks[:2])  # Use top 2 chunks
            
            prompt = f"""Based on the following context and original question, suggest 3 related questions that the user might want to ask next.

Context:
{context}

Original Question: {question}

Generate 3 related questions that would be relevant and helpful. Return only the questions, one per line:"""

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.bedrock_url}/generate",
                    json={
                        "prompt": prompt,
                        "max_tokens": 300,
                        "temperature": 0.7
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    questions = self._parse_related_questions(result["response"])
                    return questions[:3]  # Return max 3 questions
                
        except Exception as e:
            logger.error(f"Error generating related questions: {e}")
        
        return []  # Return empty list on error
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """
        Build context string from document chunks
        """
        context_parts = []
        current_length = 0
        
        for chunk in chunks:
            chunk_text = f"""Document: {chunk['filename']}
Page: {chunk.get('page_number', 'Unknown')}
Content: {chunk['text']}

---

"""
            
            # Check if adding this chunk would exceed context limit
            if current_length + len(chunk_text) > self.max_context_length:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "".join(context_parts)
    
    def _create_rag_prompt(self, question: str, context: str) -> str:
        """
        Create an enhanced RAG prompt
        """
        return f"""You are an intelligent document analysis assistant. Answer questions based ONLY on the provided context from uploaded documents.

INSTRUCTIONS:
1. Answer based solely on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Include specific references to documents and page numbers when possible
4. Be precise and factual
5. If you're uncertain, indicate your level of confidence

CONTEXT FROM UPLOADED DOCUMENTS:
{context}

QUESTION: {question}

ANSWER (be specific and include document references):"""

    def _calculate_confidence(self, question: str, answer: str, context_chunks: List[Dict]) -> float:
        """
        Calculate confidence score for the answer
        """
        try:
            # Simple confidence calculation based on multiple factors
            confidence_factors = []
            
            # Factor 1: Answer length (longer answers might be more complete)
            answer_length_score = min(len(answer) / 500, 1.0)  # Normalize to 0-1
            confidence_factors.append(answer_length_score * 0.2)
            
            # Factor 2: Number of source chunks (more sources = higher confidence)
            source_score = min(len(context_chunks) / 5, 1.0)  # Normalize to 0-1
            confidence_factors.append(source_score * 0.3)
            
            # Factor 3: Question-answer relevance (simple keyword matching)
            question_words = set(question.lower().split())
            answer_words = set(answer.lower().split())
            relevance_score = len(question_words & answer_words) / max(len(question_words), 1)
            confidence_factors.append(relevance_score * 0.3)
            
            # Factor 4: Presence of document references in answer
            reference_indicators = ['document', 'page', 'section', 'according to', 'mentioned in']
            reference_score = sum(1 for indicator in reference_indicators if indicator in answer.lower()) / len(reference_indicators)
            confidence_factors.append(reference_score * 0.2)
            
            # Calculate final confidence
            final_confidence = sum(confidence_factors)
            
            # Add some randomness to avoid always same scores
            import random
            final_confidence += random.uniform(-0.05, 0.05)
            
            return max(0.1, min(1.0, final_confidence))  # Clamp between 0.1 and 1.0
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5  # Default confidence
    
    def _parse_related_questions(self, response: str) -> List[str]:
        """
        Parse related questions from AI response
        """
        try:
            # Split by lines and clean up
            lines = response.strip().split('\n')
            questions = []
            
            for line in lines:
                line = line.strip()
                # Remove numbering, bullets, etc.
                line = re.sub(r'^[\d\.\-\*\s]+', '', line)
                
                if line and len(line) > 10 and '?' in line:  # Basic question validation
                    questions.append(line)
            
            return questions
            
        except Exception as e:
            logger.error(f"Error parsing related questions: {e}")
            return []

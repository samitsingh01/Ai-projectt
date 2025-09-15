# intelligent-qa-service/services/rag_engine.py - FIXED for t2.large
import httpx
import json
import logging
from typing import List, Dict, Any
import re
import asyncio
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, bedrock_url: str):
        self.bedrock_url = bedrock_url
        self.max_context_length = 4000  # Reduced for efficiency
        self.timeout = 30.0              # Reduced timeout
        
    async def generate_answer(self, question: str, context_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate answer using RAG with lightweight processing"""
        try:
            if not context_chunks:
                return {
                    "answer": "I don't have any relevant document context to answer this question.",
                    "confidence": 0.0,
                    "model_used": "rag-engine"
        
    async def generate_related_questions(self, question: str, context_chunks: List[Dict]) -> List[str]:
        """Generate related questions using simple template-based approach"""
        try:
            if not context_chunks:
                return []
            
            # Extract key terms from chunks
            all_text = " ".join(chunk['text'][:200] for chunk in context_chunks[:2])
            keywords = self._extract_simple_keywords(all_text)
            
            # Simple question templates
            templates = [
                "What is {}?",
                "How does {} work?", 
                "Why is {} important?",
                "Where is {} used?",
                "When was {} developed?"
            ]
            
            related_questions = []
            question_lower = question.lower()
            
            for i, keyword in enumerate(keywords[:3]):
                if keyword.lower() not in question_lower and len(keyword) > 3:
                    template = templates[i % len(templates)]
                    related_questions.append(template.format(keyword))
            
            return related_questions
            
        except Exception as e:
            logger.error(f"Error generating related questions: {e}")
            return []
    
    async def _call_bedrock_service(self, prompt: str, max_tokens: int = 800, 
                                  temperature: float = 0.3) -> Dict[str, Any]:
        """Call Bedrock service with reduced parameters"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.bedrock_url}/generate",
                    json={
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "answer": result.get("response", ""),
                        "model_used": result.get("model_used", "bedrock-service")
                    }
                else:
                    logger.warning(f"Bedrock service HTTP error: {response.status_code}")
                    return None
                    
        except httpx.TimeoutException:
            logger.warning("Bedrock service timeout")
            return None
        except Exception as e:
            logger.warning(f"Bedrock service error: {e}")
            return None
    
    def _generate_extractive_answer(self, question: str, context_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate answer using simple extractive method"""
        try:
            question_words = set(self._extract_words(question))
            best_sentences = []
            
            # Find sentences with highest word overlap
            for chunk in context_chunks[:3]:  # Use top 3 chunks
                text = chunk['text']
                sentences = self._split_sentences(text)
                
                for sentence in sentences:
                    if len(sentence.strip()) < 20:  # Skip very short sentences
                        continue
                    
                    sentence_words = set(self._extract_words(sentence))
                    overlap = len(question_words & sentence_words)
                    
                    if overlap > 0:
                        score = overlap / max(len(question_words), 1)
                        filename = chunk.get('filename', 'Unknown')
                        best_sentences.append((sentence.strip(), score, filename))
            
            if best_sentences:
                # Sort by score and combine top sentences
                best_sentences.sort(key=lambda x: x[1], reverse=True)
                
                # Build answer from top sentences
                answer_parts = []
                used_files = set()
                
                for sentence, score, filename in best_sentences[:2]:
                    if filename not in used_files:
                        answer_parts.append(f"According to {filename}: {sentence}")
                        used_files.add(filename)
                    else:
                        answer_parts.append(sentence)
                
                answer = " ".join(answer_parts)
                confidence = min(0.7, best_sentences[0][1] + 0.2)
                
                return {
                    "answer": answer,
                    "confidence": confidence,
                    "model_used": "extractive-rag"
                }
            else:
                # Last resort: return snippet from best chunk
                best_chunk = context_chunks[0]
                snippet = best_chunk['text'][:300]
                filename = best_chunk.get('filename', 'Unknown')
                
                return {
                    "answer": f"Based on {filename}: {snippet}...",
                    "confidence": 0.4,
                    "model_used": "snippet-rag"
                }
                
        except Exception as e:
            logger.error(f"Error in extractive answer: {e}")
            return {
                "answer": "I couldn't extract a specific answer from the documents.",
                "confidence": 0.0,
                "model_used": "error-extractive"
            }
    
    def _build_efficient_context(self, chunks: List[Dict]) -> str:
        """Build context with strict length control"""
        try:
            context_parts = []
            current_length = 0
            
            for i, chunk in enumerate(chunks):
                filename = chunk.get('filename', f'Document_{i+1}')
                text = chunk['text']
                
                # Create chunk header
                header = f"[{filename}]"
                
                # Calculate available space
                available_space = self.max_context_length - current_length - len(header) - 10
                
                if available_space < 100:  # Not enough space
                    break
                
                # Truncate text if needed
                if len(text) > available_space:
                    text = text[:available_space] + "..."
                
                chunk_content = f"{header}\n{text}\n"
                context_parts.append(chunk_content)
                current_length += len(chunk_content)
                
                # Stop if we've used enough chunks
                if i >= 2 or current_length > self.max_context_length * 0.8:
                    break
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error building context: {e}")
            return ""
    
    def _create_simple_rag_prompt(self, question: str, context: str) -> str:
        """Create simple, efficient RAG prompt"""
        return f"""Answer the question based on the provided documents. Be specific and concise.

Documents:
{context}

Question: {question}

Answer:"""
    
    def _calculate_simple_confidence(self, question: str, answer: str, context_chunks: List[Dict]) -> float:
        """Calculate confidence using simple metrics"""
        try:
            factors = []
            
            # Factor 1: Answer length (20%)
            length_score = min(len(answer) / 200, 1.0)
            factors.append(length_score * 0.2)
            
            # Factor 2: Word overlap between question and answer (30%)
            question_words = set(self._extract_words(question))
            answer_words = set(self._extract_words(answer))
            overlap = len(question_words & answer_words)
            overlap_score = overlap / max(len(question_words), 1)
            factors.append(overlap_score * 0.3)
            
            # Factor 3: Number of source chunks (25%)
            source_score = min(len(context_chunks) / 3, 1.0)
            factors.append(source_score * 0.25)
            
            # Factor 4: Average chunk similarity (25%)
            avg_similarity = sum(chunk.get('similarity', 0.5) for chunk in context_chunks) / len(context_chunks)
            factors.append(avg_similarity * 0.25)
            
            final_confidence = sum(factors)
            return max(0.1, min(0.9, final_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract words using simple regex"""
        try:
            return re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        except Exception as e:
            logger.error(f"Error extracting words: {e}")
            return []
    
    def _extract_simple_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords using simple frequency analysis"""
        try:
            # Common stop words
            stop_words = {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
                'with', 'by', 'is', 'are', 'was', 'were', 'will', 'would', 'could',
                'have', 'has', 'had', 'this', 'that', 'these', 'those', 'a', 'an'
            }
            
            # Extract words and filter
            words = self._extract_words(text)
            keywords = [word for word in words if word not in stop_words and len(word) > 3]
            
            # Count frequency
            word_counts = Counter(keywords)
            return [word for word, count in word_counts.most_common(max_keywords)]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple regex"""
        try:
            # Simple sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.error(f"Error splitting sentences: {e}")
            return [text]
            
            # Build optimized context
            context = self._build_efficient_context

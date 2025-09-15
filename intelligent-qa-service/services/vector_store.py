# intelligent-qa-service/services/vector_store.py - FIXED for t2.large
import json
import logging
import mysql.connector
from mysql.connector import Error
from typing import List, Dict, Any, Optional
import re
from collections import Counter

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, database_url: str):
        self.database_url = database_url
        
    def _get_db_connection(self):
        """Get database connection with proper error handling"""
        try:
            url_parts = self.database_url.replace("mysql://", "").split("/")
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

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate simple text-based 'embedding' using word frequencies"""
        try:
            # Simple approach: use word frequency as features
            words = self._extract_words(text)
            word_counts = Counter(words)
            
            # Create a simple feature vector based on common words
            common_words = [
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
                'with', 'by', 'is', 'are', 'was', 'were', 'will', 'would', 'could',
                'have', 'has', 'had', 'do', 'does', 'did', 'can', 'may', 'might',
                'should', 'must', 'shall', 'this', 'that', 'these', 'those', 'a', 'an'
            ]
            
            # Create feature vector
            features = []
            for word in common_words:
                features.append(float(word_counts.get(word, 0)))
            
            # Add text length features
            features.extend([
                float(len(text)),
                float(len(words)),
                float(len(set(words))),  # unique words
                float(sum(word_counts.values()))  # total words
            ])
            
            # Normalize to prevent huge numbers
            max_val = max(features) if features else 1
            if max_val > 0:
                features = [f / max_val for f in features]
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * 40  # Return zeros if failed

    def _extract_words(self, text: str) -> List[str]:
        """Extract and clean words from text"""
        try:
            # Convert to lowercase and extract words
            words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
            return words
        except Exception as e:
            logger.error(f"Error extracting words: {e}")
            return []

    async def search_similar_chunks(self, session_id: str, query: str, 
                                  top_k: int = 5, confidence_threshold: float = 0.1) -> List[Dict]:
        """Search for similar chunks using simple text matching"""
        try:
            chunks = self._get_session_chunks(session_id)
            
            if not chunks:
                logger.warning(f"No chunks found for session {session_id}")
                return []
            
            logger.info(f"Retrieved {len(chunks)} chunks for session {session_id}")
            
            # Extract query words and keywords
            query_words = set(self._extract_words(query))
            query_keywords = self._extract_keywords(query)
            
            scored_chunks = []
            
            for chunk in chunks:
                chunk_text = chunk['chunk_text']
                chunk_words = set(self._extract_words(chunk_text))
                chunk_keywords = self._extract_keywords(chunk_text)
                
                # Calculate different similarity scores
                word_similarity = self._calculate_word_overlap(query_words, chunk_words)
                keyword_similarity = self._calculate_keyword_similarity(query_keywords, chunk_keywords)
                phrase_similarity = self._calculate_phrase_similarity(query, chunk_text)
                
                # Combine scores with weights
                final_score = (
                    word_similarity * 0.4 +
                    keyword_similarity * 0.4 +
                    phrase_similarity * 0.2
                )
                
                logger.debug(f"Chunk similarity: {final_score:.3f} for text: {chunk_text[:50]}...")
                
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
            
            # Sort by similarity
            scored_chunks.sort(key=lambda x: x['similarity'], reverse=True)
            
            # If no results meet threshold, return top chunks with minimum similarity
            if not scored_chunks and chunks:
                logger.warning("No chunks met threshold, returning top chunks with minimum similarity")
                for chunk in chunks[:top_k]:
                    chunk_data = {
                        'id': chunk['id'],
                        'file_id': chunk['file_id'],
                        'text': chunk['chunk_text'],
                        'similarity': 0.3,  # Minimum similarity
                        'metadata': json.loads(chunk['metadata']) if chunk['metadata'] else {},
                        'filename': chunk.get('filename', 'Unknown'),
                        'page_number': 1
                    }
                    scored_chunks.append(chunk_data)
            
            logger.info(f"Found {len(scored_chunks)} relevant chunks")
            return scored_chunks[:top_k]
                
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return []

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract important keywords from text"""
        try:
            # Common stop words to ignore
            stop_words = {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
                'with', 'by', 'is', 'are', 'was', 'were', 'will', 'would', 'could',
                'have', 'has', 'had', 'do', 'does', 'did', 'can', 'may', 'might',
                'should', 'must', 'shall', 'this', 'that', 'these', 'those', 'a', 'an',
                'it', 'he', 'she', 'they', 'we', 'you', 'me', 'him', 'her', 'them',
                'us', 'my', 'your', 'his', 'its', 'our', 'their', 'be', 'been', 'being'
            }
            
            # Extract words and filter
            words = self._extract_words(text)
            keywords = [word for word in words if word not in stop_words and len(word) > 2]
            
            # Count frequency and return most common
            word_counts = Counter(keywords)
            return [word for word, count in word_counts.most_common(max_keywords)]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []

    def _calculate_word_overlap(self, words1: set, words2: set) -> float:
        """Calculate word overlap similarity"""
        try:
            if not words1 or not words2:
                return 0.0
            
            intersection = words1 & words2
            union = words1 | words2
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating word overlap: {e}")
            return 0.0

    def _calculate_keyword_similarity(self, keywords1: List[str], keywords2: List[str]) -> float:
        """Calculate keyword similarity"""
        try:
            if not keywords1 or not keywords2:
                return 0.0
            
            set1 = set(keywords1)
            set2 = set(keywords2)
            
            intersection = set1 & set2
            union = set1 | set2
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating keyword similarity: {e}")
            return 0.0

    def _calculate_phrase_similarity(self, text1: str, text2: str) -> float:
        """Calculate phrase-level similarity"""
        try:
            # Look for common 2-3 word phrases
            phrases1 = self._extract_phrases(text1)
            phrases2 = self._extract_phrases(text2)
            
            if not phrases1 or not phrases2:
                return 0.0
            
            common_phrases = set(phrases1) & set(phrases2)
            total_phrases = set(phrases1) | set(phrases2)
            
            return len(common_phrases) / len(total_phrases) if total_phrases else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating phrase similarity: {e}")
            return 0.0

    def _extract_phrases(self, text: str) -> List[str]:
        """Extract 2-3 word phrases from text"""
        try:
            words = self._extract_words(text)
            phrases = []
            
            # Extract 2-word phrases
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                if len(phrase) > 6:  # Minimum phrase length
                    phrases.append(phrase)
            
            # Extract 3-word phrases
            for i in range(len(words) - 2):
                phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                phrases.append(phrase)
            
            return phrases
            
        except Exception as e:
            logger.error(f"Error extracting phrases: {e}")
            return []

    def _get_session_chunks(self, session_id: str) -> List[Dict]:
        """Get all document chunks for a session"""
        try:
            connection = self._get_db_connection()
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
            
            results = cursor.fetchall()
            cursor.close()
            connection.close()
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting session chunks: {e}")
            return []

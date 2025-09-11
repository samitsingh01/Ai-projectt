# intelligent-qa-service/services/vector_store.py
import httpx
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import mysql.connector
from mysql.connector import Error
import os
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, database_url: str, embedding_dimension: int = 1536):
        self.database_url = database_url
        self.embedding_dimension = embedding_dimension
        
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text using a simple approach
        In production, you'd use Amazon Titan Embeddings or similar
        """
        try:
            # Create a simple embedding using text characteristics
            embedding = self._create_simple_embedding(text)
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return a default embedding on error
            return [0.0] * self.embedding_dimension
    
    async def search_similar_chunks(self, session_id: str, query: str, 
                                  top_k: int = 5, confidence_threshold: float = 0.7) -> List[Dict]:
        """
        Search for similar document chunks using vector similarity
        """
        try:
            # Generate query embedding
            query_embedding = await self.generate_embedding(query)
            
            # Get all chunks for the session
            chunks = self._get_session_chunks(session_id)
            
            if not chunks:
                return []
            
            # Calculate similarities
            similarities = []
            for chunk in chunks:
                try:
                    chunk_embedding = json.loads(chunk['chunk_embedding'])
                    similarity = self._calculate_cosine_similarity(query_embedding, chunk_embedding)
                    
                    if similarity >= confidence_threshold:
                        chunk_data = {
                            'id': chunk['id'],
                            'file_id': chunk['file_id'],
                            'text': chunk['chunk_text'],
                            'similarity': similarity,
                            'metadata': json.loads(chunk['metadata']) if chunk['metadata'] else {},
                            'filename': chunk.get('filename', 'Unknown'),
                            'page_number': chunk.get('page_number', 1)
                        }
                        similarities.append(chunk_data)
                        
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk['id']}: {e}")
                    continue
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return []
    
    def _get_session_chunks(self, session_id: str) -> List[Dict]:
        """
        Get all document chunks for a session with file information
        """
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
                    dc.chunk_embedding,
                    dc.metadata,
                    uf.original_name as filename,
                    JSON_EXTRACT(dc.metadata, '$.page_number') as page_number
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
    
    def _get_db_connection(self):
        """Get database connection"""
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
                database=url_parts[1]
            )
            return connection
        except Error as e:
            logger.error(f"Database connection error: {e}")
            return None
    
    def _create_simple_embedding(self, text: str) -> List[float]:
        """
        Create a simple embedding based on text characteristics
        This is a placeholder - in production, use proper embedding models
        """
        try:
            # Simple feature extraction
            features = []
            
            # Text length features
            features.extend([
                len(text) / 1000,  # Normalized length
                len(text.split()) / 100,  # Word count
                len(set(text.lower().split())) / 100,  # Unique words
            ])
            
            # Character frequency features
            char_counts = {}
            for char in text.lower():
                char_counts[char] = char_counts.get(char, 0) + 1
            
            # Add frequency of common characters
            common_chars = 'abcdefghijklmnopqrstuvwxyz0123456789 .,!?'
            for char in common_chars:
                features.append(char_counts.get(char, 0) / max(len(text), 1))
            
            # Word-based features
            words = text.lower().split()
            if words:
                # Average word length
                avg_word_length = sum(len(word) for word in words) / len(words)
                features.append(avg_word_length / 10)
                
                # Common word frequencies
                common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
                               'with', 'by', 'is', 'are', 'was', 'were', 'will', 'would', 'could']
                for word in common_words:
                    features.append(words.count(word) / len(words))
            else:
                features.extend([0.0] * 21)  # 1 + 20 common words
            
            # Pad or truncate to desired dimension
            while len(features) < self.embedding_dimension:
                features.append(0.0)
            
            return features[:self.embedding_dimension]
            
        except Exception as e:
            logger.error(f"Error creating simple embedding: {e}")
            return [0.0] * self.embedding_dimension
    
    def _calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([vec1], [vec2])[0][0]
            
            # Ensure similarity is between 0 and 1
            return max(0.0, min(1.0, float(similarity)))
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

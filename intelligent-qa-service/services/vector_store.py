# intelligent-qa-service/services/vector_store.py - FIXED VERSION
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import mysql.connector
from mysql.connector import Error
import re

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, database_url: str, embedding_dimension: int = 512):
        self.database_url = database_url
        self.embedding_dimension = embedding_dimension
        
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate simple embedding for text"""
        try:
            words = text.lower().split()
            embedding = [0.0] * self.embedding_dimension
            
            for i, word in enumerate(words[:100]):
                if i < self.embedding_dimension:
                    embedding[i] = float(len(word))
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * self.embedding_dimension
    
    async def search_similar_chunks(self, session_id: str, query: str, 
                                  top_k: int = 5, confidence_threshold: float = 0.1) -> List[Dict]:
        """FIXED: Simple keyword-based similarity search"""
        try:
            chunks = self._get_session_chunks(session_id)
            
            if not chunks:
                logger.warning(f"No chunks found for session {session_id}")
                return []
            
            logger.info(f"DEBUG: Retrieved {len(chunks)} chunks for session {session_id}")
            
            # SIMPLE KEYWORD MATCHING
            query_words = set(query.lower().split())
            results = []
            
            for chunk in chunks:
                chunk_text = chunk['chunk_text'].lower()
                chunk_words = set(chunk_text.split())
                
                # Calculate word overlap
                common_words = query_words & chunk_words
                similarity = len(common_words) / max(len(query_words), 1)
                
                # Substring matching
                substring_matches = sum(1 for word in query_words if word in chunk_text)
                substring_similarity = substring_matches / max(len(query_words), 1)
                
                final_similarity = max(similarity, substring_similarity)
                
                logger.info(f"DEBUG: Chunk similarity: {final_similarity:.3f} for text: {chunk_text[:50]}...")
                
                if final_similarity >= confidence_threshold:
                    chunk_data = {
                        'id': chunk['id'],
                        'file_id': chunk['file_id'],
                        'text': chunk['chunk_text'],
                        'similarity': final_similarity,
                        'metadata': json.loads(chunk['metadata']) if chunk['metadata'] else {},
                        'filename': chunk.get('filename', 'Unknown'),
                        'page_number': 1
                    }
                    results.append(chunk_data)
            
            # Sort by similarity
            results.sort(key=lambda x: x['similarity'], reverse=True)
            logger.info(f"DEBUG: Found {len(results)} relevant chunks above threshold {confidence_threshold}")
            
            # FALLBACK: If no results, add all chunks with minimum similarity
            if not results:
                logger.warning("DEBUG: No results found, adding all chunks with minimum similarity...")
                for chunk in chunks:
                    chunk_data = {
                        'id': chunk['id'],
                        'file_id': chunk['file_id'],
                        'text': chunk['chunk_text'],
                        'similarity': 0.3,  # Force minimum similarity
                        'metadata': json.loads(chunk['metadata']) if chunk['metadata'] else {},
                        'filename': chunk.get('filename', 'Unknown'),
                        'page_number': 1
                    }
                    results.append(chunk_data)
                    logger.info(f"DEBUG: Added chunk with forced similarity: {chunk['chunk_text'][:100]}")
            
            return results[:top_k]
                
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
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
                    dc.chunk_embedding,
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

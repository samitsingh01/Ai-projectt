# intelligent-qa-service/services/document_chunker.py - FIXED for t2.large
import re
import logging
from typing import List, Dict, Any
import json

logger = logging.getLogger(__name__)

class DocumentChunker:
    def __init__(self, max_chunk_size: int = 800):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = 100      # Reduced overlap for efficiency
        self.min_chunk_size = 100    # Minimum viable chunk size
        
    def chunk_document(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """Split document into manageable chunks using simple methods"""
        try:
            # Clean and preprocess text
            cleaned_text = self._clean_text(text)
            
            if len(cleaned_text) < self.min_chunk_size:
                logger.warning(f"Document {filename} too short for chunking")
                return [{
                    "text": cleaned_text,
                    "metadata": {
                        "chunk_index": 0,
                        "filename": filename,
                        "document_type": "short",
                        "page_number": 1,
                        "section": "",
                        "word_count": len(cleaned_text.split()),
                        "char_count": len(cleaned_text)
                    }
                }]
            
            # Determine document type and chunking strategy
            doc_type = self._detect_document_type(filename, cleaned_text)
            
            if doc_type == "structured":
                chunks = self._chunk_structured_document(cleaned_text)
            else:
                chunks = self._chunk_plain_text(cleaned_text)
            
            # Add metadata to chunks
            enriched_chunks = []
            for i, chunk in enumerate(chunks):
                if len(chunk["text"].strip()) >= self.min_chunk_size:
                    enriched_chunk = {
                        "text": chunk["text"],
                        "metadata": {
                            "chunk_index": i,
                            "filename": filename,
                            "document_type": doc_type,
                            "page_number": chunk.get("page_number", 1),
                            "section": chunk.get("section", ""),
                            "word_count": len(chunk["text"].split()),
                            "char_count": len(chunk["text"])
                        }
                    }
                    enriched_chunks.append(enriched_chunk)
            
            if not enriched_chunks:
                # Fallback: create single chunk
                enriched_chunks = [{
                    "text": cleaned_text[:self.max_chunk_size],
                    "metadata": {
                        "chunk_index": 0,
                        "filename": filename,
                        "document_type": "fallback",
                        "page_number": 1,
                        "section": "",
                        "word_count": len(cleaned_text.split()),
                        "char_count": len(cleaned_text)
                    }
                }]
            
            logger.info(f"Created {len(enriched_chunks)} chunks for {filename}")
            return enriched_chunks
            
        except Exception as e:
            logger.error(f"Error chunking document {filename}: {e}")
            # Return safe fallback chunk
            return [{
                "text": text[:self.max_chunk_size] if text else "Empty document",
                "metadata": {
                    "chunk_index": 0,
                    "filename": filename,
                    "document_type": "error",
                    "page_number": 1,
                    "section": "error",
                    "word_count": len(text.split()) if text else 0,
                    "char_count": len(text) if text else 0
                }
            }]
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text using simple methods"""
        try:
            if not text:
                return ""
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove control characters
            text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
            
            # Normalize line breaks
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            
            # Remove multiple consecutive newlines
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text if text else ""
    
    def _detect_document_type(self, filename: str, text: str) -> str:
        """Detect document type using simple heuristics"""
        try:
            filename_lower = filename.lower()
            
            # Check file extension
            if filename_lower.endswith('.json'):
                return "json"
            elif filename_lower.endswith('.csv'):
                return "csv"
            elif filename_lower.endswith(('.md', '.markdown')):
                return "markdown"
            elif filename_lower.endswith('.xml'):
                return "xml"
            
            # Check content patterns for structure
            if self._has_structured_content(text):
                return "structured"
            
            return "plain_text"
            
        except Exception as e:
            logger.error(f"Error detecting document type: {e}")
            return "plain_text"
    
    def _has_structured_content(self, text: str) -> bool:
        """Check if document has structured content using simple patterns"""
        try:
            patterns = [
                r'^#+\s',                    # Markdown headers
                r'^\d+\.\s+',               # Numbered lists
                r'^[-*+]\s+',               # Bullet lists
                r'^Chapter\s+\d+',          # Chapters
                r'^Section\s+\d+',          # Sections
                r'^\s*[A-Z][A-Z\s]+:',     # All caps headers
                r'^\s*\w+:\s*$',           # Key-value pairs
            ]
            
            lines = text.split('\n')[:50]  # Check first 50 lines only
            structured_lines = 0
            
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                    
                for pattern in patterns:
                    if re.match(pattern, line_stripped, re.MULTILINE | re.IGNORECASE):
                        structured_lines += 1
                        break
            
            return structured_lines >= 3  # If 3+ structured elements found
            
        except Exception as e:
            logger.error(f"Error checking structured content: {e}")
            return False
    
    def _chunk_structured_document(self, text: str) -> List[Dict[str, Any]]:
        """Chunk structured documents by sections"""
        try:
            chunks = []
            current_chunk = ""
            current_section = ""
            page_number = 1
            
            lines = text.split('\n')
            
            for line in lines:
                line_stripped = line.strip()
                
                # Check if this is a header/section
                if self._is_header_line(line_stripped):
                    # Save previous chunk if substantial
                    if len(current_chunk.strip()) >= self.min_chunk_size:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "section": current_section,
                            "page_number": page_number
                        })
                    
                    # Start new chunk
                    current_section = line_stripped[:100]  # Limit section name length
                    current_chunk = line + '\n'
                else:
                    current_chunk += line + '\n'
                    
                    # Check if chunk is getting too large
                    if len(current_chunk) >= self.max_chunk_size:
                        # Find good breaking point
                        break_point = self._find_simple_break_point(current_chunk)
                        
                        chunks.append({
                            "text": current_chunk[:break_point].strip(),
                            "section": current_section,
                            "page_number": page_number
                        })
                        
                        # Start new chunk with overlap
                        overlap_start = max(0, break_point - self.overlap_size)
                        current_chunk = current_chunk[overlap_start:]
                        page_number += 1
            
            # Add final chunk
            if len(current_chunk.strip()) >= self.min_chunk_size:
                chunks.append({
                    "text": current_chunk.strip(),
                    "section": current_section,
                    "page_number": page_number
                })
            
            return chunks if chunks else [{"text": text, "page_number": 1}]
            
        except Exception as e:
            logger.error(f"Error chunking structured document: {e}")
            return self._chunk_plain_text(text)
    
    def _chunk_plain_text(self, text: str) -> List[Dict[str, Any]]:
        """Chunk plain text using simple paragraph/sentence boundaries"""
        try:
            chunks = []
            
            # First try to split by paragraphs
            paragraphs = text.split('\n\n')
            current_chunk = ""
            page_number = 1
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # If adding this paragraph would exceed max size
                if len(current_chunk) + len(paragraph) + 2 > self.max_chunk_size:
                    # Save current chunk if it's substantial
                    if len(current_chunk.strip()) >= self.min_chunk_size:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "page_number": page_number
                        })
                        page_number += 1
                    
                    # Start new chunk
                    current_chunk = paragraph
                else:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += '\n\n' + paragraph
                    else:
                        current_chunk = paragraph
            
            # Add final chunk
            if len(current_chunk.strip()) >= self.min_chunk_size:
                chunks.append({
                    "text": current_chunk.strip(),
                    "page_number": page_number
                })
            
            # If no paragraph-based chunks work, fall back to sentence splitting
            if not chunks:
                chunks = self._chunk_by_sentences(text)
            
            # Final fallback: split by character count
            if not chunks:
                chunks = self._chunk_by_size(text)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking plain text: {e}")
            return [{"text": text[:self.max_chunk_size], "page_number": 1}]
    
    def _chunk_by_sentences(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text by sentences"""
        try:
            # Simple sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            chunks = []
            current_chunk = ""
            page_number = 1
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Check if adding sentence would exceed limit
                if len(current_chunk) + len(sentence) + 1 > self.max_chunk_size:
                    if len(current_chunk.strip()) >= self.min_chunk_size:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "page_number": page_number
                        })
                        page_number += 1
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += ' ' + sentence
                    else:
                        current_chunk = sentence
            
            # Add final chunk
            if len(current_chunk.strip()) >= self.min_chunk_size:
                chunks.append({
                    "text": current_chunk.strip(),
                    "page_number": page_number
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking by sentences: {e}")
            return []
    
    def _chunk_by_size(self, text: str) -> List[Dict[str, Any]]:
        """Simple fallback chunking by character size"""
        try:
            chunks = []
            start = 0
            page_number = 1
            
            while start < len(text):
                end = min(start + self.max_chunk_size, len(text))
                
                # Try to break at word boundary
                if end < len(text):
                    # Look for space within last 50 characters
                    search_start = max(start, end - 50)
                    last_space = text.rfind(' ', search_start, end)
                    if last_space > start:
                        end = last_space
                
                chunk_text = text[start:end].strip()
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append({
                        "text": chunk_text,
                        "page_number": page_number
                    })
                    page_number += 1
                
                start = end - self.overlap_size if end - self.overlap_size > start else end
            
            return chunks if chunks else [{"text": text, "page_number": 1}]
            
        except Exception as e:
            logger.error(f"Error chunking by size: {e}")
            return [{"text": text, "page_number": 1}]
    
    def _is_header_line(self, line: str) -> bool:
        """Simple check if line is likely a header"""
        try:
            if not line or len(line) > 200:  # Too long to be header
                return False
            
            header_patterns = [
                r'^#+\s+',                  # Markdown headers
                r'^Chapter\s+\d+',          # Chapters
                r'^Section\s+\d+',          # Sections
                r'^\d+\.\s*[A-Z]',         # Numbered sections
                r'^[A-Z][A-Z\s]+:?\s*$',   # All caps headers
                r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*:?\s*$',  # Title case
            ]
            
            for pattern in header_patterns:
                if re.match(pattern, line):
                    return True
            
            # Check if line is short and mostly uppercase
            if len(line) <= 80 and len(line.split()) <= 8:
                uppercase_chars = sum(1 for c in line if c.isupper())
                total_chars = sum(1 for c in line if c.isalpha())
                if total_chars > 0 and uppercase_chars / total_chars > 0.5:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking header line: {e}")
            return False
    
    def _find_simple_break_point(self, text: str) -> int:
        """Find a simple breaking point in text"""
        try:
            target_length = min(len(text), self.max_chunk_size)
            
            # Look for paragraph breaks
            last_double_newline = text.rfind('\n\n', 0, target_length)
            if last_double_newline > self.min_chunk_size:
                return last_double_newline + 2
            
            # Look for sentence endings
            sentence_endings = ['. ', '! ', '? ']
            best_ending = -1
            
            for ending in sentence_endings:
                pos = text.rfind(ending, 0, target_length)
                if pos > best_ending and pos > self.min_chunk_size:
                    best_ending = pos + len(ending)
            
            if best_ending > 0:
                return best_ending
            
            # Look for any period
            last_period = text.rfind('.', 0, target_length)
            if last_period > self.min_chunk_size:
                return last_period + 1
            
            # Look for line breaks
            last_newline = text.rfind('\n', 0, target_length)
            if last_newline > self.min_chunk_size:
                return last_newline + 1
            
            # Look for word boundaries
            last_space = text.rfind(' ', 0, target_length)
            if last_space > self.min_chunk_size:
                return last_space + 1
            
            # Fallback: use target length
            return target_length
            
        except Exception as e:
            logger.error(f"Error finding break point: {e}")
            return min(len(text), self.max_chunk_size)

# intelligent-qa-service/services/document_chunker.py
import re
import logging
from typing import List, Dict, Any
import json

logger = logging.getLogger(__name__)

class DocumentChunker:
    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = 200     # Overlap between chunks
        self.min_chunk_size = 100   # Minimum chunk size
        
    def chunk_document(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """
        Split document into semantic chunks
        """
        try:
            # Clean and preprocess text
            cleaned_text = self._clean_text(text)
            
            # Determine document type and use appropriate chunking strategy
            doc_type = self._detect_document_type(filename, cleaned_text)
            
            if doc_type == "structured":
                chunks = self._chunk_structured_document(cleaned_text)
            else:
                chunks = self._chunk_plain_text(cleaned_text)
            
            # Add metadata to chunks
            enriched_chunks = []
            for i, chunk in enumerate(chunks):
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
            
            logger.info(f"Created {len(enriched_chunks)} chunks for {filename}")
            return enriched_chunks
            
        except Exception as e:
            logger.error(f"Error chunking document {filename}: {e}")
            # Return a single chunk with the entire text on error
            return [{
                "text": text[:self.max_chunk_size],
                "metadata": {
                    "chunk_index": 0,
                    "filename": filename,
                    "document_type": "unknown",
                    "page_number": 1,
                    "section": "",
                    "word_count": len(text.split()),
                    "char_count": len(text)
                }
            }]
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        """
        try:
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters that might interfere with processing
            text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
            
            # Normalize line breaks
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text
    
    def _detect_document_type(self, filename: str, text: str) -> str:
        """
        Detect document type to apply appropriate chunking strategy
        """
        try:
            filename_lower = filename.lower()
            
            # Check file extension
            if filename_lower.endswith('.json'):
                return "json"
            elif filename_lower.endswith('.csv'):
                return "csv"
            elif filename_lower.endswith(('.md', '.markdown')):
                return "markdown"
            
            # Check content patterns
            if self._has_structured_content(text):
                return "structured"
            
            return "plain_text"
            
        except Exception as e:
            logger.error(f"Error detecting document type: {e}")
            return "plain_text"
    
    def _has_structured_content(self, text: str) -> bool:
        """
        Check if document has structured content (headers, lists, etc.)
        """
        patterns = [
            r'^#+\s',  # Markdown headers
            r'^\d+\.\s',  # Numbered lists
            r'^[-*+]\s',  # Bullet lists
            r'^Chapter\s+\d+',  # Chapters
            r'^Section\s+\d+',  # Sections
            r'^\s*[A-Z][A-Z\s]+:',  # All caps headers
        ]
        
        lines = text.split('\n')
        structured_lines = 0
        
        for line in lines[:50]:  # Check first 50 lines
            for pattern in patterns:
                if re.match(pattern, line.strip(), re.MULTILINE):
                    structured_lines += 1
                    break
        
        return structured_lines > 3  # If more than 3 structured elements found
    
    def _chunk_structured_document(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk structured documents by sections and headers
        """
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
                    # Save previous chunk if it's substantial
                    if len(current_chunk) > self.min_chunk_size:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "section": current_section,
                            "page_number": page_number
                        })
                    
                    # Start new chunk with header
                    current_section = line_stripped
                    current_chunk = line + '\n'
                else:
                    current_chunk += line + '\n'
                    
                    # Check if chunk is getting too large
                    if len(current_chunk) > self.max_chunk_size:
                        # Find a good breaking point
                        break_point = self._find_break_point(current_chunk)
                        
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
            if len(current_chunk.strip()) > self.min_chunk_size:
                chunks.append({
                    "text": current_chunk.strip(),
                    "section": current_section,
                    "page_number": page_number
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking structured document: {e}")
            return self._chunk_plain_text(text)
    
    def _chunk_plain_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk plain text documents
        """
        try:
            chunks = []
            start = 0
            page_number = 1
            
            while start < len(text):
                end = start + self.max_chunk_size
                
                if end >= len(text):
                    # Last chunk
                    chunk_text = text[start:].strip()
                    if len(chunk_text) > self.min_chunk_size:
                        chunks.append({
                            "text": chunk_text,
                            "page_number": page_number
                        })
                    break
                
                # Find a good breaking point
                break_point = self._find_break_point(text[start:end])
                actual_end = start + break_point
                
                chunk_text = text[start:actual_end].strip()
                if len(chunk_text) > self.min_chunk_size:
                    chunks.append({
                        "text": chunk_text,
                        "page_number": page_number
                    })
                    page_number += 1
                
                # Move start position with overlap
                start = actual_end - self.overlap_size
                if start < 0:
                    start = actual_end
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking plain text: {e}")
            return [{"text": text, "page_number": 1}]
    
    def _is_header_line(self, line: str) -> bool:
        """
        Check if a line is likely a header
        """
        header_patterns = [
            r'^#+\s',  # Markdown headers
            r'^Chapter\s+\d+',
            r'^Section\s+\d+',
            r'^\d+\.\s*[A-Z]',  # Numbered sections
            r'^[A-Z][A-Z\s]+:$',  # All caps headers ending with colon
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*:$',  # Title case headers with colon
        ]
        
        for pattern in header_patterns:
            if re.match(pattern, line):
                return True
        
        # Check if line is short and mostly uppercase
        if len(line) < 100 and len(line.split()) <= 8:
            uppercase_ratio = sum(1 for c in line if c.isupper()) / max(len(line), 1)
            if uppercase_ratio > 0.5:
                return True
        
        return False
    
    def _find_break_point(self, text: str) -> int:
        """
        Find a good point to break text (end of sentence, paragraph, etc.)
        """
        try:
            # Look for paragraph breaks first
            paragraph_breaks = [m.start() for m in re.finditer(r'\n\s*\n', text)]
            if paragraph_breaks:
                # Find the paragraph break closest to 80% of max chunk size
                target = int(self.max_chunk_size * 0.8)
                best_break = min(paragraph_breaks, key=lambda x: abs(x - target))
                if best_break > self.min_chunk_size:
                    return best_break
            
            # Look for sentence endings
            sentence_endings = [m.end() for m in re.finditer(r'[.!?]\s+', text)]
            if sentence_endings:
                target = int(self.max_chunk_size * 0.8)
                best_break = min(sentence_endings, key=lambda x: abs(x - target))
                if best_break > self.min_chunk_size:
                    return best_break
            
            # Look for line breaks
            line_breaks = [m.start() for m in re.finditer(r'\n', text)]
            if line_breaks:
                target = int(self.max_chunk_size * 0.8)
                best_break = min(line_breaks, key=lambda x: abs(x - target))
                if best_break > self.min_chunk_size:
                    return best_break
            
            # If no good break point found, use word boundaries
            words = text.split()
            current_length = 0
            for i, word in enumerate(words):
                current_length += len(word) + 1  # +1 for space
                if current_length > self.max_chunk_size * 0.8:
                    # Find the position in original text
                    break_text = ' '.join(words[:i])
                    return len(break_text)
            
            return len(text)
            
        except Exception as e:
            logger.error(f"Error finding break point: {e}")
            return min(self.max_chunk_size, len(text))

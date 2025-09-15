# intelligent-qa-service/services/citation_tracker.py - FIXED for t2.large
import re
import logging
from typing import List, Dict, Any
from collections import Counter

logger = logging.getLogger(__name__)

class CitationTracker:
    def __init__(self):
        self.max_snippet_length = 150
        self.min_snippet_length = 30
        
    def generate_citations(self, chunks: List[Dict], answer: str) -> List[Dict[str, Any]]:
        """Generate citations using simple text matching"""
        try:
            citations = []
            
            for chunk in chunks:
                # Calculate simple relevance score
                relevance_score = self._calculate_simple_relevance(chunk, answer)
                
                # Extract relevant snippet
                snippet = self._extract_simple_snippet(chunk['text'], answer)
                
                # Create citation
                citation = {
                    "source_id": chunk['id'],
                    "document": chunk.get('filename', 'Unknown'),
                    "page": chunk.get('page_number', 1),
                    "section": chunk.get('metadata', {}).get('section', ''),
                    "snippet": snippet,
                    "relevance_score": relevance_score,
                    "similarity": chunk.get('similarity', 0.0),
                    "citation_format": self._format_simple_citation(chunk),
                }
                
                citations.append(citation)
            
            # Sort by combined relevance and similarity
            citations.sort(key=lambda x: (x['relevance_score'] + x['similarity']) / 2, reverse=True)
            
            return citations
            
        except Exception as e:
            logger.error(f"Error generating citations: {e}")
            return []
    
    def _calculate_simple_relevance(self, chunk: Dict, answer: str) -> float:
        """Calculate relevance using simple word overlap"""
        try:
            chunk_text = chunk['text'].lower()
            answer_text = answer.lower()
            
            # Extract words
            chunk_words = set(self._extract_words(chunk_text))
            answer_words = set(self._extract_words(answer_text))
            
            if not chunk_words or not answer_words:
                return 0.0
            
            # Calculate word overlap
            overlap = chunk_words & answer_words
            union = chunk_words | answer_words
            
            base_score = len(overlap) / len(union) if union else 0.0
            
            # Boost for common phrases
            phrase_boost = self._calculate_phrase_overlap(chunk_text, answer_text)
            
            # Boost for exact word matches in answer
            exact_boost = self._calculate_exact_matches(chunk_text, answer_text)
            
            final_score = base_score + (phrase_boost * 0.3) + (exact_boost * 0.2)
            
            return min(1.0, final_score)
            
        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            return 0.0
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract words using simple regex"""
        try:
            return re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        except Exception as e:
            logger.error(f"Error extracting words: {e}")
            return []
    
    def _calculate_phrase_overlap(self, text1: str, text2: str) -> float:
        """Calculate overlap of 2-3 word phrases"""
        try:
            phrases1 = self._extract_simple_phrases(text1)
            phrases2 = self._extract_simple_phrases(text2)
            
            if not phrases1 or not phrases2:
                return 0.0
            
            common_phrases = set(phrases1) & set(phrases2)
            total_phrases = set(phrases1) | set(phrases2)
            
            return len(common_phrases) / len(total_phrases) if total_phrases else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating phrase overlap: {e}")
            return 0.0
    
    def _extract_simple_phrases(self, text: str) -> List[str]:
        """Extract 2-3 word phrases"""
        try:
            words = self._extract_words(text)
            phrases = []
            
            # Extract 2-word phrases
            for i in range(len(words) - 1):
                if len(words[i]) > 2 and len(words[i+1]) > 2:
                    phrase = f"{words[i]} {words[i+1]}"
                    phrases.append(phrase)
            
            # Extract 3-word phrases (limited)
            for i in range(min(len(words) - 2, 20)):  # Limit for efficiency
                phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                phrases.append(phrase)
            
            return phrases
            
        except Exception as e:
            logger.error(f"Error extracting phrases: {e}")
            return []
    
    def _calculate_exact_matches(self, source_text: str, answer_text: str) -> float:
        """Find exact word sequences that appear in both texts"""
        try:
            # Look for 4+ word sequences
            source_words = self._extract_words(source_text)
            answer_words = self._extract_words(answer_text)
            
            matches = 0
            total_sequences = 0
            
            # Check 4-word sequences in answer
            for i in range(len(answer_words) - 3):
                total_sequences += 1
                sequence = ' '.join(answer_words[i:i+4])
                source_text_joined = ' '.join(source_words)
                
                if sequence in source_text_joined:
                    matches += 1
            
            return matches / total_sequences if total_sequences > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating exact matches: {e}")
            return 0.0
    
    def _extract_simple_snippet(self, chunk_text: str, answer: str) -> str:
        """Extract relevant snippet using simple sentence scoring"""
        try:
            # Split into sentences
            sentences = self._split_sentences(chunk_text)
            
            if not sentences:
                return self._truncate_text(chunk_text)
            
            # Score sentences based on word overlap with answer
            answer_words = set(self._extract_words(answer))
            scored_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < self.min_snippet_length:
                    continue
                
                sentence_words = set(self._extract_words(sentence))
                overlap = len(answer_words & sentence_words)
                score = overlap / max(len(answer_words), 1)
                
                scored_sentences.append((sentence, score))
            
            if not scored_sentences:
                return self._truncate_text(chunk_text)
            
            # Sort by score and get best sentence
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            best_sentence = scored_sentences[0][0]
            
            # If sentence is too long, truncate it
            if len(best_sentence) > self.max_snippet_length:
                return self._truncate_text(best_sentence)
            
            # If sentence is short, try to add context
            if len(best_sentence) < self.max_snippet_length and len(scored_sentences) > 1:
                second_sentence = scored_sentences[1][0]
                combined = f"{best_sentence} {second_sentence}"
                
                if len(combined) <= self.max_snippet_length:
                    return combined
            
            return best_sentence
            
        except Exception as e:
            logger.error(f"Error extracting snippet: {e}")
            return self._truncate_text(chunk_text)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple regex"""
        try:
            # Simple sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.error(f"Error splitting sentences: {e}")
            return []
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to max snippet length"""
        try:
            if len(text) <= self.max_snippet_length:
                return text
            
            # Try to truncate at sentence boundary
            truncated = text[:self.max_snippet_length]
            
            # Look for sentence ending
            for ending in ['. ', '! ', '? ']:
                last_ending = truncated.rfind(ending)
                if last_ending > self.min_snippet_length:
                    return text[:last_ending + 1]
            
            # Look for any period
            last_period = truncated.rfind('.')
            if last_period > self.min_snippet_length:
                return text[:last_period + 1]
            
            # Fallback: truncate and add ellipsis
            return truncated.rstrip() + "..."
            
        except Exception as e:
            logger.error(f"Error truncating text: {e}")
            return text[:self.max_snippet_length] + "..."
    
    def _format_simple_citation(self, chunk: Dict) -> str:
        """Format citation in simple style"""
        try:
            filename = chunk.get('filename', 'Unknown Document')
            page = chunk.get('page_number', 1)
            
            citation = filename
            
            if page and page > 1:
                citation += f", p. {page}"
            
            return citation
            
        except Exception as e:
            logger.error(f"Error formatting citation: {e}")
            return "Unknown Source"

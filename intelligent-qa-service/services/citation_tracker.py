# intelligent-qa-service/services/citation_tracker.py
import re
import logging
from typing import List, Dict, Any
import difflib

logger = logging.getLogger(__name__)

class CitationTracker:
    def __init__(self):
        self.max_snippet_length = 150
        self.min_snippet_length = 30
        
    def generate_citations(self, chunks: List[Dict], answer: str) -> List[Dict[str, Any]]:
        """
        Generate citations by matching answer content to source chunks
        """
        try:
            citations = []
            
            for i, chunk in enumerate(chunks):
                # Calculate relevance score
                relevance_score = self._calculate_relevance(chunk, answer)
                
                # Extract relevant snippet
                snippet = self._extract_relevant_snippet(chunk['text'], answer)
                
                # Create citation
                citation = {
                    "source_id": chunk['id'],
                    "document": chunk['filename'],
                    "page": chunk.get('page_number', 1),
                    "section": chunk.get('metadata', {}).get('section', ''),
                    "snippet": snippet,
                    "relevance_score": relevance_score,
                    "similarity": chunk.get('similarity', 0.0),
                    "citation_format": self._format_citation(chunk),
                    "text_snippet": self._create_text_snippet(chunk['text'])
                }
                
                citations.append(citation)
            
            # Sort by relevance and similarity
            citations.sort(key=lambda x: (x['relevance_score'] + x['similarity']) / 2, reverse=True)
            
            return citations
            
        except Exception as e:
            logger.error(f"Error generating citations: {e}")
            return []
    
    def _calculate_relevance(self, chunk: Dict, answer: str) -> float:
        """
        Calculate how relevant a chunk is to the generated answer
        """
        try:
            chunk_text = chunk['text'].lower()
            answer_text = answer.lower()
            
            # Extract key phrases from answer
            answer_phrases = self._extract_key_phrases(answer_text)
            
            # Count phrase matches
            matches = 0
            total_phrases = len(answer_phrases)
            
            for phrase in answer_phrases:
                if phrase in chunk_text:
                    matches += 1
            
            # Calculate base relevance
            base_relevance = matches / max(total_phrases, 1)
            
            # Boost for direct quotes (text that appears exactly in both)
            direct_quotes = self._find_direct_quotes(chunk_text, answer_text)
            quote_boost = min(len(direct_quotes) * 0.1, 0.3)
            
            # Boost for similar sentence structures
            similarity_boost = self._calculate_text_similarity(chunk_text, answer_text) * 0.2
            
            final_relevance = min(1.0, base_relevance + quote_boost + similarity_boost)
            
            return final_relevance
            
        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            return 0.0
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """
        Extract key phrases from text
        """
        try:
            # Remove common stop words
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
                         'with', 'by', 'is', 'are', 'was', 'were', 'will', 'would', 'could',
                         'this', 'that', 'these', 'those', 'a', 'an'}
            
            # Extract phrases (2-4 words)
            words = re.findall(r'\b\w+\b', text.lower())
            phrases = []
            
            for i in range(len(words) - 1):
                # 2-word phrases
                if words[i] not in stop_words or words[i+1] not in stop_words:
                    phrase = f"{words[i]} {words[i+1]}"
                    if len(phrase) > 6:  # Minimum phrase length
                        phrases.append(phrase)
                
                # 3-word phrases
                if i < len(words) - 2:
                    phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                    if any(word not in stop_words for word in words[i:i+3]):
                        phrases.append(phrase)
            
            return list(set(phrases))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting key phrases: {e}")
            return []
    
    def _find_direct_quotes(self, source_text: str, answer_text: str) -> List[str]:
        """
        Find text that appears directly in both source and answer
        """
        try:
            quotes = []
            
            # Look for sequences of 4+ consecutive words
            source_words = re.findall(r'\b\w+\b', source_text.lower())
            answer_words = re.findall(r'\b\w+\b', answer_text.lower())
            
            for i in range(len(answer_words) - 3):
                for length in range(4, min(10, len(answer_words) - i + 1)):
                    phrase = ' '.join(answer_words[i:i+length])
                    source_phrase = ' '.join(source_words)
                    
                    if phrase in source_phrase and len(phrase) > 15:
                        quotes.append(phrase)
            
            return quotes
            
        except Exception as e:
            logger.error(f"Error finding direct quotes: {e}")
            return []
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using sequence matching
        """
        try:
            # Use difflib for sequence matching
            matcher = difflib.SequenceMatcher(None, text1, text2)
            return matcher.ratio()
            
        except Exception as e:
            logger.error(f"Error calculating text similarity: {e}")
            return 0.0
    
    def _extract_relevant_snippet(self, chunk_text: str, answer: str) -> str:
        """
        Extract the most relevant snippet from a chunk based on the answer
        """
        try:
            # Split chunk into sentences
            sentences = re.split(r'[.!?]+', chunk_text)
            
            if not sentences:
                return chunk_text[:self.max_snippet_length] + "..."
            
            # Score each sentence based on overlap with answer
            sentence_scores = []
            answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < self.min_snippet_length:
                    continue
                
                sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
                overlap = len(answer_words & sentence_words)
                score = overlap / max(len(sentence_words), 1)
                
                sentence_scores.append((sentence, score))
            
            if not sentence_scores:
                return chunk_text[:self.max_snippet_length] + "..."
            
            # Get the best sentence(s)
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top sentence and add context if needed
            best_sentence = sentence_scores[0][0]
            
            if len(best_sentence) < self.max_snippet_length and len(sentence_scores) > 1:
                # Add second best sentence if there's room
                second_sentence = sentence_scores[1][0]
                combined = f"{best_sentence}. {second_sentence}"
                if len(combined) <= self.max_snippet_length:
                    return combined
            
            # Truncate if too long
            if len(best_sentence) > self.max_snippet_length:
                return best_sentence[:self.max_snippet_length] + "..."
            
            return best_sentence
            
        except Exception as e:
            logger.error(f"Error extracting relevant snippet: {e}")
            return chunk_text[:self.max_snippet_length] + "..."
    
    def _format_citation(self, chunk: Dict) -> str:
        """
        Format citation in a standard academic style
        """
        try:
            filename = chunk.get('filename', 'Unknown Document')
            page = chunk.get('page_number', 1)
            section = chunk.get('metadata', {}).get('section', '')
            
            citation = f"{filename}"
            
            if page and page > 1:
                citation += f", p. {page}"
            
            if section:
                citation += f", {section}"
            
            return citation
            
        except Exception as e:
            logger.error(f"Error formatting citation: {e}")
            return "Unknown Source"
    
    def _create_text_snippet(self, text: str) -> str:
        """
        Create a clean text snippet for display
        """
        try:
            # Clean up text
            cleaned = re.sub(r'\s+', ' ', text).strip()
            
            # Truncate if needed
            if len(cleaned) > self.max_snippet_length:
                # Try to end at a sentence boundary
                truncated = cleaned[:self.max_snippet_length]
                last_period = truncated.rfind('.')
                last_question = truncated.rfind('?')
                last_exclamation = truncated.rfind('!')
                
                last_sentence_end = max(last_period, last_question, last_exclamation)
                
                if last_sentence_end > self.min_snippet_length:
                    return cleaned[:last_sentence_end + 1]
                else:
                    return truncated + "..."
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error creating text snippet: {e}")
            return text[:self.max_snippet_length] + "..."

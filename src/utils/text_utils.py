"""
Text processing utilities for PulseByte.
"""

import re
from typing import List, Dict, Any, Optional
from loguru import logger


def preprocess_text(text: str, 
                   remove_urls: bool = True,
                   remove_emails: bool = True,
                   remove_phone_numbers: bool = True,
                   normalize_whitespace: bool = True) -> str:
    """
    Preprocess text for analysis.
    
    Args:
        text: Text to preprocess
        remove_urls: Whether to remove URLs
        remove_emails: Whether to remove email addresses
        remove_phone_numbers: Whether to remove phone numbers
        normalize_whitespace: Whether to normalize whitespace
        
    Returns:
        Preprocessed text
    """
    if not text:
        return ""
    
    processed_text = text
    
    # Remove URLs
    if remove_urls:
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        processed_text = re.sub(url_pattern, '', processed_text)
    
    # Remove email addresses
    if remove_emails:
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        processed_text = re.sub(email_pattern, '', processed_text)
    
    # Remove phone numbers (basic pattern)
    if remove_phone_numbers:
        phone_pattern = r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
        processed_text = re.sub(phone_pattern, '', processed_text)
    
    # Normalize whitespace
    if normalize_whitespace:
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    return processed_text


def extract_entities(text: str, entity_types: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """
    Extract named entities from text using basic pattern matching.
    
    Args:
        text: Text to extract entities from
        entity_types: Types of entities to extract. If None, extracts all.
        
    Returns:
        Dictionary of entity_type -> list of entities
    """
    entities = {}
    
    if not text:
        return entities
    
    # Available entity types
    available_types = ['organizations', 'locations', 'money', 'dates', 'emails', 'urls']
    extract_types = entity_types or available_types
    
    try:
        if 'organizations' in extract_types:
            # Simple pattern for organizations (capitalized words, Inc, Corp, etc.)
            org_pattern = r'\b[A-Z][a-zA-Z\s&.]+(?:Inc|Corp|LLC|Ltd|Company|Group|Association|Foundation|Institute)\b'
            entities['organizations'] = list(set(re.findall(org_pattern, text)))
        
        if 'locations' in extract_types:
            # Simple pattern for locations (capitalized words that might be places)
            # This is very basic - a proper NER model would be much better
            location_pattern = r'\b[A-Z][a-zA-Z\s]+(?:City|State|Country|County|Province|Street|Avenue|Boulevard|Road)\b'
            entities['locations'] = list(set(re.findall(location_pattern, text)))
        
        if 'money' in extract_types:
            # Money amounts
            money_pattern = r'\$[0-9,]+(?:\.[0-9]{2})?|\b[0-9,]+\s*(?:dollars?|USD|cents?)\b'
            entities['money'] = list(set(re.findall(money_pattern, text, re.IGNORECASE)))
        
        if 'dates' in extract_types:
            # Basic date patterns
            date_patterns = [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
                r'\b\d{4}-\d{1,2}-\d{1,2}\b',  # YYYY-MM-DD
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'  # Month DD, YYYY
            ]
            dates = []
            for pattern in date_patterns:
                dates.extend(re.findall(pattern, text, re.IGNORECASE))
            entities['dates'] = list(set(dates))
        
        if 'emails' in extract_types:
            # Email addresses
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            entities['emails'] = list(set(re.findall(email_pattern, text)))
        
        if 'urls' in extract_types:
            # URLs
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            entities['urls'] = list(set(re.findall(url_pattern, text)))
        
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
    
    return entities


def summarize_text(text: str, max_sentences: int = 3) -> str:
    """
    Create a simple extractive summary of text.
    
    Args:
        text: Text to summarize
        max_sentences: Maximum number of sentences in summary
        
    Returns:
        Summary text
    """
    if not text:
        return ""
    
    try:
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Simple scoring: prefer sentences with important words
        important_words = [
            'said', 'announced', 'reported', 'according', 'confirmed',
            'new', 'first', 'major', 'significant', 'important',
            'million', 'billion', 'percent', 'year', 'today'
        ]
        
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            
            # Position score (earlier sentences get higher scores)
            score += max(0, len(sentences) - i) * 0.1
            
            # Important words score
            for word in important_words:
                score += sentence.lower().count(word) * 0.5
            
            # Length score (prefer medium-length sentences)
            words = sentence.split()
            if 10 <= len(words) <= 30:
                score += 1.0
            
            scored_sentences.append((score, sentence))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s[1] for s in scored_sentences[:max_sentences]]
        
        # Restore original order
        summary_sentences = []
        for sentence in sentences:
            if sentence in top_sentences:
                summary_sentences.append(sentence)
                if len(summary_sentences) >= max_sentences:
                    break
        
        return '. '.join(summary_sentences) + '.'
        
    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        return text[:500] + "..." if len(text) > 500 else text


def extract_keywords_advanced(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords using TF-IDF-like approach.
    
    Args:
        text: Text to extract keywords from
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of keywords
    """
    if not text:
        return []
    
    try:
        from collections import Counter
        import math
        
        # Clean and tokenize text
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'said', 'says', 'also', 'one', 'two', 'three', 'first', 'second',
            'new', 'old', 'good', 'bad', 'big', 'small', 'more', 'most', 'many'
        }
        
        # Filter words
        filtered_words = [
            word for word in words 
            if len(word) > 2 and word not in stop_words and word.isalpha()
        ]
        
        if not filtered_words:
            return []
        
        # Calculate word frequencies
        word_freq = Counter(filtered_words)
        
        # Calculate TF scores
        total_words = len(filtered_words)
        tf_scores = {word: freq / total_words for word, freq in word_freq.items()}
        
        # Simple IDF approximation (boost less common words)
        idf_scores = {}
        for word in tf_scores:
            # Simple heuristic: words that appear less frequently get higher IDF
            idf_scores[word] = math.log(total_words / word_freq[word])
        
        # Calculate TF-IDF scores
        tfidf_scores = {
            word: tf_scores[word] * idf_scores[word] 
            for word in tf_scores
        }
        
        # Get top keywords
        top_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, score in top_keywords[:max_keywords]]
        
        return keywords
        
    except Exception as e:
        logger.error(f"Error in advanced keyword extraction: {e}")
        # Fallback to simple extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = Counter(words)
        return [word for word, _ in word_freq.most_common(max_keywords)]


def clean_html(text: str) -> str:
    """
    Remove HTML tags and decode HTML entities.
    
    Args:
        text: Text with HTML content
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    try:
        import html
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    except Exception as e:
        logger.error(f"Error cleaning HTML: {e}")
        return text


def extract_reading_level(text: str) -> Dict[str, Any]:
    """
    Calculate basic readability metrics.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with readability metrics
    """
    if not text:
        return {}
    
    try:
        # Count sentences, words, and syllables
        sentences = len(re.findall(r'[.!?]+', text))
        words = len(text.split())
        
        # Simple syllable counting (vowel groups)
        vowels = 'aeiouyAEIOUY'
        syllables = 0
        for word in text.split():
            word = re.sub(r'[^a-zA-Z]', '', word)
            if word:
                syllable_count = len(re.findall(r'[aeiouyAEIOUY]+', word))
                if syllable_count == 0:
                    syllable_count = 1
                syllables += syllable_count
        
        if sentences == 0 or words == 0:
            return {'error': 'Insufficient text for analysis'}
        
        # Calculate metrics
        avg_sentence_length = words / sentences
        avg_syllables_per_word = syllables / words
        
        # Flesch Reading Ease (approximation)
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        flesch_score = max(0, min(100, flesch_score))  # Clamp to 0-100
        
        # Interpret Flesch score
        if flesch_score >= 90:
            reading_level = "Very Easy"
        elif flesch_score >= 80:
            reading_level = "Easy"
        elif flesch_score >= 70:
            reading_level = "Fairly Easy"
        elif flesch_score >= 60:
            reading_level = "Standard"
        elif flesch_score >= 50:
            reading_level = "Fairly Difficult"
        elif flesch_score >= 30:
            reading_level = "Difficult"
        else:
            reading_level = "Very Difficult"
        
        return {
            'sentences': sentences,
            'words': words,
            'syllables': syllables,
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_syllables_per_word': round(avg_syllables_per_word, 2),
            'flesch_score': round(flesch_score, 2),
            'reading_level': reading_level
        }
        
    except Exception as e:
        logger.error(f"Error calculating reading level: {e}")
        return {'error': str(e)}

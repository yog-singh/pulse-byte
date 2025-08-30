"""
Base scraper class for news sources.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import time
import requests
from datetime import datetime
from loguru import logger

from src.models import Article, ScrapingResult, SourceType
from config.settings import SCRAPING_CONFIG


class BaseScraper(ABC):
    """Abstract base class for all news scrapers."""
    
    def __init__(self, source_name: str, source_type: SourceType):
        self.source_name = source_name
        self.source_type = source_type
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': SCRAPING_CONFIG['USER_AGENT']
        })
        self.request_delay = SCRAPING_CONFIG['REQUEST_DELAY']
        self.max_retries = SCRAPING_CONFIG['MAX_RETRIES']
        self.timeout = SCRAPING_CONFIG['TIMEOUT']
        
    @abstractmethod
    def scrape_articles(self, keywords: Optional[List[str]] = None, 
                       max_articles: Optional[int] = None) -> ScrapingResult:
        """
        Scrape articles from the source.
        
        Args:
            keywords: Optional list of keywords to filter articles
            max_articles: Maximum number of articles to scrape
            
        Returns:
            ScrapingResult object containing scraped articles and metadata
        """
        pass
    
    def make_request(self, url: str, **kwargs) -> Optional[requests.Response]:
        """
        Make HTTP request with retry logic and rate limiting.
        
        Args:
            url: URL to request
            **kwargs: Additional arguments for requests
            
        Returns:
            Response object or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                time.sleep(self.request_delay)
                
                response = self.session.get(
                    url, 
                    timeout=self.timeout,
                    **kwargs
                )
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed for {url}: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"All attempts failed for {url}")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        # Remove common unwanted patterns
        unwanted_patterns = [
            'Advertisement',
            'Click here',
            'Read more',
            'Subscribe',
            'Sign up',
            'Continue reading'
        ]
        
        for pattern in unwanted_patterns:
            text = text.replace(pattern, '')
        
        return text.strip()
    
    def extract_keywords_from_text(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text content.
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        # This is a simple implementation - you might want to use more sophisticated
        # NLP techniques like TF-IDF, named entity recognition, etc.
        
        import re
        from collections import Counter
        
        # Simple keyword extraction based on word frequency
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        # Extract words (3+ characters, alphabetic)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out stop words and count frequency
        filtered_words = [word for word in words if word not in stop_words]
        word_counts = Counter(filtered_words)
        
        # Return most common words as keywords
        return [word for word, _ in word_counts.most_common(max_keywords)]
    
    def filter_articles_by_keywords(self, articles: List[Article], 
                                  keywords: List[str]) -> List[Article]:
        """
        Filter articles based on keywords.
        
        Args:
            articles: List of articles to filter
            keywords: Keywords to filter by
            
        Returns:
            Filtered list of articles
        """
        if not keywords:
            return articles
        
        filtered_articles = []
        for article in articles:
            if article.has_keywords(keywords):
                filtered_articles.append(article)
        
        return filtered_articles
    
    def create_scraping_result(self, articles: List[Article], 
                             success: bool = True,
                             error_message: Optional[str] = None,
                             scraping_time: float = 0.0) -> ScrapingResult:
        """
        Create a ScrapingResult object.
        
        Args:
            articles: List of scraped articles
            success: Whether scraping was successful
            error_message: Error message if scraping failed
            scraping_time: Time taken for scraping
            
        Returns:
            ScrapingResult object
        """
        return ScrapingResult(
            source=self.source_name,
            source_type=self.source_type,
            articles=articles,
            success=success,
            error_message=error_message,
            scraping_time=scraping_time,
            timestamp=datetime.now()
        )

"""
Base sentiment analyzer class.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from loguru import logger

from src.models import Article, SentimentScore, SentimentLabel


class SentimentAnalyzer(ABC):
    """Abstract base class for sentiment analyzers."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_initialized = False
        
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the sentiment analysis model.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def analyze_text(self, text: str) -> SentimentScore:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentScore object with results
        """
        pass
    
    def analyze_article(self, article: Article) -> Article:
        """
        Analyze sentiment of an article and update it.
        
        Args:
            article: Article to analyze
            
        Returns:
            Article with updated sentiment score
        """
        if not self.is_initialized:
            if not self.initialize():
                logger.error(f"Failed to initialize {self.model_name} analyzer")
                return article
        
        try:
            # Combine title and content for analysis
            text = f"{article.title}. {article.content}"
            sentiment = self.analyze_text(text)
            article.sentiment = sentiment
            
            logger.debug(f"Analyzed article '{article.title[:50]}...' with {self.model_name}: {sentiment.label.value}")
            return article
            
        except Exception as e:
            logger.error(f"Error analyzing article with {self.model_name}: {e}")
            return article
    
    def analyze_articles(self, articles: List[Article]) -> List[Article]:
        """
        Analyze sentiment of multiple articles.
        
        Args:
            articles: List of articles to analyze
            
        Returns:
            List of articles with updated sentiment scores
        """
        if not self.is_initialized:
            if not self.initialize():
                logger.error(f"Failed to initialize {self.model_name} analyzer")
                return articles
        
        analyzed_articles = []
        successful_analyses = 0
        
        for i, article in enumerate(articles):
            try:
                analyzed_article = self.analyze_article(article)
                analyzed_articles.append(analyzed_article)
                
                if analyzed_article.sentiment is not None:
                    successful_analyses += 1
                    
                # Log progress for large batches
                if (i + 1) % 10 == 0:
                    logger.info(f"Analyzed {i + 1}/{len(articles)} articles with {self.model_name}")
                    
            except Exception as e:
                logger.error(f"Error analyzing article {i}: {e}")
                analyzed_articles.append(article)  # Add original article
        
        logger.info(f"Completed analysis with {self.model_name}: {successful_analyses}/{len(articles)} successful")
        return analyzed_articles
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the analyzer model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'name': self.model_name,
            'initialized': self.is_initialized,
            'type': self.__class__.__name__
        }
    
    @staticmethod
    def normalize_sentiment_label(score: float, 
                                 positive_threshold: float = 0.1,
                                 negative_threshold: float = -0.1) -> SentimentLabel:
        """
        Normalize a sentiment score to a label.
        
        Args:
            score: Sentiment score (typically between -1 and 1)
            positive_threshold: Threshold for positive sentiment
            negative_threshold: Threshold for negative sentiment
            
        Returns:
            SentimentLabel enum value
        """
        if score > positive_threshold:
            return SentimentLabel.POSITIVE
        elif score < negative_threshold:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.NEUTRAL
    
    @staticmethod
    def calculate_confidence(scores: Dict[str, float]) -> float:
        """
        Calculate confidence based on score distribution.
        
        Args:
            scores: Dictionary of sentiment scores
            
        Returns:
            Confidence value between 0 and 1
        """
        if not scores:
            return 0.0
        
        # Find the maximum score
        max_score = max(scores.values())
        
        # Calculate confidence as the difference between max and second max
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            confidence = max_score - sorted_scores[1]
        else:
            confidence = max_score
        
        # Normalize to 0-1 range
        return min(1.0, max(0.0, confidence))
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Basic text cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Limit text length to avoid memory issues (keep first 1000 words)
        words = text.split()
        if len(words) > 1000:
            text = ' '.join(words[:1000])
            logger.debug(f"Truncated text to 1000 words for {self.model_name} analysis")
        
        return text
    
    def validate_score(self, score: SentimentScore) -> bool:
        """
        Validate sentiment score object.
        
        Args:
            score: SentimentScore to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(score, SentimentScore):
            return False
        
        if not isinstance(score.label, SentimentLabel):
            return False
        
        if not (0.0 <= score.confidence <= 1.0):
            return False
        
        if not isinstance(score.scores, dict):
            return False
        
        return True

"""
Data models for news articles and sentiment analysis results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum
import json


class SentimentLabel(Enum):
    """Sentiment classification labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SourceType(Enum):
    """Types of news sources."""
    RSS = "rss"
    API = "api"
    WEBSITE = "website"


@dataclass
class SentimentScore:
    """Sentiment analysis result."""
    label: SentimentLabel
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)
    model_used: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'label': self.label.value,
            'confidence': self.confidence,
            'scores': self.scores,
            'model_used': self.model_used
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SentimentScore':
        return cls(
            label=SentimentLabel(data['label']),
            confidence=data['confidence'],
            scores=data.get('scores', {}),
            model_used=data.get('model_used', '')
        )


@dataclass
class Article:
    """News article data model."""
    title: str
    content: str
    url: str
    source: str
    published_date: datetime
    scraped_date: datetime = field(default_factory=datetime.now)
    author: Optional[str] = None
    summary: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    category: Optional[str] = None
    source_type: SourceType = SourceType.WEBSITE
    sentiment: Optional[SentimentScore] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and process data after initialization."""
        if isinstance(self.published_date, str):
            # Try to parse string dates
            try:
                self.published_date = datetime.fromisoformat(self.published_date.replace('Z', '+00:00'))
            except ValueError:
                self.published_date = datetime.now()
        
        if isinstance(self.scraped_date, str):
            try:
                self.scraped_date = datetime.fromisoformat(self.scraped_date.replace('Z', '+00:00'))
            except ValueError:
                self.scraped_date = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert article to dictionary format."""
        return {
            'title': self.title,
            'content': self.content,
            'url': self.url,
            'source': self.source,
            'published_date': self.published_date.isoformat(),
            'scraped_date': self.scraped_date.isoformat(),
            'author': self.author,
            'summary': self.summary,
            'keywords': self.keywords,
            'category': self.category,
            'source_type': self.source_type.value,
            'sentiment': self.sentiment.to_dict() if self.sentiment else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Article':
        """Create article from dictionary."""
        sentiment_data = data.get('sentiment')
        sentiment = SentimentScore.from_dict(sentiment_data) if sentiment_data else None
        
        return cls(
            title=data['title'],
            content=data['content'],
            url=data['url'],
            source=data['source'],
            published_date=data['published_date'],
            scraped_date=data.get('scraped_date', datetime.now()),
            author=data.get('author'),
            summary=data.get('summary'),
            keywords=data.get('keywords', []),
            category=data.get('category'),
            source_type=SourceType(data.get('source_type', 'website')),
            sentiment=sentiment,
            metadata=data.get('metadata', {})
        )
    
    def to_json(self) -> str:
        """Convert article to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Article':
        """Create article from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_word_count(self) -> int:
        """Get the word count of the article content."""
        return len(self.content.split())
    
    def get_reading_time(self) -> int:
        """Estimate reading time in minutes (assuming 200 words per minute)."""
        return max(1, self.get_word_count() // 200)
    
    def has_keywords(self, target_keywords: List[str]) -> bool:
        """Check if article contains any of the target keywords."""
        text = (self.title + " " + self.content).lower()
        return any(keyword.lower() in text for keyword in target_keywords)


@dataclass
class ScrapingResult:
    """Result of a scraping operation."""
    source: str
    source_type: SourceType
    articles: List[Article]
    success: bool
    error_message: Optional[str] = None
    scraping_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'source_type': self.source_type.value,
            'articles': [article.to_dict() for article in self.articles],
            'success': self.success,
            'error_message': self.error_message,
            'scraping_time': self.scraping_time,
            'timestamp': self.timestamp.isoformat(),
            'article_count': len(self.articles)
        }


@dataclass
class AnalysisResult:
    """Result of sentiment analysis on multiple articles."""
    articles_processed: int
    sentiment_distribution: Dict[str, int]
    average_confidence: float
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'articles_processed': self.articles_processed,
            'sentiment_distribution': self.sentiment_distribution,
            'average_confidence': self.average_confidence,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp.isoformat()
        }

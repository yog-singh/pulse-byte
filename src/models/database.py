"""
Database models and operations for PulseByte.
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, JSON, Boolean, Index, text
try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.engine import Engine
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import json
import uuid
from loguru import logger

from config.settings import DATABASE_CONFIG, SQLITE_URL
from .article import Article, SentimentScore, SentimentLabel, SourceType

Base = declarative_base()


class ArticleDB(Base):
    """SQLAlchemy model for articles optimized for PostgreSQL."""
    __tablename__ = 'articles'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(1000), nullable=False)  # Increased for longer titles
    content = Column(Text, nullable=False)
    url = Column(String(2000), unique=True, nullable=False, index=True)  # Indexed for faster lookups
    source = Column(String(200), nullable=False, index=True)  # Indexed for filtering
    published_date = Column(DateTime(timezone=True), nullable=False, index=True)  # Timezone-aware, indexed
    scraped_date = Column(DateTime(timezone=True), default=datetime.now, index=True)  # Timezone-aware, indexed
    author = Column(String(300), nullable=True)
    summary = Column(Text, nullable=True)
    
    # Use JSONB for PostgreSQL (better performance), fallback to JSON for SQLite
    keywords = Column(JSONB, nullable=True)  # PostgreSQL JSONB for better performance
    category = Column(String(100), nullable=True, index=True)  # Indexed for filtering
    source_type = Column(String(50), nullable=False, index=True)  # Indexed for filtering
    
    # Sentiment analysis fields
    sentiment_label = Column(String(20), nullable=True, index=True)  # Indexed for sentiment filtering
    sentiment_confidence = Column(Float, nullable=True)
    sentiment_scores = Column(JSONB, nullable=True)  # PostgreSQL JSONB
    sentiment_model = Column(String(100), nullable=True)
    
    # Additional metadata (renamed to avoid SQLAlchemy conflict)
    article_metadata = Column(JSONB, nullable=True)  # PostgreSQL JSONB
    word_count = Column(Integer, nullable=True)  # Cache word count for performance
    reading_time = Column(Integer, nullable=True)  # Cache reading time estimate
    
    # Timestamps for tracking
    created_at = Column(DateTime(timezone=True), default=datetime.now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.now, onupdate=datetime.now, nullable=False)
    
    # Create composite indexes for common queries
    __table_args__ = (
        Index('idx_articles_source_date', 'source', 'published_date'),
        Index('idx_articles_sentiment_date', 'sentiment_label', 'published_date'),
        Index('idx_articles_category_date', 'category', 'published_date'),
        Index('idx_articles_scraped_date', 'scraped_date'),
        # GIN index for JSONB columns (PostgreSQL specific, will be ignored in SQLite)
        Index('idx_articles_keywords_gin', 'keywords', postgresql_using='gin'),
        Index('idx_articles_metadata_gin', 'article_metadata', postgresql_using='gin'),
    )
    
    def to_article(self) -> Article:
        """Convert database record to Article model."""
        sentiment = None
        if self.sentiment_label:
            sentiment = SentimentScore(
                label=SentimentLabel(self.sentiment_label),
                confidence=self.sentiment_confidence or 0.0,
                scores=self.sentiment_scores or {},
                model_used=self.sentiment_model or ""
            )
        
        return Article(
            title=self.title,
            content=self.content,
            url=self.url,
            source=self.source,
            published_date=self.published_date,
            scraped_date=self.scraped_date,
            author=self.author,
            summary=self.summary,
            keywords=self.keywords or [],
            category=self.category,
            source_type=SourceType(self.source_type),
            sentiment=sentiment,
            metadata=self.article_metadata or {}
        )
    
    @classmethod
    def from_article(cls, article: Article) -> 'ArticleDB':
        """Create database record from Article model."""
        # Calculate cached fields
        word_count = article.get_word_count()
        reading_time = article.get_reading_time()
        
        return cls(
            title=article.title,
            content=article.content,
            url=article.url,
            source=article.source,
            published_date=article.published_date,
            scraped_date=article.scraped_date,
            author=article.author,
            summary=article.summary,
            keywords=article.keywords,
            category=article.category,
            source_type=article.source_type.value,
            sentiment_label=article.sentiment.label.value if article.sentiment else None,
            sentiment_confidence=article.sentiment.confidence if article.sentiment else None,
            sentiment_scores=article.sentiment.scores if article.sentiment else None,
            sentiment_model=article.sentiment.model_used if article.sentiment else None,
            article_metadata=article.metadata,
            word_count=word_count,
            reading_time=reading_time
        )


class DatabaseManager:
    """Database operations manager with PostgreSQL support and SQLite fallback."""
    
    def __init__(self, database_config: Dict[str, Any] = None):
        """
        Initialize database manager.
        
        Args:
            database_config: Database configuration dict. Uses DATABASE_CONFIG if None.
        """
        self.config = database_config or DATABASE_CONFIG
        self.engine = None
        self.SessionLocal = None
        self.database_type = None
        self._initialize_database()
        self.create_tables()
    
    def _initialize_database(self):
        """Initialize database connection with fallback logic."""
        try:
            # Try PostgreSQL first
            self._try_postgresql()
        except Exception as pg_error:
            logger.warning(f"PostgreSQL connection failed: {pg_error}")
            logger.info("Falling back to SQLite...")
            try:
                self._try_sqlite()
            except Exception as sqlite_error:
                logger.error(f"SQLite connection also failed: {sqlite_error}")
                raise RuntimeError("Could not connect to any database")
    
    def _try_postgresql(self):
        """Try to connect to PostgreSQL."""
        # Create engine with connection pooling for PostgreSQL
        engine_args = {
            'pool_size': self.config.get('pool_size', 10),
            'max_overflow': self.config.get('max_overflow', 20),
            'pool_timeout': self.config.get('pool_timeout', 30),
            'pool_recycle': self.config.get('pool_recycle', 3600),
            'echo': self.config.get('echo', False),
        }
        
        self.engine = create_engine(self.config['url'], **engine_args)
        
        # Test connection
        with self.engine.connect() as conn:
            conn.execute(text('SELECT 1'))
        
        self.database_type = 'postgresql'
        logger.info("Connected to PostgreSQL database")
        
    def _try_sqlite(self):
        """Try to connect to SQLite as fallback."""
        sqlite_url = self.config.get('sqlite_fallback', SQLITE_URL)
        
        # For SQLite, we need to handle JSONB -> JSON conversion
        self._convert_jsonb_to_json()
        
        self.engine = create_engine(
            sqlite_url,
            echo=self.config.get('echo', False),
            connect_args={'check_same_thread': False}  # SQLite specific
        )
        
        # Test connection
        with self.engine.connect() as conn:
            conn.execute(text('SELECT 1'))
        
        self.database_type = 'sqlite'
        logger.info("Connected to SQLite database (fallback)")
    
    def _convert_jsonb_to_json(self):
        """Convert JSONB columns to JSON for SQLite compatibility."""
        # Temporarily replace JSONB with JSON for SQLite
        for column in ArticleDB.__table__.columns:
            if hasattr(column.type, 'impl') and column.type.impl.__class__.__name__ == 'JSONB':
                column.type = JSON
    
    def _initialize_session(self):
        """Initialize session maker."""
        if not self.SessionLocal:
            self.SessionLocal = sessionmaker(
                autocommit=False, 
                autoflush=False, 
                bind=self.engine
            )
    
    def create_tables(self):
        """Create all database tables."""
        if not self.engine:
            raise RuntimeError("Database not initialized")
        
        self._initialize_session()
        
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info(f"Database tables created successfully ({self.database_type})")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database connection information."""
        return {
            'type': self.database_type,
            'url': str(self.engine.url).replace(self.engine.url.password or '', '***') if self.engine else None,
            'pool_size': getattr(self.engine.pool, 'size', None) if self.engine else None,
            'checked_out': getattr(self.engine.pool, 'checkedout', None) if self.engine else None,
        }
    
    def get_session(self) -> Session:
        """Get a database session."""
        if not self.SessionLocal:
            self._initialize_session()
        return self.SessionLocal()
    
    def save_article(self, article: Article) -> bool:
        """Save an article to the database."""
        session = self.get_session()
        try:
            # Check if article already exists
            existing = session.query(ArticleDB).filter_by(url=article.url).first()
            if existing:
                # Update existing article
                for key, value in ArticleDB.from_article(article).__dict__.items():
                    if not key.startswith('_') and key != 'id':
                        setattr(existing, key, value)
            else:
                # Create new article
                db_article = ArticleDB.from_article(article)
                session.add(db_article)
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving article: {e}")
            return False
        finally:
            session.close()
    
    def save_articles(self, articles: List[Article]) -> int:
        """Save multiple articles to the database. Returns count of saved articles."""
        saved_count = 0
        for article in articles:
            if self.save_article(article):
                saved_count += 1
        return saved_count
    
    def get_articles(self, 
                    limit: Optional[int] = None,
                    keywords: Optional[List[str]] = None,
                    sentiment: Optional[SentimentLabel] = None,
                    source: Optional[str] = None,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> List[Article]:
        """Retrieve articles with optional filtering."""
        session = self.get_session()
        try:
            query = session.query(ArticleDB)
            
            # Apply filters
            if keywords:
                # Filter by keywords (this is a simple implementation)
                keyword_filter = None
                for keyword in keywords:
                    condition = ArticleDB.content.contains(keyword) | ArticleDB.title.contains(keyword)
                    keyword_filter = condition if keyword_filter is None else keyword_filter | condition
                if keyword_filter is not None:
                    query = query.filter(keyword_filter)
            
            if sentiment:
                query = query.filter(ArticleDB.sentiment_label == sentiment.value)
            
            if source:
                query = query.filter(ArticleDB.source == source)
            
            if start_date:
                query = query.filter(ArticleDB.published_date >= start_date)
            
            if end_date:
                query = query.filter(ArticleDB.published_date <= end_date)
            
            # Order by published date (most recent first)
            query = query.order_by(ArticleDB.published_date.desc())
            
            if limit:
                query = query.limit(limit)
            
            db_articles = query.all()
            return [db_article.to_article() for db_article in db_articles]
        
        finally:
            session.close()
    
    def get_article_by_url(self, url: str) -> Optional[Article]:
        """Get a specific article by URL."""
        session = self.get_session()
        try:
            db_article = session.query(ArticleDB).filter_by(url=url).first()
            return db_article.to_article() if db_article else None
        finally:
            session.close()
    
    def get_sentiment_stats(self) -> Dict[str, Any]:
        """Get sentiment analysis statistics."""
        session = self.get_session()
        try:
            total_articles = session.query(ArticleDB).count()
            analyzed_articles = session.query(ArticleDB).filter(ArticleDB.sentiment_label.isnot(None)).count()
            
            sentiment_counts = {}
            for label in SentimentLabel:
                count = session.query(ArticleDB).filter(ArticleDB.sentiment_label == label.value).count()
                sentiment_counts[label.value] = count
            
            return {
                'total_articles': total_articles,
                'analyzed_articles': analyzed_articles,
                'sentiment_distribution': sentiment_counts
            }
        finally:
            session.close()
    
    def delete_old_articles(self, days: int = 30) -> int:
        """Delete articles older than specified days. Returns count of deleted articles."""
        session = self.get_session()
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            deleted_count = session.query(ArticleDB).filter(
                ArticleDB.scraped_date < cutoff_date
            ).delete()
            session.commit()
            return deleted_count
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting old articles: {e}")
            return 0
        finally:
            session.close()


# Global database manager instance
db_manager = DatabaseManager()

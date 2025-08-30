"""
RSS feed scraper for news articles.
"""

import feedparser
from typing import List, Optional
from datetime import datetime, timezone
import time
from loguru import logger

from src.models import Article, ScrapingResult, SourceType
from .base_scraper import BaseScraper
from config.settings import SCRAPING_CONFIG


class RSScraper(BaseScraper):
    """RSS feed scraper implementation."""
    
    def __init__(self, rss_url: str):
        source_name = self._extract_source_name(rss_url)
        super().__init__(source_name, SourceType.RSS)
        self.rss_url = rss_url
        
    def _extract_source_name(self, url: str) -> str:
        """Extract source name from RSS URL."""
        # Simple extraction based on domain
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove common prefixes
        domain = domain.replace('www.', '').replace('feeds.', '').replace('rss.', '')
        
        # Extract main domain name
        parts = domain.split('.')
        if len(parts) >= 2:
            return parts[0].capitalize()
        return domain.capitalize()
    
    def scrape_articles(self, keywords: Optional[List[str]] = None, 
                       max_articles: Optional[int] = None) -> ScrapingResult:
        """
        Scrape articles from RSS feed.
        
        Args:
            keywords: Optional keywords to filter articles
            max_articles: Maximum number of articles to scrape
            
        Returns:
            ScrapingResult containing scraped articles
        """
        start_time = time.time()
        articles = []
        
        try:
            logger.info(f"Scraping RSS feed: {self.rss_url}")
            
            # Parse RSS feed
            feed = feedparser.parse(self.rss_url)
            
            if feed.bozo:
                logger.warning(f"RSS feed has parsing issues: {self.rss_url}")
            
            # Extract articles from feed entries
            for entry in feed.entries:
                try:
                    article = self._parse_rss_entry(entry)
                    if article:
                        articles.append(article)
                        
                        # Check max articles limit
                        if max_articles and len(articles) >= max_articles:
                            break
                            
                except Exception as e:
                    logger.warning(f"Error parsing RSS entry: {e}")
                    continue
            
            # Filter by keywords if provided
            if keywords:
                articles = self.filter_articles_by_keywords(articles, keywords)
            
            scraping_time = time.time() - start_time
            logger.info(f"Successfully scraped {len(articles)} articles from {self.source_name}")
            
            return self.create_scraping_result(
                articles=articles,
                success=True,
                scraping_time=scraping_time
            )
            
        except Exception as e:
            scraping_time = time.time() - start_time
            error_msg = f"Failed to scrape RSS feed {self.rss_url}: {str(e)}"
            logger.error(error_msg)
            
            return self.create_scraping_result(
                articles=[],
                success=False,
                error_message=error_msg,
                scraping_time=scraping_time
            )
    
    def _parse_rss_entry(self, entry) -> Optional[Article]:
        """
        Parse a single RSS entry into an Article object.
        
        Args:
            entry: RSS feed entry
            
        Returns:
            Article object or None if parsing fails
        """
        try:
            # Extract basic information
            title = self.clean_text(getattr(entry, 'title', ''))
            url = getattr(entry, 'link', '')
            
            if not title or not url:
                return None
            
            # Extract content
            content = self._extract_content_from_entry(entry)
            
            # Extract publication date
            published_date = self._parse_date(entry)
            
            # Extract author
            author = self._extract_author(entry)
            
            # Extract summary/description
            summary = self.clean_text(getattr(entry, 'summary', ''))
            
            # Extract category/tags
            category, keywords = self._extract_categories_and_keywords(entry)
            
            # Create Article object
            article = Article(
                title=title,
                content=content,
                url=url,
                source=self.source_name,
                published_date=published_date,
                author=author,
                summary=summary,
                keywords=keywords,
                category=category,
                source_type=SourceType.RSS,
                metadata={
                    'rss_url': self.rss_url,
                    'entry_id': getattr(entry, 'id', ''),
                    'guid': getattr(entry, 'guid', '')
                }
            )
            
            return article
            
        except Exception as e:
            logger.warning(f"Error parsing RSS entry: {e}")
            return None
    
    def _extract_content_from_entry(self, entry) -> str:
        """Extract content from RSS entry."""
        # Try different content fields in order of preference
        content_fields = ['content', 'description', 'summary']
        
        for field in content_fields:
            if hasattr(entry, field):
                value = getattr(entry, field)
                if isinstance(value, list) and len(value) > 0:
                    # Some RSS feeds have content as list
                    content = value[0].get('value', '') if isinstance(value[0], dict) else str(value[0])
                else:
                    content = str(value) if value else ''
                
                if content:
                    return self.clean_text(content)
        
        return ""
    
    def _parse_date(self, entry) -> datetime:
        """Parse publication date from RSS entry."""
        date_fields = ['published_parsed', 'updated_parsed']
        
        for field in date_fields:
            if hasattr(entry, field) and getattr(entry, field):
                time_struct = getattr(entry, field)
                try:
                    return datetime(*time_struct[:6], tzinfo=timezone.utc)
                except (TypeError, ValueError):
                    continue
        
        # Try string date fields
        string_date_fields = ['published', 'updated']
        for field in string_date_fields:
            if hasattr(entry, field) and getattr(entry, field):
                try:
                    from dateutil import parser
                    return parser.parse(getattr(entry, field))
                except:
                    continue
        
        # Fallback to current time
        return datetime.now(timezone.utc)
    
    def _extract_author(self, entry) -> Optional[str]:
        """Extract author from RSS entry."""
        author_fields = ['author', 'author_detail']
        
        for field in author_fields:
            if hasattr(entry, field):
                value = getattr(entry, field)
                if isinstance(value, dict):
                    author = value.get('name', '')
                else:
                    author = str(value) if value else ''
                
                if author:
                    return self.clean_text(author)
        
        return None
    
    def _extract_categories_and_keywords(self, entry) -> tuple[Optional[str], List[str]]:
        """Extract category and keywords from RSS entry."""
        category = None
        keywords = []
        
        # Extract category
        if hasattr(entry, 'tags') and entry.tags:
            # Use first tag as category
            if len(entry.tags) > 0:
                first_tag = entry.tags[0]
                if isinstance(first_tag, dict):
                    category = first_tag.get('term', '')
                else:
                    category = str(first_tag)
            
            # Use all tags as keywords
            for tag in entry.tags:
                if isinstance(tag, dict):
                    term = tag.get('term', '')
                else:
                    term = str(tag)
                
                if term and term not in keywords:
                    keywords.append(term.lower())
        
        # Extract keywords from content if no tags available
        if not keywords:
            content = self._extract_content_from_entry(entry)
            if content:
                keywords = self.extract_keywords_from_text(content, max_keywords=5)
        
        return category, keywords


class MultiRSScraper:
    """Scraper for multiple RSS feeds."""
    
    def __init__(self, rss_urls: List[str]):
        self.scrapers = [RSScraper(url) for url in rss_urls]
    
    def scrape_all(self, keywords: Optional[List[str]] = None,
                   max_articles_per_feed: Optional[int] = None) -> List[ScrapingResult]:
        """
        Scrape articles from all RSS feeds.
        
        Args:
            keywords: Optional keywords to filter articles
            max_articles_per_feed: Max articles per individual feed
            
        Returns:
            List of ScrapingResult objects
        """
        results = []
        
        for scraper in self.scrapers:
            try:
                result = scraper.scrape_articles(keywords, max_articles_per_feed)
                results.append(result)
            except Exception as e:
                logger.error(f"Error scraping {scraper.rss_url}: {e}")
                results.append(scraper.create_scraping_result(
                    articles=[],
                    success=False,
                    error_message=str(e)
                ))
        
        return results

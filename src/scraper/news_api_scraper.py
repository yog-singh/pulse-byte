"""
News API scraper for fetching articles from newsapi.org.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import time
from loguru import logger
from newsapi import NewsApiClient

from src.models import Article, ScrapingResult, SourceType
from .base_scraper import BaseScraper
from config.settings import NEWS_API_KEY


class NewsAPIScraper(BaseScraper):
    """News API scraper implementation."""
    
    def __init__(self, api_key: str = NEWS_API_KEY):
        super().__init__("NewsAPI", SourceType.API)
        if not api_key:
            raise ValueError("News API key is required")
        self.client = NewsApiClient(api_key=api_key)
        
    def scrape_articles(self, keywords: Optional[List[str]] = None, 
                       max_articles: Optional[int] = None,
                       sources: Optional[List[str]] = None,
                       language: str = 'en',
                       sort_by: str = 'publishedAt') -> ScrapingResult:
        """
        Scrape articles from News API.
        
        Args:
            keywords: Keywords to search for
            max_articles: Maximum number of articles to fetch
            sources: Specific news sources to search
            language: Language code (default: 'en')
            sort_by: Sort articles by ('relevancy', 'popularity', 'publishedAt')
            
        Returns:
            ScrapingResult containing scraped articles
        """
        start_time = time.time()
        articles = []
        
        try:
            logger.info(f"Scraping articles from News API with keywords: {keywords}")
            
            # Prepare search query
            query = ' OR '.join(keywords) if keywords else None
            
            # Set page size (max 100 for News API)
            page_size = min(100, max_articles) if max_articles else 100
            
            # Fetch articles using News API
            response = self._fetch_everything(
                q=query,
                sources=','.join(sources) if sources else None,
                language=language,
                sort_by=sort_by,
                page_size=page_size
            )
            
            if response['status'] == 'ok':
                # Parse articles from response
                for article_data in response['articles']:
                    try:
                        article = self._parse_api_article(article_data)
                        if article:
                            articles.append(article)
                            
                            # Check max articles limit
                            if max_articles and len(articles) >= max_articles:
                                break
                                
                    except Exception as e:
                        logger.warning(f"Error parsing News API article: {e}")
                        continue
                
                scraping_time = time.time() - start_time
                logger.info(f"Successfully scraped {len(articles)} articles from News API")
                
                return self.create_scraping_result(
                    articles=articles,
                    success=True,
                    scraping_time=scraping_time
                )
            else:
                error_msg = f"News API error: {response.get('message', 'Unknown error')}"
                logger.error(error_msg)
                
                return self.create_scraping_result(
                    articles=[],
                    success=False,
                    error_message=error_msg,
                    scraping_time=time.time() - start_time
                )
                
        except Exception as e:
            scraping_time = time.time() - start_time
            error_msg = f"Failed to scrape from News API: {str(e)}"
            logger.error(error_msg)
            
            return self.create_scraping_result(
                articles=[],
                success=False,
                error_message=error_msg,
                scraping_time=scraping_time
            )
    
    def _fetch_everything(self, **kwargs) -> Dict[str, Any]:
        """
        Fetch articles using News API everything endpoint.
        
        Args:
            **kwargs: Parameters for News API
            
        Returns:
            API response dictionary
        """
        try:
            return self.client.get_everything(**kwargs)
        except Exception as e:
            logger.error(f"News API request failed: {e}")
            return {'status': 'error', 'message': str(e), 'articles': []}
    
    def get_top_headlines(self, keywords: Optional[List[str]] = None,
                         sources: Optional[List[str]] = None,
                         category: Optional[str] = None,
                         country: str = 'us',
                         max_articles: Optional[int] = None) -> ScrapingResult:
        """
        Get top headlines from News API.
        
        Args:
            keywords: Keywords to search for
            sources: Specific news sources
            category: News category (business, entertainment, general, health, science, sports, technology)
            country: Country code (default: 'us')
            max_articles: Maximum number of articles
            
        Returns:
            ScrapingResult containing top headlines
        """
        start_time = time.time()
        articles = []
        
        try:
            logger.info(f"Fetching top headlines from News API")
            
            # Prepare parameters
            params = {
                'country': country,
                'page_size': min(100, max_articles) if max_articles else 100
            }
            
            if keywords:
                params['q'] = ' OR '.join(keywords)
            if sources:
                params['sources'] = ','.join(sources)
                # Remove country when using sources (News API limitation)
                params.pop('country', None)
            if category:
                params['category'] = category
            
            # Fetch top headlines
            response = self.client.get_top_headlines(**params)
            
            if response['status'] == 'ok':
                # Parse articles from response
                for article_data in response['articles']:
                    try:
                        article = self._parse_api_article(article_data)
                        if article:
                            articles.append(article)
                            
                            if max_articles and len(articles) >= max_articles:
                                break
                                
                    except Exception as e:
                        logger.warning(f"Error parsing top headline: {e}")
                        continue
                
                scraping_time = time.time() - start_time
                logger.info(f"Successfully fetched {len(articles)} top headlines")
                
                return self.create_scraping_result(
                    articles=articles,
                    success=True,
                    scraping_time=scraping_time
                )
            else:
                error_msg = f"News API error: {response.get('message', 'Unknown error')}"
                logger.error(error_msg)
                
                return self.create_scraping_result(
                    articles=[],
                    success=False,
                    error_message=error_msg,
                    scraping_time=time.time() - start_time
                )
                
        except Exception as e:
            scraping_time = time.time() - start_time
            error_msg = f"Failed to fetch top headlines: {str(e)}"
            logger.error(error_msg)
            
            return self.create_scraping_result(
                articles=[],
                success=False,
                error_message=error_msg,
                scraping_time=scraping_time
            )
    
    def _parse_api_article(self, article_data: Dict[str, Any]) -> Optional[Article]:
        """
        Parse News API article data into Article object.
        
        Args:
            article_data: Article data from News API
            
        Returns:
            Article object or None if parsing fails
        """
        try:
            # Extract basic information
            title = self.clean_text(article_data.get('title', ''))
            url = article_data.get('url', '')
            
            if not title or not url:
                return None
            
            # Extract content
            content = self.clean_text(article_data.get('content', ''))
            description = self.clean_text(article_data.get('description', ''))
            
            # Use description as content if content is not available or truncated
            if not content or content.endswith('[+') or content.endswith('...'):
                content = description
            
            # Extract source information
            source_info = article_data.get('source', {})
            source_name = source_info.get('name', 'Unknown')
            
            # Parse publication date
            published_at = article_data.get('publishedAt')
            if published_at:
                try:
                    published_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                except ValueError:
                    published_date = datetime.now(timezone.utc)
            else:
                published_date = datetime.now(timezone.utc)
            
            # Extract author
            author = article_data.get('author')
            
            # Extract keywords from content
            keywords = self.extract_keywords_from_text(content or description, max_keywords=5)
            
            # Create Article object
            article = Article(
                title=title,
                content=content,
                url=url,
                source=source_name,
                published_date=published_date,
                author=author,
                summary=description,
                keywords=keywords,
                source_type=SourceType.API,
                metadata={
                    'api_source': 'newsapi',
                    'source_id': source_info.get('id', ''),
                    'url_to_image': article_data.get('urlToImage', '')
                }
            )
            
            return article
            
        except Exception as e:
            logger.warning(f"Error parsing News API article: {e}")
            return None
    
    def get_sources(self, category: Optional[str] = None,
                   language: str = 'en',
                   country: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get available news sources from News API.
        
        Args:
            category: News category filter
            language: Language filter
            country: Country filter
            
        Returns:
            List of source information dictionaries
        """
        try:
            params = {'language': language}
            if category:
                params['category'] = category
            if country:
                params['country'] = country
                
            response = self.client.get_sources(**params)
            
            if response['status'] == 'ok':
                return response['sources']
            else:
                logger.error(f"Failed to get sources: {response.get('message', 'Unknown error')}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting sources: {e}")
            return []

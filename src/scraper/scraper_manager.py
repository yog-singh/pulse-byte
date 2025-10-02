"""
Scraper manager for coordinating multiple news scrapers.
"""

from typing import List, Optional, Dict, Any
import asyncio
import concurrent.futures
from datetime import datetime
from loguru import logger

from src.models import Article, ScrapingResult, SourceType
from .rss_scraper import RSScraper, MultiRSScraper
from .news_api_scraper import NewsAPIScraper
from .web_scraper import WebScraper
from .gnews_scraper import GNewsScraper
from config.settings import NEWS_SOURCES, SCRAPING_CONFIG, NEWS_API_KEY


class ScraperManager:
    """Manages and coordinates multiple news scrapers."""
    
    def __init__(self):
        self.scrapers = []
        self._initialize_scrapers()
    
    def _initialize_scrapers(self):
        """Initialize all configured scrapers."""
        logger.info("Initializing news scrapers...")
        
        # Initialize RSS scrapers
        if NEWS_SOURCES.get('rss_feeds'):
            try:
                rss_scraper = MultiRSScraper(NEWS_SOURCES['rss_feeds'])
                self.scrapers.extend(rss_scraper.scrapers)
                logger.info(f"Initialized {len(rss_scraper.scrapers)} RSS scrapers")
            except Exception as e:
                logger.error(f"Failed to initialize RSS scrapers: {e}")
        
        # Initialize News API scraper
        if NEWS_API_KEY and 'newsapi' in NEWS_SOURCES.get('news_apis', []):
            try:
                news_api_scraper = NewsAPIScraper(NEWS_API_KEY)
                self.scrapers.append(news_api_scraper)
                logger.info("Initialized News API scraper")
            except Exception as e:
                logger.error(f"Failed to initialize News API scraper: {e}")
        
        # Initialize GNews scraper (no API key required)
        if 'gnews' in NEWS_SOURCES.get('news_apis', []):
            try:
                gnews_scraper = GNewsScraper()
                self.scrapers.append(gnews_scraper)
                logger.info("Initialized GNews scraper")
            except Exception as e:
                logger.error(f"Failed to initialize GNews scraper: {e}")
        
        # Initialize web scrapers
        if NEWS_SOURCES.get('websites'):
            for website in NEWS_SOURCES['websites']:
                try:
                    web_scraper = WebScraper(website)
                    self.scrapers.append(web_scraper)
                except Exception as e:
                    logger.error(f"Failed to initialize web scraper for {website}: {e}")
            
            logger.info(f"Initialized {len(NEWS_SOURCES['websites'])} web scrapers")
        
        logger.info(f"Total scrapers initialized: {len(self.scrapers)}")
    
    def scrape_all_sources(self, 
                          keywords: Optional[List[str]] = None,
                          max_articles_per_source: Optional[int] = None,
                          use_parallel: bool = True,
                          max_workers: int = 5) -> List[ScrapingResult]:
        """
        Scrape articles from all configured sources.
        
        Args:
            keywords: Keywords to filter articles
            max_articles_per_source: Max articles per source
            use_parallel: Whether to scrape sources in parallel
            max_workers: Maximum number of concurrent workers
            
        Returns:
            List of ScrapingResult objects
        """
        if not self.scrapers:
            logger.warning("No scrapers initialized")
            return []
        
        max_articles = max_articles_per_source or SCRAPING_CONFIG.get('MAX_ARTICLES_PER_SOURCE', 50)
        
        logger.info(f"Starting scraping from {len(self.scrapers)} sources...")
        logger.info(f"Keywords: {keywords}")
        logger.info(f"Max articles per source: {max_articles}")
        
        if use_parallel:
            return self._scrape_parallel(keywords, max_articles, max_workers)
        else:
            return self._scrape_sequential(keywords, max_articles)
    
    def _scrape_parallel(self, keywords: Optional[List[str]], 
                        max_articles: int, max_workers: int) -> List[ScrapingResult]:
        """Scrape sources in parallel using ThreadPoolExecutor."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit scraping tasks
            future_to_scraper = {}
            for scraper in self.scrapers:
                future = executor.submit(self._safe_scrape, scraper, keywords, max_articles)
                future_to_scraper[future] = scraper
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_scraper):
                scraper = future_to_scraper[future]
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results.append(result)
                    logger.info(f"Completed scraping {scraper.source_name}: {len(result.articles)} articles")
                except Exception as e:
                    logger.error(f"Error scraping {scraper.source_name}: {e}")
                    # Create failed result
                    failed_result = ScrapingResult(
                        source=scraper.source_name,
                        source_type=scraper.source_type,
                        articles=[],
                        success=False,
                        error_message=str(e)
                    )
                    results.append(failed_result)
        
        return results
    
    def _scrape_sequential(self, keywords: Optional[List[str]], 
                          max_articles: int) -> List[ScrapingResult]:
        """Scrape sources sequentially."""
        results = []
        
        for scraper in self.scrapers:
            try:
                logger.info(f"Scraping {scraper.source_name}...")
                result = self._safe_scrape(scraper, keywords, max_articles)
                results.append(result)
                logger.info(f"Completed {scraper.source_name}: {len(result.articles)} articles")
            except Exception as e:
                logger.error(f"Error scraping {scraper.source_name}: {e}")
                failed_result = ScrapingResult(
                    source=scraper.source_name,
                    source_type=scraper.source_type,
                    articles=[],
                    success=False,
                    error_message=str(e)
                )
                results.append(failed_result)
        
        return results
    
    def _safe_scrape(self, scraper, keywords: Optional[List[str]], 
                    max_articles: int) -> ScrapingResult:
        """Safely scrape from a single source with error handling."""
        try:
            if isinstance(scraper, NewsAPIScraper):
                # News API scraper has different method signature
                return scraper.scrape_articles(
                    keywords=keywords,
                    max_articles=max_articles
                )
            else:
                # RSS and Web scrapers
                return scraper.scrape_articles(
                    keywords=keywords,
                    max_articles=max_articles
                )
        except Exception as e:
            logger.error(f"Error in safe_scrape for {scraper.source_name}: {e}")
            return ScrapingResult(
                source=scraper.source_name,
                source_type=scraper.source_type,
                articles=[],
                success=False,
                error_message=str(e)
            )
    
    def scrape_specific_sources(self, 
                               source_types: List[SourceType],
                               keywords: Optional[List[str]] = None,
                               max_articles_per_source: Optional[int] = None) -> List[ScrapingResult]:
        """
        Scrape from specific source types only.
        
        Args:
            source_types: List of source types to scrape
            keywords: Keywords to filter articles
            max_articles_per_source: Max articles per source
            
        Returns:
            List of ScrapingResult objects
        """
        filtered_scrapers = [
            scraper for scraper in self.scrapers 
            if scraper.source_type in source_types
        ]
        
        if not filtered_scrapers:
            logger.warning(f"No scrapers found for source types: {source_types}")
            return []
        
        logger.info(f"Scraping {len(filtered_scrapers)} sources of types: {source_types}")
        
        # Temporarily replace scrapers list
        original_scrapers = self.scrapers
        self.scrapers = filtered_scrapers
        
        try:
            results = self.scrape_all_sources(keywords, max_articles_per_source)
        finally:
            self.scrapers = original_scrapers
        
        return results
    
    def get_scraper_stats(self) -> Dict[str, Any]:
        """Get statistics about configured scrapers."""
        stats = {
            'total_scrapers': len(self.scrapers),
            'by_type': {},
            'sources': []
        }
        
        for scraper in self.scrapers:
            source_type = scraper.source_type.value
            stats['by_type'][source_type] = stats['by_type'].get(source_type, 0) + 1
            stats['sources'].append({
                'name': scraper.source_name,
                'type': source_type,
                'url': getattr(scraper, 'base_url', getattr(scraper, 'rss_url', 'N/A'))
            })
        
        return stats
    
    def add_custom_scraper(self, scraper):
        """Add a custom scraper to the manager."""
        self.scrapers.append(scraper)
        logger.info(f"Added custom scraper: {scraper.source_name}")
    
    def remove_scraper(self, source_name: str) -> bool:
        """Remove a scraper by source name."""
        original_count = len(self.scrapers)
        self.scrapers = [s for s in self.scrapers if s.source_name != source_name]
        removed = len(self.scrapers) < original_count
        
        if removed:
            logger.info(f"Removed scraper: {source_name}")
        else:
            logger.warning(f"Scraper not found: {source_name}")
        
        return removed
    
    def test_scrapers(self, max_articles: int = 1) -> Dict[str, Any]:
        """
        Test all scrapers with a small number of articles.
        
        Args:
            max_articles: Maximum articles to scrape for testing
            
        Returns:
            Dictionary with test results
        """
        logger.info("Testing all scrapers...")
        
        test_results = {
            'total_scrapers': len(self.scrapers),
            'successful': 0,
            'failed': 0,
            'results': []
        }
        
        for scraper in self.scrapers:
            try:
                logger.info(f"Testing {scraper.source_name}...")
                result = self._safe_scrape(scraper, None, max_articles)
                
                test_info = {
                    'source': scraper.source_name,
                    'type': scraper.source_type.value,
                    'success': result.success,
                    'articles_found': len(result.articles),
                    'error': result.error_message,
                    'scraping_time': result.scraping_time
                }
                
                test_results['results'].append(test_info)
                
                if result.success:
                    test_results['successful'] += 1
                    logger.info(f"✓ {scraper.source_name}: {len(result.articles)} articles")
                else:
                    test_results['failed'] += 1
                    logger.warning(f"✗ {scraper.source_name}: {result.error_message}")
                    
            except Exception as e:
                test_results['failed'] += 1
                test_info = {
                    'source': scraper.source_name,
                    'type': scraper.source_type.value,
                    'success': False,
                    'articles_found': 0,
                    'error': str(e),
                    'scraping_time': 0.0
                }
                test_results['results'].append(test_info)
                logger.error(f"✗ {scraper.source_name}: {e}")
        
        logger.info(f"Test completed: {test_results['successful']}/{test_results['total_scrapers']} scrapers working")
        return test_results

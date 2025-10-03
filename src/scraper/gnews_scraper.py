"""
GNews scraper using the gnews library.
"""

from typing import List, Optional
from datetime import datetime, timezone
import time
from loguru import logger

try:
    from gnews import GNews
except ImportError:
    GNews = None

from src.models import Article, ScrapingResult, SourceType
from .base_scraper import BaseScraper
from config.settings import GNEWS_CONFIG


class GNewsScraper(BaseScraper):
    """Scraper that fetches news using the gnews library (Google News)."""

    def __init__(self):
        super().__init__("GNews", SourceType.API)
        if GNews is None:
            raise ImportError("gnews package not installed. Install with: pip install gnews")

        self.client = GNews(language=GNEWS_CONFIG['language'], country=GNEWS_CONFIG['country'], period=GNEWS_CONFIG['period'])
        self.max_results = GNEWS_CONFIG.get('max_results', 50)

    def scrape_articles(self, keywords: Optional[List[str]] = None,
                       max_articles: Optional[int] = None) -> ScrapingResult:
        start_time = time.time()
        articles: List[Article] = []

        try:
            if not keywords:
                logger.info("GNews requires keywords; using empty search will return trending topics")
            
            query = " ".join(keywords) if keywords else None
            limit = min(self.max_results, max_articles) if max_articles else self.max_results

            logger.info(f"Fetching GNews articles (limit={limit}) for query: {query}")

            results = self.client.get_news(query) if query else self.client.get_top_news()

            for item in results[:limit]:
                try:
                    article = self._parse_gnews_item(item)
                    if article:
                        # Keyword filter if provided
                        if not keywords or article.has_keywords(keywords):
                            articles.append(article)
                except Exception as e:
                    logger.warning(f"Error parsing GNews item: {e}")

            scraping_time = time.time() - start_time
            logger.info(f"GNews scraped {len(articles)} articles")
            return self.create_scraping_result(
                articles=articles,
                success=True,
                scraping_time=scraping_time
            )

        except Exception as e:
            scraping_time = time.time() - start_time
            error_msg = f"Failed to fetch GNews articles: {e}"
            logger.error(error_msg)
            return self.create_scraping_result(
                articles=[],
                success=False,
                error_message=error_msg,
                scraping_time=scraping_time
            )

    def _parse_gnews_item(self, item) -> Optional[Article]:
        title = self.clean_text(item.get('title', ''))
        url = item.get('url', '')
        if not title or not url:
            return None

        # gnews may provide published date as string
        published = item.get('published date') or item.get('published') or item.get('published_at')
        published_date = self._parse_date(published)

        description = self.clean_text(item.get('description', '') or item.get('snippet', ''))
        source = (item.get('publisher') or {}).get('title') or item.get('source') or 'Google News'

        # Content often requires fetching the article page; use description as content
        content = description or title

        keywords = self.extract_keywords_from_text(f"{title} {description}", max_keywords=5)

        return Article(
            title=title,
            content=content,
            url=url,
            source=source,
            published_date=published_date,
            summary=description,
            keywords=keywords,
            source_type=SourceType.API,
            metadata={'api_source': 'gnews'}
        )

    def _parse_date(self, value) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            try:
                from dateutil import parser
                return parser.parse(value)
            except Exception:
                pass
        return datetime.now(timezone.utc)



"""
News scraper module for PulseByte.
"""

from .base_scraper import BaseScraper
from .rss_scraper import RSScraper
from .web_scraper import WebScraper
from .news_api_scraper import NewsAPIScraper
from .gnews_scraper import GNewsScraper
from .scraper_manager import ScraperManager

__all__ = [
    'BaseScraper',
    'RSScraper', 
    'WebScraper',
    'NewsAPIScraper',
    'GNewsScraper',
    'ScraperManager'
]

"""
Web scraper for extracting articles from news websites.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import time
from loguru import logger
from bs4 import BeautifulSoup, Tag
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import re
from urllib.parse import urljoin, urlparse

from src.models import Article, ScrapingResult, SourceType
from .base_scraper import BaseScraper
from config.settings import SELENIUM_CONFIG, SCRAPING_CONFIG


class WebScraper(BaseScraper):
    """Web scraper for news websites using BeautifulSoup and Selenium."""
    
    def __init__(self, base_url: str, site_config: Optional[Dict[str, Any]] = None):
        source_name = self._extract_source_name(base_url)
        super().__init__(source_name, SourceType.WEBSITE)
        self.base_url = base_url
        self.site_config = site_config or {}
        self.driver = None
        
    def _extract_source_name(self, url: str) -> str:
        """Extract source name from URL."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        domain = domain.replace('www.', '')
        
        parts = domain.split('.')
        if len(parts) >= 2:
            return parts[0].capitalize()
        return domain.capitalize()
    
    def _setup_selenium_driver(self) -> webdriver.Chrome:
        """Set up Selenium Chrome driver with appropriate options."""
        chrome_options = Options()
        
        if SELENIUM_CONFIG['headless']:
            chrome_options.add_argument('--headless')
            
        for option in SELENIUM_CONFIG['chrome_options']:
            chrome_options.add_argument(option)
        
        chrome_options.add_argument(f'--window-size={SELENIUM_CONFIG["window_size"][0]},{SELENIUM_CONFIG["window_size"][1]}')
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.implicitly_wait(SELENIUM_CONFIG['implicit_wait'])
            driver.set_page_load_timeout(SELENIUM_CONFIG['page_load_timeout'])
            return driver
        except Exception as e:
            logger.error(f"Failed to setup Selenium driver: {e}")
            raise
    
    def scrape_articles(self, keywords: Optional[List[str]] = None, 
                       max_articles: Optional[int] = None,
                       use_selenium: bool = False) -> ScrapingResult:
        """
        Scrape articles from the website.
        
        Args:
            keywords: Keywords to filter articles
            max_articles: Maximum number of articles to scrape
            use_selenium: Whether to use Selenium for dynamic content
            
        Returns:
            ScrapingResult containing scraped articles
        """
        start_time = time.time()
        articles = []
        
        try:
            logger.info(f"Scraping website: {self.base_url}")
            
            if use_selenium:
                articles = self._scrape_with_selenium(keywords, max_articles)
            else:
                articles = self._scrape_with_requests(keywords, max_articles)
            
            scraping_time = time.time() - start_time
            logger.info(f"Successfully scraped {len(articles)} articles from {self.source_name}")
            
            return self.create_scraping_result(
                articles=articles,
                success=True,
                scraping_time=scraping_time
            )
            
        except Exception as e:
            scraping_time = time.time() - start_time
            error_msg = f"Failed to scrape website {self.base_url}: {str(e)}"
            logger.error(error_msg)
            
            return self.create_scraping_result(
                articles=[],
                success=False,
                error_message=error_msg,
                scraping_time=scraping_time
            )
        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None
    
    def _scrape_with_requests(self, keywords: Optional[List[str]] = None,
                             max_articles: Optional[int] = None) -> List[Article]:
        """Scrape articles using requests and BeautifulSoup."""
        articles = []
        
        # Get the main page
        response = self.make_request(self.base_url)
        if not response:
            return articles
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find article links
        article_links = self._extract_article_links(soup)
        
        # Limit the number of articles
        if max_articles:
            article_links = article_links[:max_articles]
        
        # Scrape each article
        for link in article_links:
            try:
                article = self._scrape_individual_article(link)
                if article:
                    # Filter by keywords if provided
                    if not keywords or article.has_keywords(keywords):
                        articles.append(article)
                        
                        if max_articles and len(articles) >= max_articles:
                            break
                            
            except Exception as e:
                logger.warning(f"Error scraping article {link}: {e}")
                continue
        
        return articles
    
    def _scrape_with_selenium(self, keywords: Optional[List[str]] = None,
                             max_articles: Optional[int] = None) -> List[Article]:
        """Scrape articles using Selenium for dynamic content."""
        articles = []
        
        self.driver = self._setup_selenium_driver()
        
        try:
            # Load the main page
            self.driver.get(self.base_url)
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Get page source and parse with BeautifulSoup
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Find article links
            article_links = self._extract_article_links(soup)
            
            if max_articles:
                article_links = article_links[:max_articles]
            
            # Scrape each article
            for link in article_links:
                try:
                    article = self._scrape_individual_article_selenium(link)
                    if article:
                        if not keywords or article.has_keywords(keywords):
                            articles.append(article)
                            
                            if max_articles and len(articles) >= max_articles:
                                break
                                
                except Exception as e:
                    logger.warning(f"Error scraping article {link}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Selenium scraping error: {e}")
        
        return articles
    
    def _extract_article_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract article links from the main page."""
        links = []
        
        # Common selectors for article links
        selectors = [
            'a[href*="/article/"]',
            'a[href*="/story/"]',
            'a[href*="/news/"]',
            'a[href*="/post/"]',
            '.article-link a',
            '.story-link a',
            '.news-item a',
            'article a',
            '.post-title a'
        ]
        
        # Use site-specific selectors if available
        if 'article_link_selectors' in self.site_config:
            selectors = self.site_config['article_link_selectors'] + selectors
        
        for selector in selectors:
            try:
                elements = soup.select(selector)
                for element in elements:
                    href = element.get('href')
                    if href:
                        # Convert relative URLs to absolute
                        full_url = urljoin(self.base_url, href)
                        if self._is_valid_article_url(full_url):
                            links.append(full_url)
                            
                if links:  # If we found links with this selector, use them
                    break
                    
            except Exception as e:
                logger.warning(f"Error with selector {selector}: {e}")
                continue
        
        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)
        
        return unique_links
    
    def _is_valid_article_url(self, url: str) -> bool:
        """Check if URL is likely to be an article."""
        # Exclude common non-article URLs
        exclude_patterns = [
            r'/tag/', r'/category/', r'/author/', r'/page/',
            r'/search/', r'/archive/', r'/contact/', r'/about/',
            r'\.pdf$', r'\.jpg$', r'\.png$', r'\.gif$',
            r'#', r'mailto:', r'javascript:'
        ]
        
        for pattern in exclude_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False
        
        return True
    
    def _scrape_individual_article(self, url: str) -> Optional[Article]:
        """Scrape an individual article using requests."""
        response = self.make_request(url)
        if not response:
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        return self._parse_article_from_soup(soup, url)
    
    def _scrape_individual_article_selenium(self, url: str) -> Optional[Article]:
        """Scrape an individual article using Selenium."""
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            return self._parse_article_from_soup(soup, url)
            
        except (TimeoutException, WebDriverException) as e:
            logger.warning(f"Selenium error for {url}: {e}")
            return None
    
    def _parse_article_from_soup(self, soup: BeautifulSoup, url: str) -> Optional[Article]:
        """Parse article content from BeautifulSoup object."""
        try:
            # Extract title
            title = self._extract_title(soup)
            if not title:
                return None
            
            # Extract content
            content = self._extract_content(soup)
            if not content:
                return None
            
            # Extract other metadata
            author = self._extract_author(soup)
            published_date = self._extract_publish_date(soup)
            summary = self._extract_summary(soup)
            
            # Extract keywords from content
            keywords = self.extract_keywords_from_text(content, max_keywords=8)
            
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
                source_type=SourceType.WEBSITE,
                metadata={
                    'scraped_with': 'selenium' if self.driver else 'requests'
                }
            )
            
            return article
            
        except Exception as e:
            logger.warning(f"Error parsing article from {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title."""
        selectors = [
            'h1.article-title',
            'h1.entry-title', 
            'h1.post-title',
            '.article-headline h1',
            '.story-title h1',
            'h1',
            'title'
        ]
        
        if 'title_selectors' in self.site_config:
            selectors = self.site_config['title_selectors'] + selectors
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                title = self.clean_text(element.get_text())
                if title and len(title) > 10:  # Reasonable title length
                    return title
        
        return ""
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract article content."""
        selectors = [
            '.article-content',
            '.entry-content',
            '.post-content',
            '.story-body',
            '.article-body',
            'article',
            '.content'
        ]
        
        if 'content_selectors' in self.site_config:
            selectors = self.site_config['content_selectors'] + selectors
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                # Remove unwanted elements
                for unwanted in element.select('script, style, .advertisement, .ad, .social-share'):
                    unwanted.decompose()
                
                content = self.clean_text(element.get_text())
                if content and len(content) > 100:  # Reasonable content length
                    return content
        
        return ""
    
    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract article author."""
        selectors = [
            '.author',
            '.byline',
            '.article-author',
            '.post-author',
            '[rel="author"]',
            '.writer'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                author = self.clean_text(element.get_text())
                if author and len(author) < 100:  # Reasonable author name length
                    return author
        
        return None
    
    def _extract_publish_date(self, soup: BeautifulSoup) -> datetime:
        """Extract article publish date."""
        selectors = [
            'time[datetime]',
            '.publish-date',
            '.article-date',
            '.post-date',
            '[pubdate]'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                # Try datetime attribute first
                datetime_attr = element.get('datetime') or element.get('pubdate')
                if datetime_attr:
                    try:
                        from dateutil import parser
                        return parser.parse(datetime_attr)
                    except:
                        continue
                
                # Try text content
                date_text = self.clean_text(element.get_text())
                if date_text:
                    try:
                        from dateutil import parser
                        return parser.parse(date_text)
                    except:
                        continue
        
        # Fallback to current time
        return datetime.now(timezone.utc)
    
    def _extract_summary(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract article summary/description."""
        # Try meta description first
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            return self.clean_text(meta_desc['content'])
        
        # Try other meta tags
        meta_tags = [
            {'property': 'og:description'},
            {'name': 'twitter:description'},
            {'name': 'summary'}
        ]
        
        for attrs in meta_tags:
            meta = soup.find('meta', attrs=attrs)
            if meta and meta.get('content'):
                return self.clean_text(meta['content'])
        
        return None

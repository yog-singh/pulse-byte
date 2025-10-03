"""
Configuration settings for PulseByte news scraper and analyzer.
"""

import os
from decouple import config
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# API Keys (set these in your .env file)
NEWS_API_KEY = config('NEWS_API_KEY', default='')
OPENAI_API_KEY = config('OPENAI_API_KEY', default='')

# Database settings
# PostgreSQL connection (preferred)
POSTGRES_HOST = config('POSTGRES_HOST', default='localhost')
POSTGRES_PORT = config('POSTGRES_PORT', default='5432')
POSTGRES_DB = config('POSTGRES_DB', default='pulse_byte')
POSTGRES_USER = config('POSTGRES_USER', default='postgres')
POSTGRES_PASSWORD = config('POSTGRES_PASSWORD', default='')

# Construct database URL
if POSTGRES_PASSWORD:
    POSTGRES_URL = f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'
else:
    POSTGRES_URL = f'postgresql://{POSTGRES_USER}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'

# Primary database URL (PostgreSQL preferred, SQLite fallback)
DATABASE_URL = config('DATABASE_URL', default=POSTGRES_URL)

# SQLite fallback URL
SQLITE_URL = f'sqlite:///{BASE_DIR}/data/pulse_byte.db'

# Database configuration
DATABASE_CONFIG = {
    'url': DATABASE_URL,
    'sqlite_fallback': SQLITE_URL,
    'pool_size': config('DB_POOL_SIZE', default=10, cast=int),
    'max_overflow': config('DB_MAX_OVERFLOW', default=20, cast=int),
    'pool_timeout': config('DB_POOL_TIMEOUT', default=30, cast=int),
    'pool_recycle': config('DB_POOL_RECYCLE', default=3600, cast=int),
    'echo': config('DB_ECHO', default=False, cast=bool),
}

# Scraping settings
SCRAPING_CONFIG = {
    'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'REQUEST_DELAY': 1,  # Delay between requests in seconds
    'MAX_RETRIES': 3,
    'TIMEOUT': 30,
    'MAX_ARTICLES_PER_SOURCE': 100,
}

# News sources configuration
NEWS_SOURCES = {
    'rss_feeds': [
        'https://rss.cnn.com/rss/edition.rss',
        'https://feeds.bbci.co.uk/news/rss.xml',
        'https://rss.reuters.com/reuters/topNews',
        'https://feeds.npr.org/1001/rss.xml',
        'https://www.theguardian.com/world/rss',
    ],
    'news_apis': [
        'newsapi',  # Requires API key
        'gnews',    # Uses Google News (no key required)
    ],
    'websites': [
        'https://www.bbc.com/news',
        'https://edition.cnn.com',
        'https://www.reuters.com',
        'https://www.npr.org/sections/news/',
    ]
}

# GNews settings
GNEWS_CONFIG = {
    'language': config('GNEWS_LANGUAGE', default='en'),
    'country': config('GNEWS_COUNTRY', default='IN'),  # e.g., 'US'
    'max_results': config('GNEWS_MAX_RESULTS', default=50, cast=int),
    'period': config('GNEWS_PERIOD', default='7d', cast=str)
}

# Sentiment analysis settings
SENTIMENT_CONFIG = {
    'models': ['vader', 'textblob', 'transformers'],
    'default_model': 'vader',
    'transformers_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'confidence_threshold': 0.6,
}

# Logging settings
LOGGING_CONFIG = {
    'level': config('LOG_LEVEL', default='INFO'),
    'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}',
    'rotation': '10 MB',
    'retention': '30 days',
    'log_file': BASE_DIR / 'logs' / 'pulse_byte.log',
}

# Data storage paths
DATA_PATHS = {
    'raw_data': BASE_DIR / 'data' / 'raw',
    'processed_data': BASE_DIR / 'data' / 'processed',
    'models': BASE_DIR / 'data' / 'models',
    'exports': BASE_DIR / 'data' / 'exports',
}

# Create necessary directories
for path in DATA_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# Keywords for news filtering (can be updated based on requirements)
DEFAULT_KEYWORDS = [
    'technology', 'artificial intelligence', 'machine learning',
    'cryptocurrency', 'blockchain', 'climate change',
    'politics', 'economy', 'healthcare', 'science'
]

# Selenium WebDriver settings
SELENIUM_CONFIG = {
    'headless': True,
    'window_size': (1920, 1080),
    'implicit_wait': 10,
    'page_load_timeout': 30,
    'chrome_options': [
        '--no-sandbox',
        '--disable-dev-shm-usage',
        '--disable-gpu',
        '--disable-extensions',
        '--disable-web-security',
        '--allow-running-insecure-content'
    ]
}

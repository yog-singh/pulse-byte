# PulseByte 🚀

**A comprehensive news scraping and sentiment analysis platform**

PulseByte is a powerful Python application that automatically fetches news articles from multiple sources, analyzes their sentiment using state-of-the-art machine learning models, and provides comprehensive data analysis and export capabilities.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- **Multi-Source News Scraping**
  - RSS feeds from major news outlets
  - News API integration
  - Direct website scraping with Selenium
  - Parallel processing for fast data collection

- **Advanced Sentiment Analysis**
  - VADER sentiment analyzer (rule-based)
  - TextBlob sentiment analysis
  - Transformers-based models (BERT, RoBERTa)
  - Ensemble analysis for improved accuracy

- **Robust Data Management**
  - SQLite database for local storage
  - Data export to JSON, CSV, and Excel
  - Automatic data cleaning and deduplication
  - Comprehensive logging

- **Flexible Configuration**
  - Customizable news sources
  - Keyword-based filtering
  - Configurable analysis parameters
  - Environment-based settings

## 🛠️ Installation

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/pulse-byte.git
   cd pulse-byte
   ```

2. **Run the automated installer**
   ```bash
   python scripts/install.py
   ```

3. **Activate the virtual environment**
   ```bash
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

4. **Configure API keys** (optional but recommended)
   ```bash
   cp env.example .env
   # Edit .env and add your API keys
   ```

### Manual Installation

If you prefer manual setup:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Download required data
python -m textblob.download_corpora
python -c "import nltk; nltk.download('vader_lexicon')"
```

## 🚀 Quick Start

### 1. Setup Database (PostgreSQL Recommended)

#### Option A: PostgreSQL (Recommended for Production)
```bash
# Install PostgreSQL (see database setup section below)
# Configure your .env file with PostgreSQL credentials
python scripts/setup_database.py --create
```

#### Option B: SQLite (Quick Start)
```bash
# SQLite will be used automatically if PostgreSQL is not available
python main.py test  # This will create SQLite database automatically
```

### 2. Test the Installation
```bash
python main.py test
```

### 3. Run Your First Analysis
```bash
python main.py run --keywords "artificial intelligence" --max-articles 20
```

### 4. View System Statistics
```bash
python main.py stats
```

## 📖 Usage Guide

### Command Line Interface

PulseByte provides a comprehensive CLI for all operations:

#### Full Pipeline (Scrape + Analyze)
```bash
# Basic usage
python main.py run --keywords "climate change" "renewable energy"

# Advanced usage
python main.py run \
  --keywords "technology" "AI" "machine learning" \
  --max-articles 50 \
  --analyzer transformers \
  --export json
```

#### Scraping Only
```bash
# Scrape articles without analysis
python main.py scrape --keywords "politics" --max-articles 30 --export
```

#### Analyze Existing Data
```bash
# Analyze articles already in database
python main.py analyze --analyzer vader --export csv
```

### Python API

You can also use PulseByte programmatically:

```python
from src.scraper.scraper_manager import ScraperManager
from src.analyzer.analyzer_manager import AnalyzerManager

# Initialize components
scraper = ScraperManager()
analyzer = AnalyzerManager()

# Scrape articles
results = scraper.scrape_all_sources(
    keywords=['technology', 'AI'],
    max_articles_per_source=25
)

# Collect articles
articles = []
for result in results:
    if result.success:
        articles.extend(result.articles)

# Analyze sentiment
analyzed_articles = analyzer.analyze_articles(articles, analyzer_name='vader')

# Access results
for article in analyzed_articles:
    if article.sentiment:
        print(f"{article.title}: {article.sentiment.label.value}")
```

## 🔧 Configuration

### News Sources

Edit `config/settings.py` to customize news sources:

```python
NEWS_SOURCES = {
    'rss_feeds': [
        'https://rss.cnn.com/rss/edition.rss',
        'https://feeds.bbci.co.uk/news/rss.xml',
        # Add your RSS feeds
    ],
    'websites': [
        'https://www.bbc.com/news',
        # Add websites to scrape
    ]
}
```

### API Keys

Add your API keys to `.env`:

```env
# News API (https://newsapi.org/)
NEWS_API_KEY=your_news_api_key_here

# OpenAI API (optional, for advanced features)
OPENAI_API_KEY=your_openai_api_key_here
```

### Database Setup

PulseByte supports both PostgreSQL (recommended) and SQLite databases:

#### PostgreSQL Setup (Recommended)

1. **Install PostgreSQL:**

   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install postgresql postgresql-contrib
   
   # macOS (Homebrew)
   brew install postgresql && brew services start postgresql
   
   # Windows (Chocolatey)
   choco install postgresql
   
   # Docker (All platforms)
   docker run --name pulse-byte-postgres \
     -e POSTGRES_PASSWORD=your_password \
     -e POSTGRES_DB=pulse_byte \
     -p 5432:5432 -d postgres:15
   ```

2. **Create Database and User:**

   ```bash
   sudo -u postgres psql
   CREATE DATABASE pulse_byte;
   CREATE USER pulse_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE pulse_byte TO pulse_user;
   \q
   ```

3. **Configure Environment:**

   ```env
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_DB=pulse_byte
   POSTGRES_USER=pulse_user
   POSTGRES_PASSWORD=your_password
   ```

4. **Initialize Database:**

   ```bash
   python scripts/setup_database.py --create
   ```

#### Database Management Commands

```bash
# Check PostgreSQL connection
python scripts/setup_database.py --check

# Show PostgreSQL installation help
python scripts/setup_database.py --install-help

# Create database tables
python scripts/setup_database.py --create

# Migrate from SQLite to PostgreSQL
python scripts/setup_database.py --migrate

# Test database operations
python scripts/setup_database.py --test
```

### Sentiment Analysis Models

Configure analysis models in `config/settings.py`:

```python
SENTIMENT_CONFIG = {
    'models': ['vader', 'textblob', 'transformers'],
    'default_model': 'vader',
    'transformers_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest'
}
```

## 📊 Data Analysis Examples

### Sentiment Distribution
```python
from src.models.database import db_manager

# Get sentiment statistics
stats = db_manager.get_sentiment_stats()
print(f"Total articles: {stats['total_articles']}")
print(f"Sentiment distribution: {stats['sentiment_distribution']}")
```

### Export Data
```python
from src.utils.data_utils import export_articles

# Export to Excel with analysis
export_articles(articles, format='excel', filename='news_analysis')
```

### Advanced Filtering
```python
# Get articles by keywords and sentiment
positive_tech_articles = db_manager.get_articles(
    keywords=['technology', 'AI'],
    sentiment=SentimentLabel.POSITIVE,
    limit=50
)
```

## 🏗️ Architecture

```
pulse-byte/
├── src/
│   ├── scraper/          # News scraping modules
│   │   ├── base_scraper.py
│   │   ├── rss_scraper.py
│   │   ├── web_scraper.py
│   │   └── scraper_manager.py
│   ├── analyzer/         # Sentiment analysis modules
│   │   ├── vader_analyzer.py
│   │   ├── textblob_analyzer.py
│   │   ├── transformers_analyzer.py
│   │   └── analyzer_manager.py
│   ├── models/           # Data models and database
│   │   ├── article.py
│   │   └── database.py
│   └── utils/            # Utility functions
│       ├── data_utils.py
│       ├── text_utils.py
│       └── logger.py
├── config/               # Configuration files
├── data/                 # Data storage
├── tests/                # Test files
└── scripts/              # Utility scripts
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific tests
python -m pytest tests/test_basic.py

# Run with coverage
python -m pytest tests/ --cov=src
```

## 📈 Performance Tips

1. **Use PostgreSQL**: For better performance with large datasets
   - Optimized JSONB storage for metadata and sentiment scores
   - Connection pooling for concurrent operations
   - Advanced indexing for faster queries
   - Full-text search capabilities

2. **Parallel Scraping**: Enable parallel processing for faster scraping
   ```python
   results = scraper.scrape_all_sources(use_parallel=True, max_workers=5)
   ```

3. **Batch Analysis**: Use batch processing for large datasets
   ```python
   analyzer.analyze_batch(texts, batch_size=16)
   ```

4. **Database Optimization**: 
   ```python
   # Regularly clean old articles
   db_manager.delete_old_articles(days=30)
   
   # Use indexes for frequent queries
   # (Automatically created for PostgreSQL)
   ```

5. **Memory Management**: Use streaming for large exports
   ```python
   export_articles(articles, format='csv')  # More memory efficient than Excel
   ```

### Database Performance Comparison

| Feature | PostgreSQL | SQLite |
|---------|------------|--------|
| **Concurrent Writes** | ✅ Excellent | ⚠️ Limited |
| **Large Datasets** | ✅ Optimized | ⚠️ Slower |
| **JSON Operations** | ✅ JSONB | ✅ Basic JSON |
| **Full-text Search** | ✅ Native | ❌ Limited |
| **Connection Pooling** | ✅ Yes | ❌ No |
| **Setup Complexity** | ⚠️ Moderate | ✅ Simple |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check this README and inline code documentation
- **Issues**: Open an issue on GitHub
- **Examples**: See `scripts/example_usage.py` for detailed examples

## 🔮 Roadmap

- [x] **PostgreSQL Support**: Enhanced database performance and scalability
- [ ] Web dashboard for visualization
- [ ] Real-time news monitoring
- [ ] Advanced NLP features (named entity recognition, topic modeling)
- [ ] Social media integration
- [ ] RESTful API
- [ ] Docker containerization
- [ ] Cloud deployment options
- [ ] Multi-language sentiment analysis
- [ ] Advanced data analytics and reporting

## 🙏 Acknowledgments

- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [TextBlob](https://textblob.readthedocs.io/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/)
- [Selenium](https://selenium-python.readthedocs.io/)

---

**Happy news analyzing! 📰✨**
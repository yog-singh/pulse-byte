"""
Data utility functions for PulseByte.
"""

import json
import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

from src.models import Article, ScrapingResult
from config.settings import DATA_PATHS


def export_articles(articles: List[Article], 
                   format: str = 'json',
                   filename: Optional[str] = None) -> str:
    """
    Export articles to various formats.
    
    Args:
        articles: List of articles to export
        format: Export format ('json', 'csv', 'excel')
        filename: Optional filename. If None, generates timestamp-based name.
        
    Returns:
        Path to exported file
    """
    if not articles:
        logger.warning("No articles to export")
        return ""
    
    # Generate filename if not provided
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"articles_{timestamp}"
    
    export_path = DATA_PATHS['exports']
    export_path.mkdir(exist_ok=True)
    
    if format.lower() == 'json':
        return _export_to_json(articles, export_path / f"{filename}.json")
    elif format.lower() == 'csv':
        return _export_to_csv(articles, export_path / f"{filename}.csv")
    elif format.lower() == 'excel':
        return _export_to_excel(articles, export_path / f"{filename}.xlsx")
    else:
        raise ValueError(f"Unsupported export format: {format}")


def _export_to_json(articles: List[Article], filepath: Path) -> str:
    """Export articles to JSON format."""
    try:
        data = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'total_articles': len(articles),
                'format_version': '1.0'
            },
            'articles': [article.to_dict() for article in articles]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Exported {len(articles)} articles to JSON: {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Error exporting to JSON: {e}")
        raise


def _export_to_csv(articles: List[Article], filepath: Path) -> str:
    """Export articles to CSV format."""
    try:
        fieldnames = [
            'title', 'url', 'source', 'published_date', 'scraped_date',
            'author', 'summary', 'content', 'keywords', 'category',
            'source_type', 'sentiment_label', 'sentiment_confidence'
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for article in articles:
                row = {
                    'title': article.title,
                    'url': article.url,
                    'source': article.source,
                    'published_date': article.published_date.isoformat(),
                    'scraped_date': article.scraped_date.isoformat(),
                    'author': article.author or '',
                    'summary': article.summary or '',
                    'content': article.content,
                    'keywords': ', '.join(article.keywords),
                    'category': article.category or '',
                    'source_type': article.source_type.value,
                    'sentiment_label': article.sentiment.label.value if article.sentiment else '',
                    'sentiment_confidence': article.sentiment.confidence if article.sentiment else ''
                }
                writer.writerow(row)
        
        logger.info(f"Exported {len(articles)} articles to CSV: {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        raise


def _export_to_excel(articles: List[Article], filepath: Path) -> str:
    """Export articles to Excel format."""
    try:
        # Prepare data for DataFrame
        data = []
        for article in articles:
            row = {
                'Title': article.title,
                'URL': article.url,
                'Source': article.source,
                'Published Date': article.published_date,
                'Scraped Date': article.scraped_date,
                'Author': article.author or '',
                'Summary': article.summary or '',
                'Content': article.content,
                'Keywords': ', '.join(article.keywords),
                'Category': article.category or '',
                'Source Type': article.source_type.value,
                'Sentiment Label': article.sentiment.label.value if article.sentiment else '',
                'Sentiment Confidence': article.sentiment.confidence if article.sentiment else '',
                'Word Count': article.get_word_count(),
                'Reading Time (min)': article.get_reading_time()
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Create Excel file with multiple sheets
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Main articles sheet
            df.to_excel(writer, sheet_name='Articles', index=False)
            
            # Summary statistics sheet
            if any(article.sentiment for article in articles):
                summary_data = _create_summary_stats(articles)
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"Exported {len(articles)} articles to Excel: {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Error exporting to Excel: {e}")
        raise


def _create_summary_stats(articles: List[Article]) -> List[Dict[str, Any]]:
    """Create summary statistics for articles."""
    total_articles = len(articles)
    analyzed_articles = [a for a in articles if a.sentiment]
    
    stats = [
        {'Metric': 'Total Articles', 'Value': total_articles},
        {'Metric': 'Analyzed Articles', 'Value': len(analyzed_articles)},
        {'Metric': 'Analysis Coverage', 'Value': f"{len(analyzed_articles)/total_articles*100:.1f}%"}
    ]
    
    if analyzed_articles:
        # Sentiment distribution
        sentiment_counts = {}
        for article in analyzed_articles:
            label = article.sentiment.label.value
            sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
        
        for label, count in sentiment_counts.items():
            percentage = count / len(analyzed_articles) * 100
            stats.append({
                'Metric': f'{label.title()} Sentiment',
                'Value': f'{count} ({percentage:.1f}%)'
            })
        
        # Average confidence
        avg_confidence = sum(a.sentiment.confidence for a in analyzed_articles) / len(analyzed_articles)
        stats.append({'Metric': 'Average Confidence', 'Value': f'{avg_confidence:.3f}'})
        
        # Source distribution
        source_counts = {}
        for article in articles:
            source_counts[article.source] = source_counts.get(article.source, 0) + 1
        
        stats.append({'Metric': 'Number of Sources', 'Value': len(source_counts)})
    
    return stats


def import_articles(filepath: str) -> List[Article]:
    """
    Import articles from a file.
    
    Args:
        filepath: Path to the file to import
        
    Returns:
        List of imported articles
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if path.suffix.lower() == '.json':
        return _import_from_json(path)
    elif path.suffix.lower() == '.csv':
        return _import_from_csv(path)
    elif path.suffix.lower() in ['.xlsx', '.xls']:
        return _import_from_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def _import_from_json(filepath: Path) -> List[Article]:
    """Import articles from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'articles' in data:
            # New format with metadata
            articles_data = data['articles']
        elif isinstance(data, list):
            # Simple list format
            articles_data = data
        else:
            raise ValueError("Invalid JSON format")
        
        articles = []
        for article_dict in articles_data:
            try:
                article = Article.from_dict(article_dict)
                articles.append(article)
            except Exception as e:
                logger.warning(f"Error importing article: {e}")
        
        logger.info(f"Imported {len(articles)} articles from JSON: {filepath}")
        return articles
        
    except Exception as e:
        logger.error(f"Error importing from JSON: {e}")
        raise


def _import_from_csv(filepath: Path) -> List[Article]:
    """Import articles from CSV file."""
    try:
        df = pd.read_csv(filepath)
        articles = []
        
        for _, row in df.iterrows():
            try:
                # Parse dates
                published_date = pd.to_datetime(row['published_date'])
                scraped_date = pd.to_datetime(row.get('scraped_date', datetime.now()))
                
                # Parse keywords
                keywords = [k.strip() for k in str(row.get('keywords', '')).split(',') if k.strip()]
                
                # Create article
                article = Article(
                    title=str(row['title']),
                    content=str(row['content']),
                    url=str(row['url']),
                    source=str(row['source']),
                    published_date=published_date,
                    scraped_date=scraped_date,
                    author=str(row.get('author', '')) if pd.notna(row.get('author')) else None,
                    summary=str(row.get('summary', '')) if pd.notna(row.get('summary')) else None,
                    keywords=keywords,
                    category=str(row.get('category', '')) if pd.notna(row.get('category')) else None
                )
                
                articles.append(article)
                
            except Exception as e:
                logger.warning(f"Error importing CSV row: {e}")
        
        logger.info(f"Imported {len(articles)} articles from CSV: {filepath}")
        return articles
        
    except Exception as e:
        logger.error(f"Error importing from CSV: {e}")
        raise


def _import_from_excel(filepath: Path) -> List[Article]:
    """Import articles from Excel file."""
    try:
        df = pd.read_excel(filepath, sheet_name='Articles')
        return _import_from_csv(df)  # Reuse CSV logic
        
    except Exception as e:
        logger.error(f"Error importing from Excel: {e}")
        raise


def clean_data(articles: List[Article]) -> List[Article]:
    """
    Clean and validate article data.
    
    Args:
        articles: List of articles to clean
        
    Returns:
        List of cleaned articles
    """
    cleaned_articles = []
    
    for article in articles:
        try:
            # Skip articles with missing essential data
            if not article.title or not article.content or not article.url:
                logger.debug(f"Skipping article with missing essential data: {article.url}")
                continue
            
            # Clean and validate content
            article.title = article.title.strip()
            article.content = article.content.strip()
            
            # Remove duplicates by URL
            if not any(a.url == article.url for a in cleaned_articles):
                cleaned_articles.append(article)
            else:
                logger.debug(f"Skipping duplicate article: {article.url}")
                
        except Exception as e:
            logger.warning(f"Error cleaning article: {e}")
    
    logger.info(f"Cleaned {len(cleaned_articles)} articles from {len(articles)} total")
    return cleaned_articles


def save_scraping_results(results: List[ScrapingResult], filename: Optional[str] = None) -> str:
    """
    Save scraping results to JSON file.
    
    Args:
        results: List of scraping results
        filename: Optional filename
        
    Returns:
        Path to saved file
    """
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scraping_results_{timestamp}.json"
    
    export_path = DATA_PATHS['raw'] / filename
    
    try:
        data = {
            'timestamp': datetime.now().isoformat(),
            'total_sources': len(results),
            'successful_sources': sum(1 for r in results if r.success),
            'total_articles': sum(len(r.articles) for r in results),
            'results': [result.to_dict() for result in results]
        }
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved scraping results to: {export_path}")
        return str(export_path)
        
    except Exception as e:
        logger.error(f"Error saving scraping results: {e}")
        raise

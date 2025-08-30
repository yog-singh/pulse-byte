#!/usr/bin/env python3
"""
Example usage scripts for PulseByte.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.utils.logger import setup_logging
from src.scraper.scraper_manager import ScraperManager
from src.analyzer.analyzer_manager import AnalyzerManager
from src.models.database import db_manager
from src.utils.data_utils import export_articles
from loguru import logger


def example_1_basic_scraping():
    """Example 1: Basic news scraping."""
    print("=" * 60)
    print("Example 1: Basic News Scraping")
    print("=" * 60)
    
    setup_logging()
    
    # Initialize scraper manager
    scraper_manager = ScraperManager()
    
    # Scrape articles about AI and technology
    keywords = ['artificial intelligence', 'machine learning', 'technology']
    results = scraper_manager.scrape_all_sources(
        keywords=keywords,
        max_articles_per_source=5,
        use_parallel=True
    )
    
    # Display results
    total_articles = 0
    for result in results:
        if result.success:
            print(f"✓ {result.source}: {len(result.articles)} articles")
            total_articles += len(result.articles)
        else:
            print(f"✗ {result.source}: {result.error_message}")
    
    print(f"\nTotal articles scraped: {total_articles}")
    
    # Return articles for next examples
    all_articles = []
    for result in results:
        if result.success:
            all_articles.extend(result.articles)
    
    return all_articles


def example_2_sentiment_analysis(articles):
    """Example 2: Sentiment analysis with multiple models."""
    print("\n" + "=" * 60)
    print("Example 2: Sentiment Analysis")
    print("=" * 60)
    
    if not articles:
        print("No articles to analyze")
        return []
    
    # Initialize analyzer manager
    analyzer_manager = AnalyzerManager()
    
    print(f"Analyzing {len(articles)} articles...")
    
    # Test different analyzers
    analyzers = ['vader', 'textblob']
    results = {}
    
    for analyzer_name in analyzers:
        if analyzer_name in analyzer_manager.get_available_analyzers():
            print(f"\nUsing {analyzer_name.upper()} analyzer...")
            analyzed_articles = analyzer_manager.analyze_articles(
                articles.copy(), 
                analyzer_name=analyzer_name
            )
            
            # Show sentiment distribution
            sentiments = [a.sentiment.label.value for a in analyzed_articles if a.sentiment]
            sentiment_counts = {}
            for sentiment in sentiments:
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            print(f"Sentiment distribution with {analyzer_name}:")
            for sentiment, count in sentiment_counts.items():
                percentage = count / len(sentiments) * 100 if sentiments else 0
                print(f"  {sentiment}: {count} ({percentage:.1f}%)")
            
            results[analyzer_name] = analyzed_articles
        else:
            print(f"❌ {analyzer_name} analyzer not available")
    
    # Return analyzed articles
    return results.get('vader', articles)


def example_3_database_operations(articles):
    """Example 3: Database operations."""
    print("\n" + "=" * 60)
    print("Example 3: Database Operations")
    print("=" * 60)
    
    if not articles:
        print("No articles to save")
        return
    
    # Save articles to database
    print(f"Saving {len(articles)} articles to database...")
    saved_count = db_manager.save_articles(articles)
    print(f"✓ Saved {saved_count} articles")
    
    # Get database statistics
    stats = db_manager.get_sentiment_stats()
    print(f"\nDatabase statistics:")
    print(f"  Total articles: {stats['total_articles']}")
    print(f"  Analyzed articles: {stats['analyzed_articles']}")
    
    if stats['sentiment_distribution']:
        print(f"  Sentiment distribution:")
        for sentiment, count in stats['sentiment_distribution'].items():
            print(f"    {sentiment}: {count}")
    
    # Query articles by keyword
    print(f"\nQuerying articles with keyword 'AI'...")
    ai_articles = db_manager.get_articles(keywords=['AI'], limit=5)
    print(f"Found {len(ai_articles)} articles mentioning AI")
    
    for article in ai_articles[:3]:  # Show first 3
        print(f"  • {article.title[:60]}...")


def example_4_data_export(articles):
    """Example 4: Data export."""
    print("\n" + "=" * 60)
    print("Example 4: Data Export")
    print("=" * 60)
    
    if not articles:
        print("No articles to export")
        return
    
    # Export to different formats
    formats = ['json', 'csv']
    
    for format_type in formats:
        try:
            export_path = export_articles(
                articles[:10],  # Export first 10 articles
                format=format_type,
                filename=f"example_export"
            )
            print(f"✓ Exported to {format_type.upper()}: {export_path}")
        except Exception as e:
            print(f"✗ Failed to export to {format_type}: {e}")


def example_5_advanced_analysis(articles):
    """Example 5: Advanced analysis and filtering."""
    print("\n" + "=" * 60)
    print("Example 5: Advanced Analysis")
    print("=" * 60)
    
    if not articles:
        print("No articles to analyze")
        return
    
    # Analyze articles by source
    print("Articles by source:")
    source_counts = {}
    for article in articles:
        source_counts[article.source] = source_counts.get(article.source, 0) + 1
    
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count} articles")
    
    # Analyze sentiment by source
    print("\nSentiment by source:")
    for source in source_counts.keys():
        source_articles = [a for a in articles if a.source == source and a.sentiment]
        if source_articles:
            sentiments = [a.sentiment.label.value for a in source_articles]
            positive = sentiments.count('positive')
            negative = sentiments.count('negative')
            neutral = sentiments.count('neutral')
            total = len(sentiments)
            
            print(f"  {source}:")
            if total > 0:
                print(f"    Positive: {positive} ({positive/total*100:.1f}%)")
                print(f"    Negative: {negative} ({negative/total*100:.1f}%)")
                print(f"    Neutral: {neutral} ({neutral/total*100:.1f}%)")
    
    # Find most positive and negative articles
    positive_articles = [a for a in articles if a.sentiment and a.sentiment.label.value == 'positive']
    negative_articles = [a for a in articles if a.sentiment and a.sentiment.label.value == 'negative']
    
    if positive_articles:
        best_positive = max(positive_articles, key=lambda a: a.sentiment.confidence)
        print(f"\nMost positive article ({best_positive.sentiment.confidence:.3f} confidence):")
        print(f"  {best_positive.title}")
    
    if negative_articles:
        best_negative = max(negative_articles, key=lambda a: a.sentiment.confidence)
        print(f"\nMost negative article ({best_negative.sentiment.confidence:.3f} confidence):")
        print(f"  {best_negative.title}")


def main():
    """Run all examples."""
    print("🚀 PulseByte Usage Examples")
    print("This script demonstrates various features of PulseByte")
    
    try:
        # Example 1: Basic scraping
        articles = example_1_basic_scraping()
        
        if articles:
            # Example 2: Sentiment analysis
            analyzed_articles = example_2_sentiment_analysis(articles)
            
            # Example 3: Database operations
            example_3_database_operations(analyzed_articles)
            
            # Example 4: Data export
            example_4_data_export(analyzed_articles)
            
            # Example 5: Advanced analysis
            example_5_advanced_analysis(analyzed_articles)
        
        print("\n" + "=" * 60)
        print("✅ All examples completed!")
        print("Check the data/exports/ directory for exported files.")
        print("=" * 60)
    
    except KeyboardInterrupt:
        print("\n❌ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        logger.exception("Error in examples")


if __name__ == "__main__":
    main()

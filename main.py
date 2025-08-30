#!/usr/bin/env python3
"""
PulseByte - Main application entry point.

A comprehensive news scraping and sentiment analysis platform that:
- Fetches news articles from multiple sources (RSS, News API, websites)
- Analyzes sentiment using multiple ML models
- Stores results in a database
- Provides data export capabilities
"""

import argparse
import sys
import time
from typing import List, Optional
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.logger import setup_logging
from src.scraper.scraper_manager import ScraperManager
from src.analyzer.analyzer_manager import AnalyzerManager
from src.models.database import db_manager
from src.utils.data_utils import export_articles, save_scraping_results
from config.settings import DEFAULT_KEYWORDS, SENTIMENT_CONFIG
from loguru import logger


class PulseByte:
    """Main PulseByte application class."""
    
    def __init__(self):
        """Initialize PulseByte application."""
        setup_logging()
        logger.info("Initializing PulseByte...")
        
        self.scraper_manager = ScraperManager()
        self.analyzer_manager = AnalyzerManager()
        
        logger.info("PulseByte initialized successfully")
    
    def run_full_pipeline(self, 
                         keywords: Optional[List[str]] = None,
                         max_articles: Optional[int] = 100,
                         analyzer: str = 'vader',
                         export_format: Optional[str] = None,
                         save_to_db: bool = True) -> List:
        """
        Run the complete news scraping and analysis pipeline.
        
        Args:
            keywords: Keywords to search for. Uses defaults if None.
            max_articles: Maximum articles per source
            analyzer: Sentiment analyzer to use
            export_format: Export format ('json', 'csv', 'excel')
            save_to_db: Whether to save results to database
            
        Returns:
            List of analyzed articles
        """
        start_time = time.time()
        keywords = keywords or DEFAULT_KEYWORDS
        
        logger.info(f"Starting full pipeline with keywords: {keywords}")
        logger.info(f"Max articles per source: {max_articles}")
        logger.info(f"Analyzer: {analyzer}")
        
        # Step 1: Scrape articles
        logger.info("Step 1: Scraping articles from all sources...")
        scraping_results = self.scraper_manager.scrape_all_sources(
            keywords=keywords,
            max_articles_per_source=max_articles,
            use_parallel=True
        )
        
        # Collect all articles
        all_articles = []
        successful_sources = 0
        
        for result in scraping_results:
            if result.success:
                all_articles.extend(result.articles)
                successful_sources += 1
                logger.info(f"✓ {result.source}: {len(result.articles)} articles")
            else:
                logger.error(f"✗ {result.source}: {result.error_message}")
        
        logger.info(f"Scraping completed: {len(all_articles)} articles from {successful_sources} sources")
        
        if not all_articles:
            logger.warning("No articles were scraped. Exiting.")
            return []
        
        # Step 2: Analyze sentiment
        logger.info(f"Step 2: Analyzing sentiment with {analyzer}...")
        analyzed_articles = self.analyzer_manager.analyze_articles(
            all_articles, 
            analyzer_name=analyzer
        )
        
        # Step 3: Save to database
        if save_to_db:
            logger.info("Step 3: Saving articles to database...")
            saved_count = db_manager.save_articles(analyzed_articles)
            logger.info(f"Saved {saved_count} articles to database")
        
        # Step 4: Export results
        if export_format:
            logger.info(f"Step 4: Exporting results as {export_format}...")
            export_path = export_articles(analyzed_articles, format=export_format)
            logger.info(f"Results exported to: {export_path}")
        
        # Step 5: Generate summary
        analysis_summary = self.analyzer_manager.get_analysis_summary(analyzed_articles)
        
        total_time = time.time() - start_time
        logger.info(f"Pipeline completed in {total_time:.2f} seconds")
        logger.info(f"Total articles processed: {analysis_summary.articles_processed}")
        logger.info(f"Sentiment distribution: {analysis_summary.sentiment_distribution}")
        logger.info(f"Average confidence: {analysis_summary.average_confidence:.3f}")
        
        return analyzed_articles
    
    def scrape_only(self, 
                   keywords: Optional[List[str]] = None,
                   max_articles: Optional[int] = 100,
                   export_results: bool = False) -> List:
        """
        Run only the scraping part of the pipeline.
        
        Args:
            keywords: Keywords to search for
            max_articles: Maximum articles per source
            export_results: Whether to export scraping results
            
        Returns:
            List of scraping results
        """
        keywords = keywords or DEFAULT_KEYWORDS
        
        logger.info(f"Scraping articles with keywords: {keywords}")
        
        scraping_results = self.scraper_manager.scrape_all_sources(
            keywords=keywords,
            max_articles_per_source=max_articles,
            use_parallel=True
        )
        
        # Summary
        total_articles = sum(len(r.articles) for r in scraping_results)
        successful_sources = sum(1 for r in scraping_results if r.success)
        
        logger.info(f"Scraping completed: {total_articles} articles from {successful_sources} sources")
        
        if export_results:
            results_path = save_scraping_results(scraping_results)
            logger.info(f"Scraping results saved to: {results_path}")
        
        return scraping_results
    
    def analyze_existing(self, 
                        analyzer: str = 'vader',
                        keywords_filter: Optional[List[str]] = None,
                        limit: Optional[int] = None,
                        export_format: Optional[str] = None) -> List:
        """
        Analyze sentiment of existing articles in the database.
        
        Args:
            analyzer: Sentiment analyzer to use
            keywords_filter: Optional keywords to filter articles
            limit: Maximum number of articles to analyze
            export_format: Export format for results
            
        Returns:
            List of analyzed articles
        """
        logger.info(f"Analyzing existing articles with {analyzer}")
        
        # Get articles from database
        articles = db_manager.get_articles(
            limit=limit,
            keywords=keywords_filter
        )
        
        if not articles:
            logger.warning("No articles found in database")
            return []
        
        logger.info(f"Found {len(articles)} articles to analyze")
        
        # Analyze sentiment
        analyzed_articles = self.analyzer_manager.analyze_articles(
            articles,
            analyzer_name=analyzer
        )
        
        # Update database with new sentiment scores
        updated_count = db_manager.save_articles(analyzed_articles)
        logger.info(f"Updated {updated_count} articles in database")
        
        # Export if requested
        if export_format:
            export_path = export_articles(analyzed_articles, format=export_format)
            logger.info(f"Results exported to: {export_path}")
        
        return analyzed_articles
    
    def test_components(self) -> dict:
        """
        Test all system components.
        
        Returns:
            Dictionary with test results
        """
        logger.info("Testing PulseByte components...")
        
        results = {
            'scrapers': self.scraper_manager.test_scrapers(max_articles=1),
            'analyzers': self.analyzer_manager.test_analyzers(),
            'database': self._test_database()
        }
        
        logger.info("Component testing completed")
        return results
    
    def _test_database(self) -> dict:
        """Test database connectivity and operations."""
        try:
            # Test database connection by getting stats
            stats = db_manager.get_sentiment_stats()
            db_info = db_manager.get_database_info()
            
            return {
                'success': True,
                'stats': stats,
                'database_info': db_info
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        try:
            db_stats = db_manager.get_sentiment_stats()
            db_info = db_manager.get_database_info()
            scraper_stats = self.scraper_manager.get_scraper_stats()
            analyzer_info = self.analyzer_manager.get_analyzer_info()
            
            return {
                'database': {**db_stats, 'connection_info': db_info},
                'scrapers': scraper_stats,
                'analyzers': list(analyzer_info.keys())
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="PulseByte - News scraping and sentiment analysis platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py run --keywords "AI,machine learning" --max-articles 50
  python main.py scrape --keywords "climate change" --export
  python main.py analyze --analyzer vader --export json
  python main.py test
  python main.py stats
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run full pipeline
    run_parser = subparsers.add_parser('run', help='Run full scraping and analysis pipeline')
    run_parser.add_argument('--keywords', nargs='+', help='Keywords to search for')
    run_parser.add_argument('--max-articles', type=int, default=50, help='Max articles per source')
    run_parser.add_argument('--analyzer', default='vader', choices=['vader', 'textblob', 'transformers'], 
                           help='Sentiment analyzer to use')
    run_parser.add_argument('--export', choices=['json', 'csv', 'excel'], help='Export format')
    run_parser.add_argument('--no-db', action='store_true', help='Skip saving to database')
    
    # Scrape only
    scrape_parser = subparsers.add_parser('scrape', help='Scrape articles only')
    scrape_parser.add_argument('--keywords', nargs='+', help='Keywords to search for')
    scrape_parser.add_argument('--max-articles', type=int, default=50, help='Max articles per source')
    scrape_parser.add_argument('--export', action='store_true', help='Export scraping results')
    
    # Analyze existing
    analyze_parser = subparsers.add_parser('analyze', help='Analyze existing articles')
    analyze_parser.add_argument('--analyzer', default='vader', choices=['vader', 'textblob', 'transformers'],
                               help='Sentiment analyzer to use')
    analyze_parser.add_argument('--keywords', nargs='+', help='Filter by keywords')
    analyze_parser.add_argument('--limit', type=int, help='Limit number of articles')
    analyze_parser.add_argument('--export', choices=['json', 'csv', 'excel'], help='Export format')
    
    # Test components
    subparsers.add_parser('test', help='Test all system components')
    
    # Get statistics
    subparsers.add_parser('stats', help='Show system statistics')
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        app = PulseByte()
        
        if args.command == 'run':
            app.run_full_pipeline(
                keywords=args.keywords,
                max_articles=args.max_articles,
                analyzer=args.analyzer,
                export_format=args.export,
                save_to_db=not args.no_db
            )
        
        elif args.command == 'scrape':
            app.scrape_only(
                keywords=args.keywords,
                max_articles=args.max_articles,
                export_results=args.export
            )
        
        elif args.command == 'analyze':
            app.analyze_existing(
                analyzer=args.analyzer,
                keywords_filter=args.keywords,
                limit=args.limit,
                export_format=args.export
            )
        
        elif args.command == 'test':
            results = app.test_components()
            print("\n=== Test Results ===")
            
            # Scraper tests
            scraper_results = results['scrapers']
            print(f"\nScrapers: {scraper_results['successful']}/{scraper_results['total_scrapers']} working")
            for result in scraper_results['results']:
                status = "✓" if result['success'] else "✗"
                print(f"  {status} {result['source']}: {result['articles_found']} articles")
            
            # Analyzer tests
            analyzer_results = results['analyzers']
            print(f"\nAnalyzers:")
            for name, result in analyzer_results.items():
                status = "✓" if result['success'] else "✗"
                print(f"  {status} {name}")
            
            # Database test
            db_result = results['database']
            status = "✓" if db_result['success'] else "✗"
            print(f"\nDatabase: {status}")
            if db_result['success']:
                stats = db_result['stats']
                db_info = db_result.get('database_info', {})
                print(f"  Type: {db_info.get('type', 'unknown')}")
                print(f"  Total articles: {stats['total_articles']}")
                print(f"  Analyzed articles: {stats['analyzed_articles']}")
            else:
                print(f"  Error: {db_result.get('error', 'Unknown error')}")
        
        elif args.command == 'stats':
            stats = app.get_stats()
            print("\n=== System Statistics ===")
            
            if 'error' in stats:
                print(f"Error: {stats['error']}")
                return
            
            # Database stats
            db_stats = stats.get('database', {})
            connection_info = db_stats.get('connection_info', {})
            
            print(f"\nDatabase:")
            print(f"  Type: {connection_info.get('type', 'unknown')}")
            if connection_info.get('pool_size'):
                print(f"  Pool size: {connection_info.get('pool_size')}")
                print(f"  Active connections: {connection_info.get('checked_out', 0)}")
            print(f"  Total articles: {db_stats.get('total_articles', 0)}")
            print(f"  Analyzed articles: {db_stats.get('analyzed_articles', 0)}")
            
            sentiment_dist = db_stats.get('sentiment_distribution', {})
            if sentiment_dist:
                print(f"  Sentiment distribution:")
                for sentiment, count in sentiment_dist.items():
                    print(f"    {sentiment}: {count}")
            
            # Scraper stats
            scraper_stats = stats.get('scrapers', {})
            print(f"\nScrapers:")
            print(f"  Total scrapers: {scraper_stats.get('total_scrapers', 0)}")
            
            by_type = scraper_stats.get('by_type', {})
            for source_type, count in by_type.items():
                print(f"    {source_type}: {count}")
            
            # Analyzer stats
            analyzers = stats.get('analyzers', [])
            print(f"\nAvailable analyzers: {', '.join(analyzers)}")
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

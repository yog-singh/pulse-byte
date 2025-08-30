#!/usr/bin/env python3
"""
Database setup and migration script for PulseByte.
"""

import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.logger import setup_logging
    from src.models.database import DatabaseManager, ArticleDB
    from src.models.article import Article, SourceType
    from config.settings import DATABASE_CONFIG, POSTGRES_URL, SQLITE_URL
    from loguru import logger
    from datetime import datetime
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Please run 'pip install -r requirements.txt' first")
    sys.exit(1)


def check_postgres_connection():
    """Check if PostgreSQL is available and accessible."""
    try:
        # Try to connect with PostgreSQL settings
        config = DATABASE_CONFIG.copy()
        config['url'] = POSTGRES_URL
        
        db_manager = DatabaseManager(config)
        info = db_manager.get_database_info()
        
        if info['type'] == 'postgresql':
            logger.info("✓ PostgreSQL connection successful")
            logger.info(f"  Database type: {info['type']}")
            logger.info(f"  Pool size: {info['pool_size']}")
            return True
        else:
            logger.warning("✗ Connected to fallback database instead of PostgreSQL")
            return False
            
    except Exception as e:
        logger.error(f"✗ PostgreSQL connection failed: {e}")
        return False


def install_postgresql():
    """Provide instructions for installing PostgreSQL."""
    print("\n" + "="*60)
    print("PostgreSQL Installation Instructions")
    print("="*60)
    
    print("\n🐧 Linux (Ubuntu/Debian):")
    print("  sudo apt update")
    print("  sudo apt install postgresql postgresql-contrib")
    print("  sudo systemctl start postgresql")
    print("  sudo -u postgres createuser --interactive")
    print("  sudo -u postgres createdb pulse_byte")
    
    print("\n🍎 macOS:")
    print("  # Using Homebrew")
    print("  brew install postgresql")
    print("  brew services start postgresql")
    print("  createdb pulse_byte")
    
    print("\n🪟 Windows:")
    print("  # Download from: https://www.postgresql.org/download/windows/")
    print("  # Or use chocolatey:")
    print("  choco install postgresql")
    
    print("\n🐳 Docker (All platforms):")
    print("  docker run --name pulse-byte-postgres \\")
    print("    -e POSTGRES_PASSWORD=your_password \\")
    print("    -e POSTGRES_DB=pulse_byte \\")
    print("    -p 5432:5432 \\")
    print("    -d postgres:15")
    
    print("\n📝 After installation:")
    print("  1. Update your .env file with PostgreSQL credentials")
    print("  2. Run: python scripts/setup_database.py --create")
    print("  3. Test: python scripts/setup_database.py --test")


def create_database():
    """Create database and tables."""
    try:
        logger.info("Creating database tables...")
        
        db_manager = DatabaseManager()
        info = db_manager.get_database_info()
        
        logger.info(f"Database setup completed successfully!")
        logger.info(f"  Type: {info['type']}")
        logger.info(f"  Tables created for ArticleDB model")
        
        if info['type'] == 'postgresql':
            logger.info(f"  Connection pool size: {info['pool_size']}")
            logger.info(f"  Checked out connections: {info['checked_out']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        return False


def migrate_from_sqlite():
    """Migrate data from existing SQLite database to PostgreSQL."""
    try:
        logger.info("Starting migration from SQLite to PostgreSQL...")
        
        # Check if SQLite database exists
        sqlite_path = Path(SQLITE_URL.replace('sqlite:///', ''))
        if not sqlite_path.exists():
            logger.warning("No SQLite database found to migrate")
            return True
        
        # Import existing articles from SQLite
        # DatabaseManager already imported at top
        
        # Create SQLite manager
        sqlite_config = {'url': SQLITE_URL, 'sqlite_fallback': SQLITE_URL}
        sqlite_manager = DatabaseManager(sqlite_config)
        
        # Get all articles from SQLite
        logger.info("Reading articles from SQLite...")
        sqlite_articles = sqlite_manager.get_articles()
        
        if not sqlite_articles:
            logger.info("No articles found in SQLite database")
            return True
        
        logger.info(f"Found {len(sqlite_articles)} articles in SQLite")
        
        # Create PostgreSQL manager
        postgres_config = DATABASE_CONFIG.copy()
        postgres_config['url'] = POSTGRES_URL
        postgres_manager = DatabaseManager(postgres_config)
        
        # Save articles to PostgreSQL
        logger.info("Saving articles to PostgreSQL...")
        saved_count = postgres_manager.save_articles(sqlite_articles)
        
        logger.info(f"Migration completed! Migrated {saved_count}/{len(sqlite_articles)} articles")
        
        # Backup SQLite database
        backup_path = sqlite_path.parent / f"{sqlite_path.stem}_backup{sqlite_path.suffix}"
        sqlite_path.rename(backup_path)
        logger.info(f"SQLite database backed up to: {backup_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def test_database():
    """Test database connection and operations."""
    try:
        logger.info("Testing database connection and operations...")
        
        db_manager = DatabaseManager()
        info = db_manager.get_database_info()
        
        logger.info(f"✓ Database connection successful")
        logger.info(f"  Type: {info['type']}")
        logger.info(f"  URL: {info['url']}")
        
        # Test basic operations
        stats = db_manager.get_sentiment_stats()
        logger.info(f"✓ Database queries working")
        logger.info(f"  Total articles: {stats['total_articles']}")
        logger.info(f"  Analyzed articles: {stats['analyzed_articles']}")
        
        # Test article creation (without saving)
        
        test_article = Article(
            title="Test Article",
            content="This is a test article for database validation.",
            url="https://example.com/test",
            source="Test Source",
            published_date=datetime.now(),
            source_type=SourceType.WEBSITE
        )
        
        # Test conversion to database model
        db_article = ArticleDB.from_article(test_article)
        logger.info(f"✓ Article model conversion working")
        
        logger.info("Database test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Database setup and migration for PulseByte",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup_database.py --check          # Check PostgreSQL connection
  python scripts/setup_database.py --install-help   # Show installation instructions
  python scripts/setup_database.py --create         # Create database tables
  python scripts/setup_database.py --migrate        # Migrate from SQLite to PostgreSQL
  python scripts/setup_database.py --test           # Test database operations
        """
    )
    
    parser.add_argument('--check', action='store_true', help='Check PostgreSQL connection')
    parser.add_argument('--install-help', action='store_true', help='Show PostgreSQL installation instructions')
    parser.add_argument('--create', action='store_true', help='Create database tables')
    parser.add_argument('--migrate', action='store_true', help='Migrate from SQLite to PostgreSQL')
    parser.add_argument('--test', action='store_true', help='Test database operations')
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    setup_logging()
    
    try:
        if args.install_help:
            install_postgresql()
        
        if args.check:
            if check_postgres_connection():
                logger.info("PostgreSQL is ready to use!")
            else:
                logger.warning("PostgreSQL is not available. Will use SQLite fallback.")
        
        if args.create:
            if create_database():
                logger.info("Database setup completed successfully!")
            else:
                logger.error("Database setup failed!")
                sys.exit(1)
        
        if args.migrate:
            if migrate_from_sqlite():
                logger.info("Migration completed successfully!")
            else:
                logger.error("Migration failed!")
                sys.exit(1)
        
        if args.test:
            if test_database():
                logger.info("All database tests passed!")
            else:
                logger.error("Database tests failed!")
                sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

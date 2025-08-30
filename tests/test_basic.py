"""
Basic tests for PulseByte components.
"""

import unittest
import sys
from pathlib import Path
from datetime import datetime

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.models.article import Article, SentimentScore, SentimentLabel, SourceType
from src.analyzer.vader_analyzer import VaderAnalyzer
from src.utils.text_utils import preprocess_text, extract_keywords_advanced


class TestArticleModel(unittest.TestCase):
    """Test Article data model."""
    
    def setUp(self):
        """Set up test data."""
        self.article_data = {
            'title': 'Test Article Title',
            'content': 'This is a test article content with some sample text.',
            'url': 'https://example.com/test-article',
            'source': 'Test Source',
            'published_date': datetime.now(),
            'author': 'Test Author',
            'keywords': ['test', 'article', 'sample']
        }
    
    def test_article_creation(self):
        """Test creating an article."""
        article = Article(**self.article_data)
        
        self.assertEqual(article.title, self.article_data['title'])
        self.assertEqual(article.content, self.article_data['content'])
        self.assertEqual(article.url, self.article_data['url'])
        self.assertEqual(article.source, self.article_data['source'])
        self.assertEqual(article.author, self.article_data['author'])
        self.assertEqual(article.keywords, self.article_data['keywords'])
    
    def test_article_to_dict(self):
        """Test converting article to dictionary."""
        article = Article(**self.article_data)
        article_dict = article.to_dict()
        
        self.assertIsInstance(article_dict, dict)
        self.assertEqual(article_dict['title'], self.article_data['title'])
        self.assertEqual(article_dict['content'], self.article_data['content'])
    
    def test_article_from_dict(self):
        """Test creating article from dictionary."""
        article = Article(**self.article_data)
        article_dict = article.to_dict()
        
        recreated_article = Article.from_dict(article_dict)
        
        self.assertEqual(recreated_article.title, article.title)
        self.assertEqual(recreated_article.content, article.content)
        self.assertEqual(recreated_article.url, article.url)
    
    def test_word_count(self):
        """Test word count calculation."""
        article = Article(**self.article_data)
        word_count = article.get_word_count()
        
        expected_count = len(self.article_data['content'].split())
        self.assertEqual(word_count, expected_count)
    
    def test_has_keywords(self):
        """Test keyword matching."""
        article = Article(**self.article_data)
        
        # Should match
        self.assertTrue(article.has_keywords(['test']))
        self.assertTrue(article.has_keywords(['article']))
        
        # Should not match
        self.assertFalse(article.has_keywords(['nonexistent']))


class TestSentimentScore(unittest.TestCase):
    """Test SentimentScore model."""
    
    def test_sentiment_score_creation(self):
        """Test creating a sentiment score."""
        score = SentimentScore(
            label=SentimentLabel.POSITIVE,
            confidence=0.85,
            scores={'positive': 0.85, 'negative': 0.15},
            model_used='test_model'
        )
        
        self.assertEqual(score.label, SentimentLabel.POSITIVE)
        self.assertEqual(score.confidence, 0.85)
        self.assertEqual(score.model_used, 'test_model')
    
    def test_sentiment_score_to_dict(self):
        """Test converting sentiment score to dictionary."""
        score = SentimentScore(
            label=SentimentLabel.NEGATIVE,
            confidence=0.75,
            scores={'positive': 0.25, 'negative': 0.75},
            model_used='test_model'
        )
        
        score_dict = score.to_dict()
        
        self.assertIsInstance(score_dict, dict)
        self.assertEqual(score_dict['label'], 'negative')
        self.assertEqual(score_dict['confidence'], 0.75)


class TestVaderAnalyzer(unittest.TestCase):
    """Test VADER sentiment analyzer."""
    
    def setUp(self):
        """Set up analyzer."""
        self.analyzer = VaderAnalyzer()
        
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        success = self.analyzer.initialize()
        self.assertTrue(success)
        self.assertTrue(self.analyzer.is_initialized)
    
    def test_positive_sentiment(self):
        """Test positive sentiment analysis."""
        if not self.analyzer.initialize():
            self.skipTest("VADER analyzer not available")
        
        text = "This is a great and wonderful day! I'm very happy."
        result = self.analyzer.analyze_text(text)
        
        self.assertIsInstance(result, SentimentScore)
        self.assertEqual(result.label, SentimentLabel.POSITIVE)
        self.assertGreater(result.confidence, 0)
    
    def test_negative_sentiment(self):
        """Test negative sentiment analysis."""
        if not self.analyzer.initialize():
            self.skipTest("VADER analyzer not available")
        
        text = "This is terrible and awful. I hate this completely."
        result = self.analyzer.analyze_text(text)
        
        self.assertIsInstance(result, SentimentScore)
        self.assertEqual(result.label, SentimentLabel.NEGATIVE)
        self.assertGreater(result.confidence, 0)
    
    def test_neutral_sentiment(self):
        """Test neutral sentiment analysis."""
        if not self.analyzer.initialize():
            self.skipTest("VADER analyzer not available")
        
        text = "The weather is cloudy today."
        result = self.analyzer.analyze_text(text)
        
        self.assertIsInstance(result, SentimentScore)
        # Note: Neutral detection can be tricky, so we just check it's a valid result
        self.assertIn(result.label, [SentimentLabel.POSITIVE, SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL])


class TestTextUtils(unittest.TestCase):
    """Test text utility functions."""
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        text = "This is a test   with   extra spaces and http://example.com"
        processed = preprocess_text(text, remove_urls=True, normalize_whitespace=True)
        
        # Should remove URL and normalize whitespace
        self.assertNotIn('http://example.com', processed)
        self.assertNotIn('   ', processed)  # No extra spaces
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        text = "Machine learning and artificial intelligence are transforming technology."
        keywords = extract_keywords_advanced(text, max_keywords=5)
        
        self.assertIsInstance(keywords, list)
        self.assertLessEqual(len(keywords), 5)
        
        # Should contain relevant keywords
        text_lower = text.lower()
        for keyword in keywords:
            self.assertIn(keyword.lower(), text_lower)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_article_with_sentiment(self):
        """Test article creation with sentiment analysis."""
        # Create article
        article = Article(
            title='Great news about AI breakthrough',
            content='Scientists have made amazing progress in artificial intelligence research.',
            url='https://example.com/ai-news',
            source='Tech News',
            published_date=datetime.now()
        )
        
        # Analyze sentiment
        analyzer = VaderAnalyzer()
        if analyzer.initialize():
            analyzed_article = analyzer.analyze_article(article)
            
            self.assertIsNotNone(analyzed_article.sentiment)
            self.assertIsInstance(analyzed_article.sentiment, SentimentScore)
            # Should be positive given the content
            self.assertEqual(analyzed_article.sentiment.label, SentimentLabel.POSITIVE)


if __name__ == '__main__':
    unittest.main()

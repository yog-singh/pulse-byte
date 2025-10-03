"""
Analyzer manager for coordinating multiple sentiment analyzers.
"""

from typing import List, Optional, Dict, Any, Union
import time
from loguru import logger
import statistics

from src.models import Article, SentimentScore, SentimentLabel, AnalysisResult
from .vader_analyzer import VaderAnalyzer
from .textblob_analyzer import TextBlobAnalyzer
from .transformers_analyzer import TransformersAnalyzer
from config.settings import SENTIMENT_CONFIG


class AnalyzerManager:
    """Manages and coordinates multiple sentiment analyzers."""
    
    def __init__(self, analyzers: Optional[List[str]] = None):
        """
        Initialize analyzer manager.
        
        Args:
            analyzers: List of analyzer names to use. If None, uses config default.
        """
        self.analyzers = {}
        self.default_analyzer = SENTIMENT_CONFIG.get('default_model', 'vader')
        
        # Initialize requested analyzers
        requested_analyzers = analyzers or SENTIMENT_CONFIG.get('models', ['vader'])
        self._initialize_analyzers(requested_analyzers)
        
    def _initialize_analyzers(self, analyzer_names: List[str]):
        """Initialize the specified analyzers."""
        logger.info(f"Initializing sentiment analyzers: {analyzer_names}")
        
        for name in analyzer_names:
            try:
                if name.lower() == 'vader':
                    analyzer = VaderAnalyzer()
                elif name.lower() == 'textblob':
                    analyzer = TextBlobAnalyzer()
                elif name.lower() == 'transformers':
                    analyzer = TransformersAnalyzer()
                else:
                    logger.warning(f"Unknown analyzer: {name}")
                    continue
                
                if analyzer.initialize():
                    self.analyzers[name.lower()] = analyzer
                    logger.info(f"✓ {name} analyzer initialized")
                else:
                    logger.error(f"✗ Failed to initialize {name} analyzer")
                    
            except Exception as e:
                logger.error(f"Error initializing {name} analyzer: {e}")
        
        if not self.analyzers:
            logger.warning("No analyzers were successfully initialized")
        else:
            logger.info(f"Successfully initialized {len(self.analyzers)} analyzers")
    
    def analyze_article(self, article: Article, 
                       analyzer_name: Optional[str] = None) -> Article:
        """
        Analyze sentiment of a single article.
        
        Args:
            article: Article to analyze
            analyzer_name: Specific analyzer to use. If None, uses default.
            
        Returns:
            Article with updated sentiment
        """
        analyzer_name = analyzer_name or self.default_analyzer
        
        if analyzer_name not in self.analyzers:
            logger.error(f"Analyzer '{analyzer_name}' not available")
            return article
        
        try:
            analyzer = self.analyzers[analyzer_name]
            analyzed_article = analyzer.analyze_article(article)
            return analyzed_article
            
        except Exception as e:
            logger.error(f"Error analyzing article with {analyzer_name}: {e}")
            return article
    
    def analyze_articles(self, articles: List[Article],
                        analyzer_name: Optional[str] = None) -> List[Article]:
        """
        Analyze sentiment of multiple articles.
        
        Args:
            articles: List of articles to analyze
            analyzer_name: Specific analyzer to use. If None, uses default.
            
        Returns:
            List of articles with updated sentiment
        """
        if not articles:
            return articles
        
        analyzer_name = analyzer_name or self.default_analyzer
        
        if analyzer_name not in self.analyzers:
            logger.error(f"Analyzer '{analyzer_name}' not available")
            return articles
        
        start_time = time.time()
        logger.info(f"Analyzing {len(articles)} articles with {analyzer_name}")
        
        try:
            analyzer = self.analyzers[analyzer_name]
            analyzed_articles = analyzer.analyze_articles(articles)
            
            analysis_time = time.time() - start_time
            successful_analyses = sum(1 for article in analyzed_articles if article.sentiment is not None)
            
            logger.info(f"Analysis completed: {successful_analyses}/{len(articles)} successful in {analysis_time:.2f}s")
            logger.info(f"Analysis results: {analyzed_articles}")
            return analyzed_articles
            
        except Exception as e:
            logger.error(f"Error analyzing articles with {analyzer_name}: {e}")
            return articles
    
    def analyze_with_multiple_analyzers(self, articles: List[Article],
                                      analyzer_names: Optional[List[str]] = None) -> List[Article]:
        """
        Analyze articles with multiple analyzers and combine results.
        
        Args:
            articles: List of articles to analyze
            analyzer_names: List of analyzer names to use. If None, uses all available.
            
        Returns:
            List of articles with combined sentiment analysis
        """
        if not articles:
            return articles
        
        analyzer_names = analyzer_names or list(self.analyzers.keys())
        available_analyzers = [name for name in analyzer_names if name in self.analyzers]
        
        if not available_analyzers:
            logger.error("No valid analyzers specified")
            return articles
        
        logger.info(f"Analyzing {len(articles)} articles with multiple analyzers: {available_analyzers}")
        
        # Store results from each analyzer
        analyzer_results = {}
        
        for analyzer_name in available_analyzers:
            try:
                logger.info(f"Running {analyzer_name} analyzer...")
                analyzer = self.analyzers[analyzer_name]
                results = analyzer.analyze_articles(articles.copy())
                analyzer_results[analyzer_name] = results
                
            except Exception as e:
                logger.error(f"Error with {analyzer_name} analyzer: {e}")
        
        # Combine results
        combined_articles = []
        for i, original_article in enumerate(articles):
            try:
                combined_sentiment = self._combine_sentiment_results(
                    {name: results[i].sentiment for name, results in analyzer_results.items()
                     if i < len(results) and results[i].sentiment is not None}
                )
                
                # Create updated article
                updated_article = Article(
                    title=original_article.title,
                    content=original_article.content,
                    url=original_article.url,
                    source=original_article.source,
                    published_date=original_article.published_date,
                    scraped_date=original_article.scraped_date,
                    author=original_article.author,
                    summary=original_article.summary,
                    keywords=original_article.keywords,
                    category=original_article.category,
                    source_type=original_article.source_type,
                    sentiment=combined_sentiment,
                    metadata=original_article.metadata
                )
                
                combined_articles.append(updated_article)
                
            except Exception as e:
                logger.error(f"Error combining results for article {i}: {e}")
                combined_articles.append(original_article)
        
        return combined_articles
    
    def _combine_sentiment_results(self, sentiments: Dict[str, SentimentScore]) -> Optional[SentimentScore]:
        """
        Combine sentiment results from multiple analyzers.
        
        Args:
            sentiments: Dictionary of analyzer_name -> SentimentScore
            
        Returns:
            Combined SentimentScore or None if no valid sentiments
        """
        if not sentiments:
            return None
        
        # Collect all sentiment data
        labels = [sentiment.label for sentiment in sentiments.values()]
        confidences = [sentiment.confidence for sentiment in sentiments.values()]
        
        # Determine combined label by majority vote
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Get most common label
        combined_label = max(label_counts.keys(), key=lambda k: label_counts[k])
        
        # Calculate combined confidence as weighted average
        # Weight by individual confidence scores
        total_weight = sum(confidences)
        if total_weight > 0:
            weighted_confidence = sum(
                conf * (1 if sentiment.label == combined_label else 0.5)
                for sentiment, conf in zip(sentiments.values(), confidences)
            ) / total_weight
        else:
            weighted_confidence = 0.0
        
        # Combine all scores
        combined_scores = {}
        for analyzer_name, sentiment in sentiments.items():
            for score_name, score_value in sentiment.scores.items():
                key = f"{analyzer_name}_{score_name}"
                combined_scores[key] = score_value
        
        # Add aggregated scores
        combined_scores['consensus_confidence'] = weighted_confidence
        combined_scores['analyzer_agreement'] = label_counts[combined_label] / len(labels)
        
        return SentimentScore(
            label=combined_label,
            confidence=weighted_confidence,
            scores=combined_scores,
            model_used=f"ensemble_{'+'.join(sentiments.keys())}"
        )
    
    def get_analysis_summary(self, articles: List[Article]) -> AnalysisResult:
        """
        Get summary statistics of sentiment analysis results.
        
        Args:
            articles: List of analyzed articles
            
        Returns:
            AnalysisResult with summary statistics
        """
        analyzed_articles = [a for a in articles if a.sentiment is not None]
        
        if not analyzed_articles:
            return AnalysisResult(
                articles_processed=0,
                sentiment_distribution={},
                average_confidence=0.0,
                processing_time=0.0
            )
        
        # Count sentiment distribution
        sentiment_counts = {}
        confidences = []
        
        for article in analyzed_articles:
            label = article.sentiment.label.value
            sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
            confidences.append(article.sentiment.confidence)
        
        # Calculate average confidence
        avg_confidence = statistics.mean(confidences) if confidences else 0.0
        
        return AnalysisResult(
            articles_processed=len(analyzed_articles),
            sentiment_distribution=sentiment_counts,
            average_confidence=avg_confidence,
            processing_time=0.0  # This would be set by the caller
        )
    
    def get_available_analyzers(self) -> List[str]:
        """Get list of available analyzer names."""
        return list(self.analyzers.keys())
    
    def get_analyzer_info(self, analyzer_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about analyzers.
        
        Args:
            analyzer_name: Specific analyzer name. If None, returns info for all.
            
        Returns:
            Dictionary with analyzer information
        """
        if analyzer_name:
            if analyzer_name in self.analyzers:
                return self.analyzers[analyzer_name].get_model_info()
            else:
                return {'error': f'Analyzer {analyzer_name} not found'}
        else:
            return {name: analyzer.get_model_info() 
                   for name, analyzer in self.analyzers.items()}
    
    def test_analyzers(self, test_text: str = "This is a great day!") -> Dict[str, Any]:
        """
        Test all analyzers with sample text.
        
        Args:
            test_text: Text to test with
            
        Returns:
            Dictionary with test results
        """
        results = {}
        
        for name, analyzer in self.analyzers.items():
            try:
                start_time = time.time()
                sentiment = analyzer.analyze_text(test_text)
                analysis_time = time.time() - start_time
                
                results[name] = {
                    'success': True,
                    'sentiment': sentiment.to_dict(),
                    'analysis_time': analysis_time
                }
                
            except Exception as e:
                results[name] = {
                    'success': False,
                    'error': str(e),
                    'analysis_time': 0.0
                }
        
        return results
    
    def set_default_analyzer(self, analyzer_name: str) -> bool:
        """
        Set the default analyzer.
        
        Args:
            analyzer_name: Name of analyzer to set as default
            
        Returns:
            True if successful, False otherwise
        """
        if analyzer_name in self.analyzers:
            self.default_analyzer = analyzer_name
            logger.info(f"Default analyzer set to: {analyzer_name}")
            return True
        else:
            logger.error(f"Analyzer '{analyzer_name}' not available")
            return False

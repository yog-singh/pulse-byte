"""
VADER sentiment analyzer implementation.
"""

from typing import Dict, Any
from loguru import logger

from src.models import SentimentScore, SentimentLabel
from .sentiment_analyzer import SentimentAnalyzer


class VaderAnalyzer(SentimentAnalyzer):
    """VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analyzer."""
    
    def __init__(self):
        super().__init__("VADER")
        self.analyzer = None
    
    def initialize(self) -> bool:
        """Initialize VADER sentiment analyzer."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
            self.is_initialized = True
            logger.info("VADER sentiment analyzer initialized successfully")
            return True
            
        except ImportError:
            logger.error("vaderSentiment package not found. Install with: pip install vaderSentiment")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize VADER analyzer: {e}")
            return False
    
    def analyze_text(self, text: str) -> SentimentScore:
        """
        Analyze sentiment using VADER.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentScore object
        """
        if not self.is_initialized or not self.analyzer:
            raise RuntimeError("VADER analyzer not initialized")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return SentimentScore(
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                scores={'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0},
                model_used=self.model_name
            )
        
        try:
            # Get VADER scores
            scores = self.analyzer.polarity_scores(processed_text)
            
            # VADER returns: {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
            # compound score is a normalized score between -1 and 1
            compound_score = scores['compound']
            
            # Determine sentiment label based on compound score
            # VADER recommendations: positive >= 0.05, negative <= -0.05, neutral otherwise
            if compound_score >= 0.05:
                label = SentimentLabel.POSITIVE
            elif compound_score <= -0.05:
                label = SentimentLabel.NEGATIVE
            else:
                label = SentimentLabel.NEUTRAL
            
            # Calculate confidence based on the absolute value of compound score
            # and the distribution of individual scores
            confidence = self._calculate_vader_confidence(scores)
            
            # Normalize scores for consistency
            normalized_scores = {
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'compound': compound_score
            }
            
            return SentimentScore(
                label=label,
                confidence=confidence,
                scores=normalized_scores,
                model_used=self.model_name
            )
            
        except Exception as e:
            logger.error(f"Error in VADER analysis: {e}")
            return SentimentScore(
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                scores={'error': str(e)},
                model_used=self.model_name
            )
    
    def _calculate_vader_confidence(self, scores: Dict[str, float]) -> float:
        """
        Calculate confidence score for VADER results.
        
        Args:
            scores: VADER scores dictionary
            
        Returns:
            Confidence value between 0 and 1
        """
        compound = abs(scores['compound'])
        
        # Base confidence on compound score magnitude
        if compound >= 0.5:
            base_confidence = 0.8
        elif compound >= 0.1:
            base_confidence = 0.6
        else:
            base_confidence = 0.4
        
        # Adjust based on score distribution
        # Higher confidence when one sentiment dominates
        pos, neg, neu = scores['pos'], scores['neg'], scores['neu']
        max_individual = max(pos, neg, neu)
        
        if max_individual >= 0.7:
            confidence_adjustment = 0.2
        elif max_individual >= 0.5:
            confidence_adjustment = 0.1
        else:
            confidence_adjustment = 0.0
        
        confidence = min(1.0, base_confidence + confidence_adjustment)
        return confidence
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get VADER model information."""
        info = super().get_model_info()
        info.update({
            'description': 'VADER Sentiment Analysis - lexicon and rule-based sentiment analysis tool',
            'suitable_for': ['social media text', 'informal text', 'mixed case text'],
            'output_range': 'compound score: -1 (most negative) to +1 (most positive)',
            'thresholds': {
                'positive': '≥ 0.05',
                'negative': '≤ -0.05',
                'neutral': '-0.05 < score < 0.05'
            }
        })
        return info
    
    @staticmethod
    def get_feature_importance() -> Dict[str, str]:
        """Get information about VADER's key features."""
        return {
            'punctuation': 'Exclamation marks increase intensity',
            'capitalization': 'ALL CAPS increases intensity',
            'degree_modifiers': 'Words like "very", "extremely" modify intensity',
            'contrastive_conjunctions': 'But, however, etc. affect sentiment',
            'negation': 'Handles negation like "not good"',
            'emoji_support': 'Recognizes emoticons and some emojis'
        }

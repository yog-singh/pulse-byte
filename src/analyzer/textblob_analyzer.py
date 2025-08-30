"""
TextBlob sentiment analyzer implementation.
"""

from typing import Dict, Any
from loguru import logger

from src.models import SentimentScore, SentimentLabel
from .sentiment_analyzer import SentimentAnalyzer


class TextBlobAnalyzer(SentimentAnalyzer):
    """TextBlob sentiment analyzer implementation."""
    
    def __init__(self):
        super().__init__("TextBlob")
        self.textblob = None
    
    def initialize(self) -> bool:
        """Initialize TextBlob sentiment analyzer."""
        try:
            from textblob import TextBlob
            self.textblob = TextBlob
            
            # Test with a simple example to ensure it works
            test_blob = TextBlob("This is a test.")
            _ = test_blob.sentiment
            
            self.is_initialized = True
            logger.info("TextBlob sentiment analyzer initialized successfully")
            return True
            
        except ImportError:
            logger.error("textblob package not found. Install with: pip install textblob")
            logger.info("Also run: python -m textblob.download_corpora")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize TextBlob analyzer: {e}")
            logger.info("You may need to download corpora: python -m textblob.download_corpora")
            return False
    
    def analyze_text(self, text: str) -> SentimentScore:
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentScore object
        """
        if not self.is_initialized or not self.textblob:
            raise RuntimeError("TextBlob analyzer not initialized")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return SentimentScore(
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                scores={'polarity': 0.0, 'subjectivity': 0.0},
                model_used=self.model_name
            )
        
        try:
            # Create TextBlob object
            blob = self.textblob(processed_text)
            
            # Get sentiment scores
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Determine sentiment label
            # TextBlob polarity: > 0 positive, < 0 negative, 0 neutral
            if polarity > 0.1:
                label = SentimentLabel.POSITIVE
            elif polarity < -0.1:
                label = SentimentLabel.NEGATIVE
            else:
                label = SentimentLabel.NEUTRAL
            
            # Calculate confidence
            confidence = self._calculate_textblob_confidence(polarity, subjectivity)
            
            # Prepare scores dictionary
            scores = {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'positive': max(0, polarity),
                'negative': max(0, -polarity),
                'neutral': 1 - abs(polarity) if abs(polarity) <= 1 else 0
            }
            
            return SentimentScore(
                label=label,
                confidence=confidence,
                scores=scores,
                model_used=self.model_name
            )
            
        except Exception as e:
            logger.error(f"Error in TextBlob analysis: {e}")
            return SentimentScore(
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                scores={'error': str(e)},
                model_used=self.model_name
            )
    
    def _calculate_textblob_confidence(self, polarity: float, subjectivity: float) -> float:
        """
        Calculate confidence score for TextBlob results.
        
        Args:
            polarity: Polarity score (-1 to 1)
            subjectivity: Subjectivity score (0 to 1)
            
        Returns:
            Confidence value between 0 and 1
        """
        # Base confidence on polarity magnitude
        polarity_confidence = abs(polarity)
        
        # Higher subjectivity generally means more confident sentiment
        # (objective text might have neutral sentiment by default)
        subjectivity_factor = subjectivity * 0.5
        
        # Combine factors
        confidence = min(1.0, polarity_confidence + subjectivity_factor)
        
        # Boost confidence for strong polarity
        if abs(polarity) >= 0.5:
            confidence = min(1.0, confidence + 0.2)
        
        return confidence
    
    def analyze_text_detailed(self, text: str) -> Dict[str, Any]:
        """
        Perform detailed TextBlob analysis including sentence-level sentiment.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with detailed analysis results
        """
        if not self.is_initialized:
            return {}
        
        try:
            blob = self.textblob(text)
            
            # Overall sentiment
            overall_sentiment = blob.sentiment
            
            # Sentence-level sentiment
            sentences = []
            for i, sentence in enumerate(blob.sentences):
                sent_sentiment = sentence.sentiment
                sentences.append({
                    'index': i,
                    'text': str(sentence),
                    'polarity': sent_sentiment.polarity,
                    'subjectivity': sent_sentiment.subjectivity
                })
            
            # Extract key phrases (noun phrases)
            noun_phrases = list(blob.noun_phrases)
            
            # Basic statistics
            word_count = len(blob.words)
            sentence_count = len(blob.sentences)
            
            return {
                'overall_polarity': overall_sentiment.polarity,
                'overall_subjectivity': overall_sentiment.subjectivity,
                'sentences': sentences,
                'noun_phrases': noun_phrases[:10],  # Limit to first 10
                'word_count': word_count,
                'sentence_count': sentence_count,
                'average_sentence_polarity': sum(s['polarity'] for s in sentences) / len(sentences) if sentences else 0
            }
            
        except Exception as e:
            logger.error(f"Error in detailed TextBlob analysis: {e}")
            return {'error': str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get TextBlob model information."""
        info = super().get_model_info()
        info.update({
            'description': 'TextBlob Sentiment Analysis - based on movie reviews corpus',
            'features': ['polarity (-1 to 1)', 'subjectivity (0 to 1)'],
            'suitable_for': ['formal text', 'reviews', 'general text'],
            'polarity_range': '-1 (negative) to +1 (positive)',
            'subjectivity_range': '0 (objective) to 1 (subjective)',
            'additional_features': ['noun phrase extraction', 'sentence-level analysis']
        })
        return info
    
    @staticmethod
    def interpret_subjectivity(subjectivity: float) -> str:
        """
        Interpret subjectivity score.
        
        Args:
            subjectivity: Subjectivity score from 0 to 1
            
        Returns:
            Human-readable interpretation
        """
        if subjectivity >= 0.8:
            return "Very subjective (opinion-based)"
        elif subjectivity >= 0.6:
            return "Mostly subjective"
        elif subjectivity >= 0.4:
            return "Somewhat subjective"
        elif subjectivity >= 0.2:
            return "Mostly objective"
        else:
            return "Very objective (fact-based)"

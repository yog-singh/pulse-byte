"""
Transformers-based sentiment analyzer implementation.
"""

from typing import Dict, Any, List
import torch
from loguru import logger

from src.models import SentimentScore, SentimentLabel
from .sentiment_analyzer import SentimentAnalyzer
from config.settings import SENTIMENT_CONFIG


class TransformersAnalyzer(SentimentAnalyzer):
    """Transformers-based sentiment analyzer using Hugging Face models."""
    
    def __init__(self, model_name: str = None):
        self.model_name_hf = model_name or SENTIMENT_CONFIG.get('transformers_model', 'cardiffnlp/twitter-roberta-base-sentiment-latest')
        super().__init__(f"Transformers-{self.model_name_hf.split('/')[-1]}")
        self.pipeline = None
        self.tokenizer = None
        self.model = None
    
    def initialize(self) -> bool:
        """Initialize transformers sentiment analyzer."""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            
            logger.info(f"Initializing transformers model: {self.model_name_hf}")
            
            # Check if CUDA is available
            device = 0 if torch.cuda.is_available() else -1
            
            # Initialize the sentiment analysis pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name_hf,
                device=device,
                return_all_scores=True
            )
            
            # Also load tokenizer and model separately for more control
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_hf)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_hf)
            
            self.is_initialized = True
            logger.info(f"Transformers sentiment analyzer initialized successfully on device: {'GPU' if device >= 0 else 'CPU'}")
            return True
            
        except ImportError:
            logger.error("transformers package not found. Install with: pip install transformers torch")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize transformers analyzer: {e}")
            return False
    
    def analyze_text(self, text: str) -> SentimentScore:
        """
        Analyze sentiment using transformers model.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentScore object
        """
        if not self.is_initialized or not self.pipeline:
            raise RuntimeError("Transformers analyzer not initialized")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return SentimentScore(
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                scores={},
                model_used=self.model_name
            )
        
        try:
            # Handle long texts by chunking
            max_length = self.tokenizer.model_max_length if self.tokenizer else 512
            chunks = self._chunk_text(processed_text, max_length)
            
            if len(chunks) == 1:
                # Single chunk
                result = self._analyze_chunk(chunks[0])
            else:
                # Multiple chunks - average the results
                result = self._analyze_chunks(chunks)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in transformers analysis: {e}")
            return SentimentScore(
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                scores={'error': str(e)},
                model_used=self.model_name
            )
    
    def _chunk_text(self, text: str, max_length: int) -> List[str]:
        """
        Split long text into chunks that fit the model's input size.
        
        Args:
            text: Text to chunk
            max_length: Maximum tokens per chunk
            
        Returns:
            List of text chunks
        """
        if not self.tokenizer:
            # Fallback to simple word-based chunking
            words = text.split()
            # Assume average 1.3 tokens per word
            words_per_chunk = int(max_length * 0.7)
            return [' '.join(words[i:i + words_per_chunk]) 
                   for i in range(0, len(words), words_per_chunk)]
        
        # Tokenize text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Split into chunks with overlap
        chunk_size = max_length - 2  # Account for special tokens
        overlap = 50  # Overlap between chunks
        
        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
        
        return chunks
    
    def _analyze_chunk(self, text: str) -> SentimentScore:
        """Analyze sentiment for a single text chunk."""
        results = self.pipeline(text)
        
        # results[0] contains all scores for the text
        scores = {result['label'].lower(): result['score'] for result in results[0]}
        
        # Find the label with highest score
        best_result = max(results[0], key=lambda x: x['score'])
        best_label = best_result['label'].lower()
        confidence = best_result['score']
        
        # Map model labels to our sentiment labels
        sentiment_label = self._map_label_to_sentiment(best_label)
        
        return SentimentScore(
            label=sentiment_label,
            confidence=confidence,
            scores=scores,
            model_used=self.model_name
        )
    
    def _analyze_chunks(self, chunks: List[str]) -> SentimentScore:
        """Analyze sentiment for multiple chunks and combine results."""
        chunk_results = []
        
        for chunk in chunks:
            try:
                result = self._analyze_chunk(chunk)
                chunk_results.append(result)
            except Exception as e:
                logger.warning(f"Error analyzing chunk: {e}")
                continue
        
        if not chunk_results:
            return SentimentScore(
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                scores={},
                model_used=self.model_name
            )
        
        # Combine results by averaging scores
        combined_scores = {}
        total_confidence = 0
        
        # Get all unique labels
        all_labels = set()
        for result in chunk_results:
            all_labels.update(result.scores.keys())
        
        # Average scores for each label
        for label in all_labels:
            scores_for_label = [result.scores.get(label, 0) for result in chunk_results]
            combined_scores[label] = sum(scores_for_label) / len(scores_for_label)
        
        # Average confidence
        total_confidence = sum(result.confidence for result in chunk_results) / len(chunk_results)
        
        # Determine overall sentiment
        if combined_scores:
            best_label = max(combined_scores.keys(), key=lambda k: combined_scores[k])
            sentiment_label = self._map_label_to_sentiment(best_label)
        else:
            sentiment_label = SentimentLabel.NEUTRAL
        
        return SentimentScore(
            label=sentiment_label,
            confidence=total_confidence,
            scores=combined_scores,
            model_used=self.model_name
        )
    
    def _map_label_to_sentiment(self, model_label: str) -> SentimentLabel:
        """
        Map model-specific labels to our standard sentiment labels.
        
        Args:
            model_label: Label from the transformers model
            
        Returns:
            Corresponding SentimentLabel
        """
        model_label = model_label.lower()
        
        # Common mappings for different models
        positive_labels = ['positive', 'pos', 'label_2', '2']
        negative_labels = ['negative', 'neg', 'label_0', '0']
        neutral_labels = ['neutral', 'label_1', '1']
        
        if model_label in positive_labels:
            return SentimentLabel.POSITIVE
        elif model_label in negative_labels:
            return SentimentLabel.NEGATIVE
        elif model_label in neutral_labels:
            return SentimentLabel.NEUTRAL
        else:
            # Try to guess based on common patterns
            if 'pos' in model_label or 'good' in model_label:
                return SentimentLabel.POSITIVE
            elif 'neg' in model_label or 'bad' in model_label:
                return SentimentLabel.NEGATIVE
            else:
                logger.warning(f"Unknown label '{model_label}', defaulting to neutral")
                return SentimentLabel.NEUTRAL
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get transformers model information."""
        info = super().get_model_info()
        info.update({
            'huggingface_model': self.model_name_hf,
            'description': 'Transformer-based sentiment analysis using pre-trained models',
            'suitable_for': ['various text types depending on model training'],
            'features': ['high accuracy', 'contextual understanding', 'transfer learning'],
            'device': 'GPU' if torch.cuda.is_available() and self.pipeline else 'CPU'
        })
        
        if self.tokenizer:
            info['max_length'] = self.tokenizer.model_max_length
            info['vocab_size'] = self.tokenizer.vocab_size
        
        return info
    
    def analyze_batch(self, texts: List[str], batch_size: int = 8) -> List[SentimentScore]:
        """
        Analyze multiple texts in batches for efficiency.
        
        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process at once
            
        Returns:
            List of SentimentScore objects
        """
        if not self.is_initialized:
            return [SentimentScore(SentimentLabel.NEUTRAL, 0.0, {}, self.model_name) for _ in texts]
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []
            
            try:
                # Process batch
                pipeline_results = self.pipeline(batch)
                
                for j, text_results in enumerate(pipeline_results):
                    try:
                        scores = {result['label'].lower(): result['score'] for result in text_results}
                        best_result = max(text_results, key=lambda x: x['score'])
                        
                        sentiment_score = SentimentScore(
                            label=self._map_label_to_sentiment(best_result['label']),
                            confidence=best_result['score'],
                            scores=scores,
                            model_used=self.model_name
                        )
                        batch_results.append(sentiment_score)
                        
                    except Exception as e:
                        logger.error(f"Error processing batch item {i+j}: {e}")
                        batch_results.append(SentimentScore(
                            SentimentLabel.NEUTRAL, 0.0, {'error': str(e)}, self.model_name
                        ))
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size}: {e}")
                # Add neutral results for failed batch
                for _ in batch:
                    batch_results.append(SentimentScore(
                        SentimentLabel.NEUTRAL, 0.0, {'error': str(e)}, self.model_name
                    ))
            
            results.extend(batch_results)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {len(results)}/{len(texts)} texts")
        
        return results

import os
import re
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from config import Config

class EventExtractor:
    """Extract combat death events from articles."""
    
    def __init__(self, model_name=None):
        self.config = Config()
        self.model_name = model_name or self.config.EVENT_EXTRACTION_MODEL
        self.model = None
        self.tokenizer = None
        self.model_path = os.path.join(self.config.MODEL_DIR, "event_extractor")
        
        # Patterns for event extraction
        self.death_patterns = [
            # Country A killed/murdered Country B soldiers/troops
            r'(\b(?:' + '|'.join([re.escape(c) for c in self.config.DEATH_KEYWORDS]) + r')\b)',
            
            # X military personnel died/killed/casualties
            r'\b(?:' + '|'.join(self.config.MILITARY_KEYWORDS) + r')\b.{0,50}\b(?:' + 
            '|'.join([re.escape(c) for c in self.config.DEATH_KEYWORDS]) + r')\b',
            
            # died/killed/casualties ... X military personnel
            r'\b(?:' + '|'.join([re.escape(c) for c in self.config.DEATH_KEYWORDS]) + r')\b.{0,50}\b(?:' + 
            '|'.join(self.config.MILITARY_KEYWORDS) + r')\b'
        ]
        
        # Compile the patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.death_patterns]
    
    def train(self, train_texts, train_labels):
        """Train the event extractor model."""
        # Note: In a real implementation, I would recomend fine-tuning this model for this task
        # For simplicity, I will use a rule-based approach combined with pre-trained models
        pass
    
    def load_model(self):
        """Load a trained model or use a pre-trained model."""
        if os.path.exists(self.model_path):
            # Load fine-tuned model if it exists
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        else:
            # Use pre-trained model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
    
    def _extract_date_mentions(self, text):
        """Extract date mentions from text."""
        # Basic pattern for dates (e.g., January 15, 2023 or 15 January 2023 or 2023-01-15)
        date_patterns = [
            r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b'
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                dates.append(match.group(0))
                
        return dates
    
    def _extract_death_events(self, text):
        """Extract death events from text using patterns."""
        events = []
        
        # Find all matches using the compiled patterns
        for pattern in self.compiled_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                # Extract the sentence containing the match
                start_pos = max(0, text.rfind('.', 0, match.start()) + 1)
                end_pos = text.find('.', match.end())
                if end_pos == -1:
                    end_pos = len(text)
                
                event_text = text[start_pos:end_pos].strip()
                events.append(event_text)
        
        return events
    
    def extract_events(self, articles_df):
        """Extract combat death events from articles."""
        # Create new columns for events and dates
        articles_df['death_events'] = None
        articles_df['event_dates'] = None
        
        # Process each article
        for idx, row in articles_df.iterrows():
            text = row['full_text']
            
            # Extract death events
            events = self._extract_death_events(text)
            articles_df.at[idx, 'death_events'] = events
            
            # Extract dates
            dates = self._extract_date_mentions(text)
            articles_df.at[idx, 'event_dates'] = dates
        
        # Filter to keep only articles with identified death events
        filtered_df = articles_df[articles_df['death_events'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)].copy()
        
        return filtered_df 
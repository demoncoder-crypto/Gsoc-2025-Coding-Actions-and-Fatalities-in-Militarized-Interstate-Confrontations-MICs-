import os
import re
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from config import Config

class EntityExtractor:
    """Extract countries and military entities from text."""
    
    def __init__(self, model_name=None):
        self.config = Config()
        self.model_name = model_name or self.config.NER_MODEL
        self.ner_pipeline = None
        self.model_path = os.path.join(self.config.MODEL_DIR, "entity_extractor")
        self.countries_list = None
    
    def load_countries(self, data_loader):
        """Load country names from state system file."""
        self.countries_list = data_loader.get_country_names()
        # Create a dictionary to normalize country names (e.g., "United States" -> "United States of America")
        self.country_aliases = self._create_country_aliases()
    
    def _create_country_aliases(self):
        """Create aliases for countries to handle common variations."""
        aliases = {}
        for country in self.countries_list:
            # Add the original name
            aliases[country.lower()] = country
            
            # Add common variations for specific countries
            if country == "United States of America":
                aliases["united states"] = country
                aliases["usa"] = country
                aliases["u.s."] = country
                aliases["u.s"] = country
                aliases["us"] = country
                aliases["america"] = country
            elif country == "United Kingdom":
                aliases["uk"] = country
                aliases["u.k."] = country
                aliases["great britain"] = country
                aliases["britain"] = country
            elif country == "Russian Federation":
                aliases["russia"] = country
            # Add more common variations as needed
            
        return aliases
    
    def train(self, train_texts, train_labels):
        """Train the entity extractor model."""
        # Note: In a real implementation, recomended use is fine-tune the NER model
        # For simplicity, I am using a pre-trained model and focus on the rule-based components
        pass
        
    def load_model(self):
        """Load a trained model or use a pre-trained model."""
        if os.path.exists(self.model_path):
            # Load fine-tuned model if it exists
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            self.ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
        else:
            # Use pre-trained model
            self.ner_pipeline = pipeline("ner", model=self.model_name, aggregation_strategy="simple")
    
    def _identify_country_mentions(self, text):
        """Identify country mentions in text using rule-based approach."""
        countries_found = set()
        
        # Convert text to lowercase for matching
        text_lower = text.lower()
        
        # Check for each country and its aliases
        for alias, country in self.country_aliases.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, text_lower):
                countries_found.add(country)
        
        return list(countries_found)
    
    def _identify_military_entities(self, text):
        """Identify military entities in text using NER and rules."""
        if self.ner_pipeline is None:
            self.load_model()
        
        # Get NER predictions
        try:
            ner_results = self.ner_pipeline(text)
        except Exception as e:
            print(f"Error in NER pipeline: {e}")
            ner_results = []
        
        # Filter for organization entities that might be military organizations
        military_orgs = []
        for entity in ner_results:
            if entity['entity_group'] == 'ORG':
                entity_text = entity['word']
                # Check if entity contains military keywords
                if any(keyword in entity_text.lower() for keyword in self.config.MILITARY_KEYWORDS):
                    military_orgs.append(entity_text)
        
        # Also use rule-based approach to find military keyword phrases
        military_pattern = '|'.join(self.config.MILITARY_KEYWORDS)
        military_matches = re.finditer(f"\\b(\\w+\\s+)?({military_pattern})(\\s+\\w+)?\\b", text, re.IGNORECASE)
        
        for match in military_matches:
            military_orgs.append(match.group(0))
        
        return list(set(military_orgs))
    
    def extract_entities(self, articles_df):
        """Extract countries and military entities from articles."""
        if self.countries_list is None:
            raise ValueError("Country list not loaded. Call load_countries() first.")
        
        # Create new columns for extracted entities
        articles_df['countries_mentioned'] = None
        articles_df['military_entities'] = None
        
        # Process each article
        for idx, row in articles_df.iterrows():
            text = row['full_text']
            
            # Extract countries
            countries = self._identify_country_mentions(text)
            articles_df.at[idx, 'countries_mentioned'] = countries
            
            # Extract military entities
            military_entities = self._identify_military_entities(text)
            articles_df.at[idx, 'military_entities'] = military_entities
        
        return articles_df 
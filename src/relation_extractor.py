import os
import re
import torch
import numpy as np
import pandas as pd
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from config import Config

class RelationExtractor:
    """Extract relationships between countries in combat death events."""
    
    def __init__(self, model_name=None):
        self.config = Config()
        self.model_name = model_name or self.config.RELATION_MODEL
        self.model = None
        self.tokenizer = None
        self.model_path = os.path.join(self.config.MODEL_DIR, "relation_extractor")
        
        # Try to load spaCy model for dependency parsing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = None
            print("SpaCy model not found. Using rule-based approach only.")
        
        # Keywords that indicate a country's role as aggressor
        self.aggressor_verbs = [
            "killed", "attacked", "bombed", "shot", "fired", "launched", "invaded",
            "assaulted", "ambushed", "targeted", "struck", "hit", "destroyed",
            "shelled", "raided", "assassinated", "eliminated", "bombarded"
        ]
        
        # Keywords that indicate a country's role as victim
        self.victim_verbs = [
            "died", "perished", "killed", "murdered", "lost", "suffered", "casualty",
            "casualties", "fatality", "fatalities", "victim", "victims", "wounded",
            "injured", "fallen", "massacre", "slain", "defeated"
        ]
        
        # Patterns to help with determining the aggressor/victim relationship
        self.aggressor_patterns = [
            r'(COUNTRY1).*?(?:' + '|'.join(self.aggressor_verbs) + r').*?(COUNTRY2)',
            r'(COUNTRY1).*?troops.*?(?:' + '|'.join(self.aggressor_verbs) + r').*?(COUNTRY2)',
            r'(COUNTRY1).*?forces.*?(?:' + '|'.join(self.aggressor_verbs) + r').*?(COUNTRY2)',
            r'(COUNTRY1).*?military.*?(?:' + '|'.join(self.aggressor_verbs) + r').*?(COUNTRY2)'
        ]
        
        # Precompile patterns for efficiency
        self.compiled_aggressor_patterns = []
    
    def train(self, train_texts, train_labels):
        """Train the relation extractor model."""
        # Note: In a real implementation, we would fine-tune a model for this task
        # For simplicity, we'll use a rule-based approach combined with pre-trained models
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
    
    def _preprocess_countries_for_patterns(self, countries):
        """Prepare country names for pattern matching."""
        if not countries:
            return []
            
        # Precompile patterns for each country pair
        compiled_patterns = []
        for i, country1 in enumerate(countries):
            for j, country2 in enumerate(countries):
                if i != j:  # Don't match country against itself
                    for pattern in self.aggressor_patterns:
                        # Replace placeholders with actual country names
                        country_pattern = pattern.replace('COUNTRY1', re.escape(country1)).replace('COUNTRY2', re.escape(country2))
                        compiled_patterns.append((country1, country2, re.compile(country_pattern, re.IGNORECASE)))
                        
        return compiled_patterns
    
    def _identify_country_roles_with_spacy(self, event_text, countries):
        """Use spaCy's dependency parsing to identify roles."""
        if self.nlp is None or not countries or len(countries) < 2:
            return None, None
            
        # Parse the text
        doc = self.nlp(event_text)
        
        # Initialize role scores
        country_roles = {country: {'aggressor_score': 0, 'victim_score': 0} for country in countries}
        
        # Find mentions of countries
        country_mentions = {}
        for country in countries:
            country_mentions[country] = []
            # Find exact mentions and their indices
            for match in re.finditer(r'\b' + re.escape(country) + r'\b', event_text, re.IGNORECASE):
                start, end = match.span()
                country_mentions[country].append((start, end))
        
        # For each sentence, analyze the dependency structure
        for sent in doc.sents:
            # Find verbs indicating aggression or victimhood
            for token in sent:
                if token.pos_ == "VERB" and token.lemma_.lower() in self.aggressor_verbs + self.victim_verbs:
                    verb = token.lemma_.lower()
                    
                    # Find the subject and object of this verb
                    subj = None
                    obj = None
                    
                    for child in token.children:
                        if child.dep_ in ("nsubj", "nsubjpass") and not subj:
                            subj = child
                        elif child.dep_ in ("dobj", "pobj", "iobj") and not obj:
                            obj = child
                    
                    # If found both subject and object, check if they match countries
                    if subj and obj:
                        # Check each country to see if it matches subject or object
                        for country in countries:
                            # Check if country is in the subject span
                            if country.lower() in subj.subtree.text.lower():
                                # If the verb is an aggressor verb, country is the aggressor
                                if verb in self.aggressor_verbs:
                                    country_roles[country]['aggressor_score'] += 10
                                # If the verb is a victim verb, country is the victim
                                elif verb in self.victim_verbs:
                                    country_roles[country]['victim_score'] += 10
                                    
                            # Check if country is in the object span
                            if country.lower() in obj.subtree.text.lower():
                                # If the verb is an aggressor verb, country is the victim
                                if verb in self.aggressor_verbs:
                                    country_roles[country]['victim_score'] += 10
                                # If the verb is a victim verb, country is contextually related
                                elif verb in self.victim_verbs:
                                    country_roles[country]['victim_score'] += 5
        
        # Determine aggressor and victim based on scores
        aggressor = max(country_roles.items(), key=lambda x: x[1]['aggressor_score'])
        victim = max(country_roles.items(), key=lambda x: x[1]['victim_score'])
        
        # Only return values if the scores are non-zero
        aggressor_country = aggressor[0] if aggressor[1]['aggressor_score'] > 0 else None
        victim_country = victim[0] if victim[1]['victim_score'] > 0 else None
        
        # If they're the same, choose based on which score is higher
        if aggressor_country == victim_country:
            if aggressor[1]['aggressor_score'] > victim[1]['victim_score']:
                victim_country = None
            else:
                aggressor_country = None
        
        return aggressor_country, victim_country
    
    def _identify_country_roles_with_patterns(self, event_text, countries):
        """Identify roles using pattern matching on country pairs."""
        if not countries or len(countries) < 2:
            return None, None
            
        # Create patterns for all country pairs
        compiled_patterns = self._preprocess_countries_for_patterns(countries)
        
        # Match patterns
        matches = []
        for country1, country2, pattern in compiled_patterns:
            if pattern.search(event_text):
                matches.append((country1, country2))
        
        # Count matches to determine most likely aggressor and victim
        aggressor_counts = {}
        victim_counts = {}
        
        for aggressor, victim in matches:
            aggressor_counts[aggressor] = aggressor_counts.get(aggressor, 0) + 1
            victim_counts[victim] = victim_counts.get(victim, 0) + 1
        
        # Determine most frequent aggressor and victim
        aggressor_country = max(aggressor_counts.items(), key=lambda x: x[1])[0] if aggressor_counts else None
        victim_country = max(victim_counts.items(), key=lambda x: x[1])[0] if victim_counts else None
        
        # Make sure aggressor and victim aren't the same
        if aggressor_country == victim_country:
            # If they're the same, use the one with higher count
            if aggressor_counts[aggressor_country] > victim_counts[victim_country]:
                victim_country = None
            else:
                aggressor_country = None
                
        return aggressor_country, victim_country
    
    def _identify_country_roles(self, event_text, countries):
        """Identify aggressor and victim countries in a death event text."""
        if not countries or len(countries) < 2:
            return None, None
        
        event_text_lower = event_text.lower()
        
        # Try to identify using spaCy first (more accurate but may fail)
        aggressor_spacy, victim_spacy = self._identify_country_roles_with_spacy(event_text, countries)
        
        # Try to identify using patterns
        aggressor_pattern, victim_pattern = self._identify_country_roles_with_patterns(event_text, countries)
        
        # Initialize role scores for position-based analysis
        country_roles = {country: {'aggressor_score': 0, 'victim_score': 0} for country in countries}
        
        # Calculate role scores based on proximity to role-indicating verbs
        for country in countries:
            country_lower = country.lower()
            # Skip if country is not in the event text
            if country_lower not in event_text_lower:
                continue
            
            # Calculate proximity to aggressor verbs
            for verb in self.aggressor_verbs:
                if verb in event_text_lower:
                    # Calculate distance between country and verb
                    country_pos = event_text_lower.find(country_lower)
                    verb_pos = event_text_lower.find(verb)
                    distance = abs(country_pos - verb_pos)
                    
                    # Check if country appears before verb (typical for subject-verb-object)
                    if country_pos < verb_pos:
                        score = max(0, 100 - distance)
                    else:
                        score = max(0, 50 - distance)
                        
                    country_roles[country]['aggressor_score'] += score
            
            # Calculate proximity to victim verbs
            for verb in self.victim_verbs:
                if verb in event_text_lower:
                    # Calculate distance between country and verb
                    country_pos = event_text_lower.find(country_lower)
                    verb_pos = event_text_lower.find(verb)
                    distance = abs(country_pos - verb_pos)
                    
                    # Check if country appears after verb (typical for subject-verb-object)
                    if country_pos > verb_pos:
                        score = max(0, 100 - distance)
                    else:
                        score = max(0, 50 - distance)
                        
                    country_roles[country]['victim_score'] += score
        
        # Determine aggressor and victim based on scores
        aggressor_proximity = max(country_roles.items(), key=lambda x: x[1]['aggressor_score'])[0] if country_roles else None
        victim_proximity = max(country_roles.items(), key=lambda x: x[1]['victim_score'])[0] if country_roles else None
        
        # Combine results from different methods with priority
        aggressor = aggressor_spacy or aggressor_pattern or aggressor_proximity
        victim = victim_spacy or victim_pattern or victim_proximity
        
        # If aggressor and victim are the same country, set the one with the lower score to None
        if aggressor == victim:
            aggressor_score = country_roles[aggressor]['aggressor_score'] if aggressor in country_roles else 0
            victim_score = country_roles[victim]['victim_score'] if victim in country_roles else 0
            
            if aggressor_score > victim_score:
                victim = None
            else:
                aggressor = None
        
        return aggressor, victim
    
    def extract_relations(self, articles_df):
        """Extract relations between countries in death events."""
        # Create new columns for aggressor and victim countries
        articles_df['aggressor_country'] = None
        articles_df['victim_country'] = None
        
        # Process each article
        for idx, row in articles_df.iterrows():
            countries = row['countries_mentioned']
            death_events = row['death_events']
            
            if not isinstance(countries, list) or not isinstance(death_events, list) or len(countries) < 2 or len(death_events) == 0:
                continue
            
            # Process each death event
            all_aggressors = []
            all_victims = []
            
            for event_text in death_events:
                aggressor, victim = self._identify_country_roles(event_text, countries)
                if aggressor:
                    all_aggressors.append(aggressor)
                if victim:
                    all_victims.append(victim)
            
            # Determine the most frequent aggressor and victim
            if all_aggressors:
                aggressor_counts = {country: all_aggressors.count(country) for country in set(all_aggressors)}
                articles_df.at[idx, 'aggressor_country'] = max(aggressor_counts.items(), key=lambda x: x[1])[0]
            
            if all_victims:
                victim_counts = {country: all_victims.count(country) for country in set(all_victims)}
                articles_df.at[idx, 'victim_country'] = max(victim_counts.items(), key=lambda x: x[1])[0]
        
        # Filter to keep only articles with identified aggressor and victim
        filtered_df = articles_df[
            articles_df['aggressor_country'].notna() & 
            articles_df['victim_country'].notna()
        ].copy()
        
        return filtered_df 
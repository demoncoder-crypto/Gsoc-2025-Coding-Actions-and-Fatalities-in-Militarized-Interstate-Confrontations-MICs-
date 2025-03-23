import os
import re
import torch
import numpy as np
import pandas as pd
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from config import Config

class NumericExtractor:
    """Extract numeric information about fatalities from articles."""
    
    def __init__(self, model_name=None):
        self.config = Config()
        self.model_name = model_name or self.config.NER_MODEL
        self.ner_pipeline = None
        self.model_path = os.path.join(self.config.MODEL_DIR, "numeric_extractor")
        
        # Load spaCy model for better entity recognition and number parsing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If model not found, use a simpler approach
            self.nlp = None
            print("SpaCy model not found. Using regex-based approach only.")
        
        # Define patterns for numeric extraction
        self.number_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
            'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
            'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000, 'million': 1000000
        }
        
        # Patterns for numeric extraction - revised to be more precise
        self.numeric_patterns = [
            # Exact numbers of casualties: "X soldiers were killed"
            r'(\d{1,4}|' + '|'.join(self.number_words.keys()) + r')\s+(?:' + 
            '|'.join(self.config.MILITARY_KEYWORDS) + r')\s+(?:\w+\s+){0,3}(?:' + 
            '|'.join([re.escape(c) for c in self.config.DEATH_KEYWORDS]) + r')',
            
            # Ranges: "between X and Y soldiers were killed"
            r'between\s+(\d{1,4}|' + '|'.join(self.number_words.keys()) + r')\s+and\s+(\d{1,4}|' + 
            '|'.join(self.number_words.keys()) + r')\s+(?:' + '|'.join(self.config.MILITARY_KEYWORDS) + r')',
            
            # Approximate: "approximately/about/around X soldiers"
            r'(?:approximately|about|around|nearly|almost|over|more than|at least|up to)\s+(\d{1,4}|' + 
            '|'.join(self.number_words.keys()) + r')\s+(?:' + '|'.join(self.config.MILITARY_KEYWORDS) + r')',
            
            # Death toll: "death toll of X"
            r'(?:death toll|fatalities|casualties)\s+(?:of|reached|rose to|climbed to)\s+(\d{1,4}|' + 
            '|'.join(self.number_words.keys()) + r')',
            
            # "X people/individuals died/were killed"
            r'(\d{1,4}|' + '|'.join(self.number_words.keys()) + r')\s+(?:people|individuals|persons)\s+(?:\w+\s+){0,3}(?:' + 
            '|'.join([re.escape(c) for c in self.config.DEATH_KEYWORDS]) + r')'
        ]
        
        # Compile the patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.numeric_patterns]
    
    def load_model(self):
        """Load a trained model or use a pre-trained model."""
        if os.path.exists(self.model_path):
            # Load fine-tuned model if it exists
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            self.ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
        else:
            # Use pre-trained model for NER to help with numeric extraction
            self.ner_pipeline = pipeline("ner", model=self.model_name, aggregation_strategy="simple")
    
    def _convert_word_to_number(self, word):
        """Convert a word representation of a number to an integer."""
        word = word.lower()
        if word in self.number_words:
            return self.number_words[word]
        try:
            return int(word)
        except ValueError:
            return None
    
    def _is_reasonable_fatality_number(self, num):
        """Check if a number is a reasonable fatality count for an interstate military conflict."""
        # Most conflicts have casualties in the range of 1-10,000
        # Very large numbers are likely to be false positives
        return 1 <= num <= 10000
    
    def _extract_numbers_from_text(self, text):
        """Extract numbers from text using regex patterns."""
        numbers = []
        
        # Find all matches using the compiled patterns
        for pattern in self.compiled_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                # Extract the full match to get context
                full_match = match.group(0)
                
                # Extract all groups that might contain numbers
                for i in range(1, len(match.groups()) + 1):
                    try:
                        num_text = match.group(i)
                        num = self._convert_word_to_number(num_text)
                        if num is not None and self._is_reasonable_fatality_number(num):
                            # Store the number along with its context
                            numbers.append({
                                'number': num,
                                'context': full_match
                            })
                    except IndexError:
                        continue
        
        return numbers
    
    def _extract_numbers_with_spacy(self, text):
        """Extract numbers using spaCy's sophisticated NLP capabilities."""
        if self.nlp is None:
            return []
        
        numbers = []
        doc = self.nlp(text)
        
        # Check sentences with death-related terms for numeric entities
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if any(keyword in sent_text for keyword in self.config.DEATH_KEYWORDS) and \
               any(keyword in sent_text for keyword in self.config.MILITARY_KEYWORDS):
                
                # Look for numeric entities in this sentence
                for ent in sent:
                    if ent.ent_type_ in ["CARDINAL", "QUANTITY"] or ent.like_num:
                        try:
                            # Try to extract the number
                            num_text = ent.text
                            if num_text.isdigit():
                                num = int(num_text)
                            elif num_text.lower() in self.number_words:
                                num = self.number_words[num_text.lower()]
                            else:
                                continue
                                
                            if self._is_reasonable_fatality_number(num):
                                numbers.append({
                                    'number': num,
                                    'context': sent.text
                                })
                        except (ValueError, AttributeError):
                            continue
        
        return numbers
    
    def _extract_numbers_with_ner(self, text):
        """Extract numbers using NER to supplement regex patterns."""
        if self.ner_pipeline is None:
            self.load_model()
        
        numbers = []
        try:
            # Get NER predictions
            ner_results = self.ner_pipeline(text)
            
            # Filter for numeric entities
            for entity in ner_results:
                if entity['entity_group'] in ['CARDINAL', 'QUANTITY']:
                    num_text = entity['word']
                    
                    # Try to extract a number from the entity
                    try:
                        num = int(re.search(r'\d+', num_text).group(0))
                        
                        # Only consider reasonable fatality numbers
                        if not self._is_reasonable_fatality_number(num):
                            continue
                            
                        # Get context by extracting the sentence containing the number
                        start_pos = entity['start']
                        end_pos = entity['end']
                        
                        # Find sentence boundaries
                        sent_start = max(0, text.rfind('.', 0, start_pos) + 1)
                        sent_end = text.find('.', end_pos)
                        if sent_end == -1:
                            sent_end = len(text)
                        
                        context = text[sent_start:sent_end].strip()
                        
                        # Check if context contains death-related and military keywords
                        if any(keyword in context.lower() for keyword in self.config.DEATH_KEYWORDS) and \
                           any(keyword in context.lower() for keyword in self.config.MILITARY_KEYWORDS):
                            numbers.append({
                                'number': num,
                                'context': context
                            })
                    except (ValueError, AttributeError):
                        continue
        except Exception as e:
            print(f"Error in NER pipeline for numeric extraction: {e}")
        
        return numbers
    
    def _determine_fatality_range(self, numbers):
        """Determine a range of fatalities based on extracted numbers."""
        if not numbers:
            return 0, 0
        
        # Sort numbers by value
        sorted_numbers = sorted([n['number'] for n in numbers])
        
        # Validate the range: if max is more than 100x min, use a more conservative range
        if len(sorted_numbers) >= 2 and sorted_numbers[-1] > sorted_numbers[0] * 100:
            # Find the median
            median = sorted_numbers[len(sorted_numbers) // 2]
            # Use more conservative range based on median
            lower_bound = max(1, int(median * 0.5))
            upper_bound = int(median * 1.5)
            return lower_bound, upper_bound
        
        if len(sorted_numbers) == 1:
            # If only one number, use it as both min and max
            return sorted_numbers[0], sorted_numbers[0]
        else:
            # Use multiple numbers to create a range
            # We'll use a more conservative approach - 25th and 75th percentiles
            lower_idx = max(0, int(len(sorted_numbers) * 0.25))
            upper_idx = min(len(sorted_numbers) - 1, int(len(sorted_numbers) * 0.75))
            return sorted_numbers[lower_idx], sorted_numbers[upper_idx]
    
    def extract_fatality_counts(self, articles_df):
        """Extract fatality counts from articles."""
        # Create new columns for fatality counts
        articles_df['fatality_min'] = 0
        articles_df['fatality_max'] = 0
        articles_df['fatality_contexts'] = None
        
        # Process each article
        for idx, row in articles_df.iterrows():
            text = row['full_text']
            
            # Extract numbers using regex patterns
            regex_numbers = self._extract_numbers_from_text(text)
            
            # Extract numbers using spaCy (if available)
            spacy_numbers = self._extract_numbers_with_spacy(text) if self.nlp else []
            
            # Extract numbers using NER (supplement)
            ner_numbers = self._extract_numbers_with_ner(text)
            
            # Combine results (using set to avoid duplicates)
            all_numbers = regex_numbers + spacy_numbers + ner_numbers
            
            # Skip if no numbers found
            if not all_numbers:
                continue
            
            # Determine fatality range
            min_fatalities, max_fatalities = self._determine_fatality_range(all_numbers)
            
            # Update the dataframe
            articles_df.at[idx, 'fatality_min'] = min_fatalities
            articles_df.at[idx, 'fatality_max'] = max_fatalities
            articles_df.at[idx, 'fatality_contexts'] = [n['context'] for n in all_numbers]
        
        # Filter to keep only articles with fatality counts
        filtered_df = articles_df[articles_df['fatality_max'] > 0].copy()
        
        return filtered_df 
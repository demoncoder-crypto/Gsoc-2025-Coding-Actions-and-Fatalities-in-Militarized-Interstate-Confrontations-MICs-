import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from config import Config
from data_loader import DataLoader
from validation import ValidationDataset

class LabeledDataGenerator:
    """Generate a small labeled dataset for training models."""
    
    def __init__(self):
        self.config = Config()
        self.data_loader = DataLoader()
        self.validation_dataset = ValidationDataset()
        
        # Output path for labeled data
        self.output_dir = os.path.join(self.config.OUTPUT_DIR, "labeled_data")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_document_classifier_data(self, num_examples=100):
        """Generate labeled data for document classification."""
        print("Loading articles...")
        articles_df = self.data_loader.load_articles()
        print(f"Loaded {len(articles_df)} articles.")
        
        # Pre-filter to get articles with military and death terms
        print("Pre-filtering articles...")
        filtered_df = self.data_loader.prefilter_articles(articles_df)
        print(f"Pre-filtered to {len(filtered_df)} articles.")
        
        # Sample articles
        sample_size = min(num_examples, len(filtered_df))
        sample_df = filtered_df.sample(sample_size)
        
        # Get text for labeling
        texts = []
        for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Processing articles"):
            texts.append({
                'text': row['full_text'],
                'title': row['title'],
                'date': row['date'],
                'source_file': row['source_file']
            })
        
        # Add positive examples from validation dataset
        known_conflicts = self.validation_dataset.get_validation_df()
        
        # For demonstration purposes, create synthetic examples for known conflicts
        for _, conflict in known_conflicts.iterrows():
            # Create a synthetic article about this conflict
            aggressor = conflict['aggressor']
            victim = conflict['victim']
            date = conflict['date']
            fatalities = conflict['fatalities']
            description = conflict['description']
            
            text = f"""
            On {date}, {description}. The confrontation between {aggressor} and {victim} 
            resulted in {fatalities} military casualties. The {victim} military reported 
            that forces from {aggressor} conducted operations leading to fatalities among 
            {victim} troops. The incident marks a significant escalation in tensions between 
            the two countries.
            """
            
            texts.append({
                'text': text,
                'title': f"Conflict between {aggressor} and {victim}",
                'date': date,
                'source_file': 'synthetic_example',
                'is_mic': 1  # Known conflict is a positive example
            })
        
        # Label the data (in a real scenario, this would be done by human annotators)
        # For this demonstration, we'll use a simple keyword approach
        labeled_data = []
        
        for item in texts:
            if 'is_mic' in item:
                # Already labeled
                labeled_data.append(item)
                continue
                
            text = item['text'].lower()
            
            # Check for strong indicators of a militarized interstate confrontation
            is_mic = 0
            
            # Check for death terms and military terms in close proximity
            death_military_proximity = False
            for death_term in self.config.DEATH_KEYWORDS:
                if death_term in text:
                    death_pos = text.find(death_term)
                    # Check if any military term is within 50 characters
                    for mil_term in self.config.MILITARY_KEYWORDS:
                        if mil_term in text:
                            mil_pos = text.find(mil_term)
                            if abs(death_pos - mil_pos) < 50:
                                death_military_proximity = True
                                break
                
            # Check for country mentions
            country_names = self.data_loader.get_country_names()
            countries_mentioned = []
            
            for country in country_names:
                if country.lower() in text:
                    countries_mentioned.append(country)
            
            # If we have death-military proximity and at least two countries, likely a MIC
            if death_military_proximity and len(countries_mentioned) >= 2:
                is_mic = 1
            
            # Add to labeled data
            item['is_mic'] = is_mic
            labeled_data.append(item)
        
        # Balance the dataset
        positive_examples = [item for item in labeled_data if item['is_mic'] == 1]
        negative_examples = [item for item in labeled_data if item['is_mic'] == 0]
        
        # Ensure roughly equal numbers of positive and negative examples
        if len(positive_examples) < len(negative_examples):
            # Downsample negative examples
            negative_examples = random.sample(negative_examples, len(positive_examples))
        else:
            # Downsample positive examples
            positive_examples = random.sample(positive_examples, len(negative_examples))
        
        # Combine and shuffle
        balanced_data = positive_examples + negative_examples
        random.shuffle(balanced_data)
        
        # Convert to DataFrame
        labeled_df = pd.DataFrame(balanced_data)
        
        return labeled_df
    
    def generate_entity_extraction_data(self, labeled_df):
        """Generate labeled data for entity extraction."""
        if 'text' not in labeled_df.columns:
            print("Error: Input DataFrame must have a 'text' column.")
            return labeled_df
        
        # Add entity tags column
        labeled_df['entity_tags'] = None
        
        # Get country names
        country_names = self.data_loader.get_country_names()
        
        # Generate entity tags for each text
        for idx, row in tqdm(labeled_df.iterrows(), total=len(labeled_df), desc="Generating entity tags"):
            text = row['text']
            
            # Find countries mentioned in text
            countries_mentioned = []
            for country in country_names:
                if country in text:
                    countries_mentioned.append(country)
            
            # Create entity tags (simplified for demonstration)
            # In a real implementation, this would use NER with BIO tagging
            entity_tags = []
            if countries_mentioned:
                for country in countries_mentioned:
                    entity_tags.append(f"COUNTRY:{country}")
            
            # Add military entity tags
            for military_term in self.config.MILITARY_KEYWORDS:
                if military_term in text.lower():
                    entity_tags.append(f"MILITARY:{military_term}")
            
            labeled_df.at[idx, 'entity_tags'] = entity_tags
        
        return labeled_df
    
    def generate_labeled_dataset(self, num_examples=100, output_file=None):
        """Generate a complete labeled dataset for training all models."""
        if output_file is None:
            output_file = os.path.join(self.output_dir, "labeled_data.csv")
        
        # Generate document classifier data
        labeled_df = self.generate_document_classifier_data(num_examples)
        
        # Add entity extraction labels
        labeled_df = self.generate_entity_extraction_data(labeled_df)
        
        # Save to CSV
        labeled_df.to_csv(output_file, index=False)
        print(f"Labeled dataset saved to {output_file}")
        
        return labeled_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate labeled data for training models.')
    parser.add_argument('--num', type=int, default=100, help='Number of examples to generate')
    parser.add_argument('--output', type=str, help='Output file path')
    args = parser.parse_args()
    
    generator = LabeledDataGenerator()
    labeled_df = generator.generate_labeled_dataset(args.num, args.output)
    
    print(f"Generated {len(labeled_df)} labeled examples.")
    print(f"Positive examples: {sum(labeled_df['is_mic'])}")
    print(f"Negative examples: {len(labeled_df) - sum(labeled_df['is_mic'])}") 
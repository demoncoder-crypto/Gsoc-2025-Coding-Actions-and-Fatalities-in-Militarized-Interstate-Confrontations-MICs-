import os
import pandas as pd
from tqdm import tqdm
from config import Config
from data_loader import DataLoader
from document_classifier import DocumentClassifier
from entity_extractor import EntityExtractor
from event_extractor import EventExtractor
from relation_extractor import RelationExtractor
from numeric_extractor import NumericExtractor

class MICPipeline:
    """Main pipeline for processing news articles to extract militarized interstate confrontations."""
    
    def __init__(self):
        self.config = Config()
        self.data_loader = DataLoader()
        self.document_classifier = DocumentClassifier()
        self.entity_extractor = EntityExtractor()
        self.event_extractor = EventExtractor()
        self.relation_extractor = RelationExtractor()
        self.numeric_extractor = NumericExtractor()
    
    def preprocess(self, years=None):
        """Preprocess articles by loading and pre-filtering."""
        print("Loading articles...")
        articles_df = self.data_loader.load_articles()
        print(f"Loaded {len(articles_df)} articles.")
        
        print("Pre-filtering articles...")
        filtered_df = self.data_loader.prefilter_articles(articles_df)
        print(f"Pre-filtered to {len(filtered_df)} articles.")
        
        return filtered_df
    
    def run(self, articles_df=None, save_results=True):
        """Run the full pipeline on the articles."""
        # Step 1: Preprocess if not already done
        if articles_df is None:
            articles_df = self.preprocess()
        
        # Step 2: Document classification to identify true MIC articles
        print("Classifying articles...")
        mic_df = self.document_classifier.filter_mic_articles(articles_df)
        print(f"Identified {len(mic_df)} potential MIC articles.")
        
        # Step 3: Entity extraction to identify countries and military forces
        print("Extracting entities...")
        self.entity_extractor.load_countries(self.data_loader)
        entity_df = self.entity_extractor.extract_entities(mic_df)
        print(f"Extracted entities from {len(entity_df)} articles.")
        
        # Step 4: Event extraction to identify combat death events
        print("Extracting events...")
        event_df = self.event_extractor.extract_events(entity_df)
        print(f"Extracted events from {len(event_df)} articles.")
        
        # Step 5: Relation extraction to identify country roles
        print("Extracting relations...")
        relation_df = self.relation_extractor.extract_relations(event_df)
        print(f"Extracted relations from {len(relation_df)} articles.")
        
        # Step 6: Numeric extraction to identify fatality counts
        print("Extracting fatality counts...")
        results_df = self.numeric_extractor.extract_fatality_counts(relation_df)
        print(f"Extracted fatality counts from {len(results_df)} articles.")
        
        # Save results if requested
        if save_results:
            self.save_results(results_df)
        
        return results_df
    
    def save_results(self, results_df):
        """Save the pipeline results to a CSV file."""
        # Select and rename columns for the final output
        output_df = results_df[[
            'year', 'source_file', 'title', 'date', 
            'countries_mentioned', 'aggressor_country', 'victim_country',
            'event_dates', 'fatality_min', 'fatality_max'
        ]].copy()
        
        # Rename columns for clarity
        output_df = output_df.rename(columns={
            'date': 'article_date',
            'aggressor_country': 'aggressor',
            'victim_country': 'victim',
            'event_dates': 'confrontation_dates'
        })
        
        # Convert list columns to string for CSV export
        for col in ['countries_mentioned', 'confrontation_dates']:
            output_df[col] = output_df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        
        # Save to CSV
        output_df.to_csv(self.config.RESULTS_FILE, index=False)
        print(f"Results saved to {self.config.RESULTS_FILE}")
        
        return output_df
    
    def get_summary_stats(self, results_df):
        """Generate summary statistics from the results."""
        stats = {}
        
        # Count by year
        stats['count_by_year'] = results_df['year'].value_counts().to_dict()
        
        # Count by aggressor country
        stats['count_by_aggressor'] = results_df['aggressor_country'].value_counts().to_dict()
        
        # Count by victim country
        stats['count_by_victim'] = results_df['victim_country'].value_counts().to_dict()
        
        # Total fatalities (using max range)
        stats['total_fatalities_min'] = results_df['fatality_min'].sum()
        stats['total_fatalities_max'] = results_df['fatality_max'].sum()
        
        # Average fatalities per incident
        stats['avg_fatalities_min'] = results_df['fatality_min'].mean()
        stats['avg_fatalities_max'] = results_df['fatality_max'].mean()
        
        return stats 
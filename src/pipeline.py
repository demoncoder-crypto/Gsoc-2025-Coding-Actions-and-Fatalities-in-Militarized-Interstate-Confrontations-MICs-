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
        # Ensure required columns exist
        required_columns = [
            'year', 'source_file', 'title', 'date', 
            'countries_mentioned', 'aggressor_country', 'victim_country',
            'event_dates', 'fatality_min', 'fatality_max'
        ]
        
        # Check for missing columns and add them with default values
        for col in required_columns:
            if col not in results_df.columns:
                print(f"Warning: Column '{col}' missing in results. Adding with default values.")
                if col in ['fatality_min', 'fatality_max']:
                    results_df[col] = 0
                elif col in ['countries_mentioned', 'event_dates']:
                    results_df[col] = [[] for _ in range(len(results_df))]
                else:
                    results_df[col] = "Unknown"
        
        # Select and rename columns for the final output
        output_columns = [col for col in required_columns if col in results_df.columns]
        output_df = results_df[output_columns].copy()
        
        # Rename columns for clarity
        column_renames = {
            'date': 'article_date',
            'aggressor_country': 'aggressor',
            'victim_country': 'victim',
            'event_dates': 'confrontation_dates'
        }
        
        # Only rename columns that exist
        rename_dict = {k: v for k, v in column_renames.items() if k in output_df.columns}
        if rename_dict:
            output_df = output_df.rename(columns=rename_dict)
        
        # Convert list columns to string for CSV export
        for col in ['countries_mentioned', 'confrontation_dates']:
            if col in output_df.columns:
                output_df[col] = output_df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        
        # Save to CSV
        os.makedirs(os.path.dirname(self.config.RESULTS_FILE), exist_ok=True)
        output_df.to_csv(self.config.RESULTS_FILE, index=False)
        print(f"Results saved to {self.config.RESULTS_FILE}")
        
        return output_df
    
    def get_summary_stats(self, results_df):
        """Generate summary statistics from the results."""
        stats = {}
        
        # Count by year (if exists)
        if 'year' in results_df.columns:
            stats['count_by_year'] = results_df['year'].value_counts().to_dict()
        
        # Count by aggressor country (if exists)
        if 'aggressor_country' in results_df.columns:
            stats['count_by_aggressor'] = results_df['aggressor_country'].value_counts().to_dict()
        
        # Count by victim country (if exists)
        if 'victim_country' in results_df.columns:
            stats['count_by_victim'] = results_df['victim_country'].value_counts().to_dict()
        
        # Total fatalities (using max range)
        if 'fatality_min' in results_df.columns and 'fatality_max' in results_df.columns:
            stats['total_fatalities_min'] = results_df['fatality_min'].sum()
            stats['total_fatalities_max'] = results_df['fatality_max'].sum()
            
            # Average fatalities per incident
            stats['avg_fatalities_min'] = results_df['fatality_min'].mean()
            stats['avg_fatalities_max'] = results_df['fatality_max'].mean()
        
        return stats 
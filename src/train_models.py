import os
import sys
import pandas as pd
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import Config
from document_classifier import DocumentClassifier
from entity_extractor import EntityExtractor
from event_extractor import EventExtractor
from relation_extractor import RelationExtractor
from numeric_extractor import NumericExtractor

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train models for MIC extraction.')
    parser.add_argument('--training_data', type=str, required=True, 
                        help='Path to labeled training data CSV')
    parser.add_argument('--model', type=str, choices=['document', 'entity', 'event', 'relation', 'numeric', 'all'], 
                        default='all', help='Model to train (default: all)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--output_dir', type=str, help='Output directory for models')
    return parser.parse_args()

def load_training_data(filepath):
    """Load labeled training data from CSV file."""
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Training data file not found: {filepath}")
    
    # Load the CSV file
    df = pd.read_csv(filepath)
    
    # Check required columns
    required_columns = {
        'document': ['text', 'is_mic'],
        'entity': ['text', 'countries', 'military_entities'],
        'event': ['text', 'has_death_event'],
        'relation': ['text', 'aggressor', 'victim'],
        'numeric': ['text', 'fatality_min', 'fatality_max']
    }
    
    # Check if all required columns for all models are present
    missing_columns = {}
    for model, cols in required_columns.items():
        missing = [col for col in cols if col not in df.columns]
        if missing:
            missing_columns[model] = missing
    
    if missing_columns:
        print("Warning: Some required columns are missing in the training data:")
        for model, cols in missing_columns.items():
            print(f"  - {model}: {', '.join(cols)}")
    
    return df

def train_document_classifier(df, config):
    """Train the document classifier model."""
    print("Training document classifier...")
    
    # Check required columns
    if not all(col in df.columns for col in ['text', 'is_mic']):
        print("Error: Training data missing required columns for document classifier.")
        return
    
    # Prepare data
    texts = df['text'].tolist()
    labels = df['is_mic'].astype(int).tolist()
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Initialize and train model
    classifier = DocumentClassifier()
    results = classifier.train(train_texts, train_labels, val_texts, val_labels)
    
    print(f"Document classifier training complete. Evaluation results: {results}")
    return classifier

def train_entity_extractor(df, config):
    """Train the entity extractor model."""
    print("Training entity extractor...")
    
    # In a real implementation, we would train a proper NER model
    # For this example, we'll just print a placeholder
    print("Entity extractor training would be implemented here.")
    print("This would involve fine-tuning a token classification model for NER.")
    
    return EntityExtractor()

def train_event_extractor(df, config):
    """Train the event extractor model."""
    print("Training event extractor...")
    
    # In a real implementation, we would train a model for event extraction
    # For this example, we'll just print a placeholder
    print("Event extractor training would be implemented here.")
    print("This would involve fine-tuning a sequence classification model for event detection.")
    
    return EventExtractor()

def train_relation_extractor(df, config):
    """Train the relation extractor model."""
    print("Training relation extractor...")
    
    # In a real implementation, we would train a relation extraction model
    # For this example, we'll just print a placeholder
    print("Relation extractor training would be implemented here.")
    print("This would involve fine-tuning a model for relation classification.")
    
    return RelationExtractor()

def train_numeric_extractor(df, config):
    """Train the numeric extractor model."""
    print("Training numeric extractor...")
    
    # In a real implementation, we would train a model for numeric extraction
    # For this example, we'll just print a placeholder
    print("Numeric extractor training would be implemented here.")
    print("This would involve fine-tuning a token classification model for number extraction.")
    
    return NumericExtractor()

def main():
    """Main entry point for model training."""
    # Parse command line arguments
    args = parse_args()
    
    # Create config and update with command line arguments
    config = Config()
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate:
        config.LEARNING_RATE = args.learning_rate
    if args.output_dir:
        config.MODEL_DIR = args.output_dir
        os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # Load training data
    df = load_training_data(args.training_data)
    print(f"Loaded training data with {len(df)} examples.")
    
    # Train the specified model(s)
    if args.model == 'all' or args.model == 'document':
        train_document_classifier(df, config)
    
    if args.model == 'all' or args.model == 'entity':
        train_entity_extractor(df, config)
    
    if args.model == 'all' or args.model == 'event':
        train_event_extractor(df, config)
    
    if args.model == 'all' or args.model == 'relation':
        train_relation_extractor(df, config)
    
    if args.model == 'all' or args.model == 'numeric':
        train_numeric_extractor(df, config)
    
    print("Training complete. Models saved to:", config.MODEL_DIR)

if __name__ == "__main__":
    main() 
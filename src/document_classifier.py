import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from config import Config

class ArticleDataset(Dataset):
    """Dataset for document classification."""
    
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text, 
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Convert to appropriate format for dataset
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return item

class DocumentClassifier:
    """Classifier to identify true MIC articles."""
    
    def __init__(self, model_name=None):
        self.config = Config()
        self.model_name = model_name or self.config.DOCUMENT_CLASSIFIER_MODEL
        self.model = None
        self.tokenizer = None
        self.model_path = os.path.join(self.config.MODEL_DIR, "document_classifier")
        
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None):
        """Train the document classifier model."""
        # Initialize the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )
        
        # Create datasets
        train_dataset = ArticleDataset(
            train_texts, train_labels, self.tokenizer, self.config.MAX_SEQ_LENGTH
        )
        
        if val_texts is not None and val_labels is not None:
            val_dataset = ArticleDataset(
                val_texts, val_labels, self.tokenizer, self.config.MAX_SEQ_LENGTH
            )
        else:
            # Split training data if validation data not provided
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels, test_size=0.2, random_state=42
            )
            train_dataset = ArticleDataset(
                train_texts, train_labels, self.tokenizer, self.config.MAX_SEQ_LENGTH
            )
            val_dataset = ArticleDataset(
                val_texts, val_labels, self.tokenizer, self.config.MAX_SEQ_LENGTH
            )
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.model_path,
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Create data collator for batching
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train the model
        trainer.train()
        
        # Save the trained model and tokenizer
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        
        return trainer.evaluate()
    
    def load_model(self):
        """Load a trained model and tokenizer."""
        if os.path.exists(self.model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        else:
            raise FileNotFoundError(f"No trained model found at {self.model_path}")
    
    def predict(self, texts):
        """Predict whether articles are true MIC articles."""
        if self.model is None or self.tokenizer is None:
            try:
                self.load_model()
            except FileNotFoundError:
                print("No trained model found. Using pretrained model for prediction.")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, num_labels=2
                )
        
        # Create dataset
        dataset = ArticleDataset(
            texts, labels=None, tokenizer=self.tokenizer, max_length=self.config.MAX_SEQ_LENGTH
        )
        
        # Set up data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Create inference trainer
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Make predictions
        predictions = trainer.predict(dataset)
        
        # Get predicted classes and probabilities
        preds = np.argmax(predictions.predictions, axis=1)
        probs = torch.nn.functional.softmax(
            torch.tensor(predictions.predictions), dim=1
        ).numpy()
        
        return preds, probs
    
    def filter_mic_articles(self, articles_df):
        """Filter articles to keep only true MIC articles."""
        # Predict whether each article is a true MIC
        preds, probs = self.predict(articles_df['full_text'].tolist())
        
        # Add predictions to dataframe
        articles_df['is_mic'] = preds
        articles_df['mic_confidence'] = probs[:, 1]  # Probability of positive class
        
        # Filter to keep only predicted MIC articles
        filtered_df = articles_df[articles_df['is_mic'] == 1].copy()
        
        return filtered_df 
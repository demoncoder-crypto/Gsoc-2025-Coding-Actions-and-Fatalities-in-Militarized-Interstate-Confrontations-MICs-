import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding
)
from transformers import pipeline
from torch.utils.data import Dataset
from config import Config
from datasets import Dataset as HFDataset

class DocumentClassificationDataset(Dataset):
    """Dataset for document classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
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
        
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

class TokenClassificationDataset(Dataset):
    """Dataset for token classification (NER)."""
    
    def __init__(self, texts, tags, tokenizer, max_length=512, label_to_id=None):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = label_to_id or self._create_label_map()
        
    def _create_label_map(self):
        """Create mapping from tag names to IDs."""
        # Flatten the list of tags
        all_tags = [tag for tags_list in self.tags for tag in tags_list]
        unique_tags = sorted(set(all_tags))
        return {tag: i for i, tag in enumerate(unique_tags)}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tags = self.tags[idx]
        
        # Tokenize text
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            is_split_into_words=False,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create token-level labels
        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)
        
        # Simplistic approach: assign the same label to all tokens
        # In a real implementation, this would be more sophisticated to handle BIO tagging
        label_id = self.label_to_id.get(tags[0], 0)
        labels = torch.tensor([label_id] * len(input_ids), dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class FineTuner:
    """Fine-tune models with limited labeled data."""
    
    def __init__(self):
        self.config = Config()
        self.model_dir = self.config.MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)
    
    def train_document_classifier(self, texts, labels, output_dir=None, epochs=3, batch_size=8):
        """Fine-tune a document classifier."""
        if output_dir is None:
            output_dir = os.path.join(self.model_dir, "document_classifier")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.config.DOCUMENT_CLASSIFIER_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.DOCUMENT_CLASSIFIER_MODEL, 
            num_labels=len(set(labels))
        )
        
        # Create datasets
        train_dataset = DocumentClassificationDataset(
            train_texts, train_labels, tokenizer, self.config.MAX_SEQ_LENGTH
        )
        val_dataset = DocumentClassificationDataset(
            val_texts, val_labels, tokenizer, self.config.MAX_SEQ_LENGTH
        )
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Create data collator for batching
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        # Train the model
        trainer.train()
        
        # Save the trained model and tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Evaluate the model
        evaluation = trainer.evaluate()
        
        return model, tokenizer, evaluation
    
    def train_entity_extractor(self, texts, entity_tags, output_dir=None, epochs=3, batch_size=8):
        """Fine-tune an entity extraction model."""
        if output_dir is None:
            output_dir = os.path.join(self.model_dir, "entity_extractor")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create label map
        all_tags = [tag for tags_list in entity_tags for tag in tags_list]
        unique_tags = sorted(set(all_tags))
        label_to_id = {tag: i for i, tag in enumerate(unique_tags)}
        id_to_label = {i: tag for tag, i in label_to_id.items()}
        
        # Split data
        train_texts, val_texts, train_tags, val_tags = train_test_split(
            texts, entity_tags, test_size=0.2, random_state=42
        )
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.config.NER_MODEL)
        model = AutoModelForTokenClassification.from_pretrained(
            self.config.NER_MODEL,
            num_labels=len(unique_tags)
        )
        
        # Convert to HuggingFace datasets format
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.MAX_SEQ_LENGTH,
                is_split_into_words=False,
                padding="max_length"
            )
            
            # Simplistic approach for demonstration
            # In a real implementation, you would align word-level labels with token-level
            tokenized_inputs["labels"] = [
                [label_to_id.get(tag, 0) for tag in example_tags]
                for example_tags in examples["tags"]
            ]
            
            return tokenized_inputs
        
        # Create datasets
        train_dataset = HFDataset.from_dict({
            "text": train_texts,
            "tags": train_tags
        })
        val_dataset = HFDataset.from_dict({
            "text": val_texts,
            "tags": val_tags
        })
        
        # Tokenize datasets
        train_tokenized = train_dataset.map(tokenize_and_align_labels, batched=True)
        val_tokenized = val_dataset.map(tokenize_and_align_labels, batched=True)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Create data collator
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        # Train the model
        trainer.train()
        
        # Save the trained model and tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save label mappings
        with open(os.path.join(output_dir, "label_map.txt"), "w") as f:
            for tag, idx in label_to_id.items():
                f.write(f"{tag}\t{idx}\n")
        
        # Evaluate the model
        evaluation = trainer.evaluate()
        
        return model, tokenizer, evaluation
    
    def train_from_labeled_data(self, labeled_data_file):
        """Train all models from a labeled data file."""
        # Load labeled data
        try:
            labeled_df = pd.read_csv(labeled_data_file)
            print(f"Loaded {len(labeled_df)} labeled examples.")
        except Exception as e:
            print(f"Error loading labeled data: {e}")
            return
        
        results = {}
        
        # Train document classifier if labeled data has 'is_mic' column
        if 'text' in labeled_df.columns and 'is_mic' in labeled_df.columns:
            print("Training document classifier...")
            texts = labeled_df['text'].tolist()
            labels = labeled_df['is_mic'].astype(int).tolist()
            
            model, tokenizer, evaluation = self.train_document_classifier(
                texts, labels
            )
            
            results['document_classifier'] = evaluation
            print(f"Document classifier training complete. Evaluation: {evaluation}")
        
        # Train entity extractor if labeled data has 'entity_tags' column
        if 'text' in labeled_df.columns and 'entity_tags' in labeled_df.columns:
            print("Training entity extractor...")
            texts = labeled_df['text'].tolist()
            
            # Convert string representation of lists to actual lists
            entity_tags = labeled_df['entity_tags'].apply(
                lambda x: eval(x) if isinstance(x, str) else x
            ).tolist()
            
            model, tokenizer, evaluation = self.train_entity_extractor(
                texts, entity_tags
            )
            
            results['entity_extractor'] = evaluation
            print(f"Entity extractor training complete. Evaluation: {evaluation}")
        
        # Additional model training can be added here
        
        return results

# Example usage:
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune models with labeled data.')
    parser.add_argument('--data', type=str, required=True, help='Path to labeled data CSV file')
    args = parser.parse_args()
    
    fine_tuner = FineTuner()
    results = fine_tuner.train_from_labeled_data(args.data)
    print("Fine-tuning complete.") 
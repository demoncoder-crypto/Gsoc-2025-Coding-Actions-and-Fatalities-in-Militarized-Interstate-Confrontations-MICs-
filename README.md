# Militarized Interstate Confrontations (MICs) Detection System

This project analyzes newspaper articles to identify and extract information about militarized interstate confrontations where military fatalities occurred. The system uses a hybrid approach combining transformer-based models and rule-based components to process news articles and extract relevant information.

## Project Structure

- `src/` - Source code directory
  - `config.py` - Configuration settings
  - `data_loader.py` - Article loading and preprocessing
  - `document_classifier.py` - Identifies true MIC articles
  - `entity_extractor.py` - Extracts countries and military entities
  - `event_extractor.py` - Identifies combat death events
  - `relation_extractor.py` - Determines country roles (aggressor/victim)
  - `numeric_extractor.py` - Extracts fatality counts
  - `pipeline.py` - Integrates all components
  - `main.py` - Main script to run the pipeline
- `Dataset/` - Input data directory
  - `states2016.csv` - State system list
  - `New York Times/` - Directory with news articles organized by year
- `output/` - Output directory (created by the script)
  - `models/` - Trained model files (if applicable)
  - `mic_results.csv` - Final output file with MIC information

## Architecture

The system implements a multi-stage pipeline:

1. **Document filtering** - Eliminates 95% of false positives
2. **Entity extraction** - Identifies countries and military forces
3. **Event detection** - Identifies combat death events
4. **Information extraction** - Extracts dates and fatality counts
5. **Relation linking** - Connects countries to their roles in events (aggressor/victim)

The implementation uses RoBERTa-large as the base model, with specialized components for each stage of the pipeline.


## Output Format

The output CSV file contains the following columns:

- `year` - Year of the article
- `source_file` - Source file name
- `title` - Article title
- `article_date` - Publication date of the article
- `countries_mentioned` - All countries mentioned in the article
- `aggressor` - Country identified as the aggressor
- `victim` - Country identified as the victim
- `confrontation_dates` - Dates of the confrontation mentioned in the article
- `fatality_min` - Minimum number of fatalities
- `fatality_max` - Maximum number of fatalities


## Training

While the current implementation relies primarily on pre-trained models and rule-based components, it includes functionality for fine-tuning models with labeled data if available.

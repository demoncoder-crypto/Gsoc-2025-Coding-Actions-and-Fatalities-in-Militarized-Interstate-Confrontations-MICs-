# Running the MIC Detection System

This document provides instructions for running the Militarized Interstate Confrontations (MICs) detection system from start to finish.

## System Requirements

- Python 3.8+
- 8GB+ RAM
- Internet connection (for downloading pre-trained models)
- 10GB+ free disk space

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/demoncoder-crypto/Gsoc-2025-Coding-Actions-and-Fatalities-in-Militarized-Interstate-Confrontations-MICs-.git
   cd Gsoc-2025-Coding-Actions-and-Fatalities-in-Militarized-Interstate-Confrontations-MICs-
   ```

2. **Install dependencies**
   ```bash
   pip install -r src/requirements.txt
   ```

3. **Download the dataset**
   - Ensure the Dataset directory is structured as follows:
     ```
     Dataset/
     ├── states2016.csv
     └── New York Times/
         ├── 2015/
         ├── 2016/
         ├── 2017/
         ├── ...
         └── 2023/
     ```

## Running the System

### Full Analysis

To run the complete pipeline and analyze all articles:

```bash
python src/main.py --stats --validate
```

This command will:
- Load all articles from 2015-2023
- Identify militarized interstate confrontations
- Extract dates, countries involved, and fatality counts
- Generate statistics
- Validate results against known conflicts
- Save results to output/mic_results.csv

### Options

- `--years [YEARS]`: Specify specific years to analyze (e.g., `--years 2022 2023`)
- `--output OUTPUT`: Specify custom output file path
- `--stats`: Generate and display summary statistics
- `--validate`: Validate results against known conflicts
- `--sample NUM`: Process only a sample of articles (for testing)

### Examples

1. **Analyze specific years**
   ```bash
   python src/main.py --years 2022 2023 --stats
   ```

2. **Quick test with sample**
   ```bash
   python src/main.py --sample 100 --stats
   ```

3. **Generate labeled data for fine-tuning**
   ```bash
   python src/generate_labeled_data.py --num 50
   ```

4. **Fine-tune models with labeled data**
   ```bash
   python src/fine_tuning.py --data output/labeled_data/labeled_data.csv
   ```

## Output

The system generates the following outputs:

- `output/mic_results.csv`: Main results containing dates, countries, and fatality ranges
- `output/validation_results.csv`: Validation results against known conflicts
- `output/models/`: Directory containing any fine-tuned models

## Runtime Expectations

The full analysis typically takes 2-4 hours depending on hardware, with most time spent on:
- Document classification (40%)
- Entity extraction (30%)
- Relation extraction (20%)
- Other components (10%)

## Troubleshooting

- **Memory errors**: Try reducing batch size by editing `config.py`
- **Missing dependencies**: Ensure all requirements are installed
- **Model download issues**: Check internet connection; models can be manually downloaded from Hugging Face 
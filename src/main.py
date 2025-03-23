import os
import sys
import pandas as pd
import argparse
from tqdm import tqdm
from pipeline import MICPipeline
from config import Config
from validation import ResultsValidator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract militarized interstate confrontations from news articles.')
    parser.add_argument('--years', type=int, nargs='+', help='Specific years to analyze')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--stats', action='store_true', help='Generate summary statistics')
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress')
    parser.add_argument('--validate', action='store_true', help='Validate results against known conflicts')
    parser.add_argument('--sample', type=int, default=0, help='Process only a sample of articles for testing')
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    # Parse command line arguments
    args = parse_args()
    
    # Create config and update with command line arguments if provided
    config = Config()
    if args.years:
        config.YEARS_TO_ANALYZE = args.years
    if args.output:
        config.RESULTS_FILE = args.output
    
    # Set tqdm verbosity
    tqdm.pandas(disable=not args.verbose)
    
    # Run the pipeline
    print(f"Analyzing news articles for years: {config.YEARS_TO_ANALYZE}")
    pipeline = MICPipeline()
    
    # If sample mode is enabled, process only a subset of articles
    if args.sample > 0:
        print(f"Sample mode: Processing only {args.sample} articles for testing")
        articles_df = pipeline.preprocess()
        sample_df = articles_df.sample(min(args.sample, len(articles_df)))
        results_df = pipeline.run(sample_df)
    else:
        results_df = pipeline.run()
    
    # Generate statistics if requested
    if args.stats:
        stats = pipeline.get_summary_stats(results_df)
        print("\nSummary Statistics:")
        for stat_name, stat_value in stats.items():
            if isinstance(stat_value, dict):
                print(f"\n{stat_name}:")
                for k, v in sorted(stat_value.items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"  {k}: {v}")
            else:
                print(f"{stat_name}: {stat_value}")
    
    # Validate results if requested
    if args.validate:
        validator = ResultsValidator()
        validation_results, summary = validator.print_validation_report(results_df=results_df)
        
        # Save validation results
        validation_file = os.path.join(config.OUTPUT_DIR, "validation_results.csv")
        validation_results.to_csv(validation_file, index=False)
        print(f"Validation results saved to {validation_file}")
    
    # Print final results
    print(f"\nExtracted {len(results_df)} militarized interstate confrontations with fatalities.")
    print(f"Results saved to {config.RESULTS_FILE}")
    
    return results_df

if __name__ == "__main__":
    main() 
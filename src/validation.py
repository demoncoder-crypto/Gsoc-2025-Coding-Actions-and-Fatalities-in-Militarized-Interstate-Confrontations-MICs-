import os
import pandas as pd
import numpy as np
from config import Config

class ValidationDataset:
    """Known militarized interstate confrontations for validation."""
    
    # These are well-documented conflicts for our validation set
    KNOWN_CONFLICTS = [
        {
            'date': '2015-11-24',
            'aggressor': 'Turkey',
            'victim': 'Russian Federation',
            'description': 'Turkish F-16 fighter jet shot down a Russian Su-24 aircraft near the Syria–Turkey border',
            'fatalities': 1
        },
        {
            'date': '2016-07-15',
            'aggressor': 'Turkey',
            'victim': 'Turkey',
            'description': 'Failed coup attempt in Turkey resulted in military casualties',
            'fatalities': 104
        },
        {
            'date': '2019-02-26',
            'aggressor': 'India',
            'victim': 'Pakistan',
            'description': 'Indian Air Force conducted airstrikes in Balakot, Pakistan',
            'fatalities': 0
        },
        {
            'date': '2019-02-27',
            'aggressor': 'Pakistan',
            'victim': 'India',
            'description': 'Pakistan shot down an Indian MiG-21 and captured pilot',
            'fatalities': 1
        },
        {
            'date': '2020-06-15',
            'aggressor': 'China',
            'victim': 'India',
            'description': 'Galwan Valley clash between Indian and Chinese troops',
            'fatalities': 20
        },
        {
            'date': '2022-02-24',
            'aggressor': 'Russian Federation',
            'victim': 'Ukraine',
            'description': 'Russian invasion of Ukraine began',
            'fatalities': 9000
        },
        {
            'date': '2023-10-07',
            'aggressor': 'Hamas',
            'victim': 'Israel',
            'description': 'Hamas attack on Israel',
            'fatalities': 1200
        },
        {
            'date': '2023-10-08',
            'aggressor': 'Israel',
            'victim': 'Gaza',
            'description': 'Israeli airstrikes on Gaza',
            'fatalities': 3000
        }
    ]
    
    def __init__(self):
        self.config = Config()
        self.known_conflicts_df = pd.DataFrame(self.KNOWN_CONFLICTS)
    
    def get_validation_df(self):
        """Get the validation dataset as a DataFrame."""
        return self.known_conflicts_df.copy()


class ResultsValidator:
    """Validate the extracted MIC results against known conflicts."""
    
    def __init__(self):
        self.config = Config()
        self.validation_dataset = ValidationDataset()
    
    def load_results(self, results_file=None):
        """Load the extracted results."""
        if results_file is None:
            results_file = self.config.RESULTS_FILE
        
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        return pd.read_csv(results_file)
    
    def validate(self, results_df=None):
        """Validate the extracted results against known conflicts."""
        if results_df is None:
            # Load the results if not provided
            try:
                results_df = self.load_results()
            except FileNotFoundError:
                print("No results file found. Run the pipeline first.")
                return None, None
        
        # Get the validation dataset
        validation_df = self.validation_dataset.get_validation_df()
        
        # Check if required columns exist
        required_columns = ['confrontation_dates', 'aggressor', 'victim', 'fatality_min', 'fatality_max']
        
        # Add missing columns with default values
        for col in required_columns:
            if col not in results_df.columns:
                print(f"Warning: Required column '{col}' missing in results. Adding with default values.")
                if col in ['fatality_min', 'fatality_max']:
                    results_df[col] = 0
                elif col in ['confrontation_dates']:
                    results_df[col] = ""
                else:
                    results_df[col] = "Unknown"
        
        # Convert dates to strings for comparison if present
        if 'confrontation_dates' in results_df.columns:
            results_df['confrontation_dates'] = results_df['confrontation_dates'].astype(str)
        
        # Initialize validation results
        validation_results = []
        
        # Check for each known conflict
        for _, known_conflict in validation_df.iterrows():
            # Extract details
            known_date = known_conflict['date']
            known_aggressor = known_conflict['aggressor']
            known_victim = known_conflict['victim']
            known_fatalities = known_conflict['fatalities']
            
            # Default values for validation metrics
            found = False
            correct_aggressor = False
            correct_victim = False
            fatality_min = 0
            fatality_max = 0
            fatality_in_range = False
            
            # Check if the conflict was found (only if confrontation_dates column exists)
            if 'confrontation_dates' in results_df.columns and 'aggressor' in results_df.columns and 'victim' in results_df.columns:
                found_conflicts = results_df[
                    (results_df['confrontation_dates'].str.contains(known_date, na=False, regex=False)) |
                    ((results_df['aggressor'] == known_aggressor) & 
                     (results_df['victim'] == known_victim))
                ]
                
                # Calculate metrics
                found = len(found_conflicts) > 0
                correct_aggressor = found and any(found_conflicts['aggressor'] == known_aggressor)
                correct_victim = found and any(found_conflicts['victim'] == known_victim)
                
                if found and 'fatality_min' in found_conflicts.columns and 'fatality_max' in found_conflicts.columns:
                    fatality_min = found_conflicts['fatality_min'].max() if found else 0
                    fatality_max = found_conflicts['fatality_max'].max() if found else 0
                    
                    # Check if fatality range includes the known value
                    fatality_in_range = found and (fatality_min <= known_fatalities <= fatality_max)
            
            # Add to validation results
            validation_results.append({
                'date': known_date,
                'aggressor': known_aggressor,
                'victim': known_victim,
                'known_fatalities': known_fatalities,
                'found': found,
                'correct_aggressor': correct_aggressor,
                'correct_victim': correct_victim,
                'extracted_fatality_min': fatality_min,
                'extracted_fatality_max': fatality_max,
                'fatality_in_range': fatality_in_range
            })
        
        # Convert to DataFrame
        validation_results_df = pd.DataFrame(validation_results)
        
        # Calculate summary metrics
        recall = validation_results_df['found'].mean() if len(validation_results_df) > 0 else 0
        aggressor_accuracy = validation_results_df['correct_aggressor'].mean() if len(validation_results_df) > 0 else 0
        victim_accuracy = validation_results_df['correct_victim'].mean() if len(validation_results_df) > 0 else 0
        fatality_accuracy = validation_results_df['fatality_in_range'].mean() if len(validation_results_df) > 0 else 0
        
        # Overall F1 score (harmonic mean of recall and precision)
        # Note: We can't compute true precision without manual verification of all extracted events
        f1_role_identification = 2 * (aggressor_accuracy * victim_accuracy) / (aggressor_accuracy + victim_accuracy) if (aggressor_accuracy + victim_accuracy) > 0 else 0
        
        summary = {
            'recall': recall,
            'aggressor_accuracy': aggressor_accuracy,
            'victim_accuracy': victim_accuracy,
            'fatality_accuracy': fatality_accuracy,
            'f1_role_identification': f1_role_identification,
            'overall_quality': (recall + aggressor_accuracy + victim_accuracy + fatality_accuracy) / 4
        }
        
        return validation_results_df, summary
    
    def print_validation_report(self, validation_results=None, summary=None):
        """Print a validation report."""
        if validation_results is None or summary is None:
            validation_results, summary = self.validate()
            
        if validation_results is None or summary is None:
            print("Validation failed. No results to report.")
            return None, None
        
        print("\n=== Validation Report ===")
        print(f"Recall (% of known conflicts found): {summary['recall']:.2%}")
        print(f"Aggressor Accuracy: {summary['aggressor_accuracy']:.2%}")
        print(f"Victim Accuracy: {summary['victim_accuracy']:.2%}")
        print(f"Fatality Range Accuracy: {summary['fatality_accuracy']:.2%}")
        print(f"F1 Score (Role Identification): {summary['f1_role_identification']:.2%}")
        print(f"Overall Quality: {summary['overall_quality']:.2%}")
        
        print("\nDetailed Validation Results:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(validation_results[['date', 'aggressor', 'victim', 'known_fatalities', 
                                 'found', 'correct_aggressor', 'correct_victim',
                                 'extracted_fatality_min', 'extracted_fatality_max']])
        
        # Save validation results to file
        validation_file = os.path.join(self.config.OUTPUT_DIR, "validation_results.csv")
        validation_results.to_csv(validation_file, index=False)
        print(f"\nValidation results saved to {validation_file}")
        
        return validation_results, summary 
# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/pg/art/calculate_metrics_art.py
import os
import pandas as pd
import logging
from typing import Dict, List, Optional
from src.pg.art.utils.metrics_art import MetricsEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ColumnNotFoundError(Exception):
    """Custom exception for missing columns."""
    pass

def get_column_name(df: pd.DataFrame, possible_names: List[str]) -> str:
    """Helper function to find a column name from possible variations."""
    for name in possible_names:
        if name in df.columns:
            return name
    raise ColumnNotFoundError(f"Could not find any of {possible_names} in DataFrame columns")

def clean_text_column(series: pd.Series) -> List[str]:
    """Clean and convert a text column to list of strings."""
    return series.fillna('').astype(str).tolist()

def validate_text_data(texts: List[str], context: str = "") -> List[str]:
    """Validate and clean text data."""
    valid_texts = []
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            logger.warning(f"Non-string value found in {context} at index {i}: {text}")
            valid_texts.append("")
        else:
            valid_texts.append(text.strip())
    return valid_texts

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts necessary columns, computes similarity metrics using MetricsEvaluator,
    and returns a DataFrame of metrics with robust error handling.
    """
    # Define possible column name variations
    column_variations = {
        'generated_text': ['Generated Text', 'Generated_Text', 'generated text', 'generated_text'],
        'edited_ending': ['Edited Ending', 'Edited_Ending', 'edited ending', 'edited_ending'],
        'counterfactual': ['Counterfactual', 'counterfactual'],
        'initial': ['Initial', 'initial'],
        'premise': ['Premise', 'premise']
    }

    try:
        # Get the actual column names from the DataFrame
        generated_col = get_column_name(df, column_variations['generated_text'])
        edited_col = get_column_name(df, column_variations['edited_ending'])
        counterfactual_col = get_column_name(df, column_variations['counterfactual'])
        initial_col = get_column_name(df, column_variations['initial'])
        premise_col = get_column_name(df, column_variations['premise'])

        logger.info(f"Using columns: Generated={generated_col}, Edited={edited_col}, "
                   f"Counterfactual={counterfactual_col}, Initial={initial_col}, "
                   f"Premise={premise_col}")

        # Extract and clean data
        generated_texts = validate_text_data(clean_text_column(df[generated_col]), "generated texts")
        edited_endings = validate_text_data(clean_text_column(df[edited_col]), "edited endings")
        counterfactuals = validate_text_data(clean_text_column(df[counterfactual_col]), "counterfactuals")
        initials = validate_text_data(clean_text_column(df[initial_col]), "initials")
        premises = validate_text_data(clean_text_column(df[premise_col]), "premises")

        # Filter out rows where critical columns are empty
        valid_indices = [
            i for i in range(len(generated_texts))
            if (generated_texts[i] and edited_endings[i])
        ]

        if not valid_indices:
            raise ValueError("No valid rows found after filtering empty texts")

        logger.info(f"Processing {len(valid_indices)} valid rows out of {len(df)} total rows")

        # Prepare filtered data
        filtered_data = {
            'generated': [generated_texts[i] for i in valid_indices],
            'edited': [edited_endings[i] for i in valid_indices],
            'counterfactual': [counterfactuals[i] for i in valid_indices],
            'initial': [initials[i] for i in valid_indices],
            'premise': [premises[i] for i in valid_indices]
        }

        evaluator = MetricsEvaluator()
        all_metrics = {}

        # Calculate metrics with error handling for each metric type
        try:
            bart_metrics = evaluator.calculate_and_log_bart_similarity(
                filtered_data['generated'], filtered_data['edited'],
                filtered_data['counterfactual'], filtered_data['initial'],
                filtered_data['premise'], logger
            )
            all_metrics.update(bart_metrics)
        except Exception as e:
            logger.error(f"Error calculating BART metrics: {str(e)}", exc_info=True)

        try:
            rouge_metrics = evaluator.calculate_and_log_rouge_scores(
                filtered_data['generated'], filtered_data['edited'],
                filtered_data['counterfactual'], filtered_data['initial'],
                filtered_data['premise'], logger
            )
            all_metrics.update(rouge_metrics)
        except Exception as e:
            logger.error(f"Error calculating ROUGE metrics: {str(e)}", exc_info=True)

        # Create metrics DataFrame
        metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index', columns=['Score'])
        metrics_df.reset_index(inplace=True)
        metrics_df.columns = ['Metric', 'Score']
        
        # Add summary statistics
        metrics_df['Dataset'] = f"{len(valid_indices)} valid rows of {len(df)}"
        return metrics_df

    except ColumnNotFoundError as e:
        logger.error(f"Column error: {str(e)}")
        logger.info(f"Available columns: {df.columns.tolist()}")
        raise
    except Exception as e:
        logger.error(f"Error in process_data: {str(e)}", exc_info=True)
        raise

def process_file(file_path: str) -> None:
    """
    Process a single CSV file with comprehensive error handling.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"\n{'='*50}\nProcessing file: {file_path}\n{'='*50}")
        
        # Read CSV with error handling
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully read {len(df)} rows from {file_path}")
        except Exception as e:
            raise IOError(f"Failed to read CSV file: {str(e)}")

        # Log basic data info
        logger.info(f"Columns found: {df.columns.tolist()}")
        logger.info(f"Null values per column:\n{df.isnull().sum()}")
        logger.info(f"Data types:\n{df.dtypes}")

        # Process data
        metrics_df = process_data(df)

        # Save results
        base_dir = os.path.dirname(file_path)
        base_name, ext = os.path.splitext(os.path.basename(file_path))
        output_file_path = os.path.join(base_dir, f'{base_name}_metrics{ext}')
        
        metrics_df.to_csv(output_file_path, index=False)
        logger.info(f"Successfully saved metrics to {output_file_path}")
        
        return output_file_path

    except Exception as e:
        logger.error(f"Fatal error processing file {file_path}: {str(e)}", exc_info=True)
        raise

def process_repository(repo_path: str, prefix: str) -> List[str]:
    """
    Process all CSV files in the given repository that start with the specified prefix.
    Returns list of output files created.
    """
    output_files = []
    
    try:
        if not os.path.isdir(repo_path):
            raise FileNotFoundError(f"Repository not found: {repo_path}")

        csv_files = [f for f in os.listdir(repo_path) 
                    if f.endswith('.csv') and f.startswith(prefix)]
        
        if not csv_files:
            logger.warning(f"No CSV files with prefix '{prefix}' found in {repo_path}")
            return output_files
        
        logger.info(f"Found {len(csv_files)} files with prefix '{prefix}' in {repo_path}")
        
        for csv_file in csv_files:
            file_path = os.path.join(repo_path, csv_file)
            try:
                output_file = process_file(file_path)
                output_files.append(output_file)
            except Exception as e:
                logger.error(f"Skipping file {csv_file} due to error: {str(e)}")
                continue
                
        return output_files

    except Exception as e:
        logger.error(f"Error processing repository {repo_path}: {str(e)}")
        raise

def main() -> None:
    """
    Main function to process multiple repositories with comprehensive error handling.
    """
    repo_paths = [
        # '/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/pg_2025-04-29-19-02-50',
        # '/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/pg_2025-04-30-05-34-55',
        #'/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/model_2025-04-29-14', #ART MLE 10
        #'/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/model_2025-04-02-13', #Ablated MLE 10
        #'/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/pg_2025-05-03-08-11-40', #ART PG -2
        #'/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/pg_2025-05-03-12-47-35', #Ablated PG-2
        #'/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/pg_2025-05-03-12-47-35', #ABLATED
        #'/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/model_2025-04-02-13', #MLE
        '/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/pg_2025-05-04-11-29-52', # ART SCST
        '/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/pg_2025-05-04-06-43-50', # ABLATED SCST

    ]   

    all_output_files = []
    
    try:
        logger.info("Starting metric calculation process")
        
        # Process validation files
        for repo in repo_paths:
            logger.info(f"\n{'#'*50}\nProcessing validation files in {repo}\n{'#'*50}")
            outputs = process_repository(repo, prefix='validation_details_')
            all_output_files.extend(outputs or [])
        
        # Process test files
        for repo in repo_paths:
            logger.info(f"\n{'#'*50}\nProcessing test files in {repo}\n{'#'*50}")
            outputs = process_repository(repo, prefix='test_details_')
            all_output_files.extend(outputs or [])
            
        logger.info(f"\n{'*'*50}\nProcessing complete. Generated {len(all_output_files)} metrics files:")
        for f in all_output_files:
            logger.info(f"  - {f}")
            
    except Exception as e:
        logger.error(f"Fatal error in main processing: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Metric calculation process completed")

if __name__ == "__main__":
    main()
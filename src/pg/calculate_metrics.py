import os
import pandas as pd
import logging
from src.pg.utils.metrics import MetricsEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_column_name(df, possible_names):
    """Helper function to find a column name from possible variations."""
    for name in possible_names:
        if name in df.columns:
            return name
    raise KeyError(f"Could not find any of {possible_names} in DataFrame columns")

def process_data(df):
    """
    Extracts necessary columns, computes similarity metrics using MetricsEvaluator,
    and returns a DataFrame of metrics.
    Handles multiple column name variations.
    """
    # Define possible column name variations
    column_variations = {
        'generated_text': ['Generated Text', 'Generated_Text', 'generated text', 'generated_text'],
        'edited_ending': ['Edited Ending', 'Edited_Ending', 'edited ending', 'edited_ending'],
        'counterfactual': ['Counterfactual', 'counterfactual'],
        'initial': ['Initial', 'initial'],
        'premise': ['Premise', 'premise'],
        'original_ending': ['Original Ending', 'Original_Ending', 'original ending', 'original_ending']
    }

    # Get the actual column names from the DataFrame
    try:
        generated_col = get_column_name(df, column_variations['generated_text'])
        edited_col = get_column_name(df, column_variations['edited_ending'])
        counterfactual_col = get_column_name(df, column_variations['counterfactual'])
        initial_col = get_column_name(df, column_variations['initial'])
        premise_col = get_column_name(df, column_variations['premise'])
        original_col = get_column_name(df, column_variations['original_ending'])
    except KeyError as e:
        logger.error(f"Column name error: {str(e)}")
        logger.info(f"Available columns: {df.columns.tolist()}")
        raise

    # Extract data using the found column names
    generated_texts = df[generated_col].tolist()
    edited_endings = df[edited_col].tolist()
    counterfactuals = df[counterfactual_col].tolist()
    initials = df[initial_col].tolist()
    premises = df[premise_col].tolist()
    original_endings = df[original_col].tolist()

    evaluator = MetricsEvaluator()
    all_metrics = {}

    # Calculate all similarity metrics (BART, BERT, BLEU, ROUGE)
    all_metrics.update(evaluator.calculate_and_log_bart_similarity(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    ))
    all_metrics.update(evaluator.calculate_and_log_bert_similarity(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    ))
    all_metrics.update(evaluator.calculate_and_log_bleu_scores(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    ))
    all_metrics.update(evaluator.calculate_and_log_rouge_scores(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    ))

    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index', columns=['Score'])
    metrics_df.reset_index(inplace=True)
    metrics_df.columns = ['Metric', 'Score']
    return metrics_df

def process_file(file_path):
    """
    Process a single CSV file:
      - Reads the file.
      - Calculates similarity metrics.
      - Saves the output file in the same directory as the input file with suffix '_metrics.csv'.
    """
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Processing file: {file_path}")
            logger.info(f"Columns found: {df.columns.tolist()}")
            
            metrics_df = process_data(df)

            base_dir = os.path.dirname(file_path)
            base_name, ext = os.path.splitext(os.path.basename(file_path))
            output_file_path = os.path.join(base_dir, f'{base_name}_metrics{ext}')
            metrics_df.to_csv(output_file_path, index=False)
            logger.info(f"Metrics saved to {output_file_path}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
    else:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

def process_repository(repo_path, prefix):
    """
    Process all CSV files in the given repository that start with the specified prefix.
    The output metric files will be saved in the same repository.
    """
    if os.path.isdir(repo_path):
        csv_files = [f for f in os.listdir(repo_path) 
                    if f.endswith('.csv') and f.startswith(prefix)]
        
        if not csv_files:
            logger.warning(f"No CSV files with prefix '{prefix}' found in {repo_path}")
            return
        
        for csv_file in csv_files:
            file_path = os.path.join(repo_path, csv_file)
            try:
                process_file(file_path)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                continue
    else:
        logger.error(f"Repository not found: {repo_path}")
        raise FileNotFoundError(f"Repository not found: {repo_path}")

def main():
    """
    Main function to process multiple repositories.
    For each repository, you specify a prefix to select the files you want.
    """
    repo_paths = [
        '/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/ppo_2025-04-05-18',
    ]

    try:
        # Process validation files
        for repo in repo_paths:
            logger.info(f"Processing validation files in {repo}")
            process_repository(repo, prefix='validation_details_')
        
        # Process test files
        for repo in repo_paths:
            logger.info(f"Processing test files in {repo}")
            process_repository(repo, prefix='test_details_')
    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
import os
import pandas as pd
import logging
import torch
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


def process_data_line_by_line(df):
    """
    Processes each line in the DataFrame individually, calculates both ROUGE-L and BART scores
    for the generated text, and adds them to the same row in new columns.
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
    except KeyError as e:
        logger.error(f"Column name error: {str(e)}")
        logger.info(f"Available columns: {df.columns.tolist()}")
        raise

    # Initialize the metrics evaluator
    evaluator = MetricsEvaluator()

    # Create new columns for scores
    df['ROUGE_L_Score'] = 0.0
    df['BART_Score'] = 0.0

    # Process each row individually
    for idx, row in df.iterrows():
        generated_text = str(row[generated_col])
        edited_ending = str(row[edited_col])

        # Skip empty texts
        if not generated_text.strip() or not edited_ending.strip():
            df.at[idx, 'ROUGE_L_Score'] = 0.0
            df.at[idx, 'BART_Score'] = 0.0
            continue

        # Calculate ROUGE-L score
        try:
            rouge_scores = evaluator.rouge.get_scores([generated_text], [edited_ending])
            rouge_l_f1 = rouge_scores[0]['rouge-l']['f']
            df.at[idx, 'ROUGE_L_Score'] = rouge_l_f1
        except Exception as e:
            logger.error(f"Error calculating ROUGE-L for row {idx}: {str(e)}")
            df.at[idx, 'ROUGE_L_Score'] = 0.0

        # Calculate BART score
        try:
            if evaluator.bart_scorer:
                bart_score = evaluator.bart_scorer.score([generated_text], [edited_ending])[0]
                df.at[idx, 'BART_Score'] = bart_score
        except Exception as e:
            logger.error(f"Error calculating BART score for row {idx}: {str(e)}")
            df.at[idx, 'BART_Score'] = 0.0

    return df

def process_file(file_path):
    """
    Process a single CSV file:
      - Reads the file.
      - Calculates ROUGE-L and BART scores for each generated text.
      - Saves the output to a new file with '_scored' suffix.
    """
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Processing file: {file_path}")
            logger.info(f"Columns found: {df.columns.tolist()}")

            # Process the data line by line
            df_with_scores = process_data_line_by_line(df)

            # Create new output filename
            base, ext = os.path.splitext(file_path)
            output_path = f"{base}_scored{ext}"

            # Save to new file
            df_with_scores.to_csv(output_path, index=False)
            logger.info(f"Scores saved to new file: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
    else:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")


def process_repository(repo_path, prefix):
    """
    Process all CSV files in the given repository that start with the specified prefix.
    Creates new output files with '_scored' suffix.
    """
    if os.path.isdir(repo_path):
        csv_files = [f for f in os.listdir(repo_path)
                     if f.endswith('.csv') and f.startswith(prefix)]

        if not csv_files:
            logger.warning(f"No CSV files with prefix '{prefix}' found in {repo_path}")
            return []

        output_files = []
        for csv_file in csv_files:
            file_path = os.path.join(repo_path, csv_file)
            try:
                output_path = process_file(file_path)
                output_files.append(output_path)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                continue
        return output_files
    else:
        logger.error(f"Repository not found: {repo_path}")
        raise FileNotFoundError(f"Repository not found: {repo_path}")

def main():
    """
    Main function to process a specific CSV file.
    """
    file_path = '/data/agirard/Projects/TimeTravel-PolicyGradientRL/src/pg/reward.csv'
    
    try:
        logger.info(f"Processing file: {file_path}")
        process_file(file_path)
    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
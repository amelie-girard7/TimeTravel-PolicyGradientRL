import os
import pandas as pd
import logging
from src.mle.utils.metrics import MetricsEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_data(df):
    """
    Extracts necessary columns, computes similarity metrics using MetricsEvaluator,
    and returns a DataFrame of metrics.
    """
    try:
        # Verify required columns exist
        required_columns = ['Generated Text', 'Edited Ending', 'Counterfactual', 'Initial', 'Premise', 'Original Ending']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in data: {missing_cols}")
            return None

        # Extract data with correct parameter names for MetricsEvaluator
        generated_texts = df['Generated Text'].tolist()
        edited_endings = df['Edited Ending'].tolist()
        counterfactuals = df['Counterfactual'].tolist()
        initials = df['Initial'].tolist()
        premises = df['Premise'].tolist()
        original_endings = df['Original Ending'].tolist()

        evaluator = MetricsEvaluator()
        all_metrics = {}

        # Calculate metrics with correct parameter passing
        # BART similarity
        all_metrics.update(evaluator.calculate_and_log_bart_similarity(
            generated_texts, edited_endings, counterfactuals,
            initials, premises, original_endings, logger
        ))

        # BERT similarity
        all_metrics.update(evaluator.calculate_and_log_bert_similarity(
            generated_texts, edited_endings, counterfactuals,
            initials, premises, original_endings, logger
        ))

        # BLEU scores
        all_metrics.update(evaluator.calculate_and_log_bleu_scores(
            generated_texts, edited_endings, counterfactuals,
            initials, premises, original_endings, logger
        ))

        # ROUGE scores
        all_metrics.update(evaluator.calculate_and_log_rouge_scores(
            generated_texts, edited_endings, counterfactuals,
            initials, premises, original_endings, logger
        ))

        if not all_metrics:
            logger.error("No metrics calculated")
            return None

        metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index', columns=['Score'])
        metrics_df.reset_index(inplace=True)
        metrics_df.columns = ['Metric', 'Score']
        return metrics_df

    except Exception as e:
        logger.error(f"Error in process_data: {str(e)}")
        return None


def process_file(file_path):
    """
    Process a single CSV file:
      - Reads the file.
      - Calculates similarity metrics.
      - Saves the output file in the same directory as the input file with suffix '_metrics.csv'.
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False

        logger.info(f"Processing file: {file_path}")
        df = pd.read_csv(file_path)

        if df.empty:
            logger.error("Input CSV file is empty")
            return False

        metrics_df = process_data(df)
        if metrics_df is None:
            logger.error("No metrics generated")
            return False

        # Create output path
        base_dir = os.path.dirname(file_path)
        base_name, ext = os.path.splitext(os.path.basename(file_path))
        output_file_path = os.path.join(base_dir, f'{base_name}_metrics{ext}')

        # Save results
        metrics_df.to_csv(output_file_path, index=False)
        logger.info(f"Successfully saved metrics to: {output_file_path}")
        logger.debug(f"File contents:\n{metrics_df.to_string()}")
        return True

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return False


def process_repository(repo_path, prefix):
    """
    Process all CSV files in the given repository that start with the specified prefix.
    The output metric files will be saved in the same repository.
    """
    try:
        if not os.path.isdir(repo_path):
            logger.error(f"Repository not found: {repo_path}")
            return False

        # List all CSV files in the repository that start with the given prefix
        csv_files = [f for f in os.listdir(repo_path)
                     if f.endswith('.csv') and f.startswith(prefix)]

        if not csv_files:
            logger.warning(f"No CSV files with prefix '{prefix}' found in {repo_path}")
            return False

        success_count = 0
        for csv_file in csv_files:
            file_path = os.path.join(repo_path, csv_file)
            if process_file(file_path):
                success_count += 1

        logger.info(f"Processed {success_count}/{len(csv_files)} files successfully")
        return success_count > 0

    except Exception as e:
        logger.error(f"Error processing repository {repo_path}: {str(e)}")
        return False


def main():
    """
    Main function to process multiple repositories.
    For each repository, you specify a prefix to select the files you want.
    The output file is saved in the same directory as the input file.
    """
    # List of repository directories to process
    repo_paths = [
        '/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/model_2024-03-22-10', # MLE6
        # Add other repositories as needed
    ]

    # Process validation files
    logger.info("Processing validation files...")
    for repo in repo_paths:
        process_repository(repo, prefix='validation_details')

    # Process test files
    logger.info("Processing test files...")
    for repo in repo_paths:
        process_repository(repo, prefix='test_details')


if __name__ == "__main__":
    main()
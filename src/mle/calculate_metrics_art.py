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
    and returns a DataFrame of metrics with epoch information.
    """
    try:
        # Verify required columns exist
        required_columns = ['Epoch', 'Premise', 'Initial', 'Counterfactual', 
                          'Original Ending', 'Edited Ending', 'Generated Text']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in data: {missing_cols}")
            return None

        # Initialize MetricsEvaluator
        evaluator = MetricsEvaluator()
        results = []

        # Process each row individually to preserve epoch information
        for _, row in df.iterrows():
            metrics = {'Epoch': row['Epoch']}
            
            # Handle empty Original Ending
            original_ending = row['Original Ending'] if pd.notna(row['Original Ending']) else ""
            
            # Calculate metrics for this row
            metrics.update(evaluator.calculate_and_log_bart_similarity(
                [row['Generated Text']], 
                [row['Edited Ending']], 
                [row['Counterfactual']],
                [row['Initial']], 
                [row['Premise']], 
                [original_ending],
                logger
            ))

            metrics.update(evaluator.calculate_and_log_bert_similarity(
                [row['Generated Text']], 
                [row['Edited Ending']], 
                [row['Counterfactual']],
                [row['Initial']], 
                [row['Premise']], 
                [original_ending],
                logger
            ))

            metrics.update(evaluator.calculate_and_log_bleu_scores(
                [row['Generated Text']], 
                [row['Edited Ending']], 
                [row['Counterfactual']],
                [row['Initial']], 
                [row['Premise']], 
                [original_ending],
                logger
            ))

            metrics.update(evaluator.calculate_and_log_rouge_scores(
                [row['Generated Text']], 
                [row['Edited Ending']], 
                [row['Counterfactual']],
                [row['Initial']], 
                [row['Premise']], 
                [original_ending],
                logger
            ))

            results.append(metrics)

        if not results:
            logger.error("No metrics calculated")
            return None

        # Convert to DataFrame and reshape
        metrics_df = pd.DataFrame(results)
        
        # Melt the DataFrame to get metric names in one column and scores in another
        id_vars = ['Epoch']
        value_vars = [col for col in metrics_df.columns if col not in id_vars]
        
        metrics_long = pd.melt(
            metrics_df, 
            id_vars=id_vars, 
            value_vars=value_vars,
            var_name='Metric',
            value_name='Score'
        )
        
        # Add additional aggregated views
        epoch_avg = metrics_long.groupby(['Epoch', 'Metric']).mean().reset_index()
        pivot_table = metrics_long.pivot(index='Metric', columns='Epoch', values='Score')
        
        return {
            'detailed_metrics': metrics_long,
            'epoch_averages': epoch_avg,
            'pivot_table': pivot_table
        }

    except Exception as e:
        logger.error(f"Error in process_data: {str(e)}")
        return None


def process_file(file_path):
    """
    Process a single CSV file:
      - Reads the file.
      - Calculates similarity metrics.
      - Saves three output files:
        1. Detailed metrics (long format)
        2. Epoch averages
        3. Pivot table view
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

        metrics_results = process_data(df)
        if metrics_results is None:
            logger.error("No metrics generated")
            return False

        # Create output directory if it doesn't exist
        base_dir = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(base_dir, f"{base_name}_metrics")
        os.makedirs(output_dir, exist_ok=True)

        # Save all results
        metrics_results['detailed_metrics'].to_csv(
            os.path.join(output_dir, 'detailed_metrics.csv'), 
            index=False
        )
        metrics_results['epoch_averages'].to_csv(
            os.path.join(output_dir, 'epoch_averages.csv'), 
            index=False
        )
        metrics_results['pivot_table'].to_csv(
            os.path.join(output_dir, 'metrics_pivot_table.csv')
        )

        logger.info(f"Successfully saved metrics to: {output_dir}")
        logger.debug(f"Sample detailed metrics:\n{metrics_results['detailed_metrics'].head().to_string()}")
        return True

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return False


def process_repository(repo_path, prefix):
    """
    Process all CSV files in the given repository that start with the specified prefix.
    Creates organized output directories for each processed file.
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
    Creates organized output structure for each input file.
    """
    # List of repository directories to process
    repo_paths = [
        '/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/model_2025-04-03-10', # MLE6 Ablated
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

    logger.info("Processing completed.")


if __name__ == "__main__":
    main()
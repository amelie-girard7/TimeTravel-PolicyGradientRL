import os
import pandas as pd
import logging
from typing import Dict, List, Optional, Union
from src.mle.utils.metrics import MetricsEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('metrics_calculation.log')
    ]
)
logger = logging.getLogger(__name__)

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Validate the input DataFrame has required columns and data."""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Check for empty/missing critical text columns
    text_cols = ['Generated Text', 'Edited Ending', 'Counterfactual']
    for col in text_cols:
        if df[col].isna().any():
            logger.warning(f"Column '{col}' contains missing values")

def clean_text_data(text: Union[str, float]) -> str:
    """Ensure text data is properly formatted as string."""
    if pd.isna(text):
        return ""
    if isinstance(text, float):
        return str(int(text)) if text.is_integer() else str(text)
    return str(text)

def safe_metric_calculation(evaluator, texts: Dict[str, str]) -> Dict[str, float]:
    """Safely calculate metrics with error handling for each metric type."""
    metrics = {}
    
    # BART Similarity
    try:
        bart_metrics = evaluator.calculate_and_log_bart_similarity(
            [texts['generated']],
            [texts['edited']],
            [texts['counterfactual']],
            [texts['initial']],
            [texts['premise']],
            [texts['original']],
            logger
        )
        metrics.update(bart_metrics)
    except Exception as e:
        logger.error(f"BART calculation failed: {str(e)}")
    
    # ROUGE Scores
    try:
        rouge_metrics = evaluator.calculate_and_log_rouge_scores(
            [texts['generated']],
            [texts['edited']],
            [texts['counterfactual']],
            [texts['initial']],
            [texts['premise']],
            [texts['original']],
            logger
        )
        metrics.update(rouge_metrics)
    except Exception as e:
        logger.error(f"ROUGE calculation failed: {str(e)}")
    
    return metrics

def process_epoch_data(df: pd.DataFrame, evaluator: MetricsEvaluator) -> Dict[str, pd.DataFrame]:
    """Process data with epoch information and return multiple metric views."""
    results = []
    
    for idx, row in df.iterrows():
        try:
            metrics = {'Epoch': row['Epoch'], 'Row_ID': idx}
            
            # Clean text data
            texts = {
                'generated': clean_text_data(row['Generated Text']),
                'edited': clean_text_data(row['Edited Ending']),
                'counterfactual': clean_text_data(row['Counterfactual']),
                'initial': clean_text_data(row['Initial']),
                'premise': clean_text_data(row['Premise']),
                'original': clean_text_data(row['Original Ending'])
            }
            
            # Calculate metrics
            metrics.update(safe_metric_calculation(evaluator, texts))
            results.append(metrics)
            
        except Exception as e:
            logger.error(f"Error processing row {idx}: {str(e)}")
            continue
    
    if not results:
        raise ValueError("No valid metrics calculated")
    
    # Create detailed metrics DataFrame
    metrics_df = pd.DataFrame(results)
    
    # Convert to numeric where possible
    numeric_cols = [col for col in metrics_df.columns if col not in ['Epoch', 'Row_ID']]
    metrics_df[numeric_cols] = metrics_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Melt to long format
    id_vars = ['Epoch', 'Row_ID']
    value_vars = [col for col in metrics_df.columns if col not in id_vars]
    detailed_metrics = pd.melt(
        metrics_df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='Metric',
        value_name='Score'
    ).dropna(subset=['Score'])
    
    # Calculate epoch averages (numeric only)
    epoch_avg = detailed_metrics.groupby(['Epoch', 'Metric'], as_index=False)['Score'].mean()
    
    # Create pivot table view
    pivot_table = detailed_metrics.pivot_table(
        index='Metric',
        columns='Epoch',
        values='Score',
        aggfunc='mean'
    )
    
    return {
        'detailed_metrics': detailed_metrics,
        'epoch_averages': epoch_avg,
        'pivot_table': pivot_table
    }

def process_file(file_path: str) -> bool:
    """Process a single CSV file with comprehensive error handling."""
    try:
        logger.info(f"\n{'='*50}\nProcessing file: {file_path}\n{'='*50}")
        
        # Read and validate input file
        df = pd.read_csv(file_path)
        required_columns = [
            'Epoch', 'Premise', 'Initial', 'Counterfactual',
            'Original Ending', 'Edited Ending', 'Generated Text'
        ]
        validate_dataframe(df, required_columns)
        
        # Initialize metrics evaluator
        evaluator = MetricsEvaluator()
        
        # Process data with epoch information
        metrics_results = process_epoch_data(df, evaluator)
        
        # Create output directory
        base_dir = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(base_dir, f"{base_name}_metrics")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save outputs
        metrics_results['detailed_metrics'].to_csv(
            os.path.join(output_dir, 'detailed_metrics.csv'),
            index=False
        )
        metrics_results['epoch_averages'].to_csv(
            os.path.join(output_dir, 'epoch_averages.csv'),
            index=False
        )
        metrics_results['pivot_table'].to_csv(
            os.path.join(output_dir, 'metrics_pivot.csv')
        )
        
        logger.info(f"Successfully processed {len(df)} rows")
        logger.info(f"Output saved to: {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        return False

def process_repository(repo_path: str, prefix: str) -> int:
    """Process all CSV files in repository matching the prefix."""
    try:
        if not os.path.isdir(repo_path):
            logger.error(f"Repository not found: {repo_path}")
            return 0
        
        csv_files = [
            f for f in os.listdir(repo_path)
            if f.endswith('.csv') and f.startswith(prefix)
        ]
        
        if not csv_files:
            logger.warning(f"No CSV files found with prefix '{prefix}' in {repo_path}")
            return 0
        
        success_count = 0
        for csv_file in csv_files:
            file_path = os.path.join(repo_path, csv_file)
            if process_file(file_path):
                success_count += 1
        
        logger.info(f"Processed {success_count}/{len(csv_files)} files successfully")
        return success_count
        
    except Exception as e:
        logger.error(f"Error processing repository: {str(e)}")
        return 0

def main():
    """Main processing workflow."""
    repo_paths = [
        '/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/model_2025-04-29-14',
    ]
    
    total_processed = 0
    
    try:
        logger.info("Starting metric calculation pipeline")
        
        # Process validation files
        logger.info("\nProcessing validation files...")
        for repo in repo_paths:
            total_processed += process_repository(repo, 'validation_details')
        
        # Process test files
        logger.info("\nProcessing test files...")
        for repo in repo_paths:
            total_processed += process_repository(repo, 'test_details')
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing completed. Total files processed: {total_processed}")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Fatal error in main pipeline: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
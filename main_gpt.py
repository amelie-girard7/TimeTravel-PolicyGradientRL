# /data/agirard/Projects/Timetravel/main_gpt.py
import os
import sys
import pandas as pd
import uuid  # Ensure this import statement is present
from src.utils.config import CONFIG
from src.utils.utils import chatgpt_zero_shot_inference, chatgpt_one_shot_inference
from src.utils.metrics import MetricsEvaluator
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the specified inference mode or just run similarity metrics.
    """
    
    # Print the current configuration options
    print(f"Inference Mode: {CONFIG['inference_mode']}")
    print(f"Run Similarities Only: {CONFIG['run_similarities_only']}")
    print(f"Example Selection: {CONFIG['example_selection']}")

    # Ensure API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    # Path to the gold data file for fine-tuning and inference
    gold_data_path = "/data/agirard/Projects/Timetravel/data/transformed/gold_without_diff.json"

    # Check if the gold data file exists
    if not os.path.exists(gold_data_path):
        print(f"Gold data file does not exist: {gold_data_path}")
        return

    # Load the gold data
    print(f"Loading gold data from: {gold_data_path}")
    gold_data = pd.read_json(gold_data_path, lines=True)
    print("Gold data loaded successfully. Sample data:")
    print(gold_data.head())  # Print the first few rows of the gold data for debugging

    results_path = None
    if not CONFIG["run_similarities_only"]:
        # Run the specified mode
        if CONFIG["inference_mode"] == "zero_shot":
            # Run zero-shot inference using ChatGPT
            print(f"Running zero-shot inference with API key: {api_key[:4]}...")  # Print the first 4 characters of the API key for debugging
            results = chatgpt_zero_shot_inference(api_key, gold_data)
            print("Zero-shot inference completed. Sample results:")
            print(results[:2])  # Print the first few results for debugging

            # Save the results to a CSV file
            results_path = "/data/agirard/Projects/Timetravel/results/zero_shot_results.csv"
        elif CONFIG["inference_mode"] == "one_shot":
            # Run one-shot inference using ChatGPT
            print(f"Running one-shot inference with API key: {api_key[:4]}...")  # Print the first 4 characters of the API key for debugging
            results = chatgpt_one_shot_inference(api_key, gold_data, CONFIG["example_selection"])
            print("One-shot inference completed. Sample results:")
            print(results[:2])  # Print the first few results for debugging

            # Save the results to a CSV file
            if CONFIG["example_selection"] == "random":
                results_path = "/data/agirard/Projects/Timetravel/results/one_shot_results_random.csv"
            else:
                results_path = "/data/agirard/Projects/Timetravel/results/one_shot_results_fixed.csv"
        else:
            print(f"Unknown mode: {CONFIG['inference_mode']}")
            return

        # Save results to CSV with correct headers
        pd.DataFrame(results).to_csv(results_path, index=False)
        print(f"Inference results saved to {results_path}")
    else:
        # If running similarities only, load the existing results
        if CONFIG["inference_mode"] == "zero_shot":
            results_path = "/data/agirard/Projects/Timetravel/results/zero_shot_results.csv"
        elif CONFIG["inference_mode"] == "one_shot":
            if CONFIG["example_selection"] == "random":
                results_path = "/data/agirard/Projects/Timetravel/results/one_shot_results_random.csv"
            else:
                results_path = "/data/agirard/Projects/Timetravel/results/one_shot_results_fixed.csv"
        else:
            print(f"Invalid mode for running similarities: {CONFIG['inference_mode']}")
            return
        
        if not os.path.exists(results_path):
            print(f"Results file does not exist: {results_path}")
            return
        
        results = pd.read_csv(results_path).to_dict('records')

        # Only run similarity metrics if specifically required
        if CONFIG["run_similarities_only"]:
            run_similarity_metrics(results, results_path)

def run_similarity_metrics(results, results_path):
    """
    Function to run similarity metrics on the generated results and save them to a transposed CSV file.
    """
    metrics_evaluator = MetricsEvaluator()

    # Extract the relevant text fields from the results
    generated_texts = [result['generated_text'] for result in results]
    counterfactuals = [result['counterfactual'] for result in results]
    initials = [result['initial'] for result in results]
    premises = [result['premise'] for result in results]
    original_endings = [result['original_ending'] for result in results]
    edited_endings = [result.get('edited_ending', '') for result in results]  # Ensure edited_endings are extracted

    # Initialize all_metrics dictionary
    all_metrics = {}

    # Calculate BART scores
    print("Calculating BART similarity scores...")
    bart_scores = metrics_evaluator.calculate_and_log_bart_similarity(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    )
    all_metrics.update(bart_scores)

    # Calculate BERT scores
    print("Calculating BERT similarity scores...")
    bert_scores = metrics_evaluator.calculate_and_log_bert_similarity(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    )
    all_metrics.update(bert_scores)

    # Calculate BLEU scores
    print("Calculating BLEU scores...")
    bleu_scores = metrics_evaluator.calculate_and_log_bleu_scores(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    )
    all_metrics.update(bleu_scores)

    # Calculate ROUGE scores
    print("Calculating ROUGE scores...")
    rouge_scores = metrics_evaluator.calculate_and_log_rouge_scores(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    )
    all_metrics.update(rouge_scores)

    # Convert the metrics to a DataFrame and transpose it
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index', columns=['Value'])
    metrics_df.reset_index(inplace=True)
    metrics_df.columns = ['Metric', 'Value']

    # Save the transposed metrics DataFrame to a CSV file
    metrics_results_path = results_path.replace(".csv", "_metrics.csv")
    metrics_df.to_csv(metrics_results_path, index=False)
    print(f"Similarity metrics saved to {metrics_results_path}")

if __name__ == "__main__":
    main()

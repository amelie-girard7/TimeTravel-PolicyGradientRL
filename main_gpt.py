import os
import sys
import pandas as pd
from src.utils.config import CONFIG
from src.utils.utils import chatgpt_zero_shot_inference, chatgpt_one_shot_inference

def main(mode, example_selection=None):
    """
    Main function to run the specified inference mode.
    """

    # Ensure API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    # Construct the path to the test file
    test_file_path = CONFIG["data_dir"] / 'transformed' / CONFIG["test_file"]

    # Check if the test file exists
    if not test_file_path.exists():
        print(f"Test file does not exist: {test_file_path}")
        return

    # Load the test data
    print(f"Loading test data from: {test_file_path}")
    test_data = pd.read_json(test_file_path, lines=True)
    print("Test data loaded successfully. Sample data:")
    print(test_data.head())  # Print the first few rows of the test data for debugging

    # Run the specified mode
    if mode == "zero_shot":
        # Run zero-shot inference using ChatGPT
        print(f"Running zero-shot inference with API key: {api_key[:4]}...")  # Print the first 4 characters of the API key for debugging
        results = chatgpt_zero_shot_inference(api_key, test_data)
        print("Zero-shot inference completed. Sample results:")
        print(results[:2])  # Print the first few results for debugging

        # Save the results to a CSV file
        results_path = CONFIG["results_dir"] / "zero_shot_results.csv"
    elif mode == "one_shot":
        # Run one-shot inference using ChatGPT
        print(f"Running one-shot inference with API key: {api_key[:4]}...")  # Print the first 4 characters of the API key for debugging
        results = chatgpt_one_shot_inference(api_key, test_data, example_selection)
        print("One-shot inference completed. Sample results:")
        print(results[:2])  # Print the first few results for debugging

        # Save the results to a CSV file
        results_path = CONFIG["results_dir"] / "one_shot_results.csv"
    else:
        print(f"Unknown mode: {mode}")
        return

    # Save results to CSV with correct headers
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"Inference results saved to {results_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_inference.py [zero_shot|one_shot] [example_selection (optional)]")
        sys.exit(1)

    mode = sys.argv[1]
    example_selection = sys.argv[2] if len(sys.argv) > 2 else None
    main(mode, example_selection)

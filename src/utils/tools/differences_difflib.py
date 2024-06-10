import json
import csv
import difflib

def identify_new_words(original_ending, edited_ending):
    """
    Identifies words that are in the edited ending but not in the original ending using difflib.

    Parameters:
    - original_ending (str): The ending as originally written.
    - edited_ending (str): The modified version of the ending.

    Returns:
    - list: A list of words that appear in the edited ending but not in the original.
    """
    # Split the endings into lists of words
    original_words = original_ending.split()
    edited_words = edited_ending.split()
    # Use SequenceMatcher to find matches
    s = difflib.SequenceMatcher(None, original_words, edited_words)
    differences = [edited_words[i] for i, j, n in s.get_opcodes() if i == 'insert' or i == 'replace']
    return differences

def process_file(input_file_path, output_file_path, output_csv_path):
    """
    Reads a JSON file, identifies new words in edited story endings using difflib, and writes results to JSON and CSV files.

    Parameters:
    - input_file_path (str): Path to the input JSON file.
    - output_file_path (str): Path to the output JSON file.
    - output_csv_path (str): Path to the output CSV file.
    """
    # Open the input JSON file, output JSON file, and output CSV file simultaneously
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile, open(output_csv_path, 'w', newline='') as csv_outfile:
        # Define CSV columns and setup the CSV writer
        fieldnames = ['story_id', 'premise', 'initial', 'counterfactual', 'original_ending', 'edited_ending', 'differences']
        writer = csv.DictWriter(csv_outfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process each line in the input JSON file
        for line in infile:
            obj = json.loads(line)
            # Convert list of edited ending into a single string if necessary
            edited_ending_str = ' '.join(obj['edited_ending']) if isinstance(obj['edited_ending'], list) else obj['edited_ending']
            # Identify new words using difflib
            differences = identify_new_words(obj['original_ending'], edited_ending_str)

            # Update the JSON object with string formatted edited ending and differences
            obj['edited_ending'] = edited_ending_str
            obj['differences'] = differences
            # Write the updated JSON object to the output file
            json.dump(obj, outfile)
            outfile.write('\n')
            
            # Prepare and write the CSV row
            csv_row = {
                'story_id': obj['story_id'],
                'premise': obj['premise'],
                'initial': obj['initial'],
                'counterfactual': obj['counterfactual'],
                'original_ending': obj['original_ending'],
                'edited_ending': edited_ending_str,
                'differences': ', '.join(differences),
            }
            writer.writerow(csv_row)

    # Print completion status
    print(f"Processed files saved as {output_file_path} and {output_csv_path}")

# Define file paths for processing
file_paths = [
    ("/data/agirard/Projects/Timetravel/data/transformed/gold_data_raw.json", "/data/agirard/Projects/Timetravel/data/transformed/gold_data_differences.json", "/data/agirard/Projects/Timetravel/data/transformed/gold_data_differences.csv"),
]

# Loop through the file paths and process each file
for input_path, output_json_path, output_csv_path in file_paths:
    process_file(input_path, output_json_path, output_csv_path)

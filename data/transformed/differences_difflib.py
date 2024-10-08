import json
import csv
import difflib
import os

def identify_new_words(original_ending, edited_ending):
    """
    Identifies words that are in the edited ending but not in the original ending using difflib.

    Parameters:
    - original_ending (str): The ending as originally written.
    - edited_ending (str): The modified version of the ending.

    Returns:
    - list: A list of words that appear in the edited ending but not in the original.
    """
    original_words = original_ending.split()
    edited_words = edited_ending.split()
    s = difflib.SequenceMatcher(None, original_words, edited_words)
    differences = []
    
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag in ('insert', 'replace'):
            differences.extend(edited_words[j1:j2])
    
    return differences

def process_file(input_file_path, output_file_path, output_csv_path):
    """
    Reads a JSON file, identifies new words in edited story endings using difflib, and writes results to JSON and CSV files.

    Parameters:
    - input_file_path (str): Path to the input JSON file.
    - output_file_path (str): Path to the output JSON file.
    - output_csv_path (str): Path to the output CSV file.
    """
    try:
        with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile, open(output_csv_path, 'w', newline='') as csv_outfile:
            fieldnames = ['story_id', 'premise', 'initial', 'counterfactual', 'original_ending', 'edited_ending', 'differences']
            writer = csv.DictWriter(csv_outfile, fieldnames=fieldnames)
            writer.writeheader()

            for line in infile:
                obj = json.loads(line)
                edited_ending_str = ' '.join(obj['edited_ending']) if isinstance(obj['edited_ending'], list) else obj['edited_ending']
                differences = identify_new_words(obj['original_ending'], edited_ending_str)

                obj['edited_ending'] = edited_ending_str
                obj['differences'] = differences
                json.dump(obj, outfile)
                outfile.write('\n')
                
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

        print(f"Processed files saved as {output_file_path} and {output_csv_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

file_paths = [
    ("/data/agirard/Projects/Timetravel/data/raw/single_edited_endings/test_data.json", 
     "/data/agirard/Projects/Timetravel/data/transformed/test_data.json", 
     "/data/agirard/Projects/Timetravel/data/transformed/test_data.csv"),
]

for input_path, output_json_path, output_csv_path in file_paths:
    process_file(input_path, output_json_path, output_csv_path)

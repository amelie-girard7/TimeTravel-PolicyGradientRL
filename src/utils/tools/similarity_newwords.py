import json
import csv

def identify_new_words(original_ending, edited_ending):
    """Identifies words that are in the edited ending but not in the original ending."""
    original_words = set(original_ending.split())
    edited_words = set(edited_ending.split())
    differences = edited_words - original_words
    return list(differences)

def process_file(input_file_path, output_file_path, output_csv_path):
    """Reads JSON data and identifies new words in edited endings."""
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

# Example file paths
file_paths = [
    ("/data/agirard/Projects/Timetravel/data/transformed/gold_data_raw.json", "/data/agirard/Projects/Timetravel/data/transformed/gold_data_differences.json", "/data/agirard/Projects/Timetravel/data/transformed/gold_data_differences.csv"),
]

for input_path, output_json_path, output_csv_path in file_paths:
    process_file(input_path, output_json_path, output_csv_path)

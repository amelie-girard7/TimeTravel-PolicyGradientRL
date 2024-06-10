import json
import csv

def generate_ngrams(words_list, n):
    """Generate n-grams from a list of words."""
    return [' '.join(words_list[i:i + n]) for i in range(len(words_list) - n + 1)]

def identify_differences(original_ending, edited_ending, n=2):
    """Identifies n-gram differences between the edited and original endings."""
    original_ngrams = set(generate_ngrams(original_ending.split(), n))
    edited_ngrams = set(generate_ngrams(edited_ending.split(), n))
    differences = [ngram for ngram in edited_ngrams if ngram not in original_ngrams]
    return differences

def process_file(input_file_path, output_file_path, output_csv_path, n=2):
    """Processes JSON data, identifying n-gram differences and saving results."""
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile, open(output_csv_path, 'w', newline='') as csv_outfile:
        fieldnames = ['story_id', 'premise', 'initial', 'counterfactual', 'original_ending', 'edited_ending', 'differences']
        writer = csv.DictWriter(csv_outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for line in infile:
            obj = json.loads(line)
            edited_ending_str = ' '.join(obj['edited_ending']) if isinstance(obj['edited_ending'], list) else obj['edited_ending']
            differences = identify_differences(obj['original_ending'], edited_ending_str, n)
            
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

# Define n-gram degree and file paths as needed
n = 2  # You can adjust this value to change the n-gram size
file_paths = [
    ("/data/agirard/Projects/Timetravel/data/transformed/gold_data_raw.json", "/data/agirard/Projects/Timetravel/data/transformed/gold_data_n2.json", "/data/agirard/Projects/Timetravel/data/transformed/gold_data_n2.csv"),
]

for input_path, output_json_path, output_csv_path in file_paths:
    process_file(input_path, output_json_path, output_csv_path, n)

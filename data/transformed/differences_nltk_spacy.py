import json
import spacy
import nltk
import csv
from nltk.corpus import wordnet as wn

# Load spaCy's large English Language model
nlp = spacy.load("en_core_web_lg")

# Ensure NLTK data is available
nltk.download('wordnet')
nltk.download('omw-1.4')

def are_synonyms_spacy(word1, word2):
    """Checks for semantic similarity using spaCy."""
    token1 = nlp(word1)
    token2 = nlp(word2)
    return token1.similarity(token2) > 0.8  # Adjust threshold as needed

def are_synonyms_nltk(word1, word2):
    """Checks if two words are synonyms using WordNet."""
    word1_synsets = wn.synsets(word1)
    word2_synsets = wn.synsets(word2)
    return any(s1 == s2 for s1 in word1_synsets for s2 in word2_synsets)

def identify_differences(original_ending, edited_ending):
    """Identifies words in the edited_ending not in the original_ending, considering synonyms and semantic similarity."""
    original_words = set(original_ending.split())
    edited_words = set(edited_ending.split())
    differences = [
        word for word in edited_words 
        if word not in original_words and not any(
            are_synonyms_nltk(word, orig_word) or are_synonyms_spacy(word, orig_word) 
            for orig_word in original_words
        )
    ]
    return differences

def process_file(input_file_path, output_file_path, output_csv_path):
    """Reads, processes, and writes the JSON data for a given file and generates a corresponding CSV file with a differences column."""
    try:
        with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile, open(output_csv_path, 'w', newline='') as csv_outfile:
            fieldnames = ['story_id', 'premise', 'initial', 'counterfactual', 'original_ending', 'edited_ending', 'differences']
            writer = csv.DictWriter(csv_outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for line in infile:
                obj = json.loads(line)
                # Handle edited_ending being a list of sentences or a single string
                edited_ending_str = ' '.join(obj['edited_ending']) if isinstance(obj['edited_ending'], list) else obj['edited_ending']
                differences = identify_differences(obj['original_ending'], edited_ending_str)
                
                # Write JSON output
                obj['edited_ending'] = edited_ending_str
                obj['differences'] = differences
                json.dump(obj, outfile)
                outfile.write('\n')
                
                # Prepare CSV data with the new differences column
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

# Example file paths
file_paths = [
    ("/data/agirard/Projects/Timetravel/data/raw/single_edited_endings/test_data.json", 
     "/data/agirard/Projects/Timetravel/data/transformed/test_data.json", 
     "/data/agirard/Projects/Timetravel/data/transformed/test_data.csv"),
]

for input_path, output_json_path, output_csv_path in file_paths:
    process_file(input_path, output_json_path, output_csv_path)
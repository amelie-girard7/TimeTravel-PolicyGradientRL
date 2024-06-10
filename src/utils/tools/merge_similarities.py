import csv
import os

def merge_csv_files(file_paths, output_file_path):
    # Dictionary to hold data, keyed first by story_id then by row index
    combined_data = {}

    # List to store final headers
    headers = ['story_id', 'premise', 'initial', 'counterfactual', 'original_ending', 'edited_ending']

    for file_index, file_path in enumerate(file_paths):
        differences_column_name = os.path.splitext(os.path.basename(file_path))[0] + '_differences'
        # Ensure the new differences column is added to headers
        if differences_column_name not in headers:
            headers.append(differences_column_name)

        with open(file_path, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            differences_column = 'differences' if 'differences' in reader.fieldnames else None

            for row_index, row in enumerate(reader):
                story_id = row['story_id']
                key = (story_id, row_index)  # Combine story_id and row index for unique identification

                # Initialize dictionary for new story_id-row_index pair not seen before
                if key not in combined_data:
                    combined_data[key] = {h: '' for h in headers}
                    combined_data[key].update({
                        'story_id': story_id,
                        'premise': row.get('premise', ''),
                        'initial': row.get('initial', ''),
                        'counterfactual': row.get('counterfactual', ''),
                        'original_ending': row.get('original_ending', ''),
                        'edited_ending': row.get('edited_ending', ''),
                    })

                # Add or update the differences in the specific column
                if differences_column:
                    combined_data[key][differences_column_name] = row[differences_column]

    # Write the combined data to a new CSV file
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for key in sorted(combined_data):  # Sort by story_id and row index
            writer.writerow(combined_data[key])

# List of file paths
file_paths = [
    '/data/agirard/Projects/Timetravel/data/transformed/gold_data_075.csv',
    '/data/agirard/Projects/Timetravel/data/transformed/gold_data_099.csv',
    '/data/agirard/Projects/Timetravel/data/transformed/gold_data_new_words.csv'
]

# Specify the path for the output combined CSV
output_file_path = '/data/agirard/Projects/Timetravel/data/transformed/gold_data_combined.csv'

merge_csv_files(file_paths, output_file_path)

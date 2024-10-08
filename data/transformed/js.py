import json

# Define the input (original) and output (new) file paths
input_file_path = '/data/agirard/Projects/Timetravel/data/transformed/gold_data.json'
output_file_path = '/data/agirard/Projects/Timetravel/data/transformed/gold_without_diff.json'

# Read the original file without modifying it
with open(input_file_path, 'r', encoding='utf-8') as infile:
    # Load each line in the file as a separate JSON object
    data = [json.loads(line) for line in infile]

# Process the data in memory: remove the 'differences' field from each entry
for entry in data:
    if 'differences' in entry:
        del entry['differences']  # Remove the 'differences' key

# Write the modified data to a new file
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    for entry in data:
        json.dump(entry, outfile)  # Write each entry as a JSON object
        outfile.write('\n')  # Ensure each JSON object is on a new line

print(f"File saved to {output_file_path} without 'differences' column.")

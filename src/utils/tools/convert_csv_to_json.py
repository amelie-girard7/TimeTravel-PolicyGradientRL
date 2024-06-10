import pandas as pd

def convert_csv_to_json(csv_file_path, json_file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Print DataFrame shape and preview to ensure it's loaded correctly
    print("DataFrame shape:", df.shape)
    print("First few rows:", df.head())

    # Handle 'differences' column: convert string-separated values into lists
    # If 'differences' might contain NaN, handle it to prevent errors
    #df['differences'] = df['differences'].apply(lambda x: [item.strip() for item in x.split(',')] if isinstance(x, str) else [])

    # Save the DataFrame to a JSON file
    # orient='records' makes each DataFrame row a JSON object
    # lines=True writes each object on a new line with field names included in each line
    df.to_json(json_file_path, orient='records', lines=True)

    print(f"Converted {csv_file_path} to {json_file_path}")

if __name__ == "__main__":
    csv_file_path = '/data/agirard/Projects/Timetravel/data/transformed/gold_data_raw.csv'  # Ensure this is the correct path
    json_file_path = '/data/agirard/Projects/Timetravel/data/transformed/gold_data_raw.json'  # Ensure this is the correct path
    convert_csv_to_json(csv_file_path, json_file_path)

import pandas as pd

# Load your CSV file
csv_path = "/data/agirard/Projects/Timetravel/results/zero_shot_results.csv"
data = pd.read_csv(csv_path)

# Print the column names
print("Columns in CSV:", data.columns)

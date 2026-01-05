import os
import pandas as pd

# Get current script directory (both input and output will use this)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set input folder to current script directory
input_folder = BASE_DIR

# Set output file path in the same directory
output_file = os.path.join(BASE_DIR, "fever_dev_combined.csv")

# Iterate through all CSV files in the folder, read them, and store them in a list
all_dfs = []
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)
        all_dfs.append(df)

# Concatenate all DataFrames
combined_df = pd.concat(all_dfs, ignore_index=True)

# Save as a new CSV file
combined_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"✅ Merge completed! Total records: {len(combined_df)}, saved to:\n{output_file}")


import os

import pandas as pd


def find_csv_files(directory):
    """
    Recursively find all CSV files in directory and its subdirectories.

    Args:
        directory (str): Root directory to search for CSV files

    Returns:
        list: List of paths to all CSV files found
    """
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))

    csv_files = sorted(csv_files)  # !!! must sorted
    return csv_files


def concatenate_csv_files(file_list, output_file):
    # List to hold DataFrames
    dataframes = []
    print(f"Concat {len(file_list)} CSV files")

    # Iterate through all CSV files in the input folder
    for file_path in sorted(file_list):
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)  # Read the CSV file
            dataframes.append(df)  # Append DataFrame to the list
            print(f"- Reading {os.path.basename(file_path)}")

    # Concatenate all DataFrames in the list
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    # Save the concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(output_file, index=False)
    print(f"Concatenated CSV saved to: {output_file}")


if __name__ == "__main__":
    # Get all CSV files in the outputs folder and its subfolders
    output_dir = "outputs"  # Fixed
    files_to_combine = find_csv_files(output_dir)
    concatenate_csv_files(files_to_combine, "test.csv")  # Fixed

    tmp_dir = "tmps"  # Fixed
    files_to_combine = find_csv_files(tmp_dir)
    concatenate_csv_files(files_to_combine, "valid.csv")  # Fixed

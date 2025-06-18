"""
Convert output.csv to final submission format with class mappings
"""

import os

import enter
import pandas as pd
from unidecode import unidecode

# Class mappings
CLASS_MAPPINGS = {
    'bào ngư xám + trắng': 1,  # Nấm bào ngư
    'Đùi gà Baby (cắt ngắn)': 2,  # Nấm đùi gà
    'linh chi trắng': 3,  # nấm linh chi trắng
    'nấm mỡ': 0,  # nấm mỡ
}


def vietnamese_to_english(text):
    """
    Convert Vietnamese text to English and lowercase it.

    Args:
        text (str): Vietnamese text to convert

    Returns:
        str: Converted English text in lowercase
    """
    if not isinstance(text, str):
        return text
    # Convert Vietnamese characters to ASCII
    text = unidecode(text)
    # Convert to lowercase
    return text.lower().strip()


tmp = list(CLASS_MAPPINGS.items())
for k, v in tmp:
    CLASS_MAPPINGS[vietnamese_to_english(k)] = v

print(CLASS_MAPPINGS)


def list_csv_files(folder_path):
    """
    List all CSV files in the specified folder.

    Args:
        folder_path (str): Path to the folder to search

    Returns:
        list: List of CSV file paths
    """
    csv_files = []
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                csv_files.append(os.path.join(folder_path, file))
    return csv_files


def get_final_csv():
    csv_files = list_csv_files(enter.FINAL_TO_USE_DIR)
    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in {enter.FINAL_TO_USE_DIR}")
    if len(csv_files) > 1:
        raise ValueError(
            f"Multiple CSV files found in {enter.FINAL_TO_USE_DIR}: {csv_files}")
    print(f"Using {csv_files[0]}")
    return csv_files[0]


def convert_output(input_file, output_file='output/result.csv'):
    """Convert output.csv to final submission format"""
    # Read input file
    df = pd.read_csv(input_file)

    # Extract image ID (remove .jpg)
    # df['id'] = df['image'].str.replace('.jpg', '') # !!! Must new format
    df['image_name'] = df['image']

    df['label'] = df['label'].apply(vietnamese_to_english)

    # Map labels to class numbers
    # df['type'] = df['label'].map(CLASS_MAPPINGS)
    df['label'] = df['label'].map(CLASS_MAPPINGS)

    # Select and reorder columns
    df = df[['image_name', 'label']]

    # ! Sort by id column
    df = df.sort_values('image_name')
    # Save to output file
    df.to_csv(output_file, index=False)
    print(f"Converted {len(df)} records")
    print(f"Saved to {output_file}")

    # Print sample of converted data
    print("\nSample of converted data:")
    print(df.head())


if __name__ == "__main__":
    convert_output(input_file=get_final_csv())

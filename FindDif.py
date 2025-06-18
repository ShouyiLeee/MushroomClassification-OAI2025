import pandas as pd
import os
from PIL import Image

# Load the two CSV files
file1 = "C:/Users/Acer/OneDrive/Desktop/result.csv"  # Replace with your first file path
file2 = "C:/Users/Acer/OneDrive/Desktop/result-QT.csv"  # Replace with your second file path

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Merge the two DataFrames on the 'id' column
merged_df = pd.merge(df1, df2, on='image_name', suffixes=('_file1', '_file2'))

# Find rows where the 'type' values differ
diff_df = merged_df[merged_df['label_file1'] != merged_df['label_file2']]

# Print the differences
print(diff_df)

# # Save the differences to a new CSV file
# output_file = "D:\\IT\\GITHUB\\Hutech-AI-Challenge\\PatternFinding\\differences91-99.csv"  # Replace with your desired output file path
# diff_df.to_csv(output_file, index=False)
# print(f"Differences saved to {output_file}")
# root = "D:\\IT\\GITHUB\\Hutech-AI-Challenge\\data\\test"  # Replace with your image directory path

# Iterate through the rows of the diff_df and plot the images
# for _, row in diff_df.iterrows():
#     image_id = row['id']
#     image_path = os.path.join(root, f"{int(image_id):03}.jpg")  # Format the ID as a zero-padded 3-digit number

#     if os.path.exists(image_path):
#         image = Image.open(image_path)
#         image.show()
#         input("Press Enter to continue...")
#     else:
#         print(f"Image {image_path} not found.")


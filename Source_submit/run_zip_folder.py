import os
import shutil

# Define the folder to zip and the output zip file name
folder_to_zip = os.getcwd()
output_zip_file = 'aio'  # This will create libs_archive.zip

# Create a zip archive
shutil.make_archive(output_zip_file, 'zip', folder_to_zip)

print(f"Successfully created {output_zip_file}.zip")

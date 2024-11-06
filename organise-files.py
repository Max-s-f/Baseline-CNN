import os
import re

def rename_files_in_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            # Check if the file matches the pattern
            match = re.match(r"MLS-Aura_L3DB-([A-Za-z0-9]+)_v[0-9]{2}-[0-9]{2}-c[0-9]{2}_(\d{4})(\d{3})\.nc", file_name)
            print(match)
            if match:
                # Extract the components from the file name
                measurement = match.group(1)
                year = match.group(2)
                day_of_year = match.group(3)
                
                # Create the new file name
                new_file_name = f"{measurement}{year}d{day_of_year}.nc"
                
                # Create full paths
                old_file_path = os.path.join(root, file_name)
                new_file_path = os.path.join(root, new_file_name)
                
                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f"Renamed {old_file_path} to {new_file_path}")

if __name__ == "__main__":
    # Replace with the path to the top-level directory you want to search
    top_level_directory = r'\Users\simma362\Desktop\Code\Baseline-CNN\Data\Temperature'
    rename_files_in_directory(top_level_directory)

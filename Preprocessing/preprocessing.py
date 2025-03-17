import pandas as pd
import fileinput
import re
import os

def convert_date_format(filename):
# Convert date format from DD.MM.YYYY to YYYY.MM.DD in the filename
    match = re.match(r"(\d{2})\.(\d{2})\.(\d{4})", filename)  # Finds date in the file name
    if match:
        day, month, year = match.groups()
        return f"{year}.{month}.{day} " + filename[len(match.group(0)):]  # Keeps the rest of the filename
    return filename  # Returns date if nothing is changed

def find_next_available_index(folder_path, prefix):
    # Finds next available index for files with a given prefix
    existing_numbers = []

    for f in os.listdir(folder_path):
        match = re.match(rf"^{prefix}_(\d+)\.txt$", f)
        if match:
            existing_numbers.append(int(match.group(1)))

    if not existing_numbers:
        return 0  # Start at 0 if there is no files

    existing_numbers.sort()

    for i in range(len(existing_numbers)):
        if i != existing_numbers[i]:
            return i  # Return first hole in dataset
    return existing_numbers[-1] + 1  # Carries on the sequence

def rename_data(path):

    # Folder path for txt files
    pathNames = ["Grinding", "Idle"]
    activityName = ["GRIND", "IDLE"]
    folder_path_Grinding = os.path.normpath(path)

    for i in range(len(pathNames)):
        folder_path = os.path.join(folder_path_Grinding, pathNames[i])
        print(f"Processing files in: {folder_path}")

        # Fetch and sort .txt files based on new date format
        files = sorted(
            [f for f in os.listdir(folder_path) if f.endswith(".txt") and not f.startswith(activityName[i])],
            key=convert_date_format  # Sorts based on date YYYY / MM / DD
        )

        for old_name in files:
            new_index = find_next_available_index(folder_path, activityName[i])  # Find available index
            new_name = f"{activityName[i]}_{new_index}.txt"
            
            old_path = os.path.join(folder_path, old_name)
            new_path = os.path.join(folder_path, new_name)

            os.rename(old_path, new_path)
            print(f"Name updated from: {old_name} -> {new_name}")

        print("Namechanges completed!")

def delete_header(path):

    # delete first n lines before "Timestamp"
    # changes "Timestamp [ms][xx]" to "Timestamp"
    file_path = path
    found_timestamp = False

    # Read all lines
    with open(file_path, "r") as f:
        lines_to_keep = []
        
        for line in f:
            if "Timestamp" in line:
                found_timestamp = True  # Start keeping lines from here
                line = re.sub(r'[\t\s]\[ms\]\[.*?\]', '', line) # Delete Tab/space, [ms] and [xx] from Timestamp line
            if found_timestamp:
                lines_to_keep.append(line)

    if not found_timestamp:
        print(f"Warning: 'Timestamp' not found in {file_path}. No changes were made.")
        return 

    # Write only the lines after "Timestamp" back to the file
    with open(file_path, "w") as f:
        f.writelines(lines_to_keep)

def tab_txt_to_csv(txt_file, csv_file):
    # Convert tab seperated txt file to csv file
    # txt_file, csv_file format : "filename.txt", "filename.csv"
    df_txt = pd.read_csv(txt_file, delimiter=r'\s+', engine='python') # Delimiter is now all whitespace (tab and space etc)
    df_txt.to_csv(csv_file, index = None)
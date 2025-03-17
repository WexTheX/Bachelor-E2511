import pandas as pd
import fileinput
import re
import os

def rename_data(path):
    
    # This function sorts the content of each activity folder and renames the files as "GRIND_n", "IDLE_n" etc
    pathNames = ["Grinding", "Idle"]
    activityName = ["GRIND", "IDLE"]
    folder_path_data = os.path.normpath(path)

    for i in range(len(pathNames)):
        # Change folder path dynamicly
        folder_path = os.path.join(folder_path_data, pathNames[i])
        print(f"Changing names of files in folder: {folder_path}")

        # Find and sort txt files in folder
        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])

        # Give new names to the files
        for index, old_name in enumerate(files):
            new_name = f"{activityName[i]}_{index}.txt"
            old_path = os.path.join(folder_path, old_name)  # Correct file path
            new_path = os.path.join(folder_path, new_name)

            # If file exist, skip file
            if os.path.exists(new_path):
                print(f"Skipping: {old_name} (Filename {new_name} already exist)")
                continue

            os.rename(old_path, new_path)
            print(f"Name changed from: {old_name} -> {new_name}")

    print("Done renaming files!")

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
import pandas as pd
import fileinput
import re
import os
from pathlib import Path

# def downsample(Hz)
    # We should have an option to downsample to see how diff sampling frequencies affect ML accuracy
    # This is a hyperparemeter
    
    # return downsampled_fs
def convert_date_format(filename):
    # Convert date format from DD.MM.YYYY to YYYY.MM.DD in the filename
    match = re.match(r"(\d{2})\.(\d{2})\.(\d{4})", filename)  # Finds date in the file name
    if match:
        day, month, year = match.groups()
        return f"{year}.{month}.{day} " + filename[len(match.group(0)):]  # Keeps the rest of the filename
    return filename  # Returns date if nothing is changed

def rename_data(path, path_names, activity_name):
    path = os.path.normpath(path)

    for i in range(len(path_names)):
        folder_path = os.path.join(path, path_names[i])
        print(f"Re-naming files in: {folder_path}")

        
# Convert bin to txt file if bin file found
        for f in os.listdir(folder_path):
            if f.endswith(".bin") and not f.startswith(activity_name[i]):
                convert_bin_to_txt(os.path.join(folder_path, f))
# Fetch and sort .txt files based on new date format
        files = sorted(
            [f for f in os.listdir(folder_path) if f.endswith(".txt") and not f.startswith(activity_name[i])],
            key=convert_date_format  # Sorts based on date YYYY / MM / DD
        )

        for old_name in files:
            new_index = find_next_available_index(folder_path, activity_name[i])  # Find available index
            new_name = f"{activity_name[i]}_{new_index}.txt"
            
            old_path = os.path.join(folder_path, old_name)
            new_path = os.path.join(folder_path, new_name)

            os.rename(old_path, new_path)
            print(f"Name updated from: {old_name} -> {new_name}")

        print("Namechanges completed!")



def fillSets(path, path_names, activity_name, seperate_types):
    
    rename_data(path, path_names, activity_name)

    sets = []
    sets_label = []

    #### make list of folder paths
    path_names = os.listdir(path)
    path = os.path.normpath(path)

    ''' SEPERATE BY TYPE'''
    if seperate_types == 1:
        print("WIP")
      
    else: 
        for i, name in enumerate(path_names):
            folder_path = os.path.join(path,name)

    for i, name in enumerate(path_names):
        folder_path = os.path.join(path, name)
        print(f"Finding files in: {folder_path}")
        
        
        for f in os.listdir(folder_path):
            if f.endswith(".bin") and not f.startswith(activity_name[i]):
                convert_bin_to_txt(os.path.join(folder_path, f))
        txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt") and os.path.isfile(os.path.join(folder_path, f))]
        
        
        for j in range(len(txt_files)):

            sets.append(f"{folder_path}/{activity_name[i]}_" + str(j) )
            sets_label.append(activity_name[i])
    
    print("Done filling sets")
    print("\n")
    return sets, sets_label
        
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

#### Converting bin to txt ####
def convert_bin_to_txt(input_file):
    output_file = os.path.splitext(input_file)[0] + ".txt"
    # Read binary data and filter out unnecessary symbols
    with open(input_file, "rb") as bin_file:
        raw_data = bin_file.read()

    # Remove "zero bytes" and decode UTF-8
    clean_data = raw_data.replace(b"\x00", b"").decode("utf-8", errors="ignore")

    # Remove extra blank lines
    clean_lines = [line.strip() for line in clean_data.splitlines() if line.strip()]

    # Write a new text files with correct linestructure
    with open(output_file, "w", encoding="utf-8") as txt_file:
        txt_file.write("\n".join(clean_lines) + "\n")  # makes sure the ending is correct

    print(f"File convert from .bin to .txt done. file saved as '{output_file}'.")
    os.remove(input_file)

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
                line = re.sub(r'\bTEMP\b', 'Temp', line)
                line = re.sub(r'\bPressure\b', 'Press', line)
                
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
    df_txt = pd.read_csv(txt_file, delimiter=r'\t', engine='python') # Delimiter is now all whitespace (tab and space etc)
    df_txt.to_csv(csv_file, index = None)
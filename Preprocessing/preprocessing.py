import pandas as pd
import re
import os
from typing import List, Dict, Any, Tuple, Sequence
import pickle


# We should have an option to downsample to see how diff sampling frequencies affect ML accuracy
# This is a hyperparemeter ???
def downsample(df: pd.DataFrame, old_fs, new_fs):
    dropped_rows = []

    if((old_fs / new_fs).is_integer() == False):
        print(f"Old fs: {old_fs} / New fs: {new_fs} is not whole number")
        quit()
    elif((old_fs < new_fs)):
        print(f"Old fs: {old_fs} is smaller than New fs: {new_fs}")
        quit()
    else:
        for i in range(len(df['Timestamp'])):
            if((i % (old_fs / new_fs)) != 0):
                dropped_rows.append(i)

    new_df = df.drop(dropped_rows)
    return new_df

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

def tab_txt_to_csv(txt_file, csv_file):
    # Convert tab seperated txt file to csv file
    # txt_file, csv_file format : "filename.txt", "filename.csv"
    df_txt = pd.read_csv(txt_file, delimiter=r'\t', engine='python') # Delimiter is now all whitespace (tab and space etc)
    df_txt.to_csv(csv_file, index = None)

def fillSets(path, path_names, activity_name):
    
    rename_data(path, path_names, activity_name)

    sets = []
    sets_label = []

    #### make list of folder paths
    path_names = os.listdir(path)
    path = os.path.normpath(path)
    
    for i, name in enumerate(path_names):
        folder_path = os.path.join(path,name)
        print(f"Finding files in: {folder_path}")
        
        
        for f in os.listdir(folder_path):
            if f.endswith(".bin") and not f.startswith(activity_name[i]):
                convert_bin_to_txt(os.path.join(folder_path, f))

        txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt") and os.path.isfile(os.path.join(folder_path, f))]
        
        
        for j in range(len(txt_files)):

            sets.append(f"{folder_path}/{activity_name[i]}_" + str(j) )
            sets_label.append(activity_name[i])

    ''' TODO: ONLY USE ONE DATAFILES FOLDER '''
    # for i, name in enumerate(path_names):
    #     folder_path = os.path.join(path,name)
    #     print(f"Finding files in: {folder_path}")
    #     for f in os.listdir(folder_path):
    #         if f.endswith(".bin") and not f.startswith(activity_name[i]):
    #             convert_bin_to_txt(os.path.join(folder_path, f))
    #         if os.path.isdir(f):
    #             for j in os.listdir(f):
    #                 if f.endswith(".bin") and not f.startswith(activity_name[i]):
    #                     convert_bin_to_txt(os.path.join(folder_path, f))
    #     txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt") and os.path.isfile(os.path.join(folder_path, f))]
        
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

    compare_bin_and_txt(input_file, output_file)

    print(f"File convert from .bin to .txt done. file saved as '{output_file}'.")
    bin_file.close()
    txt_file.close()
    os.remove(input_file)

def compare_bin_and_txt(input_file, output_file):
    # Les innholdet fra begge filene som tekst
    with open(input_file, "rb") as f_bin:
        bin_lines = f_bin.read().replace(b"\x00", b"").decode("utf-8", errors="ignore").splitlines()
    with open(output_file, "r", encoding="utf-8") as f_txt:
        txt_lines = f_txt.read().splitlines()

    # Sammenlign linje for linje
    max_lines = max(len(bin_lines), len(txt_lines))  # Håndterer ulik lengde
    differences = []

    for i in range(8, max_lines):
        bin_line = bin_lines[i] if i < len(bin_lines) else "<Mangler i .bin>"
        txt_line = txt_lines[i] if i < len(txt_lines) else "<Mangler i .txt>"

        if bin_line != txt_line:
            differences.append(f"Forskjell på linje {i+1}:\n  BIN: '{bin_line}'\n  TXT: '{txt_line}'\n")

    # Skriv ut resultatet
    if differences:
        print(f"Fil {input_file}:")
        print(f"{len(differences)} forskjeller funnet mellom filene:\n")
        for diff in differences[:10]:  # Vis maks 10 forskjeller for oversikt
            print(diff)
        
        with open("differences_log.txt", "w", encoding="utf-8") as log_file:
            log_file.writelines(differences)
        print("Alle forskjeller er lagret i 'differences_log.txt'.")

        quit()

    f_txt.close()
    f_bin.close()

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

def pickleFiles(n_results:      list[dict[str, Any]], 
                result:        dict[str, Any],
                output_path:    str, 
                PCA_object:     Any,
                scaler:         Any
                ) -> None:

    for r in n_results:
        r_name          = r['model_name']
        r_optimizer     = r['optimalizer']
        r_clf           = r['classifier']

        with open(output_path + str(r_name) + "_" +  str(r_optimizer) + "_" + "clf.pkl", "wb") as clf_file:
            pickle.dump(r_clf, clf_file)

        clf_file.close()

    # Pickle best clf
    pickle_clf = result['classifier']

    with open(output_path + "classifier.pkl", "wb") as best_clf_file: 
        pickle.dump(pickle_clf, best_clf_file)

    best_clf_file.close()

    # print("Modell som lagres:", pickle_clf)
    # print("predict_proba tilgjengelig:", hasattr(pickle_clf, "predict_proba")) 

    # Pickle PCA and scaler
    with open(output_path + "PCA.pkl", "wb" ) as PCA_File:
        pickle.dump(PCA_object, PCA_File)

    PCA_File.close()

    with open(output_path + "scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    scaler_file.close()
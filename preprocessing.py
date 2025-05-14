import pandas as pd
import scipy as sp
import math as math
import re
import os
from typing import List, Dict, Any, Tuple, Sequence, Union
import pickle
import joblib

def downsample(df:          pd.DataFrame,
               fs:          int,
               ds_fs:       int,
               variables:   list[str]
               ) -> pd.DataFrame:
   
    # We should have an option to downsample to see how diff sampling frequencies affect ML accuracy
    # This is a hyperparemeter ???

    try:
        new_df = pd.DataFrame(columns=variables)
        for column in df:
            new_df[column] = sp.signal.decimate(df[column], math.floor(fs/ds_fs), ftype="fir")

    except Exception as ds_error:
        print(f"Error in downsampling: {ds_error}")
        quit()

    return new_df

def convert_date_format(filename: str
                        ) -> str:

    # Convert date format from DD.MM.YYYY to YYYY.MM.DD in the filename
    match = re.match(r"(\d{2})\.(\d{2})\.(\d{4})", filename)  # Finds date in the file name

    if match:
        day, month, year = match.groups()
        return f"{year}.{month}.{day} " + filename[len(match.group(0)):]  # Keeps the rest of the filename
    
    return filename # Returns date if nothing is changed

def rename_data(path:           str,
                path_names:     Sequence,
                activity_name:  str
                ) -> None:
    
    path = os.path.normpath(path)

    for i in range(len(path_names)):
        folder_path = os.path.join(path, path_names[i])
        # print(f"Re-naming files in: {folder_path}")

        
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

    return None

def tab_txt_to_csv(txt_file:    str,
                   csv_file:    str
                   ) -> None:
    
    # Convert tab seperated txt file to csv file
    # txt_file, csv_file format : "filename.txt", "filename.csv"
    df_txt = pd.read_csv(txt_file, delimiter=r'\t', engine='python') # Delimiter is now all whitespace (tab and space etc)
    df_txt.to_csv(csv_file, index = None)

    return None

def fillSets(path:  str
             ) -> Tuple[List, List, List[str]]:
    
    path_names          = os.listdir(path)
    activity_name       = [name.upper() for name in path_names]

    rename_data(path, path_names, activity_name)

    sets:       List[str] = []
    sets_label: List[str] = []

    #### make list of folder paths
    path = os.path.normpath(path)
    
    for i, name in enumerate(path_names):
        folder_path = os.path.join(path,name)
        # print(f"Finding files in: {folder_path}")
        
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
        
    print(f"Done filling sets, {len(sets)} files found.")
    print("\n")

    return sets, sets_label, activity_name
        
def find_next_available_index(folder_path:  str,
                              prefix:       str
                              ) -> int:
    
    # Finds next available index for files with a given prefix
    existing_numbers: List[int] = []

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

def convert_bin_to_txt(input_file: str
                       ) -> None:
    
    #### Converting bin to txt ####

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

    return None

def compare_bin_and_txt(input_file:         str,
                        output_file:        str,
                        differences_log:    str = "differences_log.txt"
                        ) -> None:
    
    # Read contents of both files as text
    with open(input_file, "rb") as f_bin:
        bin_lines = f_bin.read().replace(b"\x00", b"").decode("utf-8", errors="ignore").splitlines()
    with open(output_file, "r", encoding="utf-8") as f_txt:
        txt_lines = f_txt.read().splitlines()

    # Compare line by line
    max_lines = max(len(bin_lines), len(txt_lines))  # Handles different length
    differences = []

    for i in range(8, max_lines):
        bin_line = bin_lines[i] if i < len(bin_lines) else "<Missing in .bin>"
        txt_line = txt_lines[i] if i < len(txt_lines) else "<Missing in .txt>"

        if bin_line != txt_line:
            differences.append(f"Difference at line {i+1}:\n  BIN: '{bin_line}'\n  TXT: '{txt_line}'\n")

    # Skriv ut resultatet
    if differences:
        print(f"File {input_file}:")
        print(f"{len(differences)} differences found between files:\n")
        for diff in differences[:10]:  # Vis maks 10 forskjeller for oversikt
            print(diff)
        
        with open(differences_log, "w", encoding="utf-8") as log_file:
            log_file.writelines(differences)
        print(f"All differences saved in {differences_log}.")

        quit()

    f_txt.close()
    f_bin.close()

    return None


def delete_header(file_path: str
                  ) -> None:

    # delete first n lines before "Timestamp"
    # changes "Timestamp [ms][xx]" to "Timestamp"
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
                
            if found_timestamp == True:
                lines_to_keep.append(line)

    if not found_timestamp:
        print(f"Warning: 'Timestamp' not found in {file_path}. No changes were made.")
        return 

    # Write only the lines after "Timestamp" back to the file
    with open(file_path, "w") as f:
        f.writelines(lines_to_keep)

    return None

def pickleFiles(n_results:      list[dict[str, Any]], 
                result:         dict[str, Any],
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

    return None

def saveJoblibFiles(n_results:  list[dict[str, Any]], 
                result:         dict[str, Any],
                output_path:    str, 
                PCA_object:     Any,
                scaler:         Any
                ) -> None:
    
    ## Saves all classifiers from n_results
    try:
        print(f"Saving PCA object, scaler and clf to {output_path}")

        for r in n_results:
            model_name      = r['model_name']
            optimizer       = r['optimalizer']
            clf             = r['classifier']    

            joblib.dump(clf, f"{output_path}{model_name}_{optimizer}_clf.joblib")

        ## Save best classifier
        joblib.dump(result['classifier'], f"{output_path}classifier.joblib")

        joblib.dump(PCA_object, f"{output_path}PCA.joblib")
        joblib.dump(scaler, f"{output_path}scaler.joblib")
        
    except Exception as e:
        print(f"Warning: Failed to save PCA, scaler and clf: {e}")

    return None
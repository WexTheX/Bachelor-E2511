import pandas as pd
import fileinput

def delete_header(path):
    file_path = path
    found_timestamp = False

    with open(file_path, "r") as f:
        lines_to_keep = []
        
        for line in f:
            if "Timestamp" in line:
                found_timestamp = True  # Start keeping lines from here
            if found_timestamp:
                lines_to_keep.append(line)

    # Write only the lines after "Timestamp" back to the file
    with open(file_path, "w") as f:
        f.writelines(lines_to_keep)


# delete first n lines
# change Timestamp [ms][ew] til Timestamp

# Convert tab seperated txt file to csv file
# txt_file, csv_file format : "filename.txt", "filename.csv"
# Remember to remove "Information" in file when it comes directly from Muse
def tab_txt_to_csv(txt_file, csv_file):
  df_txt = pd.read_csv(txt_file, delimiter='\t')
  df_txt.to_csv(csv_file, index = None)
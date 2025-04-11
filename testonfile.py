

#### Implement solution to test a large file of various activities 
# 1) Check if bin, if bin convert to txt
# 2) Convert date format
# 3) Delete header
# 4) Convert to csv
# 5) Downsample if needed
# 6) Make Windows
# 7) Feature extraction
# 8) Predict label based on features and trained model
# 9) Save results to CSV and prints

'''
To run this file, enter the correct filepath for the files to be tested in test_file_path
and where you want the csv file saved in prediction_csv_path. Set fs to the sample frequency
for the data collection, and ds_fs to the wanted downsample frequency. Keep in mind ds_fs has
to be set the same as the model is trained for.

Set the window length to the same window length used for training the model.
'''


import joblib
import pandas as pd
import numpy as np
import os
import pandas as pd

### Local imports
from extractFeatures import extractDFfromFile, extractFeaturesFromDF
from Preprocessing.preprocessing import convert_bin_to_txt, downsample


### Folder paths
test_file_path = "testOnFile/testFiles"
prediction_csv_path = "testOnFile"
test_files = os.listdir(test_file_path)


### Variables
fs = 800                   ### Sampling rate (800 on logged data and max 200 on live data)
ds_fs = 200                ### Downsample to
window_length_seconds = 20 ### Setting the window lenth in seconds
want_prints = 1            ### Set to true if you want prints in the terminal aswell as an output file


### Lists
df_result_all = [] ##Storing results


def run_inference_on_file(file_path:            str,
                          fs:                   int,
                          ds_fs:                int,
                          window_length_sec:    int,
                          want_prints:          bool,
                          file_to_test:         str,
                          norm_accel=False):
    
    print("_______________________________________________________________________________")
    print(f"Testing file {file_to_test}")

    ### Load trained model
    clf = joblib.load("OutputFiles/Separated/classifier.pkl")
    pca = joblib.load("OutputFiles/Separated/PCA.pkl")
    scaler = joblib.load("OutputFiles/Separated/scaler.pkl")
    #print(clf.classes_)
    #print("Probability is set to?:", hasattr(clf, "predict_proba")) ##Printing 

    ### Load file and preprocess
    df = extractDFfromFile(file_path, fs)
    ### Downsample
    df = downsample(df, fs, ds_fs)

    ### Feature extraction
    features_list, _ = extractFeaturesFromDF(df, "unknown", window_length_sec, ds_fs, norm_accel)

    ### Convert to Dataframe and do PCA
    features_df     = pd.DataFrame(features_list)
    features_scaled = scaler.transform(features_df)
    features_pca    = pca.transform(features_scaled)

    ### Predictions
    preds = clf.predict(features_pca)

    ###________________________________________________________________________
    ### Output csv and prints

    results = []   ## making results collection list

    if want_prints == True:
        print(f"{'TIME':<10}{'ACTIVITY':<15}{'PROBABILITY':<12}TOP-3 PREDICTIONS")
        print(f"{'0–10':<10}{'deleted':<15}{'-':<12}-")

    ###The first 10 sec of the set gets deleted in extractFeaturesFromDF(). 
    results.append(["0–10", "deleted", "-", "-"])


    if hasattr(clf, "predict_proba"):
        probabilities = clf.predict_proba(features_pca)

        class_labels = clf.classes_

        for i, probs in enumerate(probabilities):
            start = (10 + (i * window_length_sec))
            end = start + window_length_sec
            time_range = f"{start}–{end}"

            # Get top 3 predictions
            top_3 = sorted(zip(class_labels, probs), key=lambda x: x[1], reverse=True)[:3]
            top_3_str = ", ".join([f"{lbl} ({p:.2f})" for lbl, p in top_3])

            # Use most probable as activity
            pred, pred_prob = top_3[0]
            if want_prints == True:
                print(f"{time_range:<10}{pred:<15}{pred_prob:<12.2f}{top_3_str}")
            results.append([time_range, pred, f"{pred_prob:.2f}", top_3_str])
    else:
        for i, pred in enumerate(preds):
            (10 + (i * window_length_sec))
            end = start + window_length_sec
            time_range = f"{start}–{end}"

            if want_prints == True:
                print(f"{time_range:<10}{pred:<15}{'-':<12}-")

            results.append([time_range, pred, "-", "-"])

        if want_prints == True:
            print("_______________________________________________________________________________")
        
    df_result = pd.DataFrame(results, columns=["Time", "Activity", "Probability", "Top-3"])

    return df_result
    

def offline_test():
    for filename in test_files:

        file_to_test = os.path.join(test_file_path, filename)
        file_to_test_no_ext = file_to_test.replace(".txt", "")

        if filename.endswith(".csv"):
            continue  # Skipping .csv files

        elif filename.endswith(".bin"): ##Converting .bin to .txt
            convert_bin_to_txt(file_to_test_no_ext)

        print(f"Testing file: {file_to_test}")

        df_result = run_inference_on_file(file_to_test_no_ext, fs=fs, ds_fs=ds_fs, window_length_sec=window_length_seconds, want_prints=want_prints, norm_accel=False)
        print(df_result)

        ### Line to seperate the different prediction sets       
        header_lines = [
            f"_______________________________________________________________________________",
            f"Predictions from: {os.path.basename(file_to_test)}"
        ]
        header_df = pd.DataFrame([[line, "", "", ""] for line in header_lines],
                                columns=["Time", "Activity", "Probability", "Top-3"])

        ###Adding header above every prediction set
        column_header = pd.DataFrame([["Time", "Activity", "Probability", "Top-3"]],
                                columns=["Time", "Activity", "Probability", "Top-3"])

        ### Adding the data together
        df_result_all.append(header_df)
        df_result_all.append(column_header)
        df_result_all.append(df_result)

        ### Saving as csv

    combined_df = pd.concat(df_result_all, ignore_index=True)
    filename_out = os.path.join(prediction_csv_path, "predictions.csv")
    combined_df.to_csv(filename_out, index=False)

        ### Finished, printing file  for output file
    print("Done running predictions on datasets")
    print(f"Predictions saved in: {filename_out}")

def label_filter():
    return 0

def calc_workload():
    return 0
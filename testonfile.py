import joblib
from extractFeatures import extractDFfromFile, extractFeaturesFromDF
import pandas as pd
import numpy as np
import os

# Implement solution to test a large file of various activities - EH 

# Import file 
# transform into csv
# split to windows
# do feature extraction
# send through ML
# get answer of what activity and probability from ML


# 1) check if bin, if bin convert to txt OK
# 2) Convert date format
# 3) delete header
# 4) convert to csv
# 5) make Windows
# 6) feature extraction



#     ### Testing trained model on unseen files
# test_file_path = "testFiles"
# test_files = os.listdir(test_file_path)


# for filename in test_files:
#     file_to_test = os.path.join(test_file_path, filename)
#     print(f"Testing file {file_to_test}")
#     run_inference_on_file(file_to_test, fs = fs, window_length_sec = window_length_seconds, norm_accel=True, run=True)

#------------------------------------------------------
### Testing trained model on unseen files
# test_file_path = "testFiles"
# test_files = os.listdir(test_file_path)

# for filename in test_files:
#   file_to_test = os.path.join(test_file_path, filename)
  
# ### Converts bin to txt and checks if the .bin and .txt file has errors
#     if filename.endswith(".bin") and not filename.startswith(activity_name[i]):
#         convert_bin_to_txt(file_to_test)
#     convert_date_format(filename)
#     print(f"Testing file {file_to_test}")
#     run_inference_on_file(file_to_test, fs = fs, window_length_sec = window_length_seconds, norm_accel=True, run=True)
  
### Folder path
test_file_path = "testFiles"
test_files = os.listdir(test_file_path)

### Variables


fs = 800
window_length_seconds = 20



def run_inference_on_file(file_path, fs, window_length_sec, run, norm_accel=True):


    # from extractFeatures import extractDFfromFile
    # from extractFeatures import extractFeaturesFromDF
    if not run:
        print(f"Skipping inference on {file_path}")
        return
    ### Load trained model
    clf = joblib.load("OutputFiles/Separated/classifier.pkl")
    pca = joblib.load("OutputFiles/Separated/PCA.pkl")
    scaler = joblib.load("OutputFiles/Separated/scaler.pkl")
    #print("Probability is set to?:", hasattr(clf, "predict_proba"))


    
    ### Load file and preprocess
    

    df = extractDFfromFile(file_path, fs)

    ### Feature extraction
    features_list, _ = extractFeaturesFromDF(df, "unknown", window_length_sec, fs, norm_accel)

    ### Convert to Dataframe and do PCA
    features_df = pd.DataFrame(features_list)
    features_scaled = scaler.transform(features_df)
    features_pca = pca.transform(features_scaled)

    ### Predictions
    preds = clf.predict(features_pca)
    print(f"{'TIME':<10}{'ACTIVITY':<15}{'PROBABILITY':<12}TOP-3 PREDICTIONS")
    print(f"{'0–10':<10}{'deleted':<15}{'-':<12}-")

    if hasattr(clf, "predict_proba"):
        probabilities = clf.predict_proba(features_pca)
        class_labels = clf.classes_

        for i, probs in enumerate(probabilities):
            start = (i + 1) * window_length_sec
            end = start + window_length_sec
            time_range = f"{start}–{end}"

            # Få top-3 klasser og sannsynligheter
            top_3 = sorted(zip(class_labels, probs), key=lambda x: x[1], reverse=True)[:3]
            top_3_str = ", ".join([f"{lbl} ({p:.2f})" for lbl, p in top_3])

            # Bruk top-1 som activity
            pred, pred_prob = top_3[0]
            print(f"{time_range:<10}{pred:<15}{pred_prob:<12.2f}{top_3_str}")
    else:
        for i, pred in enumerate(preds):
            start = (i + 1) * window_length_sec
            end = start + window_length_sec
            time_range = f"{start}–{end}"
            print(f"{time_range:<10}{pred:<15}{'-':<12}-")
    print("_______________________________________________________________________________")


for filename in test_files:
    if filename.endswith(".csv"):
        continue  # hopper over .csv-filer
    file_to_test = os.path.join(test_file_path, filename)
    file_to_test_no_ext = file_to_test.replace(".txt", "")
    print("_______________________________________________________________________________")
    print(f"Testing file {file_to_test}")
    run_inference_on_file(file_to_test_no_ext, fs = fs, window_length_sec = window_length_seconds, norm_accel=False, run=True)




# Implement solution to test a large file of various activities - EH 

# Import file 
# transform into csv
# split to windows
# do feature extraction
# send through ML
# get answer of what activity and probability from ML
import os
import pandas as pd
import numpy as np


### Importing functions
from Preprocessing.preprocessing import (
convert_date_format, 
tab_txt_to_csv,
convert_bin_to_txt,
downsample,
delete_header
)


### Skal i Main
# test_file_path = "testFiles"
# test_files = os.listdir(test_file_path)

for filename in test_files:
    file_to_test = os.path.join(test_file_path, filename)
    run_inference_on_file(file_to_test, fs, window_length_sec, clf, scaler, pca, norm_accel=True)

#---------------------------------------------



def run_inference_on_file(file_path, run, fs, window_length_sec, clf, scaler, pca, norm_accel=True, run):
    
    ### Load file and preprocess
    test_df = extractDFfromFile(file_path, fs)

    ### Feature extraction
    features_list, _ = extractFeaturesFromDF(test_df, "unknown", window_length_sec, fs, norm_accel)

    ### Convert to Dataframe and do PCA
    features_df = pd.DataFrame(features_list)
    features_scaled = scaler.transform(features_df)
    features_pca = pca.transform(features_scaled)

    ### Predictions
    preds = clf.predict(features_pca)


if hasattr(clf, "predict_proba"):
        probabilities = clf.predict_proba(features_pca)

        for i in range(len(predictions)):
            print(f"Window {i+1}: {predictions[i]} (probs: {probabilities[i]})")
    else:
        for i in range(len(predictions)):
            print(f"Window {i+1}: {predictions[i]}")

print(f"{'TIME':<10}{'ACTIVITY':<15}{'PROBABILITY'}")
print(f"{'0–10':<10}{'deleted':<15}{'-'}")

    for i, (pred, prob) in enumerate(zip(preds, max_probs)):
        start = (i + 1) * window_length_sec
        end = start + window_length_sec
        time_range = f"{start}–{end}"
        prob_str = f"{prob:.2f}" if isinstance(prob, float) else "-"
        print(f"{time_range:<10}{pred:<15}{prob_str}")



### Save trained model
# import joblib
# joblib.dump(clf, "clf_model.pkl")
# joblib.dump(pca, "pca_transform.pkl")
# joblib.dump(scaler, "scaler.pkl")

### Load trained model
# clf = joblib.load("clf_model.pkl")
# pca = joblib.load("pca_transform.pkl")
# scaler = joblib.load("scaler.pkl")




# Implement solution to test a large file of various activities - EH 

# Import file 
# transform into csv
# split to windows
# do feature extraction
# send through ML
# get answer of what activity and probability from ML



# ### Testing trained model on unseen files
# test_file_path = "testFiles"
# test_files = os.listdir(test_file_path)


# for filename in test_files:
#     file_to_test = os.path.join(test_file_path, filename)
#     print(f"Testing file {file_to_test}")
#     run_inference_on_file(file_to_test, fs, window_length_sec,file_name = file_to_test, norm_accel=True, run=True)

def run_inference_on_file(file_path, fs, window_length_sec, run, norm_accel=True):
    if not run:
        print(f"Skipping inference on {file_path}")
        return
    ### Load trained model
    clf = joblib.load("OutputFiles/Separated/classifier.pkl")
    pca = joblib.load("OutputFiles/Separated/PCA.pkl")
    scaler = joblib.load("OutputFiles/Separated/scaler.pkl")

    
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


        for i, (pred, probs) in enumerate(zip(preds, probabilities)):
                    start = (i + 1) * window_length_sec
                    end = start + window_length_sec
                    time_range = f"{start}–{end}"
                    top_3 = sorted(zip(class_labels, probs), key=lambda x: x[1], reverse=True)[:3]
                    top_3_str = ", ".join([f"{lbl} ({p:.2f})" for lbl, p in top_3])
                    pred_prob = dict(top_3).get(pred, "-")
                    print(f"{time_range:<10}{pred:<15}{pred_prob:<12.2f}{top_3_str}")
    else:
        for i, pred in enumerate(preds):
            start = (i + 1) * window_length_sec
            end = start + window_length_sec
            time_range = f"{start}–{end}"
            print(f"{time_range:<10}{pred:<15}{'-':<12}-")





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
import random
from collections import Counter
# from hmmlearn import hmm

### Local imports
from extractFeatures import extractDFfromFile, extractFeaturesFromDF
from Preprocessing.preprocessing import convert_bin_to_txt, downsample

def runInferenceOnFile(file_path:           str,
                        fs:                 int,
                        ds_fs:              int,
                        window_length_sec:  int,
                        want_prints:        bool,
                        file_to_test:       str,
                        norm_accel:         bool
                        ) -> pd.DataFrame:
    
    '''
    Processes a time-series data file, applies a pre-trained ML pipeline,
    and returns windowed predictions.

    This function performs the following steps:
    1. Loads pre-trained classifier, PCA transformer, and scaler models.
    2. Loads time-series data from the specified file path.
    3. Downsamples the data to the target frequency.
    4. Extracts features for overlapping or sequential time windows.
       (Note: Assumes the first `window_length_sec` data might be handled
        differently or ignored by `extractFeaturesFromDF`, offsetting results).
    5. Scales the extracted features.
    6. Applies PCA transformation to the scaled features.
    7. Predicts the activity for each window using the classifier.
    8. Calculates prediction probabilities and top-3 predictions if the
       classifier supports `predict_proba`.
    9. Formats the results, including time ranges, into a pandas DataFrame.
    10. Optionally prints formatted results to the console during processing.
    '''

    results = []

    print("___________________________________________________________________________________")
    print(f"Testing file {file_to_test}")
    
    ### Load trained model
    clf     = joblib.load("OutputFiles/Separated/classifier.pkl")
    pca     = joblib.load("OutputFiles/Separated/PCA.pkl")
    scaler  = joblib.load("OutputFiles/Separated/scaler.pkl")

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

    # Can be used to help calculate exposure_intensity_matrix 
    xyz_accel = abs(np.sqrt(np.power(features_df['mean_accel_X'], 2) +
                            np.power(features_df['mean_accel_Y'], 2) +
                            np.power(features_df['mean_accel_Z'], 2) ))
    # g_constant = xyz_accel.mean()
    # features_df['g_constant_lpf'] = xyz_accel.ewm(alpha=1).mean()
    # g_constant = 9.81
    # gravless_norm = xyz_accel * g_constant / 1000
    # print(f"g constant: {g_constant}")
    # gravless_norm = np.subtract(xyz_accel, g_constant) 
    # print(gravless_norm)

    ### Predictions
    preds = clf.predict(features_pca)

    ###________________________________________________________________________
    ### Output csv and prints

    # if want_prints == True:
    #     print(f"{'TIME':<10}{'ACTIVITY':<15}{'PROBABILITY':<12}TOP-3 PREDICTIONS")
    #     print(f"{'0–10':<10}{'deleted':<15}{'-':<12}-")

    # The first 10 sec of the set gets deleted in extractFeaturesFromDF(). 
    # results.append(["0–10", "deleted", "-", "-"])


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

            # if want_prints == True:
            #     print(f"{time_range:<10}{pred:<15}{pred_prob:<12.2f}{top_3_str}")

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

    df_result['Smoothed activity'] = moving_mode_filter(df_result['Activity'], window=3)

    if want_prints:
        print(df_result)

    return df_result, features_df
    

def offlineTest(test_file_path:        str,
                prediction_csv_path:   str,
                fs:                    int, 
                ds_fs:                 int,
                window_length_seconds: int,
                want_prints:           bool
                ) -> pd.DataFrame:
    
    '''
    Runs inference on all compatible files within a specified directory and
    combines the results into a single CSV file.

    This function iterates through files in `test_file_path`.
    - It skips any files ending with '.csv'.
    - If a file ends with '.bin', it attempts to convert it to '.txt' using
      `convert_bin_to_txt` (assuming the converted file has the same base name
      but with a '.txt' extension).
    - For each '.txt' file (either original or converted from '.bin'), it calls
      `run_inference_on_file` to get predictions.
    - Handles potential errors during the processing of individual files,
      allowing the process to continue with other files.
    - Concatenates the prediction DataFrames from all successfully processed
      files.
    - Saves the combined results to a specified CSV file in the output directory.
    '''
    
    # feature_df_all = pd.DataFrame()

    print(f"Making predictions on data from {test_file_path}: ")

    # Paths
    test_files          = os.listdir(test_file_path)
    df_result_all       = [] #Storing results

    for filename in test_files:

        file_to_test = os.path.join(test_file_path, filename)
        file_to_test_no_ext = file_to_test.replace(".txt", "")
        
        if filename.endswith(".csv"):
            continue  # Skipping .csv files

        elif filename.endswith(".bin"): ##Converting .bin to .txt
            convert_bin_to_txt(file_to_test_no_ext)

        df_result, features_df = runInferenceOnFile(file_to_test_no_ext, fs, ds_fs, window_length_seconds, want_prints, file_to_test, norm_accel=False)
    
        # header_lines = [
        #     f"_______________________________________________________________________________",
        #     f"Predictions from: {os.path.basename(file_to_test)}"
        # ]

        # header_df = pd.DataFrame([[line, "", "", ""] for line in header_lines],
        #                         columns=["Time", "Activity", "Probability", "Top-3"])

        ###Adding header above every prediction set
        # column_header = pd.DataFrame([["Time", "Activity", "Probability", "Top-3"]],
        #                         columns=["Time", "Activity", "Probability", "Top-3"])

        ### Adding the data together
        # df_result_all.append(header_df)
        # df_result_all.append(column_header)

        df_result_all.append(df_result)

        # TODO
        # df.append() er på vei ut, må finne alternativ

        ### Saving as csv

    combined_df = pd.concat(df_result_all, ignore_index=True)
    filename_out = os.path.join(prediction_csv_path, "predictions.csv")
    combined_df.to_csv(filename_out, index=False)

    ### Finished, printing file  for output file
    print("Done running predictions on datasets")
    print(f"Predictions saved in: {filename_out}")
    print("-" * 50)

    return combined_df #,features_df_all

def calcExposure(combined_df:           pd.DataFrame,
                 window_length_seconds: int,
                 labels:                list[str], 
                 exposures:             list[str],
                 safe_limit_vector:     list[float],
                 csv_path:              str,
                 filter_on:             bool
                ) -> pd.DataFrame:
    
    '''
    Calculates exposure workload based on activity durations and intensity.

    Determines the duration of each predicted activity within a given DataFrame,
    calculates the total exposure score for various hazard types using a
    predefined intensity matrix, and generates a summary comparing these
    scores to safe limits. Prints intermediate results and the final summary.
    '''

    # labels = [
    #             'GRINDBIG', 'GRINDSMALL',
    #             'IDLE','IMPA','GRINDMED', 
    #             'SANDSIM',
    #             'WELDALTIG', 'WELDSTMAG', 'WELDSTTIG'
    #     ]
    
    # exposure_list = ['CARCINOGEN', 'RESPIRATORY', 'NEUROTOXIN', 'RADIATION', 'NOISE', 'VIBRATION', 'THERMAL', 'MSK']

    default_value = 0.0

    if filter_on:
        print(f"Calculating exposure... (filter on)")
        predicted_activities    = combined_df['Smoothed activity']
    else:
        print(f"Calculating exposure... (filter off)")
        predicted_activities    = combined_df['Activity']

    activity_counts             = predicted_activities.value_counts()
    activity_length             = activity_counts * window_length_seconds / 3600
    activity_length_complete    = activity_length.reindex(labels, fill_value=default_value)

    # x
    activity_duration_vector    = activity_length_complete.values

    # A
    exposure_intensity_matrix   = initialize_exposure_intensity_matrix(exposures, labels)

    # b = Ax
    total_exposure_vector       = exposure_intensity_matrix @ activity_duration_vector

    summary_df = exposure_summary(total_exposure_vector, safe_limit_vector, exposures)

    filename_out = os.path.join(csv_path, "summary.csv")
    summary_df.to_csv(filename_out, index=False)

    print()
    print(f"Predicted hours: \n {activity_length_complete.round(decimals=2)}")
    print(f"Risk factors increased. Grind big!")
    print("-" * 99)
    print(f"Exposure intensity matrix: \n {exposure_intensity_matrix}")
    print("-" * 99)
    print(summary_df.round(decimals=1))
    print("-" * 57)

    return summary_df

def initialize_exposure_intensity_matrix(exposures:                     list[str], 
                                         activities:                    list[str],
                                         gravityless_norm_accel_mean    = round(random.uniform(10.0, 20.0), 1) - 9.81,
                                         gravityless_norm_accel_energy  = round(random.uniform(10.0, 20.0), 1) - 9.81,
                                         temperature_energy             = round(random.uniform(10.0, 20.0), 1)
                                         ) -> pd.DataFrame:
    
    '''
    Initializes and populates the exposure intensity matrix.

    Creates a DataFrame where rows represent exposure types and columns represent
    activities. Each cell (i, j) indicates the intensity rate (e.g., score
    units per hour) at which activity j contributes to exposure type i.

    The matrix is populated using a combination of:
    - Fixed estimates for certain activity-exposure pairs (e.g., RADIATION).
    - Placeholder random values for development/testing.
    - Calculations based on sensor-derived features passed as arguments,
      which act as proxies for the actual exposure intensity (e.g., using
      acceleration energy for MSK load).
    '''
    
    # TODO 
    # Finne fleire "proxies" for hvilken sensordata/features som slår ut på hvilken faktor
    # Eller finne tall på det 

    df = pd.DataFrame(0.0, index=exposures, columns=activities) 

    # If there is something common to all welding or grinding, use *WELD and *GRIND
    WELD    = df.columns[df.columns.str.startswith('WELD')]
    GRIND   = df.columns[df.columns.str.startswith('GRIND')]
    
    df.loc['CARCINOGEN',    ['WELDSTMAG', 'WELDSTTIG']] = round(random.uniform(10.0, 20.0), 1)
    df.loc['RESPIRATORY',   ['SANDSIM', 'WELDALTIG']]   = round(random.uniform(10.0, 20.0), 1) 
    df.loc['NEUROTOXIN',    ['WELDSTTIG']]              = round(random.uniform(10.0, 20.0), 1)
    df.loc['RADIATION',     [*WELD]]                    = 99999.0
    df.loc['NOISE',         [*GRIND]]                   = round(random.uniform(10.0, 20.0), 1)
    df.loc['VIBRATION',     [*GRIND]]                   = 2 * gravityless_norm_accel_mean**2 # from https://www.ergonomiportalen.no/kalkulator/#/vibrasjoner 
    df.loc['THERMAL',       [*WELD]]                    = temperature_energy
    df.loc['MSK',           [*GRIND, 'IMPA']]           = gravityless_norm_accel_energy

    return df

def exposure_summary(total_exposure_vector:     np.array,
                     safe_limit_vector:         np.array,
                     exposures:                 list[str],
                     neutral_limit = 0.80
                     ) -> pd.DataFrame:

    '''
    Creates a summary DataFrame comparing exposure levels to safe limits.

    Generates a report showing the calculated exposure level, the predefined
    safe limit, the ratio of exposure to the limit (as a percentage), and a
    status indicator (smiley face) for each exposure type.
    '''

    data = {
        'Exposure level':   total_exposure_vector,
        'Safe limit':       safe_limit_vector
    }

    df = pd.DataFrame(data, index=exposures)

    # df['Delta'] = df['Safe limit'] - df['Exposure level']

    df['Ratio [%]'] = 100 * (df['Exposure level'] / df['Safe limit'])
    
    cases = [
        # 1: Overexposed
        df['Exposure level'] > df['Safe limit'],
        
        # 2: Neutral limit reached
        df['Exposure level'] / df['Safe limit'] >= neutral_limit,

        # 3: Default: well under safe limit
    ]

    smileys = [f"😟🔴  ", f"😐🟡  "]

    df['Status'] = np.select(cases, smileys, default=f"😊🟢  ")

    return df

def moving_mode_filter(series:  pd.Series, 
                       window:  int
                       ) -> pd.Series:
    
    return pd.Series(
        [
            Counter(series[max(0, i - window // 2): i + window // 2 + 1]).most_common(1)[0][0]
            if not series[max(0, i - window // 2): i + window // 2 + 1].empty else None
            for i in range(len(series))
        ],
        index=series.index
    )

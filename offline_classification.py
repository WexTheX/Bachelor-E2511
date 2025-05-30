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

NB! If pulling test files from FilesFromAker_ALL, DON'T have any Aker files in the training set. 

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
from typing import List, Dict, Any, Tuple, Sequence, Optional

### Local imports
from extractFeatures import extractDFfromFile, extractFeaturesFromDF
from preprocessing import convert_bin_to_txt, downsample

def runInferenceOnFile(file_path:           str,
                        fs:                 int,
                        ds_fs:              int,
                        window_length_sec:  int,
                        want_prints:        bool,
                        variables:          list[str],
                        norm_IMU:               bool,
                        separated_or_combined:  str = 'Separated', # or Combined

                        start_offset:       int = 0
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
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
    clf_path:           str = f"OutputFiles/{separated_or_combined}/classifier.joblib"
    pca_path:           str = f"OutputFiles/{separated_or_combined}/PCA.joblib"
    scaler_path:        str = f"OutputFiles/{separated_or_combined}/scaler.joblib"

    results = []

    print("___________________________________________________________________________________")
    print(f"Testing file {file_path}")
    
    ### Load trained model
    clf     = joblib.load(clf_path)
    pca     = joblib.load(pca_path)
    scaler  = joblib.load(scaler_path)

    #print(clf.classes_)
    #print("Probability is set to?:", hasattr(clf, "predict_proba")) ##Printing 

    ### Load file and preprocess
    
    df = extractDFfromFile(file_path, ds_fs, drop_index=False)
    
    ### Downsample --- Does not make sense to downsample testing dataset as we want it "unedited"
    # if ds_fs < fs:
    #     df = downsample(df, fs, ds_fs, variables)
    
    ### Feature extraction
    features_list, _ = extractFeaturesFromDF(df, "unknown", window_length_sec, ds_fs, norm_IMU)

    ### Convert to Dataframe and do PCA
    try:
        features_df     = pd.DataFrame(features_list)
        features_scaled = scaler.transform(features_df)
        features_pca    = pca.transform(features_scaled)

    except Exception as e:
        print(f"Error in runInferenceOnFile, scaler.transform or pca.transform. Might indicate the need to make new CLF, and pickle clf, scaler and pca: {e}")

    # Can be used to help calculate exposure_intensity_matrix 
    # xyz_accel = abs(np.sqrt(np.power(features_df['mean_accel_X'], 2) +
    #                         np.power(features_df['mean_accel_Y'], 2) +
    #                         np.power(features_df['mean_accel_Z'], 2) ))
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

            start_idx   = (start_offset + (i * window_length_sec))
            end_idx     = start_idx + window_length_sec
            time_range  = f"{start_idx}–{end_idx}"

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

            start_idx   = (start_offset + (i * window_length_sec))
            end_idx     = start_idx + window_length_sec
            time_range  = f"{start_idx}–{end_idx}"

            if want_prints == True:
                print(f"{time_range:<10}{pred:<15}{'-':<12}-")

            results.append([time_range, pred, "-", "-"])

        if want_prints == True:
            print("_______________________________________________________________________________")
        
    df_result = pd.DataFrame(results, columns=["Time", "Activity", "Probability", "Top-3"])

    df_result['Filtered activity'] = moving_mode_filter(df_result['Activity'], window=3)

    if want_prints:
        print(df_result)

    return df_result, features_df
    
def offlineTest(test_file_path:        str,
                prediction_csv_path:   str,
                fs:                    int, 
                ds_fs:                 int,
                window_length_seconds: int,
                variables:             list[str],
                norm_IMU:              bool,
                want_prints:           bool,
                predictions_csv:       str = "predictions.csv"
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
    

    print(f"Making predictions on data from {test_file_path}: ")

    # Paths
    test_files          = os.listdir(test_file_path)
   
    df_result_all       = [] # List for storing results

    for filename in test_files:
        
        try:
            # file_to_test = test_file_path + "/" + filename

            file_to_test = os.path.join(test_file_path, filename)
            
            # file_to_test_no_ext = file_to_test.replace(".txt", "")
            
            if filename.endswith(".csv"):
                continue  # Skipping .csv files

            elif filename.endswith(".bin"): ##Converting .bin to .txt
            
                convert_bin_to_txt(file_to_test)

            df_result, features_df = runInferenceOnFile(file_to_test, fs, ds_fs, window_length_seconds, want_prints, variables, norm_IMU)
        
            df_result_all.append(df_result)
        except Exception as e:
            print(f"Warning: Skipped {filename}: {e}")

    # Save as csv
    combined_df = pd.concat(df_result_all, ignore_index=True)
    filename_out = os.path.join(prediction_csv_path, predictions_csv)
    combined_df.to_csv(filename_out, index=False)

    ### Finished, printing file  for output file
    print("Done running predictions on datasets")
    print(f"Predictions saved in: {filename_out}")
    print("-" * 50)

    return combined_df #,features_df_all

def calcExposure(combined_df:               pd.DataFrame,
                 csv_path:                  str,
                 window_length_seconds:     int,
                 labels:                    list[str],
                 exposures_and_limits:      dict[str, float],
                 use_granular_labels:       bool,
                 filter_on:                 bool = False,
                 predictions_csv:           str = "predictions.csv",
                 activity_length_csv:       str = "activity_length.csv",
                 summary_csv:               str = "exposure_summary.csv",
                 seconds_to_x:              int = 60,
                 default_activity_length:   float = 0.0,
                ) -> pd.DataFrame:
    
    '''
    Calculates exposure workload based on activity durations and intensity.

    Determines the duration of each predicted activity within a given DataFrame,
    calculates the total exposure score for various hazard types using a
    predefined intensity matrix, and generates a summary comparing these
    scores to safe limits. Prints intermediate results and the final summary.
    '''

    exposures           = exposures_and_limits.keys()
    safe_limit_vector   = exposures_and_limits.values()

    if seconds_to_x == 3600:
        time_unit = 'hours'
    elif seconds_to_x == 60:
        time_unit = 'minutes'
    elif seconds_to_x == 1:
        time_unit = 'seconds'

    # --- 1. Setup from combined DataFrame ---
    if combined_df.empty:

        try:
            full_path   = os.path.join(csv_path, predictions_csv)
            combined_df = pd.read_csv(full_path)
            print(f"Retrieving dataframe from {full_path}.")

        except Exception as e:
            print(f"Error: Unable to read {predictions_csv}: {e}")

    if filter_on == True:
        print(f"Calculating exposure... (filter on).")
        predicted_activities    = combined_df['Filtered activity']
    if filter_on == False:
        print(f"Calculating exposure... (filter off)")
        predicted_activities    = combined_df['Activity']

    activity_counts             = predicted_activities.value_counts()
    activity_length             = activity_counts * window_length_seconds / seconds_to_x
    activity_length_complete    = activity_length.reindex(labels, fill_value=default_activity_length)

    # Save as CSV
    #activity_length_complete.name = f"{"Unit:"} {time_unit}"
    activity_length_filename_out = os.path.join(csv_path, activity_length_csv)
    activity_length_complete.to_csv(activity_length_filename_out)

    # --- 2. Create exposure intensity matrix and calculate exposure, make summary ---
    # x
    activity_duration_vector    = activity_length_complete.values
    
    # A
    exposure_intensity_matrix   = initializeExposureIntensityMatrix(exposures, labels, use_granular_labels, seconds_to_x)
    
    # b = Ax
    total_exposure_vector       = exposure_intensity_matrix @ activity_duration_vector
   
    summary_df                  = exposureSummary(total_exposure_vector, safe_limit_vector, exposures)

    summary_filename_out = os.path.join(csv_path, summary_csv)
    summary_df.to_csv(summary_filename_out)

    # --- 3. Print in terminal ---
    print()
    print(f"Predicted {time_unit}: \n {activity_length_complete.round(decimals=2)}")
    print(f"Risk factors increased. Grind big!")
    print("-" * 99)
    print(f"Exposure intensity matrix ({time_unit}): \n {exposure_intensity_matrix}")
    print("-" * 99)
    print(summary_df.round(decimals=1))
    print("-" * 57)

    return summary_df

def initializeExposureIntensityMatrix(  exposures:                     list[str], 
                                        activities:                    list[str],
                                        use_granular_labels:           bool,
                                        seconds_to_x:                  int,
                                        # Dummy variables
                                        variable_0:                    float = round(random.uniform(1.0, 2.0), 1),
                                        weld_to_rad:                   float = 12.0,
                                        variable_1:                    float = round(random.uniform(1.0, 2.0), 1),
                                        variable_2:                    float = round(random.uniform(1.0, 2.0), 1),
                                        variable_3:                    float = round(random.uniform(1.0, 2.0), 1),
                                        # Dummy sensor proxies
                                        gravityless_norm_accel_mean:   float = round(random.uniform(10.0, 12.0), 1) - 9.81,
                                        gravityless_norm_accel_energy: float = round(random.uniform(10.0, 20.0), 1) - 9.81,
                                        temperature_energy:            float = round(random.uniform(1.0, 2.0), 1)
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
    # Future work: find more proxies, set up sensor readings coming in
    try:
        df = pd.DataFrame(0.0, index=exposures, columns=activities)

        WELD    = df.columns[df.columns.str.startswith('WELD')]
        GRIND   = df.columns[df.columns.str.startswith('GRIND')]
        IMPA    = df.columns[df.columns.str.startswith('IMPA')]
        
        if use_granular_labels == True:
            # If there is something common to all welding or grinding, use *WELD and *GRIND
            
            df.loc['CARCINOGEN',    ['WELDSTMAG', 'WELDSTTIG']] = variable_0
            df.loc['RESPIRATORY',   ['WELDALTIG']]              = variable_1
            df.loc['NEUROTOXIN',    ['WELDSTTIG']]              = variable_2
            df.loc['RADIATION',     [*WELD]]                    = weld_to_rad
            df.loc['NOISE',         [*GRIND]]                   = variable_3
            df.loc['VIBRATION',     [*GRIND]]                   = 2 * gravityless_norm_accel_mean**2 # from https://www.ergonomiportalen.no/kalkulator/#/vibrasjoner 
            df.loc['THERMAL',       [*WELD]]                    = temperature_energy
            df.loc['MSK',           [*GRIND, *IMPA]]            = gravityless_norm_accel_energy

        if use_granular_labels == False:
            
            df.loc['CARCINOGEN',    [*WELD]]                    = variable_0
            df.loc['RESPIRATORY',   [*WELD]]                    = variable_1
            df.loc['NEUROTOXIN',    [*WELD]]                    = variable_2
            df.loc['RADIATION',     [*WELD]]                    = weld_to_rad
            df.loc['NOISE',         [*GRIND, *IMPA]]            = variable_3
            df.loc['VIBRATION',     [*GRIND]]                   = 2 * gravityless_norm_accel_mean**2 # from https://www.ergonomiportalen.no/kalkulator/#/vibrasjoner 
            df.loc['THERMAL',       [*WELD]]                    = temperature_energy
            df.loc['MSK',           [*GRIND, *IMPA]]            = gravityless_norm_accel_energy
            
        df = df * seconds_to_x
        
    except KeyError as e:
        print(f"Error: A row or column label was not found in the DataFrame: {e}. Check 'exposures' and 'activities' lists.")
    except Exception as e:
        print(f"Error: Failed to create exposure intensity matrix: {e}")

    return df

def exposureSummary(total_exposure_vector:     np.array,
                    safe_limit_vector:         np.array,
                    exposures:                 list[str],
                    neutral_limit:             float = 0.80
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
    try:
        df = pd.DataFrame(data, index=exposures)
    except Exception as e:
        print(f"Failed to create exposure summary: {e}")
        print(f"Length of total_exposure_vector: {len(total_exposure_vector)}")
        print(f"Length of safe_limit_vector: {len(safe_limit_vector)}")

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

def moving_mode_filter(series: pd.Series, window: int) -> pd.Series:
    '''
    Applies a moving mode filter to a Pandas Series with deterministic tie-breaking:
    if multiple modes exist, the first one (in order of appearance in the window) is selected.
    '''
    result = []

    for i in range(len(series)):
        # Define the window slice
        start = max(0, i - window // 2)
        end = i + window // 2 + 1
        window_slice = series[start:end]

        if window_slice.empty:
            result.append(None)
        else:
            freqs = Counter(window_slice)
            max_freq = max(freqs.values())
            modes = {k for k, v in freqs.items() if v == max_freq}

            # Choose the first occurring mode from the original window slice
            for val in window_slice:
                if val in modes:
                    result.append(val)
                    break

    return pd.Series(result, index=series.index)
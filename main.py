''' IMPORTS '''
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import math as math
import random as random
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_halving_search_cv
from typing import List, Dict, Any, Tuple, Sequence

# Local imports
from extractFeatures import extractDFfromFile, extractFeaturesFromDF
from machineLearning import setNComponents, evaluateCLFs, makeNClassifiers
from plotting import plotDecisionBoundaries, biplot, biplot3D, plotPCATable, plotFeatureImportance, confusionMatrix, screePlot, plotLearningCurve, datasetOverview, plotScoreVsWindowLength
from preprocessing import fillSets, downsample, pickleFiles, saveJoblibFiles
from testonfile import offlineTest, calcExposure
from config import setupML, loadDataset, main_config


def main(
    # Flags
    want_feature_extraction: bool,
    save_joblib: bool, 
    separate_types: bool,
    want_new_CLFs: bool, 
    want_plots: bool, 
    want_offline_test: bool, 
    want_calc_exposure: bool,

    # Selection & Configuration
    model_selection: list[str],
    method_selection: list[str],
    variance_explained: float,
    random_seed: int,
    window_length_seconds: int,
    test_size: float,
    fs: int,
    ds_fs: int,
    cmap: str,
    norm_IMU: bool,

    # Paths
    test_file_path: str,
    prediction_csv_path: str,
    clf_results_path: str,

    # Specific Parameters
    n_iter: int,
    variables: list[str],
    exposures_and_limits: dict[str, float],

    ) -> Tuple[dict[str, Any], dict[str, Any] | Any]:

    '''
    Orchestrates the activity recognition ML pipeline.

    Covers:
    1. Data Loading & optional Feature Extraction/Loading.
    2. Splitting, Scaling, PCA.
    3. Optional Model Training/Loading.
    4. Evaluation & optional Plotting.
    5. Optional Saving of artifacts (models, scaler, PCA).
    6. Optional Offline Testing & Exposure Calculation.
    Pipeline steps are configurable via parameters.
    '''
    
    fig_list_0, fig_list_1, n_results, accuracy_list = [], [], [], []
    fig_0, fig_1, fig_2, fig_3, fig_4, fig_5 = None, None, None, None, None, None
    plots = {}
    combined_df = pd.DataFrame()
    result = {}
    

    ''' GET ML MODELS AND HPO METHODS '''

    model_names, models, optimization_methods, search_kwargs = setupML()

    ''' LOAD DATASET '''

    path, output_path, labels, original_feature_names, cmap_name, label_mapping = loadDataset(separate_types, norm_IMU, cmap)

    sets, sets_labels, activity_name = fillSets(path)

    ''' FEATURE EXTRACTION '''

    # sets, sets_labels, file, fs, ds_fs, window_length_seconds, norm_IMU, output_path
    # Dette er args til en eventuell feature_extraction()

    if want_feature_extraction:
        # Create dataframe "feature_df" containing all features deemed relevant from the raw sensor data
        # One row in feature_df is all features from one window
        all_window_features: List[Dict[str, Any]] = []
        window_labels:       List[str] = []

        start_time = time.time()
        
        for i, file in enumerate(sets):
            print(f"Extracting features from file: {file}")

            try:
                df = extractDFfromFile(file, fs)
            except Exception as e:
                print(f"Warning: Failed to extract DF from {file}: {e}. Skipping and continuing to next file")
                continue

            if (ds_fs < fs):
                df = downsample(df, fs, ds_fs, variables)
            
            window_df, df_window_labels = extractFeaturesFromDF(df, sets_labels[i], window_length_seconds, ds_fs, norm_IMU)
 

            all_window_features = all_window_features + window_df

            window_labels = window_labels + df_window_labels

            print(f"Total number of windows: {len(window_labels)}")

        feature_df = pd.DataFrame(all_window_features)

        # feature_df, window_labels = extractAllFeatures(sets, sets_labels, window_length_seconds, fs, False)

        end_time = time.time()  # End timer
        elapsed_time = end_time - start_time

        print(f"Features extracted in {elapsed_time} seconds")
            
        feature_df.to_csv(output_path+str(ds_fs)+"_feature_df.csv", index=False)

        with open(output_path+str(ds_fs)+"_window_labels.txt", "w") as fp:
            for item in window_labels:
                fp.write("%s\n" % item)

        fp.close()        

    if "feature_df" not in globals():
        window_labels   = []
        feature_df      = pd.read_csv(output_path+str(ds_fs)+"_feature_df.csv")
        f               = open(output_path+str(ds_fs)+"_window_labels.txt", "r") 
        data            = f.read()
        window_labels   = data.split("\n")
        f.close()
        window_labels.pop()


    ''' SPLITTING TEST/TRAIN + SCALING'''
    
    # for i in randomness(0, max) # velg 5 random randomnesses

    train_data, test_data, train_labels, test_labels = train_test_split(feature_df, window_labels, test_size=test_size,
                                                                        random_state=random_seed, stratify=window_labels)

    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    scaler.fit(train_data)

    train_data_scaled   = scaler.transform(train_data)
    test_data_scaled    = scaler.transform(test_data)

    ''' Principal Component Analysis (PCA)'''

    # Calculate PCA components, create PCA object, fit + transform
    # PCA_components      = setNComponents(train_data_scaled, variance_explained)
    PCA_final           = PCA(n_components = variance_explained)
    PCA_final.fit(train_data_scaled)

    PCA_train_df        = pd.DataFrame(PCA_final.transform(train_data_scaled))
    PCA_test_df         = pd.DataFrame(PCA_final.transform(test_data_scaled))

    if want_new_CLFs:

        ''' HYPERPARAMETER OPTIMIZATION AND CLASSIFIER '''

        n_results = makeNClassifiers(models, model_names, optimization_methods, model_selection,
                                     method_selection, PCA_train_df, train_labels, search_kwargs, n_iter)
        
        ''' EVALUATION '''

        result, accuracy_list = evaluateCLFs(n_results, PCA_test_df, test_labels, clf_results_path)

    else:

        ''' LOAD LAST OPTIMIZED CLFs AND EVALUATION '''

        loaded_results  = joblib.load(clf_results_path) 

        n_results       = loaded_results['n_results']
        result          = loaded_results['result']
        accuracy_list   = loaded_results['accuracy_list']

        print(f"Loading classifier {result['classifier']}.")

    if want_plots:
        
        ''' CONFUSION MATRIX '''

        # fig_0 = plotLearningCurve(n_results, PCA_train_df, train_labels)

        # fig_1 = confusionMatrix(test_labels, PCA_test_df, activity_name, result)
        
        # ''' FEATURE IMPORTANCE '''
        
        # fig_list_0  = plotPCATable(PCA_final, features_per_table=28) 

        # fig_list_1  = plotFeatureImportance(PCA_final, original_feature_names) 

        # fig_2       = screePlot(PCA_final)
        
        # ''' PLOTS OF PCA '''
        
        # fig_3 = biplot(feature_df, scaler, window_labels, label_mapping, want_arrows=False)

        # fig_4 = biplot3D(feature_df, scaler, window_labels, label_mapping, want_arrows=False)
        
        fig_5 = plotDecisionBoundaries(PCA_train_df, train_labels, label_mapping, n_results, accuracy_list, cmap_name)

        # datasetOverview(window_labels, window_length_seconds, test_size)

        plots = {
            'Learning curve': fig_0,
            'Confusion matrix': fig_1,
            'PCA table': fig_list_0,
            'Feature importance': fig_list_1,
            'Scree plot': fig_2,
            'Biplot': fig_3,
            'Biplot 3D': fig_4,
            'Decision boundaries': fig_5
        }

        ## Only to not get double plots when using/ importing main function to user interface
        if __name__ == "__main__":
            plt.show() 

    ''' PICKLING CLASSIFIER '''

    if save_joblib:

        saveJoblibFiles(n_results, result, output_path, PCA_final, scaler)

    ''' OFFLINE TEST '''
    
    if want_offline_test:

        combined_df = offlineTest(test_file_path, prediction_csv_path, fs, ds_fs, window_length_seconds,
                                  variables, norm_IMU, want_prints=True)

    if want_calc_exposure:

        summary_df  = calcExposure(combined_df, prediction_csv_path, window_length_seconds, labels,
                                   exposures_and_limits, filter_on=True)
    
    return plots, result


if __name__ == "__main__" and 1:
    main(**main_config)


if __name__ == "__main__" and 0:

    start_time = time.time()

    f1_mean = []
    f1_std = []
    accuracy_mean = []
    accuracy_std = []

    window_lengths_for_plot = []

    window_sec_lower = 180
    window_sec_upper = 195
    window_sec_interval = 5
    num_randomness: int = 5

    for i in range(window_sec_lower, window_sec_upper, window_sec_interval):
        
        print(f"Current window length: {i} seconds.")
        window_lengths_for_plot.append(i)

        main_config["want_feature_extraction"]  = True
        main_config["window_length_seconds"]    = i

        randomness_list = []
        
        current_f1          = np.zeros(num_randomness)
        current_accuracy    = np.zeros(num_randomness)

        for i in range(num_randomness):
            randomness_list.append(random.randint(i*1000000, (i+1)*1000000 - 1)) 
        
        print(randomness_list)
                               
        f1_scores = []
        accuracies = []

        for j, rand_seed in enumerate(randomness_list): 
          main_config["random_seed"] = rand_seed
          main_config["want_new_CLFs"] = True
          main_config['model_selection'] = 'svm'
          main_config['method_selection'] = []

          print(rand_seed)
          plots, result = main(**main_config)
          current_f1[j] = result["test_f1_score"]
          current_accuracy[j] = result["test_accuracy"]
          print(current_accuracy[j])
          main_config["want_feature_extraction"] = False

        f1_mean.append(current_f1.mean())
        f1_std.append(current_f1.std())

        accuracy_mean.append(current_accuracy.mean())
        accuracy_std.append(current_accuracy.std())

        print(f1_mean)

    plotScoreVsWindowLength(
        window_lengths=window_lengths_for_plot,
        mean_scores=f1_mean,
        std_scores=f1_std,
        score_type='F1 score',
        color='blue',
        filename='plots/f1_score_vs_window_length_2'
    )

    plotScoreVsWindowLength(
        window_lengths=window_lengths_for_plot,
        mean_scores=f1_mean,
        std_scores=f1_std,
        score_type='Accuracy score',
        color='red',
        filename='plots/accuracy_score_vs_window_length_2'
    )

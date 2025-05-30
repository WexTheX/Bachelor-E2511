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
from plotting import plotDecisionBoundaries, biplot, biplot3D, plotPCATable, plotFeatureImportance, confusionMatrix, screePlot, plotLearningCurve, datasetOverview, plotScoreVsWindowLength, plotScalabilityOfScoreTime
from preprocessing import fillSets, downsample, pickleFiles, saveJoblibFiles
from offline_classification import offlineTest, calcExposure
from config import setupML, loadDataset, main_config

def main(
    # Flags
    want_feature_extraction: bool,
    save_joblib: bool, 
    use_granular_labels: bool,
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
    fig_0, fig_1, fig_2, fig_3, fig_4, fig_5, fig_6 = None, None, None, None, None, None, None
    plots = {}
    combined_df, metrics_df = pd.DataFrame(), pd.DataFrame()
    result = {}
    

    ''' GET ML MODELS AND HPO METHODS '''

    model_names, models, optimization_methods, search_kwargs = setupML()

    ''' LOAD DATASET '''

    path, output_path, labels, original_feature_names, cmap_name, label_mapping = loadDataset(use_granular_labels, norm_IMU, cmap)

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

    train_data, test_data, train_labels, test_labels = train_test_split(feature_df, window_labels, test_size=test_size,
                                                                        random_state=random_seed, stratify=window_labels)

    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    scaler.fit(train_data)

    train_data_scaled   = scaler.transform(train_data)
    test_data_scaled    = scaler.transform(test_data)

    ''' Principal Component Analysis (PCA)'''

    # Calculate PCA components, create PCA object, fit + transform
    PCA_components      = setNComponents(train_data_scaled, variance_explained)
    PCA_final           = PCA(n_components = PCA_components, svd_solver='full')
    PCA_final.fit(train_data_scaled)

    PCA_train_df        = pd.DataFrame(PCA_final.transform(train_data_scaled))
    PCA_test_df         = pd.DataFrame(PCA_final.transform(test_data_scaled))

    if want_new_CLFs:

        ''' HYPERPARAMETER OPTIMIZATION AND CLASSIFIER '''

        n_results = makeNClassifiers(models, model_names, optimization_methods, model_selection,
                                     method_selection, PCA_train_df, train_labels, search_kwargs, n_iter)
        
        ''' EVALUATION '''

        result, accuracy_list, metrics_df = evaluateCLFs(n_results, PCA_test_df, test_labels, clf_results_path)

    else:

        ''' LOAD LAST OPTIMIZED CLFs AND EVALUATION '''

        loaded_results  = joblib.load(clf_results_path) 

        n_results       = loaded_results['n_results']
        result          = loaded_results['result']
        accuracy_list   = loaded_results['accuracy_list']

        print(f"Loading classifier {result['classifier']}.")

    if want_plots:
        
        ''' CONFUSION MATRIX '''

        fig_0 = plotLearningCurve(n_results, PCA_train_df, train_labels)

        fig_1 = confusionMatrix(test_labels, PCA_test_df, activity_name, result)
        
        # ''' FEATURE IMPORTANCE '''
        
        fig_list_0  = plotPCATable(PCA_final, features_per_table=28) 

        fig_list_1  = plotFeatureImportance(PCA_final, original_feature_names) 

        fig_2       = screePlot(PCA_final)
        
        ''' PLOTS OF PCA '''
        
        fig_3 = biplot(feature_df, scaler, window_labels, label_mapping, window_length_seconds)
        fig_3 = biplot(feature_df, scaler, window_labels, label_mapping, window_length_seconds)

        fig_4 = biplot3D(feature_df, scaler, window_labels, label_mapping, window_length_seconds)
        fig_4 = biplot3D(feature_df, scaler, window_labels, label_mapping, window_length_seconds)
        
        fig_5 = plotDecisionBoundaries(PCA_train_df, train_labels, label_mapping, n_results, accuracy_list, cmap_name)

        fig_6 = datasetOverview(window_labels, window_length_seconds, test_size)

        

        plots = {
            'Learning curve': fig_0,
            'Confusion matrix': fig_1,
            'PCA table': fig_list_0,
            'Feature importance': fig_list_1,
            'Scree plot': fig_2,
            'Biplot': fig_3,
            'Biplot 3D': fig_4,
            'Decision boundaries': fig_5,
            'Distribution of labels': fig_6
        }
        
        
        ## Only to not get double plots when using/ importing main function to user interface
        if __name__ == "__main__":
            plt.show()
            #fig_1.show()
            

    ''' PICKLING CLASSIFIER '''

    if save_joblib:

        saveJoblibFiles(n_results, result, output_path, PCA_final, scaler)

    ''' OFFLINE TEST '''
    
    if want_offline_test:

        combined_df = offlineTest(test_file_path, prediction_csv_path, fs, ds_fs, window_length_seconds,
                                  variables, norm_IMU, want_prints=True)

    if want_calc_exposure:

        summary_df  = calcExposure(combined_df, prediction_csv_path, window_length_seconds, labels,
                                   exposures_and_limits, use_granular_labels)
    return plots, result, metrics_df


randomness_loop = False

if __name__ == "__main__" and randomness_loop == False:
    main(**main_config)


if __name__ == "__main__" and randomness_loop == True:

    start_time = time.time()

    f1_mean, f1_std, accuracy_mean, accuracy_std, window_lengths_for_plot, randomness_list, f1_scores, accuracies = [], [], [], [], [], [], [], []

    window_sec_lower = 20
    window_sec_upper = 245
    window_sec_interval = 5

    all_window_dfs = []
    all_window_dfs_std = []

    import matplotlib.cm as cm
    colors = cm.get_cmap('tab10')

    for i in range(window_sec_lower, window_sec_upper, window_sec_interval):
        
        print(f"Current window length: {i} seconds.")
        window_lengths_for_plot.append(i)

        num_randomness: int = 20

        current_f1          = np.zeros(num_randomness)
        current_accuracy    = np.zeros(num_randomness)

        randomness_list = []

        for r in range(num_randomness):
            randomness_list.append(random.randint(r*1000000, (r+1)*1000000 - 1)) 
        
        print(randomness_list)

        main_config["want_new_CLFs"] = True
        main_config['model_selection'] = ['gnb', 'lr', 'svm', 'rf', 'knn', 'ada', 'gb']
        main_config['method_selection'] = []

        dfs = list(range(len(randomness_list)))


        main_config["want_feature_extraction"]  = True
        main_config["window_length_seconds"]    = i
        main_config["want_plots"] = False

        for j, rand_seed in enumerate(randomness_list): 
            
            main_config["random_seed"] = rand_seed
            # main_config["want_new_CLFs"] = False

            print(f"Random seed nr in inner loop: {j}: {rand_seed}")
            plots, result, metrics_df = main(**main_config)
            metrics_df = metrics_df.reset_index(level='optimalizer', drop=True)
            # print(f"Metrics_df {j}: {metrics_df}")
            
            dfs[j] = metrics_df

            main_config["want_feature_extraction"]  = False
        
        mean_dfs = sum(dfs) / len(dfs)

        # Convert list of DataFrames to 3D NumPy array: (N, rows, columns)
        data = np.array([df.values for df in dfs])

        # Compute std along axis 0 (across the N DataFrames)
        std_array = np.std(data, axis=0, ddof=1)  # ddof=1 for sample std

        # Reconstruct DataFrame with original index and columns
        std_df = pd.DataFrame(std_array, index=dfs[0].index, columns=dfs[0].columns)

        print("Mean DF: ")
        print(mean_dfs)
        print(std_df)

    
        all_window_dfs.append(mean_dfs)
        all_window_dfs_std.append(std_df)

    num_dfs = len(all_window_dfs)
    num_rows = len(main_config['model_selection'])

    for row_idx in range(num_rows):
        f1_scores = []
        std_scores = []

        for df in all_window_dfs:
            f1_scores.append(df.iloc[row_idx]['f1_score'])
        for df in all_window_dfs_std:
            std_scores.append(df.iloc[row_idx]['f1_score'])

        f1_scores = np.array(f1_scores, dtype=np.float64)
        std_scores = np.array(std_scores, dtype=np.float64)
        window_lengths_for_plot = np.array(window_lengths_for_plot, dtype=np.float64)

        lower = f1_scores - std_scores
        upper = f1_scores + std_scores

        # Cap manually (in-place) to avoid relying on np.clip magic
        lower[lower < 0.0] = 0.0
        upper[upper > 1.0] = 1.0

        # Clip f1_scores too just in case
        f1_scores = np.clip(f1_scores, 0.0, 1.0)

        color = colors(row_idx % 10)

        # range(num_dfs)
        plt.figure()
        plt.plot(window_lengths_for_plot, f1_scores, marker='o', color=color)
        plt.fill_between(window_lengths_for_plot, lower, upper,
                     color=color, alpha=0.2, label=f'Mean ± 1 STD')
        # plt.title(f'F1 Score for Row {row_idx}')
        plt.xticks(np.arange(window_sec_lower, window_sec_upper, 25))
        plt.xlabel('Window length')
        plt.ylabel('F1 score')
        # ylim = plt.ylim()       # get current y-axis limits (returns (lower, upper))
        plt.ylim(0.45, 1.05)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'plots/f1_row_{row_idx}.png', dpi=300, bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(10, 6))

    for row_idx in range(num_rows):
        f1_scores = []

        for df in all_window_dfs:
            f1_scores.append(df.iloc[row_idx]['f1_score'])

        f1_scores = np.array(f1_scores, dtype=np.float64)

        color = colors(row_idx % 10)
        label = all_window_dfs[0].index[row_idx]  # Assumes model name is in index

        plt.plot(window_lengths_for_plot, f1_scores, color=color, label=label)

    # plt.title("F1 Score Across Window Sizes for All Models")
    plt.xlabel("Window length")
    plt.ylabel("F1 Score")
    ylim = plt.ylim()       # get current y-axis limits (returns (lower, upper))
    plt.ylim(ylim[0]-0.05, 1.05)
    plt.xticks(np.arange(window_sec_lower, window_sec_upper, 25))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/f1_all_models.png', dpi=300, bbox_inches='tight')

    stop_time = time.time()

    print(f"Runtime: {stop_time-start_time}")
    plt.show()

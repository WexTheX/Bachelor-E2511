''' IMPORTS '''
import time
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import streamlit as st
import joblib

from sklearn.inspection import permutation_importance
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_halving_search_cv
from sklearn.tree import DecisionTreeClassifier
from typing import List, Dict, Any, Tuple, Sequence


# Local imports
# from FOLDER import FILE as F
from extractFeatures import extractAllFeatures, extractDFfromFile, extractFeaturesFromDF
from machineLearning import trainScaler, setNComponents, evaluateCLFs, makeNClassifiers
from plotting import plotDecisionBoundaries, biplot, biplot3D, PCA_table_plot, plotFeatureImportance, confusionMatrix
from Preprocessing.preprocessing import fillSets, downsample, pickleFiles
from testonfile import offlineTest, calcExposure
from config import setupML, loadDataset, main_config


def main(
    # Flags
    want_feature_extraction: bool, 
    want_pickle: bool, 
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

    # Paths
    test_file_path: str,
    prediction_csv_path: str,
    clf_results_path: str,

    # Specific Parameters
    n_iter: int,
    exposures: list[str],
    safe_limit_vector: list[float],
    variables: list[str]
) -> Tuple[List, dict, list[float]]:


    ''' SETUP '''

    fig_list_1, n_results, accuracy_list = [], [], []
    fig_1, fig_2, fig_3 = None, None, None
    combined_df = pd.DataFrame()
    result = {}

    ''' GET ML MODELS AND HPO METHODS '''

    model_names, models, optimization_methods, search_kwargs = setupML()

    ''' LOAD DATASET '''

    path, output_path, labels = loadDataset(separate_types)

    num_labels      = len(labels)
    cmap_name       = plt.get_cmap(cmap, num_labels)
    label_mapping   = {label: cmap_name(i) for i, label in enumerate(labels)}

    path_names          = os.listdir(path)
    activity_name       = [name.upper() for name in path_names]

    sets, sets_labels   = fillSets(path, path_names, activity_name)

    ''' FEATURE EXTRACTION '''

    if (want_feature_extraction):
        # Create dataframe "feature_df" containing all features deemed relevant from the raw sensor data
        # One row in feature_df is all features from one window
        all_window_features = []
        window_labels = []

        start_time = time.time()
        
        for i, file in enumerate(sets):
            print(f"Extracting features from file: {file}")
            fe_df = extractDFfromFile(file, fs)

            if (ds_fs != fs):
                fe_df = downsample(fe_df, fs, ds_fs)
            
            window_df, df_window_labels = extractFeaturesFromDF(fe_df, sets_labels[i], window_length_seconds, ds_fs, False)

            all_window_features = all_window_features + window_df

            window_labels = window_labels + df_window_labels

            print(f"Total number of windows: {len(window_labels)}")

        feature_df = pd.DataFrame(all_window_features)

        # feature_df, window_labels = extractAllFeatures(sets, sets_labels, window_length_seconds, fs, False)

        end_time = time.time()  # End timer
        elapsed_time = end_time - start_time
        print(f"Features extracted in {elapsed_time} seconds")
            
        feature_df.to_csv(output_path+str(ds_fs)+"feature_df.csv", index=False)

        with open(output_path+"window_labels.txt", "w") as fp:
            for item in window_labels:
                fp.write("%s\n" % item)

        fp.close()        

    if "feature_df" not in globals():
        window_labels   = []
        feature_df      = pd.read_csv(output_path+str(ds_fs)+"feature_df.csv")
        f               = open(output_path+"window_labels.txt", "r") 
        data            = f.read()
        window_labels   = data.split("\n")
        f.close()
        window_labels.pop()


    ''' SPLITTING TEST/TRAIN + SCALING'''
    
    train_data, test_data, train_labels, test_labels = train_test_split(feature_df, window_labels, test_size=test_size, random_state=random_seed, stratify=window_labels)

    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    scaler.fit(train_data)

    train_data_scaled   = scaler.transform(train_data)
    test_data_scaled    = scaler.transform(test_data)

    ''' Principal Component Analysis (PCA)'''

    # Calculate PCA components, create PCA object, fit + transform
    PCA_components      = setNComponents(train_data_scaled, variance_explained)
    PCA_final           = PCA(n_components = PCA_components)
    PCA_final.fit(train_data_scaled)

    PCA_train_df        = pd.DataFrame(PCA_final.transform(train_data_scaled))
    PCA_test_df         = pd.DataFrame(PCA_final.transform(test_data_scaled))

    if want_new_CLFs:

        ''' HYPERPARAMETER OPTIMIZATION AND CLASSIFIER '''

        n_results = makeNClassifiers(models, model_names, optimization_methods, model_selection, method_selection, PCA_train_df, train_labels, search_kwargs, n_iter)

        ''' EVALUATION '''

        result, accuracy_list = evaluateCLFs(n_results, PCA_test_df, test_labels, clf_results_path)

    else:

        ''' LOAD LAST OPTIMIZED CLFs AND EVALUATION '''

        loaded_results  = joblib.load(clf_results_path) 

        n_results       = loaded_results['n_results']
        result          = loaded_results['result']
        accuracy_list   = loaded_results['accuracy_list']

    if want_plots:
        
        ''' CONFUSION MATRIX '''

        # test_predict = result['classifier'].predict(PCA_test_df)
        confusionMatrix(test_labels, PCA_test_df, activity_name, result)
        
        ''' FEATURE IMPORTANCE '''
        
        fig_list_1 = PCA_table_plot(PCA_final, features_per_PCA=28) #train_data_scaled, n_components=PCA_components, 

        pfig_0      = plotFeatureImportance(PCA_final) 
        
        ''' PLOTS OF PCA '''
        
        fig_1 = biplot(feature_df, scaler, window_labels, label_mapping, want_arrows=False)

        fig_2 = biplot3D(feature_df, scaler, window_labels, label_mapping, want_arrows=False)
        
        fig_3 = plotDecisionBoundaries(PCA_train_df, train_labels, label_mapping, n_results, accuracy_list, cmap_name)
        
        if __name__ == "__main__":
            plt.show() 

    ''' PICKLING CLASSIFIER '''

    if want_pickle:

        pickleFiles(n_results, result, output_path, PCA_final, scaler)

    ''' OFFLINE TEST '''
    
    if want_offline_test:

        combined_df = offlineTest(test_file_path, prediction_csv_path, fs, ds_fs, window_length_seconds, want_prints=True)

    if want_calc_exposure:

        summary_df  = calcExposure(combined_df, prediction_csv_path, window_length_seconds, labels, exposures, safe_limit_vector, prediction_csv_path, filter_on=True)

    
    return [fig_list_1, fig_1, fig_2, fig_3], result, accuracy_list


if __name__ == "__main__":

    main(**main_config)
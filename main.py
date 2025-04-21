''' IMPORTS '''
import time
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import streamlit as st

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

# Local imports
# from FOLDER import FILE as F
from extractFeatures import extractAllFeatures, extractDFfromFile, extractFeaturesFromDF
from machineLearning import trainScaler, setNComponents, evaluateCLFs, makeNClassifiers
from plotting import plotDecisionBoundaries, biplot, biplot3D, PCA_table_plot, plotKNNboundries, confusionMatrix
from Preprocessing.preprocessing import fillSets, downsample, pickleFiles
from testonfile import offlineTest, calcExposure



def main(want_feature_extraction, want_pickle, separate_types, want_plots, want_offline_test, want_calc_exposure,
         model_selection, method_selection, variance_explained, random_seed ,window_length_seconds, test_size, fs,
         ds_fs, cmap, test_file_path, prediction_csv_path ):

    variables = ["Timestamp","Gyr.X","Gyr.Y","Gyr.Z","Axl.X","Axl.Y","Axl.Z","Mag.X","Mag.Y","Mag.Z","Temp"]
    
    fig_list_1, n_results, accuracy_list = [], [], []
    fig_1, fig_2, fig_3 = None, None, None
    combined_df = pd.DataFrame()
    result = {}

    ''' BASE ESTIMATORS '''

    base_params =  {'class_weight': 'balanced', 
                    'random_state': random_seed}

    SVM_base    = svm.SVC(**base_params, probability=True)
    RF_base     = RandomForestClassifier(**base_params)
    KNN_base    = KNeighborsClassifier()
    GNB_base    = GaussianNB()
    LR_base     = LogisticRegression(**base_params)
    GB_base     = GradientBoostingClassifier(random_state=random_seed)
    ADA_base    = AdaBoostClassifier(estimator=DecisionTreeClassifier(), random_state=random_seed)

    ''' HYPER PARAMETER VARIABLES '''

    num_folds = 3
    n_iter = 30

    SVM_param_grid = {
        "C":                    [0.01, 0.1,
                                # 1.0, 10.0, 100.0
                                ],
        "kernel":               ["linear", "poly", "rbf", "sigmoid"],
        # "gamma":                [0.01, 0.1, 1, 10.0, 100.0],
        # "coef0":                [0.0, 0.5, 1.0],
        # "degree":               [2, 3, 4, 5]
    }


    RF_param_grid = {
        'n_estimators':         [50, 100, 200],         # Number of trees in the forest
        'max_depth':            [10, 20, 30, None],     # Maximum depth of each tree
        'min_samples_split':    [2, 5, 10],             # Minimum samples required to split a node
        'min_samples_leaf':     [1, 2, 4],              # Minimum samples required in a leaf node
        'max_features':         ['sqrt', 'log2'],       # Number of features considered for splitting
        # 'bootstrap':            [True, False],          # Whether to use bootstrapped samples
        'criterion':            ['gini', 'entropy']     # Splitting criteria
    }

    KNN_param_grid = {
        'algorithm':            ['ball_tree', 'kd_tree', 'brute'], 
        # 'leaf_size':          [30], 
        # 'metric':             ['minkowski'], 
        # 'metric_params':      [None], 
        # 'n_jobs':             [None], 
        'n_neighbors':          [3, 4, 5], 
        'p':                    [1, 2], 
        'weights':              ['uniform', 'distance']
    }

    GNB_param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }

    LR_param_grid = {
        'C':                    [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],             #
        # 'dual':                 [False],                                    # Dual or Primal formulation
        # 'fit_intercept':        [True],                                     # Constant added to function (bias)             
        # 'intercept_scaling':    [1],                                        # Only useful when Solver = liblinear, fit_intercept = true
        # 'l1_ratio':             [None],                                     # ???
        'max_iter':             [100],                                      # Max iterations for solver to converge
        # 'multi_class':          ['deprecated'],                             # Deprecated
        # 'n_jobs':               [None],                                     # Amount of jobs that can run at the same time, (also set in CV, error if both)
        # 'penalty':              ['l1', 'l2', 'elasticnet', None],           # Norm of the penalty 
        # 'solver':               ['lbfgs', 'newton-cg', 'sag', 'saga'],      # Algorithm for optimization problem
        'tol':                  [0.0001],                                   # Tolerance for stopping criteria
        #'warm_start':           [False]                                      # Reuse previous calls solution
    }

    GB_param_grid = {
        'n_estimators':         [100, 200, 300],
        'learning_rate':        [0.01, 0.05, 0.1],
        'max_depth':            [3, 5, 7],
        'min_samples_split':    [2, 5, 10],
        'min_samples_leaf':     [1, 2, 4],
        'subsample':            [0.6, 0.8, 1.0],
        'max_features':         ['sqrt', 'log2', None],
    }
    
    ADA_param_grid = {
        'n_estimators':                 [50, 100, 200],
        'learning_rate':                [0.01, 0.1, 1.0],
        'estimator__max_depth':         [1, 3, 5],
        'estimator__min_samples_split': [2, 5]
    }

    model_names = {
        'SVM':  'Support Vector Machine', 
        'RF':   'Random Forest',
        'KNN':  'K-Nearest Neighbors',
        'GNB':  'Gaussian Naive Bayes',
        'LR':   'Linear Regression',
        'GB':   'Gradient Boosting',
        'ADA':  'AdaBoost'
    }

    models = {
        'SVM':  (SVM_base,  SVM_param_grid), 
        'RF':   (RF_base,   RF_param_grid),
        'KNN':  (KNN_base,  KNN_param_grid),
        'GNB':  (GNB_base,  GNB_param_grid),
        'LR':   (LR_base,   LR_param_grid),
        'GB':   (GB_base,   GB_param_grid),
        'ADA':  (ADA_base,  ADA_param_grid)
    }

    optimization_methods = {
        'BS':   'BayesSearchCV',
        'RS':   'RandomizedSearchCV',
        'GS':   'GridSearchCV',
        'HGS':  'HalvingGridSearchCV'
    }

    search_kwargs = {
        'n_jobs':             -1, 
        'verbose':             0,
        'cv':                  StratifiedKFold(n_splits=num_folds),
        'scoring':             'f1_weighted',
        'return_train_score':  True
    }

    ''' LOAD DATASET '''

    # Different folder for separated and not separated
    if separate_types:
        
        path            = "Preprocessing/DatafilesSeparated" 
        output_path     = "OutputFiles/Separated/"
        test_path       = "testFiles/"

        labels = [
            'GRINDBIG', 'GRINDSMALL',
            'IDLE', 'IMPA', 'GRINDMED', 'GRINDSMALLCORDED',
            'SANDSIM',
            'WELDALTIG', 'WELDSTMAG', 'WELDSTTIG'
        ]

    else:
        path            = "Preprocessing/Datafiles"
        output_path     = "OutputFiles/"
        test_path       = "testFiles/"

        labels = [
            'IDLE', 'GRINDING', 'IMPA', 'SANDSIMULATED', 'WELDING'
        ]

    exposures = [
        'CARCINOGEN', 'RESPIRATORY', 'NEUROTOXIN', 'RADIATION', 'NOISE', 'VIBRATION', 'THERMAL', 'MSK'
    ]

    safe_limit_vector = [1000.0, 750.0, 30.0, 120.0, 900.0, 400.0, 2500.0, 400]

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

    ''' HYPERPARAMETER OPTIMIZATION AND CLASSIFIER '''

    n_results = makeNClassifiers(models, model_names, optimization_methods, model_selection, method_selection, PCA_train_df, train_labels, search_kwargs, n_iter)

    ''' EVALUATION '''

    result, accuracy_list = evaluateCLFs(n_results, PCA_test_df, test_labels, want_plots, activity_name)

    if want_plots:
        
        ''' CONFUSION MATRIX '''
        test_predict = result['classifier'].predict(PCA_test_df)
        confusionMatrix(test_labels, test_predict, activity_name, result)
        
        ''' FEATURE IMPORTANCE '''
        
        fig_list_1 = PCA_table_plot(train_data_scaled, n_components=PCA_components, features_per_PCA=28)   
        
        ''' 2D PLOTS OF PCA '''
        
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

        summary_df  = calcExposure(combined_df, window_length_seconds, labels, exposures, safe_limit_vector, prediction_csv_path, filter_on=True)

    
    return [fig_list_1, fig_1, fig_2, fig_3], result, accuracy_list


if __name__ == "__main__":

    ''' GLOBAL VARIABLES '''

    want_feature_extraction = 0
    want_pickle             = 0 # Pickle the classifier, scaler and PCA objects.
    separate_types          = 1 # Granular classification
    want_plots              = 1
    want_offline_test       = 0
    want_calc_exposure      = 0

    model_selection         = ['svm']
    method_selection        = ['rs']

    ''' DATASET VARIABLES '''

    variance_explained      = 3
    random_seed             = 420
    window_length_seconds   = 20
    test_size               = 0.25
    fs                      = 800
    ds_fs                   = 200
    cmap                    = 'tab10'

    ''' LOAD PATH NAMES'''
    test_file_path = "testOnFile/testFiles"
    prediction_csv_path = "testOnFile"

    
    main(want_feature_extraction, want_pickle, 
         separate_types, want_plots, want_offline_test, want_calc_exposure,
         model_selection, method_selection, variance_explained,
         random_seed ,window_length_seconds, test_size, fs, ds_fs, 
         cmap, test_file_path, prediction_csv_path)


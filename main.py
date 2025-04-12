''' IMPORTS '''
import time
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.inspection import permutation_importance
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_halving_search_cv

# Local imports
# from FOLDER import FILE as F
from extractFeatures import extractAllFeatures, extractDFfromFile, extractFeaturesFromDF
from machineLearning import trainScaler, setNComponents, evaluateCLFs, makeNClassifiers
from plotting import plotBoundaryConditions, biplot, biplot3D, PCA_table_plot, plotKNNboundries
from Preprocessing.preprocessing import fillSets, downsample, pickleFiles
from testonfile import runInferenceOnFile, offlineTest, labelFilter, calcWorkload



def main(want_feature_extraction, want_pickle, separate_types, want_plots, model_selection, method_selection,
         variance_explained ,random_seed ,window_length_seconds ,test_size , fs, ds_fs, cmap_name, test_file_path,
         prediction_csv_path ):

    variables = ["Timestamp","Gyr.X","Gyr.Y","Gyr.Z","Axl.X","Axl.Y","Axl.Z","Mag.X","Mag.Y","Mag.Z","Temp"]

    ''' BASE ESTIMATORS '''

    base_params =  {'class_weight': 'balanced', 
                    'random_state': random_seed}

    # base_paramssvm = {
    #     'class_weight': 'balanced',
    #     'probability': True,
    #     'random_state': randomness
    # }

    SVM_base    = svm.SVC(**base_params, probability=True)
    RF_base     = RandomForestClassifier(**base_params)
    KNN_base    = KNeighborsClassifier()
    GNB_base    = GaussianNB()
    LR_base     = LogisticRegression(**base_params)

    ''' HYPER PARAMETER VARIABLES '''

    num_folds = 3
    n_iter = 30

    SVM_param_grid = {
        "C":                    [0.01, 0.1,
                                # 1, 10, 100
                                ],
        "kernel":               ["linear", "poly", "rbf", "sigmoid"],
        # "gamma":                [0.01, 0.1, 1, 10, 100],
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
        'priors':               [None], 
        'var_smoothing':        [1e-09]
    }

    LR_param_grid = {
        'C':                    [0.001, 0.01, 0.1, 1, 10, 100],             #
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

    models = {
            'SVM':  (SVM_base,  SVM_param_grid), 
            'RF':   (RF_base,   RF_param_grid),
            'KNN':  (KNN_base,  KNN_param_grid),
            'GNB':  (GNB_base,  GNB_param_grid),
            'LR':   (LR_base,   LR_param_grid)
            }

    optimization_methods = ['BayesSearchCV', 'RandomizedSearchCV', 'GridSearchCV', 'HalvingGridSearchCV', 'Base model']

    search_kwargs = {'n_jobs':             -1, 
                    'verbose':             0,
                    'cv':                  TimeSeriesSplit(n_splits=num_folds),
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
                'IDLE','IMPA','GRINDMED', 
                'SANDSIM',
                'WELDALTIG', 'WELDSTMAG', 'WELDSTTIG'
        ]

    else:
        path            = "Preprocessing/Datafiles"
        output_path     = "OutputFiles/"
        test_path       = "testFiles/"

        labels = ['IDLE', 'GRINDING', 'IMPA', 'SANDSIMULATED', 'WELDING']

    num_labels      = len(labels)
    cmap            = plt.get_cmap(cmap_name, num_labels)
    label_mapping   = {label: cmap(i) for i, label in enumerate(labels)}


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

    n_results = makeNClassifiers(models, optimization_methods, model_selection, method_selection, PCA_train_df, train_labels, search_kwargs, n_iter)

    ''' EVALUATION '''

    result, accuracy_list = evaluateCLFs(n_results, PCA_test_df, test_labels, want_plots, activity_name)

    if want_plots:
        
        ''' FEATURE IMPORTANCE '''
        
        PCA_table_plot(train_data_scaled, n_components=5, features_per_PCA=73)   

        ''' 2D PLOTS OF PCA '''

        biplot(feature_df, scaler, window_labels, label_mapping, want_arrows=False)

        biplot3D(feature_df, scaler, window_labels, label_mapping, want_arrows=False)
        
        plotBoundaryConditions(PCA_train_df, train_labels, label_mapping, n_results, accuracy_list, cmap)

        plt.show()

    ''' PICKLING CLASSIFIER '''

    pickleFiles(want_pickle, n_results, result, output_path, PCA_final, scaler)

    ''' OFFLINE TEST '''
    
    combined_df = offlineTest(test_file_path, prediction_csv_path, fs, ds_fs, window_length_seconds, want_prints=True)

    calcWorkload(combined_df, window_length_seconds, labels)




if __name__ == "__main__":

    ''' GLOBAL VARIABLES '''

    want_feature_extraction = 0
    want_pickle             = 0 # Pickle the classifier, scaler and PCA objects.
    separate_types          = 1
    want_plots              = 1

    model_selection         = ['SVM']
    method_selection        = ['GridSearchCV']

    ''' DATASET VARIABLES '''

    variance_explained      = 2
    random_seed             = 1231
    window_length_seconds   = 20
    test_size               = 0.25
    fs                      = 800
    ds_fs                   = 200
    cmap_name               = 'tab10'

    ''' LOAD PATH NAMES'''
    test_file_path = "testOnFile/testFiles"
    prediction_csv_path = "testOnFile"


    main(want_feature_extraction, want_pickle, 
         separate_types, want_plots,
         model_selection, method_selection, variance_explained ,
         random_seed ,window_length_seconds ,test_size , fs, ds_fs, 
         cmap_name, test_file_path, prediction_csv_path)


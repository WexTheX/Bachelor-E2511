''' IMPORTS '''
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.inspection import DecisionBoundaryDisplay, permutation_importance
from skopt.space import Real, Categorical, Integer
from sklearn import svm, metrics, dummy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Local imports
# from FOLDER import FILE as F
from extractFeatures import extractAllFeatures, extractDFfromFile, extractFeaturesFromDF
from machineLearning import trainScaler, setNComponents, makeClassifier, makeSVMClassifier, makeRFClassifier, makeKNNClassifier, makeGNBClassifier, evaluateCLF, evaluateCLFs, makeNClassifiers
from plotting import plotBoundaryConditions, biplot, plot_SVM_boundaries, PCA_table_plot, plotKNNboundries
from Preprocessing.preprocessing import fillSets, downsample

''' GLOBAL VARIABLES '''

want_feature_extraction = 1
pickle_files            = 1 # Pickle the classifier, scaler and PCA objects.
separate_types          = 1
want_plots              = 1
ML_models               = ["SVM", "RF", "KNN", "GNB", "COMPARE"]
ML_model                = "SVM"
Splitting_method        = ["StratifiedKFOLD", "TimeSeriesSplit"]
Splitting_method        = "TimeseriesSplit"

''' DATASET VARIABLES '''

variance_explained      = 2
random_seed             = 333
window_length_seconds   = 20
test_size               = 0.25
fs                      = 800
ds_fs                   = 800
variables               = ["Timestamp","Gyr.X","Gyr.Y","Gyr.Z","Axl.X","Axl.Y","Axl.Z","Mag.X","Mag.Y","Mag.Z","Temp"]

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

SVM_param_grid = {
    "C":                    [0.001, 0.01],
    "kernel":               ["linear", "poly", "rbf", "sigmoid"],
    "gamma":                [0.01, 0.1],
    "coef0":                [0.0, 1.0],
    "degree":               [2, 3]
}

RF_param_grid = {
    'n_estimators':         [50, 100, 200],  # Number of trees in the forest
    'max_depth':            [10, 20, 30, None],  # Maximum depth of each tree
    # 'min_samples_split':  [2, 5, 10],  # Minimum samples required to split a node
    # 'min_samples_leaf':   [1, 2, 4],  # Minimum samples required in a leaf node
    # 'max_features':       ['sqrt', 'log2'],  # Number of features considered for splitting
    # 'bootstrap':          [True, False],  # Whether to use bootstrapped samples
    'criterion':            ['gini', 'entropy']  # Splitting criteria
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
    'C':                    [0.001, 0.01], 
    'dual':                 [False], 
    'fit_intercept':        [True], 
    'intercept_scaling':    [1], 
    # 'l1_ratio':             [None], 
    'max_iter':             [100], 
    'multi_class':          ['deprecated'], 
    # 'n_jobs':               [None], 
    'penalty':              ['l2'], 
    'solver':               ['lbfgs'], 
    'tol':                  [0.0001], 
    'warm_start':           [False]
}

models = {
        'SVM':  (SVM_base,  SVM_param_grid), 
        'RF':   (RF_base,   RF_param_grid),
        'KNN':  (KNN_base,  KNN_param_grid),
        'GNB':  (GNB_base,  GNB_param_grid),
        'LR':   (LR_base,   LR_param_grid)
        }

optimization_methods = ['BayesSearchCV', 'RandomizedSearchCV', 'GridSearchCV', 'HalvingGridSearchCV', 'Base model']

search_kwargs = {'n_jobs':              -1, 
                 'verbose':             0,
                 'cv':                  TimeSeriesSplit(n_splits=num_folds),
                 'scoring':             'f1_weighted',
                 'return_train_score':  True
                }

''' USER INPUTS '''

# answer_FE = input("Do you want feature extraction? (Y | N) (Default N)")
# if(answer_FE.upper() == "Y"):
#     want_feature_extraction = True

# answer_ST = input("Do you want to separate by type (TIG and MIG vs only welding)? (Y | N) (Default N)")
# if(answer_ST.upper() == "Y"):
#     separate_types = True

# answer_ML = input(f"Choose ML model (Default SVM): {ML_models}.")
# if(answer_ML.upper() = "RF"):
#     ML_model = ML_models[1]

# answer_plot = input("Do you want plots? (Y | N) (Default N)")
# if(answer_plot.upper() == "Y"):
#     want_plots = True


''' LOAD DATASET '''

# Different folder for separated and not separated
if (separate_types):
    path            = "Preprocessing/DatafilesSeparated" 
    output_path     = "OutputFiles/Separated/"
    test_path       = "testFiles/"

    label_mapping   = {
                        'IDLE':         (0.0, 0.0, 0.0), 
                        'GRINDBIG':     (1.0, 0.0, 0.0), 'GRINDMED':    (1.0, 0.5, 0.0), 'GRINDSMALL':  (1.0, 0.0, 0.5),
                        'IMPA':         (0.5, 0.5, 0.5), 
                        'SANDSIM':      (0.0, 1.0, 0.0), 
                        'WELDALTIG':    (0.0, 0.0, 1.0), 'WELDSTMAG':   (0.5, 0.0, 1.0), 'WELDSTTIG':   (0.0, 0.5, 1.0)
    }

else:
    path            = "Preprocessing/Datafiles"
    output_path     = "OutputFiles/"
    test_path       = "testFiles/"

    label_mapping   = {'IDLE': (0.0, 0.0, 0.0), 'GRINDING': (1.0, 0.0, 0.0), 'IMPA': (0.5, 0.5, 0.5), 'SANDSIMULATED': (0.0, 1.0, 0.0), 'WELDING': (0.0, 0.0, 1.0)}

path_names          = os.listdir(path)
activity_name       = [name.upper() for name in path_names]

sets, sets_labels = fillSets(path, path_names, activity_name)

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

mapped_labels = np.array([label_mapping[label] for label in train_labels])

scaler = StandardScaler()
scaler.set_output(transform="pandas")

train_data_scaled   = scaler.fit_transform(train_data)
test_data_scaled    = scaler.transform(test_data)

total_data_scaled   = scaler.fit_transform(feature_df)


# total_data_scaled = scaleFeatures(feature_df, 1)
# train_data_scaled = scaleFeatures(train_data, 1)
# test_data_scaled = scaleFeatures(test_data, 0)


''' Principal Component Analysis (PCA)'''

# Calculate PCA components, create PCA object, fit + transform
PCA_components      = setNComponents(train_data_scaled, variance_explained=variance_explained)
PCA_final           = PCA(n_components = PCA_components)

PCA_train_df        = pd.DataFrame(PCA_final.fit_transform(train_data_scaled))
PCA_test_df         = pd.DataFrame(PCA_final.transform(test_data_scaled))

''' HYPERPARAMETER OPTIMIZATION AND CLASSIFIER '''

model_selection     = ['LR', 'GNB', 'KNN', 'SVM']
method_selection    = ['GridSearchCV', 'RandomizedSearchCV']

results = makeNClassifiers(models, optimization_methods, model_selection, method_selection, PCA_train_df, train_labels, search_kwargs, n_iter=30)

# models = (
#         (SVM_base, SVM_param_grid), 
#         (RF_base, RF_param_grid),
#         (KNN_base, KNN_param_grid),
#         (GNB_base, GNB_param_grid) )

# for base_model, param_grid in models:
#     for method in optimization_methods:

#         clf, best_params = makeClassifier(base_model, param_grid, method, PCA_train_df, train_labels, search_kwargs, n_iter=30)

#         classifiers.append(clf)
#         best_clf_params.append(best_params)
#         optimization_list.append(method)

# print(f"Optimization list: {optimization_list}")
# print(f"Classifiers: {classifiers}")
# print(f"best_params: {best_clf_params}")

# if (ML_model.upper() == "SVM"):
#     for method in optimization_methods:
#         t_clf, t_best_clf_params = makeSVMClassifier(method, SVM_base, num_folds, hyperparams_SVM, PCA_train_df, train_labels, train_data, variance_explained)
#         classifiers.append(t_clf)
#         best_clf_params.append(t_best_clf_params)

# elif (ML_model.upper() == "RF"):
#     for method in optimization_methods:
#         t_clf, t_best_clf_params = makeRFClassifier(method, RF_base, num_folds, hyperparams_RF, PCA_train_df, train_labels)
#         classifiers.append(t_clf)
#         best_clf_params.append(t_best_clf_params)

# elif (ML_model.upper() == "KNN"):
#     for method in optimization_methods:
#         t_clf, t_best_clf_params = makeKNNClassifier(method, KNN_base, num_folds, hyperparams_KNN, PCA_train_df, train_labels)
#         classifiers.append(t_clf)
#         best_clf_params.append(t_best_clf_params)

# elif (ML_model.upper() == "GNB"):
#     for method in optimization_methods:
#         t_clf, t_best_clf_params = makeGNBClassifier(method, GNB_base, num_folds, hyperparams_GNB, PCA_train_df, train_labels)
#         classifiers.append(t_clf)
#         best_clf_params.append(t_best_clf_params)

''' EVALUATION '''

accuracy_list = evaluateCLFs(results, PCA_test_df, test_labels, want_plots, activity_name)



    # results.append( {
    #                     'model_name':       model_name_str,
    #                     'optimalizer':      method,

    #                     'classifier':       clf,
    #                     'best_score':       best_score,
    #                     'best_params':      best_params,      
    #                     'train_test_delta': train_test_delta, # Higher value -> more overfitted
    #                     'mean_test_score':  mean_test_score,
    #                     'std_test_score':   std_test_score
    #                   })

# clf_dict = {}

# for i, classifier in enumerate(classifiers):
#     clf_dict[optimization_list[i]] = classifier

# print(clf_dict)

# # clf_dict brukes ikke, clf_names er tom

# for name, clf, clf_name in zip(optimization_list, classifiers, clf_dict):
    
#     accuracy_score = evaluateCLF(name, clf, PCA_test_df, test_labels, want_plots, activity_name, clf_name)
#     accuracy_list.append(np.round(accuracy_score, 3))
    

if want_plots:
    
    ''' FEATURE IMPORTANCE '''
    
    PCA_table_plot(train_data_scaled, 5)   

    ''' 2D PLOTS OF PCA '''

    biplot(total_data_scaled, window_labels, label_mapping)
    
    plotBoundaryConditions(PCA_train_df, train_labels, label_mapping, results, accuracy_list)

    # plot_SVM_boundaries(PCA_train_df, train_labels, label_mapping,
    #                      classifiers, optimization_methods, best_clf_params, accuracy_list)

    ''' KNN PLOT '''
    # if(ML_model.upper() == "KNN"):
    #     if (PCA_components == 2):
    #         plotKNNboundries(PCA_train_df, clf, mapped_labels)
    
    plt.show()


''' Real time streaming '''
# import pickle
# from muse_api_main import ble_conn, Muse_Utils, ble_TESTING
# from bleak import BleakScanner, BleakClient
# import asyncio

''' REAL TEST '''

'''
# File path to testing file
# test_data_df = pd.read_csv(test_path+"test_data.csv")

test_feature, windowLabel = extractAllFeatures('testFiles/06-03-2025 123529', sets_labels, window_length_seconds*Fs, False, 800)

# Scale incoming data
test_data_scaled = scaleFeatures(test_feature)

# PCA transform test_data_scaled using already fitted PCA
PCA_test_df = pd.DataFrame(PCA_final.transform(test_data_scaled))

# Use best CLF
guess = clf1.predict(PCA_test_df)

guess = guess.sort()
'''



''' Pickling classifier '''

import pickle
halving_classifier = results[0]['classifier']

if (pickle_files):
    with open(output_path + "classifier.pkl", "wb") as CLF_File: 
        pickle.dump(halving_classifier, CLF_File) 
    


    final_model = None
    # for clf, method in zip(classifiers, optimization_list):
    #     if method == "HalvingGridSearchCV":
    #         final_model = clf.best_estimator_ if hasattr(clf, "best_estimator_") else clf
    #         break

    # if final_model is None:
    #     print("Fant ikke modell med HalvingGridSearchCV – bruker første som fallback.")
    #     final_model = classifiers[0]

    # Dobbeltsjekk før lagring
    print("Modell som lagres:", final_model)
    print("predict_proba tilgjengelig:", hasattr(final_model, "predict_proba"))

    # Lagre
    with open(output_path + "classifier.pkl", "wb") as CLF_File: 
        pickle.dump(final_model, CLF_File)


    with open(output_path + "PCA.pkl", "wb" ) as PCA_File:
        pickle.dump(PCA_final, PCA_File)

    with open(output_path + "scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

# from testonfile import run_inference_on_file
#     ### Testing trained model on unseen files
# test_file_path = "testFiles"
# test_files = os.listdir(test_file_path)


# for filename in test_files:
#     if filename.endswith(".csv"):
#         continue  # hopper over .csv-filer
#     file_to_test = os.path.join(test_file_path, filename)
#     file_to_test_no_ext = file_to_test.replace(".txt", "")
#     print("_______________________________________________________________________________")
#     print(f"Testing file {file_to_test}")
#     run_inference_on_file(file_to_test_no_ext, fs = fs, window_length_sec = window_length_seconds, norm_accel=False, run=True)
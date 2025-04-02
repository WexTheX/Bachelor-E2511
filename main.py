''' IMPORTS '''
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.inspection import DecisionBoundaryDisplay
from skopt.space import Real, Categorical, Integer
from sklearn import svm, metrics, dummy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV

# Local imports
# from FOLDER import FILE as F
from extractFeatures import extractAllFeatures
from machineLearning import splitData, scaleFeatures, setNComponents, makeSVMClassifier, makeRFClassifier
from plotting import plotWelch, biplot
from SignalProcessing import ExtractIMU_Features as IMU_F
from SignalProcessing import get_Freq_Domain_features_of_signal as freq
from Preprocessing.preprocessing import fillSets
from sklearn.ensemble import RandomForestClassifier



''' GLOBAL VARIABLES '''

want_feature_extraction = 0
separate_types = 1
want_plots = 0
ML_models = ["SVM", "RF"]
ML_models = 0
accuracy_list = []

''' DATASET VARIABLES '''

variance_explained = 0.4
randomness = 181
window_length_seconds = 15
split_value = 0.75
Fs = 800
variables = ["Timestamp","Gyr.X","Gyr.Y","Gyr.Z","Axl.X","Axl.Y","Axl.Z","Mag.X","Mag.Y","Mag.Z","Temp"]

''' BASE ESTIMATORS '''

base_params =  {'class_weight': 'balanced',
                'random_state': randomness}

SVM_base = svm.SVC(**base_params)
RF_base = RandomForestClassifier(**base_params)

''' HYPER PARAMETER VARIABLES '''

num_folds = 3

hyperparams_SVM = {
    "C": [0.001, 0.01, 0.1, 1],
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "gamma": [1],
    "coef0": [0, 0.5, 1],
    "degree": [2, 3]
}

# hyperparams_SVM_space = {
#     # "C": Categorical(1),  # Continuous log-scale for C
#     # "kernel": Categorical(["linear", "poly", "rbf", "sigmoid"]),  # Discrete choices
#     # "gamma": Categorical(1),  # Log-uniform scale for gamma
#     # "coef0": Categorical(1),
#     # "degree": Categorical(2)
# }

# hyperparams_SVM = {
#     "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
#     "kernel": ["linear", "poly", "rbf", "sigmoid"],
#     "gamma": [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2],
#     "coef0": [0, 0.5, 1],
#     "degree": [2, 3, 4, 5]
# }

hyperparams_SVM_space = {
    "C": Real(1e-3, 1e3, prior="log-uniform"),  # Continuous log-scale for C
    "kernel": Categorical(["linear", "poly", "rbf", "sigmoid"]),  # Discrete choices
    "gamma": Real(1e-3, 1e2, prior="log-uniform"),  # Log-uniform scale for gamma
    "coef0": Real(0, 1),
    "degree": Integer(2, 5)
}

hyperparams_RF = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    # 'max_depth': [10, 20, 30, None],  # Maximum depth of each tree
    # 'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    # 'min_samples_leaf': [1, 2, 4],  # Minimum samples required in a leaf node
    # 'max_features': ['sqrt', 'log2'],  # Number of features considered for splitting
    # 'bootstrap': [True, False],  # Whether to use bootstrapped samples
    'criterion': ['gini', 'entropy']  # Splitting criteria
}

''' USER INPUTS '''

# answer_FE = input("Do you want feature extraction? (Y | N) (Default N)")
# if(answer_FE == "Y"):
#     want_feature_extraction = True

# if(answer_FE):
#     answer_ST = input("Do you want to separate by type (TIG and MIG vs only welding)? (Y | N) (Default N)")
# if(answer_ST == "Y"):
#     separate_types = True



# answer_ML = input(f"Choose ML model (Default SVM): {ML_models}.")
# if(answer_ML = "RF"):
#     ML_model = ML_models[0]

# answer_plot = input("Do you want plots? (Y | N) (Default N)")
# if(answer_plot == "Y"):
#     want_plots = True


''' LOAD DATASET '''

# Different folder for separated and not separated
if(separate_types):
    path = "Preprocessing/DatafilesSeparated" 
    output_path = "OutputFiles/Separated/"
    test_path = "testFiles/"
else:
    path = "Preprocessing/Datafiles"
    output_path = "OutputFiles/"
    test_path = "testFiles/"

path_names = os.listdir(path)
activity_name = [name.upper() for name in path_names]

sets, sets_labels = fillSets(path, path_names, activity_name, separate_types)
# print(f"Content of sets: \n {sets}")
# print(f"Content of sets_labels: \n {sets_labels}")


''' FEATURE EXTRACTION '''

if (want_feature_extraction):
    # Create dataframe "feature_df" containing all features deemed relevant from the raw sensor data
    # One row in feature_df is all features from one window
    feature_df, window_labels = extractAllFeatures(sets, sets_labels, window_length_seconds*Fs, False, 800)
    feature_df.to_csv(output_path+"feature_df.csv", index=False)
    with open(output_path+"window_labels.txt", "w") as fp:
        for item in window_labels:
            fp.write("%s\n" % item)

if "feature_df" not in globals():
    window_labels = []
    feature_df = pd.read_csv(output_path+"feature_df.csv")
    f = open(output_path+"window_labels.txt", "r")
    data = f.read()
    window_labels = data.split("\n")
    f.close()
    window_labels.pop()


''' SPLITTING TEST/TRAIN + SCALING'''
train_data, test_data, train_labels, test_labels = splitData(feature_df, window_labels, randomness, split_value)

train_data_scaled = scaleFeatures(train_data)
test_data_scaled = scaleFeatures(test_data)


''' Principal Component Analysis (PCA)'''
PCA_components = setNComponents(train_data_scaled, variance_explained=variance_explained)
PCA_final = PCA(n_components = PCA_components)

PCA_train_df = pd.DataFrame(PCA_final.fit_transform(train_data_scaled))
PCA_test_df = pd.DataFrame(PCA_final.transform(test_data_scaled))


''' HYPERPARAMETER OPTIMIZATION AND CLASSIFIER '''

# comment out here + in clf_dict to remove 

optimization_methods = ['ManualGridSearchCV', 'RandomizedSearchCV', 'GridSearchCV', 'HalvingGridSearchCV']

clf1, clf1_best_params = makeSVMClassifier(optimization_methods[0], SVM_base, num_folds, hyperparams_SVM_space, hyperparams_SVM, want_plots, PCA_train_df, train_data, train_labels, variance_explained, separate_types)
clf2, clf2_best_params = makeSVMClassifier(optimization_methods[1], SVM_base, num_folds, hyperparams_SVM_space, hyperparams_SVM, want_plots, PCA_train_df, train_data, train_labels, variance_explained, separate_types)
clf3, clf3_best_params = makeSVMClassifier(optimization_methods[2], SVM_base, num_folds, hyperparams_SVM_space, hyperparams_SVM, want_plots, PCA_train_df, train_data, train_labels, variance_explained, separate_types)
clf4, clf4_best_params = makeSVMClassifier(optimization_methods[3], SVM_base, num_folds, hyperparams_SVM_space, hyperparams_SVM, want_plots, PCA_train_df, train_data, train_labels, variance_explained, separate_types)

models = (clf1, clf2, clf3, clf4)
titles = (
    clf1_best_params,
    clf2_best_params,
    clf3_best_params,
    clf4_best_params )

clf_dict = {
    optimization_methods[0]: clf1,
    optimization_methods[1]: clf2,
    optimization_methods[2]: clf3,
    optimization_methods[3]: clf4
    }

''' EVALUATION '''

for name, clf in clf_dict.items():
    
    print(f"Evaluating {name}: ")

    test_predict = clf.predict(PCA_test_df)

    accuracy_score = metrics.balanced_accuracy_score(test_labels, test_predict)
    precision_score = metrics.precision_score(test_labels, test_predict, average="weighted")
    recall_score = metrics.recall_score(test_labels, test_predict, average="weighted")
    f1_score = metrics.f1_score(test_labels, test_predict, average="weighted")
    
    accuracy_list.append(np.round(accuracy_score, 3))

    print(f"Accuracy: \t {accuracy_score}")
    print(f"Precision: \t {precision_score}")
    print(f"Recall: \t {recall_score}")
    print(f"f1: \t\t {f1_score}")

dummy_clf = dummy.DummyClassifier(strategy="most_frequent")
dummy_clf.fit(PCA_train_df, train_labels)
dummy_score = dummy_clf.score(PCA_test_df, test_labels)

print("Baseline Accuracy (Dummy Classifier):", dummy_score)
print(accuracy_list)

if(want_plots):

    # # Set-up 2x2 grid for plotting.
    # fig, sub = plt.subplots(2, 2)
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)

    # X0, X1 = PCA_train_df[0], PCA_train_df[1]

    # titles = (
    # "SVC with linear kernel",
    # "LinearSVC (linear kernel)",
    # "SVC with RBF kernel",
    # "SVC with polynomial (degree 3) kernel",
    # )

    # label_mapping = {'IDLE': (0.0, 0.0, 0.0)  , 
    #                    'GRINDBIG': (1.0, 0.0, 0.0),'GRINDMED': (0.6, 0.0, 0.0), 'GRINDSMALL': (0.3, 0.0, 0.0),
    #                    'SANDSIM': (0.0, 1.0, 0.0), 
    #                    'WELDALTIG': (0.0, 0.0, 1.0), 'WELDSTMAG': (0.0, 0.0, 0.6), 'WELDSTTIG': (0.0, 0.0, 0.3)}
    
    # y_labels = np.array(train_labels)
    # mappedLabels = np.array([label_mapping[label] for label in train_labels])
    # # print(mappedLabels)

    # for clf, title, ax in zip(models, titles, sub.flatten()):
    #     disp = DecisionBoundaryDisplay.from_estimator(
    #         clf,
    #         PCA_train_df,
    #         response_method="predict",
    #         cmap=plt.cm.coolwarm,
    #         alpha=0.8,
    #         ax=ax,
    #         xlabel='PC1',
    #         ylabel='PC2',
    #     )
    #     ax.scatter(X0, X1, c=mappedLabels, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    #     ax.set_xticks(())
    #     ax.set_yticks(())
    #     ax.set_title(title)

    # plt.show()



    ''' PCA CHECK '''
    # print("Printing PCA compontents for entire set")
    total_data_scaled = pd.DataFrame(scaleFeatures(feature_df))
    PCA_plot = PCA(n_components = 5)
    print(f"Total amount of features: {len(total_data_scaled.columns)}")

    # Displays tables for how much each feature is contributing to PC1-5
    for i in range(len(total_data_scaled.columns) // 34):
        PCA_total_columns_part = total_data_scaled.columns[i*34:(i*34+34)]
        # print(f"List of columns: {PCA_total_columns_part}")

        PCA_total_part = total_data_scaled[PCA_total_columns_part]
        PCA_total_df = pd.DataFrame(PCA_plot.fit_transform(PCA_total_part))
        
        # biplot(PCA_total_df, window_labels, PCA_plot, 5, separate_types, clf1)


    # Plot 2D plot of PC's regardless of how many components are in the model
    PCA_plot = PCA(n_components = 2)
    PCA_plot_df = pd.DataFrame(PCA_plot.fit_transform(total_data_scaled))
    
    biplot(PCA_plot_df, window_labels, PCA_plot, 2, separate_types, models, optimization_methods, titles, accuracy_list)

    conf_matrix = metrics.confusion_matrix(test_labels, test_predict, labels=activity_name)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='coolwarm', xticklabels=activity_name, yticklabels=activity_name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title('Confusion matrix')
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
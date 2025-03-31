''' IMPORTS '''
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from skopt.space import Real, Categorical, Integer
from sklearn import svm, metrics, dummy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV

# Local imports
# from FOLDER import FILE as F
from extractFeatures import extractAllFeatures
from machineLearning import splitData, scaleFeatures, setNComponents, makeSVMClassifier
from plotting import plotWelch, biplot
from SignalProcessing import ExtractIMU_Features as IMU_F
from SignalProcessing import get_Freq_Domain_features_of_signal as freq
from Preprocessing.preprocessing import fillSets


''' GLOBAL VARIABLES '''

want_feature_extraction = 0
separate_types = 1
want_plots = 1
ML_models = ["SVM"]
ML_models = 0
method = 'ManualGridSearch'

''' DATASET VARIABLES '''

variance_explained = 0.9
randomness = 123334
window_length_seconds = 30
split_value = 0.75
Fs = 800
variables = ["Timestamp","Gyr.X","Gyr.Y","Gyr.Z","Axl.X","Axl.Y","Axl.Z","Mag.X","Mag.Y","Mag.Z","Temp"]

''' HYPER PARAMETER VARIABLES '''

num_folds = 3
C_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
kernel_types = ['linear', 'poly',
                 'rbf', 'sigmoid']
gamma_list = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]
coef0_list = [0, 0.5, 1]
deg_list = [2, 3, 4, 5]
hyperparams_list = []

hyperparams_dict = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "gamma": [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2],
    "coef0": [0, 0.5, 1],
    "degree": [2, 3, 4, 5]
}

hyperparams_space = {
    "C": Real(1e-3, 1e3, prior="log-uniform"),  # Continuous log-scale for C
    "kernel": Categorical(["linear", "poly", "rbf", "sigmoid"]),  # Discrete choices
    "gamma": Real(1e-3, 1e2, prior="log-uniform"),  # Log-uniform scale for gamma
    "coef0": Real(0, 1),
    "degree": Integer(2, 5)
}

''' USER INPUTS '''

# answer_FE = input("Do you want feature extraction? (Y | N)")
# if(answer_FE == "Y"):
#     want_feature_extraction = True

# if(answer_FE):
#     answer_ST = input("Do you want to separate by type (TIG and MIG vs only welding)? (Y | N)")
#     if(answer_ST == "Y"):
#         separate_types = True


# answer_ML = input(f"Choose ML model: {ML_models}")

# answer_plot = input("Do you want plots? (Y | N)")
# if(answer_plot == "Y"):
#     want_lots = True


''' LOAD DATASET '''

# Spesify path for input and output of files
if(separate_types):
    path = "Preprocessing/DatafilesSeparated" 
else:
    path = "Preprocessing/Datafiles"

output_path = "OutputFiles/"   
path_names = os.listdir(path)
activity_name = [name.upper() for name in path_names]

sets, sets_labels = fillSets(path, path_names, activity_name, separate_types)
# print(f"Content of sets: \n {sets}")
# print(f"Content of sets_labels: \n {sets_labels}")



''' FEATURE EXTRACTION '''

if (want_feature_extraction):
    # Create dataframe "feature_df" containing all features deemed relevant from the raw sensor data
    # One row in feature_df is all features from one window
    feature_df, window_labels = extractAllFeatures(sets, sets_labels, window_length_seconds*Fs, False, 800, path)
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

# Debugging prints
# print(f"Content of feature dataframe: \n {feature_df}")
# print(f"Content of window label list: \n {window_labels}")

''' PCA CHECK '''
if(want_plots):
    print("Printing PCA compontents for entire set")
    total_data_scaled = pd.DataFrame(scaleFeatures(feature_df))
    PCA_plot = PCA(n_components = 5)
    print(f"Total amount of features: {len(total_data_scaled.columns)}")

    for i in range(len(total_data_scaled.columns) // 34):
        PCA_total_columns_part = total_data_scaled.columns[i*34:(i*34+34)]
        # print(f"List of columns: {PCA_total_columns_part}")
        PCA_total_part = total_data_scaled[PCA_total_columns_part]
        PCA_total_df = pd.DataFrame(PCA_plot.fit_transform(PCA_total_part))
        
        biplot(PCA_total_df, window_labels, PCA_plot, 5, separate_types)
    

    PCA_plot3D = PCA(n_components = 3)
    PCA_plot_df = pd.DataFrame(PCA_plot3D.fit_transform(total_data_scaled))
    biplot(PCA_plot_df, window_labels, PCA_plot3D, 3, separate_types)
    plt.show()
    print("Done. \n")


''' SPLITTING TEST/TRAIN '''
train_data, test_data, train_labels, test_labels = splitData(feature_df, window_labels, randomness, split_value)
# print(f"Content of training data: \n {train_data}")
# print(f"Content of training labels: \n {train_labels}")
# print(f"Content of testing data: \n {test_data}")
# print(f"Content of testing labels: \n {test_labels}")

''' SCALING '''
train_data_scaled = scaleFeatures(train_data)
test_data_scaled = scaleFeatures(test_data)

# print(f"Content of training data scaled: \n {train_data_scaled}")
# print(f"Content of testing data scaled: \n {test_data_scaled}")

''' Principal Component Analysis (PCA)'''
PCA_components = setNComponents(train_data_scaled, variance_explained=0.90)
PCA_final = PCA(n_components = PCA_components)

PCA_train_df = pd.DataFrame(PCA_final.fit_transform(train_data_scaled))
PCA_test_df = pd.DataFrame(PCA_final.transform(test_data_scaled))

start_time = time.time()

''' HYPERPARAMETER OPTIMIZATION '''
# best_hyperparams = makeSVMClassifier(method, num_folds, hyperparams_dict, C_list, kernel_types, gamma_list, coef0_list, deg_list,
#                                     want_plots, train_data, train_labels, variance_explained)


''' HYPERPARAMETER OPTIMIZATION AND CLASSIFIER '''

clf1 = makeSVMClassifier('ManualGridSearch', num_folds, hyperparams_dict, want_plots, PCA_train_df, train_data, train_labels, variance_explained)
clf2 = makeSVMClassifier('GridSearchCV', num_folds, hyperparams_dict, want_plots, PCA_train_df, train_data, train_labels, variance_explained)
clf3 = makeSVMClassifier('HalvingGridSearchCV', num_folds, hyperparams_dict, want_plots, PCA_train_df, train_data, train_labels, variance_explained)
clf4 = makeSVMClassifier('BayesSearchCV', num_folds, hyperparams_dict, want_plots, PCA_train_df, train_data, train_labels, variance_explained)

''' EVALUATION '''

print("ManualGridSearch: ")

test_predict = clf1.predict(PCA_test_df)

accuracy_score = metrics.accuracy_score(test_labels, test_predict)
precision_score = metrics.precision_score(test_labels, test_predict, average=None)
recall_score = metrics.recall_score(test_labels, test_predict, average=None)
f1_score = metrics.f1_score(test_labels, test_predict, average=None)

print(f"Accuracy: \t {accuracy_score}")
print(f"Precision: \t {precision_score}")
print(f"Recall: \t {recall_score}")
print(f"f1: \t {f1_score}")


print("GridSearchCV: ")

test_predict = clf2.predict(PCA_test_df)

accuracy_score = metrics.accuracy_score(test_labels, test_predict)
precision_score = metrics.precision_score(test_labels, test_predict, average=None)
recall_score = metrics.recall_score(test_labels, test_predict, average=None)
f1_score = metrics.f1_score(test_labels, test_predict, average=None)

print(f"Accuracy: \t {accuracy_score}")
print(f"Precision: \t {precision_score}")
print(f"Recall: \t {recall_score}")
print(f"f1: \t {f1_score}")


print("HalvingGridSearchCV: ")

test_predict = clf3.predict(PCA_test_df)

accuracy_score = metrics.accuracy_score(test_labels, test_predict)
precision_score = metrics.precision_score(test_labels, test_predict, average=None)
recall_score = metrics.recall_score(test_labels, test_predict, average=None)
f1_score = metrics.f1_score(test_labels, test_predict, average=None)

print(f"Accuracy: \t {accuracy_score}")
print(f"Precision: \t {precision_score}")
print(f"Recall: \t {recall_score}")
print(f"f1: \t {f1_score}")

print("BayesSearchCV: ")

test_predict = clf4.predict(PCA_test_df)

accuracy_score = metrics.accuracy_score(test_labels, test_predict)
precision_score = metrics.precision_score(test_labels, test_predict, average=None)
recall_score = metrics.recall_score(test_labels, test_predict, average=None)
f1_score = metrics.f1_score(test_labels, test_predict, average=None)

print(f"Accuracy: \t {accuracy_score}")
print(f"Precision: \t {precision_score}")
print(f"Recall: \t {recall_score}")
print(f"f1: \t {f1_score}")


dummy_clf = dummy.DummyClassifier(strategy="most_frequent")
dummy_clf.fit(PCA_train_df, train_labels)
dummy_score = dummy_clf.score(PCA_test_df, test_labels)

print("Baseline Accuracy (Dummy Classifier):", dummy_score)

# Confusion matrix
conf_matrix = metrics.confusion_matrix(test_labels, test_predict, labels=activity_name)

if(want_plots):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='coolwarm', xticklabels=activity_name, yticklabels=activity_name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title('Confusion matrix')
    # print(conf_matrix)

if (want_plots):
    # for i in range(1, len(variables)):
    #     plotWelch(sets[0], variables[i], Fs, False)
    #     plotWelch(sets[0], variables[i], Fs, True)
    #     plt.xlabel('Frequency (Hz)')
    #     plt.ylabel('Power Spectral Density')
    #     plt.title('Welch PSD, %s' % variables[i])
    #     plt.grid()
    #     plt.figure()

    # biplot(dfPCAtrain, train_labels, PCATest, PCA_components, separate_types)
    plt.show()

''' IMPORTS '''
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score

# Local imports
# from FOLDER import FILE as F
from extractFeatures import extractAllFeatures
from machineLearning import splitData, scaleFeatures, setHyperparams
from plotting import plotWelch, biplot
from SignalProcessing import ExtractIMU_Features as IMU_F
from SignalProcessing import get_Freq_Domain_features_of_signal as freq
from Preprocessing.preprocessing import fillSets

''' GLOBAL VARIABLES '''
# Input variables
want_feature_extraction = 0
seperate_types = 0
want_plots = 0
ML_models = ["SVM"]
ML_models = 0


# Dataset parameters
randomness = 0
window_length_seconds = 30
split_value = 0.75
Fs = 800
variables = ["Timestamp","Gyr.X","Gyr.Y","Gyr.Z","Axl.X","Axl.Y","Axl.Z","Mag.X","Mag.Y","Mag.Z","Temp"]

# Hyper parameter variables
num_folds = 5
C_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
kernel_types = ['linear', 'poly',
                 'rbf', 'sigmoid']
gamma_list = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]
coef0_list = [0, 0.5, 1]
deg_list = [2, 3, 4, 5]
hyper_param_list = []

# Initialize arrays for evaluating score = (mean - std)
accuracy_array = np.zeros( (num_folds, len(C_list), len(kernel_types), len(gamma_list), len(coef0_list), len(deg_list)) )
mean_accuracy_array = np.zeros( (len(C_list), len(kernel_types), len(gamma_list), len(coef0_list), len(deg_list)) )
std_accuracy_array = np.zeros( (len(C_list), len(kernel_types), len(gamma_list), len(coef0_list), len(deg_list)) )


''' USER INPUTS '''

# answer_FE = input("Do you want feature extraction? (Y | N)")
# if(answer_FE == "Y"):
#     want_feature_extraction = True

# if(answer_FE):
#     answer_ST = input("Do you want to seperate by type (TIG and MIG vs only welding)? (Y | N)")
#     if(answer_ST == "Y"):
#         seperate_types = True


# answer_ML = input(f"Choose ML model: {ML_models}")

# answer_plot = input("Do you want plots? (Y | N)")
# if(answer_plot == "Y"):
#     want_lots = True


''' LOAD DATASET '''
# Spesify path for input and output of files
if(seperate_types):
    path = "Preprocessing/DatafilesSeperated" 
else:
    path = "Preprocessing/Datafiles"

output_path = "OutputFiles/"   
path_names = os.listdir(path)
activity_name = [name.upper() for name in path_names]

sets, sets_labels = fillSets(path, path_names, activity_name, seperate_types)
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

window_labels_numeric = LabelEncoder().fit_transform(window_labels)

# Debugging prints
# print(f"Content of feature dataframe: \n {feature_df}")
# print(f"Content of window label list: \n {window_labels}")

''' PCA CHECK '''
if(want_plots):
    total_data_scaled = scaleFeatures(feature_df)
    PCA_components = setHyperparams(total_data_scaled, variance_explained=variance_explained)

    PCA_check = PCA(n_components = PCA_components)    
    PCA_train_df = pd.DataFrame(PCA_check.fit_transform(total_data_scaled))
    
    biplot(PCA_train_df, window_labels, PCA_check, PCA_components)

# Start timer to evaluate performance
start_time = time.time()

''' SPLITTING TEST/TRAIN '''
train_data, test_data, train_labels, test_labels = splitData(feature_df, window_labels, randomness, split_value)
# print(f"Content of training data: \n {train_data}")
# print(f"Content of training labels: \n {train_labels}")

# print(f"Content of testing data: \n {test_data}")
# print(f"Content of testing labels: \n {test_labels}")

train_labels_numeric = LabelEncoder().fit_transform(train_labels)


''' K-FOLD SPLIT '''
skf = StratifiedKFold(n_splits = num_folds)

''' HYPERPARAMETER OPTIMIZATION '''
for i, (train_index, test_index) in enumerate(skf.split(train_data, train_labels)):

    print(f"PCA fitting on fold {i}")

    # Debug prints
    # print(f"  Train: index={train_index}")
    # print(f"  Test:  index={test_index}")
    # print(f"Train labels: {train_labels}")

    kfold_train_labels = [train_labels[j] for j in train_index]
    kfold_test_labels = [train_labels[j] for j in test_index]

    # print(kfold_testLabels)
    
    kfold_train_data = train_data.iloc[train_index]
    kfold_validation_data = train_data.iloc[test_index]
    #print(f"Dette er stuffet: {kfold_TrainData}")
    # Scale training and validation seperately
    kfold_train_data_scaled = scaleFeatures(kfold_train_data)
    kfold_validation_data_scaled = scaleFeatures(kfold_validation_data)
    
    PCA_components = setHyperparams(kfold_train_data_scaled, variance_explained=0.90)
    
    PCA_fold = PCA(n_components = PCA_components)
    
    kfold_PCA_train_df = pd.DataFrame(PCA_fold.fit_transform(kfold_train_data_scaled))
    kfold_PCA_validation_df = pd.DataFrame(PCA_fold.transform(kfold_validation_data_scaled))

    if (want_plots):
        print(f"Plotting PCA plots for fold {i}")
        biplot(kfold_PCA_train_df, kfold_train_labels, PCA_fold, PCA_components)


    for j, C_value in enumerate(C_list):

        for k, kernel in enumerate(kernel_types):
                
                # print("Work in progress")

                if kernel == 'linear':
                    l, m, n = 0, 0, 0
                    
                    clf = svm.SVC(C=C_value, kernel=kernel)
                    clf.fit(kfold_PCA_train_df, kfold_train_labels)
                    test_predict = clf.predict(kfold_PCA_validation_df)
                    # accuracy_array[i, j, k, :, :, :] = 0
                    accuracy_array[i, j, k, l, m, n] = metrics.accuracy_score(kfold_test_labels, test_predict)
                    
                    # Only append for fold 0
                    if i == 0:
                        hyper_param_list.append((C_value, kernel))

                elif kernel == 'poly':
                     
                    for l, gamma_value in enumerate(gamma_list):
                        for m, coef0_value in enumerate(coef0_list):
                            for n, deg_value in enumerate(deg_list):

                                # print(f"Working on {j} {k} {l} {m} {n}")
                                clf = svm.SVC(C=C_value, kernel=kernel, gamma=gamma_value, coef0=coef0_value, degree=deg_value)
                                clf.fit(kfold_PCA_train_df, kfold_train_labels)
                                test_predict = clf.predict(kfold_PCA_validation_df)
                                accuracy_array[i, j, k, l, m, n] = metrics.accuracy_score(kfold_test_labels, test_predict)
                                
                                if i == 0:
                                    hyper_param_list.append((C_value, kernel, gamma_value, coef0_value, deg_value))

                elif kernel == 'sigmoid': 
                    
                    for l, gamma_value in enumerate(gamma_list):
                        for m, coef0_value in enumerate(coef0_list):    
                            n = 0
                            clf = svm.SVC(C=C_value, kernel=kernel, gamma=gamma_value, coef0=coef0_value)
                            clf.fit(kfold_PCA_train_df, kfold_train_labels)
                            test_predict = clf.predict(kfold_PCA_validation_df)
                            accuracy_array[i, j, k, l, m, n] = metrics.accuracy_score(kfold_test_labels, test_predict)   

                            if i == 0:
                                hyper_param_list.append((C_value, kernel, gamma_value, coef0_value))

                elif kernel == 'rbf':

                    for l, gamma_value in enumerate(gamma_list):
                        m, n = 0, 0
                        clf = svm.SVC(C=C_value, kernel=kernel, gamma=gamma_value)
                        clf.fit(kfold_PCA_train_df, kfold_train_labels)
                        test_predict = clf.predict(kfold_PCA_validation_df)
                        accuracy_array[i, j, k, l, m, n] = metrics.accuracy_score(kfold_test_labels, test_predict)  

                        if i == 0:
                            hyper_param_list.append((C_value, kernel, gamma_value))
print("\n")

# Exhaustive grid search: calculate which hyperparams gives highest score = max|mean - std|
for j in range(len(C_list)):
    for k in range(len(kernel_types)):
        for l in range(len(gamma_list)):
            for m in range(len(coef0_list)):
                for n in range(len(deg_list)):
                    mean_accuracy_array[j, k, l, m, n] = accuracy_array[:, j, k, l, m, n].mean()
                    std_accuracy_array[j, k, l, m, n] = accuracy_array[:, j, k, l, m, n].std()


score_array = mean_accuracy_array - std_accuracy_array
# print(score_array)
print(f"Dimensions of score_array: {score_array.shape}")

'''
Tror problemet ligger i at score_array er en cube med mange 0 verdier
0 for alle verdier som ikke settes i loopen (degree 2,3,4 for 'linear' fr.eks)
Potensielt bevis: score_array uten alle 0 verdier er like lang som hyper_param_list
Videre gir multi_dim_index og hyper_param_list(best_param_test) like parametre
'''
score_array_test = score_array.flatten()
score_array_test = [i for i in score_array_test if i != 0]
# print(score_array_1D)
print(f"Size of score_array_test: {len(score_array_test)}")

# print(f"Indices: {max_index}")
# print(f"Max value = {max_value} at index {max_index}")

# Find location and value of highest score
best_param = np.argmax(score_array)
print(f"Index of best parameter, converted to 2D array (cube): {best_param}")
best_param_test = np.argmax(score_array_test) 
print(f"Index of best parameter, converted to 2D array (not cube): {best_param}")

max_value = np.max(score_array)
multi_dim_index = np.unravel_index(best_param, score_array.shape)
print(f"Max value: {max_value} at {multi_dim_index}")
print("\n")


# print(score_array.shape)
# print(best_param)
# print(len(multi_dim_index))

# Unknown error !!!!!!!!!!!!!!! ^ must be investigated

end_time = time.time()  # End timer
elapsed_time = end_time - start_time

print(f"All combinations of hyper params: {len(hyper_param_list)}")
print(f"Created and evaluated {len(hyper_param_list) * num_folds} instances of SVM classifiers in seconds: {elapsed_time:.6f}")
print(f"Highest score found (mean - std): {max_value}")
print(f"Best combination of hyperparameters (C, kernel, gamma, coef0, degree): {hyper_param_list[best_param_test]}")
print(f"\n")

# TESTING 
C_value = C_list[multi_dim_index[0]]
kernel = kernel_types[multi_dim_index[1]]
gamma_value = gamma_list[multi_dim_index[2]]
coef0_value = coef0_list[multi_dim_index[3]]
deg_value = deg_list[multi_dim_index[4]]


#return multi_dim_index or array of [C_value ... etc]









''' SCALING '''
train_data_scaled = scaleFeatures(train_data)
test_data_scaled = scaleFeatures(test_data)

# print(f"Content of training data scaled: \n {train_data_scaled}")
# print(f"Content of testing data scaled: \n {test_data_scaled}")


''' HYPERPARAMETER OPTIMIZATION '''



''' Principal Component Analysis (PCA)'''
PCA_components = setHyperparams(train_data_scaled, variance_explained=0.90)
PCA_final = PCA(n_components = PCA_components)

PCA_train_df = pd.DataFrame(PCA_final.fit_transform(train_data_scaled))
PCA_validation_df = pd.DataFrame(PCA_final.transform(test_data_scaled))


''' CLASSIFIER '''
clf = svm.SVC(C=C_value, kernel=kernel, gamma=gamma_value, coef0=coef0_value, degree=deg_value)
clf.fit(PCA_train_df, train_labels)


''' EVALUATION '''
# Total accuracy
test_predict = clf.predict(PCA_validation_df)
accuracy_score = metrics.accuracy_score(test_labels, test_predict)
print(f"Accuracy on unseen test data: {accuracy_score}")

# Confusion matrix
conf_matrix = metrics.confusion_matrix(test_labels, test_predict, labels=activity_name)

if(want_plots):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='coolwarm', xticklabels=activity_name, yticklabels=activity_name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title('Confusion matrix')
    # print(conf_matrix)


''' ALT: PIPELINE (WIP) '''
# clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
# clf.fit(train_data,train_labels)

# print(clf.named_steps['linearsvc'].coef_)

# Plotting part:
# Plot FFT:
# variables = ["Axl.X", "Axl.Y", "Axl.Z"]

# plotWelch(sets, variables, Fs)

# testWelch(sets[0], variables[0], Fs)

if (want_plots):
    # for i in range(1, len(variables)):
    #     plotWelch(sets[0], variables[i], Fs, False)
    #     plotWelch(sets[0], variables[i], Fs, True)
    #     plt.xlabel('Frequency (Hz)')
    #     plt.ylabel('Power Spectral Density')
    #     plt.title('Welch PSD, %s' % variables[i])
    #     plt.grid()
    #     plt.figure()

    # biplot(dfPCAtrain, train_labels, PCATest, PCA_components)

    plt.show()

import time
start_time = time.time()  # Start timer


''' IMPORTS '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
import seaborn as sns


# Local imports
# from FOLDER import FILE as F
from extractFeatures import Extract_All_Features
from machineLearning import splitData, scaleFeatures, setHyperparams
from plotting import plotWelch, biplot
from SignalProcessing import ExtractIMU_Features as IMU_F
from SignalProcessing import get_Freq_Domain_features_of_signal as freq
from Preprocessing.preprocessing import fillSets


''' GLOBAL VARIABLES '''
# Spesify path for input and output of files
path = "Preprocessing/Datafiles"
output_path = "OutputFiles/"
path_names = os.listdir(path)
activity_name = [name[:4].upper() for name in path_names]

# Input variables
want_feature_extraction = 0
want_plots = 1
ML_models = ["SVM"]
ML_models = 0

# Dataset parameters
randomness = 0
window_length_seconds = 30
Fs = 800
variables = ["Timestamp","Gyr.X","Gyr.Y","Gyr.Z","Axl.X","Axl.Y","Axl.Z","Mag.X","Mag.Y","Mag.Z","Temp"]
split_value = 0.75

# Hyper parameter variables
hyper_param_list = []
num_folds = 5
C_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
kernel_types = ['linear', 'poly', 'rbf', 'sigmoid']
gamma_list = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]
coef0_list = [0, 0.5, 1]
deg_list = [2, 3, 4, 5]

accuracy_array = np.zeros( (num_folds, len(C_list), len(kernel_types), len(gamma_list), len(coef0_list), len(deg_list)) )
mean_accuracy_array = np.zeros( (len(C_list), len(kernel_types), len(gamma_list), len(coef0_list), len(deg_list)) )
std_accuracy_array = np.zeros( (len(C_list), len(kernel_types), len(gamma_list), len(coef0_list), len(deg_list)) )


''' LOAD DATASET '''
sets, sets_labels = fillSets(path, path_names, activity_name)
# print(f"Content of sets: \n {sets}")
# print(f"Content of sets_labels: \n {sets_labels}")


''' USER INPUTS '''
# answer_FE = input("Do you want feature extraction? (Y | N)")
# if(answer_FE == "Y"):
#     want_feature_extraction = True

# answer_ML = input(f"Choose ML model: {ML_models}")

# answer_plot = input("Do you want plots? (Y | N)")
# if(answer_plot == "Y"):
#     want_lots = True


''' FEATURE EXTRACTION '''

if(want_feature_extraction):
    feature_df, window_labels = Extract_All_Features(sets, sets_labels, window_length_seconds*Fs, False, 800, path)
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

# print(f"Content of feature dataframe: \n {feature_df}")
# print(f"Content of window label list: \n {window_labels}")

window_labels_numeric = LabelEncoder().fit_transform(window_labels)

''' PCA CHECK '''
if(want_plots):
    total_data_scaled = scaleFeatures(feature_df)
    PCA_components = setHyperparams(total_data_scaled, variance_explained=0.90)

    PCA_check = PCA(n_components = PCA_components)    
    kfold_dfPCA_train = pd.DataFrame(PCA_check.fit_transform(total_data_scaled))
    biplot(kfold_dfPCA_train, window_labels, PCA_check, PCA_components)


''' SPLITTING TEST/TRAIN '''
train_data, test_data, train_labels, test_labels = splitData(feature_df, window_labels, randomness)
# print(f"Content of training data: \n {train_data}")
# print(f"Content of training labels: \n {train_labels}")

# print(f"Content of testing data: \n {test_data}")
# print(f"Content of testing labels: \n {test_labels}")

train_labels_numeric = LabelEncoder().fit_transform(train_labels)


''' K-FOLD SPLIT '''
skf = StratifiedKFold(n_splits = num_folds)

''' HYPERPARAMETER OPTIMIZATION '''
for i, (train_index, test_index) in enumerate(skf.split(train_data, train_labels)):

    index = 0

    print(f"PCA fitting on fold {i}")
    # print(f"  Train: index={train_index}")
    # print(f"  Test:  index={test_index}")
    # print(f"Train labels: {train_labels}")

    kfold_trainLabels = [train_labels[j] for j in train_index]
    kfold_testLabels = [train_labels[j] for j in test_index]

    # print(kfold_testLabels)
    
    kfold_TrainData = train_data.iloc[train_index]
    kfold_ValidationData = train_data.iloc[test_index]
    #print(f"Dette er stuffet: {kfold_TrainData}")
    # Scale training and validation seperately
    kfold_TrainDataScaled = scaleFeatures(kfold_TrainData)
    kfold_ValidationDataScaled = scaleFeatures(kfold_ValidationData)

    # 
    
    PCA_components = setHyperparams(kfold_TrainDataScaled, variance_explained=0.90)
    
    PCATest = PCA(n_components = PCA_components)
    
    kfold_dfPCA_train = pd.DataFrame(PCATest.fit_transform(kfold_TrainDataScaled))
    kfold_dfPCA_validation = pd.DataFrame(PCATest.transform(kfold_ValidationDataScaled))

    if (want_plots):
        print("Plotting plots for each fold")
        biplot(kfold_dfPCA_train, kfold_trainLabels, PCATest, PCA_components)

    
    
    # print(f"Testing accurracy with different C and kernels: ")


    for j, C_value in enumerate(C_list):

        for k, kernel in enumerate(kernel_types):
                
                # print("Work in progress")

                if kernel == 'linear':
                    l, m, n = 0, 0, 0
                    
                    clf = svm.SVC(C=C_value, kernel=kernel)
                    clf.fit(kfold_dfPCA_train, kfold_trainLabels)
                    testPredict = clf.predict(kfold_dfPCA_validation)
                    # accuracy_array[i, j, k, :, :, :] = 0
                    accuracy_array[i, j, k, l, m, n] = metrics.accuracy_score(kfold_testLabels, testPredict)
                    
                    # Only append for fold 0
                    if i == 0:
                        hyper_param_list.append((C_value, kernel))

                elif kernel == 'poly':
                     
                    for l, gamma_value in enumerate(gamma_list):
                        for m, coef0_value in enumerate(coef0_list):
                            for n, deg_value in enumerate(deg_list):

                                # print(f"Working on {j} {k} {l} {m} {n}")
                                clf = svm.SVC(C=C_value, kernel=kernel, gamma=gamma_value, coef0=coef0_value, degree=deg_value)
                                clf.fit(kfold_dfPCA_train, kfold_trainLabels)
                                testPredict = clf.predict(kfold_dfPCA_validation)
                                accuracy_array[i, j, k, l, m, n] = metrics.accuracy_score(kfold_testLabels, testPredict)
                                
                                if i == 0:
                                    hyper_param_list.append((C_value, kernel, gamma_value, coef0_value, deg_value))

                elif kernel == 'sigmoid': 
                    
                    for l, gamma_value in enumerate(gamma_list):
                        for m, coef0_value in enumerate(coef0_list):    
                            n = 0
                            clf = svm.SVC(C=C_value, kernel=kernel, gamma=gamma_value, coef0=coef0_value)
                            clf.fit(kfold_dfPCA_train, kfold_trainLabels)
                            testPredict = clf.predict(kfold_dfPCA_validation)
                            accuracy_array[i, j, k, l, m, n] = metrics.accuracy_score(kfold_testLabels, testPredict)   

                            if i == 0:
                                hyper_param_list.append((C_value, kernel, gamma_value, coef0_value))

                elif kernel == 'rbf':

                    for l, gamma_value in enumerate(gamma_list):
                        m, n = 0, 0
                        clf = svm.SVC(C=C_value, kernel=kernel, gamma=gamma_value)
                        clf.fit(kfold_dfPCA_train, kfold_trainLabels)
                        testPredict = clf.predict(kfold_dfPCA_validation)
                        accuracy_array[i, j, k, l, m, n] = metrics.accuracy_score(kfold_testLabels, testPredict)  

                        if i == 0:
                            hyper_param_list.append((C_value, kernel, gamma_value))

            

            # # Make new SVM instance with specific hyperparams
            # clf = svm.SVC(C = C_value, kernel = kernel, probability = True)
            # clf.fit(kfold_dfPCA_train, kfold_trainLabels)
            # testPredict = clf.predict(kfold_dfPCA_validation)
            
            # Add accuracy for params to a 3D array

# accuracy_array[i, j, k, l, m] = metrics.accuracy_score(kfold_testLabels, testPredict)
            
# print(f"C = {C_value}, Kernel = {k} \t\t ", metrics.accuracy_score(kfold_testLabels, testPredict))
# print(hyper_param_list)
print(f"\n")
print(f"Length of Hyper param list: {len(hyper_param_list)}")
print(f"Shape of accuracy array: {accuracy_array.shape}")
print(f"Memory size of accuracy array in bytes: {accuracy_array.nbytes}")

end_time = time.time()  # End timer
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")

# print(clf.get_params())

for j in range(len(C_list)):
    for k in range(len(kernel_types)):
        for l in range(len(gamma_list)):
            for m in range(len(coef0_list)):
                for n in range(len(deg_list)):
                    mean_accuracy_array[j, k, l, m, n] = accuracy_array[:, j, k, l, m, n].mean()
                    std_accuracy_array[j, k, l, m, n] = accuracy_array[:, j, k, l, m, n].std()

score_array = mean_accuracy_array - std_accuracy_array
# print(score_array)

# max_value = np.max(score_array)
# max_index = np.where(score_array == max_value)

# print(f"Indices: {max_index}")
# print(f"Max value = {max_value} at index {max_index}")

best_param = np.argmax(score_array)
max_value = np.max(score_array)
multi_dim_index = np.unravel_index(best_param, score_array.shape)

print(f"\n")
print(f"Highest score of mean - std: {max_value}")

# TESTING 
C_value = C_list[multi_dim_index[0]]
kernel = kernel_types[multi_dim_index[1]]
gamma_value = gamma_list[multi_dim_index[2]]
coef0_value = coef0_list[multi_dim_index[3]]
deg_value = deg_list[multi_dim_index[4]]



# print(f"Mean accuracy array across all folds: {mean_accuracy_array}")
# print(f"STD accuracy array across all folds: {std_accuracy_array}")
# print(f"Array of all scores: {score_array}")

print(f"Best combination of hyperparameters through exhaustive grid search {hyper_param_list[best_param]}")

# print(f"Hyper param list: {hyper_param_list}")

# argMaxValue = np.argmax(accuracyArray / num_folds)

# print(f"Average value of accuracy: {accuracyArray / num_folds}")
# print(f"Index of max value: {argMaxValue}")
# print(f"Hyper param list: {hyper_param_list[argMaxValue]}")
# clf = svm.SVC(C = C, kernel = j)


#  dfPCAtest = pd.DataFrame(PCATest.transform(test_data_scaled))   


''' SCALING '''
train_data_scaled = scaleFeatures(train_data)
test_data_scaled = scaleFeatures(test_data)

# print(f"Content of training data scaled: \n {train_data_scaled}")
# print(f"Content of testing data scaled: \n {test_data_scaled}")


''' HYPERPARAMETER OPTIMIZATION '''



''' Principal Component Analysis (PCA)'''
PCA_components = setHyperparams(train_data_scaled, variance_explained=0.90)
PCATest = PCA(n_components = PCA_components)

dfPCA_train = pd.DataFrame(PCATest.fit_transform(train_data_scaled))
dfPCA_validation = pd.DataFrame(PCATest.transform(test_data_scaled))


# Decide nr of PC's
# PCA_components = setHyperparams(variance_explained = 0.95)

# PCATest = PCA(n_components = PCA_components)

# dfPCAtrain = pd.DataFrame(PCATest.fit_transform(train_data_scaled))
# dfPCAtest = pd.DataFrame(PCATest.transform(test_data_scaled))

# def biplot(score, coeff, train_labels, labels=None):

#     loadings = PCATest.components_.T * np.sqrt(PCATest.explained_variance_)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(loadings, annot=True, cmap='coolwarm', xticklabels=['PC1', 'PC2'], yticklabels=PCATest.feature_names_in_)
#     plt.title('Feature Importance in Principal Components')

#     xs = score[0]
#     ys = score[1]
#     plt.figure(figsize=(10, 8))

#     label_mapping = {'GRIN': 0  , 'IDLE': 1, 'WELD': 2}
#     y_labels = np.array(train_labels)
#     mappedLabels = np.array([label_mapping[label] for label in train_labels])

#     plt.scatter(xs, ys, c=mappedLabels, cmap='viridis')


#     for i in range(len(coeff)):

#         plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)

#         plt.text(coeff[i, 0] * 1.2, coeff[i, 1] * 1.2, labels[i], color='g')

#     plt.xlabel("PC1")
#     plt.ylabel("PC2")
#     plt.title("Biplot")
#     plt.show()



# Uncomment to see 2D / 3D plot of PCA


# print(f"New training data, fitted and PCA-transformed with {PCA_components} components: \n {dfPCAtrain}")


''' CLASSIFIER '''
clf = svm.SVC(C=C_value, kernel=kernel, gamma=gamma_value, coef0=coef0_value, degree=deg_value)
clf.fit(dfPCA_train, train_labels)

# scores = cross_val_score(clf, train_data, train_labels_numeric, cv=5)
# print(scores)
# print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


''' EVALUATION '''
# Total accuracy
testPredict = clf.predict(dfPCA_validation)
accuracy_score = metrics.accuracy_score(test_labels, testPredict)
print(f"Accuracy on unseen test data: {accuracy_score}")

# Confusion matrix
conf_matrix = metrics.confusion_matrix(test_labels, testPredict, labels=activity_name)
if(want_plots):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='coolwarm', xticklabels=activity_name, yticklabels=activity_name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title('Confusion matrix')
# print(conf_matrix)


# Testing different kernels and C values for the classifier
# Ideally should cross validate across different training sets

# C = 0
# kernel_types = ['linear', 'poly', 'rbf', 'sigmoid']

# print(f"Testing accurracy with different C and kernels: ")

# for i in range(-3, 3):
#     C = 10**(i)

#     for j in kernel_types:
#         clf = svm.SVC(C = C, kernel = j)
#         clf.fit(dfPCAtrain, train_labels)

#         testPredict = clf.predict(dfPCAtest)
        # print(f"C = {C}, Kernel = {j} \t\t ", metrics.accuracy_score(test_labels, testPredict))

# testPredict = clf.predict(dfPCAtest)
# print("Accuracy of assigning correct label using SVM: ", metrics.accuracy_score(test_labels, testPredict))


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

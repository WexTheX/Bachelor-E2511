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
from machineLearning import splitData, scaleFeatures
from plotting import plotWelch, biplot
from SignalProcessing import ExtractIMU_Features as IMU_F
from SignalProcessing import get_Freq_Domain_features_of_signal as freq
from Preprocessing.preprocessing import fillSets

''' GLOBAL VARIABLES '''
# Spesify path for input and output of files
path = "Preprocessing/Datafiles"
outputPath = "OutputFiles/"
pathNames = os.listdir(path)
activityName = [name[:4].upper() for name in pathNames]

# Input variables
want_feature_extraction = 0
want_plots = 1

# Dataset parameters
randomness = 12533
windowLengthSeconds = 13
Fs = 800
variables = ["Timestamp","Gyr.X","Gyr.Y","Gyr.Z","Axl.X","Axl.Y","Axl.Z","Mag.X","Mag.Y","Mag.Z","Temp"]

# Hyper parameter variables
hyper_param_list = []
num_folds = 5

C_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
kernelTypes = ['linear', 'poly', 'rbf', 'sigmoid']
gamma_list = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]
coef0_list = [0, 0.5, 1]
deg_list = [2, 3, 4, 5]

accuracy_array = np.zeros( (num_folds, len(C_list), len(kernelTypes), len(gamma_list), len(coef0_list), len(deg_list)) )
mean_accuracy_array = np.zeros( (len(C_list), len(kernelTypes), len(gamma_list), len(coef0_list), len(deg_list)) )
std_accuracy_array = np.zeros( (len(C_list), len(kernelTypes), len(gamma_list), len(coef0_list), len(deg_list)) )


''' LOAD DATASET '''
sets, setsLabel = fillSets(path, pathNames, activityName)
# print(f"Content of sets: \n {sets}")
# print(f"Content of setsLabel: \n {setsLabel}")


''' USER INPUTS '''
# answer_FE = input("Do you want feature extraction? (Y | N)")
# if(answer_FE == "Y"):
#     want_feature_extraction = True

# answer_plot = input("Do you want plots? (Y | N)")
# if(answer_plot == "Y"):
#     want_lots = True


''' FEATURE EXTRACTION '''

if(want_feature_extraction):
    feature_df, windowLabels = Extract_All_Features(sets, setsLabel, windowLengthSeconds*Fs, False, 800, path)
    feature_df.to_csv(outputPath+"feature_df.csv", index=False)
    with open(outputPath+"windowLabels.txt", "w") as fp:
        for item in windowLabels:
            fp.write("%s\n" % item)

if "feature_df" not in globals():
    windowLabels = []
    feature_df = pd.read_csv(outputPath+"feature_df.csv")
    f = open(outputPath+"windowLabels.txt", "r")
    data = f.read()
    windowLabels = data.split("\n")
    f.close()
    windowLabels.pop()

# print(f"Content of feature dataframe: \n {feature_df}")
# print(f"Content of window label list: \n {windowLabels}")

windowLabelsNumeric = LabelEncoder().fit_transform(windowLabels)


''' SPLITTING TEST/TRAIN '''
trainData, testData, trainLabels, testLabels = splitData(feature_df, windowLabels, randomness)
# print(f"Content of training data: \n {trainData}")
# print(f"Content of training labels: \n {trainLabels}")

# print(f"Content of testing data: \n {testData}")
# print(f"Content of testing labels: \n {testLabels}")

trainLabelsNumeric = LabelEncoder().fit_transform(trainLabels)


''' K-FOLD SPLIT '''
skf = StratifiedKFold(n_splits = num_folds)

def setHyperparams(kfold_TrainDataScaled, varianceExplained):

    C = np.cov(kfold_TrainDataScaled, rowvar=False) # 140x140 Co-variance matrix
    eigenvalues, eigenvectors = np.linalg.eig(C)

    eigSum = 0
    for i in range(len(eigenvalues)):
        
        eigSum += eigenvalues[i]
        totalVariance = eigSum / eigenvalues.sum()

        if totalVariance >= varianceExplained:
            n_components = i + 1
            print(f"Variance explained by {i + 1} PCA components: {eigSum / eigenvalues.sum()}")
            break

    # n_components = 2

    return n_components

''' HYPERPARAMETER OPTIMIZATION '''
for i, (train_index, test_index) in enumerate(skf.split(trainData, trainLabels)):

    index = 0

    print(f"Fold {i}:")
    # print(f"  Train: index={train_index}")
    # print(f"  Test:  index={test_index}")
    # print(f"Train labels: {trainLabels}")

    kfold_trainLabels = [trainLabels[j] for j in train_index]
    kfold_testLabels = [trainLabels[j] for j in test_index]

    # print(kfold_testLabels)
    
    kfold_TrainData = trainData.iloc[train_index]
    kfold_ValidationData = trainData.iloc[test_index]
    #print(f"Dette er stuffet: {kfold_TrainData}")
    # Scale training and validation seperately
    kfold_TrainDataScaled = scaleFeatures(kfold_TrainData)
    kfold_ValidationDataScaled = scaleFeatures(kfold_ValidationData)

    # 

    PCA_components = setHyperparams(kfold_TrainDataScaled, varianceExplained=0.90)
    PCATest = PCA(n_components = PCA_components)

    kfold_dfPCA_train = pd.DataFrame(PCATest.fit_transform(kfold_TrainDataScaled))
    kfold_dfPCA_validation = pd.DataFrame(PCATest.transform(kfold_ValidationDataScaled))

    # biplot(kfold_dfPCA_train, kfold_trainLabels, PCATest, PCA_components)

    
    
    # print(f"Testing accurracy with different C and kernels: ")


    for j, C_value in enumerate(C_list):

        for k, kernel in enumerate(kernelTypes):
                
                # print("Work in progress")

                if kernel == 'linear':
                    
                    clf = svm.SVC(C=C_value, kernel=kernel)
                    clf.fit(kfold_dfPCA_train, kfold_trainLabels)
                    testPredict = clf.predict(kfold_dfPCA_validation)
                    # accuracy_array[i, j, k, :, :, :] = 0
                    accuracy_array[i, j, k, 0, 0, 0] = metrics.accuracy_score(kfold_testLabels, testPredict)
                    
                    # Only append for fold 0
                    if i == 0:
                        hyper_param_list.append((C_value, kernel))

                elif kernel == 'poly':
                     
                    for l, gamma_value in enumerate(gamma_list):
                        for m, coef0_value in enumerate(coef0_list):
                            for n, deg_value in enumerate(deg_list):

                                clf = svm.SVC(C=C_value, kernel=kernel, gamma=gamma_value, coef0=coef0_value, degree=deg_value)
                                clf.fit(kfold_dfPCA_train, kfold_trainLabels)
                                testPredict = clf.predict(kfold_dfPCA_validation)
                                accuracy_array[i, j, k, l, m, n] = metrics.accuracy_score(kfold_testLabels, testPredict)
                                
                                if i == 0:
                                    hyper_param_list.append((C_value, kernel, gamma_value, coef0_value, deg_value))

                elif kernel == 'sigmoid': 

                    for l, gamma_value in enumerate(gamma_list):
                        for m, coef0_value in enumerate(coef0_list):    

                            clf = svm.SVC(C=C_value, kernel=kernel, gamma=gamma_value, coef0=coef0_value)
                            clf.fit(kfold_dfPCA_train, kfold_trainLabels)
                            testPredict = clf.predict(kfold_dfPCA_validation)
                            accuracy_array[i, j, k, l, m, 0] = metrics.accuracy_score(kfold_testLabels, testPredict)   

                            if i == 0:
                                hyper_param_list.append((C_value, kernel, gamma_value, coef0_value))

                elif kernel == 'rbf':

                    for l, gamma_value in enumerate(gamma_list):
                        
                        clf = svm.SVC(C=C_value, kernel=kernel, gamma=gamma_value)
                        clf.fit(kfold_dfPCA_train, kfold_trainLabels)
                        testPredict = clf.predict(kfold_dfPCA_validation)
                        accuracy_array[i, j, k, l, m, 0] = metrics.accuracy_score(kfold_testLabels, testPredict)  

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
print(f"Length of Hyper param list: {len(hyper_param_list)}")

print(f"Shape of accuracy array: {accuracy_array.shape}")
print(f"Memory size of accuracy array: {accuracy_array.nbytes}")

end_time = time.time()  # End timer
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")

# print(clf.get_params())

for j in range(len(C_list)):
    for k in range(len(kernelTypes)):
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

print(f"Highest score: {max_value} found at index: {multi_dim_index}")

# TESTING 
C_value = C_list[multi_dim_index[0]]
kernel = kernelTypes[multi_dim_index[1]]
gamma_value = gamma_list[multi_dim_index[2]]
coef0_value = coef0_list[multi_dim_index[3]]
deg_value = deg_list[multi_dim_index[4]]

# final scaling
trainDataScaled = scaleFeatures(trainData)
testDataScaled = scaleFeatures(testData)

PCA_components = setHyperparams(trainDataScaled, varianceExplained=0.90)
PCATest = PCA(n_components = PCA_components)

dfPCA_train = pd.DataFrame(PCATest.fit_transform(trainDataScaled))
dfPCA_validation = pd.DataFrame(PCATest.transform(testDataScaled))


clf = svm.SVC(C=C_value, kernel=kernel, gamma=gamma_value, coef0=coef0_value, degree=deg_value)
clf.fit(dfPCA_train, trainLabels)
testPredict = clf.predict(dfPCA_validation)
accuracy_score = metrics.accuracy_score(testLabels, testPredict)

print(f"Accuracy on unseen test data: {accuracy_score}")


# print(f"Mean accuracy array across all folds: {mean_accuracy_array}")
# print(f"STD accuracy array across all folds: {std_accuracy_array}")
# print(f"Array of all scores: {score_array}")

# print(f"Best combination of hyperparameters through exhaustive grid search {hyper_param_list[best_param]}")

# print(f"Hyper param list: {hyper_param_list}")

# argMaxValue = np.argmax(accuracyArray / num_folds)

# print(f"Average value of accuracy: {accuracyArray / num_folds}")
# print(f"Index of max value: {argMaxValue}")
# print(f"Hyper param list: {hyper_param_list[argMaxValue]}")
# clf = svm.SVC(C = C, kernel = j)


#  dfPCAtest = pd.DataFrame(PCATest.transform(testDataScaled))   
   
''' SCALING '''

# trainDataScaled = scaleFeatures(trainData)
# testDataScaled = scaleFeatures(testData)

# print(f"Content of training data scaled: \n {trainDataScaled}")
# print(f"Content of testing data scaled: \n {testDataScaled}")


''' Principal Component Analysis (PCA)'''


# Decide nr of PC's
# PCA_components = setHyperparams(varianceExplained = 0.95)

# PCATest = PCA(n_components = PCA_components)

# dfPCAtrain = pd.DataFrame(PCATest.fit_transform(trainDataScaled))
# dfPCAtest = pd.DataFrame(PCATest.transform(testDataScaled))

# def biplot(score, coeff, trainLabels, labels=None):

#     loadings = PCATest.components_.T * np.sqrt(PCATest.explained_variance_)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(loadings, annot=True, cmap='coolwarm', xticklabels=['PC1', 'PC2'], yticklabels=PCATest.feature_names_in_)
#     plt.title('Feature Importance in Principal Components')

#     xs = score[0]
#     ys = score[1]
#     plt.figure(figsize=(10, 8))

#     label_mapping = {'GRIN': 0  , 'IDLE': 1, 'WELD': 2}
#     y_labels = np.array(trainLabels)
#     mappedLabels = np.array([label_mapping[label] for label in trainLabels])

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
# clf = svm.SVC(kernel='rbf')
# clf.fit(dfPCAtrain, trainLabels)
# clf = svm.SVC(kernel='linear', C=1, random_state=42)

# scores = cross_val_score(clf, trainData, trainLabelsNumeric, cv=5)
# print(scores)
# print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

''' EVALUATION '''

# Testing different kernels and C values for the classifier
# Ideally should cross validate across different training sets

# C = 0
# kernelTypes = ['linear', 'poly', 'rbf', 'sigmoid']

# print(f"Testing accurracy with different C and kernels: ")

# for i in range(-3, 3):
#     C = 10**(i)

#     for j in kernelTypes:
#         clf = svm.SVC(C = C, kernel = j)
#         clf.fit(dfPCAtrain, trainLabels)

#         testPredict = clf.predict(dfPCAtest)
        # print(f"C = {C}, Kernel = {j} \t\t ", metrics.accuracy_score(testLabels, testPredict))

# testPredict = clf.predict(dfPCAtest)
# print("Accuracy of assigning correct label using SVM: ", metrics.accuracy_score(testLabels, testPredict))


''' ALT: PIPELINE (WIP) '''
# clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
# clf.fit(trainData,trainLabels)

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
    # plt.show()

    # biplot(dfPCAtrain, trainLabels, PCATest, PCA_components)


    plt.show()

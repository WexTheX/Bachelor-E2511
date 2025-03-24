# Main file
# Global imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
import seaborn as sns


# Local imports
# from FOLDER import FILE as F
from extractFeatures import Extract_All_Features
from machineLearning import splitData, scaleFeatures
from plotting import plotWelch, biplot_3D
from SignalProcessing import ExtractIMU_Features as IMU_F
from SignalProcessing import get_Freq_Domain_features_of_signal as freq
from Preprocessing.preprocessing import fillSets

# Spesify path for input and output of files
path = "Preprocessing/Datafiles"
outputPath = "OutputFiles/"

wantFeatureExtraction = 0
wantPlots = False
windowLengthSeconds = 13
Fs = 800
randomness = 19
variables = ["Timestamp","Gyr.X","Gyr.Y","Gyr.Z","Axl.X","Axl.Y","Axl.Z","Mag.X","Mag.Y","Mag.Z","Temp"]

# Load sets and label for those sets from given path
''' LOAD DATASET '''
sets, setsLabel = fillSets(path)
# print(f"Content of sets: \n {sets}")
# print(f"Content of setsLabel: \n {setsLabel}")

# User inputs, expand or ditch?
# answerFE = input("Do you want feature extraction? (Y | N)")
# if(answerFE == "Y"):
#     wantFeatureExtraction = True

# answerPlot = input("Do you want plots? (Y | N)")
# if(answerPlot == "Y"):
#     wantPlots = True


''' FEATURE EXTRACTION '''

if(wantFeatureExtraction):
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

windowLabelsNumeric = LabelEncoder().fit_transform(windowLabels)


# print(f"Content of feature dataframe: \n {feature_df}")
# print(f"Content of window label list: \n {windowLabels}")


''' SPLITTING '''
trainData, testData, trainLabels, testLabels = splitData(feature_df, windowLabels, randomness)
# print(f"Content of training data: \n {trainData}")
print(f"Content of training labels: \n {trainLabels}")
# print(f"Content of testing data: \n {testData}")
print(f"Content of testing labels: \n {testLabels}")

trainLabelsNumeric = LabelEncoder().fit_transform(trainLabels)

'''K-fold split'''

kf = KFold(n_splits=3)
kf.get_n_splits(trainData)


for i, (train_index, test_index) in enumerate(kf.split(trainData)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")



''' SCALING '''

trainDataScaled = scaleFeatures(trainData)
testDataScaled = scaleFeatures(testData)

# print(f"Content of training data scaled: \n {trainDataScaled}")
# print(f"Content of testing data scaled: \n {testDataScaled}")


''' Principal Component Analysis (PCA)'''

def setHyperparams(varianceExplained):

    C = np.cov(trainDataScaled, rowvar=False) # 140x140 Co-variance matrix
    eigenvalues, eigenvectors = np.linalg.eig(C)

    eigSum = 0
    for i in range(len(trainDataScaled)):
        
        eigSum += eigenvalues[i]
        totalVariance = eigSum / eigenvalues.sum()

        if totalVariance >= varianceExplained:
            n_components = i + 1
            print(f"Variance explained by {i + 1} PCA components: {eigSum / eigenvalues.sum()}")
            break
    n_components = 3
    return n_components

# Decide nr of PC's
PCA_components = setHyperparams(varianceExplained = 0.95)

PCATest = PCA(n_components = PCA_components)

dfPCAtrain = pd.DataFrame(PCATest.fit_transform(trainDataScaled))
dfPCAtest = pd.DataFrame(PCATest.transform(testDataScaled))

def biplot(score, coeff, trainLabels, labels=None):

    loadings = PCATest.components_.T * np.sqrt(PCATest.explained_variance_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(loadings, annot=True, cmap='coolwarm', xticklabels=['PC1', 'PC2'], yticklabels=PCATest.feature_names_in_)
    plt.title('Feature Importance in Principal Components')

    xs = score[0]
    ys = score[1]
    plt.figure(figsize=(10, 8))

    label_mapping = {'GRIN': 0  , 'IDLE': 1, 'WELD': 2}
    y_labels = np.array(trainLabels)
    mappedLabels = np.array([label_mapping[label] for label in trainLabels])

    plt.scatter(xs, ys, c=mappedLabels, cmap='viridis')


    for i in range(len(coeff)):

        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)

        plt.text(coeff[i, 0] * 1.2, coeff[i, 1] * 1.2, labels[i], color='g')

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Biplot")
    plt.show()



# Uncomment to see 2D / 3D plot of PCA
biplot_3D(dfPCAtrain, trainLabels, PCATest)

print(f"New training data, fitted and PCA-transformed with {PCA_components} components: \n {dfPCAtrain}")


''' CLASSIFIER '''
# clf = svm.SVC(kernel='rbf')
# clf.fit(dfPCAtrain, trainLabels)
clf = svm.SVC(kernel='linear', C=1, random_state=42)

scores = cross_val_score(clf, trainData, trainLabelsNumeric, cv=5)
print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

''' EVALUATION '''

# Testing different kernels and C values for the classifier
# Ideally should cross validate across different training sets

C = 0
kernelTypes = ['linear', 'poly', 'rbf', 'sigmoid']

print(f"Testing accurracy with different C and kernels: ")

for i in range(-3, 3):
    C = 10**(i)

    for j in kernelTypes:
        clf = svm.SVC(C = C, kernel = j)
        clf.fit(dfPCAtrain, trainLabels)

        testPredict = clf.predict(dfPCAtest)
        print(f"C = {C}, Kernel = {j} \t\t ", metrics.accuracy_score(testLabels, testPredict))

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

if (wantPlots):
    for i in range(1, len(variables)):
        plotWelch(sets[0], variables[i], Fs, False)
        plotWelch(sets[0], variables[i], Fs, True)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.title('Welch PSD, %s' % variables[i])
        plt.grid()
        plt.figure()
    plt.show()
# Main file
# Global imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.datasets import load_iris

# Local imports
# from FOLDER import FILE as F
from extractFeatures import Extract_All_Features
from machineLearning import splitData, scaleFeatures
from plotting import plotWelch
from SignalProcessing import ExtractIMU_Features as IMU_F
from SignalProcessing import get_Freq_Domain_features_of_signal as freq
from Preprocessing.preprocessing import fillSets

# Spesify path for input and output of files
path = "Preprocessing/Datafiles"
outputPath = "OutputFiles/"

wantFeatureExtraction = 0
wantPlots = False
windowLengthSeconds = 20
Fs = 800
randomness = 192
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
# TODO: Går det an å sjekke ka som allerede e extracta og kun hente ut det som ikkje e gjort fra før?
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

# print(f"Content of feature dataframe: \n {feature_df}")
# print(f"Content of window label list: \n {windowLabels}")


''' SPLITTING '''
trainData, testData, trainLabels, testLabels = splitData(feature_df, windowLabels, randomness)
# print(f"Content of training data: \n {trainData}")
# print(f"Content of training labels: \n {trainLabels}")
# print(f"Content of testing data: \n {testData}")
# print(f"Content of testing labels: \n {testLabels}")

''' SCALING '''

trainDataScaled = scaleFeatures(trainData)
testDataScaled = scaleFeatures(testData)

# print(f"Content of training data scaled: \n {trainDataScaled}")
# print(f"Content of testing data scaled: \n {testDataScaled}")


''' Principal Component Analysis (PCA)'''

## temp location
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
    
    return n_components

PCA_components = setHyperparams(varianceExplained = 0.95)

PCATest = PCA(n_components = 3)

# The training data is fitted using PCA. The training data is then transformed from 140 -> "n_components" dimensions.
# The test data is then transformed to the same space as the test data.
dfPCAtrain = pd.DataFrame(PCATest.fit_transform(trainDataScaled))
dfPCAtest = pd.DataFrame(PCATest.transform(testDataScaled))

##############################

loadings = PCATest.components_.T * np.sqrt(PCATest.explained_variance_)
# print("Loadings:")
# print(loadings)

# plt.figure(figsize=(10, 8))
# sns.heatmap(loadings, annot=True, cmap='coolwarm', xticklabels=['PC1', 'PC2'], yticklabels=PCATest.feature_names_in_)
# plt.title('Feature Importance in Principal Components')
# plt.show()

##############################

def biplot(score, coeff, trainLabels, labels=None):

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

def biplot_3D(score, coeff, trainLabels, labels=None):
    xs = score[0]
    ys = score[1]
    zs = score[2]

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    label_mapping = {'GRIN': 0, 'IDLE': 1, 'WELD': 2}
    y_labels = np.array(trainLabels)
    mappedLabels = np.array([label_mapping[label] for label in trainLabels])

    # Create 3D scatter plot
    sc = ax.scatter(xs, ys, zs, c=mappedLabels, cmap='inferno')

    # Draw arrows for the components
    for i in range(len(coeff)):
        ax.quiver(0, 0, 0, coeff[i, 0], coeff[i, 1], coeff[i, 2], color='r', alpha=0.5)

        ax.text(coeff[i, 0] * 1.2, coeff[i, 1] * 1.2, coeff[i, 2] * 1.2, labels[i], color='g')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title("3D Biplot")

    plt.show()

biplot_3D(dfPCAtrain, PCATest.components_.T, trainLabels, PCATest.feature_names_in_)


##############################

print(f"Training data fitted and PCA-transformed with {PCA_components} components: \n {dfPCAtrain}")

''' CLASSIFIER '''
clf = svm.SVC(kernel='poly')
clf.fit(dfPCAtrain, trainLabels)

testPredict = clf.predict(dfPCAtest)

''' EVALUATION '''
print("Accuracy of SVM: ", metrics.accuracy_score(testLabels, testPredict))


''' ALT: PIPELINE (WIP) '''
# clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
# clf.fit(trainData,trainLabels)

# print(clf.named_steps['linearsvc'].coef_)

# Plotting part:
# Plot FFT:
# variables = ["Axl.X", "Axl.Y", "Axl.Z"]

# plotWelch(sets, variables, Fs)

# testWelch(sets[0], variables[0], Fs)

if(wantPlots):
    for i in range(1, len(variables)):
        plotWelch(sets[0], variables[i], Fs, False)
        plotWelch(sets[0], variables[i], Fs, True)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.title('Welch PSD, %s' % variables[i])
        plt.grid()
        plt.figure()
    plt.show()
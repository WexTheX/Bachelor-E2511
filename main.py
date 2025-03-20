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

wantFeatureExtraction = False
wantPlots = False
windowLengthSeconds = 10
Fs = 800
randomness = 18
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
testDataScaled = scaleFeatures(testData)
trainDataScaled = scaleFeatures(trainData)


# print(f"Content of training data scaled: \n {trainDataScaled}")
# print(f"Content of testing data scaled: \n {testDataScaled}")


''' Principal Component Analysis (PCA)'''

## temp location
def setHyperparams(varianceExplained):

    C = np.cov(trainDataScaled, rowvar=False) # 140x140
    eigenvalues, eigenvectors = np.linalg.eig(C)

    # eigenvalues = np.array(eigenvalues)
    eigSum = 0
    # TESTING
    for i in range(len(trainDataScaled)):
        
        eigSum += eigenvalues[i]
        totalVariance = eigSum / eigenvalues.sum()

        if totalVariance >= varianceExplained:
            n_components = i + 1
            print(f"Variance explained by {i + 1} PCA components: {eigSum / eigenvalues.sum()}")
            break
    
    return n_components

PCA_components = setHyperparams(varianceExplained = 0.95)

PCATest = PCA(n_components = PCA_components)

# The training data is fitted using PCA. The training data is then transformed from 140 -> "n_components" dimensions.
# The test data is then transformed to the same space as the test data.
dfPCAtrain = pd.DataFrame(PCATest.fit_transform(trainDataScaled))
dfPCAtest = pd.DataFrame(PCATest.transform(testDataScaled))

print(f"Training data fitted and PCA-transformed with {PCA_components} components: \n {dfPCAtrain}")

''' CLASSIFIER '''
clf = svm.SVC(kernel='linear')
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
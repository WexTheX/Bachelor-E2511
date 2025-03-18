# Main file
# Global imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


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
windowLengthSeconds = 0
Fs = 800
randomness = 25
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

'''
# GRIN_features = pd.read_csv("OutputFiles/GRIN_features.csv")

# mean_accel_x = GRIN_features["mean_accel_X"]    
# print(mean_accel_x)

# 140 elements per row
# row n = accel xyz TD, accel xyz FD, gyro xyz TD, gyro xyz FD, mag xyz TD, mag xyz FD, temp TD, temp FD from window n
'''

''' SPLITTING '''
trainData, testData, trainLabels, testLabels = splitData(feature_df, windowLabels, randomness)
print(f"Content of training data: \n {trainData}")
print(f"Content of training labels: \n {trainLabels}")
print(f"Content of testing data: \n {testData}")
print(f"Content of testing labels: \n {testLabels}")
    
''' SCALING '''
trainDataScaled = scaleFeatures(trainData)
testDataScaled = scaleFeatures(testData)
print(f"Content of training data scaled: \n {trainDataScaled}")
print(f"Content of testing data scaled: \n {testDataScaled}")

''' CLASSIFIER '''
clf = svm.SVC(kernel='linear')
clf.fit(trainDataScaled, trainLabels)

testPredict = clf.predict(testDataScaled)

''' EVALUATION '''
print("Accuracy:", metrics.accuracy_score(testLabels, testPredict))


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

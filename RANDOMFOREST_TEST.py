from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from machineLearning import splitData, scaleFeatures
from extractFeatures import Extract_All_Features
from Preprocessing.preprocessing import fillSets
import os
from sklearn import svm, metrics
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
path = "Preprocessing/Datafiles"
outputPath = "OutputFiles/"
windowLengthSeconds = 13
Fs = 800
randomness = 1222


pathNames = os.listdir(path)
activityName = [name[:4].upper() for name in pathNames]


wantFeatureExtraction = False
sets, setsLabel = fillSets(path, pathNames, activityName)


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
trainData, testData, trainLabels, testLabels = splitData(feature_df, windowLabels, randomness)

print(trainData)


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

    # n_components = 3
    return n_components


trainDataScaled = scaleFeatures(trainData)
testDataScaled = scaleFeatures(testData)




PCA_components = setHyperparams(trainDataScaled, varianceExplained=0.90)
PCATest = PCA(n_components = PCA_components)


RF_dfPCA_train = pd.DataFrame(PCATest.fit_transform(trainDataScaled))
RF_dfPCA_test = pd.DataFrame(PCATest.transform(testDataScaled))


for i in range(1, 100):
    clf = RandomForestClassifier(max_depth=3, random_state=i)
    clf.fit(RF_dfPCA_train, trainLabels)
    testPredict = clf.predict(RF_dfPCA_test)

    print(metrics.accuracy_score(testLabels, testPredict))
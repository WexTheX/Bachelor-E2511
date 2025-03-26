from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from machineLearning import splitData, scaleFeatures
from extractFeatures import Extract_All_Features
from Preprocessing.preprocessing import fillSets
import os
from sklearn import svm, metrics
import pandas as pd
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


trainDataScaled = scaleFeatures(trainData)
testDataScaled = scaleFeatures(testData)
for i in range(1, 100):
    clf = RandomForestClassifier(max_depth=1, random_state=i)
    clf.fit(trainDataScaled, trainLabels)
    testPredict = clf.predict(testDataScaled)

    print(metrics.accuracy_score(testLabels, testPredict))



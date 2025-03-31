from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from machineLearning import splitData, scaleFeatures
from extractFeatures import extractAllFeatures
from Preprocessing.preprocessing import fillSets
import os
from sklearn import svm, metrics
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import GridSearchCV 

path = "Preprocessing/Datafiles"
outputPath = "OutputFiles/"
windowLengthSeconds = 13
Fs = 800
randomness = 1222
split_value = 0.75


pathNames = os.listdir(path)
activityName = [name[:4].upper() for name in pathNames]


wantFeatureExtraction = False
sets, setsLabel = fillSets(path, pathNames, activityName, seperate_types=False)

if "feature_df" not in globals():
    windowLabels = []
    feature_df = pd.read_csv(outputPath+"feature_df.csv")
    f = open(outputPath+"window_labels.txt", "r")
    data = f.read()
    windowLabels = data.split("\n")
    f.close()
    windowLabels.pop()
trainData, testData, trainLabels, testLabels = splitData(feature_df, windowLabels, randomness, split_value=split_value)

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


param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'max_features': ['sqrt', 'log2', None],  # Number of features to consider for split
    'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
}



#grid_search = GridSearchCV(RandomForestClassifier(),
                           #param_grid=param_grid)
#grid_search.fit(trainData, trainLabels)

#print(grid_search.best_params_)





for i in range(1, 100):
    clf = RandomForestClassifier(max_depth=None, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, n_estimators=200, random_state=i)
    clf.fit(RF_dfPCA_train, trainLabels)
    testPredict = clf.predict(RF_dfPCA_test)

    print(metrics.accuracy_score(testLabels, testPredict))

print()

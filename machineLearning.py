import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Splits data
def splitData(df, labelList, randomness):

  xTrain, xTest, yTrain, yTest = train_test_split(
    df, labelList, test_size=0.25, random_state=randomness
  )

  return xTrain, xTest, yTrain, yTest

def scaleFeatures(df):
  scaler = StandardScaler()
  scaler.set_output(transform="pandas")

  scaledFeatures = scaler.fit_transform(df)

  return scaledFeatures
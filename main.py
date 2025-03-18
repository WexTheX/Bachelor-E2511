# Main file
# Global imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Local imports
# from FOLDER import FILE as F
from extractFeatures import Extract_All_Features
# from machineLearning import 
from plotting import plotFFT, plotWelch, testWelch

from SignalProcessing import ExtractIMU_Features as IMU_F
from SignalProcessing import get_Freq_Domain_features_of_signal as freq
from Preprocessing.preprocessing import fillSets

# Spesify path for input and output of files
path = "Preprocessing/Datafiles"
outputPath = "OutputFiles/features_df.csv"

windowLengthSeconds = 10
Fs = 800

# Load sets and label for those sets from given path
sets, setsLabel = fillSets(path)

# Feature extraction:

# TODO: Går det an å sjekke ka som allerede e extracta og kun hente ut det som ikkje e gjort fra før?
# Make a df with all features and saving it to a .csv file with a random name for now

feature_df, windowLabels = Extract_All_Features(sets, setsLabel, windowLengthSeconds*Fs, False, 800, path)
feature_df.to_csv("OutputFiles/feature_df.csv", index=False)

if "feature_df" not in globals():
    feature_df = pd.read_csv("OutputFiles/features_df.csv")

print(feature_df)
# print(windowLabels)

# GRIN_features = pd.read_csv("OutputFiles/GRIN_features.csv")

# mean_accel_x = GRIN_features["mean_accel_X"]    
# print(mean_accel_x)

# 140 elements per row
# row n = accel xyz TD, accel xyz FD, gyro xyz TD, gyro xyz FD, mag xyz TD, mag xyz FD, temp TD, temp FD from window n

# Machine learning part:



# Plotting part:
# Plot FFT:
variables = ["Axl.X"]
'''
FFTfeature = []
FFTfeature.append("Axl.X")
plotFFT(sets, FFTfeature)
'''

# plotWelch(sets, variables, Fs)

testWelch(sets[0], variables[0], Fs)
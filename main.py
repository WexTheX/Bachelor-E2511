# Main file
# Global imports
import numpy as np
import pandas as pd
import matplotlib as plt
from pathlib import Path

# Local imports
# from FOLDER import FILE as F
from SignalProcessing import ExtractIMU_Features as IMU_F
from SignalProcessing import get_Freq_Domain_features_of_signal as freq
from extractFeatures import Extract_All_Features
from Preprocessing.preprocessing import fillSets

windowLength = 8000
Fs = 800
outputPath = "OutputFiles/GRIN_features.csv"
path = "Preprocessing/Datafiles"
sets, setsLabel = fillSets(path)

# Extract features
feature_df = Extract_All_Features(sets, windowLength, False, 800, path)
feature_df.to_csv("OutputFiles/feature_df.csv", index=False)

# print(feature_df)

GRIN_features = pd.read_csv("OutputFiles/GRIN_features.csv")

mean_accel_x = GRIN_features["mean_accel_X"]    
print(mean_accel_x)

# 140 elements per row
# row n = accel xyz TD, accel xyz FD, gyro xyz TD, gyro xyz FD, mag xyz TD, mag xyz FD, temp TD, temp FD from window n
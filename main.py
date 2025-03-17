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

sets, setsLabel = fillSets("Preprocessing/Datafiles")
windowLength = 8000
Fs = 800
outputPath = "OutputFiles/GRIN_features.csv"
path = "Preprocessing/Datafiles"

# TODO: Går det an å sjekke ka som allerede e extracta og kun hente ut det som ikkje e gjort fra før?
# Make a df with all features and saving it to a .csv file with a random name for now

feature_df = Extract_All_Features(sets, windowLength, False, 800, path)

feature_df.to_csv("OutputFiles/GRIN_features.csv", index=False)
print(feature_df)

GRIN_features = pd.read_csv("OutputFiles/GRIN_features.csv")

mean_accel_x = GRIN_features["mean_accel_X"]    
print(mean_accel_x)

# 140 elements per row
# row n = accel xyz TD, accel xyz FD, gyro xyz TD, gyro xyz FD, mag xyz TD, mag xyz FD, temp TD, temp FD from window n
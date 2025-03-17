# Main file
# Global imports
import numpy as np
import pandas as pd
import matplotlib as plt
from pathlib import Path

# Local imports
# from FOLDER import FILE as F
from SignalProcessing.FFT_plots import plotFFT
from SignalProcessing import ExtractIMU_Features as IMU_F
from SignalProcessing import get_Freq_Domain_features_of_signal as freq
from SignalProcessing.extractFeatures import Extract_All_Features

# Add sets to plot, sets.append("path from main file")
sets = []
setsLabel = []
variables = []
windowLength = 8000
Fs = 800
outputPath = "OutputFiles/GRIN_features.csv"

# Grinding path
folder_path = Path("Preprocessing/Datafiles/Grinding")
txt_files = list(folder_path.glob("*.txt"))
print(txt_files)

for i in range(len(txt_files)):
    sets.append("Preprocessing/Datafiles/Grinding/GRIND_"+ str(i) )
    setsLabel.append("GRIN")

# Idle path
folder_path = Path("Preprocessing/Datafiles/Idle")
txt_files = list(folder_path.glob("*.txt"))

for i in range(len(txt_files)):
    sets.append("Preprocessing/Datafiles/Idle/IDLE_" + str(i) )
    setsLabel.append("IDLE")

print(f"Sets: \n {sets}")
print(f"setsLabel: \n {setsLabel}")

# TODO: Går det an å sjekke ka som allerede e extracta og kun hente ut det som ikkje e gjort fra før?
# Make a df with all features and saving it to a .csv file with a random name for now
feature_df = Extract_All_Features(sets, windowLength, False, 800)
feature_df.to_csv("OutputFiles/GRIN_features.csv", index=False)
print(feature_df)

GRIN_features = pd.read_csv("OutputFiles/GRIN_features.csv")

mean_accel_x = GRIN_features["mean_accel_X"]    
print(mean_accel_x)

# 140 elements per row
# row n = accel xyz TD, accel xyz FD, gyro xyz TD, gyro xyz FD, mag xyz TD, mag xyz FD, temp TD, temp FD from window n


# Choose variables to compare, variables.append("column name")
# variables.append("Axl.X")
# variables.append("Axl.Y")
# variables.append("Axl.Z")

# plotFFT(sets, variables)
# plotFFT(sets, variables)

# plot Welch
# df = pd.read_csv(sets[0]+".csv")
# signal = df[variables[0]]
# welchDiagram = freq.get_Freq_Domain_features_of_signal(signal, "accel_x", 800)

# Det som var i ExtractIMU_Features før:

# sensorTypes = "Gyr.X Gyr.Y Gyr.Z Axl.X Axl.Y Axl.Z Mag.X Mag.Y Mag.Z Temp Hum".split()
# variables.extend(sensorTypes)

# print(variables, sets)

# Tester om det er nok med 10 sekunder vindu
# sets.append("Preprocessing/Datafiles/Grinding/GRIND_0")
# sets.append("Preprocessing/Datafiles/Grinding/TEST")
#TEST er fra samme fil GRIND0 men fra 800 til 1600
# plotFFT(sets, variables)
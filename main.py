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

print(sets)
print(setsLabel)


feature_dataframe = Extract_All_Features(sets, windowLength, 0, 800)
print(feature_dataframe)


# Choose variables to compare, variables.append("column name")
# variables.append("Axl.X")
# variables.append("Axl.Y")
# variables.append("Axl.Z")

# plotFFT(sets, variables)

# plot Welch
# df = pd.read_csv(sets[0]+".csv")
# signal = df[variables[0]]
# welchDiagram = freq.get_Freq_Domain_features_of_signal(signal, "accel_x", 800)

# Det som var i ExtractIMU_Features før:

# sensorTypes = "Gyr.X Gyr.Y Gyr.Z Axl.X Axl.Y Axl.Z Mag.X Mag.Y Mag.Z Temp Hum".split()
# variables.extend(sensorTypes)

# print(variables, sets)
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

## Test code
# windowLength = 10

# imu_data = np.zeros((windowLength, 7))

# for i in range(0, windowLength):
    # imu_data[i:,0] = 1741840 + i

# print(imu_data)

# ExtractIMU_Features(imu_data, windowLength, 0)

# Add sets to plot, sets.append("path from main file")
sets = []
setsLabel = []
windowLength = 800
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


Extract_All_Features(sets, windowLength, 0, 800)

# Choose variables to compare, variables.append("column name")
variables = []
# variables.append("Axl.X")
# variables.append("Axl.Y")
variables.append("Axl.Z")

plotFFT(sets, variables)

# plt.figure()
df = pd.read_csv(sets[0]+".csv")
x = df[variables[0]]

# plot Welsh
a = freq.get_Freq_Domain_features_of_signal(x, "accel_x", 800)



# print(a)

# Det som var i ExtractIMU_Features før:

# sensorTypes = "Gyr.X Gyr.Y Gyr.Z Axl.X Axl.Y Axl.Z Mag.X Mag.Y Mag.Z Temp Hum".split()
# variables.extend(sensorTypes)

# print(variables, sets)
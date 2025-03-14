# Main file
# Global imports
import numpy as np
import pandas as pd

# Local imports
# from FOLDER import FILE as F
from SignalProcessing.FFT_plots import plotFFT
from SignalProcessing import ExtractIMU_Features as IMU_F
from SignalProcessing import get_Freq_Domain_features_of_signal as freq


## Test code
# windowLength = 10

# imu_data = np.zeros((windowLength, 7))

# for i in range(0, windowLength):
    # imu_data[i:,0] = 1741840 + i

# print(imu_data)

# ExtractIMU_Features(imu_data, windowLength, 0)

# Add sets to plot, sets.append("path from main file")
sets = []
sets.append("Datafiles/20250226 Angle Grinder/26.02.2025 094702")
sets.append("Datafiles/20250226 Angle Grinder/26.02.2025 094951")
sets.append("Datafiles/20250226 Angle Grinder/26.02.2025 095305")
print(sets)

# Choose variables to compare, variables.append("column name")
variables = []
variables.append("Axl.X")
variables.append("Axl.Y")
variables.append("Axl.Z")

df = pd.read_csv(sets[0]+".csv")
x = df[variables[1]]

a = freq.get_Freq_Domain_features_of_signal(x, "accel_x", 800)
# b = FFT.

print(a)

# Det som var i ExtractIMU_Features før:

# sensorTypes = "Gyr.X Gyr.Y Gyr.Z Axl.X Axl.Y Axl.Z Mag.X Mag.Y Mag.Z Temp Hum".split()
# variables.extend(sensorTypes)

# print(variables, sets)
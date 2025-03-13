# Main file
# Global imports
import numpy as np

# Local imports
# from FOLDER import FILE as F
from SignalProcessing import FFT_plots as FFT
from SignalProcessing import ExtractIMU_Features as IMU_F

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

# Choose variables to compare, variables.append("column name")
variables = []
variables.append("Axl.X")
variables.append("Axl.Y")
variables.append("Axl.Z")

FFT.plotFFT(sets, variables)

# Det som var i ExtractIMU_Features før:

sensorTypes = "Gyr.X Gyr.Y Gyr.Z Axl.X Axl.Y Axl.Z Mag.X Mag.Y Mag.Z Temp Hum".split()
variables.extend(sensorTypes)

print(variables, sets)
# Main file
# Global imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

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
#folder_path = Path("Preprocessing/Datafiles/Grinding")
#txt_files = list(folder_path.glob("*.txt"))
#print(txt_files)

#for i in range(len(txt_files)):
    #sets.append("Preprocessing/Datafiles/Grinding/GRIND_"+ str(i) )
    #setsLabel.append("GRIN")

# Idle path
folder_path = Path("Preprocessing/Datafiles/Idle")
txt_files = list(folder_path.glob("*.txt"))

#for i in range(len(txt_files)):
    #sets.append("Preprocessing/Datafiles/Idle/IDLE_" + str(i) )
    #setsLabel.append("IDLE")

print(f"Sets: \n {sets}")
print(f"setsLabel: \n {setsLabel}")

# TODO: Går det an å sjekke ka som allerede e extracta og kun hente ut det som ikkje e gjort fra før?
# Make a df with all features and saving it to a .csv file with a random name for now
#feature_dataframe = Extract_All_Features(sets, windowLength, False, 800)
#feature_dataframe.to_csv("Outputfiles/features4.csv", index=False)
#print(feature_dataframe)

# 140 elements per row
# row n = accel xyz TD, accel xyz FD, gyro xyz TD, gyro xyz FD, mag xyz TD, mag xyz FD, temp TD, temp FD from window n



# Choose variables to compare, variables.append("column name")
#variables.append("Axl.X")
#variables.append("Axl.Y")
variables.append("Axl.Z")

# plotFFT(sets, variables)
# plotFFT(sets, variables)

# plot Welch
sets.append("Preprocessing/Datafiles/Grinding/GRIND_0")
sets.append("WINDOW TESTING/10_TIL_20_GRIND0")
sets.append("WINDOW TESTING/20_TIL_30_GRIND0")
sets.append("WINDOW TESTING/30_TIL_40_GRIND0")
sets.append("WINDOW TESTING/40_TIL_50_GRIND0")

df = pd.read_csv(sets[0]+".csv")
df2 = pd.read_csv(sets[1]+".csv")
df3 = pd.read_csv(sets[2]+".csv")
df4 = pd.read_csv(sets[3]+".csv")
df5 = pd.read_csv(sets[4]+".csv")

signal1 = df[variables[0]]
signal2 = df2[variables[0]]
signal3 = df3[variables[0]]
signal4 = df4[variables[0]]
signal5 = df5[variables[0]]

freq1, psd1, features1 = freq.get_Freq_Domain_features_of_signal(signal1, "accel_x", 800)
freq2, psd2, features2 = freq.get_Freq_Domain_features_of_signal(signal2, "accel_x", 800)
freq3, psd3, features3 = freq.get_Freq_Domain_features_of_signal(signal3, "accel_x", 800)
freq4, psd4, features4 = freq.get_Freq_Domain_features_of_signal(signal4, "accel_x", 800)
freq5, psd5, features5 = freq.get_Freq_Domain_features_of_signal(signal5, "accel_x", 800)

plt.semilogy(freq1, psd1, label='60 sec (Whole set)')
plt.semilogy(freq2, psd2, label='10 to 20 sec')
plt.semilogy(freq3, psd3, label='20 to 30 sec')
plt.semilogy(freq4, psd4, label='30 to 40 sec')
plt.semilogy(freq5, psd5, label='40 to 50 sec')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (PSD)')
plt.title('PSD from Welsch method split into windows and whole set, Z-dir, angle grinding')

plt.grid()
plt.show()
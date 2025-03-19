''' PLOTS STANDALONE, NOT CALLED IN MAIN '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SignalProcessing.get_Freq_Domain_features_of_signal import getFFT, getWelch

''' VARIABLES '''
fs = 800
filtering = False
omega_n = 15
order = 3

datasets = []
variables = []

''' ADD DATASETS '''
# datasets.append("Preprocessing/Datafiles/Grinding/GRIND_0.txt")
# datasets.append("Preprocessing/Datafiles/Grinding/GRIND_4.txt")
# datasets.append("Preprocessing/Datafiles/Grinding/GRIND_11.txt")
# datasets.append("Preprocessing/Datafiles/Idle/IDLE_0.txt")
# datasets.append("Preprocessing/Datafiles/Idle/IDLE_1.txt")
# datasets.append("Preprocessing/Datafiles/Idle/IDLE_2.txt")
datasets.append("Preprocessing/Datafiles/Welding/WELD_0.txt")
datasets.append("Preprocessing/Datafiles/Welding/WELD_6.txt")
datasets.append("Preprocessing/Datafiles/Welding/18.03.2025 143542.txt")

''' ADD VARIABLES '''
# variables.append("Axl.X")
# variables.append("Axl.Y")
# variables.append("Axl.Z")

variables.append("Mag.X")
# variables.append("Mag.Y")
# variables.append("Mag.Z")

''' WELCH PLOTTING '''
# for i in datasets:
#   df = pd.read_csv(i, delimiter="\t")
#   for j in variables:
#     x = df[j]
#     x_yf, x_xf, x_size = getFFT(x)
#     plt.plot(x_xf, 2.0/x_size*np.abs(x_yf[0:x_size//2]))
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Amplitude')
#     plt.title('FFT, %s' % j)
#     plt.grid()
#     plt.figure()

''' WELCH PLOTTING '''
legendNames = []
for i in datasets:
  df = pd.read_csv(i, delimiter="\t")
  for j in variables:
    x = df[j]

    freq, psd = getWelch(x, fs, filtering, omega_n, order)
    plt.semilogy(freq, psd)  # Log scale for better visibility
    legendNames.append(i + ", " + j)


plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('Welch PSD, %s' % j)
plt.grid()
plt.legend(legendNames)
plt.figure()    
    

plt.show()


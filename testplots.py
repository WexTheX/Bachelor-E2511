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
# datasets.append("Preprocessing/Datafiles/Welding/WELD_0.txt")
# datasets.append("Preprocessing/Datafiles/Welding/WELD_6.txt")
# datasets.append("Preprocessing/Datafiles/Welding/WELD_10.txt")

# datasets.append("Preprocessing/DatafilesSeparated/WeldAlTIG/WELDALTIG_0.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStTIG/WELDSTTIG_0.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStMAG/WELDSTMAG_0.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStMAG/WELDSTMAG_1.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStMAG/WELDSTMAG_2.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStMAG/WELDSTMAG_3.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStMAG/WELDSTMAG_4.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStMAG/WELDSTMAG_5.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStMAG/WELDSTMAG_6.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStMAG/WELDSTMAG_7.txt")




''' ADD VARIABLES '''
variables.append("Axl.X")
variables.append("Axl.Y")
variables.append("Axl.Z")

variables.append("Mag.X")
variables.append("Mag.Y")
variables.append("Mag.Z")

variables.append("Gyr.X")
variables.append("Gyr.Y")
variables.append("Gyr.Z")

variables.append("Temp")
# variables.append("Press")

# variables.append("Range")
# variables.append("Lum")
# variables.append("IRLum")

''' TIME PLOTTING '''
def plotTime(sets, vars):
  # for i in sets:
  #   df = pd.read_csv(i, delimiter="\t")

  #   for j in vars:
  #     df[j].plot()

  for i in vars:
    legendNames = []

    for j in sets:
      df = pd.read_csv(j, delimiter="\t")
      df[i].plot()
      legendNames.append(j + ", " + i)

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time, %s' % i)
    plt.legend(legendNames)
    plt.grid()
    plt.show()


''' FFT PLOTTING '''
def plotFFT(sets, vars):
  for i in sets:
    plt.figure()

    df = pd.read_csv(i, delimiter="\t")
    for j in vars:
      x = df[j]
      x_yf, x_xf, x_size = getFFT(x)
      plt.plot(x_xf, 2.0/x_size*np.abs(x_yf[0:x_size//2]))

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('FFT, %s' % j)
    plt.grid()
      

''' WELCH PLOTTING '''
def plotWelch(sets, vars):
  # for i in sets:
  #   legendNames = []
  #   df = pd.read_csv(i, delimiter="\t")
  #   for j in vars:
  #     plt.figure()
  #     x = df[j]

  #     freq, psd = getWelch(x, fs, filtering, omega_n, order)
  #     plt.semilogy(freq, psd)  # Log scale for better visibility
  #     legendNames.append(i + ", " + j)

  for i in vars:
    plt.figure()
    legendNames = []

    for j in sets:
      df = pd.read_csv(j, delimiter="\t")
      x = df[i]

      freq, psd = getWelch(x, fs, filtering, omega_n, order)
      plt.semilogy(freq, psd)  # Log scale for better visibility
      legendNames.append(j + ", " + i)
    
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Welch PSD')
    plt.legend(legendNames)
    plt.show()

plotTime(datasets, variables)
# plotFFT(datasets, variables)
# plotWelch(datasets, variables)  

# plt.show()


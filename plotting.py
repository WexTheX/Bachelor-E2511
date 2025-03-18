import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SignalProcessing.get_Freq_Domain_features_of_signal import getFFT, getWelch, butter_highpass

# Plot of normal distribution, WIP
def normDistPlot(dataset, size):
  mean = dataset["mean_accel_X"]
  sd = dataset["sd_accel_X"]
    
  values = np.random.normal(mean, sd, size)

  plt.hist(values, 100)
  plt.axvline(values.mean(), color='k', linestyle='dashed', linewidth=2)
  plt.figure()

'''
df = pd.read_csv("OutputFiles/features4.csv")
size = 10
print(df)

normDistPlot(df[:1], 800*size)
plt.show()
'''

# Plot of FFT of sets and variables, REDO needed
def plotFFT(sets, variables):
  label_names = []

  for i in sets:
    for j in variables:
      #tab_txt_to_csv(i+".txt", i+".csv")
      x_yf, x_xf, x_size = FFTofFile(i, j)
      plt.plot(x_xf, 2.0/x_size*np.abs(x_yf[0:x_size//2]))
      #plt.semilogy(x_xf, 2.0/x_size*np.abs(x_yf[0:x_size//2]))
      label_names.append(i + ", " + j)

      # Plot one plot for each variable, for each set
      # plt.legend(label_names)
      # plt.figure()

    # Plot all variables in same plot, one for each set
    # plt.legend(label_names)
    # plt.figure()
  
  # Plot all sets and variables in same plot
  plt.legend(label_names)
  plt.figure()

# Plot of Welch Method
def plotWelch(sets, variables, fs):
  # df = pd.read_csv(sets[0]+".csv")
  # signal = df[variables[0]]
  # welchDiagram = freq.get_Freq_Domain_features_of_signal(signal, "accel_x", 800)

  # # TBD, this won't work as get_Freq_Domain_features_of_signal() doesn't plot anymore


  for i in sets:
    for j in variables:
      butter_highpass(fs)
      freq, psd = getWelch(i, j, fs)
      plt.semilogy(freq, psd)  # Log scale for better visibility
      plt.xlabel('Frequency (Hz)')
      plt.ylabel('Power Spectral Density')
      plt.title('Welch PSD')
      plt.grid()
      plt.figure()
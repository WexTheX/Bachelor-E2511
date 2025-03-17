import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plot of normal distribution, WIP
def normDistPlot(dataset, size):
    mean = dataset["mean_accel_X"]
    sd = dataset["sd_accel_X"]
    
    values = np.random.normal(mean, sd, size)

    plt.hist(values, 100)
    plt.axvline(values.mean(), color='k', linestyle='dashed', linewidth=2)
    plt.figure()

df = pd.read_csv("OutputFiles/features4.csv")
size = 10
print(df)

normDistPlot(df[:1], 800*size)
plt.show()

# Make FFT of given file, with given feature 
def FFTofFile(file, feature):
  df = pd.read_csv(file+".csv")
  x = df[feature]
  x_size = len(x)
  x_space = 1/800

  x_yf = fft(x)
  x_xf = fftfreq(x_size, x_space)[:x_size//2]
  return x_yf, x_xf, x_size

# Plot of FFT of sets and variables, REDO needed
def plotFFT(sets, variables):
  label_names = []

  for i in sets:
    for j in variables:
      #tab_txt_to_csv(i+".txt", i+".csv")
      x_yf, x_xf, x_size = FFTofFile(i, j)
      plt.semilogy(x_xf, 2.0/x_size*np.abs(x_yf[0:x_size//2]))
      label_names.append(i + ", " + j)

  # Plot FFT result
  plt.legend(label_names)
  plt.grid()
  plt.show()
  plt.figure()
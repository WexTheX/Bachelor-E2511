from scipy.fft import fft, fftfreq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Convert tab seperated txt file to csv file
# txt_file, csv_file format : "filename.txt", "filename.csv"
# Remember to remove "Information" in file when it comes directly from Muse
def tab_txt_to_csv(txt_file, csv_file):
  df_txt = pd.read_csv(txt_file, delimiter='\t')
  df_txt.to_csv(csv_file, index = None)

# Make FFT of given file, with given feature
def FFTofFile(file, feature):
  df = pd.read_csv(file+".csv")
  x = df[feature]
  x_size = len(x)
  x_space = 1/800

  x_yf = fft(x)
  x_xf = fftfreq(x_size, x_space)[:x_size//2]
  return x_yf, x_xf, x_size

# Add sets to plot, sets.append("path from main file")
# Add name of file, not file type (.txt, .csv)
sets = []
sets.append("Datafiles/20250226 Angle Grinder/26.02.2025 094702")
#sets.append("Datafiles/20250226 Angle Grinder/26.02.2025 094951")
#sets.append("Datafiles/20250226 Angle Grinder/26.02.2025 095305")

# Choose variables to compare
variables = []
variables.append("Axl.X")
variables.append("Axl.Y")
variables.append("Axl.Z")

label_names = []

# For loop that plots every set in sets
for i in sets:
  for j in variables:
    tab_txt_to_csv(i+".txt", i+".csv")
    x_yf, x_xf, x_size = FFTofFile(i, j)
    plt.plot(x_xf, 2.0/x_size*np.abs(x_yf[0:x_size//2]))
    label_names.append(i + ", " + j)

# Plot FFT result
plt.legend(label_names)
plt.grid()
plt.show()

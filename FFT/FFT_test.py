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

# Choose the amount of sets used for comparisons
setAmount = 3
#set1 = "25-02-2025 123109_OA_HF"
#set2 = "25-02-2025 123422_FA_HF"
#set3 = "25-02-2025 123720_W_HF"
set1 = "25-02-2025 123031_OA_LF"
set2 = "25-02-2025 123348_FA_LF"
set3 = "25-02-2025 123655_W_LF"

# Choose variable to compare
variabel = "Axl.Z"

# Convert txtfiles to csv files (Possibly redundant)
tab_txt_to_csv(set1+".txt", set1+".csv")
tab_txt_to_csv(set2+".txt", set2+".csv")
tab_txt_to_csv(set3+".txt", set3+".csv")

x1_yf, x1_xf, x1_size = FFTofFile(set1, variabel)
x2_yf, x2_xf, x2_size = FFTofFile(set2, variabel)
x3_yf, x3_xf, x3_size = FFTofFile(set3, variabel)

# Plot FFT result
plt.plot(x1_xf, 2.0/x1_size*np.abs(x1_yf[0:x1_size//2]))
plt.plot(x2_xf, 2.0/x2_size*np.abs(x2_yf[0:x2_size//2]))
plt.plot(x3_xf, 2.0/x3_size*np.abs(x3_yf[0:x3_size//2]))
plt.legend(['OA','FA', 'W'])
plt.grid()
plt.show()

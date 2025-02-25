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

#tab_txt_to_csv("20250221112108_Drill_10s.txt", "20250221112108_Drill_10s.csv")

# Choose the amount of sets used for comparisons
setAmount = 3
set1 = "25-02-2025 123109_OA_HF"
set2 = "25-02-2025 123422_FA_HF"
set3 = "25-02-2025 123720_W_HF"

# Choose variable to compare
variabel = "Gyr.Z"

# Convert txtfiles to csv files (Possibly redundant)
tab_txt_to_csv(set1+".txt", set1+".csv")
tab_txt_to_csv(set2+".txt", set2+".csv")
tab_txt_to_csv(set3+".txt", set3+".csv")

# Read csv file
df_1 = pd.read_csv(set1+".csv")
df_2 = pd.read_csv(set2+".csv")
df_3 = pd.read_csv(set3+".csv")

# Define variables for set 1
x1 = df_1[variabel]
x1_size = len(x1)
x1_space = 1/800 # Usikkert på denne variablen
# FFT for set 1
x1_yf = fft(x1)
x1_xf = fftfreq(x1_size, x1_space)[:x1_size//2]

# Define variables for set 2
x2 = df_2[variabel]
x2_size = len(x2)
x2_space = 1/800 
# FFT for set 2
x2_yf = fft(x2)
x2_xf = fftfreq(x2_size, x2_space)[:x2_size//2]

# Define variables for set 3
x3 = df_3[variabel]
x3_size = len(x3)
x3_space = 1/800 
# FFT for set 3
x3_yf = fft(x3)
x3_xf = fftfreq(x3_size, x3_space)[:x3_size//2]

# Plot FFT result
plt.plot(x1_xf, 2.0/x1_size*np.abs(x1_yf[0:x1_size//2]))
plt.plot(x2_xf, 2.0/x2_size*np.abs(x2_yf[0:x2_size//2]))
plt.plot(x3_xf, 2.0/x3_size*np.abs(x3_yf[0:x3_size//2]))
plt.legend(['OA','FA', 'W'])
plt.grid()
plt.show()
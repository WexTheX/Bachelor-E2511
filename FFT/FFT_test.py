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

# Read csv file
df = pd.read_csv("20250221112108_Drill_10s.csv")

# Get x-coordinates for axeleration, get sample size
axlx = df['Axl.X']
axlx_size = len(axlx)
axlx_space = 1/1600 # Unsure of this is correct (sample frequency was 1600 for the log)

# FFT of axlx
yf = fft(axlx)
xf = fftfreq(axlx_size, axlx_space)[:axlx_size//2]

# Plot FFT result
plt.plot(xf, 2.0/axlx_size*np.abs(yf[0:axlx_size//2]))
plt.grid()
plt.show()
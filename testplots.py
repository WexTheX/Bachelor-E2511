''' PLOTS STANDALONE, NOT CALLED IN MAIN '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import math as math

from SignalProcessing.get_Freq_Domain_features_of_signal import getFFT, getWelch

''' VARIABLES '''
fs = 800
filtering = False
omega_n = 15
order = 3

column_names = ["Timestamp","Gyr.X","Gyr.Y","Gyr.Z","Axl.X","Axl.Y","Axl.Z","Mag.X","Mag.Y","Mag.Z","Temp"]
datasets = []
variables = []

''' ADD DATASETS '''
# datasets.append("Preprocessing/DatafilesSeparated/GrindBig/GRINDBIG_0.txt")
# datasets.append("Preprocessing/DatafilesSaperated/GrindBig/GRINDBIG_1.txt")
# datasets.append("Preprocessing/DatafilesSaperated/GrindBig/GRINDBIG_2.txt")

# datasets.append("Preprocessing/DatafilesSeparated/GrindMed/GRINDMED_0.txt")
# datasets.append("Preprocessing/DatafilesSeparated/GrindMed/GRINDMED_1.txt")
# datasets.append("Preprocessing/DatafilesSeparated/GrindMed/GRINDMED_2.txt")

# datasets.append("Preprocessing/DatafilesSeparated/GrindSmall/GRINDSMALL_0.txt")
# datasets.append("Preprocessing/DatafilesSeparated/GrindSmall/GRINDSMALL_1.txt")
# datasets.append("Preprocessing/DatafilesSeparated/GrindSmall/GRINDSMALL_2.txt")
# datasets.append("Preprocessing/DatafilesSeparated/GrindSmall/GRINDSMALL_6.txt")
# datasets.append("Preprocessing/DatafilesSeparated/GrindSmall/GRINDSMALL_8.txt")
# datasets.append("Preprocessing/DatafilesSeparated/GrindSmall/GRINDSMALL_11.txt")


datasets.append("Preprocessing/DatafilesSeparated/WeldAlTIG/WELDALTIG_0.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldAlTIG/WELDALTIG_2.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldAlTIG/WELDALTIG_4.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldAlTIG/WELDALTIG_6.txt")

datasets.append("Preprocessing/DatafilesSeparated/WeldStTIG/WELDSTTIG_0.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStTIG/WELDSTTIG_1.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStTIG/WELDSTTIG_2.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStTIG/WELDSTTIG_3.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStTIG/WELDSTTIG_4.txt")

datasets.append("Preprocessing/DatafilesSeparated/WeldStMAG/WELDSTMAG_0.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStMAG/WELDSTMAG_1.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStMAG/WELDSTMAG_2.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStMAG/WELDSTMAG_3.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStMAG/WELDSTMAG_4.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStMAG/WELDSTMAG_5.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStMAG/WELDSTMAG_6.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStMAG/WELDSTMAG_7.txt")


# datasets.append("Preprocessing/DatafilesSeparated/Impa/IMPA_0.txt")
# datasets.append("Preprocessing/DatafilesSeparated/Impa/IMPA_1.txt")
# datasets.append("Preprocessing/DatafilesSeparated/Impa/IMPA_2.txt")


datasets.append("Preprocessing/DatafilesSeparated/Idle/IDLE_0.txt")
# datasets.append("Preprocessing/DatafilesSeparated/Idle/IDLE_4.txt")
# datasets.append("Preprocessing/DatafilesSeparated/Idle/IDLE_8.txt")

''' ADD VARIABLES '''
variables.append("Axl.X")
# variables.append("Axl.Y")
# variables.append("Axl.Z")

variables.append("Mag.X")
# variables.append("Mag.Y")
# variables.append("Mag.Z")

# variables.append("Gyr.X")
# variables.append("Gyr.Y")
# variables.append("Gyr.Z")

# variables.append("Temp")
# variables.append("Press")

# variables.append("Range")
# variables.append("Lum")
# variables.append("IRLum")

def butter_lowpass_filter(data, fs, omega_n, order):

    b, a = sp.signal.butter(order, Wn=omega_n, fs=fs, btype='low', analog=False)
    result = sp.signal.lfilter(b, a, data)

    return result

''' DOWNSAMPELING '''
def downsample(df: pd.DataFrame, old_fs, new_fs):
    '''
    dropped_rows = []

    if((old_fs / new_fs).is_integer() == False):
        print(f"Old fs: {old_fs} / New fs: {new_fs} is not whole number")
        quit()
    elif((old_fs < new_fs)):
        print(f"Old fs: {old_fs} is smaller than New fs: {new_fs}")
        quit()
    else:
        for i in range(len(df['Timestamp'])):
            if((i % (old_fs / new_fs)) != 0):
                dropped_rows.append(i)

    new_df = df.drop(dropped_rows)
    '''
    # print(df.describe())
    new_df = pd.DataFrame(columns=column_names)
    for column in df:
      new_df[column] = sp.signal.decimate(df[column], math.floor(fs/ds_fs), ftype="fir")
    
    # print(new_df.describe())

    # df = df.set_index(pd.timedelta_range(start='0us', periods=len(df['Timestamp']), freq="1250us"))
    # print(info_df)
    # print(info_df.describe())

    # new_df = df.resample('5ms').interpolate()
    # print(new_info_df.describe())
    return new_df


''' TIME PLOTTING '''
def plotTime(sets, vars, fs, ds_fs):
  # for i in sets:
  #   df = pd.read_csv(i, delimiter="\t")

  #   for j in vars:
  #     df[j].plot()

  for i in vars:
    plt.figure()
    legendNames = []

    for j in sets:
      og_df = pd.read_csv(j, delimiter="\t")
      # og_x = og_df[i]
      if(ds_fs == fs):
        df = og_df
        # x = og_x
      else:
        # x = sp.signal.decimate(og_x, math.floor(fs/ds_fs), ftype="fir")
        
        # x_length = math.floor(len(og_x)/ds_fs)
        # x = sp.signal.resample(og_x, x_length) 
        df = downsample(og_df, fs, ds_fs)
      x = df[i]

      plt.plot(x)
      # x.plot()
      legendNames.append(j + ", " + i)

    plt.xlabel(f'Time, {ds_fs}', size=20)
    plt.ylabel('Value', size=20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.title('Time, %s' % i, size=20)
    plt.legend(legendNames, prop={'size': 20})
    plt.grid()
    # plt.show()


''' FFT PLOTTING '''
def plotFFT(sets, vars, fs, ds_fs):
  for i in sets:
    plt.figure()

    df = pd.read_csv(i, delimiter="\t")
    
    for j in vars:
      og_df = pd.read_csv(j, delimiter="\t")
      # og_x = og_df[i]
      if(ds_fs == fs):
        df = og_df
        # x = og_x
      else:
        # x = sp.signal.decimate(og_x, math.floor(fs/ds_fs), ftype="fir")
        
        # x_length = math.floor(len(og_x)/ds_fs)
        # x = sp.signal.resample(og_x, x_length) 
        df = downsample(og_df, fs, ds_fs)
      x = df[i]

      x_yf, x_xf, x_size = getFFT(x)
      plt.plot(x_xf, 2.0/x_size*np.abs(x_yf[0:x_size//2]))

    plt.xlabel(f'Frequency (Hz), {fs/ds_fs}', size=20)
    plt.ylabel('Amplitude', size=20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.title('FFT, %s' % j, size=20)
    plt.grid()
      

''' WELCH PLOTTING '''
def plotWelch(sets, vars, fs, ds_fs):
  # for i in sets:
  #   legendNames = []
  #   og_df = pd.read_csv(i, delimiter="\t")
  #   df = sp.signal.decimate(og_df, ds_fs)
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
      og_df = pd.read_csv(j, delimiter="\t")
      # og_x = og_df[i]
      if(ds_fs == fs):
        df = og_df
        # x = og_x
      else:
        # x = sp.signal.decimate(og_x, math.floor(fs/ds_fs), ftype="fir")
        
        # x_length = math.floor(len(og_x)/ds_fs)
        # x = sp.signal.resample(og_x, x_length) 
        df = downsample(og_df, fs, ds_fs)
      x = df[i]

      freq, psd = getWelch(x, ds_fs, filtering, omega_n, order)
      plt.semilogy(freq, psd)  # Log scale for better visibility
      tname = j.split("/")
      name = tname[2]
      legendNames.append(name + ", " + i)
    
    plt.grid()
    plt.xlabel(f'Frequency (Hz), {ds_fs} Hz signal', size=20)
    plt.ylabel('Power Spectral Density', size=20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.title('Welch PSD', size=20)
    plt.legend(legendNames, loc='upper right', prop={'size': 20})
    # plt.show()

# info_df = pd.read_csv(datasets[0], delimiter="\t")
# info_df = info_df.set_index(pd.timedelta_range(start='0us', periods=len(info_df['Timestamp']), freq="1250us"))
# print(info_df)
# print(info_df.describe())
# new_info_df = info_df.resample('5ms').interpolate()
# print(new_info_df.describe())
# ds_fs = 200
# new_info_df = downsample(info_df, fs, ds_fs)
# print(new_info_df.describe)


ds_fs = 800
plotTime(datasets, variables, fs, ds_fs)
# plotFFT(datasets, variables, fs, ds_fs)
plotWelch(datasets, variables, fs, ds_fs)  

# ds_fs = 200
# plotTime(datasets, variables, fs, ds_fs)
# plotFFT(datasets, variables, fs, ds_fs)
# plotWelch(datasets, variables, fs, ds_fs)  

plt.show()


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
size = 10

column_names = ["Timestamp","Gyr.X","Gyr.Y","Gyr.Z","Axl.X","Axl.Y","Axl.Z","Mag.X","Mag.Y","Mag.Z","Temp"]
datasets = []
variables = []

''' ADD DATASETS '''
# datasets.append("Preprocessing/DatafilesSeparated/GrindBig/GRINDBIG_0.txt")
# datasets.append("Preprocessing/DatafilesSeparated/GrindBig/GRINDBIG_1.txt")
# datasets.append("Preprocessing/DatafilesSeparated/GrindBig/GRINDBIG_2.txt")

# datasets.append("Preprocessing/DatafilesSeparated/GrindMed/GRINDMED_0.txt")
# datasets.append("Preprocessing/DatafilesSeparated/GrindMed/GRINDMED_1.txt")
# datasets.append("Preprocessing/DatafilesSeparated/GrindMed/GRINDMED_2.txt")

datasets.append("Preprocessing/DatafilesSeparated/GrindSmall/GRINDSMALL_0.txt")
# datasets.append("Preprocessing/DatafilesSeparated/GrindSmall/GRINDSMALL_1.txt")
# datasets.append("Preprocessing/DatafilesSeparated/GrindSmall/GRINDSMALL_2.txt")
# datasets.append("Preprocessing/DatafilesSeparated/GrindSmall/GRINDSMALL_6.txt")
# datasets.append("Preprocessing/DatafilesSeparated/GrindSmall/GRINDSMALL_8.txt")
# datasets.append("Preprocessing/DatafilesSeparated/GrindSmall/GRINDSMALL_11.txt")


# datasets.append("Preprocessing/DatafilesSeparated/WeldAlTIG/WELDALTIG_0.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldAlTIG/WELDALTIG_2.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldAlTIG/WELDALTIG_4.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldAlTIG/WELDALTIG_6.txt")

# datasets.append("Preprocessing/DatafilesSeparated/WeldStTIG/WELDSTTIG_0.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStTIG/WELDSTTIG_1.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStTIG/WELDSTTIG_2.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStTIG/WELDSTTIG_3.txt")
# datasets.append("Preprocessing/DatafilesSeparated/WeldStTIG/WELDSTTIG_4.txt")

# datasets.append("Preprocessing/DatafilesSeparated/WeldStMAG/WELDSTMAG_0.txt")
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


# datasets.append("Preprocessing/DatafilesSeparated/Idle/IDLE_0.txt")
# datasets.append("Preprocessing/DatafilesSeparated/Idle/IDLE_4.txt")
# datasets.append("Preprocessing/DatafilesSeparated/Idle/IDLE_8.txt")

#datasets.append("800v200hz/800hz.txt")

''' ADD VARIABLES '''
variables.append("Axl.X")
# variables.append("Axl.Y")
# variables.append("Axl.Z")

# variables.append("Mag.X")
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

def myDecimate(x, fs, ds_fs, omega_n, numtaps):
  ds_factor   = math.floor(fs/ds_fs)
  nyquist_fs  = fs/2

  fir_coeff = sp.signal.firwin(numtaps=numtaps, cutoff=omega_n/nyquist_fs)
  filter_x = sp.signal.filtfilt(b=fir_coeff, a=[1.0], x=x, axis=0)

  ds_x = filter_x[::ds_factor]
  return ds_x

''' DOWNSAMPELING '''
def downsample(df: pd.DataFrame, old_fs, new_fs, f_type):
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
    # t_df = pd.DataFrame(columns=column_names)
    # for column in df:
    #   t_df[column] = butter_lowpass_filter(df[column], old_fs, 200, 10)

    # dropped_rows = []
    # for i in range(len(df['Timestamp'])):
    #         if((i % (old_fs / new_fs)) != 0):
    #             dropped_rows.append(i)
    
    # new_df = df.drop(dropped_rows)

    # print(df.describe())
    new_df = pd.DataFrame(columns=column_names)
    for column in df:
      new_df[column] = sp.signal.decimate(df[column], math.floor(fs/ds_fs), ftype=f_type, zero_phase=True)
    
    # new_df = pd.DataFrame(columns=column_names)
    # for column in df:
    #   new_df[column] = myDecimate(x=df[column], fs=fs, ds_fs=ds_fs, omega_n=90, numtaps=13)

    # new_df = pd.DataFrame(columns=column_names)
    # for column in df:
    #   new_df[column] = sp.signal.resample_poly(df[column], up=1, down=4, window=sp.signal.windows.kaiser(15, 14), axis=0)
    # print(new_df.describe())


    # df = df.set_index(pd.timedelta_range(start='0us', periods=len(df['Timestamp']), freq="1250us"))
    # print(info_df)
    # print(info_df.describe())

    # new_df = df.resample('5ms').interpolate()
    # print(new_info_df.describe())
    return new_df

def normalizeSets(datasets):
  for sets in datasets:
    tname = sets.split("/")
    name = tname[2]
    df = pd.read_csv(sets, delimiter="\t")
    axlx = df['Axl.X']
    axly = df['Axl.Y']
    axlz = df['Axl.Z']

    gyrx = df['Gyr.X']
    gyry = df['Gyr.Y']
    gyrz = df['Gyr.Z']

    magx = df['Mag.X']
    magy = df['Mag.Y']
    magz = df['Mag.Z']

    axl = np.sqrt( np.power(axlx, 2) + np.power(axly, 2) + np.power(axlz, 2))
    gyr = np.sqrt( np.power(gyrx, 2) + np.power(gyry, 2) + np.power(gyrz, 2))
    mag = np.sqrt( np.power(magx, 2) + np.power(magy, 2) + np.power(magz, 2))

    plt.figure()
    plt.plot(axl)
    plt.xlabel(f'Time')
    plt.ylabel('Value')
    plt.title(f'Time, axl, {name}')
    plt.grid()
    
    plt.figure()
    plt.plot(gyr)
    plt.xlabel(f'Time')
    plt.ylabel('Value')
    plt.title(f'Time, gyr, {name}')
    plt.grid()

    plt.figure()
    plt.plot(mag)
    plt.xlabel(f'Time')
    plt.ylabel('Value')
    plt.title(f'Time, gyr, {name}')
    plt.grid()

    plt.figure()
    freq, psd = getWelch(axl, 800, filtering, omega_n, order)
    plt.semilogy(freq, psd)  # Log scale for better visibility
    plt.grid()
    plt.xlabel(f'Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title(f'Welch PSD, axl, {name}')

    plt.figure()
    freq, psd = getWelch(gyr, 800, filtering, omega_n, order)
    plt.semilogy(freq, psd)  # Log scale for better visibility
    plt.grid()
    plt.xlabel(f'Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title(f'Welch PSD, gyr, {name}')

    plt.figure()
    freq, psd = getWelch(mag, 800, filtering, omega_n, order)
    plt.semilogy(freq, psd)  # Log scale for better visibility
    plt.grid()
    plt.xlabel(f'Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title(f'Welch PSD, mag, {name}')

    plt.show()

    

  return

    

''' TIME PLOTTING '''
def plotTime(sets, vars, fs, ds_fs, f_type="fir", size=20):
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
        df = downsample(og_df, fs, ds_fs, f_type=f_type)
      x = df[i]

      plt.plot(x)

      tname = j.split("/")
      name = tname[1]
      # x.plot()
      legendNames.append(name + ", " + i)

    plt.xlabel(f'Time, {ds_fs}', size=size)
    plt.ylabel('Value', size=size)
    plt.xticks(fontsize = size)
    plt.yticks(fontsize = size)
    plt.title('Time, %s' % i, size=size)
    plt.legend(legendNames, prop={'size': size})
    plt.grid()
    # plt.show()


''' FFT PLOTTING '''
def plotFFT(sets, vars, fs, ds_fs, f_type="fir", size=20):
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
        df = downsample(og_df, fs, ds_fs, f_type)
      x = df[i]

      x_yf, x_xf, x_size = getFFT(x)
      plt.plot(x_xf, 2.0/x_size*np.abs(x_yf[0:x_size//2]))

    plt.xlabel(f'Frequency (Hz), {fs/ds_fs}', size=size)
    plt.ylabel('Amplitude', size=size)
    plt.xticks(fontsize = size)
    plt.yticks(fontsize = size)
    plt.title('FFT, %s' % j, size=size)
    plt.grid()
      

''' WELCH PLOTTING '''
def plotWelch(sets, vars, fs, ds_fs, f_type="fir", size=20):
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
        df = downsample(og_df, fs, ds_fs, f_type)
      x = df[i]

      freq, psd = getWelch(x, ds_fs, filtering, omega_n, order)
      plt.semilogy(freq, psd)  # Log scale for better visibility
      tname = j.split("/")
      name = tname[2]
      legendNames.append(name + ", " + i)
    
    plt.grid()
    plt.xlabel(f'Frequency (Hz), {ds_fs} Hz signal', size=size)
    plt.ylabel('Power Spectral Density', size=size)
    plt.xticks(fontsize = size)
    plt.yticks(fontsize = size)
    plt.title('Welch PSD', size=size)
    plt.legend(legendNames, loc='upper right', prop={'size': size})
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

f_type = "fir"

# normalizeSets(datasets)

ds_fs = 800
plotTime(datasets, variables, fs, ds_fs, size=size)
# plotFFT(datasets, variables, fs, ds_fs, size=size)
plotWelch(datasets, variables, fs, ds_fs, size=size)  

ds_fs = 400
plotTime(datasets, variables, fs, ds_fs, size=size)
# plotFFT(datasets, variables, fs, ds_fs, size=size)
plotWelch(datasets, variables, fs, ds_fs, size=size)

# f_type = "iir"
# ds_fs = 200
# plotTime(datasets, variables, fs, ds_fs, f_type = "iir", size=size)
# plotFFT(datasets, variables, fs, ds_fs, f_type = "iir", size=size)
# plotWelch(datasets, variables, fs, ds_fs, f_type = "iir", size=size)

datasets = []
datasets.append("800v200hz/200hz.txt")

fs = 200
ds_fs = 200
#plotTime(datasets, variables, fs, ds_fs, size=size)
# plotFFT(datasets, variables, fs, ds_fs, size=size)
# plotWelch(datasets, variables, fs, ds_fs, size=size)  



plt.show()


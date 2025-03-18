import matplotlib.pyplot as plt
import scipy
from scipy.signal import butter, lfilter, freqz, welch
from scipy.fft import fft, ifft, fftfreq
from scipy import stats
# from scipy import signal
from math import log, e
import numpy as np
import pandas as pd

def get_Freq_Domain_features_of_signal(signal, signal_name, Fs):
    
    features = {} 

    suffix = ""
    if signal_name:
        suffix = f"_{signal_name}"
    
    signal_t = np.transpose(signal)
    
    # Compute PSD via Welch algorithm
    
    freq, psd = getWelch(signal_t, Fs, True, 15, 3)
    
    # if(plot):
    #     plt.semilogy(freq, psd)  # Log scale for better visibility
    #     plt.xlabel('Frequency (Hz)')
    #     plt.ylabel('Power Spectral Density')
    #     plt.title('Welch PSD')
    #     plt.figure()

    # Convert to [ug / sqrt(Hz)]
    psd = np.sqrt(psd) #* accel2ug

    # Compute noise spectral densities        
    ndaz = np.mean(psd)


    features[f'psd_mean{suffix}']           = psd.mean()   
    features[f'psd_max{suffix}']            = psd.max()
    features[f'psd_min{suffix}']            = psd.min()
    features[f'psd_max(Hz)_{suffix}']       = freq[np.argmax(psd)]
    # features[f'Hz_mean{suffix}']          = freq.mean()  # This does not make sense

    return features

# Make FFT of given file, with given feature 
def getFFT(file, feature):
  df = pd.read_csv(file+".csv")
  x = df[feature]
  x_size = len(x)
  x_space = 1/800

  x_yf = fft(x)
  x_xf = fftfreq(x_size, x_space)[:x_size//2]
  return x_yf, x_xf, x_size

# x = signal
def getWelch(x, fs, filterOn, omega_n, order):

    if filterOn:
        signal = butter_highpass_filter(x, fs, omega_n, order)
    else:
        signal = x

    freq, psd = welch(signal, fs, nperseg=1024, scaling='density')

    return freq, psd

def butter_highpass_filter(data, fs, omega_n, order):

    b, a = butter(order, Wn=omega_n, fs=fs, btype='high', analog=False)
    result = lfilter(b, a, data)

    return result
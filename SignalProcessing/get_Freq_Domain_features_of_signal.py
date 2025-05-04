import matplotlib.pyplot as plt
import scipy
from typing import List, Dict, Any, Tuple, Sequence, Optional
from scipy.signal import butter, lfilter, freqz, welch
from scipy.fft import fft, ifft, fftfreq
from scipy import stats
# from scipy import signal
from math import log, e
import numpy as np
import pandas as pd
from pandas import Series

def get_Freq_Domain_features_of_signal(signal:      Series, 
                                       signal_name: str,
                                       Fs:          int
                                       ) -> dict:
    
    '''Calculates frequency domain features from a signal's spectral density.

    Computes the Power Spectral Density (PSD) using the `getWelch` helper
    function (which wraps `scipy.signal.welch`) with predefined
    filter settings and Welch parameters. It then calculates the square root
    of the PSD (related to Amplitude Spectral Density) and extracts features
    such as the mean, max, min spectral density magnitude, and the frequency
    at which the maximum magnitude occurs.'''

    features = {} 

    suffix = ""
    if signal_name:
        suffix = f"_{signal_name}"
    
    # signal_t = np.transpose(signal)
    
    # Compute PSD via Welch algorithm
    freq, psd = getWelch(signal, Fs, True, 15.0, 3)

    # plt.semilogy(freq, psd)  # Log scale for better visibility
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power Spectral Density')
    # plt.title('Welch PSD {suffix}')
    # plt.figure()

    # Convert to [ug / sqrt(Hz)]
    psd = np.sqrt(psd) #* accel2ug

    # Compute noise spectral densities        
    # ndaz = np.mean(psd)

    features[f'psd_mean{suffix}']           = psd.mean()   
    features[f'psd_max{suffix}']            = psd.max()
    features[f'psd_min{suffix}']            = psd.min()
    features[f'psd_max(Hz)_{suffix}']       = freq[np.argmax(psd)]
    # features[f'Hz_mean{suffix}']          = freq.mean()  # This does not make sense

    return features

def getFFT(signal: Series
           ) -> Any:
  
  # Make FFT of given file, with given feature  
  x_size = len(signal)
  x_space = 1/800

  x_yf = fft(signal)
  x_xf = fftfreq(x_size, x_space)[:x_size//2]

  return x_yf, x_xf, x_size

def getWelch(x:         Series, # signal
             fs:        int, 
             filter_on: bool, 
             omega_n:   float, 
             order:     int
             ) -> Tuple[np.ndarray, np.ndarray]:

    if filter_on:
        signal = butter_highpass_filter(x, fs, omega_n, order)
    else:
        signal = x

    #TODO: Compare performance to different nperseg and noverlap
    # ish 512 sec with nperseg = 1024
    # ish 400 sec with nperseg = 256
    # ish 486 sec with nperseg = 128, slightly worse results
    freq, psd = welch(signal, fs=fs, nperseg=256, noverlap=0, scaling='density')

    return freq, psd

def butter_highpass_filter(data:    Series, 
                           fs:      int, 
                           omega_n: float,
                           order:   int
                           ) -> np.ndarray:
      
    '''Applies a high-pass Butterworth filter to the input data.'''
    
    b, a = butter(order, Wn=omega_n, fs=fs, btype='high', analog=False)
    result = lfilter(b, a, data)

    return result
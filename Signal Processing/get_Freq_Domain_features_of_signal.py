import matplotlib.pyplot as plt
import scipy
from scipy.signal import butter, lfilter, freqz, welch
from scipy.fft import fft, ifft
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
    freq, psd = welch(signal_t, Fs, nperseg=1024, scaling='density')  # az
    
    # Convert to [ug / sqrt(Hz)]
    psd = np.sqrt(psd) #* accel2ug

    # Compute noise spectral densities        
    ndaz = np.mean(psd)


    features[f'psd_mean{suffix}']           = psd.mean()   
    features[f'psd_max{suffix}']            = psd.max()
    features[f'psd_min{suffix}']            = psd.min()
    features[f'psd_max(Hz)_{suffix}']       = freq[np.argmax(psd)]
    # features[f'Hz_mean{suffix}']            = freq.mean()  # This does not make sense

    return features


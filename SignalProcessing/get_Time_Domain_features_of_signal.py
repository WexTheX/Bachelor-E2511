import matplotlib.pyplot as plt
import scipy
from scipy.signal import butter, lfilter, freqz, welch
from scipy.fft import fft, ifft
from scipy import stats
# from scipy import signal
from math import log, e
import numpy as np
import pandas as pd

def get_Time_Domain_features_of_signal(signal,signal_name):
    features = {} 
    signal = butter(1, 100, fs=800, btype='low', analog=False)

    suffix = ""
    if signal_name:
        suffix = f"_{signal_name}"
    
    features[f'mean{suffix}']       = signal.mean()   
    features[f'sd{suffix}']         = signal.std()
    features[f'mad{suffix}']        = stats.median_abs_deviation(signal, scale=1/1.4826)
    features[f'max{suffix}']        = signal.max()
    features[f'min{suffix}']        = signal.min()
    features[f'energy{suffix}']     = sum(pow(abs(signal), 2)) # https://stackoverflow.com/questions/34133680/calculate-energy-of-time-domain-data
    features[f'entropy{suffix}']    = entropy(signal) # https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
    features[f'iqr{suffix}']        = interquartile_range(signal) # https://www.statology.org/interquartile-range-python/
    features[f'kurtosis{suffix}']   = stats.kurtosis(signal, fisher=True)
    features[f'skewness{suffix}']   = stats.skew(signal)
    # features['auto_regression_coefficient']   = 1 #TODO auto_regression_coefficient https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
    # features['simple_moving_average_x']       = simple_moving_average(x_features, moving_average_window)
    # features['correlation']                   = 1 #TODO correlation https://realpython.com/numpy-scipy-pandas-correlation-python/#example-numpy-correlation-calculation
    # features['angular_velocity']              = 1 #TODO angular velocity | how to get? 
    # features['linear_acceleration']           = 1 # for Norm data this is already there

    return features

def median_filter(array):
    """returns median filtered signal"""
    return scipy.ndimage.median_filter(array, size=10)

def butter_lowpass_with_cutoff(cutoff, fs, order=3):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter_with_cutoff(data, cutoff, fs, order=3):
    # TODO check the filters
    # For cutoff frequency https://www.electrical4u.com/cutoff-frequency/
    b, a = butter_lowpass_with_cutoff(cutoff, fs, order=order)
    result = lfilter(b, a, data)
    return result

def butter_lowpass(fs, order=3):
    return butter(order, fs=fs, btype='low', analog=False, Wn=15) # If fs is specified, Wn is in the same units as fs.

def butter_lowpass_filter(data, fs, order=3):
    b, a = butter_lowpass(fs, order=order)
    result = lfilter(b, a, data)
    return result

def entropy(labels, base=None):
    """ Computes entropy of label distribution. """
    #https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent


def interquartile_range(signal):
    """finds the interquartile range of a 1d numpy array"""
    # https://www.statology.org/interquartile-range-python/

    q3, q1 = np.percentile(signal, [75, 25])
    iqr = q3 - q1
    return iqr


def simple_moving_average(signal, window_size):
    """calculates the moving"""
    # https://www.investopedia.com/terms/s/sma.asp
    # https://www.geeksforgeeks.org/how-to-calculate-moving-averages-in-python/
    i = 0
    moving_averages = []

    while i < len(signal) - window_size + 1:
        window_average = round(np.sum(
            signal[i:i+window_size]) / window_size, 2)
        
        moving_averages.append(window_average)

        i += 1

    return moving_averages

#navn = "Vinkelslipertest1"
#signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
#features = get_Time_Domain_features_of_signal(signal, navn)
#print(features)
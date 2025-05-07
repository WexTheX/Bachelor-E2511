import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas as pd

from scipy.signal import butter, lfilter, freqz, welch
from scipy.fft import fft, ifft
from scipy import stats
from math import log, e
from pandas import Series
from typing import List, Dict, Any, Tuple, Sequence, Optional


def get_Time_Domain_features_of_signal(signal:      Series,
                                       signal_name: str
                                       ) -> dict:
    features = {} 
    # signal = butter(1, 100, fs=800, btype='low', analog=False)

    suffix = ""
    if signal_name:
        suffix = f"_{signal_name}"
    
    features[f'mean{suffix}']           = signal.mean()   
    features[f'sd{suffix}']             = signal.std()
    features[f'mad{suffix}']            = stats.median_abs_deviation(signal, scale=1/1.4826)
    features[f'max{suffix}']            = signal.max()
    features[f'min{suffix}']            = signal.min()
    features[f'energy{suffix}']         = sum(pow(abs(signal), 2)) # https://stackoverflow.com/questions/34133680/calculate-energy-of-time-domain-data
    features[f'entropy{suffix}']        = entropy(signal) # https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
    features[f'iqr{suffix}']            = interquartile_range(signal) # https://www.statology.org/interquartile-range-python/
    features[f'kurtosis{suffix}']       = stats.kurtoKursis(signal, fisher=True)
    features[f'skewness{suffix}']       = stats.skew(signal)
    features[f'correlation{suffix}']    = autocorrelation(signal) # https://realpython.com/numpy-scipy-pandas-correlation-python/#example-numpy-correlation-calculation

    # features['auto_regression_coefficient']   = 1 #TODO auto_regression_coefficient https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
    # features['simple_moving_average_x']       = simple_moving_average(x_features, moving_average_window)
    # features['angular_velocity']              = 1 #TODO angular velocity | how to get? 
    # features['linear_acceleration']           = 1 # for Norm data this is already there
    # Number/Percentage of Outliers: Defined by a rule (e.g., > Q3 + 1.5IQR or < Q1 - 1.5IQR). ????
    # Percentage range???? IQR for 90-10 etc
    
    return features

def median_filter(array:    pd.Series,
                  size:     int = 10
                  ):
    
    """returns median filtered signal"""
    return scipy.ndimage.median_filter(array, size=size)

def butter_lowpass_with_cutoff(cutoff:  float,
                               fs:      int,
                               order:   int = 3,
                               btype:   str ='low'):
    
    return butter(order, cutoff, fs=fs, btype=btype, analog=False)

def butter_lowpass_filter_with_cutoff(data,
                                      cutoff,
                                      fs,
                                      order=3):

    # TODO check the filters
    # For cutoff frequency https://www.electrical4u.com/cutoff-frequency/
    b, a = butter_lowpass_with_cutoff(cutoff, fs, order=order)
    result = lfilter(b, a, data)

    return result

def butter_lowpass(fs,
                   order=3
                   ):
    
    return butter(order, fs=fs, btype='low', analog=False, Wn=15) # If fs is specified, Wn is in the same units as fs.

def butter_lowpass_filter(data,
                          fs, 
                          order=3
                          ):

    b, a = butter_lowpass(fs, order=order)
    result = lfilter(b, a, data)

    return result

def entropy(labels, 
            base=None
            ):
    
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

def interquartile_range(signal: pd.Series
                        ) -> float:

    """finds the interquartile range of a 1d numpy array"""
    # https://www.statology.org/interquartile-range-python/

    q3, q1 = np.percentile(signal, [75, 25])
    iqr = q3 - q1
    return iqr

def percentage_range(signal:            pd.Series,
                     lower_percentage:  int = 10,
                     upper_percentage:  int = 90
                     ) -> float:
    
    q_upper, q_lower = np.percentile(signal, [lower_percentage, upper_percentage])
    percentage_range = q_upper - q_lower
    
    return percentage_range

def simple_moving_average(signal, 
                          window_size
                          ):
    
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

def autocorrelation(signal: pd.Series
                    ) -> float:
    
    dataframe = pd.concat([signal.shift(1), signal], axis=1)
    dataframe.columns = ['t-1', 't+1']
    result_matrix = dataframe.corr()

    result = result_matrix.iloc[0,1]

    return result


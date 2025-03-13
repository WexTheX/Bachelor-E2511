import pandas as pd
import numpy as np
import scipy.io
import scipy
from scipy import stats
from math import log, e
import matplotlib.pyplot as plt

def get_features_from_hr_signal(signal):

    signal = np.array(signal)
    
    # signal = signal[~np.isnan(signal)]

    results = {}

    results['hr_mean']          = signal.mean()
    results['hr_min']           = signal.min()
    results['hr_max']           = signal.max()
    results['hr_sd']            = signal.std()
    results['hr_energy']        = sum(pow(abs(signal), 2))
    results['hr_entropy']       = entropy(signal)
    results['hr_iqr']           = interquartile_range(signal)
    results['hr_kurtosis']      = stats.kurtosis(signal, fisher=True)
    results['hr_skewness']      = stats.skew(signal)
    results['hr_mad']           = stats.median_abs_deviation(signal, scale='normal') # new
    # results['hr_mad']           = stats.median_abs_deviation(signal, scale=1/1.4826) # new
        
    return results

def Extract_HR_Features(HR_data, WindowLength_HR, PeakHR):
   
    all_window_features = []
    
    # Calculate the number of windows
    num_samples = len(HR_data)
    num_windows = num_samples // WindowLength_HR
    # print(f"Number of HR windows: {num_windows}")
    
    # Loop over windows
    for i in range(num_windows):
        # Define the start and end index for the window
        start_idx = i * WindowLength_HR
        end_idx = start_idx + WindowLength_HR

        # Extract the window of HR data
        window_HR_data = HR_data[start_idx:end_idx]

        # Compute features for the window
        windowed_features = get_features_from_hr_signal(window_HR_data)

        # Calculate the percentage of peak HR
        windowed_features['PeakHR_percentage']= (windowed_features['hr_max'] / PeakHR) * 100

        # Append the features of the current window to the list
        all_window_features.append(windowed_features)
    
    # Convert the list of features into a DataFrame for easy manipulation
    HR_feature_df = pd.DataFrame(all_window_features)

    return HR_feature_df


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
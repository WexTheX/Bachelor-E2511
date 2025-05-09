import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Sequence, Optional

# SignalProcessing part is needed when this file is imported in main.py
from SignalProcessing.get_Time_Domain_features_of_signal import get_Time_Domain_features_of_signal
from SignalProcessing.get_Freq_Domain_features_of_signal import get_Freq_Domain_features_of_signal
from preprocessing import tab_txt_to_csv, delete_header, rename_data, compare_bin_and_txt

# Changed name for clarity
# Still based on Roya's "ExtractIMU_Features"
import time
def extractDFfromFile(file_path:    str,
                      fs:           int,
                      drop_index:   bool = True
                      ) -> pd.DataFrame:

    # try:
    #   df = pd.read_csv(file_path+".txt", delimiter="\t")

    # except pd.errors.ParserError as pe:
    #   try:
    #     # print("Kom inni delete header")
    #     delete_header(file_path + ".txt")
    #     df = pd.read_csv(file_path+".txt", delimiter="\t")
    #   except Exception as e:
    #     print(f"Error in delete_header in file {file_path}: {e}")

    # except Exception as e:
    #   print(f"Error in read_csv in file {file_path}: {e}")
    
    # delete_header(file_path + ".txt")
    df = pd.read_csv(file_path+".txt", delimiter="\t")

    ''' REMOVE 10 SECONDS '''
    if drop_index == True:
        df.drop(df.index[:fs*10]) # Drop everything before 10 seconds
        df.drop(df.index[fs*10:]) # Drop everything after 10 seconds

    return df

def extractFeaturesFromDF(df:                   pd.DataFrame,
                          df_label:             Sequence,
                          window_length_sec:    int,
                          fs:                   int,
                          norm_IMU:             bool
                          ) -> Tuple[List[Dict[str, Any]], Sequence]:

    '''Extracts time and frequency domain features from fixed-length, non-overlapping windows.

    Segments the input DataFrame into windows based on the specified duration
    and sampling frequency. For each window, it calculates a set of statistical
    and spectral features. The feature calculation strategy depends on the
    `norm_IMU` flag:
    - If True: Calculates features on the Euclidean norm of Accelerometer,
      Gyroscope, and Magnetometer signals, plus the Temperature signal.
    - If False: Calculates features on each individual axis (X, Y, Z) for
      Accel, Gyro, Mag, plus the Temperature signal.'''
    
    window_labels = []
    all_window_features = []

    ''' EXTRACT COLUMNS '''
    time_data   = df["Timestamp"]  # Timedata, only used to measure amount of samples

    gyro_X      = df["Gyr.X"]  # Gyroscope data in the X direction
    gyro_Y      = df["Gyr.Y"]  # Gyroscope data in the Y direction
    gyro_Z      = df["Gyr.Z"]  # Gyroscope data in the Z direction

    accel_X     = df["Axl.X"]  # Accelerometer data in the X direction
    accel_Y     = df["Axl.Y"]  # Accelerometer data in the Y direction
    accel_Z     = df["Axl.Z"]  # Accelerometer data in the Z direction

    mag_X       = df["Mag.X"]  # Magnetometer data in the X direction
    mag_Y       = df["Mag.Y"]  # Magnetometer data in the Y direction
    mag_Z       = df["Mag.Z"]  # Magnetometer data in the Z direction

    temp        = df["Temp"]   # Ambient temperature (°C)
    press   = df["Press"]  # Air pressure (Pa or hPa)

    # range   = df["Range"]  # Distance to object (meters)
    # lum     = df["Lum"]    # Light intensity (lux)
    # IR_lum  = df["IRLum"]  # Infrared light intensity
        
    ''' COUNT AND LOAD WINDOWS '''
    num_samples = len(time_data) # Number of measurements
    window_length = window_length_sec * fs
    num_windows = num_samples // window_length # Rounds down when deciding numbers

    # Iterate through each window
    for j in range(0, num_windows):
        
        # Define the start and end index for the window
        start_idx = j * window_length
        end_idx = start_idx + window_length
        # print(f"Getting features from window {start_idx} to {end_idx}")  
                
        window_temp = temp[start_idx:end_idx]

        if norm_IMU == True:

            # Normalize IMU (calculate only once)
            if j == 0:
                norm_acceleration   = np.sqrt( np.power(accel_X, 2) + np.power(accel_Y, 2) + np.power(accel_Z, 2) )
                norm_gyro           = np.sqrt( np.power(gyro_X, 2) + np.power(gyro_Y, 2) + np.power(gyro_Z, 2) )
                norm_mag            = np.sqrt( np.power(mag_X, 2) + np.power(mag_Y, 2) + np.power(mag_Z, 2) )

            # g_constant = np.mean(norm_acceleration)
            # print(f"g constant: {g_constant}")
            # gravless_norm = np.subtract(norm_acceleration, g_constant)  

            window_accel_Norm   = norm_acceleration[start_idx:end_idx]
            window_gyro_Norm    = norm_gyro[start_idx:end_idx]
            window_mag_Norm     = norm_mag[start_idx:end_idx]

            # Get temporal features (mean, std, MAD, etc as datatype dict)
            # 4 columns, 11 Time domain features each = 44 elements
            window_features_accel_Norm_Time = get_Time_Domain_features_of_signal(window_accel_Norm, "accel_Norm")
            window_features_gyro_Norm_Time  = get_Time_Domain_features_of_signal(window_gyro_Norm, "gyro_Norm")
            window_features_mag_Norm_Time   = get_Time_Domain_features_of_signal(window_mag_Norm, "mag_Norm")
            window_features_temp_Time       = get_Time_Domain_features_of_signal(window_temp, "temp")

            # Get frequency features (Welch's method)
            # 3 columns, 4 Frequency domain features each = 12 elements
            window_features_accel_Norm_Freq = get_Freq_Domain_features_of_signal(window_accel_Norm, "accel_Norm", fs)
            window_features_gyro_Norm_Freq  = get_Freq_Domain_features_of_signal(window_gyro_Norm, "gyro_Norm", fs)
            window_features_mag_Norm_Freq   = get_Freq_Domain_features_of_signal(window_mag_Norm, "mag_Norm", fs)
        
            
            ## merge all
            window_features = {
                            **window_features_accel_Norm_Time,
                            **window_features_gyro_Norm_Time,
                            **window_features_mag_Norm_Time,
                            **window_features_temp_Time,

                            **window_features_accel_Norm_Freq,
                            **window_features_gyro_Norm_Freq,
                            **window_features_mag_Norm_Freq
                            }


        if norm_IMU == False:
            
            # Windowing the signals
            window_accel_X = accel_X[start_idx:end_idx]
            window_accel_Y = accel_Y[start_idx:end_idx]
            window_accel_Z = accel_Z[start_idx:end_idx]

            window_gyro_X  = gyro_X[start_idx:end_idx]
            window_gyro_Y  = gyro_Y[start_idx:end_idx]
            window_gyro_Z  = gyro_Z[start_idx:end_idx]

            window_mag_X   = mag_X[start_idx:end_idx]
            window_mag_Y   = mag_Y[start_idx:end_idx]
            window_mag_Z   = mag_Z[start_idx:end_idx]


            # Get temporal features (mean, std, MAD, etc as datatype dict)
            # 10 columns, 11 Time domain features each = 110 elements
            window_features_accel_X_Time    = get_Time_Domain_features_of_signal(window_accel_X, "accel_X")
            window_features_accel_Y_Time    = get_Time_Domain_features_of_signal(window_accel_Y, "accel_Y")
            window_features_accel_Z_Time    = get_Time_Domain_features_of_signal(window_accel_Z, "accel_Z")
            window_features_gyro_X_Time     = get_Time_Domain_features_of_signal(window_gyro_X, "gyro_X")
            window_features_gyro_Y_Time     = get_Time_Domain_features_of_signal(window_gyro_Y, "gyro_Y")
            window_features_gyro_Z_Time     = get_Time_Domain_features_of_signal(window_gyro_Z, "gyro_Z")
            window_features_mag_X_Time      = get_Time_Domain_features_of_signal(window_mag_X, "mag_X")
            window_features_mag_Y_Time      = get_Time_Domain_features_of_signal(window_mag_Y, "mag_Y")
            window_features_mag_Z_Time      = get_Time_Domain_features_of_signal(window_mag_Z, "mag_Z")
            window_features_temp_Time       = get_Time_Domain_features_of_signal(window_temp, "temp")
            # window_features_press_Time       = get_Time_Domain_features_of_signal(window_temp, "press")
            
            # Get frequency features from Welch's method
            # 9 columns, 4 Frequency domain features each = 36 elements
            window_features_accel_X_Freq    = get_Freq_Domain_features_of_signal(window_accel_X, "accel_X", fs)
            window_features_accel_Y_Freq    = get_Freq_Domain_features_of_signal(window_accel_Y, "accel_Y", fs)
            window_features_accel_Z_Freq    = get_Freq_Domain_features_of_signal(window_accel_Z, "accel_Z", fs)
            window_features_gyro_X_Freq     = get_Freq_Domain_features_of_signal(window_gyro_X, "gyro_X", fs)
            window_features_gyro_Y_Freq     = get_Freq_Domain_features_of_signal(window_gyro_Y, "gyro_Y", fs)
            window_features_gyro_Z_Freq     = get_Freq_Domain_features_of_signal(window_gyro_Z, "gyro_Z", fs)
            window_features_mag_X_Freq      = get_Freq_Domain_features_of_signal(window_mag_X, "mag_X", fs)
            window_features_mag_Y_Freq      = get_Freq_Domain_features_of_signal(window_mag_Y, "mag_Y", fs)
            window_features_mag_Z_Freq      = get_Freq_Domain_features_of_signal(window_mag_Z, "mag_Z", fs)


            # Merge all
            window_features = {
                            **window_features_accel_X_Time, 
                            **window_features_accel_Y_Time,
                            **window_features_accel_Z_Time,
                            **window_features_gyro_X_Time,
                            **window_features_gyro_Y_Time,
                            **window_features_gyro_Z_Time,
                            **window_features_mag_X_Time,
                            **window_features_mag_Y_Time,
                            **window_features_mag_Z_Time,
                            **window_features_temp_Time,
                            # **window_features_press_Time,

                            **window_features_accel_X_Freq,
                            **window_features_accel_Y_Freq,
                            **window_features_accel_Z_Freq,
                            **window_features_gyro_X_Freq,
                            **window_features_gyro_Y_Freq,
                            **window_features_gyro_Z_Freq,
                            **window_features_mag_X_Freq,
                            **window_features_mag_Y_Freq,
                            **window_features_mag_Z_Freq
                            }

        # Append the features of the current window to the list
        all_window_features.append(window_features)
        window_labels.append(df_label)

    return all_window_features, window_labels
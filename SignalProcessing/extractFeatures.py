import numpy as np
import pandas as pd
# SignalProcessing part is needed when this file is imported in main.py
from SignalProcessing.get_Time_Domain_features_of_signal import get_Time_Domain_features_of_signal
from SignalProcessing.get_Freq_Domain_features_of_signal import get_Freq_Domain_features_of_signal
from Preprocessing.preprocessing import tab_txt_to_csv
from Preprocessing.preprocessing import delete_header
from Preprocessing.preprocessing import rename_data

# Changed name for clarity
# Still based on Roya's "ExtractIMU_Features"

def Extract_All_Features(datasets, WindowLength, Norm_Accel, Fs):
    
    features_df = []
    
    # Renames data inside Datafiles/xxx folder
    rename_data()

    for i in datasets:
        

        delete_header(i + ".txt") # Deletes lines before Timestamp and does some regex
        tab_txt_to_csv(i + ".txt", i + ".csv") # Converts from .txt to .csv

        df = pd.read_csv(i+".csv")

        # 1, 2, 4, 5, or 10 data points must be selected
        time_data = df["Timestamp"]  # Assuming 1st column is time
        
        gyro_X  = df["Gyr.X"]  # Gyroscope data in the X direction
        gyro_Y  = df["Gyr.Y"]  # Gyroscope data in the Y direction
        gyro_Z  = df["Gyr.Z"]  # Gyroscope data in the Z direction

        accel_X = df["Axl.X"]  # Accelerometer data in the X direction
        accel_Y = df["Axl.Y"]  # Accelerometer data in the Y direction
        accel_Z = df["Axl.Z"]  # Accelerometer data in the Z direction

        mag_X   = df["Mag.X"]  # Magnetometer data in the X direction
        mag_Y   = df["Mag.Y"]  # Magnetometer data in the Y direction
        mag_Z   = df["Mag.Z"]  # Magnetometer data in the Z direction

        temp    = df["Temp"]   # Ambient temperature (°C)
        # press   = df["Press"]  # Air pressure (Pa or hPa)

        # range   = df["Range"]  # Distance to object (meters)
        # lum     = df["Lum"]    # Light intensity (lux)
        # IR_lum  = df["IRLum"]  # Infrared light intensity

        # Define a list to store features for each window
        all_window_features = []

        # Calculate the number of windows
        num_samples = len(time_data) # Number of measurements
        num_windows = num_samples // WindowLength
        # 42 = 84 322 // 2 000

        # Trying to remove first and last 10 seconds from the IMU_data sets
        # To avoid time wasted during start and stop of tests

        num_windows_cut = (10 * Fs) // WindowLength
        # print(num_windows_cut)
        # 4 = 10 * 800 // 2 000

        print(f"Number of IMU windows after cut: {num_windows - 2 * num_windows_cut}")
        
        # Only does feature extraction on windows in the middle
        for i in range(num_windows_cut, num_windows - num_windows_cut):

            # Define the start and end index for the window
            start_idx = i * WindowLength
            end_idx = start_idx + WindowLength
            print(start_idx, end_idx)  
                    
            # Windowing the signals
            window_gyro_X  = gyro_X[start_idx:end_idx]
            window_gyro_Y  = gyro_Y[start_idx:end_idx]
            window_gyro_Z  = gyro_Z[start_idx:end_idx]

            window_mag_X  = mag_X[start_idx:end_idx]
            window_mag_Y  = mag_Y[start_idx:end_idx]
            window_mag_Z  = mag_Z[start_idx:end_idx]

            window_temp = temp[start_idx:end_idx]

            if Norm_Accel == True:
                # Normalize acceleration
                norm_acceleration = np.sqrt( np.power(accel_X, 2) + np.power(accel_Y, 2) + np.power(accel_Z, 2))

                g_constant = np.mean(norm_acceleration)
                # print(f"g constant: {g_constant}")
                gravless_norm = np.subtract(norm_acceleration, g_constant)  
                window_accel_Norm = gravless_norm[start_idx:end_idx]

                # Get temporal features (mean, std, MAD, etc as datatype dict)
                window_features_accel_Norm_Time = get_Time_Domain_features_of_signal(window_accel_Norm, "accel_Norm")
                window_features_gyro_X_Time     = get_Time_Domain_features_of_signal(window_gyro_X, "gyro_X")
                window_features_gyro_Y_Time     = get_Time_Domain_features_of_signal(window_gyro_Y, "gyro_Y")
                window_features_gyro_Z_Time     = get_Time_Domain_features_of_signal(window_gyro_Z, "gyro_Z")
                window_features_mag_X_Time      = get_Time_Domain_features_of_signal(window_mag_X, "mag_X")
                window_features_mag_Y_Time      = get_Time_Domain_features_of_signal(window_mag_Y, "mag_Y")
                window_features_mag_Z_Time      = get_Time_Domain_features_of_signal(window_mag_Z, "mag_Z")
                window_features_temp_Time       = get_Time_Domain_features_of_signal(window_temp, "temp")

                # Get frequency features (Welch's method)
                window_features_accel_Norm_Freq = get_Freq_Domain_features_of_signal(window_accel_Norm, "accel_Norm", Fs)
                window_features_gyro_X_Freq     = get_Freq_Domain_features_of_signal(window_gyro_X, "gyro_X", Fs)
                window_features_gyro_Y_Freq     = get_Freq_Domain_features_of_signal(window_gyro_Y, "gyro_Y", Fs)
                window_features_gyro_Z_Freq     = get_Freq_Domain_features_of_signal(window_gyro_Z, "gyro_Z", Fs)
                window_features_mag_X_Freq      = get_Freq_Domain_features_of_signal(window_mag_X, "mag_X", Fs)
                window_features_mag_Y_Freq      = get_Freq_Domain_features_of_signal(window_mag_Y, "mag_Y", Fs)
                window_features_mag_Z_Freq      = get_Freq_Domain_features_of_signal(window_mag_Z, "mag_Z", Fs)
                window_features_temp_Freq       = get_Freq_Domain_features_of_signal(window_temp, "temp", Fs)

                ## merge all
                window_features = {
                                **window_features_accel_Norm_Time, 
                                **window_features_accel_Norm_Freq,

                                **window_features_gyro_X_Time,
                                **window_features_gyro_Y_Time,
                                **window_features_gyro_Z_Time,
                                **window_features_mag_X_Time,
                                **window_features_mag_Y_Time,
                                **window_features_mag_Z_Time,
                                **window_features_temp_Time,
                        
                                **window_features_gyro_X_Freq,
                                **window_features_gyro_Y_Freq,
                                **window_features_gyro_Z_Freq,
                                **window_features_mag_X_Freq,
                                **window_features_mag_Y_Freq,
                                **window_features_mag_Z_Freq,
                                **window_features_temp_Freq}
                        

            if Norm_Accel == False:

                window_accel_X = accel_X[start_idx:end_idx]
                window_accel_Y = accel_Y[start_idx:end_idx]
                window_accel_Z = accel_Z[start_idx:end_idx]

                # Get temporal features (mean, std, MAD, etc as datatype dict)
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
                
                # Get frequency features from Welch's method
                window_features_accel_X_Freq    = get_Freq_Domain_features_of_signal(window_accel_X, "accel_X", Fs)
                window_features_accel_Y_Freq    = get_Freq_Domain_features_of_signal(window_accel_Y, "accel_Y", Fs)
                window_features_accel_Z_Freq    = get_Freq_Domain_features_of_signal(window_accel_Z, "accel_Z", Fs)
                window_features_gyro_X_Freq     = get_Freq_Domain_features_of_signal(window_gyro_X, "gyro_X", Fs)
                window_features_gyro_Y_Freq     = get_Freq_Domain_features_of_signal(window_gyro_Y, "gyro_Y", Fs)
                window_features_gyro_Z_Freq     = get_Freq_Domain_features_of_signal(window_gyro_Z, "gyro_Z", Fs)
                window_features_mag_X_Freq      = get_Freq_Domain_features_of_signal(window_mag_X, "mag_X", Fs)
                window_features_mag_Y_Freq      = get_Freq_Domain_features_of_signal(window_mag_Y, "mag_Y", Fs)
                window_features_mag_Z_Freq      = get_Freq_Domain_features_of_signal(window_mag_Z, "mag_Z", Fs)
                window_features_temp_Freq       = get_Freq_Domain_features_of_signal(window_temp, "temp", Fs)



                ## merge all
                window_features = {
                                **window_features_accel_X_Time, 
                                **window_features_accel_Y_Time,
                                **window_features_accel_Z_Time,
                                **window_features_accel_X_Freq,
                                **window_features_accel_Y_Freq,
                                **window_features_accel_Z_Freq,

                                **window_features_gyro_X_Time,
                                **window_features_gyro_Y_Time,
                                **window_features_gyro_Z_Time,
                                **window_features_gyro_X_Freq,
                                **window_features_gyro_Y_Freq,
                                **window_features_gyro_Z_Freq,
                                
                                **window_features_mag_X_Time,
                                **window_features_mag_Y_Time,
                                **window_features_mag_Z_Time,
                                **window_features_mag_X_Freq,
                                **window_features_mag_Y_Freq,
                                **window_features_mag_Z_Freq,

                                **window_features_temp_Time,
                                **window_features_temp_Freq}


            # Append the features of the current window to the list
            all_window_features.append(window_features)

        # Convert the list of features to a Pandas DataFrame for easy manipulation
        feature_df = pd.DataFrame(all_window_features)

    return feature_df
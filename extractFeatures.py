import numpy as np
import pandas as pd

# SignalProcessing part is needed when this file is imported in main.py
from SignalProcessing.get_Time_Domain_features_of_signal import get_Time_Domain_features_of_signal
from SignalProcessing.get_Freq_Domain_features_of_signal import get_Freq_Domain_features_of_signal
from Preprocessing.preprocessing import tab_txt_to_csv, delete_header, rename_data, compare_bin_and_txt

# Changed name for clarity
# Still based on Roya's "ExtractIMU_Features"

def extractAllFeatures(datasets, datasetsLabel, window_length_sec, fs, Norm_Accel):
    windowLabel = []
    all_window_features = []
    windowSum = 0
    window_length = window_length_sec * fs
    
    # Renames data inside Datafiles/xxx folder

    for i, name in enumerate(datasets):
        ''' PREPROCESS FILES '''
        delete_header(name + ".txt") # Deletes lines before Timestamp and does some regex
        tab_txt_to_csv(name + ".txt", name + ".csv") # Converts from .txt to .csv

        ''' LOAD .CSV DATAFILES '''
        df = pd.read_csv(name+".csv")

        ''' REMOVE 10 SECONDS '''
        df.drop(df.index[:fs*10]) # Drop everything before 10 seconds
        df.drop(df.index[fs*10:]) # Drop everything after 10 seconds

        # 1, 2, 4, 5, or 10 data points must be selected
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
        # press   = df["Press"]  # Air pressure (Pa or hPa)

        # range   = df["Range"]  # Distance to object (meters)
        # lum     = df["Lum"]    # Light intensity (lux)
        # IR_lum  = df["IRLum"]  # Infrared light intensity

        num_samples = len(time_data) # Number of measurements
        
        ''' LOAD WINDOWS '''
        num_windows = num_samples // window_length # Rounds down when deciding numbers
        print(f"Number of IMU windows in {name} after cut: {num_windows}")
        windowSum += num_windows
        
        # Only does feature extraction on windows in the middle
        for j in range(0, num_windows):
            
            # Define the start and end index for the window
            start_idx = j * window_length
            end_idx = start_idx + window_length
            # print(f"Getting features from window {start_idx} to {end_idx}")  
                    
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
                # 10 columns, 10 Time domain features each = 100 elements
                window_features_accel_Norm_Time = get_Time_Domain_features_of_signal(window_accel_Norm, "accel_Norm")
                window_features_gyro_X_Time     = get_Time_Domain_features_of_signal(window_gyro_X, "gyro_X")
                window_features_gyro_Y_Time     = get_Time_Domain_features_of_signal(window_gyro_Y, "gyro_Y")
                window_features_gyro_Z_Time     = get_Time_Domain_features_of_signal(window_gyro_Z, "gyro_Z")
                window_features_mag_X_Time      = get_Time_Domain_features_of_signal(window_mag_X, "mag_X")
                window_features_mag_Y_Time      = get_Time_Domain_features_of_signal(window_mag_Y, "mag_Y")
                window_features_mag_Z_Time      = get_Time_Domain_features_of_signal(window_mag_Z, "mag_Z")
                window_features_temp_Time       = get_Time_Domain_features_of_signal(window_temp, "temp")

                # Get frequency features (Welch's method)
                # 10 columns, 4 Frequency domain features each = 40 elements
                window_features_accel_Norm_Freq = get_Freq_Domain_features_of_signal(window_accel_Norm, "accel_Norm", fs)
                window_features_gyro_X_Freq     = get_Freq_Domain_features_of_signal(window_gyro_X, "gyro_X", fs)
                window_features_gyro_Y_Freq     = get_Freq_Domain_features_of_signal(window_gyro_Y, "gyro_Y", fs)
                window_features_gyro_Z_Freq     = get_Freq_Domain_features_of_signal(window_gyro_Z, "gyro_Z", fs)
                window_features_mag_X_Freq      = get_Freq_Domain_features_of_signal(window_mag_X, "mag_X", fs)
                window_features_mag_Y_Freq      = get_Freq_Domain_features_of_signal(window_mag_Y, "mag_Y", fs)
                window_features_mag_Z_Freq      = get_Freq_Domain_features_of_signal(window_mag_Z, "mag_Z", fs)
                # window_features_temp_Freq       = get_Freq_Domain_features_of_signal(window_temp, "temp", fs)


                ## merge all
                window_features = {
                                **window_features_accel_Norm_Time, 
                                **window_features_accel_Norm_Freq,

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
                                # **window_features_temp_Freq
                                }
                        
            if Norm_Accel == False:

                window_accel_X = accel_X[start_idx:end_idx]
                window_accel_Y = accel_Y[start_idx:end_idx]
                window_accel_Z = accel_Z[start_idx:end_idx]

                # Get temporal features (mean, std, MAD, etc as datatype dict)
                # 10 columns, 10 Time domain features each = 100 elements
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
                # 10 columns, 4 Frequency domain features each = 40 elements
                window_features_accel_X_Freq    = get_Freq_Domain_features_of_signal(window_accel_X, "accel_X", fs)
                window_features_accel_Y_Freq    = get_Freq_Domain_features_of_signal(window_accel_Y, "accel_Y", fs)
                window_features_accel_Z_Freq    = get_Freq_Domain_features_of_signal(window_accel_Z, "accel_Z", fs)
                window_features_gyro_X_Freq     = get_Freq_Domain_features_of_signal(window_gyro_X, "gyro_X", fs)
                window_features_gyro_Y_Freq     = get_Freq_Domain_features_of_signal(window_gyro_Y, "gyro_Y", fs)
                window_features_gyro_Z_Freq     = get_Freq_Domain_features_of_signal(window_gyro_Z, "gyro_Z", fs)
                window_features_mag_X_Freq      = get_Freq_Domain_features_of_signal(window_mag_X, "mag_X", fs)
                window_features_mag_Y_Freq      = get_Freq_Domain_features_of_signal(window_mag_Y, "mag_Y", fs)
                window_features_mag_Z_Freq      = get_Freq_Domain_features_of_signal(window_mag_Z, "mag_Z", fs)
                # window_features_temp_Freq       = get_Freq_Domain_features_of_signal(window_temp, "temp", fs)


                # Merge all
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

                                **window_features_temp_Time
                                # , **window_features_temp_Freq
                                }

            # Append the features of the current window to the list
            all_window_features.append(window_features)
            windowLabel.append(datasetsLabel[i])

        print(f"Total number on IMU windows: {windowSum}") 

    # Convert the list of features to a Pandas DataFrame for easy manipulation
    feature_df = pd.DataFrame(all_window_features)

    # print(f"Total number of windows: {activityWindowsCounter}")
    return feature_df, windowLabel

def extractDFfromFile(file, fs):
    ''' TURN FILE INTO CSV READABLE FORMAT '''
    delete_header(file + ".txt") # Deletes lines before Timestamp and does some regex
    tab_txt_to_csv(file + ".txt", file + ".csv") # Converts from .txt to .csv

    df = pd.read_csv(file+".csv")

    ''' REMOVE 10 SECONDS '''
    df.drop(df.index[:fs*10]) # Drop everything before 10 seconds
    df.drop(df.index[fs*10:]) # Drop everything after 10 seconds
    return df

def extractFeaturesFromDF(df, df_label, window_length_sec, fs, Norm_Accel):

    window_labels = []
    all_window_features = []
    window_sum = 0
    
    window_length = window_length_sec * fs

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
    # press   = df["Press"]  # Air pressure (Pa or hPa)

    # range   = df["Range"]  # Distance to object (meters)
    # lum     = df["Lum"]    # Light intensity (lux)
    # IR_lum  = df["IRLum"]  # Infrared light intensity

    ''' COUNT WINDOWS '''
    num_samples = len(time_data) # Number of measurements
        
    ''' LOAD WINDOWS '''
    num_windows = num_samples // window_length # Rounds down when deciding numbers
    window_sum += num_windows
        
    # Only does feature extraction on windows in the middle
    for j in range(0, num_windows):
        
        # Define the start and end index for the window
        start_idx = j * window_length
        end_idx = start_idx + window_length
        # print(f"Getting features from window {start_idx} to {end_idx}")  
                
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
            # 10 columns, 10 Time domain features each = 100 elements
            window_features_accel_Norm_Time = get_Time_Domain_features_of_signal(window_accel_Norm, "accel_Norm")
            window_features_gyro_X_Time     = get_Time_Domain_features_of_signal(window_gyro_X, "gyro_X")
            window_features_gyro_Y_Time     = get_Time_Domain_features_of_signal(window_gyro_Y, "gyro_Y")
            window_features_gyro_Z_Time     = get_Time_Domain_features_of_signal(window_gyro_Z, "gyro_Z")
            window_features_mag_X_Time      = get_Time_Domain_features_of_signal(window_mag_X, "mag_X")
            window_features_mag_Y_Time      = get_Time_Domain_features_of_signal(window_mag_Y, "mag_Y")
            window_features_mag_Z_Time      = get_Time_Domain_features_of_signal(window_mag_Z, "mag_Z")
            window_features_temp_Time       = get_Time_Domain_features_of_signal(window_temp, "temp")

            # Get frequency features (Welch's method)
            # 10 columns, 4 Frequency domain features each = 40 elements
            window_features_accel_Norm_Freq = get_Freq_Domain_features_of_signal(window_accel_Norm, "accel_Norm", fs)
            window_features_gyro_X_Freq     = get_Freq_Domain_features_of_signal(window_gyro_X, "gyro_X", fs)
            window_features_gyro_Y_Freq     = get_Freq_Domain_features_of_signal(window_gyro_Y, "gyro_Y", fs)
            window_features_gyro_Z_Freq     = get_Freq_Domain_features_of_signal(window_gyro_Z, "gyro_Z", fs)
            window_features_mag_X_Freq      = get_Freq_Domain_features_of_signal(window_mag_X, "mag_X", fs)
            window_features_mag_Y_Freq      = get_Freq_Domain_features_of_signal(window_mag_Y, "mag_Y", fs)
            window_features_mag_Z_Freq      = get_Freq_Domain_features_of_signal(window_mag_Z, "mag_Z", fs)
            # window_features_temp_Freq       = get_Freq_Domain_features_of_signal(window_temp, "temp", fs)


            ## merge all
            window_features = {
                            **window_features_accel_Norm_Time, 
                            **window_features_accel_Norm_Freq,

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
                            # **window_features_temp_Freq
                            }
                    
        if Norm_Accel == False:

            window_accel_X = accel_X[start_idx:end_idx]
            window_accel_Y = accel_Y[start_idx:end_idx]
            window_accel_Z = accel_Z[start_idx:end_idx]

            # Get temporal features (mean, std, MAD, etc as datatype dict)
            # 10 columns, 10 Time domain features each = 100 elements
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
            # 10 columns, 4 Frequency domain features each = 40 elements
            window_features_accel_X_Freq    = get_Freq_Domain_features_of_signal(window_accel_X, "accel_X", fs)
            window_features_accel_Y_Freq    = get_Freq_Domain_features_of_signal(window_accel_Y, "accel_Y", fs)
            window_features_accel_Z_Freq    = get_Freq_Domain_features_of_signal(window_accel_Z, "accel_Z", fs)
            window_features_gyro_X_Freq     = get_Freq_Domain_features_of_signal(window_gyro_X, "gyro_X", fs)
            window_features_gyro_Y_Freq     = get_Freq_Domain_features_of_signal(window_gyro_Y, "gyro_Y", fs)
            window_features_gyro_Z_Freq     = get_Freq_Domain_features_of_signal(window_gyro_Z, "gyro_Z", fs)
            window_features_mag_X_Freq      = get_Freq_Domain_features_of_signal(window_mag_X, "mag_X", fs)
            window_features_mag_Y_Freq      = get_Freq_Domain_features_of_signal(window_mag_Y, "mag_Y", fs)
            window_features_mag_Z_Freq      = get_Freq_Domain_features_of_signal(window_mag_Z, "mag_Z", fs)
            # window_features_temp_Freq       = get_Freq_Domain_features_of_signal(window_temp, "temp", fs)


            # Merge all
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

                            **window_features_temp_Time
                            # , **window_features_temp_Freq
                            }

        # Append the features of the current window to the list
        all_window_features.append(window_features)
        window_labels.append(df_label)

    # print(f"Total number on IMU windows: {window_sum}") 

    # print(f"Total number of windows: {activityWindowsCounter}")

    return all_window_features, window_labels
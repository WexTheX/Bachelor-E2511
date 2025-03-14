import numpy as np
import pandas as pd
# SignalProcessing part is needed when this file is imported in main.py
from SignalProcessing.get_Time_Domain_features_of_signal import get_Time_Domain_features_of_signal
from SignalProcessing.get_Freq_Domain_features_of_signal import get_Freq_Domain_features_of_signal

def ExtractIMU_Features(imu_data, WindowLength, Norm_Accel):

    # Extract acceleration and gyroscope data from the IMU dataset
    time_data = imu_data[:, 0]  # Assuming 1st column is time
    accel_X = imu_data[:, 1]  # Acceleration in X direction
    accel_Y = imu_data[:, 2]  # Acceleration in Y direction
    accel_Z = imu_data[:, 3]  # Acceleration in Z direction
    gyro_X  = imu_data[:, 4]  # Gyroscope in X direction
    gyro_Y  = imu_data[:, 5]  # Gyroscope in Y direction
    gyro_Z  = imu_data[:, 6]  # Gyroscope in Z direction


    # Define a list to store features for each window
    all_window_features = []

    # Calculate the number of windows
    num_samples = len(time_data) # Number of measurements
    num_windows = num_samples // WindowLength
    # 42 = 84 322 // 2 000

    # Trying to remove first and last 10 seconds from the IMU_data sets
    # To avoid time wasted during start and stop of tests

    # cutLength = (10 * Fs) // WindowLength
    # 4 = 10 * 800 // 2 000

    print(f"Number of IMU  windows: {num_windows}")

    # for i in range(cutLength, num_windows - cutLength):

    for i in range(num_windows):
        
        # Define the start and end index for the window
        start_idx = i * WindowLength
        end_idx = start_idx + WindowLength
        print(start_idx, end_idx)  
                
        # Windowing the signals
        window_gyro_X  = gyro_X[start_idx:end_idx]
        window_gyro_Y  = gyro_Y[start_idx:end_idx]
        window_gyro_Z  = gyro_Z[start_idx:end_idx]

        if Norm_Accel == True:
            # Extract acceleration and gyroscope data from the IMU dataset
            norm_acceleration = np.sqrt( np.power(accel_X, 2) + np.power(accel_Y, 2) + np.power(accel_Z, 2))

            g_constant = np.mean(norm_acceleration)
            # print(f"g constant: {g_constant}")
            gravless_norm = np.subtract(norm_acceleration, g_constant)  
            window_accel_Norm = gravless_norm[start_idx:end_idx]

            
            window_features_accel_Norm_Time = get_Time_Domain_features_of_signal(window_accel_Norm, "accel_Norm")
            window_features_gyro_X_Time     = get_Time_Domain_features_of_signal(window_gyro_X, "gyro_X")
            window_features_gyro_Y_Time     = get_Time_Domain_features_of_signal(window_gyro_Y, "gyro_Y")
            window_features_gyro_Z_Time     = get_Time_Domain_features_of_signal(window_gyro_Z, "gyro_Z")

            window_features_accel_Norm_Freq = get_Freq_Domain_features_of_signal(window_accel_Norm, "accel_Norm", Fs=200)
            window_features_gyro_X_Freq     = get_Freq_Domain_features_of_signal(window_gyro_X, "gyro_X", Fs=200)
            window_features_gyro_Y_Freq     = get_Freq_Domain_features_of_signal(window_gyro_Y, "gyro_Y", Fs=200)
            window_features_gyro_Z_Freq     = get_Freq_Domain_features_of_signal(window_gyro_Z, "gyro_Z", Fs=200)

            ## merge all
            window_features = {**window_features_accel_Norm_Time, 
                               **window_features_accel_Norm_Freq,

                               **window_features_gyro_X_Time,
                               **window_features_gyro_Y_Time,
                               **window_features_gyro_Z_Time,
                               **window_features_gyro_X_Freq,
                               **window_features_gyro_Y_Freq,
                               **window_features_gyro_Z_Freq}
            

        if Norm_Accel == False:
            window_accel_X = accel_X[start_idx:end_idx]
            window_accel_Y = accel_Y[start_idx:end_idx]
            window_accel_Z = accel_Z[start_idx:end_idx]

            window_features_accel_X_Time = get_Time_Domain_features_of_signal(window_accel_X, "accel_X")
            window_features_accel_Y_Time = get_Time_Domain_features_of_signal(window_accel_Y, "accel_Y")
            window_features_accel_Z_Time = get_Time_Domain_features_of_signal(window_accel_Z, "accel_Z")

            window_features_gyro_X_Time  = get_Time_Domain_features_of_signal(window_gyro_X, "gyro_X")
            window_features_gyro_Y_Time  = get_Time_Domain_features_of_signal(window_gyro_Y, "gyro_Y")
            window_features_gyro_Z_Time  = get_Time_Domain_features_of_signal(window_gyro_Z, "gyro_Z")

            window_features_accel_X_Freq = get_Freq_Domain_features_of_signal(window_accel_X, "accel_X", Fs=200)
            window_features_accel_Y_Freq = get_Freq_Domain_features_of_signal(window_accel_Y, "accel_Y", Fs=200)
            window_features_accel_Z_Freq = get_Freq_Domain_features_of_signal(window_accel_Z, "accel_Z", Fs=200)

            window_features_gyro_X_Freq  = get_Freq_Domain_features_of_signal(window_gyro_X, "gyro_X", Fs=200)
            window_features_gyro_Y_Freq  = get_Freq_Domain_features_of_signal(window_gyro_Y, "gyro_Y", Fs=200)
            window_features_gyro_Z_Freq  = get_Freq_Domain_features_of_signal(window_gyro_Z, "gyro_Z", Fs=200)

            ## merge all
            window_features = {**window_features_accel_X_Time, 
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
                               **window_features_gyro_Z_Freq}


        # Append the features of the current window to the list
        all_window_features.append(window_features)

    # Convert the list of features to a Pandas DataFrame for easy manipulation
    feature_df = pd.DataFrame(all_window_features)

    return feature_df

'''
def splitTrainingAndTest(num_of_windows):
    trainingShare = 0.9
    testShare = (1 - trainingShare)
    n = 3
    j = "Datafiles/20250226 Angle Grinder/26.02.2025 095305"

    feature_df = ExtractIMU_Features("imu_data", "WindowLength", "Norm_Accel")
    
    for i in feature_df:
        random = random.randint(0, 9)
        if random != 9:
            trainingData = i  
            name = set[j]
            label = name[n:]

            # append trainingData til df_training

        elif random == 9:
            testData = i
            # Label testData

            # append testData til df_testing

    


    # trainingData = num_of_windows * trainingShare

    # testData = num_of_windows * testShare'
'''
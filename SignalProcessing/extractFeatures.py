import numpy as np
import pandas as pd
# SignalProcessing part is needed when this file is imported in main.py
from SignalProcessing.get_Time_Domain_features_of_signal import get_Time_Domain_features_of_signal
from SignalProcessing.get_Freq_Domain_features_of_signal import get_Freq_Domain_features_of_signal
from Preprocessing.preprocessing import tab_txt_to_csv
from Preprocessing.preprocessing import delete_header

# Changed name for clarity
# Still based on Roya's "ExtractIMU_Features"

def Extract_All_Features(datasets, WindowLength, Norm_Accel, Fs):
    
    features_df = []

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

        cutLength = (10 * Fs) // WindowLength
        # print(cutLength)
        # 4 = 10 * 800 // 2 000

        print(f"Number of IMU windows after cut: {num_windows - 2 * cutLength}")

        
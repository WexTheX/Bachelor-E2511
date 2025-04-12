import time
import numpy as np
import pandas as pd

import pickle

from bleak import BleakScanner, BleakClient
import asyncio

from muse_api_main.Muse_Utils import *
from muse_api_main.Muse_HW import *

from extractFeatures import extractFeaturesFromDF

''' Command and data characteristics '''
CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7" 
DATA_UUID = "09bf2c52-d1d9-c0b7-4145-475964544307"

myDev = None
device_list = ["Muse_E2511_GREY", "Muse_E2511_RED"]
device_name = device_list[1]

window_length_sec = 20                  # Length of one window for prediction
fs = 200                                # Frequency of sensor sampling
window_size = window_length_sec * fs
real_time_window_sec = 50               # Time period the program will stream

sample_queue = asyncio.Queue()
shutdown_event = asyncio.Event()
start_time = time.time()

data_mode = MH.DataMode.DATA_MODE_IMU_MAG_TEMP_PRES_LIGHT       # Data mode, what sensors is used
DATA_SIZE = 6 * 5;                                              # Dimension of incomming packet (6 bytes * number of sensors)
DATA_BUFFER_SIZE = int((128 - 8) / DATA_SIZE)                   # Number of packets for each 128-bytes notification

notification_counter = 0    # Counter for total notifications              
sample_counter = 0          # Counter for number of samples, resets on new prediction
prediction_counter = 0
num_values = 14             # Number of values the sensors will give, (3 (axl, xyz) + 3 (gyo, xyz) + 3 (mag, xyz) + 2 (temp, press) + 3 (range, light, ir-light))

# sample_list = np.zeros((fs*window_length_sec, num_values+1))    # Preloads list of samples as zeroes
columns = ["Timestamp",                                         # Predefines columns names of dataframe
           "Axl.X","Axl.Y","Axl.Z",
           "Gyr.X","Gyr.Y","Gyr.Z",
           "Mag.X","Mag.Y","Mag.Z",
           "Temp","Press",
           "Range","Lum","IRLum"]    
prediction_list = {}                                            # Prepares list of predictions

''' PICKLE IMPORTS '''
output_path = "OutputFiles/Separated/"                          # Define import path
with open(output_path + "classifier.pkl", "rb") as CLF_file:    # Import classifier
# with open(output_path + "classifiers/SVM_Grid_SearchCV_clf.pkl", "rb") as CLF_file:    # Import classifier
# with open(output_path + "classifiers/LR_GridSearchCV_clf.pkl", "rb") as CLF_file:    # Import classifier
# with open(output_path + "classifiers/KNN_GridSearchCV_clf.pkl", "rb") as CLF_file:    # Import classifier
    clf = pickle.load(CLF_file)
CLF_file.close()

with open(output_path + "PCA.pkl", "rb" ) as PCA_file:          # Import PCA
    PCA_final = pickle.load(PCA_file)
PCA_file.close()

with open(output_path + "scaler.pkl", "rb" ) as scaler_File:    # Import Scaler
    scaler = pickle.load(scaler_File)
scaler_File.close()


''' NOTIFICATION FUNCTIONS '''
def cmd_notification_handler(sender, data):
    """Simple notification handler which prints the data received."""
    print("{0}: {1}".format(sender, data))

    return

async def dataNotificationHandler(sender: int, data: bytearray):
    """Decode data"""
    global notification_counter, sample_counter, start_time
    header_offset = 8   # Ignore part of notification that is header data (8 bytes)

    for k in range(DATA_BUFFER_SIZE):
        current_packet = bytearray(DATA_SIZE)                                   # Define size of first packet in notification
        current_packet[:] = data[header_offset : header_offset + DATA_SIZE + 1] # Get packet data
        temp_data = Muse_Utils.DecodePacket(                                    # Decode packet to covert values and store them in MuseData Object
            current_packet, 0, stream_mode.value, 
            gyrConfig.Sensitivity, axlConfig.Sensitivity, magConfig.Sensitivity, hdrConfig.Sensitivity
        )   

        sample = [
            time.time(),
            temp_data.axl[0], temp_data.axl[1], temp_data.axl[2],
            temp_data.gyr[0], temp_data.gyr[1], temp_data.gyr[2],
            temp_data.mag[0], temp_data.mag[1], temp_data.mag[2],
            temp_data.tp[0], temp_data.tp[1],
            temp_data.light.range, temp_data.light.lum_vis, temp_data.light.lum_ir            
        ]

        await sample_queue.put(sample)

        # sample_counter += 1
        # if sample_counter % 1000 == 0:
        #     elapsed = time.time() - start_time
        #     print(f"Sample rate: {sample_counter / elapsed:.2f} sample/sec")
    return

async def processSamples():
    global prediction_counter, prediction_list
    sample_list = []

    while True:
        sample = await sample_queue.get()
        sample_list.append(sample)

        if len(sample_list) >= window_size:
            try:
                ''' CONVERT TO DF, FEATURE EXTRACT AND SCALE '''
                feature_df = pd.DataFrame(data=sample_list, columns=columns)                                    # Convert samples_list into dataframe to make it usable in extractFeaturesFromDF
                feature_df_extraction, label = extractFeaturesFromDF(feature_df, "Realtime", window_length_sec, fs, False)
                feature_df_scaled = scaler.transform(pd.DataFrame(feature_df_extraction))                       # Scale data with scaled from training data
            
                ''' PCA AND PREDICT '''
                PCA_feature_df = pd.DataFrame(PCA_final.transform(feature_df_scaled))                           # Convert to PC found from training data
                prediction = clf.predict(PCA_feature_df)                                                        # Predict label using classifier                                                         
                prediction_list[time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime())] = prediction[0] # Add prediction to prediction list dict, with timemark as reference
                prediction_counter += 1

                print(prediction)  
            except Exception as e:
                print(f"Error when predicting: {e}.")

            sample_list.clear()

async def waitForQuit():
    loop = asyncio.get_event_loop()
    print("Type q for to end streaming.")
    while True:
        user_input = await loop.run_in_executor(None, input)
        if user_input.strip().lower() in ("q", "exit"):
            shutdown_event.set()
            break


def list_services(client):
    #print all services and characteristic

    services = client.services
    for service in services:
        print('\nservice', service.handle, service.uuid, service.description)

        characteristics = service.characteristics

        for char in characteristics:
            print('  characteristic', char.handle, char.uuid, char.description, char.properties)

            descriptors = char.descriptors

            for desc in descriptors:
                print('    descriptor', desc)

async def main():

    global device_ID, stream_mode
    global gyrConfig, axlConfig, magConfig, hdrConfig
    global start_time


    #DEVICE ENUMERATION
    devices = await BleakScanner.discover()
    myDev = None
    for d in devices:
        print(d)
        if d.name == device_name:
            myDev = d
    

    if(myDev != None):
        #DEVICE CONNECTION
        async with BleakClient(str(myDev.address)) as client:
            list_services(client)
            
            # Start notify on command characteristic
            await client.start_notify(CMD_UUID, cmd_notification_handler)

            # Get device ID
            await client.write_gatt_char(CMD_UUID,Muse_Utils.Cmd_GetDeviceID(),True)
            response = await client.read_gatt_char(CMD_UUID)
            device_ID = Muse_Utils.Dec_DeviceID(CommandResponse(response))
            print("Device successfully connected !")
            
            # Get sensors full scales
            await client.write_gatt_char(CMD_UUID,Muse_Utils.Cmd_GetSensorsFullScale(),True)
            response = await client.read_gatt_char(CMD_UUID)
            gyrConfig, axlConfig, magConfig, hdrConfig = Muse_Utils.Dec_SensorsFullScales(CommandResponse(response))

            
            # Set up the command
            stream_mode = data_mode
            cmd_stream = Muse_Utils.Cmd_StartStream(mode=stream_mode, frequency=MH.DataFrequency.DATA_FREQ_200Hz, enableDirect=False)

            # Start notify on data characteristic
            start_time = time.time()
            await client.start_notify(DATA_UUID, dataNotificationHandler)

            start_time_local = time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime())
            print(f"Start streaming, {start_time_local}")
            # Start Streaming using the above configuration (direct streaming, IMU mode and Sampling Frequency = 200 Hz)
            await client.write_gatt_char(CMD_UUID, cmd_stream, True)
            processing_task = asyncio.create_task(processSamples())
            wait_for_quit_task = asyncio.create_task(waitForQuit())

            # Set streaming duration to real_time_window_sec seconds, then stop
            # await asyncio.sleep(real_time_window_sec) 
            await shutdown_event.wait()   
           
            await client.write_gatt_char(CMD_UUID, Muse_Utils.Cmd_StopAcquisition(), response=True)
            end_time_local = time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime())
            print(f"Streaming stopped, {end_time_local}")

            print(f"Prediction list: \n {prediction_list}")

asyncio.run(main())

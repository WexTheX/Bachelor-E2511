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

''' VARIABLES '''
myDev = None
device_list = ["Muse_E2511_GREY", "Muse_E2511_RED", "muse_v3_3"] # List of bluetooth devices 
device_name = device_list[2]                        # Choose device to connect to from list

window_length_sec = 20                  # Length of one window for prediction
fs = 200                                # Frequency of sensor sampling
window_size = window_length_sec * fs
real_time_window_sec = 30               # Time period the program will stream, deprecated after quitProgram function, still nice for debug
start_time = time.time()  
last_notification_time = time.time()

sample_queue = asyncio.Queue()          # Queue to store samples as they arrive
shutdown_event = asyncio.Event()        # Event to listen for quitting of program

data_mode = MH.DataMode.DATA_MODE_IMU_MAG_TEMP_PRES             # Data mode, what sensors is used
DATA_SIZE = 6 * 4;                                              # Dimension of incomming packet (6 bytes * number of sensors)
DATA_BUFFER_SIZE = int((128 - 8) / DATA_SIZE)                   # Number of packets for each 128-bytes notification

notification_counter = 0             
sample_counter = 0
prediction_counter = 0
num_values = 14             # Number of values the sensors will give, (3 (axl, xyz) + 3 (gyo, xyz) + 3 (mag, xyz) + 2 (temp, press) + 3 (range, light, ir-light))

df_columns = ["Timestamp",                                         # Predefines columns names of dataframe
           "Axl.X","Axl.Y","Axl.Z",
           "Gyr.X","Gyr.Y","Gyr.Z",
           "Mag.X","Mag.Y","Mag.Z",
           "Temp","Press",
           "Range","Lum","IRLum"]    
prediction_list = {}   
tot_sample_log = []                                      # Prepares dict of predictions

''' PICKLE IMPORTS '''
output_path = "OutputFiles/Separated/"                          # Define import path
with open(output_path + "classifier.pkl", "rb") as CLF_file:    # Import classifier
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
    """ DECODE DATA FUNCTION 
    This function runs every time the device sends a bluetooth notification.
    The notification is 128 bytes, where the last 120 is the data needed.
    Every DATA_SIZE amount of bytes contains samples from that 1/fs interval.
    For each DATA_SIZE segment in the 120 bytes data, it runds thorugh 221e's DecodePacket function
    which converts the data using the senosor calibration data and stores it in a MuseData object

    The MuseData Object is then stored in a sample queue, 
    so it can be ready to recieve the next packet as soon as possible. 
    """
    global notification_counter, sample_counter, start_time, last_notification_time
    ''' DEBUG AND PRINTS TO MEASURE NOTIFICATION DELTA '''
    # notification_delta = time.time() - last_notification_time
    # print(f"Notifcation delta: {notification_delta}")
    # print(notification_delta)
    # last_notification_time = time.time()
    # print(f"Size of data: {len(data)}")
        
    header_offset = 8                                   # Ignore part of notification that is header data (8 bytes)

    ''' RUN THOUGH EACH SEGMENT OF THE DATA '''
    for k in range(DATA_BUFFER_SIZE):
        current_packet = bytearray(DATA_SIZE)           # Define size of the segment (packet)
        
        start_idx   = k * DATA_SIZE + header_offset     # Define where current segment begins
        end_idx     = start_idx + DATA_SIZE + 1         # Define where current segment ends

        # Define size of first packet in notification
        current_packet[:] = data[start_idx : end_idx]   # Extract packet data
        temp_data = Muse_Utils.DecodePacket(            # Decode packet to convert values and store them in MuseData Object
            current_packet, 0, stream_mode.value, 
            gyrConfig.Sensitivity, axlConfig.Sensitivity, magConfig.Sensitivity, hdrConfig.Sensitivity
        )   

        await sample_queue.put(temp_data)               # Put segment as MuseData object in queue

        ''' DEBUG AND PRINT TO MEASURE SAMPLES PER SECOND'''
        sample_counter += 1
        if sample_counter % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Sample rate: {sample_counter / elapsed:.2f} sample/sec")
    return


async def processSamples():
    ''' PROCESS SAMPLES FUNCTION
    This function is run as a task that constantly listens to the sample queue.
    Once a new element is added, it takes it in and converts it from a MuseData Object to an array.
    This array is then appended to a list of samples which contains all samples from the current window.

    Once the samples list is is large enough to contain window_length number of samples
    it will run the processing part.
    This consists of:
        Convert samples list to dataframe, combining it with the df_columns to give the values a column name
        Extract features from the dataframe
        Scale the data using the training datas scaler
        Transform the data using the training datas PCA component
        Predict the class using the classifier.
    The prediction is then stored in a dict containing the class and the time the prediction occured.
    The sample list is then cleared to prepare for the next segment.
    '''
    global prediction_counter, prediction_list, tot_sample_log
    sample_log = []

    
    while True:                                 # Run allways
        temp_data = await sample_queue.get()    # Get sample from queue
        sample = np.array([                              # Convert from MuseData object to array
            time.time(),
            temp_data.axl[0], temp_data.axl[1], temp_data.axl[2],
            temp_data.gyr[0], temp_data.gyr[1], temp_data.gyr[2],
            temp_data.mag[0], temp_data.mag[1], temp_data.mag[2],
            temp_data.tp[0], temp_data.tp[1],
            temp_data.light.range, temp_data.light.lum_vis, temp_data.light.lum_ir            
        ])
        sample_log.append(sample)              # Add sample to list

        if len(sample_log) >= window_size:
            ''' DEBUG TO MEASURE PROCESS TIME '''
            start_time = time.time()
            try:
                ''' CONVERT TO DF, FEATURE EXTRACT AND SCALE '''
                feature_df = pd.DataFrame(data=sample_log, columns=df_columns)                                 # Convert samples_list into dataframe to make it usable in extractFeaturesFromDF
                feature_df_extraction, label = extractFeaturesFromDF(feature_df, "Realtime", window_length_sec, fs, False)
                feature_df_scaled = scaler.transform(pd.DataFrame(feature_df_extraction))                       # Scale data with scaled from training data
            
                ''' PCA AND PREDICT '''
                PCA_feature_df = pd.DataFrame(PCA_final.transform(feature_df_scaled))                           # Convert to PC found from training data
                prediction = clf.predict(PCA_feature_df)                                                        # Predict label using classifier                                                         
                prediction_list[time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime())] = prediction[0] # Add prediction to prediction list dict, with timemark as reference
                prediction_counter += 1

                print(prediction)

                tot_sample_log = tot_sample_log + sample_log

            except Exception as e:
                print(f"Error when predicting: {e}.")
            
            sample_log.clear()

            ''' CONT. DEBUG TO MEASURE PROCESS TIME'''
            end_time = time.time()  # End timer
            elapsed_time = end_time - start_time
            print(f"Process sampling completed in {elapsed_time} seconds.")

async def waitForQuit():
    ''' QUITTING FUNCTION
    Simple function to wait for input from terminal before quitting the program
    '''
    loop = asyncio.get_event_loop()
    print("Type q for to end streaming.")
    while True:
        user_input = await loop.run_in_executor(None, input)
        if user_input.strip().lower() in ("q"):                 # Choose what input to wait for
            shutdown_event.set()                                # Set flag so main() can continue
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

async def RT_main():

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
            await client.start_notify(DATA_UUID, dataNotificationHandler)       # Start notification handler

            start_time_local = time.strftime("%d%b%Y_%H%M%S", time.localtime())
            print(f"Start streaming, {start_time_local}")
            await client.write_gatt_char(CMD_UUID, cmd_stream, True)            # Tell device to start streaming
            processing_task = asyncio.create_task(processSamples())             # Start processing task
            wait_for_quit_task = asyncio.create_task(waitForQuit())             # Start listen for quit signal task

            ''' WAITING FUNCTIONS '''
            # await asyncio.sleep(real_time_window_sec)       # Wait real_time_window_sec time before moving on
            await shutdown_event.wait()                     # Wait until shutdown_event flag is set before moving on
           
            await client.write_gatt_char(CMD_UUID, Muse_Utils.Cmd_StopAcquisition(), response=True)
            end_time_local = time.strftime("%d%b%Y_%H%M%S", time.localtime())
            print(f"Streaming stopped, {end_time_local}")

            print(f"Prediction list: \n {prediction_list}")

            ''' STORE SAMPLE LOG FOR FUTURE ANALYSIS '''
            sample_df = pd.DataFrame(data=tot_sample_log, columns=df_columns)
            sample_df.to_csv(output_path+"/livedetection/"+str(start_time_local)+".csv", index=False)      
            print(f"Stored log as file: {str(start_time_local)}.csv")


asyncio.run(RT_main())

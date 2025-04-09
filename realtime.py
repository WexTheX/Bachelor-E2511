""" ble_conn.py: main file of example usage for Python Muse API.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

__authors__ = ["Francesca Palazzo", "Roberto Bortoletto", "Luigi Mattiello"]
__contact__ = "info@221e.com"
__copyright__ = "Copyright (c) 2020 by 221e srl."
__credits__ = ["Francesca Palazzo", "Roberto Bortoletto", "Luigi Mattiello"]
__deprecated__ = False
__email__ =  "roberto.bortoletto@221e.com"
__license__ = "GNU General Public License"
__maintainer__ = "Roberto Bortoletto"
__status__ = "Production"
__version__ = "1.3.0"

from bleak import BleakScanner, BleakClient
import asyncio
import threading
from muse_api_main.Muse_Utils import *
from muse_api_main.Muse_HW import *
from sklearn.decomposition import PCA
import pickle
import pandas as pd
from extractFeatures import extractAllFeatures, extractDFfromFile, extractFeaturesFromDF
import time
import numpy as np

CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7" 
DATA_UUID = "09bf2c52-d1d9-c0b7-4145-475964544307"

myDev = None
muse_name = "Muse_E2511_RED"

window_length_sec = 20                                              # Length of 1 window
fs = 200  

data_mode = MH.DataMode.DATA_MODE_IMU_MAG_TEMP_PRES_LIGHT
DATA_SIZE = 6*5;  # dimension of packet
DATA_BUFFER_SIZE = int((128 - 8) / DATA_SIZE) # number of packets for each 128-bytes notification

notification_counter = 0
sample_counter = 0
num_values = 14
data_buffer = np.zeros((DATA_BUFFER_SIZE, num_values))  # buffer to store accelerometer data

feature_list = np.zeros((fs*window_length_sec, num_values+1))
columns = ["Timestamp","Axl.X","Axl.Y","Axl.Z","Gyr.X","Gyr.Y","Gyr.Z","Mag.X","Mag.Y","Mag.Z","Temp","Press","Range","Lum","IRLum"]
prediction_list = {}

delta_time = 0.04           # Time difference between samples at 25Hz
real_time_window_sec = 30  # Time period the program will stream

'''
# Packet dimension and size, according to mode (30 and 4)
pck_dim = Muse_Utils.GetPacketDimension(data_mode)
print(pck_dim)
pck_num = Muse_Utils.GetNumberOfPackets(data_mode)
print(pck_num)
quit()
'''

''' Pickled PCA and CLF from main '''


output_path = "OutputFiles/Separated/"
with open(output_path + "classifier.pkl", "rb") as CLF_file:
    clf = pickle.load(CLF_file)

with open(output_path + "PCA.pkl", "rb" ) as PCA_File:
    PCA_final = pickle.load(PCA_File)

with open(output_path + "scaler.pkl", "rb" ) as Scaler_File:
    scaler = pickle.load(Scaler_File)

def cmd_notification_handler(sender, data):
    """Simple notification handler which prints the data received."""
    print("{0}: {1}".format(sender, data))

    return



async def data_notification_handler(sender: int, data: bytearray):
    """Decode data"""
    global feature_list, prediction_list, notification_counter, sample_counter
    header_offset = 8   # ignore packet header

    for k in range(DATA_BUFFER_SIZE):
        current_packet = bytearray(DATA_SIZE)
        current_packet[:] = data[header_offset : header_offset + DATA_SIZE + 1]
        temp_data = Muse_Utils.DecodePacket(current_packet, 0, stream_mode.value, gyrConfig.Sensitivity, axlConfig.Sensitivity, magConfig.Sensitivity, hdrConfig.Sensitivity)
        
        feature_list[k+(sample_counter)][0] = time.time()
        feature_list[k+(sample_counter)][1] = temp_data.axl[0]
        feature_list[k+(sample_counter)][2] = temp_data.axl[1]
        feature_list[k+(sample_counter)][3] = temp_data.axl[2]
        feature_list[k+(sample_counter)][4] = temp_data.gyr[0]
        feature_list[k+(sample_counter)][5] = temp_data.gyr[1]
        feature_list[k+(sample_counter)][6] = temp_data.gyr[2]
        feature_list[k+(sample_counter)][7] = temp_data.mag[0]
        feature_list[k+(sample_counter)][8] = temp_data.mag[1]
        feature_list[k+(sample_counter)][9] = temp_data.mag[2]
        feature_list[k+(sample_counter)][10] = temp_data.tp[0]
        feature_list[k+(sample_counter)][11] = temp_data.tp[1]
        feature_list[k+(sample_counter)][12] = temp_data.light.range
        feature_list[k+(sample_counter)][13] = temp_data.light.lum_vis
        feature_list[k+(sample_counter)][14] = temp_data.light.lum_ir

    sample_counter += 4

    # for k in range(DATA_BUFFER_SIZE):
    #     #Decode Accelerometer reading
    #     current_packet = bytearray(6)
    #     # print(f"Start_idx: {start_idx}, stop: {start_idx + DATA_SIZE + 1}")
    #     current_packet[:] = data[header_offset : header_offset + DATA_SIZE + 1] 
    #     # print(f"Current Packet: {current_packet}")

    #     temp_data = [0.0] * num_values
    #     for i in range(num_values):
    #         # Extract channel raw value (i.e., 2-bytes each)
    #         raw_value = current_packet[2*i:2*(i+1)]
    #         # Convert to Int16 and apply sensitivity scaling
    #         if(0 <= i < 3):
    #             temp_data[i] = int.from_bytes(raw_value, byteorder='little', signed=True) * gyrConfig.Sensitivity
    #         elif(3 <= i < 6):
    #             temp_data[i] = int.from_bytes(raw_value, byteorder='little', signed=True) * axlConfig.Sensitivity
    #         elif(6 <= i < 9):
    #             temp_data[i] = int.from_bytes(raw_value, byteorder='little', signed=True) * magConfig.Sensitivity
    #         elif(i == 10):
    #             temp_data[i] = int.from_bytes(raw_value, byteorder='little', signed=False) 
    #             temp_data[i] /= 100
    #         elif(i == 9):
    #             temp_data[i] = int.from_bytes(raw_value, byteorder='little', signed=False)
    #             temp_data[i] /= 4096
    #         else:
    #             temp_data[i] = int.from_bytes(raw_value, byteorder='little', signed=False)

        
        # feature_list[k+(sample_counter)][0] = time.time()
        # feature_list[k+(sample_counter)][1] = temp_data[3]
        # feature_list[k+(sample_counter)][2] = temp_data[4]
        # feature_list[k+(sample_counter)][3] = temp_data[5]
        # feature_list[k+(sample_counter)][4] = temp_data[0]
        # feature_list[k+(sample_counter)][5] = temp_data[1]
        # feature_list[k+(sample_counter)][6] = temp_data[2]
        # feature_list[k+(sample_counter)][7] = temp_data[6]
        # feature_list[k+(sample_counter)][8] = temp_data[7]
        # feature_list[k+(sample_counter)][9] = temp_data[8]
        # feature_list[k+(sample_counter)][10] = temp_data[10]
        # feature_list[k+(sample_counter)][11] = temp_data[9]
        # feature_list[k+(sample_counter)][12] = temp_data[11]
        # feature_list[k+(sample_counter)][13] = temp_data[12]
        # feature_list[k+(sample_counter)][14] = temp_data[13]


    #     header_offset += DATA_SIZE
    
    # sample_counter += DATA_BUFFER_SIZE

    # print(f"data_buffer (post): \n {data_buffer}")

    # features = np.array([
    #     time.time(),
    #     tempData.axl[0], 
    #     tempData.axl[1], 
    #     tempData.axl[2],                             
    #     tempData.gyr[0], 
    #     tempData.gyr[1], 
    #     tempData.gyr[2],                              
    #     tempData.mag[0], 
    #     tempData.mag[1], 
    #     tempData.mag[2],                             
    #     tempData.tp[0], 
    #     tempData.tp[1],                                                
    #     tempData.light.range, 
    #     tempData.light.lum_vis, 
    #     tempData.light.lum_ir
    # ])

    
    print(sample_counter)
    
    if ((sample_counter) > window_length_sec*fs-1):
        ''' FEATURE EXTRACTION AND SCALE '''
        feature_df = pd.DataFrame(data=feature_list, columns=columns)  
        print(feature_df)
        feature_df_extraction, label = extractFeaturesFromDF(feature_df, "Realtime", window_length_sec, fs, False)
        feature_df_scaled = scaler.transform(pd.DataFrame(feature_df_extraction))
       
        ''' PCA AND PREDICT '''
        PCA_feature_df = pd.DataFrame(PCA_final.transform(feature_df_scaled))
        prediction = clf.predict(PCA_feature_df)
        print(prediction)
        prediction_list[time.time()] = prediction

        sample_counter = 0

    notification_counter += 1   
    return



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


    #DEVICE ENUMERATION
    devices = await BleakScanner.discover()
    myDev = None
    for d in devices:
        print(d)
        if d.name == 'Muse_E2511_RED':
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
            await client.start_notify(DATA_UUID, data_notification_handler)

            print("Start streaming")
            # Start Streaming using the above configuration (direct streaming, IMU mode and Sampling Frequency = 200 Hz)
            await client.write_gatt_char(CMD_UUID, cmd_stream, True)
            
            # Set streaming duration to 10 seconds
            await asyncio.sleep(real_time_window_sec)

            # Stop data acquisition in STREAMING mode      
            await client.write_gatt_char(CMD_UUID, Muse_Utils.Cmd_StopAcquisition(), response=True)
            print("Streaming stopped")


asyncio.run(main())




           
            

                    

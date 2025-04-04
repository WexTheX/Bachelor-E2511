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
from Muse_Utils import *
from sklearn.decomposition import PCA
import pickle
from machineLearning import scaleFeatures
import pandas as pd

CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7" 
DATA_UUID = "09bf2c52-d1d9-c0b7-4145-475964544307"




''' Pickled PCA and CLF from main '''


output_path = "OutputFiles/"
with open(output_path + "classifier.pkl", "rb") as CLF_file:
        halving_classifier = pickle.load(CLF_file)

with open(output_path + "PCA.pkl", "wb" ) as PCA_File:
    PCA_final = pickle.load(PCA_File)



def cmd_notification_handler(sender, data):
    """Simple notification handler which prints the data received."""
    print("{0}: {1}".format(sender, data))

    return



async def data_notification_handler(sender: int, data: bytearray):
    """Decode data"""

    header_offset = 8   # ignore packet header

    # decode packet data
    tempData = Muse_Utils.DecodePacket(data[header_offset:], 0, stream_mode.value, gyrConfig.Sensitivity, axlConfig.Sensitivity, magConfig.Sensitivity, hdrConfig.Sensitivity)
    # print data as: device_ID, axl_X, axl_Y, axl_Z, gyr_X, gyr_Y, gyr_Z

    #print("{0} {1} {2} {3} {4} {5} {6}".format(device_ID,tempData.axl[0],tempData.axl[1],tempData.axl[2],tempData.gyr[0],tempData.gyr[1],tempData.gyr[2]))
    
    imu_features = [
        tempData.axl[0], tempData.axl[1], tempData.axl[2],                              # Accelerometer
        tempData.gyr[0], tempData.gyr[1], tempData.gyr[2],                              # Gyroscope
        tempData.mag[0], tempData.mag[1], tempData.mag[2],                              # Magnetometer
        tempData.tp[0], tempData.tp[1],                                                 # Temperature pressure
        tempData.light.range, tempData.light.lum_vis, tempData.light.lum_ir             # Light (range, lum, irlum)
    ]
    
    
    feature_segment = splitWindow(featureList=data)

    feature_segment_scaled = scaleFeatures(feature_segment)
    
    
    PCA_final_df  = pd.DataFrame(PCA_final.transform(feature_segment_scaled))
    

    prediction = halving_classifier.predict(PCA_final_df)




    return



def splitWindow(data, feature_list):
    
    
    if len(data) < 10000:
        feature_list.append()

    else:
        
        segment = feature_list
        feature_list.clear()
        return segment


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
            stream_mode = MH.DataMode.DATA_MODE_IMU_MAG_TEMP_PRES_LIGHT
            cmd_stream = Muse_Utils.Cmd_StartStream(mode=stream_mode, frequency=MH.DataFrequency.DATA_FREQ_200Hz, enableDirect=True)

            # Start notify on data characteristic
            await client.start_notify(DATA_UUID, data_notification_handler)

            print("Start streaming")
            # Start Streaming using the above configuration (direct streaming, IMU mode and Sampling Frequency = 200 Hz)
            await client.write_gatt_char(CMD_UUID, cmd_stream, True)
            
            # Set streaming duration to 10 seconds
            await asyncio.sleep(10)

            # Stop data acquisition in STREAMING mode      
            await client.write_gatt_char(CMD_UUID, Muse_Utils.Cmd_StopAcquisition(), response=True)
            print("Streaming stopped")


asyncio.run(main())




           
            

                    

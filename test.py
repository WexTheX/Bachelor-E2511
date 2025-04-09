from bleak import BleakScanner, BleakClient
import asyncio

import time

import numpy as np

from muse_api_main.Muse_Utils import *
from muse_api_main.Muse_HW import *

# Command and Data characteristics
CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
DATA_UUID = "09bf2c52-d1d9-c0b7-4145-475964544307"

myDev = None
data_mode = MH.DataMode.DATA_MODE_IMU_MAG_TEMP_PRES_LIGHT
DATA_SIZE = 6*5;  # dimension of packet
DATA_BUFFER_SIZE = int((128 - 8) / DATA_SIZE) # number of packets for each 128-bytes notification

notification_counter = 0
num_values = 14
data_buffer = np.zeros((DATA_BUFFER_SIZE, num_values))  # buffer to store accelerometer data


def data_notification_handler(sender, data):
    """Decode data"""

    global notification_counter, axlConfig, gyrConfig, magConfig, hdrConfig

    # The first 8 bytes of the data buffer correspond to the Timestamp: we ignore them in this example
    start_idx = 8
    # print(f"Data: {data}")

    # Define acceleration buffer to store decoded data
    data_buffer = np.zeros((DATA_BUFFER_SIZE, num_values))
    # print(f"Time: {time.time()}")
    # print(f"data_buffer (pre): \n {data_buffer}")
    # print(f"DATA_BUFFER_SIZE: \n {DATA_BUFFER_SIZE}")

    for k in range(DATA_BUFFER_SIZE):
        #Decode Accelerometer reading
        current_packet = bytearray(6)
        # print(f"Start_idx: {start_idx}, stop: {start_idx + DATA_SIZE + 1}")
        current_packet[:] = data[start_idx : start_idx + DATA_SIZE + 1] 
        # print(f"Current Packet: {current_packet}")

        temp_data = [0.0] * num_values
        for i in range(num_values):
            # Extract channel raw value (i.e., 2-bytes each)
            raw_value = current_packet[2*i:2*(i+1)]
            # Convert to Int16 and apply sensitivity scaling
            if(0 <= i < 3):
                temp_data[i] = int.from_bytes(raw_value, byteorder='little', signed=True) * gyrConfig.Sensitivity
            elif(3 <= i < 6):
                temp_data[i] = int.from_bytes(raw_value, byteorder='little', signed=True) * axlConfig.Sensitivity
            elif(6 <= i < 9):
                temp_data[i] = int.from_bytes(raw_value, byteorder='little', signed=True) * magConfig.Sensitivity
            else:
                temp_data[i] = int.from_bytes(raw_value, byteorder='little', signed=True)
        
        
        data_buffer[k][0] = temp_data[0]
        data_buffer[k][1] = temp_data[1]
        data_buffer[k][2] = temp_data[2]
        data_buffer[k][3] = temp_data[3]
        data_buffer[k][4] = temp_data[4]
        data_buffer[k][5] = temp_data[5]
        data_buffer[k][6] = temp_data[6]
        data_buffer[k][7] = temp_data[7]
        data_buffer[k][8] = temp_data[8]
        data_buffer[k][9] = temp_data[9]
        data_buffer[k][10] = temp_data[10]
        data_buffer[k][11] = temp_data[11]
        data_buffer[k][12] = temp_data[12]
        data_buffer[k][13] = temp_data[13]
        # data_buffer[k][14] = temp_data[14] # 3(axl)+3(gyr)+3(mag)+2(tp)+3(light)

        start_idx += DATA_SIZE

        notification_counter += 1 

    print(f"data_buffer (post): \n {data_buffer}")
    return

async def main():

    global ax, fig, canvas, x, acceleration_x, acceleration_y, acceleration_z, myDev
    global gyrConfig, axlConfig, magConfig, hdrConfig

    # Device name to be searched for
    my_device_name = 'Muse_E2511_RED'

    # Device Enumeration
    devices = await BleakScanner.discover(timeout=10.0)
    myDevice = None
    for d in devices:
        print(d)
        if d.name == my_device_name:
            myDevice = d

    if(myDevice != None):
        # Device Connection
        async with BleakClient(str(myDevice.address)) as client:
            
            # Check device status
            await client.write_gatt_char(char_specifier=CMD_UUID, data=bytearray([0x82,0x00]), response=True)
            response = await client.read_gatt_char(CMD_UUID)

            if (response[2] == 0x82 and response[3] == 0x00 and response[4] == 0x02):
                print('System in IDLE state')

                # Get Muse sensors full scales to decode data
                await client.write_gatt_char(CMD_UUID,Muse_Utils.Cmd_GetSensorsFullScale(),True)
                response = await client.read_gatt_char(CMD_UUID)
                gyrConfig, axlConfig, magConfig, hdrConfig = Muse_Utils.Dec_SensorsFullScales(CommandResponse(response))

                # Subscribe to Data characteristic
                await client.start_notify(DATA_UUID, data_notification_handler)
                
                # Start BUFFERED streaming of accelerometer data and read data
                stream_mode = data_mode
                cmd_stream = Muse_Utils.Cmd_StartStream(mode=stream_mode, frequency=MH.DataFrequency.DATA_FREQ_25Hz, enableDirect=False)
                await client.write_gatt_char(CMD_UUID, cmd_stream, True)

                # Make the streaming last for 30 seconds
                await asyncio.sleep(5)
                # Stop the streaming
                await client.write_gatt_char(char_specifier=CMD_UUID, data=Muse_Utils.Cmd_StopAcquisition(MH.CommunicationChannel.CHANNEL_BLE), response=True)

            await client.stop_notify(DATA_UUID)

asyncio.run(main())                   

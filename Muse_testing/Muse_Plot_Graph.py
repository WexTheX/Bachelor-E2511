from bleak import BleakScanner, BleakClient
import asyncio

import matplotlib.pyplot as plt
import numpy as np

from Muse_Utils import *
from Muse_HW import *

# Command and Data characteristics
CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
DATA_UUID = "09bf2c52-d1d9-c0b7-4145-475964544307"

myDev = None
DATA_SIZE_AXL = 6;  # dimension of AXL packet
AXL_BUFFER_SIZE = int((128 - 8) / DATA_SIZE_AXL) # number of packets for each 128-bytes notification
notification_counter = 0
axl_buffer = np.zeros((AXL_BUFFER_SIZE,3))  # buffer to store accelerometer data
PLOT_NUM_OF_SAMPLES = 500   # maximum number of samples to be displayed in the plot
DELTA_TIME = 0.0025   # time difference between consecutive samples (@25 Hz)

def data_notification_handler(sender, data):
    """Decode data"""

    global notification_counter, axlConfig, acceleration_x, acceleration_y, acceleration_z, x, ax, fig, canvas

    # The first 8 bytes of the data buffer correspond to the Timestamp: we ignore them in this example
    start_idx = 8

    # Define acceleration buffer to store decoded data
    axl_buffer = np.zeros((AXL_BUFFER_SIZE,3))

    for k in range(AXL_BUFFER_SIZE):
        #Decode Accelerometer reading
        current_packet = bytearray(6)
        current_packet[:] = data[start_idx : start_idx + DATA_SIZE_AXL + 1] 

        axl = [0.0] * 3
        for i in range(3):
            # Extract channel raw value (i.e., 2-bytes each)
            raw_value = current_packet[2*i:2*(i+1)]
            # Convert to Int16 and apply sensitivity scaling
            axl[i] = int.from_bytes(raw_value, byteorder='little', signed=True) * axlConfig.Sensitivity
        
        axl_buffer[k][0] = axl[0]
        axl_buffer[k][1] = axl[1]
        axl_buffer[k][2] = axl[2]

        start_idx += DATA_SIZE_AXL

        notification_counter += 1 

    """Update plot"""
    if (notification_counter > PLOT_NUM_OF_SAMPLES):
        # The data points filled all the plot area, so perform shift of old data
        acceleration_x = np.roll(acceleration_x,-AXL_BUFFER_SIZE)
        acceleration_y = np.roll(acceleration_y,-AXL_BUFFER_SIZE)
        acceleration_z = np.roll(acceleration_z,-AXL_BUFFER_SIZE)
        x = np.roll(x,-AXL_BUFFER_SIZE)
        for i in range(AXL_BUFFER_SIZE):
            acceleration_x[(PLOT_NUM_OF_SAMPLES-1)-AXL_BUFFER_SIZE+1+i] = axl_buffer[i][0]
            acceleration_y[(PLOT_NUM_OF_SAMPLES-1)-AXL_BUFFER_SIZE+1+i] = axl_buffer[i][1]
            acceleration_z[(PLOT_NUM_OF_SAMPLES-1)-AXL_BUFFER_SIZE+1+i] = axl_buffer[i][2]    
            x[(PLOT_NUM_OF_SAMPLES-1)-AXL_BUFFER_SIZE+1+i] = x[(PLOT_NUM_OF_SAMPLES-1)-AXL_BUFFER_SIZE+i] + DELTA_TIME
    else:
        for i in range(AXL_BUFFER_SIZE):
            acceleration_x[notification_counter-AXL_BUFFER_SIZE+i] = axl_buffer[i][0]
            acceleration_y[notification_counter-AXL_BUFFER_SIZE+i] = axl_buffer[i][1]
            acceleration_z[notification_counter-AXL_BUFFER_SIZE+i] = axl_buffer[i][2]     

    # Update maximum and minimum limits
    max_yp = np.nanmax([np.nanmax(acceleration_x), np.nanmax(acceleration_y), np.nanmax(acceleration_z)]) + 50
    min_yp = np.nanmin([np.nanmin(acceleration_x), np.nanmin(acceleration_y), np.nanmin(acceleration_z)]) - 50

    ax.plot(x,acceleration_x,'b.',linestyle='dotted')
    ax.plot(x,acceleration_y,'r.',linestyle='dotted')
    ax.plot(x,acceleration_z,'y.',linestyle='dotted')
    ax.set_xlim(x[0],x[(PLOT_NUM_OF_SAMPLES-1)])
    ax.set_ylim(min_yp,max_yp)
    plt.tight_layout()

    fig.canvas.draw()
    plt.pause(0.0001)

    return

async def main():

    global ax, fig, canvas, x, acceleration_x, acceleration_y, acceleration_z, myDev
    global gyrConfig, axlConfig, magConfig, hdrConfig

    # Device name to be searched for
    my_device_name = 'Muse'

    # Device Enumeration
    devices = await BleakScanner.discover()
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

                # Setup of figure
                fig, ax = plt.subplots()
                canvas = np.zeros((480,640))
                x = np.linspace(1, PLOT_NUM_OF_SAMPLES, PLOT_NUM_OF_SAMPLES)
                x *= DELTA_TIME   # Time in [sec]
                acceleration_x = np.zeros(PLOT_NUM_OF_SAMPLES) * np.nan
                acceleration_y = np.zeros(PLOT_NUM_OF_SAMPLES) * np.nan
                acceleration_z = np.zeros(PLOT_NUM_OF_SAMPLES) * np.nan
                ax.set_xlim(x[0],x[(PLOT_NUM_OF_SAMPLES-1)])
                ax.plot(x,acceleration_x,'b.',linestyle='dotted')
                ax.plot(x,acceleration_y,'r.',linestyle='dotted')
                ax.plot(x,acceleration_z,'y.',linestyle='dotted')
                ax.set_xlabel('Time [sec]')
                ax.set_ylabel('Acceleration [mg]')
                ax.set_title('Muse Streaming')
                fig.canvas.draw()   
                plt.pause(0.0001)

                # Subscribe to Data characteristic
                await client.start_notify(DATA_UUID, data_notification_handler)
                
                # Start BUFFERED streaming of accelerometer data and read data
                stream_mode = MH.DataMode.DATA_MODE_AXL
                cmd_stream = Muse_Utils.Cmd_StartStream(mode=stream_mode, frequency=MH.DataFrequency.DATA_FREQ_25Hz, enableDirect=False)
                await client.write_gatt_char(CMD_UUID, cmd_stream, True)

                # Make the streaming last for 30 seconds
                await asyncio.sleep(60)
                # Stop the streaming
                await client.write_gatt_char(char_specifier=CMD_UUID, data=Muse_Utils.Cmd_StopAcquisition(MH.CommunicationChannel.CHANNEL_BLE), response=True)

            await client.stop_notify(DATA_UUID)

asyncio.run(main())

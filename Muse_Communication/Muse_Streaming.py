
from bleak import BleakScanner, BleakClient
import asyncio

from Muse_Utils import *
from Muse_HW import Muse_HW as MH

from functools import partial
import contextlib


# Command and Data characteristics
CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
DATA_UUID = "09bf2c52-d1d9-c0b7-4145-475964544307"

# Muse characteristics
device_name = ["Muse_E2511_1", "Muse_E2511"]
stream_duration_seconds = 30
stream_start_timeout = 3

# Function for printing streaming data
async def data_notification_handler(client: BleakClient, sender: int, data: bytearray):
    """Decode data"""

    header_offset = 8   # ignore packet header

    if (client.address == client_1.address):
        # decode packet data
        tempData = Muse_Utils.DecodePacket(data[header_offset:], 0, stream_mode.value, gyrConfig1.Sensitivity, axlConfig1.Sensitivity, magConfig1.Sensitivity, hdrConfig1.Sensitivity)
        print("{0} {1} {2} {3} {4} {5} {6}".format(device_1_ID,tempData.axl[0],tempData.axl[1],tempData.axl[2],tempData.gyr[0],tempData.gyr[1],tempData.gyr[2]))
    else:
        # decode packet data
        tempData = Muse_Utils.DecodePacket(data[header_offset:], 0, stream_mode.value, gyrConfig2.Sensitivity, axlConfig2.Sensitivity, magConfig2.Sensitivity, hdrConfig2.Sensitivity)
        print("{0} {1} {2} {3} {4} {5} {6}".format(device_2_ID,tempData.axl[0],tempData.axl[1],tempData.axl[2],tempData.gyr[0],tempData.gyr[1],tempData.gyr[2]))
        
    return

async def main():

    global client_1, client_2, device_1_ID, device_2_ID, stream_mode
    global gyrConfig1, axlConfig1, magConfig1, hdrConfig1, gyrConfig2, axlConfig2, magConfig2, hdrConfig2

    # DEVICE ENUMERATION
    devices = await BleakScanner.discover(timeout=10.0)
    myDev_1 = None
    myDev_2 = None
    for d in devices:
        print(d)
        if d.name == device_name[0]:
            myDev_1 = d
        if d.name == device_name[1]:
            myDev_2 = d

    lock = asyncio.Lock()

    if(myDev_1 != None and myDev_2 != None):

        try:
            async with contextlib.AsyncExitStack() as stack:

                # Trying to establish a connection to two devices at the same time
                # can cause errors, so use a lock to avoid this.
                async with lock:

                    print("Connecting ...")

                    client_1 = BleakClient(myDev_1)
                    await stack.enter_async_context(client_1)
                   
                    # Get Muse n.1 device ID
                    await client_1.write_gatt_char(CMD_UUID,Muse_Utils.Cmd_GetDeviceID(),True)
                    response = await client_1.read_gatt_char(CMD_UUID)
                    device_1_ID = Muse_Utils.Dec_DeviceID(CommandResponse(response))
                    print("Device #1 successfully connected !")
                    # Get Muse n.1 sensors full scales
                    await client_1.write_gatt_char(CMD_UUID,Muse_Utils.Cmd_GetSensorsFullScale(),True)
                    response = await client_1.read_gatt_char(CMD_UUID)
                    gyrConfig1, axlConfig1, magConfig1, hdrConfig1 = Muse_Utils.Dec_SensorsFullScales(CommandResponse(response))
                    
                    client_2 = BleakClient(myDev_2)
                    await stack.enter_async_context(client_2)

                    # Get Muse n.2 device ID
                    await client_2.write_gatt_char(CMD_UUID,Muse_Utils.Cmd_GetDeviceID(),True)
                    response = await client_2.read_gatt_char(CMD_UUID)
                    device_2_ID = Muse_Utils.Dec_DeviceID(CommandResponse(response))
                    print("Device #2 successfully connected !")
                    # Get Muse n.2 sensors full scales
                    await client_2.write_gatt_char(CMD_UUID,Muse_Utils.Cmd_GetSensorsFullScale(),True)
                    response = await client_2.read_gatt_char(CMD_UUID)
                    gyrConfig2, axlConfig2, magConfig2, hdrConfig2 = Muse_Utils.Dec_SensorsFullScales(CommandResponse(response))


                    for i in range(stream_start_timeout,0,-1):
                        print("Streaming will start in :",f"{i}"," sec, get ready in position !!", end="\r\n", flush=True)
                        await asyncio.sleep(1)

                    await client_1.start_notify(DATA_UUID, partial(data_notification_handler, client_1))
                    await client_2.start_notify(DATA_UUID, partial(data_notification_handler, client_2))
                    
                    # Set up the command
                    stream_mode = MH.DataMode.DATA_MODE_IMU
                    cmd_stream = Muse_Utils.Cmd_StartStream(mode=stream_mode, frequency=MH.DataFrequency.DATA_FREQ_25Hz, enableDirect=True)

                    # Start Streaming using the above configuration (direct streaming, IMU mode and Sampling Frequency = 25 Hz)
                    await client_1.write_gatt_char(CMD_UUID, cmd_stream, True)
                    await client_2.write_gatt_char(CMD_UUID, cmd_stream, True)

                    # Wait for 'stream_duration_seconds' seconds
                    await asyncio.sleep(stream_duration_seconds)

                    # Stop the acquisition streaming
                    await client_1.write_gatt_char(CMD_UUID, Muse_Utils.Cmd_StopAcquisition(), True)
                    await client_2.write_gatt_char(CMD_UUID, Muse_Utils.Cmd_StopAcquisition(), True)

                    print("Streaming Stopped !")

                    # Unsubscribe from data characteristic
                    await client_1.stop_notify(DATA_UUID)
                    await client_2.stop_notify(DATA_UUID)

            # The stack context manager exits here, triggering disconnection!

        except Exception as e:
            print("Error during procedure: \n", e)

asyncio.run(main())

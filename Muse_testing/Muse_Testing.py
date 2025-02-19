
#EXP: EXP = Explanation

#EXP: Muse API
from Muse_Utils import *
from Muse_HW import *

#EXP: Bluetooth and async
from bleak import BleakScanner, BleakClient
import asyncio

#EXP: Command and Data characteristics
CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
DATA_UUID = "09bf2c52-d1d9-c0b7-4145-475964544307"

#EXP: Info-variables:
my_device_name = "Muse"
stream_duration_seconds = 30
stream_start_timeout_seconds = 3

#EXP: Handle data streamed from Muse
def data_notification_handler(sender, data):

  header_offset = 8

  tempData = Muse_Utils.DecodePacket(data[header_offset:], 0, stream_mode.value, gyrConfig.Sensitivity, axlConfig.Sensitivity, magConfig.Sensitivity, hdrConfig.Sensitivity)
  print("{0} {1} {2}".format(device_ID,tempData.tp[0],tempData.th[0]))

#EXP: Main function
async def main():

  #EXP: Global variables
  global client, device_ID
  global stream_mode, gyrConfig, axlConfig, magConfig, hdrConfig
  
  #EXP: Find devices, if correct device found: remember it
  devices = await BleakScanner.discover()
  myDevice = None
  for d in devices:
    print(d)
    if d.name == my_device_name:
      myDevice = d

  #EXP: If correct device is found: Do this
  if(myDevice != None):
    async with BleakClient(str(myDevice.address)) as client:
      #EXP: Get device ID
      await client.write_gatt_char(char_specifier=CMD_UUID, data=bytearray([0x82, 0x00]), response=True)
      response = await client.read_gatt_char(CMD_UUID)
      device_ID = Muse_Utils.Dec_DeviceID(CommandResponse(response))

      if(response[2] == 0x82 and response[3] == 0x00 and response[4] == 0x02):
        print("System in IDLE")

        await client.write_gatt_char(CMD_UUID, Muse_Utils.Cmd_GetSensorsFullScale(), True)
        response = await client.read_gatt_char(CMD_UUID)
        gyrConfig, axlConfig, magConfig, hdrConfig = Muse_Utils.Dec_SensorsFullScales(CommandResponse(response))

        await client.start_notify(DATA_UUID, data_notification_handler)

        stream_mode = MH.DataMode.DATA_MODE_AXL
        cmd_stream = Muse_Utils.Cmd_StartStream(mode=stream_mode, frequency=MH.DataFrequency.DATA_FREQ_25Hz, enableDirect=False)
        await client.write_gatt_char(CMD_UUID, cmd_stream, True)

        #EXP: How long stream lasts, then terminate stream
        await asyncio.sleep(stream_duration_seconds)
        await client.write_gatt_char(char_specifier=CMD_UUID, data=Muse_Utils.Cmd_StopAcquisition(MH.CommunicationChannel.CHANNEL_BLE), response=True)

      print("Disconnecting")
      await client.stop_notify(DATA_UUID)

  else:
    print("No corrent client was found")

asyncio.run(main())
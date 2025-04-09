""" Muse_Utils.py: Muse utilities for encoding and decoding of command and messages.

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

__authors__ = ["Luigi Mattiello", "Francesca Palazzo", "Roberto Bortoletto"]
__contact__ = "info@221e.com"
__copyright__ = "Copyright (c) 2020 by 221e srl."
__credits__ = ["Luigi Mattiello", "Francesca Palazzo", "Roberto Bortoletto"]
__deprecated__ = False
__email__ =  "roberto.bortoletto@221e.com"
__license__ = "GNU General Public License"
__maintainer__ = "Roberto Bortoletto"
__status__ = "Production"
__version__ = "2.0.0"

import copy
import ctypes
import datetime
import struct
import math

from muse_api_main.Muse_HW import Muse_HW as MH
from muse_api_main.Muse_Data import *
from typing import Optional

class Muse_Utils:
    """Utilities related to communication protocol specifications.
    """
    
    # USB WRAPPER FUNCTIONS

    @staticmethod
    def _WrapMessage(buffer):
        """
        Adds header and trailer to a standard command.
        .. code-block:: python

        Args:
            buffer:
                Input byte array to be wrapped.
        """
        wrapped_buffer = None
        if (buffer != None):
            #Define wrapped message buffer
            wrapped_buffer = [0]* (len(buffer)+4)
            
            #Add header to input buffer
            wrapped_buffer[0] = 0x3f # ?
            wrapped_buffer[1] = 0x21 # !

            #Copy main content
            buffer_copy = copy.copy(buffer)
            wrapped_buffer[2:] = buffer_copy
            
            #Add trailer to input buffer
            wrapped_buffer.append(0x21) # !
            wrapped_buffer.append(0x3f) # ?
        
        return wrapped_buffer

    @staticmethod
    def _ExtractMessage(buffer):
        """
        Removes header and trailer from a standard command.
        .. code-block:: python

        Args:
            buffer:
                Input byte array from which the header and trailer must be removed.
        """
        #Remove header and trailer from input buffer
        dewrapped_buffer = None
        if (buffer != None):
            #copio dal secondo carattere in poi, poi tolgo il trailer 
            dewrapped_buffer = copy.copy(buffer[2:])[:-2]
        return dewrapped_buffer

    # end of USB WRAPPER FUNCTIONS

    # ENCODING COMMAND IMPLEMENTATION
    
    @staticmethod
    def Cmd_Acknowledge(ack: MH.AcknowledgeType, channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Builds command to send Acknowledge.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_Acknowledge(ack=MH.AcnowledgeType.ACK_SUCCESS)
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            ack:
                Acknowledge type (ACK_SUCCESS = 0x00, ACK_ERROR = 0x01).
            channel:
                Communication channel over which the command will be sent.
        """
        # Definition of message buffer        
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_ACKNOWLEDGE)

        buffer[0] = 0x00
        
        #Set payload length
        buffer[1] = MH.CommandLength.CMD_LENGTH_ACKNOWLEDGE.value - 2

        buffer[2] = MH.Command.CMD_STATE.value
        buffer[3] = ack
        buffer[4] = MH.SystemState.SYS_CALIB.value
        
        return buffer
    
    @staticmethod
    def Cmd_GetSystemState(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Builds command to retrieve current system state.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_GetSystemState()
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_STATE)

        #Get state command
        buffer[0] = MH.Command.CMD_STATE.value + MH.READ_BIT_MASK

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_StartStream(mode: MH.DataMode, frequency: MH.DataFrequency, enableDirect = True, channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Builds command to start acquisition in stream mode.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_StartStream(MH.DataMode.DATA_MODE_AXL, MH.DataFrequency.DATA_FREQ_50Hz, enableDirect=True)
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            mode:
                Identifies the set of data to be acquired.
            frequency:
                Identifies the data acquisition frequency.
            enableDirect:
                Allows to select the stream type (i.e., direct=True or buffered=False).
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_START_STREAM)

        #Start stream acquisition using set state command
        buffer[0] = MH.Command.CMD_STATE.value

        #Set payload length
        buffer[1] = MH.CommandLength.CMD_LENGTH_START_STREAM.value - 2

        #Set tx type based on boolean flag value
        if (enableDirect):
            buffer[2] =  MH.SystemState.SYS_TX_DIRECT.value 
        else:
            buffer[2] = MH.SystemState.SYS_TX_BUFFERED.value

        #Set acquisition mode
        tmp = struct.pack("I", mode.value)
        buffer[3:6] = tmp[:3]

        #Set acquisition frequency
        buffer[6] = frequency.value

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer

    @staticmethod
    def Cmd_StartLog(mode: MH.DataMode, frequency: MH.DataFrequency, channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Builds command to stop any data acquisition procedure.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_StartLog(MH.DataMode.DATA_MODE_AXL, MH.DataFrequency.DATA_FREQ_50Hz)
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            mode:
                Identifies the set of data to be acquired.
            frequency:
                Identifies the data acquisition frequency.
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_START_LOG)

        #Start stream acquisition using set state command
        buffer[0] = MH.Command.CMD_STATE.value

        #Set payload length
        buffer[1] = MH.CommandLength.CMD_LENGTH_START_LOG.value - 2

        #Set state - LOG
        buffer[2] =  MH.SystemState.SYS_LOG.value

        #Set acquisition mode
        tmp = struct.pack("I", mode.value)
        buffer[3:6] = tmp[:3]
  
        #Set acquisition frequency
        buffer[6] = frequency.value

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_StopAcquisition(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Builds command to stop any data acquisition procedure.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_StopAcquisition()
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_STOP_ACQUISITION)

        #Set IDLE state to stop any acquisition procedure
        buffer[0] = MH.Command.CMD_STATE.value
        buffer[1] = MH.CommandLength.CMD_LENGTH_STOP_ACQUISITION.value - 2
        buffer[2] = MH.SystemState.SYS_IDLE.value

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer

    @staticmethod
    def Cmd_Restart(restartMode : MH.RestartMode, channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Builds command to set restart device.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            restart_mode = MH.RestartMode.APPLICATION
            cmd = Muse_Utils.Cmd_Restart(restart_mode)
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            restartMode:
                Allows to select the restart mode (i.e., APPLICATION, BOOT or RESET).
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_RESTART)
        
        #Set RESTART command with the specified restart mode (i.e., boot or app)
        buffer[0] = MH.Command.CMD_RESTART.value
        buffer[1] = MH.CommandLength.CMD_LENGTH_RESTART.value - 2
        buffer[2] = restartMode.value

        # Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_GetApplicationInfo(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Builds command to retrive application firmware information.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_GetApplicationInfo()
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_APP_INFO)

        #Get firmware application info
        buffer[0] = MH.Command.CMD_APP_INFO.value + MH.READ_BIT_MASK

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_GetBatteryCharge(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Builds command to retrieve battery charge level [%].
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_GetBatteryCharge()
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_BATTERY_CHARGE)

        #Get battery charge
        buffer[0] = MH.Command.CMD_BATTERY_CHARGE.value + MH.READ_BIT_MASK

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_GetBatteryVoltage(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Builds command to retrieve battery voltage level [mV].
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_GetBatteryVoltage()
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_BATTERY_VOLTAGE)

        #Get battery voltage
        buffer[0] = MH.Command.CMD_BATTERY_VOLTAGE.value + MH.READ_BIT_MASK

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer

    @staticmethod
    def Cmd_GetDeviceCheckup(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Builds command to retrieve system check-up register value.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_GetDeviceCheckup()
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_CHECK_UP)
        
        #Get check up register value
        buffer[0] = MH.Command.CMD_CHECK_UP.value + MH.READ_BIT_MASK

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_GetFirmwareVersion(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Builds command to retrieve current firmware versions (i.e., both bootloader and application).
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_GetFirmwareVersion()
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_FW_VERSION)

        #Get firmware version labels (i.e., bootloader and application firmware)
        buffer[0] = MH.Command.CMD_FW_VERSION.value + MH.READ_BIT_MASK

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer

    @staticmethod
    def Cmd_SetTime(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Builds command to update date/time.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_SetTime()
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_SET_TIME)
       
        #Set time using current timespan since 1970
        buffer[0] = MH.Command.CMD_TIME.value
        buffer[1] = MH.CommandLength.CMD_LENGTH_SET_TIME.value - 2

        #Get current timespan since 1/1/1970
        time_span = datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1, 0, 0, 0)
        seconds_since_epoch = ctypes.c_long(int(time_span.total_seconds())).value
        
        #Set payload - seconds since epoch (4 bytes)
        #pack takes values that aren't bytes and convert them into bytes
        payload = struct.pack("<I", seconds_since_epoch)
        
        buffer[2:] = payload[:]
        
        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer

    @staticmethod
    def Cmd_GetTime(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Builds command to retrieve current date/time.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_GetTime()
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_TIME)

        #Get current datetime
        buffer[0] = MH.Command.CMD_TIME.value + MH.READ_BIT_MASK

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_GetDeviceName(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Builds command to retrieve device name (i.e., the name advertised by the Bluetooth module).
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_GetDeviceName()
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            channel:
                Communication channel over which the command will be sent.
        """
             
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_BLE_NAME)

        #Get device name (i.e., ble name)
        buffer[0] = MH.Command.CMD_BLE_NAME.value + MH.READ_BIT_MASK

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer

    @staticmethod
    def Cmd_SetDeviceName(bleName, channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Builds command to change the device name (i.e., the name advertised by the Bluetooth module).
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_SetDeviceName(bleName="my_muse")
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            bleName:
                New device name to be set.
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_SET_BLE_NAME)
        
        #Check bleName to be set consistency before proceeding
        if (bleName != "" and len(bleName) < 16):
        
            buffer[0] = MH.Command.CMD_BLE_NAME.value

            respLen = len(bleName)
            buffer[1:1] = respLen.to_bytes(1, 'big')
            
            for i in range(len(bleName)):
                buffer[2 + i:] = bytearray(bleName[i],'utf-8')
                

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_GetDeviceID(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Builds command to retrieve the device unique identifier.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_GetDeviceID()
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_DEVICE_ID)

        #Get device unique identifier
        buffer[0] = MH.Command.CMD_DEVICE_ID.value + MH.READ_BIT_MASK

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_GetDeviceSkills(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Builds command to retrieve the devices skills (i.e. hardware and software features provided by the device).
        The response to this command will return a byte array containing 4 bytes corresponding to hardware skills and the next 4 bytes corresponding to software skills.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_GetDeviceSkills()
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_DEVICE_SKILLS)

        #Get device skills (i.e., hardware or software based on specified flag)
        buffer[0] = MH.Command.CMD_DEVICE_SKILLS.value + MH.READ_BIT_MASK
        buffer[1] = MH.CommandLength.CMD_LENGTH_GET_DEVICE_SKILLS.value - 2

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer

    @staticmethod
    def Cmd_GetMemoryStatus(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Builds command to retrieve current memory status.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_GetMemoryStatus()
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_MEM_CONTROL)

        # Get available memory
        buffer[0] = MH.Command.CMD_MEM_CONTROL.value + MH.READ_BIT_MASK

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_EraseMemory(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Builds command to erase device memory.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_EraseMemory()
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_SET_MEM_CONTROL)
        
        #Erase memory 
        bulk_erase = 1 # Parameter to erase all memory
        buffer[0] = MH.Command.CMD_MEM_CONTROL.value
        buffer[1] = MH.CommandLength.CMD_LENGTH_SET_MEM_CONTROL.value - 2
        buffer[2] = bulk_erase
        
        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer

    @staticmethod
    def Cmd_MemoryFileInfo(fileId, channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Builds command to retrieve file information.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_MemoryFileInfo(fileId=0)
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            fileId:
                File identifier.
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_MEM_FILE_INFO)

        #Retrieve file information given a specific file identifier
        buffer[0] = MH.Command.CMD_MEM_FILE_INFO.value + MH.READ_BIT_MASK
        buffer[1] = MH.CommandLength.CMD_LENGTH_GET_MEM_FILE_INFO.value - 2

        #Set file id as a 2-bytes unsigned integer value
        valueBytes = fileId.to_bytes(2, byteorder='little')
        buffer[2] = valueBytes[0]
        buffer[3] = valueBytes[1]

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_MemoryFileDownload(fileId, channel = MH.CommunicationChannel.CHANNEL_USB):
        """
        Builds command to activate a memory file offload procedure.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_MemoryFileDownload(fileId=0)
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            fileId:
                File identifier.
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_MEM_FILE_DOWNLOAD)

        #Start file offload procedure
        buffer[0] = MH.Command.CMD_MEM_FILE_DOWNLOAD.value
        buffer[1] = MH.CommandLength.CMD_LENGTH_GET_MEM_FILE_DOWNLOAD.value - 2

        #Set file identifier
        valueBytes = fileId.to_bytes(2, byteorder='little')
        buffer[2] = valueBytes[0]
        buffer[3] = valueBytes[1]

        #Set file offload channel (USB vs BLE)
        buffer[4] = channel.value
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            buffer[4] = 0x00; #set by default to USB channel (if 0x01 it manages the BLE transfer)

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_GetClockOffset(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """Builds command to retrieve current clock offset.

        Args:
            channel: Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_CLK_OFFSET)

        #Get clock offset
        buffer[0] = MH.Command.CMD_CLK_OFFSET.value + MH.READ_BIT_MASK

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_SetClockOffset(inOffset = 0, channel = MH.CommunicationChannel.CHANNEL_BLE):
        """Builds command to trigger a clock offset estimation procedure.

        Args:
            inOffset: Allows to specify a custom clock offset to be set.
            channel: Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_SET_CLK_OFFSET)

        #Set clock offset
        buffer[0] = MH.Command.CMD_CLK_OFFSET.value
        buffer[1] = MH.CommandLength.CMD_LENGTH_SET_CLK_OFFSET.value - 2

        valueBytes = inOffset.to_bytes(8, byteorder="little")
        buffer[9] = valueBytes[7]
        buffer[8] = valueBytes[6]
        buffer[7] = valueBytes[5]
        buffer[6] = valueBytes[4]
        buffer[5] = valueBytes[3]
        buffer[4] = valueBytes[2]
        buffer[3] = valueBytes[1]
        buffer[2] = valueBytes[0]

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_EnterTimeSync(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """Builds command to enter timesync routine.

        Args:
            channel: Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_ENTER_TIME_SYNC)

        # Start timesync procedure
        buffer[0] = MH.Command.CMD_TIME_SYNC.value

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_ExitTimeSync(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """Builds command to exit timesync routine.

        Args:
            channel: Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_EXIT_TIME_SYNC)

        #Stop timesync procedure
        buffer[0] = MH.Command.CMD_EXIT_TIME_SYNC.value

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_SetSensorsFullScale(gyrFS: MH.GyroscopeFS, axlFS: MH.AccelerometerFS, magFS: MH.MagnetometerFS, hdrFS : MH.AccelerometerHDRFS, channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Builds command to set a custom sensors full scale configuration.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            gyro_FS = MH.GyroscopeFS.GYR_FS_2000dps
            axl_FS = MH.AccelerometerFS.AXL_FS_32g
            mag_FS = MH.MagnetometerFS.MAG_FS_08G
            hdr_FS = MH.AccelerometerHDRFS.HDR_FS_100g
            cmd = Muse_Utils.Cmd_SetSensorsFullScale(gyro_FS, axl_FS, mag_FS, hdr_FS)
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            gyrFS:
                Gyroscope full scale code.
            axlFS:
                Accelerometer full scale code.
            magFS:
                Magnetometer full scale code.
            hdrFS:
                High Dynamic Range (HDR) Accelerometer code
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_SET_SENSORS_FS)

        #Set sensors full scale configuration
        buffer[0] = MH.Command.CMD_SENSORS_FS.value
        buffer[1] = MH.CommandLength.CMD_LENGTH_SET_SENSORS_FS.value - 2

        #Build integer configuration code to be sent
        buffer[2] = (axlFS.value | gyrFS.value | hdrFS.value | magFS.value)

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_GetSensorsFullScale(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Get the current stored sensors full scales.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_GetSensorsFullScale()
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_SENSORS_FS)

        #Get current sensors full scale configuration
        buffer[0] = MH.Command.CMD_SENSORS_FS.value + MH.READ_BIT_MASK

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    

    #CMD_CALIB_MATRIX = 0x48
    @staticmethod
    def Cmd_SetCalibrationMatrix(rowId: bytes, colId: bytes, r1: float,  r2: float,  r3: float, channel = MH.CommunicationChannel.CHANNEL_BLE):
    
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_SET_CALIB_MATRIX)

        #Set calibration matrix components
        buffer[0] = MH.Command.CMD_CALIB_MATRIX.value
        buffer[1] = MH.CommandLength.CMD_LENGTH_SET_CALIB_MATRIX.value - 2

        #Payload
        buffer[2] = rowId;                                  #Row index in the range 0-2
        buffer[3] = colId;                                  #Column index in the range 0-3

        # Values for a given row and col
        valByteArray = struct.pack("<f", r1)
        buffer[4:] = valByteArray[:]
        valByteArray = struct.pack("<f", r2)
        buffer[8:] = valByteArray[:]
        valByteArray = struct.pack("<f", r3)
        buffer[12:] = valByteArray[:]
        
        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        
        return buffer
    
    @staticmethod
    def Cmd_GetCalibrationMatrix(calibrationType: bytes, colId: bytes, channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Get a specific row of a calibration matrix.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_GetCalibrationMatrix(calibrationType=0x01, colId=0)
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            calibrationType:
                Define the MEMS calibration you want to retrieve (Accelerometer: 0x01, Gyroscope: 0x02, Magnetometer: 0x03).
            colId:
                Column index in the range 0-3.
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_CALIB_MATRIX)

        #Get current calibration matrix values
        buffer[0] = MH.Command.CMD_CALIB_MATRIX.value + MH.READ_BIT_MASK
        buffer[1] = MH.CommandLength.CMD_LENGTH_GET_CALIB_MATRIX.value - 2

        #Set row/col payload indexes to be retrieved
        buffer[2] = calibrationType
        buffer[3] = colId

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_SetButtonLogConfiguration(mode: MH.DataMode, frequency : MH.DataFrequency, channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Set the button log configuration.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_SetButtonLogConfiguration(mode=MH.DataMode.DATA_MODE_IMU, frequency=MH.DataFrequency.DATA_FREQ_50Hz)
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            mode:
                Identifies the set of data to be acquired.
            frequency:
                Identifies the data acquisition frequency.
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_SET_BUTTON_LOG)
        
        #Set log mode command
        buffer[0] = MH.Command.CMD_BUTTON_LOG.value
        buffer[1] = MH.CommandLength.CMD_LENGTH_SET_BUTTON_LOG.value - 2

        #Set log mode code
        tmp = struct.pack("I", mode.value)
        buffer[2:5] = tmp[:3]
        
        #Set log frequency
        buffer[5] = frequency.value

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        
        return buffer
    
    @staticmethod
    def Cmd_GetButtonLogConfiguration(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Get the current button log configuration.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_GetButtonLogConfiguration()
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_BUTTON_LOG)

        #Get current log mode
        buffer[0] = MH.Command.CMD_BUTTON_LOG.value + MH.READ_BIT_MASK

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_GetUserConfiguration(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Get the current user configuration parameters.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            cmd = Muse_Utils.Cmd_GetUserConfiguration()
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_USER_CONFIG)

        #Get current user configuration
        buffer[0] = MH.Command.CMD_USER_CONFIG.value + MH.READ_BIT_MASK

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_SetUserConfiguration(standby: Optional[bool]=None, memory:Optional[bool]=None, 
                                 streaming_channel:Optional[MH.StreamingChannel]=None, mpe9dof:Optional[bool]=None, 
                                 slowfreq:Optional[bool]=None, channel=MH.CommunicationChannel.CHANNEL_BLE):
        """
        Set the User Configuration.
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            # Enable Standby
            cmd = Muse_Utils.Cmd_SetUserConfiguration(standby = True)
            # Enable BLE channel for streaming
            cmd = Muse_Utils.Cmd_SetUserConfiguration(streaming_channel = MH.StreamingChannel.CHANNEL_BLE)
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            standby (bool):
                It allows to enable (True) or disable (False) automatic standby. If enabled, the device will move to standby condition after 120 seconds of inactivity. The default value is ENABLED (value: 1). The system can also be forced into standby by using the button when the automatic standby is disabled. If the device is connected, the system is prevented from going into standby condition.
            memory (bool):
                It allows to enable (True) or disable (False) memory management using a circular buffer instead of a linear one. The default value is DISABLED (value: 0). If enabled, the memory is treated as a circular buffer, which means that when the memory runs out, the oldest data is deleted to make space for new. This means that a log can never be interrupted by an end-of-memory condition.
            streaming_channel (Muse_HW.StreamingChannel):
                It allows to select the streaming channel. It is mutually exlusive among BLE, USB, TCP and MQTT.
            mpe9dof (bool):
                It allows to enable (True) or disable (False) the use of the magnetometer in the MPE. The default value is DISABLED (value: 0). If enabled, it indicates that the MPE algorithm will take into account the magnetometer in its orientation estimation.
            slowfreq (bool):
                It allows to enable (True) or disable (False) the use of the slow frequency mode.
            channel (Muse_HW.CommunicationChannel):
                Communication channel over which the command will be sent.
        
        """
        # Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_SET_USER_CONFIG)

        # Set a custom user configuration
        buffer[0] = MH.Command.CMD_USER_CONFIG.value
        buffer[1] = MH.CommandLength.CMD_LENGTH_SET_USER_CONFIG.value - 2

        # Initialize bit-mask
        bitMask = 0

        # Build configuration code to be sent using user input
        config_code = 0
        if not(standby is None):
            bitMask = bitMask + MH.UserConfigMask.USER_CFG_MASK_AUTO_STANDBY.value
            if (standby):
                config_code |= MH.UserConfigMask.USER_CFG_MASK_AUTO_STANDBY.value
        if not(memory is None):
            bitMask = bitMask + MH.UserConfigMask.USER_CFG_MASK_CIRCULAR_MEMORY.value
            if (memory):
                config_code |= MH.UserConfigMask.USER_CFG_MASK_CIRCULAR_MEMORY.value

        # STEAMING configurations are mutually exclusive
        if not(streaming_channel is None):
            
            bitMask = bitMask + MH.UserConfigMask.USER_CFG_MASK_STREAMING_CHANNEL.value
            
            mqttcommands=False
            if (streaming_channel == MH.StreamingChannel.CHANNEL_BLE):
                config_code |= MH.UserConfigMask.USER_CFG_MASK_BLE_STREAM.value
            elif (streaming_channel == MH.StreamingChannel.CHANNEL_USB):
                config_code |= MH.UserConfigMask.USER_CFG_MASK_USB_STREAM.value
            elif (streaming_channel == MH.StreamingChannel.CHANNEL_TCP):
                config_code |= MH.UserConfigMask.USER_CFG_MASK_TCP_STREAM.value
                mqttcommands = True
            elif (streaming_channel == MH.StreamingChannel.CHANNEL_MQTT):
                config_code |= MH.UserConfigMask.USER_CFG_MASK_MQTT_STREAM.value
                mqttcommands = True

            # Update mqtt commands flag
            bitMask = bitMask + MH.UserConfigMask.USER_CFG_MASK_MQTT_COMMANDS.value
            if (mqttcommands):
                config_code |= MH.UserConfigMask.USER_CFG_MASK_MQTT_COMMANDS.value
            

        if not(mpe9dof is None):
            bitMask = bitMask + MH.UserConfigMask.USER_CFG_MASK_9DOF_MPE.value
            if (mpe9dof):
                config_code |= MH.UserConfigMask.USER_CFG_MASK_9DOF_MPE.value
        if not(slowfreq is None):
            bitMask = bitMask + MH.UserConfigMask.USER_CFG_MASK_SLOW_FREQUENCY.value
            if (slowfreq):
                config_code |= MH.UserConfigMask.USER_CFG_MASK_SLOW_FREQUENCY.value
        
        # Set bit-mask
        tmp = bitMask.to_bytes(2, byteorder='little')
        buffer[2:4] = tmp

        # Set configuration code
        tmp = config_code.to_bytes(2, byteorder='little')
        buffer[4:6] = tmp

        # Wrap message with header and trailer in the case of USB communication
        if channel == MH.CommunicationChannel.CHANNEL_USB:
            return Muse_Utils._WrapMessage(buffer)

        return bytes(buffer)

    # WiFi commands

    @staticmethod
    def Cmd_SetWifiSSIDHead(ssid: str, channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Set the SSID for WiFi connection (header part).
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            SSID_string = "221e_wifi"
            cmd = Muse_Utils.Cmd_SetWifiSSIDHead(ssid = SSID_string)
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            ssid (str):
                It is a 2 up to 32 bytes hex string representing the SSID to be set.
            channel:
                Communication channel over which the command will be sent.
        """
        # Definition of message buffer
        buffer = bytearray()
        
        #Check bleName to be set consistency before proceeding
        if (ssid != "" and len(ssid) <= 18):
        
            # Definition of message buffer
            buffer = bytearray(len(ssid))

            # Set the SSID of the network – head part
            buffer[0] = MH.Command.CMD_WIFI_SSID_HEAD.value
            buffer[1] = len(ssid)

            for i in range(len(ssid)):
                buffer[2 + i:] = bytearray(ssid[i],'utf-8')

            # Wrap message with header and trailer in the case of USB communication
            if channel == MH.CommunicationChannel.CHANNEL_USB:
                return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_SetWifiSSIDCont(ssid: str, channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Set the SSID for WiFi connection (continuation part).
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            SSID_string = "221e_wifi_with_long_SSID"
            # Send the first part of SSID to be set
            cmd = Muse_Utils.Cmd_SetWifiSSIDHead(ssid = SSID_string[:18])
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)
            # Send the remaining part of SSID to be set
            cmd = Muse_Utils.Cmd_SetWifiSSIDCont(ssid = SSID_string[18:])
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            ssid (str):
                It is a 2 up to 32 bytes hex string representing the SSID to be set.
            channel:
                Communication channel over which the command will be sent.
        """
        # Definition of message buffer
        buffer = bytearray()
        
        #Check bleName to be set consistency before proceeding
        if (ssid != "" and len(ssid) <= 14):
        
            # Definition of message buffer
            buffer = bytearray(len(ssid))

            # Set the SSID of the network – head part
            buffer[0] = MH.Command.CMD_WIFI_SSID_CONT.value
            buffer[1] = len(ssid)

            for i in range(len(ssid)):
                buffer[2 + i:] = bytearray(ssid[i],'utf-8')

            # Wrap message with header and trailer in the case of USB communication
            if channel == MH.CommunicationChannel.CHANNEL_USB:
                return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_SetWifiPassowrdHead(password: str, channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Set the password for WiFi connection (header part).
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            password_string = "221e_wifiPassword"
            cmd = Muse_Utils.Cmd_SetWifiPassowrdHead(password = password_string)
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            password (str):
                It is a hex string representing the password to be set.
            channel:
                Communication channel over which the command will be sent.
        """
        # Definition of message buffer
        buffer = bytearray()
        
        #Check bleName to be set consistency before proceeding
        if (password != "" and len(password) <= 18):
        
            # Definition of message buffer
            buffer = bytearray(len(password))

            # Set the Password of the network – head part
            buffer[0] = MH.Command.CMD_WIFI_PSW_HEAD.value
            buffer[1] = len(password)

            for i in range(len(password)):
                buffer[2 + i:] = bytearray(password[i],'utf-8')

            # Wrap message with header and trailer in the case of USB communication
            if channel == MH.CommunicationChannel.CHANNEL_USB:
                return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_SetWifiPassowrdCont(password: str, channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Set the password for WiFi connection (header part).
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            password_string = "221e_very_long_wifi_password"
            # Send the first part of the password to be set
            cmd = Muse_Utils.Cmd_SetWifiPassowrdHead(password = password_string[:18])
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)
            # Send continuation part of the password to be set
            cmd = Muse_Utils.Cmd_SetWifiPassowrdCont(password = password_string[18:])
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            password (str):
                It is a hex string representing the password to be set.
            channel:
                Communication channel over which the command will be sent.
        """
        # Definition of message buffer
        buffer = bytearray()
        
        #Check bleName to be set consistency before proceeding
        if (password != "" and len(password) <= 18):
        
            # Definition of message buffer
            buffer = bytearray(len(password))

            # Set the Password of the network – head part
            buffer[0] = MH.Command.CMD_WIFI_PSW_CONT.value
            buffer[1] = len(password)

            for i in range(len(password)):
                buffer[2 + i:] = bytearray(password[i],'utf-8')

            # Wrap message with header and trailer in the case of USB communication
            if channel == MH.CommunicationChannel.CHANNEL_USB:
                return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_SetWifiStreamHostHead(hostname: str, channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Set the hostname for WiFi connection (header part).
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            host = "192.168.137.1"
            cmd = Muse_Utils.Cmd_SetWifiStreamHostHead(hostname = host)
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            hostname (str):
                It is a hex string representing the hostname to be set.
            channel:
                Communication channel over which the command will be sent.
        """
        # Definition of message buffer
        buffer = bytearray()
        
        #Check bleName to be set consistency before proceeding
        if (hostname != "" and len(hostname) <= 18):
        
            # Definition of message buffer
            buffer = bytearray(len(hostname))

            # Set the Host of the network – head part
            buffer[0] = MH.Command.CMD_WIFI_STREAM_HOST_HEAD.value
            buffer[1] = len(hostname)

            for i in range(len(hostname)):
                buffer[2 + i:] = bytearray(hostname[i],'utf-8')

            # Wrap message with header and trailer in the case of USB communication
            if channel == MH.CommunicationChannel.CHANNEL_USB:
                return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_SetWifiStreamHostCont(hostname: str, channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Set the hostname for WiFi connection (continuation part).
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            host = "my_very_long_hostname.com"
            # header part
            cmd = Muse_Utils.Cmd_SetWifiStreamHostHead(hostname = host[:18])
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)
            # continuation part
            cmd = Muse_Utils.Cmd_SetWifiStreamHostCont(hostname = host[18:])
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            hostname (str):
                It is a hex string representing the continuation part of hostname to be set.
            channel:
                Communication channel over which the command will be sent.
        """
        # Definition of message buffer
        buffer = bytearray()
        
        #Check bleName to be set consistency before proceeding
        if (hostname != "" and len(hostname) <= 18):
        
            # Definition of message buffer
            buffer = bytearray(len(hostname))

            # Set the Host of the network – head part
            buffer[0] = MH.Command.CMD_WIFI_STREAM_HOST_CONT.value
            buffer[1] = len(hostname)

            for i in range(len(hostname)):
                buffer[2 + i:] = bytearray(hostname[i],'utf-8')

            # Wrap message with header and trailer in the case of USB communication
            if channel == MH.CommunicationChannel.CHANNEL_USB:
                return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_SetWifiStreamHostPort(port: int, channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Set the hostname for WiFi connection (header part).
        .. code-block:: python

            CMD_UUID = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
            host_port = 1883
            cmd = Muse_Utils.Cmd_GetWifiStreamHostHead(port = host_port)
            await client.write_gatt_char(char_specifier=CMD_UUID, data=cmd, response=True)

        Args:
            port (int):
                It is a 16-bit unsigned integer representing the host port to which the muse device will be connected to.
            channel:
                Communication channel over which the command will be sent.
        """
        # Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_SET_WIFI_STREAM_HOST_PORT)

        # Set the Host of the network – head part
        buffer[0] = MH.Command.CMD_WIFI_STREAM_HOST_PORT.value
        buffer[1] = MH.CommandLength.CMD_LENGTH_SET_WIFI_STREAM_HOST_PORT.value - 2

        payload = struct.pack('<H', port)
        
        buffer[2:] = payload[:]

        # Wrap message with header and trailer in the case of USB communication
        if channel == MH.CommunicationChannel.CHANNEL_USB:
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_GetWifiSSIDHead(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Get the SSID currently set for WiFi connection (header part).
        .. code-block:: python

            SSID_head = Muse_Utils.Cmd_GetWifiSSIDHead(channel = MH.CommunicationChannel.CHANNEL_BLE)
            print(SSID_head)

        Args:
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_WIFI_SSID_HEAD)

        #Get device name (i.e., ble name)
        buffer[0] = MH.Command.CMD_WIFI_SSID_HEAD.value + MH.READ_BIT_MASK

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_GetWifiSSIDCont(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Get the SSID currently set for WiFi connection (continuation part).
        .. code-block:: python

            SSID_continuation = Muse_Utils.Cmd_GetWifiSSIDCont(channel = MH.CommunicationChannel.CHANNEL_BLE)
            print(SSID_continuation)

        Args:
            channel:
                Communication channel over which the command will be sent.
        """
        #Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_WIFI_SSID_CONT)

        buffer[0] = MH.Command.CMD_WIFI_SSID_CONT.value + MH.READ_BIT_MASK

        #Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_GetWifiStreamHostHead(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Get the hostname currently set for WiFi connection (header part).
        .. code-block:: python

            host_head = Muse_Utils.Cmd_GetWifiStreamHostHead(channel = MH.CommunicationChannel.CHANNEL_BLE)
            print(host_head)

        Args:
            channel:
                Communication channel over which the command will be sent.
        """
        # Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_WIFI_STREAM_HOST_HEAD)

        # Set the header part of Host of the network
        buffer[0] = MH.Command.CMD_WIFI_STREAM_HOST_HEAD.value + MH.READ_BIT_MASK
        buffer[1] = MH.CommandLength.CMD_LENGTH_GET_WIFI_STREAM_HOST_HEAD.value - 2

        # Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_GetWifiStreamHostCont(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Get the hostname currently set for WiFi connection (continuation part).
        .. code-block:: python

            host_continuation = Muse_Utils.Cmd_GetWifiStreamHostCont(channel = MH.CommunicationChannel.CHANNEL_BLE)
            print(host_continuation)

        Args:
            channel:
                Communication channel over which the command will be sent.
        """
        # Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_WIFI_STREAM_HOST_CONT)

        # Set the continuation part of Host of the network
        buffer[0] = MH.Command.CMD_WIFI_STREAM_HOST_CONT.value + MH.READ_BIT_MASK
        buffer[1] = MH.CommandLength.CMD_LENGTH_GET_WIFI_STREAM_HOST_CONT.value - 2

        # Wrap message with header and trailer in the case of USB communication
        if (channel == MH.CommunicationChannel.CHANNEL_USB):
            return Muse_Utils._WrapMessage(buffer)

        return buffer
    
    @staticmethod
    def Cmd_GetWifiStreamHostPort(channel = MH.CommunicationChannel.CHANNEL_BLE):
        """
        Get the host port set for WiFi connection.
        .. code-block:: python

            port = Muse_Utils.Cmd_GetWifiStreamHostPort(channel = MH.CommunicationChannel.CHANNEL_BLE)
            print(port)

        Args:
            channel:
                Communication channel over which the command will be sent.
        """
        # Definition of message buffer
        buffer = bytearray(MH.CommandLength.CMD_LENGTH_GET_WIFI_STREAM_HOST_PORT)

        # Set the Port of the network
        buffer[0] = MH.Command.CMD_WIFI_STREAM_HOST_PORT.value + MH.READ_BIT_MASK
        buffer[1] = MH.CommandLength.CMD_LENGTH_GET_WIFI_STREAM_HOST_PORT.value - 2

        # Wrap message with header and trailer in the case of USB communication
        if channel == MH.CommunicationChannel.CHANNEL_USB:
            return Muse_Utils._WrapMessage(buffer)

        return buffer
 
    # end ENCODING COMMAND IMPLEMENTATION

    # DECODING FUNCTIONS
    
    @staticmethod
    def ParseCommandCharacteristic(channel: MH.CommunicationChannel, buffer: bytearray):
        """
        Parse command characteristic to get a command response object. Manage also header and trailer removal in case of BLE or USB channel

        Args:
            channel:
                Communication channel over which the command will be sent.
            buffer (bytearray):
                Byte array to be parsed.

        Returns:
            response:
                CommandResponse type output
        """               
        if (channel == MH.CommunicationChannel.CHANNEL_BLE):
            response = CommandResponse(buffer)
        else:
            response = CommandResponse(Muse_Utils._ExtractMessage(buffer))
    
        return response

    @staticmethod
    def Dec_SystemState(response: CommandResponse):
        """
        Decode system state.

        Args:
            response:
                CommandResponse object to be decoded.

        Returns:
            state:
                Output reference to the SystemState value.
        """ 
        state = MH.SystemState.SYS_NONE.value
        
        #Decode system state given command response
        if ((response.tx & 0x7F) == MH.Command.CMD_STATE.value and
            response.ack == MH.AcknowledgeType.ACK_SUCCESS):
                
            state = response.payload[0]
            
        return state
    
    @staticmethod
    def Dec_ApplicationInfo(response: CommandResponse):
        """
        Decode firmware application information.

        Args:
            response:
                CommandResponse object to be decoded.

        Returns:
            (crc, length):
                - Output reference to the Circular Redundancy Check (CRC) as 32-bit unsigned integer.
                - Output reference to the length of the application firmware (i.e., number of bytes), as a 32-bit unsigned integer.
        """ 
        crc = 0
        length = 0

        #Decode firmware application info given command response payload
        if ( response.tx == MH.Command.CMD_APP_INFO and
            response.ack== MH.AcknowledgeType.ACK_SUCCESS.value):
        
            crc = int.from_bytes(response.payload[:4], byteorder='little', signed=False) 
            length = int.from_bytes(response.payload[4: ], byteorder='little', signed=False)
            
        return crc, length
    
    @staticmethod
    def Dec_BatteryCharge(response: CommandResponse):
        """
        Decode battery charge level.

        Args:
            response:
                CommandResponse object to be decoded.

        Returns:
            charge:
                Output reference to battery charge value [%].
        """ 
        charge = -1

        #Decode battery charge percentage value given command response payload
        if (response.tx  == MH.Command.CMD_BATTERY_CHARGE and
            response.ack == MH.AcknowledgeType.ACK_SUCCESS.value):
        
            charge = response.payload[0]
            
        return charge
    
    @staticmethod
    def Dec_BatteryVoltage(response: CommandResponse):
        """
        Decode battery voltage level.

        Args:
            response:
                CommandResponse object to be decoded.

        Returns:
            voltage:
                Output reference to battery voltage value [mV].
        """ 
        #Decode battery voltage [mV] value given command response payload
        if (response.tx == MH.Command.CMD_BATTERY_VOLTAGE and
            response.ack == MH.AcknowledgeType.ACK_SUCCESS.value):

            voltage = struct.unpack('<H', response.payload)[0]
            
            return voltage

    @staticmethod
    def Dec_CheckUp(response: CommandResponse):
        """
        Decode check-up register code.

        Args:
            response:
                CommandResponse object to be decoded.

        Returns:
            checkup:
                Output reference to check-up register code.
        """ 
        #Decode checkup register value, as string, given the command response payload
        tmpOut = ""

        #Decode current checkup register value given command response payload
        if (response.tx == MH.Command.CMD_CHECK_UP and
            response.ack == MH.AcknowledgeType.ACK_SUCCESS):
    
            for i in range(response.len-2):
                tmpOut += format(response.payload[i], '02X') + ' '
        
        checkup = tmpOut

        return checkup
    
    @staticmethod
    def Dec_FirmwareVersion(response: CommandResponse):
        """
        Decode firmware version labels.

        Args:
            response:
                CommandResponse object to be decoded.

        Returns:
            (bootrev, apprev):
                - bootrev: Output reference to boot loader firmware version label.
                - apprev: Output reference to application firmware version label.
        """ 
        boot_rev = "x.x.x"
        app_rev = "x.x.x"

        #Decode current firmware versions given command response payload
        if (response.tx == MH.Command.CMD_FW_VERSION and
            response.ack == MH.AcknowledgeType.ACK_SUCCESS):
        
            #Decode bootloader and application firmware versions
            words = str(response.payload, 'utf-8').replace('\0', ' ').split()

            boot_rev = words[0]
            app_rev = str(response.payload[7]) + '.' + str(response.payload[8]) + '.' + str(response.payload[9]) # words[1]
        
        return boot_rev, app_rev
    
    @staticmethod
    def Dec_DateTime(response: CommandResponse):
        """
        Decode date/time.

        Args:
            response:
                CommandResponse object to be decoded.

        Returns:
            current_time:
                Output reference to the current date/time.
        """ 
        time = datetime.datetime(1970, 1, 1, 0, 0, 0)

        #Decode current Date/Time given command response payload
        if (response.tx == MH.Command.CMD_TIME and
            response.ack == MH.AcknowledgeType.ACK_SUCCESS):
    
            epoch = datetime.datetime(1970, 1, 1) # UTC time
            time = epoch + datetime.timedelta(seconds=int.from_bytes(response.payload, byteorder='little'))
        
        return time

    @staticmethod
    def Dec_DeviceName(response: CommandResponse):
        """
        Decode device name (i.e., it is the name advertised by Bluetooth module).

        Args:
            response:
                CommandResponse object to be decoded.

        Returns:
            name:
                Output reference to device name.
        """
        name = "-"

        #Decode current device name (i.e., ble name) given command response payload
        if (response.tx == MH.Command.CMD_BLE_NAME.value and
            response.ack == MH.AcknowledgeType.ACK_SUCCESS):
        
            name = str(response.payload, 'utf-8')
            
        return name
    
    @staticmethod
    def Dec_DeviceID(response: CommandResponse):
        """
        Decode device unique identifier.

        Args:
            response:
                CommandResponse object to be decoded.

        Returns:
            id:
                Output reference to device unique identifier.
        """
        id = ""

        #Decode current device unique identifier given command response payload
        if (response.tx == MH.Command.CMD_DEVICE_ID and
            response.ack == MH.AcknowledgeType.ACK_SUCCESS):
        
                #Revert array before printing string in order to keep endianness in byte representation
                response.payload.reverse
                for i in range(response.len - 2):
                    id += format(response.payload[i], '02X')
            
        return id
    
    @staticmethod
    def Dec_DeviceSkills(response: CommandResponse):
        """
        Decode hardware and software skills.

        Args:
            response:
                CommandResponse object to be decoded.

        Returns:
            hw_skills:
                Output reference to hardware skills.
            sw_skills:
                Output reference to software skills.
        """
        hw_skills = {}
        sw_skills = {}

        #Get skills code to be parsed
        skillsCode = struct.unpack("<I", response.payload[:4])[0]

        #Extract hardware skills given the overall skills code
        if ((skillsCode & MH.HardwareSkills.SKILLS_HW_GYRO.value) > 0):
            hw_skills.update({MH.HardwareSkills.SKILLS_HW_GYRO.value: "GYR"})

        if ((skillsCode & MH.HardwareSkills.SKILLS_HW_AXL.value) > 0):
            hw_skills.update({MH.HardwareSkills.SKILLS_HW_AXL.value: "AXL"})

        if ((skillsCode & MH.HardwareSkills.SKILLS_HW_MAGN.value) > 0):
            hw_skills.update({MH.HardwareSkills.SKILLS_HW_MAGN.value: "MAG"})

        if ((skillsCode & MH.HardwareSkills.SKILLS_HW_HDR.value) > 0):
            hw_skills.update({MH.HardwareSkills.SKILLS_HW_HDR.value: "HDR"})

        if ((skillsCode & MH.HardwareSkills.SKILLS_HW_TEMP.value) > 0):
            hw_skills.update({MH.HardwareSkills.SKILLS_HW_TEMP.value: "TEMP"})

        if ((skillsCode & MH.HardwareSkills.SKILLS_HW_RH.value) > 0):
            hw_skills.update({MH.HardwareSkills.SKILLS_HW_RH.value: "RH"})

        if ((skillsCode & MH.HardwareSkills.SKILLS_HW_BAR.value) > 0):
            hw_skills.update({MH.HardwareSkills.SKILLS_HW_BAR.value: "BAR"})

        if ((skillsCode & MH.HardwareSkills.SKILLS_HW_LUM_VIS.value) > 0):
            hw_skills.update({MH.HardwareSkills.SKILLS_HW_LUM_VIS.value: "LUM/VIS"})
   
        if ((skillsCode & MH.HardwareSkills.SKILLS_HW_LUM_IR.value) > 0):
            hw_skills.update({MH.HardwareSkills.SKILLS_HW_LUM_IR.value: "LUM/IR"})
   
        if ((skillsCode & MH.HardwareSkills.SKILLS_HW_RANGE.value) > 0):
            hw_skills.update({MH.HardwareSkills.SKILLS_HW_RANGE.value: "RANGE"})
        
        if ((skillsCode & MH.HardwareSkills.SKILLS_HW_MIC.value) > 0):    
            hw_skills.update({MH.HardwareSkills.SKILLS_HW_MIC.value: "MIC"})
 
        
        #Get skills code to be parsed
        skillsCode = struct.unpack("<I", response.payload[4:])[0]

        #Extract software skills given the overall skills code
        if ((skillsCode & MH.SoftwareSkills.SKILLS_SW_MPE.value) > 0):
            sw_skills.update({MH.SoftwareSkills.SKILLS_SW_MPE.value: "MPE"})

        if ((skillsCode & MH.SoftwareSkills.SKILLS_SW_MAD.value) > 0):
            sw_skills.update({MH.SoftwareSkills.SKILLS_SW_MAD.value: "MAD"})

        return hw_skills, sw_skills

    @staticmethod
    def Dec_MemoryStatus(response: CommandResponse):
        """
        Decode memory status information.

        Args:
            response:
                CommandResponse object to be decoded.

        Returns:
            (available_memory, number_of_files):
                - Output reference to available memory.
                - Output reference to number of files currently saved in memory.
        """
        available_memory = 100
        number_of_files = 0

        #Decode current memory status given command response payload
        if (response.tx == MH.Command.CMD_MEM_CONTROL and
            response.ack == MH.AcknowledgeType.ACK_SUCCESS):
        
                #Get available free memory (i.e., percentage value)
                available_memory = response.payload[0]

                #Get number of files currently saved in memory
                
                number_of_files = int.from_bytes(response.payload[1:3], byteorder='little', signed=False)   

        return available_memory, number_of_files

    @staticmethod
    def Dec_FileInfo(response: CommandResponse):
        """
        Decode file information.

        Args:
            response:
                CommandResponse object to be decoded.

        Returns:
            file_info:
                Output reference to a FileInfo structure.
        """
        #Decode current file info given command response payload
        if (response.tx == MH.Command.CMD_MEM_FILE_INFO and
            response.ack == MH.AcknowledgeType.ACK_SUCCESS):
        
                #Decode file timestamp
                tmp = bytearray(8)
                tmp[:5] = response.payload[:5]
                ts = struct.unpack("<Q", tmp)[0]
                ts += MH.REFERENCE_EPOCH * 1000

                #Decode sensors full scales

                #Pad 1-bytes response with further 3-bytes before converting to UInt32 and extracting configuration codes
                tmp = bytearray(4)
                tmp[0] = response.payload[5]
                code = struct.unpack("<I", tmp)[0]
                #DecodeMEMSConfiguration(response.payload[5], out gyrConfig, out axlConfig, out magConfig, out hdrConfig);
                gyrConfig, axlConfig, magConfig,  hdrConfig = Muse_Utils.DecodeMEMSConfiguration(code)

                #Decode data acquisition mode
                tmp = bytearray(4)
                tmp[:3] = response.payload[6:9]
                dm = struct.unpack("<I", tmp)[0]

                #Update file_info object
                file_info = FileInfo(ts, gyrConfig, axlConfig, magConfig, hdrConfig, dm, response.payload[9])

                return file_info
            
    @staticmethod    
    def Dec_ClockOffset(response: CommandResponse):
        """
        Decode clock offset.

        Args:
            response:
                CommandResponse object to be decoded.

        Returns:
            clock_offset:
                Output reference to clock offset.
        """
        clock_offset = 0

        #Decode current file info given command response payload
        if (response.tx == MH.Command.CMD_CLK_OFFSET and
            response.ack == MH.AcknowledgeType.ACK_SUCCESS):
            
            tmp = bytearray(8)
            tmp[:5] = response.payload[:5]
            clock_offset = struct.unpack("<Q", tmp)[0]
        
        return clock_offset

    @staticmethod
    def Dec_SensorsFullScales(response: CommandResponse):
        """
        Decode sensors full scale / sensitivity configuration.

        Args:
            response:
                CommandResponse object to be decoded.

        Returns:
            (gyrConfig, axlConfig, magConfig, hdrConfig):
                - gyrConfig: Output reference to Gyroscope configuration.
                - axlConfig: Output reference to Accelerometer configuration.
                - magConfig: Output reference to Magnetometer configuration.
                - hdrConfig: Output reference to High Dynamic Range (HDR) Accelerometer configuration.
        """
        gyrConfig = SensorConfig(0,0)
        axlConfig = SensorConfig(0,0)
        magConfig = SensorConfig(0,0)
        hdrConfig = SensorConfig(0,0)

        #Decode current sensors full scales (i.e., gyr / axl / hdr / mag) given command response payload
        if (response.tx == MH.Command.CMD_SENSORS_FS and
            response.ack == MH.AcknowledgeType.ACK_SUCCESS):
        
        
                #Pad 3-bytes response.payload array with a further byte before converting to UInt32 and extracting configuration codes
                tmp = bytearray(4)
                tmp = response.payload[:3]
                code = struct.unpack("<I", tmp + b"\x00")[0]

                #Extract MEMS configurations
                #DecodeMEMSConfiguration(response.payload[0], out gyrConfig, out axlConfig, out magConfig, out hdrConfig);
                gyrConfig, axlConfig, magConfig, hdrConfig =  Muse_Utils.DecodeMEMSConfiguration(code)
            
        

        return gyrConfig, axlConfig, magConfig, hdrConfig
    
    @staticmethod
    def Dec_CalibrationMatrixValues(col_val: bytearray):
        """
        Decode calibration matric values

        Args:
            col_val (bytearray):
                bytearray containing column values to be decoded.

        Returns:
            col_values:
                float list of decoded column values.
        """
        col_values = [0.0, 0.0, 0.0]

        for i in range(3):
            start_index = i * 4
            tmp = col_val[start_index:start_index+4]
            col_values[i] = struct.unpack("<f", tmp)[0]

        return col_values
    
    @staticmethod
    def Dec_ButtonLogConfiguration(response: CommandResponse):
        """
        Decode button log configuration.

        Args:
            response:
                CommandResponse object to be decoded.

        Returns:
            (mode, frequency):
                - mode: Output reference to current acquisition mode configured.
                - frequency: Output reference to current acquisition frequency configured.
        """
        mode = ""
        frequency = ""

        #Decode current sensors full scales (i.e., gyr / axl / hdr / mag) given command response payload
        if (response.tx  == MH.Command.CMD_BUTTON_LOG and
            response.ack == MH.AcknowledgeType.ACK_SUCCESS):
    
                #Pad 3-bytes response.payload array with a further byte before converting to UInt32
                tmp = bytearray(4)
                tmp = response.payload[:3]
                code = struct.unpack("<I", tmp + b"\x00")[0]

                #Build acquisition mode string description
                mode = Muse_Utils.DataModeToString(code)

                #Set acquisition frequency string representation
                frequency = str(MH.DataFrequency(response.payload[3]).value)

        return mode, frequency
            
    @staticmethod
    def Dec_UserConfiguration(response: CommandResponse):
        """
        Decode user configuration packet.

        Args:
            response:
                CommandResponse object to be decoded.

        Returns:
            userConfig:
                Output reference to user configuration object (type UserConfig).
        """
        #Get code from which the user configuration must be extracted
        code = struct.unpack('<H', response.payload)[0]

        #Set internal boolean flags
        standby = False 
        memory = False 
        ble = False 
        usb = False 
        tcp = False 
        mqtt = False 
        mpe9dof = False 
        slowfreq = False 
        mqttcommands = False

        if ((code & MH.UserConfigMask.USER_CFG_MASK_AUTO_STANDBY.value) > 0):
            standby = True
        if ((code & MH.UserConfigMask.USER_CFG_MASK_CIRCULAR_MEMORY.value) > 0):
            memory = True 
        if ((code & MH.UserConfigMask.USER_CFG_MASK_STREAMING_CHANNEL.value) == MH.UserConfigMask.USER_CFG_MASK_BLE_STREAM.value):
            ble = True 
        if ((code & MH.UserConfigMask.USER_CFG_MASK_STREAMING_CHANNEL.value) == MH.UserConfigMask.USER_CFG_MASK_USB_STREAM.value):
            usb = True 
        if ((code & MH.UserConfigMask.USER_CFG_MASK_STREAMING_CHANNEL.value) == MH.UserConfigMask.USER_CFG_MASK_TCP_STREAM.value):
            tcp = True 
        if ((code & MH.UserConfigMask.USER_CFG_MASK_STREAMING_CHANNEL.value) == MH.UserConfigMask.USER_CFG_MASK_MQTT_STREAM.value):
            mqtt = True 
        if ((code & MH.UserConfigMask.USER_CFG_MASK_9DOF_MPE.value) > 0):
            mpe9dof = True 
        if ((code & MH.UserConfigMask.USER_CFG_MASK_SLOW_FREQUENCY.value) > 0):
            slowfreq = True
        if ((code & MH.UserConfigMask.USER_CFG_MASK_MQTT_COMMANDS.value) > 0): 
            mqttcommands = True

        return UserConfig(standby, memory, ble, usb, tcp, mqtt, mpe9dof, slowfreq, mqttcommands)
    
    @staticmethod
    def GetPacketDimension(inMode: MH.DataMode) -> int:
        """
        Returns the data packet dimension corresponding to a given data acquisition mode.

        Args:
            inMode:
                Data acquisition mode as DataMode type.

        Returns:
            packet_dimension:
                A integer value representing the data packet dimension.
        """
        packet_dimension = 0

        #Compute packet dimension given data acquisition mode
        if ((inMode.value & MH.DataMode.DATA_MODE_GYRO.value) > 0):
            packet_dimension += int(MH.DataSize.DATA_SIZE_GYRO.value)
        if ((inMode.value & MH.DataMode.DATA_MODE_AXL.value) > 0):
            packet_dimension += int(MH.DataSize.DATA_SIZE_AXL.value)
        if ((inMode.value & MH.DataMode.DATA_MODE_HDR.value) > 0):
            packet_dimension += int(MH.DataSize.DATA_SIZE_HDR.value)
        if ((inMode.value & MH.DataMode.DATA_MODE_MAGN.value) > 0):
            packet_dimension += int(MH.DataSize.DATA_SIZE_MAGN.value)
        if ((inMode.value & MH.DataMode.DATA_MODE_ORIENTATION.value) > 0):
            packet_dimension += int(MH.DataSize.DATA_SIZE_ORIENTATION.value)
        if ((inMode.value & MH.DataMode.DATA_MODE_TIMESTAMP.value) > 0):
            packet_dimension += int(MH.DataSize.DATA_SIZE_TIMESTAMP.value)
        if ((inMode.value & MH.DataMode.DATA_MODE_TEMP_HUM.value) > 0):
            packet_dimension += int(MH.DataSize.DATA_SIZE_TEMP_HUM.value)
        if ((inMode.value & MH.DataMode.DATA_MODE_TEMP_PRESS.value) > 0):
            packet_dimension += int(MH.DataSize.DATA_SIZE_TEMP_PRESS)
        if ((inMode.value & MH.DataMode.DATA_MODE_RANGE.value) > 0):
            packet_dimension += int(MH.DataSize.DATA_SIZE_RANGE.value)
        if ((inMode.value & MH.DataMode.DATA_MODE_SOUND.value) > 0):
            packet_dimension += int(MH.DataSize.DATA_SIZE_SOUND.value)

        #Check packet dimension consistency
        #It MUST be a dividend of 120 (2040)
        if (packet_dimension == 6 or packet_dimension == 12 or packet_dimension == 24 or
                packet_dimension == 30 or packet_dimension == 60):
            return packet_dimension

        #Return -1 in case of data dimension inconsistency
        return -1
    
    @staticmethod
    def GetNumberOfPackets(inMode: MH.DataMode):
        """
        Returns the overall number of data packets contained into a 128-bytes data characteristic given a data acquisition mode.

        Args:
            inMode:
                Data acquisition mode as DataMode type.

        Returns:
            num_of_packets:
                A integer value representing the number of data packets.
        """
        # Compute packet dimension and check its consistency given data acquisition mode
        packet_dimension = Muse_Utils.GetPacketDimension(inMode)

        num_of_packets = -1
        # Compute overall number of packets contained into a data characteristic update
        if (packet_dimension != -1):
            num_of_packets = (120 / packet_dimension)

        # Return -1 in case of data dimension inconsistency
        return num_of_packets

    @staticmethod
    def DecodePacket(buffer: bytearray, timestamp: int, mode: MH.DataMode, gyr_res: float, axl_res: float, mag_res: float, hdr_res: float):
        """
        Decode data packet.

        Args:
            buffer (bytearray):
                Bytearray to be decoded.
            timestamp (int):
                Reference timestamp related to the last notification.
            mode (Muse_HW.DataMode):
                Data acquisition mode.
            gyr_res (float):
                Gyroscope sensitivity coefficient.
            axl_res (float):
                Accelerometer sensitivity coefficient.
            mag_res (float):
                Magnetometer sensitivity coefficient.
            hdr_res (float):
                High Dynamic Range (HDR) Accelerometer sensitivity coefficient. 

        Returns:
            current_data (Muse_Data):
                A Muse_Data object with decoded data.
        """
        #Check input buffer consistency
        if (buffer != None and len(buffer) > 0):
        
            #Define muse data container
            packet_index_offset = 0
            
            current_data = Muse_Data()
            #Set overall timestamp reference
            current_data.overall_timestamp = timestamp
            current_data.timestamp = timestamp

            #Ther order of IF statements is related to the order with which the data coming within each packet in order to properly update the packet_index_offset
            if ((mode & MH.DataMode.DATA_MODE_GYRO.value) > 0):
            
                #Decode Gyroscope reading
                current_packet = bytearray(int(MH.DataSize.DATA_SIZE_GYRO.value))

                current_packet[:] = buffer[packet_index_offset:packet_index_offset+int(MH.DataSize.DATA_SIZE_GYRO.value)] 
                packet_index_offset += int(MH.DataSize.DATA_SIZE_GYRO.value)

                current_data.gyr = Muse_Utils.DataTypeGYR(current_packet, gyr_res)
            

            if ((mode & MH.DataMode.DATA_MODE_AXL.value) > 0):
            
                #Decode Accelerometer reading
                current_packet = bytearray(int(MH.DataSize.DATA_SIZE_AXL.value))

                current_packet[:] = buffer[packet_index_offset:packet_index_offset+int(MH.DataSize.DATA_SIZE_AXL.value)] 
                packet_index_offset += int(MH.DataSize.DATA_SIZE_AXL.value)

                current_data.axl = Muse_Utils.DataTypeAXL(current_packet, axl_res)
            

            if ((mode & MH.DataMode.DATA_MODE_MAGN.value) > 0):
            
                # Decode Magnetometer reading
                current_packet = bytearray(int(MH.DataSize.DATA_SIZE_MAGN.value))
                current_packet[:] = buffer[packet_index_offset:packet_index_offset+int(MH.DataSize.DATA_SIZE_MAGN.value)] 
                packet_index_offset += int(MH.DataSize.DATA_SIZE_MAGN.value)

                current_data.mag = Muse_Utils.DataTypeMAGN(current_packet, mag_res)
            

            if ((mode & MH.DataMode.DATA_MODE_HDR.value) > 0):
            
                #Decode Accelerometer HDR reading
                current_packet = bytearray(int(MH.DataSize.DATA_SIZE_HDR.value))
                current_packet[:] = buffer[packet_index_offset:packet_index_offset+int(MH.DataSize.DATA_SIZE_HDR.value)] 
                packet_index_offset += int(MH.DataSize.DATA_SIZE_HDR.value)

                current_data.hdr = Muse_Utils.DataTypeHDR(current_packet, hdr_res)
            

            if ((mode & MH.DataMode.DATA_MODE_ORIENTATION.value) > 0):
            
                #Decode orientation quaternion
                current_packet = bytearray(int(MH.DataSize.DATA_SIZE_ORIENTATION.value))
                current_packet[:] = buffer[packet_index_offset:packet_index_offset+int(MH.DataSize.DATA_SIZE_ORIENTATION.value)] 
                packet_index_offset += int(MH.DataSize.DATA_SIZE_ORIENTATION.value)

                current_data.quat = Muse_Utils.DataTypeOrientation(current_packet)
                current_data.euler = Muse_Utils.GetAnglesFromQuaternion(current_data.quat)
            

            if ((mode & MH.DataMode.DATA_MODE_TIMESTAMP.value) > 0):
            
                #Decode timestamp
                current_packet = bytearray(int(MH.DataSize.DATA_SIZE_TIMESTAMP.value))
                current_packet[:] = buffer[packet_index_offset:packet_index_offset+int(MH.DataSize.DATA_SIZE_TIMESTAMP.value)] 
                packet_index_offset += int(MH.DataSize.DATA_SIZE_TIMESTAMP.value)

                current_data.timestamp = Muse_Utils.DataTypeTimestamp(current_packet)
            

            if ((mode & MH.DataMode.DATA_MODE_TEMP_HUM.value) > 0):
            
                #Decode temperature and humidity reading
                current_packet = bytearray(int(MH.DataSize.DATA_SIZE_TEMP_HUM.value))
                current_packet[:] = buffer[packet_index_offset:packet_index_offset+int(MH.DataSize.DATA_SIZE_TEMP_HUM.value)] 
                packet_index_offset += int(MH.DataSize.DATA_SIZE_TEMP_HUM.value)

                current_data.th = Muse_Utils.DataTypeTempHum(current_packet)
            

            if ((mode & MH.DataMode.DATA_MODE_TEMP_PRESS.value) > 0):
            
                #Decode temperature and barometric pressure reading
                current_packet = bytearray(int(MH.DataSize.DATA_SIZE_TEMP_PRESS.value))
                current_packet[:] = buffer[packet_index_offset:packet_index_offset+int(MH.DataSize.DATA_SIZE_TEMP_PRESS.value)] 
                packet_index_offset += int(MH.DataSize.DATA_SIZE_TEMP_PRESS.value)

                current_data.tp = Muse_Utils.DataTypeTempPress(current_packet)
            

            if ((mode & MH.DataMode.DATA_MODE_RANGE.value) > 0):
            
                #Decode distance / luminosity reading
                current_packet = bytearray(int(MH.DataSize.DATA_SIZE_RANGE.value))
                current_packet[:] = buffer[packet_index_offset:packet_index_offset+int(MH.DataSize.DATA_SIZE_RANGE.value)] 
                packet_index_offset += int(MH.DataSize.DATA_SIZE_RANGE.value)

                tmp = Muse_Utils.DataTypeRange(current_packet)
                current_data.light = Light(tmp[0],tmp[1],tmp[2],tmp[3])
            

            if ((mode & MH.DataMode.DATA_MODE_SOUND.value) > 0):
            
                #Decode sound reading
                current_packet = bytearray(int(MH.DataSize.DATA_SIZE_SOUND.value))
                current_packet[:] = buffer[packet_index_offset:packet_index_offset+int(MH.DataSize.DATA_SIZE_SOUND.value)] 
                packet_index_offset += int(MH.DataSize.DATA_SIZE_SOUND.value)

                current_data.sound = Muse_Utils.DataTypeSound(current_packet)
            
            return current_data
        
        return None
    
    @staticmethod
    def Dec_WifiSSIDHead(response: CommandResponse):
        """
        Decode SSID Head(i.e., it is the SSID advertised by Bluetooth module).

        Args:
            response:
                CommandResponse object to be decoded.

        Returns:
            ssid:
                Output reference to SSID head part.
                
        """
        ssid = ""

        #Decode SSID given command response payload
        if (response.tx == MH.Command.CMD_WIFI_SSID_HEAD.value and
            response.ack == MH.AcknowledgeType.ACK_SUCCESS):
        
            ssid = str(response.payload, 'utf-8')
            
        return ssid
    
    @staticmethod
    def Dec_WifiSSIDCont(response: CommandResponse):
        """
        Decode device SSID Continuation part(i.e., it is the SSID advertised by Bluetooth module).

        Args:
            response:
                CommandResponse object to be decoded.

        Returns:
            ssid:
                Output reference to SSID continuation part. 
        """
        ssid = ""

        #Decode SSID given command response payload
        if (response.tx == MH.Command.CMD_WIFI_SSID_CONT.value and
            response.ack == MH.AcknowledgeType.ACK_SUCCESS):
        
            ssid = str(response.payload, 'utf-8')
            
        return ssid

    @staticmethod
    def Dec_WifiStreamHostHead(response: CommandResponse):
        """
        Decode Stream Host Head.

        Args:
            response:
                CommandResponse object to be decoded.

        Returns:
            host:
                Output reference to host head part.   
        """
        #Decode battery voltage [mV] value given command response payload
        if (response.tx == MH.Command.CMD_WIFI_STREAM_HOST_HEAD and
            response.ack == MH.AcknowledgeType.ACK_SUCCESS.value):

            host = str(response.payload, 'utf-8')
            
            return host
        
    @staticmethod
    def Dec_WifiStreamHostCont(response: CommandResponse):
        """
        Decode Stream Host continuation part.

        Args:
            response:
                CommandResponse object to be decoded.

        Returns:
            host:
                Output reference to the remaining part of host.   
        """
        #Decode battery voltage [mV] value given command response payload
        if (response.tx == MH.Command.CMD_WIFI_STREAM_HOST_CONT and
            response.ack == MH.AcknowledgeType.ACK_SUCCESS.value):

            host = str(response.payload, 'utf-8')
            
            return host
    
    @staticmethod
    def Dec_WifiStreamHostPort(response: CommandResponse):
        """
        Decode Stream Host port.

        Args:
            response:
                CommandResponse object to be decoded.

        Returns:
            port:
                Output reference to the WiFi port.   
        """
        #Decode battery voltage [mV] value given command response payload
        if (response.tx == MH.Command.CMD_WIFI_STREAM_HOST_PORT and
            response.ack == MH.AcknowledgeType.ACK_SUCCESS.value):

            port = struct.unpack('<H', response.payload)[0]
            
            return port
    
    @staticmethod
    def DataTypeGYR(current_payload: bytearray, gyr_res: float):
        """Decode Gyroscope reading.

        Args:
            current_payload (bytearray):
                Bytes array to be decoded.
            gyr_res (float):
                Gyroscope sensitivity coefficient.

        Returns:
            current_data: 3-elements array of float (i.e., x, y, z channels) containing Gyroscope data
        """
        #Define 3-elements array of float (i.e., x, y, z channels)
        current_data = [0.0]*3

        #Iterate across Gyroscope channels
        for i in range(3):
        
            #Extract channel raw value (i.e., 2-bytes each)
            raw_value = current_payload[2*i:2*(i+1)]
            # Convert to Int16 and apply sensitivity scaling
            current_data[i] = int.from_bytes(raw_value, byteorder='little', signed=True) * gyr_res
    

        #Return decoded Gyroscope reading
        return current_data
    
    @staticmethod
    def DataTypeAXL(current_payload: bytearray, axl_res: float):
        """Decode Accelerometer reading.

        Args:
            current_payload (bytearray):
                Bytes array to be decoded.
            axl_res (float):
                Accelerometer sensitivity coefficient.

        Returns:
            current_data: 3-elements array of float (i.e., x, y, z channels) containing Accelerometer data
        """
        current_data = [0.0] * 3

        #Accelerometer
        for i in range(3):
            #Extract channel raw value (i.e., 2-bytes each)
            raw_value = current_payload[2*i:2*(i+1)]
            # Convert to Int16 and apply sensitivity scaling
            current_data[i] = int.from_bytes(raw_value, byteorder='little', signed=True) * axl_res
    

        return current_data
    
    @staticmethod
    def DataTypeMAGN(current_payload: bytearray, mag_res: float):
        """Decode Magnetometer reading.

        Args:
            current_payload (bytearray):
                Bytes array to be decoded.
            mag_res (float):
                Magnetometer sensitivity coefficient.

        Returns:
            current_data: 3-elements array of float (i.e., x, y, z channels) containing Magnetometer data
        """
        current_data = [0.0] * 3

        #Magnetometer
        for i in range(3):
            #Extract channel raw value (i.e., 2-bytes each)
            raw_value = current_payload[2*i:2*(i+1)]
            # Convert to Int16 and apply sensitivity scaling
            current_data[i] = int.from_bytes(raw_value, byteorder='little', signed=True) * mag_res
    

        return current_data
    
    @staticmethod
    def DataTypeHDR(current_payload: bytearray, hdr_res: float):
        """Decode High Dynamic Range (HDR) Accelerometer reading.

        Args:
            current_payload (bytearray):
                Bytes array to be decoded.
            hdr_res (float):
                High Dynamic Range (HDR) Accelerometer sensitivity coefficient.

        Returns:
            current_data: 3-elements array of float (i.e., x, y, z channels) containing High Dynamic Range (HDR) Accelerometer data
        """
        current_data = [0.0]*3
        tmp = bytearray(2)

        #Accelerometer HDR
        for i in range(3):
            #Extract channel raw value (i.e., 2-bytes each)
            raw_value = current_payload[2*i:2*(i+1)]
            # Convert to Int16 and apply sensitivity scaling
            current_data[i] = int.from_bytes(raw_value, byteorder='little', signed=True) * hdr_res

        return current_data
    
    @staticmethod
    def DataTypeOrientation(current_payload: bytearray):
        """Decode orientation (unit) quaternion

        Args:
            current_payload (bytearray):
                Bytes array to be decoded.

        Returns:
            current_data: 4-elements array of float (i.e., qw, qi, qj, qk) containing Quaternion components
        """
        #Define 4-elements array of float (i.e., qw, qi, qj, qk channels)
        current_data = [0.0] * 4

        #Orientation Quaternion (i.e., UNIT QUATERNION)
        tempFloat = [0.0] * 3
        for i in range(3):
        
            #Extract channel raw value (i.e., 2-bytes each) and convert to Int16
            tempFloat[i] = int.from_bytes(current_payload[2*i:2*i+2], byteorder='little', signed=True)

            #Keep into account the data resolution
            tempFloat[i] /= 32767
            
            #Assign imaginary parts of quaternion
            current_data[i + 1] = tempFloat[i]
        

        #Compute real component of quaternion given immaginary parts
        current_data[0] = math.sqrt(1 - (current_data[1] * current_data[1] +
                    current_data[2] * current_data[2] + current_data[3] * current_data[3]))

        #Return decoded Unit Quaternion
        return current_data
    
    @staticmethod
    def GetAnglesFromQuaternion(q):
        """Compute Euler Angles given a unit quaternion.

        Args:
            q:
                Unit quaternion to be converted (4-elements array of float).

        Returns:
            result: 3-elements array of float (i.e., roll, pitch and yaw) containing Euler Angles
        """
        result = [0.0] * 3

        #Compute Euler Angles given a unit quaternion
        result[0] = float(math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] * q[1] + q[2] * q[2])) * 180 / math.pi)
        result[1] = float(math.asin(2 * q[0] * q[2] - 2 * q[3] * q[1]) * 180 / math.pi)
        result[2] = float(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3])) * 180 / math.pi)
        
        return result    

    @staticmethod
    def DataTypeTimestamp(current_payload: bytearray):
        """Decode timestamp.

        Args:
            current_payload (bytearray):
                Bytes array to be decoded.

        Returns:
            currentData: A unsigned long integer value representing the timestamp in epoch format.
        """
        #Set current data variable to 0
        currentData = 0

        #Get raw byte array representation to be decoded
        tmp = bytearray(8)
        
        # Copy the first 6 bytes of currentPayload into tmp
        tmp[:6] = current_payload[:6]

        # Convert the first 6 bytes of tmp to a 64-bit unsigned integer
        tempTime = struct.unpack("<Q", tmp)[0] & 0x0000FFFFFFFFFFFF

        # Add the reference epoch (in milliseconds) to tempTime
        tempTime += MH.REFERENCE_EPOCH * 1000

        currentData = tempTime

        return currentData
    
    @staticmethod
    def DataTypeTempHum(current_payload: bytearray):
        """Decode temperature and humidity reading.

        Args:
            current_payload (bytearray):
                Bytes array to be decoded.

        Returns:
            current_data: An array of float (i.e., temperature, humidity)
        """
        #Define 2-elements array of float (i.e., temperature, humidity)
        current_data = [0.0] * 2

        #Temperature
        #Define temporary internal variable used for data conversion
        tmp = bytearray(2)  
        # Copy the first 2 bytes of currentPayload into tmp
        tmp[:2] = current_payload[0:2]
        # Convert to UInt16 and apply sensitivity scaling
        current_data[0] = int.from_bytes(tmp, byteorder='little', signed=False)
        current_data[0] *= 0.002670 
        current_data[0] -= 45

        #Humidity
        tmp[:2] = current_payload[2:4]
        # Convert to UInt16 and apply sensitivity scaling
        current_data[1] = int.from_bytes(tmp, byteorder='little', signed=False)
        current_data[1] *= 0.001907
        current_data[1] -= 6

        #Return decoded data
        return current_data
    
    @staticmethod
    def DataTypeTempPress(current_payload: bytearray):
        """Decode temperature and barometric pressure reading.

        Args:
            current_payload (bytearray):
                Bytes array to be decoded.

        Returns:
            current_data: An array of float (i.e., temperature, pressure)
        """
        #Define 2-elements array of float (i.e., temperature, pressure)
        current_data = [0.0] * 2

        #Temperature
        #Define temporary internal variable used for data conversion
        tmp = bytearray(2)  
        # Copy the first 2 bytes of currentPayload into tmp
        tmp[:2] = current_payload[3:5]
        # Convert to UInt16 and apply sensitivity scaling
        current_data[0] = int.from_bytes(tmp, byteorder='little', signed=False)
        current_data[0] /= 100

        #Pressure
        tmp = bytearray(3)
        tmp[:3] = current_payload[0:3]
        # Convert to UInt16 and apply sensitivity scaling
        current_data[1] = int.from_bytes(tmp, byteorder='little', signed=False)
        current_data[1] /= 4096

        #Return decoded data
        return current_data
  
    @staticmethod
    def DataTypeRange(current_payload: bytearray):
        """Decode Light sensor data.

        Args:
            current_payload (bytearray):
                Bytes array to be decoded.

        Returns:
            current_data: A 4-element float array (i.e., range, vis, ir, lux)
        """
        current_data = [0] * 4

        #Define temporary internal variable used for data conversion
        tmp = bytearray(2)  
        # Copy the first 2 bytes of currentPayload into tmp
        tmp[:2] = current_payload[0:2]
        # Convert to UInt16
        range = int.from_bytes(tmp, byteorder='little', signed=False)

        tmp[:2] = current_payload[2:4]
        # Convert to UInt16
        vis = int.from_bytes(tmp, byteorder='little', signed=False)

        tmp[:2] = current_payload[4:6]
        # Convert to UInt16
        ir = int.from_bytes(tmp, byteorder='little', signed=False)

        current_data[0] = range
        current_data[1] = vis
        current_data[2] = ir

        lux = 0.0
        if vis > 0:
            if (ir / vis < 0.109):
                lux = 1.534 * vis - 3.759 * ir
            elif (ir / vis < 0.429):
                lux = 1.339 * vis - 1.972 * ir
            elif (ir / vis < 0.95 * 1.45):
                lux = 0.701 * vis - 0.483 * ir
            elif (ir / vis < 1.5 * 1.45):
                lux = 2.0 * 0.701 * vis - 1.18 * 0.483 * ir
            elif (ir / vis < 2.5 * 1.45):
                lux = 4.0 * 0.701 * vis - 1.33 * 0.483 * ir
            else:
                lux = 8.0 * 0.701 * vis
        else:
            # manage division by zero
            lux = 0.0

        current_data[3] = round(lux)       

        return current_data
    
    @staticmethod
    def DataTypeSound(current_payload: bytearray):
        """Decode Sound data. TO BE IMPLEMENTED
        """
        current_data = 0.0

        return current_data
    
    @staticmethod
    def DecodeMEMSConfiguration(code: int):
        """Decode sensors configuration in terms of full scale and sensitivity.

        Args:
            code (int):
                24-bit unsigned integer code.

        Returns:
            (gyrConfig, axlConfig, magConfig, hdrConfig):
                - gyrConfig: Output reference to Gyroscope configuration.
                - axlConfig: Output reference to Accelerometer configuration.
                - magConfig: Output reference to Magnetometer configuration.
                - hdrConfig: Output reference to HDR Accelerometer configuration.
        """
        #Apply bitwise mask to get sensor full scale code (i.e., gyr, axl, hdr, mag LSB-order)
        gyrCode = (code & MH.MEMS_FullScaleMask.SENSORSFS_MASK_GYRO.value)
        axlCode = (code & MH.MEMS_FullScaleMask.SENSORSFS_MASK_AXL.value)
        hdrCode = (code & MH.MEMS_FullScaleMask.SENSORSFS_MASK_HDR.value)
        magCode = (code & MH.MEMS_FullScaleMask.SENSORSFS_MASK_MAGN.value)

        #Gyroscope
        gyrConfig = MH.Gyroscope_CFG[gyrCode]
        #Accelerometer
        axlConfig = MH.Accelerometer_CFG[axlCode]
        #Magnetometer
        magConfig = MH.Magnetometer_CFG[magCode]
        # HDR Accelerometer
        hdrConfig = MH.AccelerometerHDR_CFG[hdrCode]

        return gyrConfig, axlConfig, magConfig, hdrConfig
    
    @staticmethod
    def DataModeToString(code: int) -> str:
        """Create a string representation of data acquisition mode.

        Args:
            code (int):
                32-bit unsigned integer code.

        Returns:
            modeString:
                string representation of data acquisition mode.
        """
        modeString = ""
        
        #Build acquisition mode string description
        if ((code & ctypes.c_uint32(MH.DataMode.DATA_MODE_GYRO.value).value) > 0):
            if (len(modeString)>0):
                modeString += " | "
            modeString += "GYR"
        if ((code & ctypes.c_uint32(MH.DataMode.DATA_MODE_AXL.value).value) > 0):
            if (len(modeString)>0):
                modeString += " | "
            modeString += "AXL"
        if ((code & ctypes.c_uint32(MH.DataMode.DATA_MODE_HDR.value).value) > 0):
            if (len(modeString)>0):
                modeString += " | "
            modeString += "HDR"
        if ((code & ctypes.c_uint32(MH.DataMode.DATA_MODE_MAGN.value).value) > 0):
            if (len(modeString)>0):
                modeString += " | "
            modeString += "MAG"
        if ((code & ctypes.c_uint32(MH.DataMode.DATA_MODE_ORIENTATION.value).value) > 0):
            if (len(modeString)>0):
                modeString += " | "
            modeString += "QUAT"
        if ((code & ctypes.c_uint32(MH.DataMode.DATA_MODE_TIMESTAMP.value).value) > 0):
            if (len(modeString)>0):
                modeString += " | "
            modeString += "TIME"
        if ((code & ctypes.c_uint32(MH.DataMode.DATA_MODE_TEMP_HUM.value).value) > 0):
            if (len(modeString)>0):
                modeString += " | "
            modeString += "TH"
        if ((code & ctypes.c_uint32(MH.DataMode.DATA_MODE_TEMP_PRESS.value).value) > 0):
            if (len(modeString)>0):
                modeString += " | "
            modeString += "TP"
        if ((code & ctypes.c_uint32(MH.DataMode.DATA_MODE_RANGE.value).value) > 0):
            if (len(modeString)>0):
                modeString += " | "
            modeString += "RANGE"
        if ((code & ctypes.c_uint32(MH.DataMode.DATA_MODE_SOUND.value).value) > 0):
            if (len(modeString)>0):
                modeString += " | "
            modeString += "MIC"

        if (modeString != ""):
            modeString.join(", ") 
        
        return modeString 
    
    @staticmethod
    def DataFrequencyToString(frequency: MH.DataFrequency) -> str:
        """Create a string representation of data acquisition mode.

        Args:
            frequency (Muse_HW.DataFrequency):
                DataFrequency value of sampling frequency

        Returns:
            string representation of data acquisition frequency.
        """
        if frequency == MH.DataFrequency.DATA_FREQ_25Hz :
            return "25 Hz"
            
        if frequency ==  MH.DataFrequency.DATA_FREQ_50Hz:
            return "50 Hz"
        
        if frequency ==  MH.DataFrequency.DATA_FREQ_100Hz:
            return "100 Hz"
        
        if frequency == MH.DataFrequency.DATA_FREQ_200Hz:
            return "200 Hz"
        
        if frequency ==  MH.DataFrequency.DATA_FREQ_400Hz:
            return "400 Hz"
        
        if frequency == MH.DataFrequency.DATA_FREQ_800Hz:
            return "800 Hz"
        
        if frequency ==  MH.DataFrequency.DATA_FREQ_1600Hz:
            return "1600 Hz"
        else:
            return "NONE"

    # end DECODING FUNCTIONS

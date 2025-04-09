""" Muse_HW.py: Muse communication protocol specifications.

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

from enum import Enum, Flag, IntEnum
from frozendict import frozendict
from muse_api_main.Muse_Data import *

class Muse_HW:
    
    # CONSTANTS

    COMM_MESSAGE_LEN_CMD = 20 # Command characteristic size [bytes]
    
    DATA_MESSAGE_LEN_CMD = 128 # Data characteristic size [bytes]
    
    REFERENCE_EPOCH = 1580000000 # Timestamp reference epoch (corisponds to Sunday 26 January 2020 00:53:20)

    READ_BIT_MASK = 0x80 # Bit-mask used to manage read/wrte command modes

    # end CONSTANTS

    # DEVICE section
    class Command(IntEnum):
        """ Command codes """
        #<summary> NOT A COMMAND - used only on software side </summary>
        CMD_NONE = 0xff
        #<summary>Acknowledge</summary>
        CMD_ACK = 0x00
        # <summary>State [get/set]</summary>
        CMD_STATE = 0x02
        # <summary>Restart</summary>
        CMD_RESTART = 0x03
        # <summary>Application firmware information [readonly]</summary>
        CMD_APP_INFO = 0x04
        # <summary>Application firmware upload </summary>
        CMD_FW_UPLOAD = 0x05
        # <summary>Battery Charge [readonly]</summary>
        CMD_BATTERY_CHARGE = 0x07
        # <summary>Battery Voltage [readonly]</summary>
        CMD_BATTERY_VOLTAGE = 0x08
        # <summary>Device check up flag [readonly]</summary>
        CMD_CHECK_UP = 0x09
        # <summary>Installed Firmware Version [readonly]</summary>
        CMD_FW_VERSION = 0x0a
        # <summary>Current Time [get/set]</summary>
        CMD_TIME = 0x0b
        # <summary>Bluetooth Module Name [get/set]</summary>
        CMD_BLE_NAME = 0x0c
        # <summary>Device Identification code [readonly]</summary>
        CMD_DEVICE_ID = 0x0e
        # <summary>Device features [readonly]</summary>
        CMD_DEVICE_SKILLS = 0x0f
        # <summary>Memory state [readonly]</summary>
        CMD_MEM_CONTROL = 0x20        
        # <summary>Get file information [readonly]</summary>
        CMD_MEM_FILE_INFO = 0x21       
        # <summary>File download</summary>
        CMD_MEM_FILE_DOWNLOAD = 0x22
        # <summary>Clock offset [get/set]</summary>
        CMD_CLK_OFFSET = 0x31          
        # <summary>Enter time sync mode</summary>
        CMD_TIME_SYNC = 0x32           
        # <summary>Exit time sync mode</summary>
        CMD_EXIT_TIME_SYNC = 0x33
        # <summary>Sensors full scales [get/set]</summary>
        CMD_SENSORS_FS = 0x40
        # <summary>Sensors calibration matricies [get/set]</summary>
        CMD_CALIB_MATRIX = 0x48
        # <summary>Log button confiugration [get/set]</summary>
        CMD_BUTTON_LOG = 0x50
        # <summary> User configuration settings [get/set]</summary>
        CMD_USER_CONFIG = 0x51

        ## Muse WiFi-enabled commands
        # <summary> Set the SSID of the network – head part [get/set]</summary>
        CMD_WIFI_SSID_HEAD = 0x71
        # <summary> Set the SSID of the network – continuation part [get/set]</summary>
        CMD_WIFI_SSID_CONT = 0x72
        # <summary> Set the password for network authentication – head part [set]</summary>
        CMD_WIFI_PSW_HEAD = 0x73
        # <summary> Set the password for network authentication – continuation part [set]</summary>
        CMD_WIFI_PSW_CONT = 0x74
        # <summary> Set the host name – head part [get/set]</summary>
        CMD_WIFI_STREAM_HOST_HEAD = 0x75
        # <summary> Set the host name – continuation part [get/set]</summary>
        CMD_WIFI_STREAM_HOST_CONT = 0x76
        # <summary> Set the host port [get/set]</summary>
        CMD_WIFI_STREAM_HOST_PORT = 0x77

    class CommandLength(IntEnum):
        """ Command lengths """
        #<summary> Get state command length </summary>
        CMD_LENGTH_GET_STATE = 2
        # <summary> Set state command length </summary>
        CMD_LENGTH_SET_STATE = 3
        # <summary> Acknowledge command length </summary>
        CMD_LENGTH_ACKNOWLEDGE = 5
        # <summary> Start stream acquisition command length </summary>
        CMD_LENGTH_START_STREAM = 7
        # <summary> Start log acquisition command length </summary>
        CMD_LENGTH_START_LOG = 7
        # <summary> Stop acquisition command length </summary>
        CMD_LENGTH_STOP_ACQUISITION = 3
        # <summary> Restart command length </summary>
        CMD_LENGTH_RESTART = 3
        # <summary> Get firmware application info command length </summary>
        CMD_LENGTH_GET_APP_INFO = 2
        # <summary> Get battery charge level command length </summary>
        CMD_LENGTH_GET_BATTERY_CHARGE = 2
        # <summary> Get battery voltage level command length </summary>
        CMD_LENGTH_GET_BATTERY_VOLTAGE = 2
        # <summary> Get check-up register value command length </summary>
        CMD_LENGTH_GET_CHECK_UP = 2
        # <summary> Get firmware versions command length </summary>
        CMD_LENGTH_GET_FW_VERSION = 2
        # <summary> Get time command length </summary>
        CMD_LENGTH_GET_TIME = 2
        # <summary> Set time command length </summary>
        CMD_LENGTH_SET_TIME = 6
        # <summary> Get device name command length </summary>
        CMD_LENGTH_GET_BLE_NAME = 2
        # <summary> Set device command length </summary>
        CMD_LENGTH_SET_BLE_NAME = 20
        # <summary> Get device unique identifier command length </summary>
        CMD_LENGTH_GET_DEVICE_ID = 2
        # <summary> Get device skills command length </summary>
        CMD_LENGTH_GET_DEVICE_SKILLS = 2
        # <summary> Get memory status command length </summary>
        CMD_LENGTH_GET_MEM_CONTROL = 2
        # <summary> Erase memory command length </summary>
        CMD_LENGTH_SET_MEM_CONTROL = 3
        # <summary> Get memory file info command length </summary>
        CMD_LENGTH_GET_MEM_FILE_INFO = 4
        # <summary> Start file offload command length </summary>
        CMD_LENGTH_GET_MEM_FILE_DOWNLOAD = 5
        # <summary> Get clock offset command length </summary>
        CMD_LENGTH_GET_CLK_OFFSET = 2
        # <summary> Set clock offset command length </summary>
        CMD_LENGTH_SET_CLK_OFFSET = 10
        # <summary> Enter timesync command length </summary>
        CMD_LENGTH_ENTER_TIME_SYNC = 2
        # <summary> Exit timesync command length </summary>
        CMD_LENGTH_EXIT_TIME_SYNC = 2
        # <summary> Get sensors full scale configuration command length </summary>
        CMD_LENGTH_GET_SENSORS_FS = 2
        # <summary> Set sensors full scale configuration command length </summary>
        CMD_LENGTH_SET_SENSORS_FS = 3
        # <summary> Get calibration matrix command length </summary>
        CMD_LENGTH_GET_CALIB_MATRIX = 4
        # <summary> Set calibration matrix command length </summary>
        CMD_LENGTH_SET_CALIB_MATRIX = 16
        # <summary> Get button log configuration command length </summary>
        CMD_LENGTH_GET_BUTTON_LOG = 2
        # <summary> Set button log configuration command length </summary>
        CMD_LENGTH_SET_BUTTON_LOG = 6
        # <summary> Get user configuration command length </summary>
        CMD_LENGTH_GET_USER_CONFIG = 2
        # <summary> Set user configuration command length </summary>
        CMD_LENGTH_SET_USER_CONFIG = 6

        ## Muse WiFi-enabled command lengths
        # <summary> Set the SSID of the network – head part [get/set]</summary>
        CMD_LENGTH_GET_WIFI_SSID_HEAD = 2
        # <summary> Set the SSID of the network – continuation part [get/set]</summary>
        CMD_LENGTH_GET_WIFI_SSID_CONT = 2  
        # <summary> Set the host name – head part [get/set]</summary>
        CMD_LENGTH_GET_WIFI_STREAM_HOST_HEAD = 2
        # <summary> Set the host name – continuation part [get/set]</summary>
        CMD_LENGTH_GET_WIFI_STREAM_HOST_CONT = 2
        # <summary> Set the host port [get/set]</summary>
        CMD_LENGTH_GET_WIFI_STREAM_HOST_PORT = 2
        # <summary> Set the host port [get/set]</summary>
        CMD_LENGTH_SET_WIFI_STREAM_HOST_PORT = 4

    class AcknowledgeType(IntEnum):
        """Acknowledge types"""
        #<summary>NOT AN ACKNOWLEDGE - used only on software side </summary>
        ACK_NONE = 0xff,
        #<summary>Success</summary>
        ACK_SUCCESS = 0x00,                             
        # <summary>Error</summary>
        ACK_ERROR = 0x01	    
        
    class SystemState(Enum):
        """System states"""
        #<summary>System state NONE - used only on software side</summary>
        SYS_NONE = 0x00   
        # <summary>System state ERROR</summary>
        SYS_ERROR = 0xff      
        # <summary>System state STARTUP</summary>
        SYS_STARTUP = 0x01     
        # <summary>System state IDLE</summary>
        SYS_IDLE = 0x02        
        # <summary>System state STANDBY</summary>
        SYS_STANDBY = 0x03     
        #<summary>System state LOG - acquisition mode</summary>
        SYS_LOG = 0x04         
        #<summary>System state READOUT - memory file download</summary>
        SYS_READOUT = 0x05     
        #<summary>System state STREAM - buffered acquisition (realtime)</summary>
        SYS_TX_BUFFERED = 0x06 
        #<summary>System state CALIB - calibration routines</summary>
        SYS_CALIB = 0x07       
        #<summary>System state STREAM - direct acquisition (realtime)</summary>
        SYS_TX_DIRECT = 0x08

    class RestartMode(Enum):
        """Restart modes
        - APPLICATION: 0x00 Restart device in application mode
        - BOOT: 0x01 Restart device in bootloader mode
        - RESET: 0x02 Restart device in application mode after performing a factory reset
        """        
        #<summary>Restart device in application mode</summary>
        APPLICATION = 0x00
        #<summary>Restart device in bootloader mode</summary>
        BOOT = 0x01
        #<summary>Restart device in application mode after performing a factory reset</summary>
        RESET = 0x02

    class CommunicationChannel(Enum):
        """Communication Channels
        - CHANNEL_NONE: 0x00 Channel NONE - used only on software side
        - CHANNEL_BLE: 0x01 Bluetooth Low Energy
        - CHANNEL_USB: 0x02 USB channel
        """ 
        #<summary>Channel NONE - used only on software side</summary>
        CHANNEL_NONE = 0x00
        #<summary>Bluetooth Low Energy</summary>
        CHANNEL_BLE = 0x01
        #<summary>USB</summary>
        CHANNEL_USB = 0x02

    class StreamingChannel(Enum):
        """Streaming Channels
        - CHANNEL_BLE: 0x00 Bluetooth Low Energy
        - CHANNEL_USB: 0x01 USB channel
        - CHANNEL_TCP: 0x02 TCP channel
        - CHANNEL_MQTT: 0x03 MQTT channel
        """ 
        #<summary>Bluetooth Low Energy</summary>
        CHANNEL_BLE = 0x00
        #<summary>USB</summary>
        CHANNEL_USB = 0x01
        #<summary>TCP</summary>
        CHANNEL_TCP = 0x02
        #<summary>MQTT</summary>
        CHANNEL_MQTT = 0x03
    
    # end DEVICE section

    # DATA ACQUISITION section
    class DataMode(Enum):
        """Acquisition modes"""
        #<summary>Acquisition mode NONE - used only on software side</summary>
        DATA_MODE_NONE = 0x00000000
        #<summary>Acquisition mode Gyroscope</summary>
        DATA_MODE_GYRO = 0x00000001
        #<summary>Acquisition mode Accelerometer</summary>
        DATA_MODE_AXL = 0x00000002
        #<summary>Acquisition mode IMU: Gyroscope + Accelerometer</summary>
        DATA_MODE_IMU = DATA_MODE_AXL | DATA_MODE_GYRO
        #<summary>Acquisition mode Magnetometer</summary>
        DATA_MODE_MAGN = 0x00000004
        #<summary>Acquisition mode 9DOF: Gyroscope + Accelerometer + Magnetometer</summary>
        DATA_MODE_9DOF = DATA_MODE_MAGN | DATA_MODE_IMU
        #<summary>Acquisition mode High Dynamic Range (HDR) Accelerometer</summary>
        DATA_MODE_HDR = 0x00000008
        #<summary>Acquisition mode IMU + HDR</summary>
        DATA_MODE_IMU_HDR = DATA_MODE_IMU | DATA_MODE_HDR
        #<summary>Acquisition mode orientation quaternion</summary>
        DATA_MODE_ORIENTATION = 0x00000010
        #<summary>Acquisition mode IMU + orientation quaternion</summary>
        DATA_MODE_IMU_ORIENTATION = DATA_MODE_ORIENTATION | DATA_MODE_IMU
        #<summary>Acquisition mode 9DOF + orientation quaternion</summary>
        DATA_MODE_9DOF_ORIENTATION = DATA_MODE_9DOF | DATA_MODE_ORIENTATION
        # <summary>Acquisition mode timestamp</summary>
        DATA_MODE_TIMESTAMP = 0x00000020
        #<summary>Acquisition mode 9DOF + timestamp </summary>
        DATA_MODE_9DOF_TIMESTAMP = DATA_MODE_9DOF | DATA_MODE_TIMESTAMP
        #<summary>Acquisition mode orientation quaternion + timestamp</summary>
        DATA_MODE_ORIENTATION_TIMESTAMP = DATA_MODE_ORIENTATION | DATA_MODE_TIMESTAMP
        #<summary>Acquisition mode IMU + orientation quaternion + timestamp</summary>
        DATA_MODE_IMU_ORIENTATION_TIMESTAMP = DATA_MODE_IMU_ORIENTATION | DATA_MODE_TIMESTAMP
        #<summary>Acquisition mode temperature and humidity</summary>
        DATA_MODE_TEMP_HUM = 0x00000040
        # <summary>Acquisition mode temperature and barometric pressure</summary>
        DATA_MODE_TEMP_PRESS = 0x00000080
        #<summary>Acquisition mode range and light intensity</summary>
        DATA_MODE_RANGE = 0x00000100
        #<summary>Acquisition mode microphone</summary>
        DATA_MODE_SOUND = 0x00000400
        #IMU plus more
        DATA_MODE_IMU_MAG_TEMP_PRES_LIGHT =  DATA_MODE_IMU | DATA_MODE_MAGN | DATA_MODE_TEMP_PRESS | DATA_MODE_RANGE
        DATA_MODE_IMU_MAG_TEMP_PRES =  DATA_MODE_IMU | DATA_MODE_MAGN | DATA_MODE_TEMP_PRESS


    class DataSize(IntEnum):
        """Acquisition packet sizes"""
        #<summary>Packet size Gyroscope</summary>
        DATA_SIZE_GYRO = 6
        #<summary>Packet size Accelerometer</summary>
        DATA_SIZE_AXL = 6
        #<summary>Packet size IMU: Gyroscope + Accelerometer</summary>
        DATA_SIZE_IMU = DATA_SIZE_AXL + DATA_SIZE_GYRO
        #<summary>Packet size Magnetometer</summary>
        DATA_SIZE_MAGN = 6
        #<summary>Packet size 9DOF: Gyroscope + Accelerometer + Magnetometer</summary>
        DATA_SIZE_9DOF = DATA_SIZE_MAGN | DATA_SIZE_IMU
        #<summary>Packet size High Dynamic Range (HDR) Accelerometer</summary>
        DATA_SIZE_HDR = 6
        #<summary>Packet size IMU + HDR</summary>
        DATA_SIZE_IMU_HDR = DATA_SIZE_IMU | DATA_SIZE_HDR
        #<summary>Packet size orientation quaternion</summary>
        DATA_SIZE_ORIENTATION = 6
        #<summary>Packet size IMU + orientation quaternion</summary>
        DATA_SIZE_IMU_ORIENTATION = DATA_SIZE_ORIENTATION | DATA_SIZE_IMU
        #<summary>Packet size 9DOF + orientation quaternion</summary>
        DATA_SIZE_9DOF_ORIENTATION = DATA_SIZE_9DOF | DATA_SIZE_ORIENTATION
        #<summary>Packet size timestamp</summary>
        DATA_SIZE_TIMESTAMP = 6
        #<summary>Packet size temperature and humidity</summary>
        DATA_SIZE_TEMP_HUM = 6
        #<summary>Packet size temperature and barometric pressure</summary>
        DATA_SIZE_TEMP_PRESS = 6
        # <summary>Packet size luminosity</summary>
        DATA_SIZE_RANGE = 6
        #<summary>Packet size microphone</summary>
        DATA_SIZE_SOUND = 6                               
    
    class DataFrequency(Enum):
        """Acquisition frequencies"""   
        #<summary>Acquisition Frequency NONE - used only on software side</summary>
        DATA_FREQ_NONE = 0x00
        #<summary>Acquisition frequency 25 Hz</summary>
        DATA_FREQ_25Hz = 0x01
        #<summary>Acquisition frequency 50 Hz</summary>
        DATA_FREQ_50Hz = 0x02
        #<summary>Acquisition frequency 100 Hz</summary>
        DATA_FREQ_100Hz = 0x04
        #<summary>Acquisition frequency 200 Hz</summary>
        DATA_FREQ_200Hz = 0x08
        #<summary>Acquisition frequency 400 Hz</summary>
        DATA_FREQ_400Hz = 0x10
        #<summary>Acquisition frequency 800 Hz</summary>
        DATA_FREQ_800Hz = 0x20
        #<summary>Acquisition frequency 1600 Hz</summary>
        DATA_FREQ_1600Hz = 0x40         
    
    class AcquisitionType(Enum):
        """Acquisition types"""
        #<summary>Type NONE - used only on software side</summary>
        ACQ_NONE = 0x00
        #<summary>Buffered - streaming</summary>
        ACQ_TX_BUFFERED = 0x01
        #<summary>Direct - streaming</summary>
        ACQ_TX_DIRECT = 0x02
        #<summary>Memory file offload</summary>
        ACQ_READOUT = 0x03
    
    # DATA ACQUISITION section

    # CONFIGURATION section
    class HardwareSkills(Enum):
        """Hardware features"""
        #<summary>Hardware feature NONE - used only on software side</summary>
        SKILLS_HW_NONE = 0x0000
        #<summary>Gyroscope</summary>
        SKILLS_HW_GYRO = 0x0001
        #<summary>Accelerometer</summary>
        SKILLS_HW_AXL = 0x0002
        #<summary>Magnetometer</summary>
        SKILLS_HW_MAGN = 0x0004
        #<summary>High Dynamic Range (HDR) Accelerometer</summary>
        SKILLS_HW_HDR = 0x0008
        #<summary>Temperature</summary>
        SKILLS_HW_TEMP = 0x0010
        #<summary>Relative Humidity</summary>
        SKILLS_HW_RH = 0x0020
        #<summary>Barometric Pressure</summary>
        SKILLS_HW_BAR = 0x0040
        #<summary>Light intensity (i.e., visible)</summary>
        SKILLS_HW_LUM_VIS = 0x0080
        #<summary>Light intensity (i.e., infrared)</summary>
        SKILLS_HW_LUM_IR = 0x0100
        #<summary>Distance / Range</summary>
        SKILLS_HW_RANGE = 0x0200
        #<summary>Microphone</summary>
        SKILLS_HW_MIC = 0x0400

    class SoftwareSkills(Enum):
        """Software features"""
        #<summary>Software feature NONE - used only on software side</summary>
        SKILLS_SW_NONE = 0x0000
        #<summary>MPE library</summary>
        SKILLS_SW_MPE = 0x0001
        #<summary>MAD library</summary>
        SKILLS_SW_MAD = 0x0010

    class MEMS_ID(Enum):
        """Sensors identifiers within the system"""
        #<summary>Gyroscope</summary>
        SENSORS_GYRO = 0x01
        #<summary>Accelerometer</summary>
        SENSORS_AXL = 0x02
        #<summary>High Dynamic Range (HDR) Accelerometer</summary>
        SENSORS_HDR = 0x04
        #<summary>Magnetometer</summary>
        SENSORS_MAGN = 0x08	

    class MEMS_FullScaleMask(Enum):
        """Sensors full scale bit-mask"""
        #<summary>Gyroscope</summary>
        SENSORSFS_MASK_GYRO = 0x03
        #summary>Accelerometer</summary>
        SENSORSFS_MASK_AXL = 0x0c
        #<summary>High Dynamic Range (HDR) Accelerometer</summary>
        SENSORSFS_MASK_HDR = 0x30
        #<summary>Magnetometer</summary>
        SENSORSFS_MASK_MAGN = 0xc0

    # Gyroscope configurations dictionary (i.e., full scale and sensitivity coefficient)
    Gyroscope_CFG = frozendict({
        0x00: SensorConfig(245, 0.00875),
        0x01: SensorConfig(500, 0.0175),
        0x02: SensorConfig(1000, 0.035),
        0x03: SensorConfig(2000, 0.070)}) 
    
    # Accelerometer configurations dictionary (i.e., full scale and sensitivity coefficient)
    Accelerometer_CFG = frozendict(
        {0x00 : SensorConfig(4, 0.122) ,
        0x08 : SensorConfig(8, 0.244) ,
        0x0c : SensorConfig(16, 0.488) ,
        0x04 : SensorConfig(32, 0.976) }) 
    
    # Magnetometer configurations dictionary (i.e., full scale and sensitivity coefficient)
    Magnetometer_CFG = frozendict(
        {0x00 : SensorConfig(4, 1000.0/6842.0) ,
        0x40 : SensorConfig(8, 1000.0/3421.0) ,
        0x80 : SensorConfig(12, 1000.0/2281.0) ,
        0xc0 : SensorConfig(16, 1000.0/1711.0) }) 
    
    # High Dynamic Range (HDR) Accelerometer configurations dictionary (i.e., full scale and sensitivity coefficient)
    AccelerometerHDR_CFG = frozendict(
        {0x00 : SensorConfig(100, 49.0) ,
        0x10 : SensorConfig(200, 98.0) ,
        0x30 :SensorConfig(400, 195.0) ,
        }) 

    class GyroscopeFS(Enum):
        """Gyroscope full scales"""
        #<summary>245 dps</summary>
        GYR_FS_245dps = 0x00
        #<summary>500 dps</summary>
        GYR_FS_500dps = 0x01
        #<summary>1000 dps</summary>
        GYR_FS_1000dps = 0x02
        # <summary>2000 dps</summary>
        GYR_FS_2000dps = 0x03
        
    class AccelerometerFS(Enum): 
        """Accelerometer full scales"""
        #<summary>4 g</summary>
        AXL_FS_4g = 0x00
        #<summary>8 g</summary>
        AXL_FS_08g = 0x08
        #<summary>16 g</summary>
        AXL_FS_16g = 0x0c
        #<summary>32 g</summary>
        AXL_FS_32g = 0x04

    class MagnetometerFS(Enum):
        """Magnetometer full scales"""
        #<summary>4 Gauss</summary>
        MAG_FS_04G = 0x00
        #<summary>8 Gauss</summary>
        MAG_FS_08G = 0x40
        #<summary>12 Gauss</summary>
        MAG_FS_12G = 0x80
        #<summary>16 Gauss</summary>
        MAG_FS_16G = 0xc0

    class AccelerometerHDRFS(Enum):
        """High Dynamic Range (HDR) Accelerometer full scales"""
        #<summary>100 g</summary>
        HDR_FS_100g = 0x00
        #<summary>200 g</summary>
        HDR_FS_200g = 0x10
        #<summary>400 g</summary>
        HDR_FS_400g = 0x30

    class UserConfigMask(Enum):
        """Bit-masks for user configuration encoding / decoding operations"""
        #<summary>Extracts STANDBY configuration channel</summary>
        USER_CFG_MASK_AUTO_STANDBY = 0x0001
        #<summary>Extracts CIRCULAR MEMORY configuration channel</summary>
        USER_CFG_MASK_CIRCULAR_MEMORY = 0x0002
        #<summary>Extracts BLE STREAM configuration channel</summary>
        USER_CFG_MASK_BLE_STREAM = 0x0000
        #<summary>Extracts USB STREAM configuration channel</summary>
        USER_CFG_MASK_USB_STREAM = 0x0004
        #<summary>Extracts TCP STREAM configuration channel</summary>
        USER_CFG_MASK_TCP_STREAM = 0x0008
        #<summary>Extracts MQTT STREAM configuration channel</summary>
        USER_CFG_MASK_MQTT_STREAM = 0x000C
        #<summary>Extracts 9DOF MPE configuration channel</summary>
        USER_CFG_MASK_9DOF_MPE = 0x0020
        #<summary>Extracts SLOW FREQUENCY configuration channel</summary>
        USER_CFG_MASK_SLOW_FREQUENCY = 0x0040
        #<summary>Extracts MQTT commands configuration channel</summary>
        USER_CFG_MASK_MQTT_COMMANDS = 0x0080
        #<summary>mask for streaming channel decoding</summary>
        USER_CFG_MASK_STREAMING_CHANNEL = 0x001C

    # end CONFIGURATION section

    # CALIBRATION section
    class CalibrationType(IntEnum):
        """Calibration types"""
        #<summary>Accelerometer</summary>
        CALIB_TYPE_AXL = 0
        #<summary>Gyroscope</summary>
        CALIB_TYPE_GYR = 1
        #<summary>Magnetometer</summary>
        CALIB_TYPE_MAG = 2

    # end CALIBRATION section



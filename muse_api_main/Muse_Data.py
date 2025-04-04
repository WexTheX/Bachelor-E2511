""" Muse_Data.py: Muse data objects definitions.

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

class CommandResponse:
    """Command response object"""
    def __init__(self, buffer: bytearray):
        """Command response object constructor

        Args:
            buffer (bytearray): response buffer
        """
        from muse_api_main.Muse_HW import Muse_HW as MH
        if buffer is not None:
            self.rx = MH.Command(buffer[0])
            if buffer[2] > 0x80:
                self.tx = MH.Command(buffer[2] & 0x7F)
                self.read = True
            else:
                self.tx = MH.Command(buffer[2])
                self.read = False
            

            self.len = buffer[1]
            self.ack = MH.AcknowledgeType(buffer[3])

            self.payload = buffer[4:4+self.len-2]
        else:
            self.rx = MH.Command.CMD_NONE
            self.tx = MH.Command.CMD_NONE
            self.len = -1
            self.ack = MH.AcknowledgeType.ACK_NONE
            self.payload = None
    
class SensorConfig:
    """Sensors configuration structure
    """
    def __init__(self, fs: int, sens: float):
        """MEMS configuration object constructor

        Args:
            fs (int): Full scale
            sens (float): Sensitivity
        """
        self.FullScale = fs
        self.Sensitivity = sens
    
    def __str__(self):
        """Override of ToString method

        Returns:
            string containing the full scale and sensitivity of the sensor
        """
        details = '{}, {}'.format(self.FullScale, self.Sensitivity)
        return details

class FileInfo:
    """File Information structure
    """
    def __init__(self, timestamp: int, gyrConfig: SensorConfig, axlConfig: SensorConfig, magConfig: SensorConfig, hdrConfig: SensorConfig, mode: int, frequency: int):
        """FileInfo constructor

        Args:
            timestamp (int): Timestamp
            gyrConfig (SensorConfig): Gyroscope configuration
            axlConfig (SensorConfig): Accelerometer configuration
            magConfig (SensorConfig): Magnetometer configuration
            hdrConfig (SensorConfig): HDR Accelerometer configuration
            mode (int): Acquisition mode
            frequency (int): Acquisition frequency
        """
        from Muse_HW import Muse_HW as MH
        from Muse_Utils import Muse_Utils as MU
        self.timestamp = timestamp

        self.GyrConfig = gyrConfig
        self.AxlConfig = axlConfig
        self.MagConfig = magConfig
        self.HDRConfig = hdrConfig

        self.ModeString = MU.DataModeToString(mode)
        self.Mode = MH.DataMode(mode)
        

        self.Frequency = MH.DataFrequency(frequency).value
        self.FrequencyString = MU.DataFrequencyToString(MH.DataFrequency(frequency))
    
    def __str__(self) -> str:  
        """Override of ToString method
        """  
        return str(self.timestamp) + ", " + str(self.GyrConfig) + ", " +  str(self.AxlConfig) + ", " + str(self.MagConfig) + ", " + str(self.HDRConfig) + ", " + str(self.ModeString) + ", " + str(self.FrequencyString)

class UserConfig:
    """User Configuration object"""
   
    def __init__(self, standby: bool, memory: bool, ble: bool, usb: bool, tcp: bool, mqtt: bool, mpe9dof: bool, slowfreq: bool, mqttcommands: bool):
        """UserConfig object constructor

        Args:
            standby (bool): Boolean enabled/disabled standby status
            memory (bool): Boolean enabled/disabled circular memory status
            ble (bool): Boolean enabled/disabled BLE stream status
            usb (bool): Boolean enabled/disabled USB stream status
            tcp (bool): Boolean enabled/disabled TCP stream status
            mqtt (bool): Boolean enabled/disabled MQTT stream status
            mpe9dof (bool): Boolean enabled/disabled 9DOF MPE (only if MPE is present)
            slowfreq (bool): Boolean enabled/disabled slow frequency streaming
            mqttcommands (bool): Boolean enabled/disabled MQTT commands
        """
        self.AutoStandby = standby
        self.CircularMemory = memory
        self.StreamingSelectBLE = ble
        self.StreamingSelectUSB = usb
        self.StreamingSelectTCP = tcp
        self.StreamingSelectMQTT = mqtt
        self.MPE9DOF = mpe9dof
        self.SlowFrequency = slowfreq
        self.MQTTcommands = mqttcommands
    

    def __str__(self) -> str:
        """Override of ToString method
        """
        return ("USER CONFIGURATION\n"+"standby: "+str(self.AutoStandby) + "\ncircular memory: " + str(self.CircularMemory) + "\nchannel BLE streaming: " + str(self.StreamingSelectBLE) + 
                "\nchannel USB streaming: " + str(self.StreamingSelectUSB) + "\nchannel TCP streaming: " + str(self.StreamingSelectTCP) + "\nchannel MQTT streaming: " + str(self.StreamingSelectMQTT) + 
                "\nMPE 9DOF: " + str(self.MPE9DOF) + "\nslow frequency: " + str(self.SlowFrequency) + "\nMQTT commands: " + str(self.MQTTcommands))
    
class Light:
    """Light object"""
    def __init__(self, range: int , vis: int, ir: int, lux: int):
        """Light object constructor

        Args:
            range (int): range value
            vis (int): visible luminosity
            ir (int): infrared luminosity
            lux (int): LUX value computed from visible and infrared luminosity
        """
        self.range = range
        self.lum_vis = vis
        self.lum_ir = ir
        self.lux = lux
    

    def __str__(self) -> str:  
        """Override of ToString method
        """
        return str(self.range) + "\t" + str(self.lum_vis) + "\t" + str(self.lum_ir) + "\t" + str(self.lux)
        
class Muse_Data:
    """Muse Data object"""
    def __init__(self):
        """Muse Data object constructor
        """
        self.gyr = [0.0] * 3
        self.axl = [0.0] * 3
        self.mag = [0.0] * 3
        self.hdr = [0.0] * 3
        self.th = [0.0] * 3
        self.tp = [0.0] * 3
    
        self.light = Light(0,0,0,0)
        
        self.sound = 0

        self.quat = [0.0] * 4
        self.euler = [0.0] * 3

        self.timestamp = 0
        self.overall_timestamp = 0
    

    def ChannelsToString(arg) -> str:
    
        str_out = ""

        for i in range(len(arg) -1):
            str_out += str(arg[i]) + "\t"
            
        str_out += str(arg[len(arg) - 1])

        return str_out


    def __str__(self):
        """Override of ToString method
        """
        str = str(self.overall_timestamp) + "\t" + \
            str(self.timestamp) + "\t" + \
            self.ChannelsToString(self.gyr) + "\t" + \
            self.ChannelsToString(self.axl) + "\t" + \
            self.ChannelsToString(self.mag) + "\t" + \
            self.ChannelsToString(self.hdr) + "\t" + \
            self.ChannelsToString(self.th) + "\t" + \
            self.ChannelsToString(self.tp) + "\t" + \
            self.light + "\t" + \
            self.ChannelsToString(self.quat)

        return str
    




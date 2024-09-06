import machine
import time

class EEGSensorInterface:
    def __init__(self):
        # Pin definitions for EEG sensor and FPGA communication
        self.eeg_sensor_pin = machine.Pin(2, machine.Pin.IN)
        self.fpga_comm_pin = machine.Pin(4, machine.Pin.OUT)

        # I2C communication setup for sending data to the FPGA
        self.i2c = machine.I2C(0, scl=machine.Pin(5), sda=machine.Pin(6))

    def read_eeg_data(self):
        # Read EEG data from the sensor pin
        try:
            eeg_value = self.eeg_sensor_pin.value()
            return eeg_value
        except Exception as e:
            print(f"Error reading EEG data: {e}")
            return None

    def send_data_to_fpga(self, data):
        # Send the EEG data to the FPGA over I2C
        try:
            self.i2c.writeto(0x42, bytes([data]))
            time.sleep(0.1)  # Delay for signal stability
            print(f"Data sent to FPGA: {data}")
        except Exception as e:
            print(f"Error sending data to FPGA: {e}")

    def continuous_read_and_send(self):
        # Continuously read EEG data and send it to the FPGA
        while True:
            eeg_data = self.read_eeg_data()
            if eeg_data is not None:
                self.send_data_to_fpga(eeg_data)
            time.sleep(0.5)  # Adjust the frequency of reading/sending as needed
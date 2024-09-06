import machine
import time

class GlassesHardwareInit:
    def __init__(self):
        # Power pin for controlling the system's power state
        self.power_pin = machine.Pin(5, machine.Pin.OUT)
        
        # Pin for monitoring the EEG sensor
        self.eeg_sensor = machine.Pin(15, machine.Pin.IN)
        
        # LED indicator to visualize the system's state (on/off)
        self.led_indicator = machine.Pin(2, machine.Pin.OUT)
        
        # Additional pin for Bluetooth communication (e.g., to send data to a mobile device)
        self.bluetooth_comm = machine.Pin(10, machine.Pin.OUT)

    def power_on(self):
        # Turn on the power and indicate that the system is active with the LED
        self.power_pin.on()
        time.sleep(1)  # Small delay for power stabilization
        self.led_indicator.on()

    def power_off(self):
        # Turn off the power and LED indicator
        self.led_indicator.off()
        self.power_pin.off()

    def check_sensor(self):
        # Check the EEG sensor's state (active or inactive)
        if self.eeg_sensor.value() == 1:
            print("EEG Sensor Active")
        else:
            print("EEG Sensor Inactive")

    def bluetooth_send(self, data):
        # Send data via Bluetooth (placeholder for actual communication protocol)
        self.bluetooth_comm.on()
        time.sleep(0.5)  # Delay to simulate transmission
        self.bluetooth_comm.off()
        print(f"Data sent: {data}")
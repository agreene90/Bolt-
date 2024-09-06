import machine
import time

class GlassesHardwareInit:
    def __init__(self):
        self.power_pin = machine.Pin(5, machine.Pin.OUT)
        self.eeg_sensor = machine.Pin(15, machine.Pin.IN)
        self.led_indicator = machine.Pin(2, machine.Pin.OUT)

    def power_on(self):
        self.power_pin.on()
        time.sleep(1)
        self.led_indicator.on()

    def power_off(self):
        self.led_indicator.off()
        self.power_pin.off()

    def check_sensor(self):
        if self.eeg_sensor.value() == 1:
            print("EEG Sensor Active")
        else:
            print("EEG Sensor Inactive")

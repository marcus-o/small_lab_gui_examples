from modules.digitizers.digitizer import digitizer

import time
import numpy as np
from threading import Event
import seabreeze.spectrometers as seaspec


class oceanoptics_spectrometer(digitizer):
    def __init__(self, devnum=None, devserial=None):
        self.num_sensors = 1
        devices = seaspec.list_devices()
        print(devices)
        try:
            if devnum:
                self.dev = seaspec.Spectrometer(devices[devnum])
            elif devserial:
                self.dev = seaspec.Spectrometer.from_serial_number(devserial)
            else:
                self.dev = seaspec.Spectrometer(devices[0])
        except Exception:
            print('couldnt find spectrometer')
        self.lastframe = 0
        self.stop_event = Event()

    def setup(self, integration):
        self.integration = integration
        self.dev.integration_time_micros(integration*1000)
        time.sleep(integration/1000.*2)

    def frame(self, stop_event=Event(), inp=None, init_output=None):
        self.stop_event.clear()
        intensities = self.dev.intensities()
        while (np.sum(np.abs(self.lastframe-intensities)) == 0
               and (not self.stop_event.is_set())):
            intensities = self.dev.intensities()
            time.sleep(0.001)
        self.lastframe = intensities
        if (not self.stop_event.is_set()):
            return {
                'wavelength': self.dev.wavelengths(),
                'intensity': intensities,
                'source': 'oceanoptics_spectrometer',
                'success': True}
        else:
            return {
                'wavelength': 0,
                'intensity': 0,
                'source': 'oceanoptics_spectrometer',
                'success': False}

    def readout_continuous(self, stop_event=Event(),
                           inp=None, init_output=None):
        self.stop_event.clear()
        return {
            'wavelength': self.dev.wavelengths(),
            'intensity': self.dev.intensities(),
            'source': 'oceanoptics_spectrometer',
            'success': True}

    def stop(self):
        self.stop_event.set()

    def close(self):
        print('closing ocean optics spectrometer')

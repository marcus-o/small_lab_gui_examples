from modules.digitizers.digitizer import digitizer

import time
import numpy as np
from threading import Event


class oceanoptics_spectrometer_dummy(digitizer):
    def __init__(self, devnum=None, devserial=None):
        self.num_sensors = 1
        self.stop_event = Event()
        self.datalength = 1024

    def setup(self, integration):
        self.integration = integration

    def frame(self, stop_event=Event(), inp=None, init_output=None):
        time.sleep(self.integration/1000.)
        return {
            'wavelength': np.arange(self.datalength)+200.,
            'intensity': np.random.rand(self.datalength),
            'source': 'oceanoptics_spectrometer'}

    def readout_continuous(self, stop_event=Event(),
                           inp=None, init_output=None):
        time.sleep(self.integration/1000.)
        return {
            'wavelength': np.arange(self.datalength)+200.,
            'intensity': np.random.rand(self.datalength),
            'source': 'oceanoptics_spectrometer'}

    def stop(self):
        self.stop_event.set()

    def close(self):
        print('closing ocean optics spectrometer')

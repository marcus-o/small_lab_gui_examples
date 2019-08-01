from small_lab_gui.digitizers.digitizer import digitizer

import numpy as np
import time
from threading import Event


class mcs6a_dummy(digitizer):
    def __init__(self, path=''):
        self.num_sensors = 4
        self.nDev = 0
        self.range = 8192
        self.stop_event = Event()

        self.sweeps = 0

    def setup(self, integration, timepreset=0):
        self.integration = integration
        self.sweeps = 0

    def frame(self, stop_event=Event(), inp=None, init_output=None):
        self.stop_event.clear()
        time.sleep(self.integration/3000)
        self.sweeps = self.sweeps + 1
        self.data = np.random.rand(self.num_sensors, self.range)
        return {'data': self.data,
                'sweeps': self.sweeps,
                'starts': self.sweeps,
                'runtime': self.sweeps,
                'ofls': 0,
                'totalsum': self.data.sum(),
                'roisum': self.data.sum(),
                'roirate': self.data.sum(),
                'source': 'fast_mcs6a',
                'success': True}

    def start_continuous(self):
        self.data = np.zeros((self.num_sensors, self.range))
        self.sweeps = 0

    def readout_continuous(self, stop_event=Event(),
                           inp=None, init_output=None):
        self.sweeps = self.sweeps + 1
        self.data = self.data + np.random.rand(self.num_sensors, self.range)
        return {'data': self.data,
                'sweeps': self.sweeps,
                'starts': self.sweeps,
                'runtime': self.sweeps,
                'ofls': 0,
                'totalsum': self.data.sum(),
                'roisum': self.data.sum(),
                'roirate': self.data.sum(),
                'source': 'fast_mcs6a',
                'success': True}

    def stop(self):
        self.stop_event.set()
        pass

    def close(self):
        print('closing mcs6a')

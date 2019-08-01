# digitizer class for zurich instrument's hf2li lockin
from small_lab_gui.digitizers.digitizer import digitizer

import numpy as np
import time
from threading import Event


class hf2li_dummy(digitizer):
    def __init__(self, dev='dev1251'):
        self.num_sensors = 6
        self.dev = dev

    def setup(self, integration, timeconstant=0.1, order=2):
        self.integration = integration
        self.timeconstant = timeconstant
        self.order = order

    def frame(self, stop_event=None, inp=None, init_output=None):
        time.sleep(self.integration)
        return {
            'sig': np.random.rand(1)[0],
            'r': np.random.rand(1)[0],
            'theta': np.random.rand(1)[0],
            'sig2': np.random.rand(1)[0],
            'r2': np.random.rand(1)[0],
            'theta2': np.random.rand(1)[0],
            'source': 'zi_hf2li',
            'success': True}

    def readout_continuous(self, stop_event=Event(),
                           inp=None, init_output=None):
        time.sleep(self.integration)
        return {
            'sig': np.random.rand(1)[0],
            'r': np.random.rand(1)[0],
            'theta': np.random.rand(1)[0],
            'sig2': np.random.rand(1)[0],
            'r2': np.random.rand(1)[0],
            'theta2': np.random.rand(1)[0],
            'source': 'zi_hf2li',
            'success': True}

    def stop(self):
        pass

    def close(self):
        print('closing digitizer')

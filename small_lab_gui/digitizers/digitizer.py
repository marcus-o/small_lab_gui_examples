# abstract digitizer class (for everything that acquires data)
import numpy as np
import time


class digitizer():
    def __init__(self):
        self.num_sensors = 1
        pass

    def setup(self, integration=None):
        pass

    def frame(self, stop_event=None, inp=None, init_output=None):
        pass

    def stop(self):
        pass

    def close(self):
        pass


class digitizer_dummy(digitizer):
    def setup(self, integration):
        self.integration = integration

    def frame(self, stop_event=None, inp=None, init_output=None):
        time.sleep(self.integration)
        return {'data': np.random.rand(4096)*65300., 'sweeps': 100,
                'starts': 100, 'runtime': 0.53, 'source': 'dummy_digitizer'}

    def stop(self):
        pass

    def close(self):
        print('closing digitizer')

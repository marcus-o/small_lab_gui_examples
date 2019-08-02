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

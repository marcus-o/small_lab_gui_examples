from modules.digitizers.digitizer import digitizer

import numpy as np
import time
from math import floor


class camera_dummy_roi:
    xmin = 0
    xmax = 2048
    ymin = 0
    ymax = 512
    xbin = 1
    ybin = 1


class camera_dummy(digitizer):
    def setup(self, integration, roi):
        self.num_sensors = 1
        self.integration = integration
        self.roi = roi

    def frame(self, stop_event=None, inp=None, init_output=None):
        time.sleep(self.integration/1000.)
        return np.random.rand(
            floor((self.roi.ymax - self.roi.ymin)/self.roi.ybin),
            floor((self.roi.xmax - self.roi.xmin)/self.roi.xbin))*65300.

    def stop(self):
        pass

    def close(self):
        print('closing camera')

from modules.digitizers.digitizer import digitizer

import numpy as np


class pressure_controller_pfeiffer_tpg366_dummy(digitizer):
    def __init__(self, port, baud):
        self.num_sensors = 6

    def frame(self, stop_event=None, inp=None, init_output=None):
        prss = np.random.rand(8).tolist()
        valid = [True]*8
        prss[3] = 1.e8
        valid[3] = False
        return {
            'pressures': prss,
            'valid': valid,
            'source': 'tpg366',
            'success': True}

    def close(self):
        pass

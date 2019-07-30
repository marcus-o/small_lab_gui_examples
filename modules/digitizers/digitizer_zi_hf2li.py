# digitizer class for zurich instrument's hf2li lockin
from modules.digitizers.digitizer import digitizer

import numpy as np
import time
from threading import Event
import zhinst
import zhinst.ziPython


class hf2li(digitizer):
    def __init__(self, dev='dev1251'):
        self.num_sensors = 6
        self.dev = dev
        d = zhinst.ziPython.ziDiscovery()
        props = d.get(d.find(self.dev))
        self.daq = zhinst.ziPython.ziDAQServer(
            props['serveraddress'], props['serverport'], props['apilevel'])
        self.daq.connectDevice(self.dev, props['interfaces'][0])
        self.stop_event = Event()

    def setup(self, integration, timeconstant=0.1, order=2):
        self.integration = integration
        self.timeconstant = timeconstant
        self.order = order

        # set reference pll to DI0
        # self.daq.setInt('/{}/plls/0/adcselect'.format(self.dev), 4)
        # self.daq.setInt('/{}/plls/0/enable'.format(self.dev), 0)
        # self.daq.setInt('/{}/plls/0/enable'.format(self.dev), 1)

        # input settings channel 1
        # self.daq.setDouble('/{}/sigins/0/range'.format(self.dev), 0.3)
        # self.daq.setInt('/{}/sigins/0/ac'.format(self.dev), 1)
        # self.daq.setInt('/{}/sigins/0/imp50'.format(self.dev), 0)
        # self.daq.setInt('/{}/sigins/0/diff'.format(self.dev), 1)

        # demod 1 reference signal
        # self.daq.setInt('/{}/demods/0/oscselect'.format(self.dev), 0)
        # self.daq.setInt('/{}/demods/1/oscselect'.format(self.dev), 0)
        # self.daq.setDouble('/{}/demods/0/harmonic'.format(self.dev), 2)
        # self.daq.setDouble('/{}/demods/1/harmonic'.format(self.dev), 1)
        # self.daq.setDouble('/{}/demods/0/phaseshift'.format(self.dev), 42)
        # self.daq.setDouble('/{}/demods/1/phaseshift'.format(self.dev), 80)

        # low pass settings
        self.daq.setInt('/{}/demods/0/order'.format(self.dev), order)
        self.daq.setInt('/{}/demods/1/order'.format(self.dev), order)
        self.daq.setInt('/{}/demods/3/order'.format(self.dev), order)
        self.daq.setInt('/{}/demods/4/order'.format(self.dev), order)
        self.daq.setDouble(
            '/{}/demods/0/timeconstant'.format(self.dev), self.timeconstant)
        self.daq.setDouble(
            '/{}/demods/1/timeconstant'.format(self.dev), self.timeconstant)
        self.daq.setDouble(
            '/{}/demods/3/timeconstant'.format(self.dev), self.timeconstant)
        self.daq.setDouble(
            '/{}/demods/4/timeconstant'.format(self.dev), self.timeconstant)
        # computer transfer
        self.daq.setInt('/{}/demods/0/enable'.format(self.dev), 1)
        self.daq.setInt('/{}/demods/1/enable'.format(self.dev), 1)
        self.daq.setInt('/{}/demods/3/enable'.format(self.dev), 1)
        self.daq.setInt('/{}/demods/4/enable'.format(self.dev), 1)
        self.daq.setDouble('/{}/demods/0/rate'.format(self.dev), 200)
        self.daq.setDouble('/{}/demods/1/rate'.format(self.dev), 200)
        self.daq.setDouble('/{}/demods/3/rate'.format(self.dev), 200)
        self.daq.setDouble('/{}/demods/4/rate'.format(self.dev), 200)

    def frame(self, stop_event=Event(), inp=None, init_output=None):
        self.stop_event.clear()
        # wait for settle
        # settling times from zurich instruments for 99% signal
        settlingsTimeFactor = [4.61, 6.64, 8.41, 10.05,
                               11.60, 13.11, 14.57, 16.00]
        print(settlingsTimeFactor[self.order-1]*self.timeconstant)
        time.sleep(settlingsTimeFactor[self.order-1]*self.timeconstant)
        # clear old data
        self.daq.flush()
        self.daq.subscribe('/{}/demods/1/sample'.format(self.dev))
        self.daq.subscribe('/{}/demods/4/sample'.format(self.dev))
        # measurement time in seconds, timeout in ms
        data = self.daq.poll(self.integration, 500, 0, True)
        if len(data) > 0:
            # calulate mean over integration time
            sig = (
                data['/{}/demods/1/sample'.format(self.dev)]['x']
                + 1j*data['/{}/demods/1/sample'.format(self.dev)]['y']).mean()
            sig2 = (
                data['/{}/demods/4/sample'.format(self.dev)]['x']
                + 1j*data['/{}/demods/4/sample'.format(self.dev)]['y']).mean()
            r = np.abs(sig)
            theta = np.angle(sig)
            r2 = np.abs(sig2)
            theta2 = np.angle(sig2)
            return {'sig': sig, 'r': r, 'theta': theta,
                    'sig2': sig2, 'r2': r2, 'theta2': theta2,
                    'source': 'zi_hf2li',
                    'success': True}
        else:
            return {'sig': 0, 'r': 0, 'theta': 0,
                    'sig2': 0, 'r2': 0, 'theta2': 0,
                    'source': 'zi_hf2li',
                    'success': False}

    def readout_continuous(
          self, stop_event=Event(), inp=None, init_output=None):
        self.stop_event.clear()
        self.daq.subscribe('/{}/demods/1/sample'.format(self.dev))
        self.daq.subscribe('/{}/demods/4/sample'.format(self.dev))
        # measurement time in seconds, timeout in ms
        data = self.daq.poll(self.integration, 500, 0, True)
        if len(data) > 0:
            sig = (
                data['/{}/demods/1/sample'.format(self.dev)]['x']
                + 1j*data['/{}/demods/1/sample'.format(self.dev)]['y']).mean()
            sig2 = (
                data['/{}/demods/4/sample'.format(self.dev)]['x']
                + 1j*data['/{}/demods/4/sample'.format(self.dev)]['y']).mean()
            r = np.abs(sig)
            theta = np.angle(sig)
            r2 = np.abs(sig2)
            theta2 = np.angle(sig2)
            return {'sig': sig, 'r': r, 'theta': theta,
                    'sig2': sig2, 'r2': r2, 'theta2': theta2,
                    'source': 'zi_hf2li',
                    'success': True}
        else:
            return {'sig': 0, 'r': 0, 'theta': 0,
                    'sig2': 0, 'r2': 0, 'theta2': 0,
                    'source': 'zi_hf2li',
                    'success': False}

    def stop(self):
        self.stop_event.set()

    def close(self):
        print('closing h2fli')

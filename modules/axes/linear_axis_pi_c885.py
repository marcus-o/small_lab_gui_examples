# implement a physikinstrumente gcs controller
from modules.axes.linear_axis import linear_axis_controller
from modules.axes.linear_axis import linear_axis

import pipython
import pipython.pitools


class linear_axis_controller_pimicosc885(linear_axis_controller):
    num_axes = 0

    def __init__(self, controllername='C-885.XX SN 0',
                 gcsdll='C:/Users/Public/PI/PI_Programming_Files_PI_GCS2_DLL/PI_GCS2_DLL_x64.dll'):
        try:
            if gcsdll:
                self.gcsdll = gcsdll
                self.gcs = pipython.GCSDevice('C-885', gcsdll=self.gcsdll)
            else:
                self.gcsdll = gcsdll
                self.gcs = pipython.GCSDevice('C-885')
            # check if the controller is available
            print('Check if ' + controllername
                  + ' is present, otherwise you have to modify the code!')
            print(self.gcs.EnumerateUSB())
        except Exception:
            print('could not open gcs dll')
        try:
            # connect to the controller
            # self.gcs.ConnectUSB(controllername)
            self.gcs.OpenUSBDaisyChain(description=controllername)
            self.daisychainid = self.gcs.dcid
            self.gcs.ConnectDaisyChainDevice(1, self.daisychainid)
            # get contoller identification
            print(self.gcs.qIDN())
            # get detail info
            if self.gcs.HasqVER():
                print('version info: {}'.format(self.gcs.qVER().strip()))
        except Exception:
            print('could not connect to ' + controllername)

    def stop(self):
        self.gcs.STP()

    def close(self):
        print('closing c885')
        self.gcs.CloseConnection()


class linear_axis_pimicosc885(linear_axis):
    referenced = False

    def __init__(self, controller, addr):
        self.controller = controller
        if self.controller.gcsdll:
            self.gcs = pipython.GCSDevice('', gcsdll=self.controller.gcsdll)
        else:
            self.gcs = pipython.GCSDevice('')
        self.gcs.ConnectDaisyChainDevice(addr+2, self.controller.daisychainid)
        self.addr = addr
        self.position = self.get_position()
        self.gcs.SVO(self.gcs.axes[0], True)
        self.referenced = self.gcs.qFRF(self.gcs.axes[0])[self.gcs.axes[0]]

    def abs_move(self, aim_position):
        self.aim_position = aim_position
        self.gcs.MOV(self.gcs.axes[0], self.aim_position)

    def follow_move(self, stop_event=None, inp=None, init_output=None):
        self.position = self.get_position()
        if stop_event is not None:
            if self.gcs.qONT()[self.gcs.axes[0]]:
                stop_event.set()
        self.referenced = self.gcs.qFRF(self.gcs.axes[0])[self.gcs.axes[0]]
        return self.position

    def get_position(self):
        positions = self.gcs.qPOS()
        p = positions[self.gcs.axes[0]]
        self.position = float(p)
        return self.position

    def stop(self):
        self.gcs.HLT(self.gcs.axes[0])

    def home(self):
        self.gcs.FNL(self.gcs.axes[0])

    def zero(self):
        self.gcs.POS(self.gcs.axes[0], (0,))

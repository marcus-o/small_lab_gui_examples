from small_lab_gui.digitizers.digitizer import digitizer
from small_lab_gui.company_dll.greateyes_wrapper import greatEyesDll

from math import floor
from threading import Event


class greateyes(digitizer):
    def __init__(self, path='C:\\Users\\Labor\\Documents\\python\\small_lab_gui\\company_dll\\greateyes_sdk\\c\\bin\\x64'):
        self.num_sensors = 1
        self.ge = greatEyesDll(path, 1)
        if self.ge.GetNumberOfConnectedCams():
            print('found camera')
            print('connection status (==6?):')
            print(self.ge.ConnectCamera(0).statusMSG)
            print('cam settings status:')
            self.ge.ClearFifo(0)
            print(self.ge.CamSettings(self.ge.readoutSpeed_3_MHz,
                                      1000, 1, 1, 0).statusMSG)
            print('camera temperature:')
            self.ge.ClearFifo(0)
            print(self.ge.TemperatureControl_GetTemperature(1, 0).temperature)
            print('camera temperature status:')
            print(self.ge.TemperatureControl_GetTemperature(1, 0).statusMSG)
            print('camera level string:')
            print(self.ge.TemperatureControl_GetLevelString(8, 0).levels)

        else:
            print('no camera')

    def setup(self, integration, roi):
        self.integration = integration
        self.ge.ClearFifo(0)
        print('cam settings status:')
        print(self.ge.CamSettings(
            self.ge.readoutSpeed_3_MHz,
            int(floor(self.integration)), 1, 1, 0).statusMSG)
        # set roi and bin here somehow
        self.roi = roi
        # apply to camera roi
        self.ge.OpenShutter(1, 0)

    def frame(self, stop_event=Event(), inp=None, init_output=None):
        # start a frame
        print('start status (==0?):')
        self.ge.ClearFifo(0)
        print(
            self.ge.StartMeasurement(True, False, True, False, 1, 0).statusMSG)

        # wait for frame to finish or stop event
        while self.ge.DllIsBusy(0) and (not stop_event.is_set()):
            pass

        # in case of stop event stop frame
        if stop_event.is_set():
            self.stop()
            ret = self.ge.GetMeasurementData(0)

        # in case of frame check if it is fine and process
        if not self.ge.DllIsBusy(0):
            ret = self.ge.GetMeasurementData(0)
            data = ret.pInDataStart
            print('data status (==0?):')
            status = ret.statusMSG
            print(status)
            if status == 0:
                data = data.transpose()
                data = data.reshape(self.ge.numPixelInX[0],
                                    self.ge.numPixelInY[0])
                return data

    def stop(self):
        print('stop success:')
        print(self.ge.StopMeasurement(0))
        pass

    def close(self):
        print('closing greateyes camera')
        self.ge.DisconnectCamera(0)

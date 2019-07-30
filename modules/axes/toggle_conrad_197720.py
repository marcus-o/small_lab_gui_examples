from modules.axes.toggle import toggle_controller
from modules.axes.toggle import toggle_axis

import serial


class toggle_controller_conrad_197720(toggle_controller):
    num_toggles = 8

    def __init__(self, port='COM1', init_state=[False]*8):
        self.port = port
        try:
            self.ser = serial.Serial(
                port=self.port, baudrate=19200, bytesize=8,
                parity=serial.PARITY_NONE, stopbits=1, timeout=10,
                writeTimeout=10)
            print('opened')
            print(self.ser)
            for cnt in range(0, self.num_toggles):
                self.axes.append(toggle_axis_conrad_197720(
                    ser=self.ser, address=0, relais=cnt))
                self.axes[-1].toggle(inp={'on': init_state[cnt]})
        except Exception:
            print('could not open conrad_197720 com port')

    def setup(self, ser, addr=1):
        # not necessary for one card
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        command = 1
        data = 0
        chksum = command ^ addr ^ data
        mes = bytearray(4)
        mes[0] = command
        mes[1] = addr
        mes[2] = data
        mes[3] = chksum
        ser.write(mes)
        ser.read(4)

    def close(self):
        print('closing conrad relais card')
        for axis in self.axes:
            axis.toggle(inp={'on': False})
        self.ser.close()


class toggle_axis_conrad_197720(toggle_axis):
    def __init__(self, ser, address=0, relais=0):
        self.ser = ser
        self.addr = address
        self.relais = relais

    def toggle(self, stop_event=None, inp=None, init_output=None):
        # stop_event and init_output are ignored,
        # but there for compatability with measurement class
        if not ('on' in inp):
            inp['on'] = False
        # build message
        if inp['on']:
            command = 6
        else:
            command = 7
        data = 2**(self.relais)
        chksum = command ^ self.addr ^ data
        mes = bytearray(4)
        mes[0] = command
        mes[1] = self.addr
        mes[2] = data
        mes[3] = chksum

        # send message
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        self.ser.write(mes)
        return self.ser.read(4)

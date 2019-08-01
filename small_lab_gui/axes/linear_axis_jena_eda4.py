# implement a piezo jena eda4 controller
from small_lab_gui.axes.linear_axis import linear_axis_controller
from small_lab_gui.axes.linear_axis import linear_axis

import serial


class linear_axis_controller_jena_eda4(linear_axis_controller):
    num_axes = 0

    def __init__(self, port, baud=9600):
        print('opening jena eda4 com port')
        self.port = port
        self.baud = baud
        try:
            self.ser = serial.Serial(port=port, baudrate=baud, timeout=1)
        except Exception:
            print('could not open jena eda4 com port')

    def reinit(self):
        print('reopening jena eda4 com port')
        self.ser.close()
        try:
            self.ser = serial.Serial(
                port=self.port, baudrate=self.baud, timeout=1)
        except Exception:
            print('could not open jena eda4 com port')

    def command(self, command):
        self.ser.write(command.encode())

    def readline(self):
        return self.ser.readline()

    def stop(self):
        pass

    def close(self):
        print('closing jena eda4 com port')
        self.ser.close()


class linear_axis_piezojena_eda4(linear_axis):
    def __init__(self, controller, addr, tol_um=0.1, in_min_um=0.0,
                 in_max_um=79.9, out_min=0.0, out_max=65.5):
        self.tol_um = tol_um
        self.controller = controller
        self.addr = addr
        self.referenced = True
        command = 'setk,' + '{0}'.format(int(self.addr)) + ',1\r\n'
        self.controller.command(command)
        self.posval_to_um = (in_max_um - in_min_um) / (out_max - out_min)
        self.get_position()

    def reinit(self):
        command = 'setk,' + '{0}'.format(int(self.addr)) + ',1\r\n'
        self.controller.command(command)
        self.get_position()

    def abs_move(self, aim_position):
        self.aim_position = aim_position
        command = (
            'set,' + '{0}'.format(int(self.addr)) + ','
            + '{0}'.format(int(self.aim_position*1000./self.posval_to_um))
            + '\r\n')
        self.controller.command(command)

    def follow_move(self, stop_event=None, inp=None, init_output=None):
        self.position = self.get_position()
        if (self.position - self.aim_position)**2 < self.tol_um**2:
            stop_event.set()
        return self.position

    def get_position(self):
        command = 'measure\r\n'
        self.controller.command(command)
        p = self.controller.readline().decode("utf-8")
        aws = p.split(',')
        if aws[0] == 'aw':
            self.position = float(aws[self.addr+1])/1000.*self.posval_to_um
            return self.position

    def stop(self):
        pass

    def home(self):
        pass

    def zero(self):
        pass

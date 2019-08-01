# implement a micos pollux controller
from small_lab_gui.axes.linear_axis import linear_axis_controller
from small_lab_gui.axes.linear_axis import linear_axis

import serial


class linear_axis_controller_pollux(linear_axis_controller):
    num_axes = 0

    def __init__(self, port, baud):
        try:
            self.ser = serial.Serial(port=port, baudrate=baud, timeout=1)
            print('opened pollux com port')
        except Exception:
            print('could not open pollux com port')

    def command(self, command):
        self.ser.write(command.encode())

    def readline(self):
        return self.ser.readline()

    def stop(self):
        self.ser.write(3)

    def close(self):
        print('closing pollux com port')
        self.ser.close()


class linear_axis_pollux(linear_axis):
    def __init__(self, controller, addr):
        self.controller = controller
        self.addr = addr
        self.get_position()
        self.referenced = True

    def abs_move(self, aim_position):
        self.aim_position = aim_position
        command = ('{:.6f}'.format(self.aim_position)
                   + ' ' + self.addr + ' nmove\r\n')
        self.controller.command(command)

    def follow_move(self, stop_event=None, inp=None, init_output=None):
        command = self.addr + ' nstatus\r\n'
        self.controller.command(command)
        p = self.controller.readline()
        if not p[0] % 2:
            stop_event.set()

        self.position = self.get_position()
        return self.position

    def get_position(self):
        command = self.addr + ' npos\r\n'
        self.controller.command(command)
        p = self.controller.readline()
        self.position = float(p)
        return float(p)

    def stop(self):
        command = self.addr + ' nabort\r\n'
        self.controller.command(command)

    def home(self):
        command = self.addr + ' ncalibrate\r\n'
        self.controller.command(command)

    def zero(self):
        self.aim_position = 0
        command = '0 ' + self.addr + ' setnpos\r\n'
        self.controller.command(command)

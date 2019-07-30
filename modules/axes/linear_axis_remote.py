# implement a remote axis controller
from modules.axes.linear_axis import linear_axis_controller
from modules.axes.linear_axis import linear_axis

from modules.helper import asynczmq


class linear_axis_controller_remote(linear_axis_controller):
    num_axes = 0

    def __init__(self, address='localhost', port=5557):
        print('opening zmq')
        self.zmq = asynczmq.zmqRequester(address, port)

    def command(self, command, value):
        self.zmq.send(command, value)

    def close(self):
        print('closing zmq')


class linear_axis_remote(linear_axis):
    def __init__(self, controller, name):
        self.controller = controller
        self.name = name
        self.referenced = True

    def abs_move(self, aim_position):
        self.aim_position = aim_position
        command = self.name + '_abs_move'
        self.controller.command(command, self.aim_position)

    def rel_move(self, aim_position):
        self.aim_position = aim_position
        command = self.name + '_rel_move'
        self.controller.command(command, self.aim_position)

    def get_position(self):
        command = self.name + '_get_position'
        p = self.controller.command(command)
        self.position = float(p)
        return self.position

    def follow_move(self, stop_event=None, inp=None, init_output=None):
        self.position = self.get_position()
        if stop_event is not None:
            if self.aim_position == self.position:
                stop_event.set()
        return self.position

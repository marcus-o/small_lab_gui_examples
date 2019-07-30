from modules.axes.linear_axis import linear_axis_controller
from modules.axes.linear_axis import linear_axis


class linear_axis_controller_dummy(linear_axis_controller):
    num_axes = 4

    def __init__(self, port=0, baud=0, *argv):
        for _ in range(0, self.num_axes):
            self.axes.append(linear_axis())


class linear_axis_dummy(linear_axis):
    referenced = False

    def __init__(self, *argv):
        pass

    def setup(self, *argv):
        pass

    def home(self, *argv):
        self.referenced = True

    def zero(self, *argv):
        self.position = 0
        self.aim_position = 0

    def stop(self, *argv):
        pass

    def close(self):
        pass

    def abs_move(self, aim_position):
        self.aim_position = aim_position

    def follow_move(self, stop_event=None, inp=None, init_output=None):
        self.position = (self.position
                         + (self.aim_position - self.position) / 2.)
        if abs(self.aim_position - self.position) < 0.1:
            self.position = self.aim_position

        if self.position == self.aim_position:
            stop_event.set()
        return self.position

    def get_position(self):
        return self.position

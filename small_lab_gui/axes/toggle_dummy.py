from small_lab_gui.axes.toggle import toggle_controller
from small_lab_gui.axes.toggle import toggle_axis


class toggle_controller_dummy(toggle_controller):
    num_toggles = 8

    def __init__(self, *argv):
        for _ in range(0, self.num_toggles):
            self.axes.append(toggle_axis_dummy())

    def setup(self, *argv):
        pass

    def close(self):
        pass


class toggle_axis_dummy(toggle_axis):
    def __init__(self, *argv):
        pass

    def setup(self, *argv):
        pass

    def toggle(self, stop_event=None, inp=None, init_output=None):
        pass

    def close(self):
        pass

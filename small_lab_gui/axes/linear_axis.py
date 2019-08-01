# abstract linear axis movement class
# (for everything that moves along a direction)


class linear_axis_controller():
    num_axes = 0
    axes = []

    def __init__(self):
        pass

    def setup(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


class linear_axis():
    position = 0
    aim_position = 0
    referenced = True

    def __init__(self):
        pass

    def setup(self):
        pass

    def abs_move(self, position):
        pass

    def home(self):
        pass

    def zero(self):
        pass

    def get_position(self):
        pass

    def follow_move(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass

from modules.helper import bokeh_gui_helper as bgh

from modules.axes.linear_axis_remote import linear_axis_controller_remote
from modules.axes.linear_axis_remote import linear_axis_remote

import bokeh
import bokeh.layouts
import bokeh.plotting


class overlap_gui():
    def __init__(self, doc, delayer):
        self.running = bgh.running()
        self.title = 'Overlap Finder'
        # measurement thread
        self.cont_readout = None
        # bokeh doc for callback
        self.doc = doc
        # delay piezo
        self.delayer = delayer

        self.offset_slider = bokeh.models.widgets.Slider(
            start=0, end=250, value=0, step=1., title="Piezo Position")
        # arrange layout
        self.inputs = bokeh.layouts.widgetbox(self.offset_slider)
        self.layout = bokeh.layouts.row(self.inputs)
        # callback
        self.offset_slider.on_change('value', self.move_piezo)

    def move_piezo(self, attr, old_val, new_val):
        # in case this measurement is running, do nothing
        if self.running.am_i_running(self):
            pass
        # in case a different measurement is running, do nothing
        elif self.running.is_running():
            pass
        # otherwise request piezo move
        else:
            self.running.now_running(self)
            self.delayer.abs_move(new_val)
            self.running.now_stopped()

    def close(self):
        pass


class tof_session_handler(bgh.bokeh_gui_session_handler):
    def open_session(self, doc):
        # connect to movement server piezo axis
        movement_server = linear_axis_controller_remote()
        piezo = linear_axis_remote(movement_server, 'experimental_target_x')

        # alignment tab
        overlapgui = overlap_gui(doc=doc, delayer=piezo)

        self.title = 'Overlapfinder'
        self.tabs = [
            {'layout': overlapgui.layout, 'title': overlapgui.title}]

        # this list is auto-closed, all close functions of the added objects
        # are called at session destruction
        self.close_list.append(overlapgui)
        self.close_list.append(piezo)


print('start overlapfinder')
# start the server
bgh.bokeh_gui_helper(tof_session_handler(), 5026)

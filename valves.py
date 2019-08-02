from functools import partial
import numpy as np

import bokeh
import bokeh.layouts
import bokeh.plotting

import jinja2

from small_lab_gui.helper import measurement
from small_lab_gui.helper import bokeh_gui_helper as bgh
from small_lab_gui.helper import bokeh_plot_helper as bph

testing = True
if not testing:
    # for the lab
    from small_lab_gui.digitizers.digitizer_pfeiffer_tpg366 \
        import pressure_controller_pfeiffer_tpg366
    from small_lab_gui.axes.toggle_conrad_197720 \
        import toggle_controller_conrad_197720
else:
    # for testing
    from small_lab_gui.digitizers.digitizer_pfeiffer_tpg366_dummy \
        import pressure_controller_pfeiffer_tpg366_dummy as pressure_controller_pfeiffer_tpg366
    from small_lab_gui.axes.toggle_dummy import toggle_controller_dummy \
        as toggle_controller_conrad_197720

template = jinja2.Template('''<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <title>{{ title }}</title>
        {{ bokeh_css }}
        {{ bokeh_js }}
        <style>
        .bk-root .bk-widget button {
            min-width: 100%;
            font-size: 40px;
        }
        </style>

    </head>
    <body>
        {{ plot_div|indent(8) }}
        {{ plot_script|indent(8) }}
    </body>
</html>
''')


class toggle_gui():
    def __init__(self, doc, toggler, name='default', toggle_names=None):
        self.toggler = toggler
        self.name = name
        self.areyousureyouaresure = False

        self.inputs = bokeh.layouts.widgetbox(width=200)
        self.titleBar = bokeh.models.widgets.Paragraph(text=self.name)
        self.inputs.children.append(self.titleBar)

        self.toggle_names = []
        self.axis_btns = []
        for cnt in range(0, self.toggler.num_toggles):

            if toggle_names:
                self.toggle_names.append(toggle_names[cnt])
            else:
                self.toggle_names.append(str(cnt+1))

            axis = bokeh.models.widgets.Button(
                label='Open ' + self.toggle_names[cnt],
                button_type='danger', disabled=False)
            axis.on_click(partial(self.toggle, axis=cnt))
            self.axis_btns.append(axis)
            self.inputs.children.append(axis)

        self.areyousureyouaresureBtn = bokeh.models.widgets.Button(
            label='Override', button_type='danger', disabled=False)
        self.areyousureyouaresureBtn.on_click(self.override)
        self.inputs.children.append(self.areyousureyouaresureBtn)

        self.layout = bokeh.layouts.row(self.inputs)

    def toggle(self, axis):
        if self.axis_btns[axis].button_type == 'danger':
            self.axis_btns[axis].label = 'Close ' + self.toggle_names[axis]
            self.axis_btns[axis].button_type = 'success'
            self.toggler.axes[axis].toggle(inp={'on': True})
        else:
            if self.axis_btns[axis].button_type == 'success':
                self.axis_btns[axis].label = 'Open ' + self.toggle_names[axis]
                self.axis_btns[axis].button_type = 'danger'
                self.toggler.axes[axis].toggle(inp={'on': False})

    def override(self):
        if self.areyousureyouaresure:
            self.areyousureyouaresure = False
            self.areyousureyouaresureBtn.button_type = 'danger'
        else:
            self.areyousureyouaresure = True
            self.areyousureyouaresureBtn.button_type = 'success'
            for btn in self.axis_btns:
                btn.disabled = False


class pressure_gui():
    def __init__(self, doc, pressure_controller,
                 prsmax, name='default', sensor_names=None):

        self.name = name
        self.doc = doc

        self.pressure_controller = pressure_controller
        self.prsmax = prsmax

        self.inputs = bokeh.layouts.widgetbox(width=500)
        self.titleBar = bokeh.models.widgets.Paragraph(text=self.name)
        self.inputs.children.append(self.titleBar)

        self.sensor_names = []
        self.prs_displays = []
        for cnt in range(0, self.pressure_controller.num_sensors):
            self.prs_displays.append(bokeh.models.widgets.Button(
                label='', button_type='primary', disabled=True))
            self.inputs.children.append(self.prs_displays[-1])
            self.sensor_names.append(str(cnt+1))

        if sensor_names:
            self.sensor_names = sensor_names

        self.layout = bokeh.layouts.row(self.inputs)

        if type(self) == pressure_gui:
            self.start()

    def start(self):
        self.cont_pressure_readout = measurement.measurement(
            inputs=None,
            sequence=[
                self.pressure_controller.frame,
                measurement.sleep_function(0.2)],
            update=measurement.bokeh_update_function(self.update, self.doc),
            init=None,
            finish=None,
            save_output=False)
        self.cont_pressure_readout.start()

    def stop(self):
        self.cont_pressure_readout.stop()
        self.cont_pressure_readout.join()

    def close(self):
        self.stop()

    def update(self, data):
        pressures = data[0]['pressures']
        valid = data[0]['valid']
        if data[0]['success']:
            for idx, handle in enumerate(self.prs_displays):
                if not valid[idx]:
                    handle.button_type = 'primary'
                    handle.label = 'unknown'
                else:
                    handle.label = (self.sensor_names[idx] + ': '
                                    + '{:.2e}'.format(pressures[idx])
                                    + ' mbar')
                    if pressures[idx] < self.prsmax[idx]:
                        handle.button_type = 'success'
                    else:
                        handle.button_type = 'danger'


class pressure_valve_gui(pressure_gui):
    def __init__(self, doc, pressure_controller, prsmax, toggler,
                 valve_prs_max, sensor_names, valve_names):
        # get the pressure gui
        super().__init__(doc, pressure_controller, prsmax,
                         name='Pressure Readout', sensor_names=sensor_names)

        # add the valve control
        self.toggler = toggler
        self.valve_control = toggle_gui(
            doc, toggler, name='Valve Control', toggle_names=valve_names)

        self.layout = bokeh.layouts.row(
            bokeh.layouts.column(self.inputs),
            bokeh.layouts.column(self.valve_control.inputs))

        # remember the contraints
        self.valve_prs_max = valve_prs_max
        # start pressure readout
        self.start()

    def stop(self):
        self.cont_pressure_readout.stop()
        for handle in self.toggler.axes:
            handle.toggle(inp={'on': False})

    def update(self, data):
        pressures = data[0]['pressures']
        valid = data[0]['valid']
        if data[0]['success']:
            # update pressure display
            for idx, handle in enumerate(self.prs_displays):
                if not valid[idx]:
                    handle.button_type = 'primary'
                    handle.label = 'unknown'
                else:
                    handle.label = (
                        self.sensor_names[idx]
                        + ': '
                        + '{:.2e}'.format(pressures[idx])
                        + ' mbar')
                    if pressures[idx] < self.prsmax[idx]:
                        handle.button_type = 'success'
                    else:
                        handle.button_type = 'danger'
            # check contraints
            if not self.valve_control.areyousureyouaresure:
                for constr in self.valve_prs_max:
                    if (
                          (not valid[constr['prs1']])
                          or (not valid[constr['prs2']])
                          or (np.abs(pressures[constr['prs1']]
                                     - pressures[constr['prs2']])
                              > constr['diff_max'])):
                        if self.valve_control.axis_btns[constr['valve_num']].button_type == 'danger':
                            self.valve_control.axis_btns[constr['valve_num']].disabled = True
                    else:
                        self.valve_control.axis_btns[constr['valve_num']].disabled = False


class pressure_session_handler(bgh.bokeh_gui_session_handler):
    def open_session(self, doc):
        # use template to change font size (see across room)
        doc.template = template

        # open pressure sensors
        port = "COM5"
        baud = 115200
        # sensor = pressure_sensors.pressure_controller_dummy()
        sensor = pressure_controller_pfeiffer_tpg366(port=port, baud=baud)

        # open the valve toggler
        port = "COM14"
        # toggler = toggles.toggle_controller_dummy()
        toggler = toggle_controller_conrad_197720(port)

        raw_readout = pressure_valve_gui(
            doc=doc,
            pressure_controller=sensor,
            prsmax=[5e-1, 5e-1, 5e-1, 5e-1, 5e-1, 5e-1],
            toggler=toggler,
            valve_prs_max=[
                {'prs1': 0, 'prs2': 1, 'diff_max': 0.1, 'valve_num': 0},
                {'prs1': 1, 'prs2': 2, 'diff_max': 0.2, 'valve_num': 1},
                {'prs1': 1, 'prs2': 2, 'diff_max': 0.3, 'valve_num': 2},
                {'prs1': 2, 'prs2': 3, 'diff_max': 0.5, 'valve_num': 5}, ],
            sensor_names=[
                'SwB', 'HHG', 'Delay', 'Experiment', 'Prevac', 'Target'],
            valve_names=[
                'SwB/HHG', 'XUV ARM', 'IR ARM',
                '?', '?', 'Exp Chamber', '?', '?'])
        tab1 = bokeh.models.widgets.Panel(
            child=raw_readout.layout,
            title='Pressure Readout and Valve Control')
        tabs = bokeh.models.widgets.Tabs(tabs=[tab1])
        doc.add_root(tabs)
        doc.title = 'Valve Control'

        # this list is auto-closed, all close functions of the added objects
        # are called at session destruction
        self.close_list.append(raw_readout)
        self.close_list.append(toggler)
        self.close_list.append(sensor)


print('start valves')
# start the server
bgh.bokeh_gui_helper(pressure_session_handler(), 4089)

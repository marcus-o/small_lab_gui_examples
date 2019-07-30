import numpy as np
from functools import partial
import os
import datetime
import matplotlib.pyplot as plt
import h5py

from modules.helper import bokeh_gui_helper as bgh
from modules.helper import measurement

import bokeh

# for the lab
# from modules.digitizers.digitizer_fast_mcs6a import mcs6a
# from modules.axes.linear_axis_jena_eda4 import linear_axis_controller_jena_eda4
# from modules.axes.linear_axis_jena_eda4 import linear_axis_piezojena_eda4

testing = True
if not testing:
    # for the lab
    from modules.digitizers.digitizer_greateyes_camera import greateyes
    from modules.digitizers.digitizer_camera_dummy import camera_dummy_roi \
        as camera_dummy_roi
    from modules.axes.linear_axis_jena_eda4 import \
        linear_axis_controller_jena_eda4
    from modules.axes.linear_axis_jena_eda4 import \
        linear_axis_piezojena_eda4
    from modules.helper.postToELOG import elog
else:
    # for testing
    from modules.digitizers.digitizer_camera_dummy import camera_dummy \
        as greateyes
    from modules.digitizers.digitizer_camera_dummy import camera_dummy_roi \
        as camera_dummy_roi
    from modules.axes.linear_axis_dummy import linear_axis_controller_dummy \
        as linear_axis_controller_jena_eda4
    from modules.axes.linear_axis_dummy import linear_axis_dummy \
        as linear_axis_piezojena_eda4
    from modules.helper.postToELOG_dummy import elog_dummy as elog


class continuous_acquisition_gui():
    def __init__(self, doc, sensor, running, roi):
        self.cont_camera_readout = None
        self.doc = doc
        self.sensor = sensor
        self.running = running
        self.roi = roi

        # pcolor plot to display results
        self.imagePlot = bgh.plot_false_color()
        self.imagePlot.image()

        # Set up widgets
        self.startBtn = bokeh.models.widgets.Button(
            label='Start', button_type='success')
        self.integrationInput = bokeh.models.widgets.TextInput(
            title='Integration time [s]', value='1.0')
        self.updateBtn = bokeh.models.widgets.Button(
            label='Update Roi', button_type='success')

        # start thread callback
        self.startBtn.on_click(self.start)
        self.updateBtn.on_click(self.change_roi)
        self.integrationInput.on_change('value', self.change_integration)
        # arrange items
        self.inputs = bokeh.layouts.widgetbox(
            self.startBtn, self.integrationInput, self.updateBtn)
        self.layout = bokeh.layouts.row(
            self.inputs, self.imagePlot.element, width=800)

    def start(self):
        # in case this measurement is running, stop it
        if self.running.am_i_running(self):
            self.stop()
        # in case a different measurement is running, do nothing
        elif self.running.is_running():
            pass
        else:
            # set running indicator to block double readout
            self.running.now_running(self)

            # bin is wrong for spectrum
            if type(self) == continuous_acquisition_gui:
                self.roi.xbin = 1
                self.roi.ybin = 1

            self.startBtn.label = 'Stop'
            self.startBtn.button_type = 'danger'
            self.integration = float(self.integrationInput.value)*1000.

            self.cont_camera_readout = measurement.measurement(
                inputs=None,
                sequence=[self.sensor.frame],
                update=measurement.bokeh_update_function(
                    self.update, self.doc),
                init=partial(
                    self.sensor.setup,
                    integration=self.integration,
                    roi=self.roi),
                finish=None,
                save_output=False)
            self.cont_camera_readout.start()

    def change_integration(self, attr, old_val, new_val):
        if self.running.am_i_running(self):
            self.stop()
            self.start()

    def change_roi(self):
        self.roi.xmin = int(np.floor(self.imagePlot.plot.x_range.start))
        self.roi.xmax = int(np.ceil(self.imagePlot.plot.x_range.end))
        self.roi.ymin = int(np.floor(self.imagePlot.plot.y_range.start))
        self.roi.ymax = int(np.ceil(self.imagePlot.plot.y_range.end))

        if self.running.is_running():
            self.stop()
            self.start()

    def stop(self):
        if not (self.cont_camera_readout is None):
            self.cont_camera_readout.stop()
            self.cont_camera_readout.join()
        self.running.now_stopped()
        self.startBtn.label = 'Start'
        self.startBtn.button_type = 'success'

    def close(self):
        self.stop()

    def update(self, data):
        im_data = data[0]
        # update plots
        try:
            self.imagePlot.update(
                num=0,
                x=np.arange(im_data.shape[1]),
                y=np.arange(im_data.shape[0]),
                z=im_data)
        except Exception as e:
            print('plot error')
            print(e)


class continuous_spectrum_gui(continuous_acquisition_gui):
    def __init__(self, doc, sensor, running, roi):
        super().__init__(doc, sensor, running, roi)
        # spectrum source and line plot instead of image plot
        self.linePlot = bgh.plot_2d()
        self.linePlot.line(legend='Spectrum')

        # save button
        self.saveBtn = bokeh.models.widgets.Button(
            label='Save Spectrum', button_type='success')
        self.saveBtn.on_click(partial(self.save_spectrum))
        self.saveNameInput = bokeh.models.widgets.TextInput(
            title='Legend Name', value='Name')
        # arrange items
        self.inputs = bokeh.layouts.widgetbox(
            self.startBtn, self.integrationInput,
            self.saveBtn, self.saveNameInput)
        self.layout = bokeh.layouts.row(
            self.inputs, self.linePlot.element, width=800)

    def start(self):
        self.roi.ybin = self.roi.ymax - self.roi.ymin
        super().start()

    def save_spectrum(self):
        self.linePlot.save_current(name=self.saveNameInput.value, num=0)

    def update(self, data):
        line_data = data[0]
        # update plots
        try:
            self.linePlot.update(
                num=0,
                x=np.arange(0, line_data.size),
                y=line_data.flatten())
        except Exception as e:
            print('plot error')
            print(e)


class measurement_gui(continuous_spectrum_gui):
    def __init__(self, doc, sensor, running, roi, delayer, logbook):
        super().__init__(doc, sensor, running, roi)
        self.logbook = logbook

        # delay piezo
        self.delayer = delayer

        # scan table button
        self.scanTableBtn = bokeh.models.widgets.Toggle(label='Scan Table')
        # measuement inputs
        self.zeroDelInput = bokeh.models.widgets.TextInput(
            title='Zero Delay [um]', value='50.')
        self.startDelInput = bokeh.models.widgets.TextInput(
            title='Start Delay [fs]', value='-10.')
        self.stopDelInput = bokeh.models.widgets.TextInput(
            title='Stop Delay [fs]', value='10.')
        self.stepSizeInput = bokeh.models.widgets.TextInput(
            title='Step Size [fs]', value='0.2')
        self.comment = bokeh.models.widgets.TextAreaInput(
            title='Comment', value='', rows=10)

        # arrange items
        self.inputs = bokeh.layouts.widgetbox(
            self.startBtn, self.integrationInput, self.zeroDelInput,
            self.startDelInput, self.stopDelInput, self.stepSizeInput,
            self.saveBtn, self.saveNameInput, self.scanTableBtn, self.comment)
        self.layout = bokeh.layouts.row(
            self.inputs,
            bokeh.layouts.column(
                self.imagePlot.element, self.linePlot.element), width=800)

    def start(self):
        # in case this measurement is running, stop it
        if self.running.am_i_running(self):
            self.stop()
        # in case a different measurement is running, do nothing
        elif self.running.is_running():
            pass
        else:
            # set running indicator to block double readout
            self.running.now_running(self)
            self.roi.ybin = self.roi.ymax - self.roi.ymin
            self.startBtn.label = 'Stop'
            self.startBtn.button_type = 'danger'
            # integration time
            self.integration = float(self.integrationInput.value)*1000
            # delay vector setup
            if self.scanTableBtn.active:
                self.delays = np.loadtxt('scantable.txt')
            else:
                self.delays = np.arange(
                    float(self.startDelInput.value),
                    float(self.stopDelInput.value),
                    float(self.stepSizeInput.value))
            self.piezo_values = (float(self.zeroDelInput.value)
                                 + 3.0e8 * self.delays*1.0e-15 / 2. * 1e6)
            # scan start time for save name
            self.now = datetime.datetime.now()

            self.cont_camera_readout = measurement.measurement(
                inputs=[[pv, None] for pv in self.piezo_values],
                sequence=[
                    measurement.single_input_sequence_function(
                        self.delayer.abs_move),
                    self.sensor.frame],
                update=measurement.bokeh_update_function(
                    self.update, self.doc),
                init=partial(
                    self.sensor.setup,
                    integration=self.integration,
                    roi=self.roi),
                finish=measurement.bokeh_no_input_finish_function(
                    self.restore, self.doc),
                save_output=True)
            self.cont_camera_readout.start()

    def restore(self):
        self.stop()

    def update(self, data):
        # if data makes sense (remotely)
        curdelays = self.delays[0:len(data)]
        im_data = [d[1].flatten() for d in data]
        # get current spectrum
        curspec = im_data[-1]
        im_data = np.transpose(np.array(im_data))

        # update plots
        try:
            self.linePlot.update(
                num=0, x=np.arange(0, curspec.size), y=curspec)
            self.imagePlot.update(
                num=0,
                x=curdelays,
                y=np.arange(0, im_data.shape[0]),
                z=im_data)
        except Exception as e:
            print('plot error')
            print(e)

        # save scan every step
        try:
            os.makedirs(
                self.now.strftime('%Y-%m') + '/'
                + self.now.strftime('%Y-%m-%d'), exist_ok=True)
            fname = (self.now.strftime('%Y-%m') + '/'
                     + self.now.strftime('%Y-%m-%d')
                     + '/scan_ta_'
                     + self.now.strftime('%Y-%m-%d-%H-%M-%S'))
            # save data hdf5
            with h5py.File(fname + '.hdf5', 'w') as f:
                f.create_dataset('delays', data=self.delays)
                f.create_dataset('data', data=im_data)
                f.create_dataset(
                        'comment', data=self.comment.value,
                        dtype=h5py.special_dtype(vlen=str))
                f.flush()

            # save comment to separate txt file
                with open(fname + '.txt', 'w') as f:
                    f.write(self.comment.value)

            # last step, save picture and scan to logbook
            if len(curdelays) == len(self.delays):
                plt.clf()
                plt.pcolormesh(im_data)
                # plt.savefig(fname + '.pdf')
                plt.savefig(fname + '.png')
                if self.logbook:
                    self.logbook.post(
                        author='auto-save',
                        subject='Current Scan: ' + fname + '.hdf5',
                        text=self.comment.value, filename=fname + '.png')
        except Exception as e:
            print('save error')
            print(e)


class ta_session_handler(bgh.bokeh_gui_session_handler):
    def open_session(self, doc):
        self.running = bgh.running()

        sensor = greateyes()
        roi = camera_dummy_roi()

        jena_controller = linear_axis_controller_jena_eda4(
            port='COM9', baud=9600)
        piezo = linear_axis_piezojena_eda4(
            jena_controller, 0)

        logbook = None

        # create tabs
        raw_readout = continuous_acquisition_gui(
            doc, sensor, self.running, roi)
        spectrum_readout = continuous_spectrum_gui(
            doc, sensor, self.running, roi)
        measurement_readout = measurement_gui(
            doc, sensor, self.running, roi, piezo, logbook)

        self.title = 'Transient Absorption'
        self.tabs = [
            {'layout': raw_readout.layout, 'title': 'Show Chip'},
            {'layout': spectrum_readout.layout, 'title': 'Show Spectrum'},
            {'layout': measurement_readout.layout, 'title': 'Measurement'}]

        # this list is auto-closed, all close function of the added objects
        # are called at session destruction
        self.close_list.append(raw_readout)
        self.close_list.append(spectrum_readout)
        self.close_list.append(measurement_readout)
        self.close_list.append(sensor)


print('start ta')
# start the server
bgh.bokeh_gui_helper(ta_session_handler(), 5010)

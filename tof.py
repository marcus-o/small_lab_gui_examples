import numpy as np
from functools import partial
import os
import datetime
import matplotlib.pyplot as plt
import h5py

from small_lab_gui.helper import bokeh_gui_helper as bgh
from small_lab_gui.helper import measurement

import bokeh

testing = True
if not testing:
    # for the lab
    from small_lab_gui.digitizers.digitizer_fast_mcs6a import mcs6a
    from small_lab_gui.axes.linear_axis_jena_eda4 import \
        linear_axis_controller_jena_eda4
    from small_lab_gui.axes.linear_axis_jena_eda4 import \
        linear_axis_piezojena_eda4
    from small_lab_gui.helper.postToELOG import elog
else:
    # for testing
    from small_lab_gui.digitizers.digitizer_fast_mcs6a_dummy import mcs6a_dummy \
        as mcs6a
    from small_lab_gui.axes.linear_axis_dummy import linear_axis_controller_dummy \
        as linear_axis_controller_jena_eda4
    from small_lab_gui.axes.linear_axis_dummy import linear_axis_dummy \
        as linear_axis_piezojena_eda4
    from small_lab_gui.helper.postToELOG_dummy import elog_dummy as elog


class tof_alignment_gui():
    def __init__(self, doc, running, tof_digitizer, delayer):
        self.title = 'Alignment'
        self.spectrum_length = tof_digitizer.range
        self.spectrum = []

        # measurement thread
        self.measurement = None
        # bokeh doc for callback
        self.doc = doc
        # digitizer card
        self.tof_digitizer = tof_digitizer
        # global measurement running indicator
        self.running = running
        # delay piezo
        self.delayer = delayer

        # for continuous mode
        self.subtractsweeps = 0
        self.subtractstarts = 0
        self.subtractruntime = 0
        self.subtractspectrum = 0
        self.lastspectrum = 0

        # Set up widgets
        self.startBtn = bokeh.models.widgets.Button(
            label='Start', button_type='success')
        self.integrationInput = bokeh.models.widgets.TextInput(
            title='Integration time [ksweeps]', value='6')
        self.piezoPos = bokeh.models.widgets.TextInput(
            title='Piezo Position [um]', value='1.0')

        # spectrum plot
        self.linePlot = bgh.plot_2d()
        self.linePlot.line(legend='Current')

        # save button
        self.saveBtn = bokeh.models.widgets.Button(
            label='Save Spectrum', button_type='success')
        self.saveBtn.on_click(self.save_spectrum)
        self.saveNameInput = bokeh.models.widgets.TextInput(
            title='Legend Name', value='Name')

        # arrange layout
        self.inputs = bokeh.layouts.widgetbox(
            self.startBtn, self.integrationInput, self.saveBtn,
            self.saveNameInput, self.piezoPos)
        self.layout = bokeh.layouts.row(
            self.inputs, self.linePlot.element, width=800)

        # start thread callback
        self.startBtn.on_click(self.start)

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
            # initialize data array
            self.spectrum = 0

            self.subtractsweeps = 0
            self.subtractstarts = 0
            self.subtractruntime = 0
            self.subtractspectrum = 0
            self.lastspectrum = 0

            # switch start to stop button
            self.startBtn.label = 'Stop'
            self.startBtn.button_type = 'danger'

            # set integration time
            self.integration = float(self.integrationInput.value)*1000.

            # create the measurment thread
            self.measurement = measurement.measurement(
                inputs=None,
                sequence=[
                    self.tof_digitizer.readout_continuous,
                    measurement.sleep_function(1)],
                update=measurement.bokeh_update_function(
                    self.update, self.doc),
                init=self.tof_digitizer.start_continuous,
                finish=lambda in1, in2: self.tof_digitizer.stop(),
                save_output=False)
            # start the measurment thread
            self.measurement.start()

    def save_spectrum(self):
        self.linePlot.save_current(name=self.saveNameInput.value, num=0)

    def stop(self):
        if not (self.measurement is None):
            self.measurement.stop()
            self.measurement.join()
        self.running.now_stopped()
        self.startBtn.label = 'Start'
        self.startBtn.button_type = 'success'

    def close(self):
        self.stop()

    def update(self, data):
        if not (data is None):
            curspec = data[0]['data'][0]
            sweeps = data[0]['starts']
            if np.abs(curspec - self.lastspectrum).sum():
                self.lastspectrum = curspec
                self.spectrum = curspec-self.subtractspectrum
                self.linePlot.update(
                    num=0, x=np.arange(0, self.spectrum.size), y=self.spectrum)
                if sweeps - self.subtractsweeps >= self.integration:
                    self.subtractsweeps = sweeps
                    self.subtractspectrum = curspec


class tof_measurement_gui(tof_alignment_gui):
    def __init__(self, doc, running, tof_digitizer, delayer, logbook):
        super().__init__(doc, running, tof_digitizer, delayer)
        self.logbook = logbook
        self.title = 'Measurement'

        # pcolor plot to display results
        self.imagePlot = bgh.plot_false_color()
        self.imagePlot.image()

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
            self.saveBtn, self.saveNameInput,  self.scanTableBtn, self.comment)
        self.layout = bokeh.layouts.row(
            self.inputs,
            bokeh.layouts.column(
                self.imagePlot.element, self.linePlot.element),
            width=800)

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

            # measurement thread
            self.measurement = measurement.measurement(
                inputs=[[pv, None] for pv in self.piezo_values],
                sequence=[
                    measurement.single_input_sequence_function(
                        self.delayer.abs_move),
                    self.tof_digitizer.frame],
                update=measurement.bokeh_update_function(
                    self.update, self.doc),
                init=partial(
                    self.tof_digitizer.setup, integration=self.integration),
                finish=measurement.bokeh_no_input_finish_function(
                    self.stop, self.doc),
                save_output=True)
            self.measurement.start()

    def update(self, data):
        # loop steps
        curdelays = self.delays[0:len(data)]
        im_data = [d[1]['data'][0] for d in data]
        # get current spectrum
        curspec = im_data[-1]
        im_data = np.transpose(np.array(im_data))

        # update plots
        try:
            self.linePlot.update(
                num=0, x=np.arange(0, curspec.size), y=curspec)
            self.imagePlot.update(
                num=0, x=curdelays, y=np.arange(im_data.shape[0]), z=im_data)
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
                     + '/scan_tof_'
                     + self.now.strftime('%Y-%m-%d-%H-%M-%S'))
            # save data hdf5
            with h5py.File(fname + '.hdf5', 'w') as f:
                f.create_dataset('delays', data=curdelays)
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


class tof_session_handler(bgh.bokeh_gui_session_handler):
    def open_session(self, doc):
        self.running = bgh.running()
        # hardware
        digitizer = mcs6a()

        # movement_server = linear_axis.linear_axis_controller_remote()
        # piezo = linear_axis.linear_axis_remote(movement_server, 'piezo')

        jena_controller = linear_axis_controller_jena_eda4(
            port='COM9', baud=9600)
        piezo = linear_axis_piezojena_eda4(
            jena_controller, 0)

        # open logbook to auto-save scans
        logbook = elog(host='localhost', port=8080, logbook='demo')

        # alignment tab
        alignmentgui = tof_alignment_gui(
            doc=doc,
            running=self.running,
            tof_digitizer=digitizer,
            delayer=piezo)

        # measurement tab
        measurementgui = tof_measurement_gui(
            doc=doc,
            running=self.running,
            tof_digitizer=digitizer,
            delayer=piezo,
            logbook=logbook)

        self.title = 'TOF Readout'
        self.tabs = [
            {'layout': alignmentgui.layout, 'title': alignmentgui.title},
            {'layout': measurementgui.layout, 'title': measurementgui.title}]

        # this list is auto-closed, all close functions of the
        # added objects are called at session destruction
        self.close_list.append(alignmentgui)
        self.close_list.append(measurementgui)
        self.close_list.append(piezo)
        self.close_list.append(digitizer)


print('start tof')
# start the server
bgh.bokeh_gui_helper(tof_session_handler(), 5024)

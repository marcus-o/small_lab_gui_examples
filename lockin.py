import numpy as np
from functools import partial
import os
import datetime
import matplotlib.pyplot as plt
import h5py

import bokeh

from small_lab_gui.helper import bokeh_gui_helper as bgh
from small_lab_gui.helper import measurement

testing = True
if not testing:
    # for the lab
    from small_lab_gui.digitizers.digitizer_zi_hf2li import hf2li
    from small_lab_gui.axes.linear_axis_jena_eda4 import \
        linear_axis_controller_jena_eda4
    from small_lab_gui.axes.linear_axis_jena_eda4 import \
        linear_axis_piezojena_eda4
    from small_lab_gui.helper.postToELOG import elog
else:
    # for testing
    from small_lab_gui.digitizers.digitizer_zi_hf2li_dummy import hf2li_dummy \
        as hf2li
    from small_lab_gui.axes.linear_axis_dummy import linear_axis_controller_dummy \
        as linear_axis_controller_jena_eda4
    from small_lab_gui.axes.linear_axis_dummy import linear_axis_dummy \
        as linear_axis_piezojena_eda4
    from small_lab_gui.helper.postToELOG_dummy import elog_dummy as elog


class lockin_alignment_gui():
    def __init__(self, doc, running, lockin_digitizer, delayer):
        self.title = 'Alignment'
        # measurement thread
        self.measurement = None
        # bokeh doc for callback
        self.doc = doc
        # digitizer card
        self.lockin_digitizer = lockin_digitizer
        # global measurement running indicator
        self.running = running
        # delay piezo
        self.delayer = delayer

        # set up inputs
        self.startBtn = bokeh.models.widgets.Button(
            label='Start', button_type='success')
        self.timeconstantInput = bokeh.models.widgets.TextInput(
            title='Time constant [sec]', value='0.05')
        self.orderInput = bokeh.models.widgets.TextInput(
            title='Filter Order', value='6')
        self.integrationInput = bokeh.models.widgets.TextInput(
            title='Integration time [sec]', value='1')
        self.piezoPos = bokeh.models.widgets.TextInput(
            title='Piezo Position [um]', value='1.0')

        # set up outputs
        self.signalIndicator = bokeh.models.widgets.Button(
            label='Rate', button_type='primary')
        self.angleIndicator = bokeh.models.widgets.Button(
            label='Angle', button_type='primary')

        # arrange layout
        self.inputs = bokeh.layouts.widgetbox(
            self.startBtn, self.timeconstantInput,
            self.integrationInput, self.piezoPos)
        self.indicators = bokeh.layouts.widgetbox(
            self.signalIndicator, self.angleIndicator)
        self.layout = bokeh.layouts.row(
            self.inputs, self.indicators, width=800)

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
            # switch start to stop button
            self.startBtn.label = 'Stop'
            self.startBtn.button_type = 'danger'
            # set timeconstant and integration time
            self.timeconstant = float(self.timeconstantInput.value)
            self.integration = float(self.integrationInput.value)

            # create the measurement thread
            self.measurement = measurement.measurement(
                inputs=None,
                sequence=[
                    self.lockin_digitizer.readout_continuous,
                    measurement.sleep_function(0.1)],
                update=measurement.bokeh_update_function(
                    self.update, self.doc),
                init=partial(
                    self.lockin_digitizer.setup,
                    integration=self.integration,
                    timeconstant=self.timeconstant),
                finish=None,
                save_output=False)
            # start the measurement thread
            self.measurement.start()

    def stop(self):
        if self.measurement is not None:
            # measurement is inherited from Thread,
            # so one can wait for it using join
            self.measurement.stop()
            self.measurement.join()
        self.running.now_stopped()
        self.startBtn.label = 'Start'
        self.startBtn.button_type = 'success'

    def close(self):
        self.stop()

    def update(self, data):
        if not (data is None):
            r = data[0]['r']
            theta = data[0]['theta']
            x = np.real(r*np.exp(1j*theta))
            y = np.imag(r*np.exp(1j*theta))
            self.signalIndicator.label = str(x)
            self.angleIndicator.label = str(y)


class lockin_measurement_gui(lockin_alignment_gui):
    def __init__(self, doc, running, lockin_digitizer, delayer, logbook=None):
        super().__init__(doc, running, lockin_digitizer, delayer)
        self.title = 'Measurement'
        self.logbook = logbook

        # measurement source and line plot
        self.linePlot = bgh.plot_2d()
        self.linePlot.line(legend='Current 2', line_color='red')
        self.linePlot.line(legend='Current 5', line_color='blue')

        self.linePlotMean = bgh.plot_2d()
        self.linePlotMean.line(legend='Current 2', line_color='red')
        self.linePlotMean.line(legend='Current 5', line_color='blue')

        # save button
        self.saveBtn = bokeh.models.widgets.Button(
            label='Save Spectrum', button_type='success')
        self.saveBtn.on_click(self.save_spectrum)
        self.saveNameInput = bokeh.models.widgets.TextInput(
            title='Legend Name', value='Name')

        # scan table button
        self.scanTableBtn = bokeh.models.widgets.Toggle(label='Scan Table')

        # measurement inputs
        self.zeroDelInput = bokeh.models.widgets.TextInput(
            title='Zero Delay [um]', value='50.')
        self.startDelInput = bokeh.models.widgets.TextInput(
            title='Start Delay [fs]', value='-10.')
        self.stopDelInput = bokeh.models.widgets.TextInput(
            title='Stop Delay [fs]', value='10.')
        self.stepSizeInput = bokeh.models.widgets.TextInput(
            title='Step Size [fs]', value='0.5')
        self.comment = bokeh.models.widgets.TextAreaInput(
            title='Comment', value='', rows=10)

        # reclaim piezo button
        self.reclaimPiezoBtn = bokeh.models.widgets.Button(
            label='Reclaim Piezo', button_type='success')
        self.reclaimPiezoBtn.on_click(self.reclaimPiezo)

        # arrange items
        self.inputs = bokeh.layouts.widgetbox(
            self.startBtn, self.timeconstantInput, self.orderInput,
            self.integrationInput, self.zeroDelInput, self.startDelInput,
            self.stopDelInput, self.stepSizeInput, self.saveBtn,
            self.saveNameInput, self.scanTableBtn, self.comment,
            self.reclaimPiezoBtn)
        self.layout = bokeh.layouts.row(
            self.inputs,
            bokeh.layouts.column(
                self.linePlot.element, self.linePlotMean.element))

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
            self.integration = float(self.integrationInput.value)
            self.order = int(self.orderInput.value)
            self.timeconstant = float(self.timeconstantInput.value)
            # delay vector setup
            if self.scanTableBtn.active:
                self.delays = np.loadtxt('scantable.txt')
            else:
                self.delays = np.arange(float(self.startDelInput.value),
                                        float(self.stopDelInput.value),
                                        float(self.stepSizeInput.value))
            self.piezo_values = (float(self.zeroDelInput.value) +
                                 3.0e8 * self.delays*1.0e-15 / 2. * 1e6)

            # scan start time for save name
            self.now = datetime.datetime.now()

            # measurement thread
            self.measurement = measurement.measurement(
                inputs=[[pv, None] for pv in self.piezo_values],
                sequence=[
                    measurement.single_input_sequence_function(
                        self.delayer.abs_move),
                    self.lockin_digitizer.frame],
                update=measurement.bokeh_update_function(
                    self.update, self.doc),
                init=partial(
                    self.lockin_digitizer.setup,
                    integration=self.integration,
                    timeconstant=self.timeconstant, order=self.order),
                finish=measurement.bokeh_no_input_finish_function(
                    self.stop, self.doc),
                save_output=True)
            self.measurement.start()

    def save_spectrum(self):
        self.linePlot.save_current(name=self.saveNameInput.value, num=0)

    def reclaimPiezo(self):
        self.delayer.controller.reinit()
        self.delayer.reinit()

    def update(self, data):
        # check if data has contents
        if not (data is None):
            # loop sub steps
            curdelays = self.delays[0:len(data)]
            xs = np.real([d[1]['sig'] for d in data])
            ys = np.imag([d[1]['sig'] for d in data])
            rs = [d[1]['r'] for d in data]
            thetas = [d[1]['theta'] for d in data]
            xs2 = np.real([d[1]['sig2'] for d in data])
            ys2 = np.imag([d[1]['sig2'] for d in data])
            rs2 = [d[1]['r2'] for d in data]
            thetas2 = [d[1]['theta2'] for d in data]

            # update plots
            try:
                self.linePlot.update(num=0, x=curdelays, y=xs)
                self.linePlot.update(num=1, x=curdelays, y=xs2)
                self.linePlotMean.update(
                    num=0, x=curdelays, y=xs-np.mean(xs))
                self.linePlotMean.update(
                    num=1, x=curdelays, y=xs2-np.mean(xs2))
            except Exception as e:
                print('plot error')
                print(e)

            # save data
            try:
                os.makedirs(
                    self.now.strftime('%Y-%m') + '/' +
                    self.now.strftime('%Y-%m-%d'), exist_ok=True)
                fname = (self.now.strftime('%Y-%m') + '/'
                         + self.now.strftime('%Y-%m-%d')
                         + '/scan_lockin_'
                         + self.now.strftime('%Y-%m-%d-%H-%M-%S'))

                # save data to hdf5 file
                with h5py.File(fname + '.hdf5', 'w') as f:
                    f.create_dataset('delays', data=curdelays)
                    f.create_dataset('rs', data=rs)
                    f.create_dataset('thetas', data=thetas)
                    f.create_dataset('xs', data=xs)
                    f.create_dataset('ys', data=ys)
                    f.create_dataset('rs2', data=rs2)
                    f.create_dataset('thetas2', data=thetas2)
                    f.create_dataset('xs2', data=xs2)
                    f.create_dataset('ys2', data=ys2)
                    f.create_dataset(
                        'comment', data=self.comment.value,
                        dtype=h5py.special_dtype(vlen=str))
                    f.flush()

                # save comment to separate txt file
                with open(fname + '.txt', 'w') as f:
                    f.write(self.comment.value)

                # last step, save picture and scan to logbook
                if len(data) == len(self.delays):
                    plt.clf()
                    plt.plot(curdelays, xs)
                    plt.plot(curdelays, xs2)
                    plt.savefig(fname + '.pdf')
                    plt.savefig(fname + '.png')
                    if self.logbook:
                        self.logbook.post(
                            author='auto-save',
                            subject='Current Scan: ' + fname + '.hdf5',
                            text=self.comment.value, filename=fname + '.png')
            except Exception as e:
                print('save error')
                print(e)


class lockin_session_handler(bgh.bokeh_gui_session_handler):
    def open_session(self, doc):
        # create object to determine if some measurement is currently running
        running = bgh.running()

        # open hardware
        # zurich instruments lock-in digitzer
        digitizer = hf2li()

        # delay light pulse via piezo
        # pi piezo
        # movement_server = linear_axis.linear_axis_controller_remote()
        # piezo = linear_axis.linear_axis_remote(movement_server, 'piezo')
        # jena piezo
        jena_controller = linear_axis_controller_jena_eda4(
            port='COM9', baud=9600)
        piezo = linear_axis_piezojena_eda4(jena_controller, 0)

        # open logbook to auto-save scans
        logbook = elog(host='localhost', port=8080, logbook='demo')

        # alignment tab
        alignmentgui = lockin_alignment_gui(
            doc=doc,
            running=running,
            lockin_digitizer=digitizer,
            delayer=piezo)

        # measurement tab
        measurementgui = lockin_measurement_gui(
            doc=doc,
            running=running,
            lockin_digitizer=digitizer,
            delayer=piezo,
            logbook=logbook)

        self.title = 'Lockin Readout'
        self.tabs = [
            {'layout': alignmentgui.layout, 'title': alignmentgui.title},
            {'layout': measurementgui.layout, 'title': measurementgui.title}]

        # this list is auto-closed, all close functions of the
        # added objects are called at session destruction
        self.close_list.append(alignmentgui)
        self.close_list.append(measurementgui)
        self.close_list.append(piezo)
        self.close_list.append(digitizer)


print('start lockin')
# start the server
bgh.bokeh_gui_helper(lockin_session_handler(), 5025)
